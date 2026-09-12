import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, set_seed
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef, f1_score, 
    precision_score, recall_score, accuracy_score, confusion_matrix,
    roc_auc_score, brier_score_loss
)
import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece
import os
import gc
from torch.utils.data import DataLoader

FROZEN_DIR = "./data/manifests"
OUTPUT_DIR = "./outputs/e2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Loading datasets")
train_df = pd.read_json(f"{FROZEN_DIR}/train_frozen.jsonl", lines=True)
val_df = pd.read_json(f"{FROZEN_DIR}/val_frozen.jsonl", lines=True)
test_pv_df = pd.read_json(f"{FROZEN_DIR}/test_primevul_frozen.jsonl", lines=True)

print("2. Building CWE Vocabulary")
vul_train = train_df[train_df['label'] == 1].copy()
if 'cwe_label' not in vul_train.columns:
    vul_train['cwe_label'] = 'CWE-Other'
vul_train['cwe_label'] = vul_train['cwe_label'].fillna('CWE-Other')

unique_cwes = sorted(vul_train['cwe_label'].unique().tolist())
cwe2id = {cwe: idx for idx, cwe in enumerate(unique_cwes)}
NUM_CWE = len(unique_cwes)

print("3. Applying CWE Masking")
def apply_cwe_mask(df):
    if 'cwe_label' not in df.columns:
        df['cwe_label'] = 'CWE-Other'
    else:
        df['cwe_label'] = df['cwe_label'].fillna('CWE-Other')
    df['cwe_encoded'] = df.apply(
        lambda x: cwe2id.get(x['cwe_label'], cwe2id.get('CWE-Other')) if x['label'] == 1 else -100, 
        axis=1
    )
    return df

train_df = apply_cwe_mask(train_df)
val_df = apply_cwe_mask(val_df)
test_pv_df = apply_cwe_mask(test_pv_df)

print("4. Tokenizing")
MODEL_NAME = "microsoft/unixcoder-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["func_code"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

train_ds = Dataset.from_pandas(train_df[['func_code', 'label', 'cwe_encoded']])
val_ds = Dataset.from_pandas(val_df[['func_code', 'label', 'cwe_encoded']])
test_pv_ds = Dataset.from_pandas(test_pv_df[['func_code', 'label', 'cwe_encoded']])

tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=['func_code'], num_proc=4)
tokenized_val = val_ds.map(tokenize_function, batched=True, remove_columns=['func_code'], num_proc=4)
tokenized_test_pv = test_pv_ds.map(tokenize_function, batched=True, remove_columns=['func_code'], num_proc=4)

columns_to_keep = ["input_ids", "attention_mask", "label", "cwe_encoded"]
tokenized_train.set_format("torch", columns=columns_to_keep)
tokenized_val.set_format("torch", columns=columns_to_keep)
tokenized_test_pv.set_format("torch", columns=columns_to_keep)

print("5. Defining Multi-task Architecture")
class UniXCoderMultiTask(nn.Module):
    def __init__(self, model_name="microsoft/unixcoder-base", num_cwe_classes=15):
        super(UniXCoderMultiTask, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        hidden_size = self.config.hidden_size
        
        self.classifier_bin = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier_cwe = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_cwe_classes)
        )

    def forward(self, input_ids, attention_mask, labels=None, cwe_encoded=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] 
        cls_embedding = sequence_output[:, 0, :] 
        bin_logits = self.classifier_bin(cls_embedding).squeeze(-1)
        cwe_logits = self.classifier_cwe(cls_embedding)
        return {"logits": bin_logits, "cwe_logits": cwe_logits}

class MultiTaskTrainer(Trainer):
    def __init__(self, lambda_cwe=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_cwe = lambda_cwe

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        cwe_labels = inputs.pop("cwe_encoded")
        outputs = model(**inputs)
        bin_logits = outputs["logits"]
        cwe_logits = outputs["cwe_logits"]
        
        pos_weight = torch.tensor([45.44]).to(bin_logits.device)
        loss_fct_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_bin = loss_fct_bin(bin_logits, labels.float())
        
        loss_fct_cwe = nn.CrossEntropyLoss(ignore_index=-100)
        loss_cwe = loss_fct_cwe(cwe_logits, cwe_labels)
        
        total_loss = loss_bin + self.lambda_cwe * loss_cwe
        return (total_loss, outputs) if return_outputs else total_loss

def custom_evaluate_test(model, test_dataset, device):
    """Manual evaluation loop on the test set to collect both Vul and CWE metrics"""
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_bin_labels = []
    all_bin_preds = []
    all_bin_probs = []
    all_cwe_labels = []
    all_cwe_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            cwe_encoded = batch["cwe_encoded"].to(device)
            
            outputs = model(input_ids, attention_mask)
            bin_logits = outputs["logits"]
            cwe_logits = outputs["cwe_logits"]
            
            probs = torch.sigmoid(bin_logits)
            preds = (probs > 0.5).int()
            
            cwe_preds = torch.argmax(cwe_logits, dim=1)
            
            all_bin_labels.extend(labels.cpu().numpy())
            all_bin_preds.extend(preds.cpu().numpy())
            all_bin_probs.extend(probs.cpu().numpy())
            
            valid_cwe_idx = cwe_encoded != -100
            if valid_cwe_idx.sum() > 0:
                all_cwe_labels.extend(cwe_encoded[valid_cwe_idx].cpu().numpy())
                all_cwe_preds.extend(cwe_preds[valid_cwe_idx].cpu().numpy())
                
    all_bin_labels = np.array(all_bin_labels)
    all_bin_preds = np.array(all_bin_preds)
    all_bin_probs = np.array(all_bin_probs)

    precision_curve, recall_curve, _ = precision_recall_curve(all_bin_labels, all_bin_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    accuracy = accuracy_score(all_bin_labels, all_bin_preds)
    mcc = matthews_corrcoef(all_bin_labels, all_bin_preds)
    precision = precision_score(all_bin_labels, all_bin_preds, zero_division=0)
    recall = recall_score(all_bin_labels, all_bin_preds, zero_division=0)
    vul_f1 = f1_score(all_bin_labels, all_bin_preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(all_bin_labels, all_bin_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    roc_auc = roc_auc_score(all_bin_labels, all_bin_probs)
    brier = brier_score_loss(all_bin_labels, all_bin_probs)
    ece = expected_calibration_error(all_bin_labels, all_bin_probs)

    cwe_macro_f1 = f1_score(all_cwe_labels, all_cwe_preds, average='macro', zero_division=0) if len(all_cwe_labels) > 0 else 0.0
    
    return {
        "accuracy": accuracy, "mcc": mcc, "precision": precision, "recall": recall,
        "vul_f1": vul_f1, "fpr": fpr, "pr_auc": pr_auc, "roc_auc": roc_auc,
        "brier": brier, "ece": ece, "cwe_macro_f1": cwe_macro_f1
    }

print("6. Starting Ablation Loop")
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEEDS = [13, 21, 42, 87, 101]
LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0]

results = []

for seed in SEEDS:
    for l_cwe in LAMBDA_VALUES:
        print(f" Training Model with Lambda CWE = {l_cwe} | Seed = {seed} ")
        set_seed(seed)
        
        model = UniXCoderMultiTask(num_cwe_classes=NUM_CWE)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        model_output_dir = f"{OUTPUT_DIR}/lambda_{l_cwe}_seed_{seed}"
        
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            disable_tqdm=False,
            logging_strategy="steps",    
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", 
            greater_is_better=False,
            bf16=True if torch.cuda.is_available() else False,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            optim="adamw_torch_fused",
            report_to="none"
        )
        
        trainer = MultiTaskTrainer(
            lambda_cwe=l_cwe,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val
        )
        
        trainer.train()

        torch.save(model.state_dict(), f"{model_output_dir}/best_model.pt")
        
        print(f"Evaluating Lambda = {l_cwe} | Seed = {seed} on Test PrimeVul")
        metrics = custom_evaluate_test(model, tokenized_test_pv, device)
        
        print(f"-> Vul-F1: {metrics['vul_f1']:.4f} | CWE Macro-F1: {metrics['cwe_macro_f1']:.4f}")
        results.append({
            "Seed": seed,
            "Lambda": l_cwe,
            "Vulnerability-F1": metrics["vul_f1"],
            "CWE-Macro-F1": metrics["cwe_macro_f1"],
            "Accuracy": metrics["accuracy"],
            "MCC": metrics["mcc"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "FPR": metrics["fpr"],
            "PR-AUC": metrics["pr_auc"],
            "ROC-AUC": metrics["roc_auc"],
            "Brier": metrics["brier"],
            "ECE": metrics["ece"]
        })
        
        del model, trainer, training_args
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


import scipy.stats as st
import numpy as np
import os

df_results = pd.DataFrame(results)

report_path = "outputs/e2/per_seed_metrics.csv"
df_results.to_csv(report_path, index=False)

print("\nComputing Aggregated Metrics and Paired Differences...")

metrics = ["Vulnerability-F1", "CWE-Macro-F1", "Accuracy", "MCC", "Precision", "Recall", "FPR", "PR-AUC", "ROC-AUC", "Brier", "ECE"]

# 1. Compute Mean, SD, 95% CI
agg_records = []
for l_cwe in df_results['Lambda'].unique():
    subset = df_results[df_results['Lambda'] == l_cwe]
    n = len(subset)
    record = {"Lambda": l_cwe, "N_Seeds": n}
    for m in metrics:
        if m in subset.columns:
            mean_val = subset[m].mean()
            std_val = subset[m].std(ddof=1) if n > 1 else 0.0
            t_crit = st.t.ppf(0.975, n-1) if n > 1 else 1.96
            ci_val = t_crit * std_val / np.sqrt(n) if n > 1 else 0.0
            record[f"{m}_mean"] = mean_val
            record[f"{m}_std"] = std_val
            record[f"{m}_95CI"] = ci_val
    agg_records.append(record)

agg_df = pd.DataFrame(agg_records)
agg_path = "outputs/e2/lambda_summary.csv"
agg_df.to_csv(agg_path, index=False)

print("\nABLATION COMPLETE!")
print(f"Raw results saved to {report_path}")
print(f"Aggregated results saved to {agg_path}")
