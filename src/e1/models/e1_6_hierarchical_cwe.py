import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, set_seed
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef, f1_score, 
    precision_score, recall_score, accuracy_score, confusion_matrix
)
import numpy as np
import os
import gc

FROZEN_DIR = "./data/manifests"
MODEL_NAME = "microsoft/unixcoder-base"

print("1. Loading datasets")
train_df = pd.read_json(f"{FROZEN_DIR}/train_frozen.jsonl", lines=True)
val_df = pd.read_json(f"{FROZEN_DIR}/val_frozen.jsonl", lines=True)
test_pv_df = pd.read_json(f"{FROZEN_DIR}/test_primevul_frozen.jsonl", lines=True)

print("2. Mapping CWEs to Hierarchical Families")
CWE_FAMILY_MAP = {
    "CWE-119": "Memory",
    "CWE-125": "Memory",
    "CWE-787": "Memory",
    "CWE-416": "Memory",
    "CWE-476": "Memory",
    "CWE-399": "Memory",
    
    "CWE-20": "Input Validation",
    "CWE-78": "Input Validation",
    "CWE-79": "Input Validation",
    "CWE-89": "Input Validation",
    
    "CWE-190": "Numeric",
    "CWE-189": "Numeric",
    
    "CWE-400": "Resource/Lifetime",
    "CWE-362": "Resource/Lifetime",
    "CWE-401": "Resource/Lifetime",
}

def map_cwe(cwe_code):
    if pd.isna(cwe_code) or cwe_code == "CWE-Other":
        return "Other"
    return CWE_FAMILY_MAP.get(cwe_code, "Other")

for df in [train_df, val_df, test_pv_df]:
    if 'cwe_label' not in df.columns:
        df['cwe_label'] = 'CWE-Other'

train_df['cwe_family'] = train_df['cwe_label'].apply(map_cwe)
val_df['cwe_family'] = val_df['cwe_label'].apply(map_cwe)
test_pv_df['cwe_family'] = test_pv_df['cwe_label'].apply(map_cwe)

unique_families = ["Memory", "Input Validation", "Numeric", "Resource/Lifetime", "Other"]
cwe2id = {fam: idx for idx, fam in enumerate(unique_families)}
NUM_CWE = len(unique_families)

def apply_cwe_mask(df):
    df['cwe_encoded'] = df.apply(
        lambda x: cwe2id[x['cwe_family']] if x['label'] == 1 else -100, 
        axis=1
    )
    return df

train_df = apply_cwe_mask(train_df)
val_df = apply_cwe_mask(val_df)
test_pv_df = apply_cwe_mask(test_pv_df)

print(f"3. Tokenizing with {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["func_code"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

def df_to_tokenized_dataset(df):
    ds = Dataset.from_pandas(df[['func_code', 'label', 'cwe_encoded']])
    tokenized = ds.map(tokenize_function, batched=True, remove_columns=['func_code'], num_proc=4)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label", "cwe_encoded"])
    return tokenized

tokenized_train = df_to_tokenized_dataset(train_df)
tokenized_val = df_to_tokenized_dataset(val_df)
tokenized_test_pv = df_to_tokenized_dataset(test_pv_df)

class UniXCoderMultiTask(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_cwe=NUM_CWE):
        super(UniXCoderMultiTask, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
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
            nn.Linear(hidden_size, num_cwe)
        )

    def forward(self, input_ids, attention_mask, labels=None, cwe_encoded=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :] 
        bin_logits = self.classifier_bin(cls_embedding).squeeze(-1)
        cwe_logits = self.classifier_cwe(cls_embedding)
        return {"logits": bin_logits, "cwe_logits": cwe_logits}

class MultiTaskTrainer(Trainer):
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
        loss_cwe = loss_fct_cwe(cwe_logits.view(-1, NUM_CWE), cwe_labels.view(-1))
        
        LAMBDA_CWE = 0.2
        total_loss = loss_bin + LAMBDA_CWE * loss_cwe
        return (total_loss, outputs) if return_outputs else total_loss

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

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        bin_logits = logits[0]
    else:
        bin_logits = logits
        
    probs = 1.0 / (1.0 + np.exp(-bin_logits))
    preds = (probs > 0.5).astype(int)
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    accuracy = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    from sklearn.metrics import roc_auc_score, brier_score_loss
    roc_auc = roc_auc_score(labels, probs)
    brier = brier_score_loss(labels, probs)
    ece = expected_calibration_error(labels, probs)
    
    return {
        "accuracy": accuracy,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece": ece
    }

def run_e1_6(seed):
    print(f"\nStarting E1.6 Hierarchical CWE for seed {seed}")
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXCoderMultiTask(num_cwe=NUM_CWE)
    model.to(device)
    
    EPOCHS = 3
    BATCH_SIZE = 16
    OUTPUT_DIR = f"./saved_models_e1_6/seed_{seed}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        report_to="none"
    )
    
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    del model, trainer, training_args
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for seed in [13, 21, 42, 87, 101]:
        run_e1_6(seed)
