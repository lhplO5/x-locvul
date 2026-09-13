import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import Dataset
import os
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef, f1_score, 
    precision_score, recall_score, accuracy_score, confusion_matrix,
    roc_auc_score, brier_score_loss
)
from statsmodels.stats.contingency_tables import mcnemar

print("1. Loading datasets...")
FROZEN_DIR = "./data/manifests"
test_pv_df = pd.read_json(f"{FROZEN_DIR}/test_primevul_frozen.jsonl", lines=True)
test_bv_df = pd.read_json(f"{FROZEN_DIR}/test_bigvul_frozen.jsonl", lines=True)
test_pair_df = pd.read_json(f"{FROZEN_DIR}/test_paired_frozen.jsonl", lines=True)

# Add dummy cwe_encoded to test sets to avoid KeyError in mapping
for df in [test_pv_df, test_bv_df, test_pair_df]:
    if 'cwe_encoded' not in df.columns:
        df['cwe_encoded'] = -100

print("2. Setting up Tokenizers...")
cb_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
ux_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")

def tokenize_df(df, tokenizer):
    ds = Dataset.from_pandas(df[['func_code', 'label', 'cwe_encoded']])
    def tokenize_func(examples):
        return tokenizer(examples["func_code"], padding="max_length", truncation=True, max_length=512)
    tokenized = ds.map(tokenize_func, batched=True, remove_columns=['func_code'])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label", "cwe_encoded"])
    return tokenized

cb_pv = tokenize_df(test_pv_df, cb_tokenizer)
cb_bv = tokenize_df(test_bv_df, cb_tokenizer)
cb_pair = tokenize_df(test_pair_df, cb_tokenizer)

ux_pv = tokenize_df(test_pv_df, ux_tokenizer)
ux_bv = tokenize_df(test_bv_df, ux_tokenizer)
ux_pair = tokenize_df(test_pair_df, ux_tokenizer)

print("3. Defining Model Architectures...")

class CodeBERTSingleTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        hidden_size = self.encoder.config.hidden_size
        self.classifier_bin = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden_size, 1))

    def forward(self, input_ids, attention_mask, labels=None, cwe_encoded=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :] 
        bin_logits = self.classifier_bin(cls_embedding).squeeze(-1)
        return {"logits": bin_logits}

class UniXCoderSingleTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/unixcoder-base")
        hidden_size = self.encoder.config.hidden_size
        self.classifier_bin = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden_size, 1))

    def forward(self, input_ids, attention_mask, labels=None, cwe_encoded=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :] 
        bin_logits = self.classifier_bin(cls_embedding).squeeze(-1)
        return {"logits": bin_logits}

class UniXCoderMultiTask(nn.Module):
    def __init__(self, num_cwe=16):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/unixcoder-base")
        hidden_size = self.encoder.config.hidden_size
        self.classifier_bin = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden_size, 1))
        self.classifier_cwe = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden_size, num_cwe))

    def forward(self, input_ids, attention_mask, labels=None, cwe_encoded=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :] 
        bin_logits = self.classifier_bin(cls_embedding).squeeze(-1)
        cwe_logits = self.classifier_cwe(cls_embedding)
        return {"logits": bin_logits, "cwe_logits": cwe_logits}


# Helper for metrics
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = y_true[in_bin].mean()
            conf_in_bin = y_prob[in_bin].mean()
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return ece

def calculate_metrics(labels, probs):
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
    
    roc_auc = roc_auc_score(labels, probs)
    brier = brier_score_loss(labels, probs)
    ece = expected_calibration_error(labels, probs)
    
    return {
        "Accuracy": f"{accuracy:.4f}",
        "MCC": f"{mcc:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1-score": f"{f1:.4f}",
        "FPR": f"{fpr:.4f}",
        "PR-AUC": f"{pr_auc:.4f}",
        "ROC-AUC": f"{roc_auc:.4f}",
        "Calibration ECE": f"{ece:.4f}",
        "Brier score": f"{brier:.4f}"
    }

def calculate_mcnemar(labels, preds1, preds2):
    correct1 = (preds1 == labels)
    correct2 = (preds2 == labels)
    
    both_correct = np.sum(correct1 & correct2)
    a_correct_b_wrong = np.sum(correct1 & ~correct2)
    a_wrong_b_correct = np.sum(~correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)
    
    table = [[both_correct, a_correct_b_wrong],
             [a_wrong_b_correct, both_wrong]]
             
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue

# Configurations
CONFIGS = [
    {
        "run": "E1.1", 
        "desc": "CodeBERT + BCE Loss", 
        "purpose": "Huggingface Baseline",
        "model_cls": CodeBERTSingleTask,
        "tokenizer": "cb",
        "checkpoint": "./saved_models_e1_1/codebert_seed_13/best_model.pt",
        "num_cwe": None
    },
    {
        "run": "E1.2", 
        "desc": "UniXCoder + BCE Loss", 
        "purpose": "Encoder Baseline",
        "model_cls": UniXCoderSingleTask,
        "tokenizer": "ux",
        "checkpoint": "./saved_models_e1_2/unixcoder_seed_13/best_model.pt",
        "num_cwe": None
    },
    {
        "run": "E1.3", 
        "desc": "UniXCoder + Multi-task (Bin+CWE)", 
        "purpose": "Proposed Methodology",
        "model_cls": UniXCoderMultiTask,
        "tokenizer": "ux",
        "checkpoint": "./saved_models_e1_3/seed_13/best_model.pt",
        "num_cwe": 16 # E1.3 uses 15 CWEs + 1 Other = 16
    },
    {
        "run": "E1.4", 
        "desc": "UniXCoder + Hard-Mining", 
        "purpose": "Ablation: Data Sampling",
        "model_cls": UniXCoderMultiTask,
        "tokenizer": "ux",
        "checkpoint": "./saved_models_e1_4/stage2_seed_13/best_model.pt",
        "num_cwe": 16
    },
    {
        "run": "E1.5", 
        "desc": "UniXCoder + Shuffled CWE", 
        "purpose": "Negative Control",
        "model_cls": UniXCoderMultiTask,
        "tokenizer": "ux",
        "checkpoint": "./saved_models_e1_5/seed_13/best_model.pt",
        "num_cwe": 16
    },
    {
        "run": "E1.6", 
        "desc": "UniXCoder + Hierarchical CWE", 
        "purpose": "Ablation: Granularity",
        "model_cls": UniXCoderMultiTask,
        "tokenizer": "ux",
        "checkpoint": "./saved_models_e1_6/seed_13/best_model.pt",
        "num_cwe": 5 # E1.6 uses 5 families
    }
]

# We need a Custom Trainer to extract outputs nicely
class EvalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Handle both single and multi-task models
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs[0]
        
        loss = torch.tensor(0.0).to(logits.device) # Dummy loss
        return (loss, outputs) if return_outputs else loss


# To store paired predictions for McNemar
paired_predictions = {}
labels_paired = test_pair_df['label'].values

table1_rows = []
table2_rows = []
table3_rows = []

print("\n4. Running Evaluations...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=32, report_to="none")

for cfg in CONFIGS:
    print(f"Evaluating {cfg['run']}...")
    if not os.path.exists(cfg['checkpoint']):
        print(f"  -> Checkpoint {cfg['checkpoint']} not found! Skipping...")
        continue
    
    if cfg['num_cwe']:
        model = cfg['model_cls'](num_cwe=cfg['num_cwe'])
    else:
        model = cfg['model_cls']()
        
    model.load_state_dict(torch.load(cfg['checkpoint'], map_location=device))
    model.to(device)
    model.eval()
    
    # Select datasets based on tokenizer
    if cfg['tokenizer'] == 'cb':
        ds_pv, ds_bv, ds_pair = cb_pv, cb_bv, cb_pair
    else:
        ds_pv, ds_bv, ds_pair = ux_pv, ux_bv, ux_pair
        
    trainer = EvalTrainer(model=model, args=args)
    
    def get_probs(dataset):
        preds = trainer.predict(dataset)
        logits = preds.predictions
        # Output shape could be tuple if multitask
        if isinstance(logits, tuple):
            bin_logits = logits[0]
        elif isinstance(logits, dict) and "logits" in logits:
            bin_logits = logits["logits"]
        else:
            bin_logits = logits
        
        # Handle array shapes correctly
        if bin_logits.ndim == 2:
            bin_logits = bin_logits[:, 0] if bin_logits.shape[1] == 1 else bin_logits
            
        probs = 1.0 / (1.0 + np.exp(-bin_logits))
        return probs

    print("  -> PrimeVul...")
    probs_pv = get_probs(ds_pv)
    metrics_pv = calculate_metrics(test_pv_df['label'].values, probs_pv)
    row_pv = {"Run": cfg['run'], "Encoder/loss/sampling": cfg['desc'], "Mục đích": cfg['purpose']}
    row_pv.update(metrics_pv)
    table1_rows.append(row_pv)
    
    print("  -> BigVul...")
    probs_bv = get_probs(ds_bv)
    metrics_bv = calculate_metrics(test_bv_df['label'].values, probs_bv)
    row_bv = {"Run": cfg['run'], "Encoder/loss/sampling": cfg['desc'], "Mục đích": cfg['purpose']}
    row_bv.update(metrics_bv)
    table2_rows.append(row_bv)
    
    print("  -> Paired Set...")
    probs_pair = get_probs(ds_pair)
    preds_pair = (probs_pair > 0.5).astype(int)
    paired_predictions[cfg['run']] = preds_pair
    
    acc_pair = accuracy_score(labels_paired, preds_pair)
    
    row_pair = {
        "Run": cfg['run'], 
        "Encoder/loss/sampling": cfg['desc'], 
        "Mục đích": cfg['purpose'],
        "Paired Accuracy": f"{acc_pair:.4f}",
        "McNemar's p-value (vs E1.1)": "N/A",
        "McNemar's p-value (vs E1.2)": "N/A"
    }
    
    # Calculate McNemar vs E1.1
    if 'E1.1' in paired_predictions and cfg['run'] != 'E1.1':
        p1 = calculate_mcnemar(labels_paired, paired_predictions[cfg['run']], paired_predictions['E1.1'])
        row_pair["McNemar's p-value (vs E1.1)"] = f"{p1:.4e}"
        
    # Calculate McNemar vs E1.2
    if 'E1.2' in paired_predictions and cfg['run'] != 'E1.2':
        p2 = calculate_mcnemar(labels_paired, paired_predictions[cfg['run']], paired_predictions['E1.2'])
        row_pair["McNemar's p-value (vs E1.2)"] = f"{p2:.4e}"
        
    table3_rows.append(row_pair)
    
    # Save individual results to the model's directory
    model_dir = os.path.dirname(cfg['checkpoint'])
    import json
    eval_dict = {
        "PrimeVul": metrics_pv,
        "BigVul": metrics_bv,
        "Paired": row_pair
    }
    with open(os.path.join(model_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_dict, f, indent=4)
    print(f"  -> Saved individual metrics to {model_dir}/eval_metrics.json")


print("\n5. Saving CSV Reports...")
os.makedirs("outputs/e1", exist_ok=True)

# Define column orders specifically requested
col_order = ["Run", "Encoder/loss/sampling", "Mục đích", "PR-AUC", "F1-score", "MCC", "Precision", "Recall", "Accuracy", "FPR", "ROC-AUC", "Calibration ECE", "Brier score"]
t3_order = ["Run", "Encoder/loss/sampling", "Mục đích", "Paired Accuracy", "McNemar's p-value (vs E1.1)", "McNemar's p-value (vs E1.2)"]

if table1_rows:
    df1 = pd.DataFrame(table1_rows)
    df1 = df1[[c for c in col_order if c in df1.columns]]
    df1.to_csv("outputs/e1/primevul_seed_metrics.csv", index=False)
    print("Saved outputs/e1/primevul_seed_metrics.csv")

if table2_rows:
    df2 = pd.DataFrame(table2_rows)
    df2 = df2[[c for c in col_order if c in df2.columns]]
    df2.to_csv("outputs/e1/bigvul_seed_metrics.csv", index=False)
    print("Saved outputs/e1/bigvul_seed_metrics.csv")

if table3_rows:
    df3 = pd.DataFrame(table3_rows)
    df3 = df3[[c for c in t3_order if c in df3.columns]]
    df3.to_csv("outputs/e1/paired_predictions.csv", index=False)
    print("Saved outputs/e1/paired_predictions.csv")

print("[OK] Evaluation completed successfully!")
