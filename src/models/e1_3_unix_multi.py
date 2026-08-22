import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

FROZEN_DIR = "./data/manifests"

print("1. Loading datasets...")
train_df = pd.read_json(f"{FROZEN_DIR}/train_frozen.jsonl", lines=True)
val_df = pd.read_json(f"{FROZEN_DIR}/val_frozen.jsonl", lines=True)
test_pv_df = pd.read_json(f"{FROZEN_DIR}/test_primevul_frozen.jsonl", lines=True)
test_pair_df = pd.read_json(f"{FROZEN_DIR}/test_paired_frozen.jsonl", lines=True)
test_bv_df = pd.read_json(f"{FROZEN_DIR}/test_bigvul_frozen.jsonl", lines=True)

print("2. Building CWE Vocabulary...")
vul_train = train_df[train_df['label'] == 1].copy()

if 'cwe_label' not in vul_train.columns:
    vul_train['cwe_label'] = 'CWE-Other'
vul_train['cwe_label'] = vul_train['cwe_label'].fillna('CWE-Other')

unique_cwes = sorted(vul_train['cwe_label'].unique().tolist())
cwe2id = {cwe: idx for idx, cwe in enumerate(unique_cwes)}
NUM_CWE = len(unique_cwes)

print("3. Applying CWE Masking (-100 for Clean Codes)...")

def apply_cwe_mask(df):
    """Assigns CWE ID for vulnerabilities and -100 for clean codes."""
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
test_pair_df = apply_cwe_mask(test_pair_df)
test_bv_df = apply_cwe_mask(test_bv_df)

print("4. Tokenizing...")
MODEL_NAME = "microsoft/unixcoder-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = Dataset.from_pandas(train_df[['func_code', 'label', 'cwe_encoded']])
val_ds = Dataset.from_pandas(val_df[['func_code', 'label', 'cwe_encoded']])
test_pv_ds = Dataset.from_pandas(test_pv_df[['func_code', 'label', 'cwe_encoded']])
test_pair_ds = Dataset.from_pandas(test_pair_df[['func_code', 'label', 'cwe_encoded']])
test_bv_ds = Dataset.from_pandas(test_bv_df[['func_code', 'label', 'cwe_encoded']])

def tokenize_function(examples):
    return tokenizer(
        examples["func_code"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=['func_code'])
tokenized_val = val_ds.map(tokenize_function, batched=True, remove_columns=['func_code'])
tokenized_test_pv = test_pv_ds.map(tokenize_function, batched=True, remove_columns=['func_code'])
tokenized_test_pair = test_pair_ds.map(tokenize_function, batched=True, remove_columns=['func_code'])
tokenized_test_bv = test_bv_ds.map(tokenize_function, batched=True, remove_columns=['func_code'])

columns_to_keep = ["input_ids", "attention_mask", "label", "cwe_encoded"]
tokenized_train.set_format("torch", columns=columns_to_keep)
tokenized_val.set_format("torch", columns=columns_to_keep)
tokenized_test_pv.set_format("torch", columns=columns_to_keep)
tokenized_test_pair.set_format("torch", columns=columns_to_keep)
tokenized_test_bv.set_format("torch", columns=columns_to_keep)

import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

print("5. Defining Multi-task Architecture...")

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
        
        return {
            "logits": bin_logits,
            "cwe_logits": cwe_logits
        }

from transformers import Trainer
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef, f1_score, 
    precision_score, recall_score, accuracy_score, confusion_matrix
)
import numpy as np

print("6. Defining Trainer and Metrics...")

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
        loss_cwe = loss_fct_cwe(cwe_logits, cwe_labels)
        
        LAMBDA = 0.2
        total_loss = loss_bin + LAMBDA * loss_cwe
        
        return (total_loss, outputs) if return_outputs else total_loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions[0], eval_pred.label_ids
    probs = 1.0 / (1.0 + np.exp(-logits))
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
    
    return {
        "accuracy": accuracy,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "pr_auc": pr_auc
    }

from transformers import TrainingArguments, set_seed
import os

print("7. Configuring Training Loop...")

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEEDS = [13] 
OUTPUT_DIR = "./saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for seed in SEEDS:
    print(f"\nStarting training for seed: {seed}")
    set_seed(seed)
    
    model = UniXCoderMultiTask(num_cwe_classes=NUM_CWE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    seed_output_dir = f"{OUTPUT_DIR}/unixcoder_seed_{seed}"
    
    training_args = TrainingArguments(
        output_dir=seed_output_dir,
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
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=2,
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
    torch.save(model.state_dict(), f"{seed_output_dir}/best_model.pt")

print("Training completed successfully!")
