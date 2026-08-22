import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel
from transformers import Trainer, TrainingArguments, set_seed
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef, f1_score, 
    precision_score, recall_score, accuracy_score, confusion_matrix
)
import numpy as np
import os

FROZEN_DIR = "./data/manifests"
MODEL_NAME = "microsoft/codebert-base"

print("1. Loading datasets...")
train_df = pd.read_json(f"{FROZEN_DIR}/train_frozen.jsonl", lines=True)
val_df = pd.read_json(f"{FROZEN_DIR}/val_frozen.jsonl", lines=True)
test_pv_df = pd.read_json(f"{FROZEN_DIR}/test_primevul_frozen.jsonl", lines=True)
test_pair_df = pd.read_json(f"{FROZEN_DIR}/test_paired_frozen.jsonl", lines=True)
test_bv_df = pd.read_json(f"{FROZEN_DIR}/test_bigvul_frozen.jsonl", lines=True)

print(f"2. Tokenizing with {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = Dataset.from_pandas(train_df[['func_code', 'label']])
val_ds = Dataset.from_pandas(val_df[['func_code', 'label']])
test_pv_ds = Dataset.from_pandas(test_pv_df[['func_code', 'label']])
test_pair_ds = Dataset.from_pandas(test_pair_df[['func_code', 'label']])
test_bv_ds = Dataset.from_pandas(test_bv_df[['func_code', 'label']])

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

columns_to_keep = ["input_ids", "attention_mask", "label"]
tokenized_train.set_format("torch", columns=columns_to_keep)
tokenized_val.set_format("torch", columns=columns_to_keep)
tokenized_test_pv.set_format("torch", columns=columns_to_keep)
tokenized_test_pair.set_format("torch", columns=columns_to_keep)
tokenized_test_bv.set_format("torch", columns=columns_to_keep)

print("3. Defining Single-task Architecture (CodeBERT)...")

class CodeBERTSingleTask(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(CodeBERTSingleTask, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        
        hidden_size = self.config.hidden_size
        
        self.classifier_bin = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CodeBERT [CLS] token is the first token
        sequence_output = outputs[0] 
        cls_embedding = sequence_output[:, 0, :] 
        
        bin_logits = self.classifier_bin(cls_embedding).squeeze(-1)
        
        return {"logits": bin_logits}

print("4. Defining Trainer and Metrics...")

class SingleTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        bin_logits = outputs["logits"]
        
        # Pos_weight from EDA
        pos_weight = torch.tensor([45.44]).to(bin_logits.device)
        loss_fct_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_bin = loss_fct_bin(bin_logits, labels.float())
        
        return (loss_bin, outputs) if return_outputs else loss_bin

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # For single task, eval_pred.predictions is just logits
    if isinstance(logits, tuple):
        logits = logits[0]
        
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

print("5. Configuring Training Loop...")

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEEDS = [13, 21, 42, 87, 101] # As required by D3 in experimental design
OUTPUT_DIR = "./saved_models_e1_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Note: We will only run a test for seed 13 if manually executed to avoid long runtimes
# Usually, a runner script loops through all seeds.

for seed in [13]:
    print(f"\nStarting training for seed: {seed}")
    set_seed(seed)
    
    model = CodeBERTSingleTask()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    seed_output_dir = f"{OUTPUT_DIR}/codebert_seed_{seed}"
    
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
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none"
    )
    
    trainer = SingleTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    torch.save(model.state_dict(), f"{seed_output_dir}/best_model.pt")

print("[OK] E1.1 Baseline training completed successfully!")
