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

print("1. Loading datasets...")
train_df = pd.read_json(f"{FROZEN_DIR}/train_frozen.jsonl", lines=True)
val_df = pd.read_json(f"{FROZEN_DIR}/val_frozen.jsonl", lines=True)
test_pv_df = pd.read_json(f"{FROZEN_DIR}/test_primevul_frozen.jsonl", lines=True)

# Define CWEs
vul_train = train_df[train_df['label'] == 1].copy()
if 'cwe_label' not in vul_train.columns:
    vul_train['cwe_label'] = 'CWE-Other'
vul_train['cwe_label'] = vul_train['cwe_label'].fillna('CWE-Other')
unique_cwes = sorted(vul_train['cwe_label'].unique().tolist())
cwe2id = {cwe: idx for idx, cwe in enumerate(unique_cwes)}
NUM_CWE = len(unique_cwes)

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

print(f"2. Tokenizing with {MODEL_NAME}...")
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
    tokenized = ds.map(tokenize_function, batched=True, remove_columns=['func_code'])
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
    
    return {"accuracy": accuracy, "mcc": mcc, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "pr_auc": pr_auc}

def run_stage_1_and_2(seed):
    print(f"\n======================================")
    print(f"Starting E1.4 Hard Mining for seed {seed}")
    print(f"======================================")
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_stage1 = UniXCoderMultiTask()
    model_stage1.to(device)
    
    EPOCHS_S1 = 1
    BATCH_SIZE = 16
    OUTPUT_DIR_S1 = f"./saved_models_e1_4/stage1_seed_{seed}"
    os.makedirs(OUTPUT_DIR_S1, exist_ok=True)
    
    training_args_s1 = TrainingArguments(
        output_dir=OUTPUT_DIR_S1,
        num_train_epochs=EPOCHS_S1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="no", 
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer_s1 = MultiTaskTrainer(
        model=model_stage1,
        args=training_args_s1,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    
    print("\n--- STAGE 1: Standard Training ---")
    trainer_s1.train()
    
    print("\n--- STAGE 1: Generating Inference on Train Set ---")
    # Generate predictions on train set to find hard examples
    train_preds = trainer_s1.predict(tokenized_train)
    bin_logits = train_preds.predictions[0] if isinstance(train_preds.predictions, tuple) else train_preds.predictions
    probs = 1.0 / (1.0 + np.exp(-bin_logits))
    
    # FP: True label=0, Prob >= 0.90
    # FN: True label=1, Prob <= 0.10
    labels_np = train_df['label'].values
    hard_fp_idx = np.where((labels_np == 0) & (probs >= 0.90))[0]
    hard_fn_idx = np.where((labels_np == 1) & (probs <= 0.10))[0]
    
    print(f"Found {len(hard_fp_idx)} Hard False Positives and {len(hard_fn_idx)} Hard False Negatives.")
    
    # We want hard examples to be ~20% of the new dataset
    # So normal examples = 80%, hard examples = 20%
    # Normal examples count = len(train_df)
    # Target hard examples count = 0.25 * len(train_df)
    total_hard_needed = int(0.25 * len(train_df))
    
    all_hard_idx = np.concatenate([hard_fp_idx, hard_fn_idx])
    if len(all_hard_idx) == 0:
        print("No hard examples found. Using original dataset for Stage 2.")
        upsampled_train_df = train_df.copy()
    else:
        # Upsample the hard examples to reach the target count
        times_to_repeat = total_hard_needed // len(all_hard_idx)
        times_to_repeat = max(1, times_to_repeat) # at least 1
        
        hard_df = train_df.iloc[all_hard_idx]
        hard_df_repeated = pd.concat([hard_df] * times_to_repeat, ignore_index=True)
        
        # Shuffle to mix hard and normal examples
        upsampled_train_df = pd.concat([train_df, hard_df_repeated]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print(f"Stage 2 Train Size: {len(upsampled_train_df)} (Original: {len(train_df)})")
    
    # Free memory
    del trainer_s1
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n--- STAGE 2: Hard Mining Training ---")
    tokenized_train_s2 = df_to_tokenized_dataset(upsampled_train_df)
    
    EPOCHS_S2 = 2 # Finish the rest of the epochs
    OUTPUT_DIR_S2 = f"./saved_models_e1_4/stage2_seed_{seed}"
    os.makedirs(OUTPUT_DIR_S2, exist_ok=True)
    
    training_args_s2 = TrainingArguments(
        output_dir=OUTPUT_DIR_S2,
        num_train_epochs=EPOCHS_S2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer_s2 = MultiTaskTrainer(
        model=model_stage1, # Warm start (same initialization)
        args=training_args_s2,
        train_dataset=tokenized_train_s2,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    
    trainer_s2.train()
    torch.save(model_stage1.state_dict(), f"{OUTPUT_DIR_S2}/best_model.pt")
    
    print(f"[OK] Seed {seed} completed successfully.")

if __name__ == "__main__":
    for seed in [13]: # Run 1 seed for test
        run_stage_1_and_2(seed)
