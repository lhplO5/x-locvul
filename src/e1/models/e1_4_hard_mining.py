import pandas as pd
import torch
import torch.nn as nn
import math
from datasets import Dataset
from torch.utils.data import Sampler, DataLoader
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

print(f"2. Tokenizing with {MODEL_NAME}")
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

class HardMiningBatchSampler(Sampler):
    """
    Yields batches where hard samples occupy at most max_hard_ratio (20%) of the batch.
    """
    def __init__(self, normal_indices, hard_indices, batch_size=16, max_hard_ratio=0.20, seed=13):
        self.normal_indices = np.array(normal_indices)
        self.hard_indices = np.array(hard_indices)
        self.batch_size = batch_size
        self.max_hard = max(1, int(batch_size * max_hard_ratio))  # e.g., 3 for batch=16
        self.min_normal = batch_size - self.max_hard              # e.g., 13
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        # Shuffle indices for every epoch
        shuffled_normal = self.rng.permutation(self.normal_indices)
        shuffled_hard = self.rng.permutation(self.hard_indices) if len(self.hard_indices) > 0 else np.array([], dtype=int)

        n_batches = len(shuffled_normal) // self.min_normal
        hard_ptr = 0

        for b in range(n_batches):
            batch = []
            # 1. Take normal samples
            batch.extend(shuffled_normal[b * self.min_normal : (b + 1) * self.min_normal])

            # 2. Take at most 20% hard samples (with cyclic reuse if hard samples are fewer)
            if len(shuffled_hard) > 0:
                for _ in range(self.max_hard):
                    batch.append(shuffled_hard[hard_ptr % len(shuffled_hard)])
                    hard_ptr += 1

            # Shuffle items within the single batch
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.normal_indices) // self.min_normal


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

class HardMiningTrainer(MultiTaskTrainer):
    """Trainer subclass that injects the HardMiningBatchSampler into the train DataLoader."""
    def __init__(self, *args, hard_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._hard_sampler = hard_sampler

    def get_train_dataloader(self):
        if self._hard_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self._hard_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return super().get_train_dataloader()

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


def run_stage_1_and_2(seed):
    print(f"Starting E1.4 Hard Mining for seed {seed}")
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_stage1 = UniXCoderMultiTask(num_cwe=NUM_CWE)
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
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        report_to="none"
    )
    
    trainer_s1 = MultiTaskTrainer(
        model=model_stage1,
        args=training_args_s1,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    
    print("\nSTAGE 1: Standard Training")
    trainer_s1.train()
    
    print("\nSTAGE 1: Generating Inference on Train Set")
    # Generate predictions on train set to find hard examples
    train_preds = trainer_s1.predict(tokenized_train)
    bin_logits = train_preds.predictions[0] if isinstance(train_preds.predictions, tuple) else train_preds.predictions
    probs = 1.0 / (1.0 + np.exp(-bin_logits))
    
    # FP: True label=0, Prob >= 0.90
    # FN: True label=1, Prob <= 0.10
    labels_np = train_df['label'].values
    hard_fp_idx = np.where((labels_np == 0) & (probs >= 0.90))[0]
    hard_fn_idx = np.where((labels_np == 1) & (probs <= 0.10))[0]
    all_hard_idx = np.concatenate([hard_fp_idx, hard_fn_idx])
    all_normal_idx = np.array([i for i in range(len(train_df)) if i not in set(all_hard_idx)])
    
    print(f"Found {len(hard_fp_idx)} Hard FP + {len(hard_fn_idx)} Hard FN = {len(all_hard_idx)} total hard samples.")
    print(f"Normal samples: {len(all_normal_idx)}")
    
    # Free Stage 1 trainer memory
    del trainer_s1
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nSTAGE 2: Hard Mining Training (Custom Batch Sampler, ≤20% hard/batch)")
    
    EPOCHS_S2 = 2
    OUTPUT_DIR_S2 = f"./saved_models_e1_4/stage2_seed_{seed}"
    os.makedirs(OUTPUT_DIR_S2, exist_ok=True)
    
    hard_sampler = HardMiningBatchSampler(
        normal_indices=all_normal_idx,
        hard_indices=all_hard_idx,
        batch_size=BATCH_SIZE,
        max_hard_ratio=0.20,
        seed=seed
    )
    
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
    
    trainer_s2 = HardMiningTrainer(
        model=model_stage1,
        args=training_args_s2,
        train_dataset=tokenized_train, 
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        hard_sampler=hard_sampler
    )
    
    trainer_s2.train()
    torch.save(model_stage1.state_dict(), f"{OUTPUT_DIR_S2}/best_model.pt")
    tokenizer.save_pretrained(OUTPUT_DIR_S2)
    
    del model_stage1, trainer_s2, training_args_s2
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Seed {seed} completed successfully.")

if __name__ == "__main__":
    for seed in [13, 21, 42, 87, 101]:
        run_stage_1_and_2(seed)

