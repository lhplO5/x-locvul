import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import os
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_vulnerable_data_with_cwe(jsonl_path):
    # Only load vulnerable functions (label == 1)
    df = pd.read_json(jsonl_path, lines=True)
    if 'label' in df.columns:
        df = df[df['label'] == 1].copy()
    
    if 'vul_lines' not in df.columns:
        df['vul_lines'] = "Missing Ground Truth" 
    
    if 'cwe' not in df.columns:
        df['cwe'] = "CWE-Other"
        
    # Convert list of lines to a single string
    df['target_text'] = df['vul_lines'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    
    # Prepend CWE to the function code
    df['input_text'] = df.apply(lambda row: f"<{row['cwe']}> {row['func_code']}", axis=1)
    
    return df[['input_text', 'target_text']]

FROZEN_DIR = "./data/manifests"
OUTPUT_DIR = "./saved_models_e2_4"
MODEL_NAME = "Salesforce/codet5-base"

def run_e2_4(seed):
    print(f"\n{'='*50}")
    print(f"Starting E2.4/E2.5: CWE-Conditioned CodeT5 (Seed {seed})")
    print(f"{'='*50}")
    set_seed(seed)

    print("1. Loading ONLY vulnerable functions & Prepending Ground-Truth CWE...")
    train_df = load_vulnerable_data_with_cwe(f"{FROZEN_DIR}/train_frozen.jsonl")
    val_df = load_vulnerable_data_with_cwe(f"{FROZEN_DIR}/val_frozen.jsonl")
    
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add special tokens for CWE if needed (optional, but good practice)
    # We will just rely on standard tokenization for "<CWE-119>" etc.

    def tokenize_function(examples):
        # Input: <CWE-XXX> Source Code
        inputs = tokenizer(examples["input_text"], max_length=512, padding="max_length", truncation=True)
        # Target: Vulnerable Line String
        targets = tokenizer(examples["target_text"], max_length=128, padding="max_length", truncation=True)
        
        # Replace pad_token_id with -100 to ignore in loss calculation
        labels = targets["input_ids"]
        labels = [[-100 if token == tokenizer.pad_token_id else token for token in l] for l in labels]
        
        inputs["labels"] = labels
        return inputs

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=val_ds.column_names)
    
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    print("2. Initializing CodeT5 Model...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    run_output_dir = f"{OUTPUT_DIR}/seed_{seed}"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=10, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=1
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    print("3. Training Model...")
    trainer.train()
    
    print(f"Saving best model to {run_output_dir}...")
    trainer.save_model(run_output_dir)
    print("Training Complete!")

if __name__ == "__main__":
    run_e2_4(seed=13)
