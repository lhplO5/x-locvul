import os
import random
import numpy as np
import pandas as pd
import torch
from transformers import (
    T5ForConditionalGeneration, 
    AutoTokenizer, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate

DATA_DIR = "./data/linevul"
MODEL_NAME = "Salesforce/codet5-base"
OUTPUT_DIR = "./models"
SEEDS = [10, 42, 100, 420, 1000]

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

exact_match_metric = evaluate.load("exact_match")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    preds = np.clip(preds, 0, len(tokenizer) - 1)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    em_result = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "exact_match": em_result["exact_match"]
    }

def load_and_prepare_data(tokenizer):
    print("Loading data...")
    train_df = pd.read_csv(f"{DATA_DIR}/train_cleaned.csv")
    val_df = pd.read_csv(f"{DATA_DIR}/val_cleaned.csv")
    
    train_df = train_df[train_df['target'] == 1].copy()
    val_df = val_df[val_df['target'] == 1].copy()
    
    print(f"Training samples (Target=1): {len(train_df)}")
    print(f"Validation samples (Target=1): {len(val_df)}")
    
    train_df = train_df.dropna(subset=['func_code', 'flaw_line'])
    val_df = val_df.dropna(subset=['func_code', 'flaw_line'])
    
    train_ds = Dataset.from_pandas(train_df[['func_code', 'flaw_line']])
    val_ds = Dataset.from_pandas(val_df[['func_code', 'flaw_line']])
    
    def preprocess_function(examples):
        inputs = examples['func_code']
        targets = examples['flaw_line']
        model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    train_tokenized = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)
    
    return train_tokenized, val_tokenized

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    train_tokenized, val_tokenized = load_and_prepare_data(tokenizer)
    
    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING FOR SEED: {seed}")
        print(f"{'='*50}")
        
        set_seed(seed)
        
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        run_name = f"e3_locator_seed_{seed}"
        output_dir = os.path.join(OUTPUT_DIR, run_name)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            save_total_limit=1,
            num_train_epochs=MAX_EPOCHS,
            predict_with_generate=True, 
            generation_max_length=MAX_TARGET_LENGTH,
            load_best_model_at_end=True,
            metric_for_best_model="exact_match",
            greater_is_better=True,
            seed=seed,
            fp16=torch.cuda.is_available(), 
            bf16=False,
            logging_steps=50,
            dataloader_num_workers=4,
            report_to="none"
        )
        
        callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        print(f"Training Seed {seed}")
        trainer.train()
        
        print(f"Saving best model for Seed {seed}")
        trainer.save_model(os.path.join(OUTPUT_DIR, f"{run_name}_best"))
        print(f"Seed {seed} completed!")

if __name__ == "__main__":
    main()
