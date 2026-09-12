import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel, Trainer, TrainingArguments
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef, f1_score, 
    precision_score, recall_score, accuracy_score, confusion_matrix,
    roc_auc_score
)
import numpy as np
import os
import gc

DATA_DIR = "./data/processed"
MODEL_NAME = "microsoft/unixcoder-base"
SEEDS = [13, 21, 42, 87, 101]

class UniXCoderSingleTask(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(UniXCoderSingleTask, self).__init__()
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
        sequence_output = outputs[0] 
        cls_embedding = sequence_output[:, 0, :] 
        bin_logits = self.classifier_bin(cls_embedding).squeeze(-1)
        
        output = {"logits": bin_logits}
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            output["loss"] = loss_fct(bin_logits, labels.float())
            
        return output

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
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
    roc_auc = roc_auc_score(labels, probs)
    
    return {
        "Accuracy": accuracy,
        "MCC": mcc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "raw_probs": probs,
        "raw_labels": labels
    }

def main():
    print("1. Loading test dataset")
    test_df = pd.read_csv(f"{DATA_DIR}/test_chrono.csv")
    test_df = test_df.dropna(subset=['func', 'target'])
    test_df['label'] = test_df['target'].astype(int)
    test_df['func_code'] = test_df['func'].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_ds = Dataset.from_pandas(test_df[['func_code', 'label']])
    
    def tokenize_function(examples):
        return tokenizer(examples["func_code"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_test = test_ds.map(tokenize_function, batched=True, remove_columns=['func_code', '__index_level_0__'] if '__index_level_0__' in test_ds.column_names else ['func_code'])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print("Loading val dataset for recalibration...")
    val_df = pd.read_csv(f"{DATA_DIR}/val_chrono.csv")
    val_df = val_df.dropna(subset=['func', 'target'])
    val_df['label'] = val_df['target'].astype(int)
    val_df['func_code'] = val_df['func'].astype(str)
    val_ds = Dataset.from_pandas(val_df[['func_code', 'label']])
    tokenized_val = val_ds.map(tokenize_function, batched=True, remove_columns=['func_code', '__index_level_0__'] if '__index_level_0__' in val_ds.column_names else ['func_code'])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    results = []
    
    for seed in SEEDS:
        print(f"\nEvaluating Seed: {seed}")
        model_path = f"./saved_models_e5/detector_seed_{seed}/best_model.pt"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Skipping.")
            continue
            
        model = UniXCoderSingleTask()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=32, report_to="none"),
            compute_metrics=compute_metrics
        )
        
        metrics = trainer.evaluate(tokenized_test)
        
        row = {"Seed": seed}
        for k, v in metrics.items():
            if k.startswith("eval_") and k not in ["eval_raw_probs", "eval_raw_labels"]:
                row[k.replace("eval_", "")] = v
                
        results.append(row)

        np.savez_compressed(
            f"./results/cache_E5_seed_{seed}.npz",
            probs=metrics["eval_raw_probs"],
            labels=metrics["eval_raw_labels"]
        )
        
        # Evaluate on Validation Set for recalibration
        val_metrics = trainer.evaluate(tokenized_val)
        np.savez_compressed(
            f"./results/cache_E5_val_seed_{seed}.npz",
            probs=val_metrics["eval_raw_probs"],
            labels=val_metrics["eval_raw_labels"]
        )
        
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    if results:
        results_df = pd.DataFrame(results)
        
        # Remove loss columns if they exist
        cols_to_drop = [c for c in results_df.columns if 'loss' in c.lower()]
        results_df = results_df.drop(columns=cols_to_drop)
        
        # Calculate Average
        avg_row = results_df.mean().to_dict()
        avg_row["Seed"] = "AVERAGE"
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        
        print("\n" + "="*80)
        print("  FINAL E5 DETECTION RESULTS (CHRONOLOGICAL SPLIT)")
        print("="*80)
        print(results_df.round(4).to_string(index=False))
        print("="*80)
        
        os.makedirs("./outputs/e5", exist_ok=True)
        results_df.to_csv("./outputs/e5/per_seed_aggregate_metrics.csv", index=False)
        print("\nResults saved to ./outputs/e5/per_seed_aggregate_metrics.csv")
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    main()
