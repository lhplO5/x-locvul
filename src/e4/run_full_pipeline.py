import os
import json
import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, AutoModelForCausalLM
import gc
import difflib

STAGE1_WEIGHTS = "./saved_models_e1_4/stage2_seed_42/best_model.pt"
STAGE2_REPO = "jsbe/x-locvul-e3-locator"
STAGE3_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_FILE = "./data/processed/e4_sample_150_linevul.jsonl"
OUTPUT_FILE = "./results/e4/full_pipeline_results.jsonl"
CHECKPOINT_FILE = "./results/e4/full_pipeline_checkpoint.jsonl"
MANIFEST_FILE = "./data/manifests/train_frozen.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_A1 = """You are a software security expert. Analyze the following C/C++ function and explain its vulnerability.
Provide your response strictly using these four headers:
### Vulnerability Type
### Source Evidence
### Root Cause
### Repair

Keep each section concise. For Repair, you may include a short corrected code snippet.

Function:
```c
{func_code}
```"""

PROMPT_A2 = """You are a software security expert. Analyze the following C/C++ function.
A vulnerability is suspected near this line: `{line}`.
Provide your response strictly using these four headers:
### Vulnerability Type
### Source Evidence
### Root Cause
### Repair

Keep each section concise. For Repair, you may include a short corrected code snippet.

Function:
```c
{func_code}
```"""

PROMPT_A3 = """You are a software security expert. Analyze the following C/C++ function.
The vulnerability is predicted as `{cwe}` and is located near this line: `{line}`.
Provide your response strictly using these four headers:
### Vulnerability Type
### Source Evidence
### Root Cause
### Repair

Keep each section concise. For Repair, you may include a short corrected code snippet.

Function:
```c
{func_code}
```"""


#STAGE 1: UniXCoder Multi-Task (Detector)
def get_cwe_mapping():
    print("[STAGE 1 PREP] Loading CWE mapping from train_frozen.jsonl")
    train_df = pd.read_json(MANIFEST_FILE, lines=True)
    vul_train = train_df[train_df['label'] == 1].copy()
    vul_train['cwe_label'] = vul_train['cwe_label'].fillna('CWE-Other')
    unique_cwes = sorted(vul_train['cwe_label'].unique().tolist())
    id2cwe = {idx: cwe for idx, cwe in enumerate(unique_cwes)}
    print(f"[STAGE 1 PREP] Found {len(unique_cwes)} CWE classes.")
    return id2cwe, len(unique_cwes)

id2cwe, NUM_CWE = get_cwe_mapping()

class UniXCoderMultiTask(nn.Module):
    def __init__(self, model_name="microsoft/unixcoder-base", num_cwe=NUM_CWE):
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

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :]
        bin_logits = self.classifier_bin(cls_embedding).squeeze(-1)
        cwe_logits = self.classifier_cwe(cls_embedding)
        return bin_logits, cwe_logits

def run_stage1(samples):
    print("\n" + "="*60)
    print("STAGE 1: VULNERABILITY DETECTION & CWE CLASSIFICATION")
    print("="*60)
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = UniXCoderMultiTask(num_cwe=NUM_CWE)
    
    print(f"[STAGE 1] Loading weights from {STAGE1_WEIGHTS}")
    model.load_state_dict(torch.load(STAGE1_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    print(f"[STAGE 1] Model loaded on {device}. Starting inference on {len(samples)} samples\n")

    vul_count = 0
    for i, item in enumerate(samples):
        func_code = item['func_code']
        inputs = tokenizer(func_code, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            bin_logits, cwe_logits = model(inputs.input_ids, inputs.attention_mask)
        
        prob = torch.sigmoid(bin_logits).item()
        cwe_pred_id = torch.argmax(cwe_logits, dim=-1).item()
        
        item['pred_vul'] = int(prob > 0.5)
        item['pred_prob'] = round(prob, 4)
        item['pred_cwe'] = id2cwe[cwe_pred_id]
        
        if item['pred_vul'] == 1:
            vul_count += 1
        
        print(f"  [S1] [{i+1}/{len(samples)}] pred_vul={item['pred_vul']} (prob={item['pred_prob']:.4f}) pred_cwe={item['pred_cwe']}")
            
    elapsed = time.time() - t0
    print(f"\n[STAGE 1 DONE] {len(samples)} samples processed in {elapsed:.1f}s. Detected {vul_count}/{len(samples)} as vulnerable.")
    print("[STAGE 1] Cleaning up memory")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return samples

#STAGE 2: CodeT5 (Locator)
def source_constrained_projection(generated_line, func_code):
    """Matches the generated line to the most similar line in the actual source code."""
    lines = func_code.split('\n')
    best_match = ""
    highest_ratio = 0.0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        ratio = difflib.SequenceMatcher(None, generated_line.strip(), stripped).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = line
    return best_match.strip() if highest_ratio > 0.5 else generated_line

def run_stage2(samples):
    print("\n" + "="*60)
    print("STAGE 2: VULNERABILITY LOCALIZATION (CodeT5)")
    print("="*60)
    
    t0 = time.time()
    print(f"[STAGE 2] Loading model from Hugging Face: {STAGE2_REPO}")
    tokenizer = AutoTokenizer.from_pretrained(STAGE2_REPO, use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(STAGE2_REPO)
    model.to(device)
    model.eval()
    print(f"[STAGE 2] Model loaded on {device}. Starting inference on {len(samples)} samples\n")

    for i, item in enumerate(samples):
        func_code = item['func_code']
        inputs = tokenizer(func_code, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=256)
        
        raw_pred_line = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        item['raw_pred_line'] = raw_pred_line
        
        # Apply Source-Constrained Projection
        item['pred_line'] = source_constrained_projection(raw_pred_line, func_code)
        
        # Ground truth line
        gt_line = item.get('flaw_line', item.get('vul_lines', ''))
        if isinstance(gt_line, list):
            gt_line = gt_line[0] if gt_line else ''
        item['gt_line'] = gt_line
        
        # Ground truth CWE
        item['gt_cwe'] = item.get('cwe', item.get('cwe_label', 'CWE-Other'))
        
        # Quick match check for logging
        match_status = "EXACT" if item['pred_line'].strip() == str(gt_line).strip() else "MISMATCH"
        print(f"  [S2] [{i+1}/{len(samples)}] match={match_status} | raw=\"{raw_pred_line[:60]}...\" | proj=\"{item['pred_line'][:60]}...\"")
            
    elapsed = time.time() - t0
    print(f"\n[STAGE 2 DONE] {len(samples)} samples processed in {elapsed:.1f}s.")
    print("[STAGE 2] Cleaning up memory")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return samples

# STAGE 3: Qwen2.5-Coder (Explanation)
def run_stage3(samples):
    print("\n" + "="*60)
    print("STAGE 3: EXPLANATION GENERATION (Qwen2.5-Coder-1.5B)")
    print("="*60)
    
    t0 = time.time()
    print(f"[STAGE 3] Loading model: {STAGE3_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(STAGE3_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        STAGE3_MODEL, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    model_device = next(model.parameters()).device
    print(f"[STAGE 3] Model loaded on {model_device}. Starting generation for {len(samples)} samples x 4 conditions = {len(samples)*4} total\n")

    results = []
    resume_idx = 0
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"[STAGE 3] Found existing checkpoint at {CHECKPOINT_FILE}. Loading...")
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        # Each sample produces 4 conditions, so resume_idx = len(results) // 4
        resume_idx = len(results) // 4
        print(f"[STAGE 3] Resuming from sample index {resume_idx}")
    
    for i, item in enumerate(samples):
        if i < resume_idx:
            continue
            
        func_code = item.get('func_code', '')
        pred_cwe = item.get('pred_cwe', 'CWE-Other')
        pred_line = item.get('pred_line', '')
        oracle_cwe = item.get('gt_cwe', 'CWE-Other')
        oracle_line = item.get('gt_line', '')
        
        ablations = {
            "A1_FunctionOnly": PROMPT_A1.format(func_code=func_code),
            "A2_FunctionLine": PROMPT_A2.format(func_code=func_code, line=pred_line),
            "A3_FunctionLineCWE": PROMPT_A3.format(func_code=func_code, line=pred_line, cwe=pred_cwe),
            "A4_Oracle": PROMPT_A3.format(func_code=func_code, line=oracle_line, cwe=oracle_cwe)
        }
        
        for cond_name, prompt_text in ablations.items():
            messages = [
                {"role": "system", "content": "You are a helpful and precise security assistant."},
                {"role": "user", "content": prompt_text}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model_device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=768, 
                    temperature=0.0, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Build output row with all required columns
            res_item = {
                "idx": item.get('idx', i),
                "func_hash": item.get('func_hash', ''),
                "func_code": func_code,
                "gt_cwe": oracle_cwe,
                "gt_line": oracle_line,
                "pred_vul": item.get('pred_vul', 1),
                "pred_prob": item.get('pred_prob', 0.0),
                "pred_cwe": pred_cwe,
                "raw_pred_line": item.get('raw_pred_line', ''),
                "pred_line": pred_line,
                "condition": cond_name,
                "explanation": response.strip()
            }
            results.append(res_item)
            
            print(f"  [S3] [{i+1}/{len(samples)}] {cond_name} -> {len(response.strip())} chars generated")
        
        # Checkpoint: save every 10 cases to prevent data loss
        if (i+1) % 10 == 0:
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            elapsed = time.time() - t0
            print(f"  [S3 CHECKPOINT] Saved {len(results)} rows after {i+1}/{len(samples)} cases ({elapsed:.1f}s elapsed)\n")
            
    elapsed = time.time() - t0
    print(f"\n[STAGE 3 DONE] {len(results)} explanation rows generated in {elapsed:.1f}s.")
    print("[STAGE 3] Cleaning up memory")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results

def main():
    os.makedirs("./results_e4", exist_ok=True)
    
    print("="*60)
    print("  X-LOCVUL FULL END-TO-END PIPELINE")
    print("="*60)
    print(f"  Stage 1 weights : {STAGE1_WEIGHTS}")
    print(f"  Stage 2 model   : {STAGE2_REPO}")
    print(f"  Stage 3 model   : {STAGE3_MODEL}")
    print(f"  Input data      : {DATA_FILE}")
    print(f"  Output file     : {OUTPUT_FILE}")
    print(f"  Device          : {device}")
    print("="*60)
    
    print(f"\n[MAIN] Loading data from {DATA_FILE}...")
    samples = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            samples.append(json.loads(line))
    print(f"[MAIN] Loaded {len(samples)} samples.\n")
    
    total_t0 = time.time()
    
    samples = run_stage1(samples)
    samples = run_stage2(samples)
    results = run_stage3(samples)
    
    print(f"\n[MAIN] Saving final results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  [DONE] Full Pipeline completed successfully!")
    print(f"  Total time     : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Total rows     : {len(results)} (= {len(samples)} cases x 4 conditions)")
    print(f"  Output saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
