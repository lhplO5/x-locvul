import pandas as pd
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SAMPLE_FILE = "./data/processed/e3_sample_180.jsonl"
OUT_DIR = "./results_e3"
OUT_FILE = f"{OUT_DIR}/qwen_ablations.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

PROMPT_A1 = """You are a software security expert. Analyze the following C/C++ function and explain its vulnerability.
Provide your response strictly using these four headers:
### Vulnerability Type
### Source Evidence
### Root Cause
### Repair

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

Function:
```c
{func_code}
```"""

def run_explanations():
    print("==================================================")
    print(" E3: Generating Explanations with Qwen2.5-Coder ")
    print("==================================================")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    df = pd.read_json(SAMPLE_FILE, lines=True)
    
    # We will simulate the generations for a small batch to test the pipeline.
    # In a real run, you iterate over all 180 cases.
    df_to_run = df.head(3) # Limit for dry-run
    
    results = []
    
    for idx, row in df_to_run.iterrows():
        func_code = row['func_code']
        gt_cwe = row.get('cwe', 'Unknown')
        gt_line = row.get('vul_lines', [''])[0] if isinstance(row.get('vul_lines'), list) else row.get('vul_lines', '')
        
        # E1/E2 mock predictions (in a real pipeline, we load them from E1/E2 results)
        pred_cwe = gt_cwe # Simulated E1 prediction
        pred_line = gt_line # Simulated E2 prediction
        
        ablations = {
            "A1_FunctionOnly": PROMPT_A1.format(func_code=func_code),
            "A2_FunctionLine": PROMPT_A2.format(func_code=func_code, line=pred_line),
            "A3_FunctionLineCWE": PROMPT_A3.format(func_code=func_code, line=pred_line, cwe=pred_cwe),
            "A4_Oracle": PROMPT_A3.format(func_code=func_code, line=gt_line, cwe=gt_cwe)
        }
        
        case_results = {
            "idx": int(row.get('idx', idx)),
            "func_hash": row.get('func_hash', ''),
            "gt_cwe": gt_cwe,
            "gt_line": gt_line,
            "pred_cwe": pred_cwe,
            "pred_line": pred_line,
            "explanations": {}
        }
        
        for ablation_name, prompt_text in ablations.items():
            messages = [
                {"role": "system", "content": "You are a helpful and precise security assistant."},
                {"role": "user", "content": prompt_text}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # Temperature = 0 for deterministic generation (E3 requirement)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    temperature=0.0, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            case_results["explanations"][ablation_name] = response
            print(f"Generated {ablation_name} for case {idx}")
            
        results.append(case_results)
        
    with open(OUT_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
            
    print(f"Finished generating explanations. Saved to {OUT_FILE}")

if __name__ == "__main__":
    run_explanations()
