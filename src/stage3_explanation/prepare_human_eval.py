import pandas as pd
import json
import os
import random
import re

IN_FILE = "./results_e3/qwen_ablations.jsonl"
OUT_DIR = "./results_e3"

def parse_explanation(text):
    """
    Extracts the 4 fields from the markdown explanation.
    """
    fields = {
        "Vulnerability Type": "",
        "Source Evidence": "",
        "Root Cause": "",
        "Repair": ""
    }
    
    current_field = None
    lines = text.split('\n')
    
    for line in lines:
        if line.startswith("### Vulnerability Type"):
            current_field = "Vulnerability Type"
        elif line.startswith("### Source Evidence"):
            current_field = "Source Evidence"
        elif line.startswith("### Root Cause"):
            current_field = "Root Cause"
        elif line.startswith("### Repair"):
            current_field = "Repair"
        elif current_field:
            fields[current_field] += line + "\n"
            
    for k in fields:
        fields[k] = fields[k].strip()
        
    return fields

def validate_cwe_consistency(llm_vuln_type, anchor_cwe):
    """
    Lightweight validator to flag if the stated vulnerability type contradicts the Stage-1 CWE anchor.
    Returns True if consistent, False if flagged as contradiction.
    """
    if not anchor_cwe or anchor_cwe == "Unknown":
        return True # Can't contradict if anchor is unknown
        
    llm_lower = llm_vuln_type.lower()
    anchor_lower = anchor_cwe.lower()
    
    # Simple check: Does the LLM output contain the CWE ID?
    if anchor_lower in llm_lower:
        return True
        
    # Heuristic mapping for common CWEs to keywords
    cwe_keywords = {
        "cwe-119": ["buffer", "memory", "bounds", "overflow"],
        "cwe-125": ["out-of-bounds", "read", "buffer"],
        "cwe-787": ["out-of-bounds", "write", "buffer"],
        "cwe-476": ["null", "pointer", "dereference"],
        "cwe-416": ["use-after-free", "uaf", "free"],
        "cwe-20": ["input", "validation", "parameter"],
        "cwe-89": ["sql", "injection"],
        "cwe-79": ["xss", "cross-site scripting"]
    }
    
    base_cwe = anchor_lower.split()[0] if anchor_lower else ""
    if base_cwe in cwe_keywords:
        for kw in cwe_keywords[base_cwe]:
            if kw in llm_lower:
                return True
                
    # If no keywords matched, flag it (returns False)
    return False

def prepare():
    print("==================================================")
    print(" E3: Preparing Blinded Human Evaluation CSVs ")
    print("==================================================")
    
    if not os.path.exists(IN_FILE):
        print(f"Error: {IN_FILE} not found. Run e3_qwen_explainer.py first.")
        # Create a dummy file for dry-run if needed
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(IN_FILE, 'w') as f:
            f.write(json.dumps({
                "idx": 0, "func_hash": "abc", "gt_cwe": "CWE-119", "gt_line": "x = 1;", "pred_cwe": "CWE-119", "pred_line": "x = 1;",
                "explanations": {
                    "A1_FunctionOnly": "### Vulnerability Type\nBuffer Overflow\n### Source Evidence\nline\n### Root Cause\nbad\n### Repair\ngood",
                    "A2_FunctionLine": "### Vulnerability Type\nBuffer Overflow\n### Source Evidence\nline\n### Root Cause\nbad\n### Repair\ngood",
                    "A3_FunctionLineCWE": "### Vulnerability Type\nCWE-119\n### Source Evidence\nline\n### Root Cause\nbad\n### Repair\ngood",
                    "A4_Oracle": "### Vulnerability Type\nCWE-119\n### Source Evidence\nline\n### Root Cause\nbad\n### Repair\ngood"
                }
            }) + "\n")
            
    records = []
    
    with open(IN_FILE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            idx = data['idx']
            
            for ablation, explanation in data['explanations'].items():
                fields = parse_explanation(explanation)
                
                # Determine the Anchor CWE based on ablation
                anchor_cwe = None
                if ablation == "A3_FunctionLineCWE":
                    anchor_cwe = data['pred_cwe']
                elif ablation == "A4_Oracle":
                    anchor_cwe = data['gt_cwe']
                    
                is_consistent = validate_cwe_consistency(fields['Vulnerability Type'], anchor_cwe) if anchor_cwe else True
                
                records.append({
                    "original_idx": idx,
                    "ablation_name": ablation,
                    "Vulnerability_Type": fields['Vulnerability Type'],
                    "Source_Evidence": fields['Source Evidence'],
                    "Root_Cause": fields['Root Cause'],
                    "Repair": fields['Repair'],
                    "Auto_CWE_Consistent": is_consistent
                })
                
    random.seed(42)
    random.shuffle(records)
    
    master_rows = []
    blind_rows = []
    
    for i, rec in enumerate(records):
        review_id = f"REV_{i:04d}"
        
        master_rows.append({
            "Review_ID": review_id,
            "original_idx": rec["original_idx"],
            "ablation_name": rec["ablation_name"]
        })
        
        blind_rows.append({
            "Review_ID": review_id,
            "Vulnerability_Type": rec["Vulnerability_Type"],
            "Source_Evidence": rec["Source_Evidence"],
            "Root_Cause": rec["Root_Cause"],
            "Repair": rec["Repair"],
            "Auto_CWE_Consistent": rec["Auto_CWE_Consistent"],
            "Score_RootCause_1_5": "",
            "Score_Evidence_1_5": "",
            "Score_Repair_1_5": "",
            "Rater_Notes": ""
        })
        
    # Save Master Key
    pd.DataFrame(master_rows).to_csv(f"{OUT_DIR}/master_key.csv", index=False)
    
    # Save blinded CSVs for Rater A and Rater B
    blind_df = pd.DataFrame(blind_rows)
    blind_df.to_csv(f"{OUT_DIR}/rater_A.csv", index=False)
    blind_df.to_csv(f"{OUT_DIR}/rater_B.csv", index=False)
    
    print(f"Generated {len(blind_rows)} blinded cases.")
    print(f"Saved master_key.csv (DO NOT share with raters!)")
    print(f"Saved rater_A.csv and rater_B.csv")

if __name__ == "__main__":
    prepare()
