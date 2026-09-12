import json
import random
import re
import pandas as pd
import os

IN_FILE = "./results/e4/full_pipeline_results.jsonl"
OUT_DIR = "./results/e4"
RATER_A_OUT = f"{OUT_DIR}/E4_RaterA.csv"
RATER_B_OUT = f"{OUT_DIR}/E4_RaterB.csv"
MASTER_KEY_OUT = f"{OUT_DIR}/master_key.csv"

def parse_explanation(text):
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

def normalize_code(code_str):
    if not code_str:
        return ""
    # Remove all whitespace including spaces, tabs, newlines
    return re.sub(r'\s+', '', code_str)

def get_line_status(condition, pred_line, gt_line):
    if condition == "A4_Oracle":
        return "Oracle"
    if condition == "A1_FunctionOnly":
        return "N/A"
    
    norm_pred = normalize_code(pred_line)
    norm_gt = normalize_code(gt_line)
    
    if not norm_pred or not norm_gt:
        return "No Match"
        
    if norm_pred == norm_gt:
        return "Exact Match"
    elif norm_pred in norm_gt or norm_gt in norm_pred:
        return "Partial Match"
    else:
        return "No Match"

def load_flat_results():
    """Load flat JSONL from run_full_pipeline.py and group by idx."""
    all_rows = []
    with open(IN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            all_rows.append(json.loads(line))
    
    print(f"Total rows read from pipeline output: {len(all_rows)}")
    
    # Group by idx → reconstruct case dict with explanations nested
    cases_by_idx = {}
    for row in all_rows:
        idx = row.get('idx')
        if idx not in cases_by_idx:
            cases_by_idx[idx] = {
                "idx": idx,
                "func_hash": row.get('func_hash', ''),
                "func_code": row.get('func_code', ''),
                "gt_cwe": row.get('gt_cwe', ''),
                "gt_line": row.get('gt_line', ''),
                "pred_vul": row.get('pred_vul', 1),
                "pred_prob": row.get('pred_prob', 0.0),
                "pred_cwe": row.get('pred_cwe', ''),
                "raw_pred_line": row.get('raw_pred_line', ''),
                "pred_line": row.get('pred_line', ''),
                "explanations": {}
            }
        condition = row.get('condition', '')
        explanation = row.get('explanation', '')
        cases_by_idx[idx]["explanations"][condition] = explanation
    
    records = list(cases_by_idx.values())
    print(f"Grouped into {len(records)} unique cases.")
    return records

def sample_cases():
    records = load_flat_results()
    
    cwe_groups = {}
    for r in records:
        cwe_family = r['gt_cwe'].split()[0] if r['gt_cwe'] else "Unknown"
        cwe_groups.setdefault(cwe_family, []).append(r)
        
    cwe_counts = {k: len(v) for k, v in cwe_groups.items()}
    top_cwes = sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_cwe_names = [x[0] for x in top_cwes]
    print(f"Top 10 CWEs for sampling: {top_cwe_names}")
    
    random.seed(42)
    sampled_cases = []
    for cwe in top_cwe_names:
        pool = cwe_groups[cwe]
        if len(pool) >= 5:
            sampled_cases.extend(random.sample(pool, 5))
        else:
            sampled_cases.extend(pool)
            
    print(f"Sampled exactly {len(sampled_cases)} cases.")
    return sampled_cases

def prepare():
    sampled_cases = sample_cases()
    
    master_rows = []
    blind_rows = []
    
    for case in sampled_cases:
        idx = case.get('idx')
        gt_cwe = case.get('gt_cwe', '')
        pred_cwe = case.get('pred_cwe', '')
        gt_line = case.get('gt_line', '')
        pred_line = case.get('pred_line', '')
        
        for condition, explanation in case.get('explanations', {}).items():
            fields = parse_explanation(explanation)
            
            line_status = get_line_status(condition, pred_line, gt_line)
            
            master_rows.append({
                "original_idx": idx,
                "Condition": condition,
                "gt_cwe": gt_cwe,
                "pred_cwe": pred_cwe,
                "gt_line": gt_line,
                "pred_line": pred_line,
                "line_status": line_status
            })
            
            blind_rows.append({
                "Base_Case_ID": idx,
                "Condition": condition,
                "ground-truth CWE": gt_cwe,
                "predicted CWE": pred_cwe,
                "ground-truth line": gt_line,
                "predicted line": pred_line,
                "predicted/oracle line status": line_status,
                "Vulnerability Type": fields['Vulnerability Type'],
                "Source Evidence": fields['Source Evidence'],
                "Root Cause": fields['Root Cause'],
                "Repair": fields['Repair'],
                "Score_RootCause_1_5": "",
                "Score_Evidence_1_5": "",
                "Score_CWE_1_5": "",
                "Score_Repair_1_5": "",
                "Score_UnsupportedClaim_0_1": ""
            })
            
    combined = list(zip(master_rows, blind_rows))
    random.shuffle(combined)
    
    final_master = []
    final_blind = []
    for i, (m, b) in enumerate(combined):
        review_id = f"REV_{i+1:03d}"
        
        m_copy = dict(m)
        m_copy['Review_ID'] = review_id
        final_master.append(m_copy)
        
        b_copy = dict(b)
        ordered_b = {"blind order": review_id}
        ordered_b.update(b_copy)
        final_blind.append(ordered_b)
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    pd.DataFrame(final_master).to_csv(MASTER_KEY_OUT, index=False)
    
    blind_df = pd.DataFrame(final_blind)
    
    def sanitize_xml(val):
        if isinstance(val, str):
            return re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', val)
        return val
        
    blind_df = blind_df.map(sanitize_xml)
    
    blind_df.to_csv(RATER_A_OUT, index=False, escapechar='\\')
    blind_df.to_csv(RATER_B_OUT, index=False, escapechar='\\')
    
    print(f"Generated {len(blind_df)} blinded cases.")
    print(f"Saved {MASTER_KEY_OUT}")
    print(f"Saved {RATER_A_OUT} and {RATER_B_OUT}")

if __name__ == "__main__":
    prepare()
