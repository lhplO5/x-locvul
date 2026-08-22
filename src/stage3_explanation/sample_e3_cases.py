import pandas as pd
import json
import os
import random

FROZEN_TEST = "./data/manifests/test_primevul_frozen.jsonl"
OUT_DIR = "./data/processed"
OUT_FILE = f"{OUT_DIR}/e3_sample_180.jsonl"

# Broad categorization for C/C++ CWEs based on typical top-25 lists
MEMORY_SAFETY_CWES = [
    "CWE-119", "CWE-125", "CWE-787", "CWE-416", "CWE-476", 
    "CWE-120", "CWE-190", "CWE-134", "CWE-415"
]
INPUT_VALIDATION_CWES = [
    "CWE-20", "CWE-79", "CWE-89", "CWE-78", "CWE-77", 
    "CWE-400", "CWE-399", "CWE-22", "CWE-94", "CWE-352"
]

def categorize_cwe(cwe):
    if not isinstance(cwe, str):
        return "Other"
    
    # Extract base CWE if it's a compound string
    cwe_id = cwe.split()[0] if cwe else ""
    
    if cwe_id in MEMORY_SAFETY_CWES:
        return "Memory Safety"
    elif cwe_id in INPUT_VALIDATION_CWES:
        return "Input Validation"
    else:
        return "Other"

def sample_cases():
    print("=============================================")
    print(" E3: Sampling 180 Cases for Explanation Task ")
    print("=============================================")
    
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load test set
    df = pd.read_json(FROZEN_TEST, lines=True)
    
    # Filter only vulnerable functions
    if 'label' in df.columns:
        df = df[df['label'] == 1].copy()
    else:
        print("Warning: 'label' column not found, assuming all are vulnerable.")
        
    print(f"Total vulnerable samples in test set: {len(df)}")
    
    # Categorize
    if 'cwe' not in df.columns:
        df['cwe'] = "CWE-Other"
    df['cwe_category'] = df['cwe'].apply(categorize_cwe)
    
    # Separate into buckets
    mem_safety_df = df[df['cwe_category'] == 'Memory Safety']
    input_val_df = df[df['cwe_category'] == 'Input Validation']
    other_df = df[df['cwe_category'] == 'Other']
    
    print(f"  - Memory Safety available: {len(mem_safety_df)}")
    print(f"  - Input Validation available: {len(input_val_df)}")
    print(f"  - Others available: {len(other_df)}")
    
    # Sample 60 from each (50 main + 10 pilot)
    # If not enough, sample with replacement or take all
    def safe_sample(sub_df, n):
        if len(sub_df) >= n:
            return sub_df.sample(n, random_state=42)
        else:
            print(f"Warning: Not enough samples ({len(sub_df)} < {n}). Taking all available.")
            return sub_df
            
    sampled_mem = safe_sample(mem_safety_df, 60)
    sampled_input = safe_sample(input_val_df, 60)
    sampled_other = safe_sample(other_df, 60)
    
    # Combine
    final_sample = pd.concat([sampled_mem, sampled_input, sampled_other])
    
    # If total is less than 180, backfill from other categories
    remaining_needed = 180 - len(final_sample)
    if remaining_needed > 0:
        used_indices = final_sample.index
        unused_df = df.drop(used_indices)
        if len(unused_df) >= remaining_needed:
            backfill = unused_df.sample(remaining_needed, random_state=42)
            final_sample = pd.concat([final_sample, backfill])
            print(f"Backfilled {remaining_needed} cases from unused pool.")
            
    print(f"\nFinal sampled cases: {len(final_sample)}")
    print(final_sample['cwe_category'].value_counts())
    
    # Save
    final_sample.to_json(OUT_FILE, orient='records', lines=True)
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    sample_cases()
