import pandas as pd
import numpy as np
import hashlib
import re
import os
import shutil
from huggingface_hub import hf_hub_download

DATA_PATH = "./data/processed"
OUTPUT_DIR = "./data/linevul"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Downloading LineVul dataset")
splits = ['train', 'validation', 'test']
for s in splits:
    target_path = f"{OUTPUT_DIR}/{s}.csv"
    if not os.path.exists(target_path):
        print(f"Downloading {s} split")
        try:
            cached_path = hf_hub_download(repo_id="starsofchance/LineVul", repo_type="dataset", filename=s, cache_dir="./data/hf_cache")
            shutil.copy(cached_path, target_path)
            print(f"Successfully downloaded {s}.")
        except Exception as e:
            print(f"Error downloading {s}: {e}")

print("2. Loading downloaded datasets")
cols = ['processed_func', 'target', 'flaw_line', 'flaw_line_index', 'CWE ID', 'CVE ID', 'project']
train_df = pd.read_csv(f"{OUTPUT_DIR}/train.csv", usecols=cols)
val_df = pd.read_csv(f"{OUTPUT_DIR}/validation.csv", usecols=cols)
test_df = pd.read_csv(f"{OUTPUT_DIR}/test.csv", usecols=cols)

def standardize_cols(df, split_name):
    df = df.copy()
    df = df.rename(columns={
        'processed_func': 'func_code',
        'CWE ID': 'cwe',
        'CVE ID': 'cve'
    })
    df['split'] = split_name
    return df

train_df = standardize_cols(train_df, 'train')
val_df = standardize_cols(val_df, 'val')
test_df = standardize_cols(test_df, 'test')

manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Unified Manifest Created: {len(manifest_df)} total samples.")

def remove_comments_and_whitespace(code):
    """Remove C/C++ comments and extra whitespaces"""
    code = str(code) if pd.notna(code) else ""
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'\s+', ' ', code).strip()
    return code

def compute_sha256(text):
    text = str(text) if pd.notna(text) else ""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

print("3. Normalizing code and computing hashes")
manifest_df['func_code'] = manifest_df['func_code'].astype(str)
manifest_df['normalized_code'] = manifest_df['func_code'].apply(remove_comments_and_whitespace)
manifest_df['exact_hash'] = manifest_df['func_code'].apply(compute_sha256)
manifest_df['normalized_hash'] = manifest_df['normalized_code'].apply(compute_sha256)

print("4. Checking for Cross-Split Leakage")
hash_split_counts = manifest_df.groupby('normalized_hash')['split'].nunique()
leaked_hashes = hash_split_counts[hash_split_counts > 1].index.tolist()
print(f"Found {len(leaked_hashes)} clusters of code appearing in multiple splits!")

print("5. Cleaning Leakage")
initial_train_len = len(manifest_df[manifest_df['split'] == 'train'])
initial_val_len = len(manifest_df[manifest_df['split'] == 'val'])

# Protect test set
test_hashes = set(manifest_df[manifest_df['split'] == 'test']['normalized_hash'])
mask_test_leak = (manifest_df['split'].isin(['train', 'val'])) & (manifest_df['normalized_hash'].isin(test_hashes))

# Protect validation set
val_hashes = set(manifest_df[manifest_df['split'] == 'val']['normalized_hash'])
mask_val_leak = (manifest_df['split'] == 'train') & (manifest_df['normalized_hash'].isin(val_hashes))

manifest_df = manifest_df[~(mask_test_leak | mask_val_leak)]
manifest_df = manifest_df.dropna(subset=['func_code', 'target'])
manifest_df['target'] = manifest_df['target'].astype(int)

final_train_len = len(manifest_df[manifest_df['split'] == 'train'])
final_val_len = len(manifest_df[manifest_df['split'] == 'val'])

print(f"Dropped {initial_train_len - final_train_len} leaked samples from Train.")
print(f"Dropped {initial_val_len - final_val_len} leaked samples from Val.")

print("6. Saving Cleaned Data")
for s in ['train', 'val', 'test']:
    split_df = manifest_df[manifest_df['split'] == s].copy()
    if 'normalized_code' in split_df.columns:
        split_df = split_df.drop(columns=['normalized_code', 'split'])
    
    file_path = f"{OUTPUT_DIR}/{s}_cleaned.csv"
    split_df.to_csv(file_path, index=False)
    print(f"Saved {s}_cleaned.csv -> {len(split_df)} samples.")

print("\nData download and audit complete!")
