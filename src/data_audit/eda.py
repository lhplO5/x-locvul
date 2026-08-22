# ==========================================
# CELL 1: LOAD FROZEN DATA (BẢN GỠ LỖI)
# ==========================================
import pandas as pd
import numpy as np

FROZEN_DIR = "./data/manifests"

print("1. Loading frozen datasets...")

train_df = pd.read_json(f"{FROZEN_DIR}/train_frozen.jsonl", lines=True)
val_df = pd.read_json(f"{FROZEN_DIR}/val_frozen.jsonl", lines=True)
test_pv_df = pd.read_json(f"{FROZEN_DIR}/test_primevul_frozen.jsonl", lines=True)
test_pair_df = pd.read_json(f"{FROZEN_DIR}/test_paired_frozen.jsonl", lines=True)
test_bv_df = pd.read_json(f"{FROZEN_DIR}/test_bigvul_frozen.jsonl", lines=True)

print("--- DATASET SHAPES ---")
print(f"Train Set:         {len(train_df)} samples")
print(f"Val Set:           {len(val_df)} samples")
print(f"Test PrimeVul Set: {len(test_pv_df)} samples")
print(f"Test Paired Set:   {len(test_pair_df)} samples")
print(f"Test BigVul Set:   {len(test_bv_df)} samples")

assert 'func_code' in train_df.columns, "Missing 'func_code' column!"
assert 'label' in train_df.columns, "Missing 'label' column!"
print("\n[OK] Data loaded successfully.")

# ==========================================
# CELL 2: BINARY LABEL DISTRIBUTION CHECK
# ==========================================
print("2. Analyzing Binary Label Distribution (Vulnerable vs Clean)...")

def print_distribution(df, split_name):
    counts = df['label'].value_counts()
    total = len(df)
    vul_count = counts.get(1, 0)
    clean_count = counts.get(0, 0)
    
    vul_pct = (vul_count / total) * 100 if total > 0 else 0
    clean_pct = (clean_count / total) * 100 if total > 0 else 0
    
    print(f"\n[{split_name.upper()} SET]")
    print(f"Vulnerable (1): {vul_count} ({vul_pct:.2f}%)")
    print(f"Clean (0):      {clean_count} ({clean_pct:.2f}%)")
    
    # Calculate recommended pos_weight for BCE Loss (Negative / Positive)
    if vul_count > 0:
        recommended_weight = clean_count / vul_count
        print(f"-> Recommended BCE pos_weight: ~{recommended_weight:.2f}")

print_distribution(train_df, "train")
print_distribution(val_df, "val")
print_distribution(test_pv_df, "test_primevul")

# ==========================================
# CELL 3: CWE MULTI-CLASS DISTRIBUTION
# ==========================================
print("3. Analyzing CWE Distribution (Only for Vulnerable samples)...")

# Filter only vulnerable samples
vul_train = train_df[train_df['label'] == 1]

if 'cwe_label' in vul_train.columns:
    cwe_counts = vul_train['cwe_label'].value_counts()
    
    print(f"Total unique CWE classes in Train: {len(cwe_counts)}")
    print("\nTop 5 Most Frequent CWEs:")
    print(cwe_counts.head(5))
    
    print("\nBottom 5 Least Frequent CWEs:")
    print(cwe_counts.tail(5))
    
    # Check for extremely rare classes (< 10 samples)
    rare_classes = cwe_counts[cwe_counts < 10]
    if len(rare_classes) > 0:
        print(f"\nWARNING: Found {len(rare_classes)} CWE classes with less than 10 samples.")
else:
    print("Column 'cwe_label' not found. Skipping CWE analysis.")

# ==========================================
# CELL 4: CODE LENGTH & TRUNCATION RISK
# ==========================================
print("4. Analyzing Code Lengths (Approximated by word count)...")

# Fast word count approximation (casting to string to avoid errors)
train_df['word_count'] = train_df['func_code'].astype(str).apply(lambda x: len(x.split()))

mean_len = train_df['word_count'].mean()
median_len = train_df['word_count'].median()
max_len = train_df['word_count'].max()

print(f"Mean word count:   {mean_len:.1f}")
print(f"Median word count: {median_len:.1f}")
print(f"Max word count:    {max_len}")

# UniXCoder max_length is typically 512 tokens (roughly 400 words)
SAFE_THRESHOLD = 400
truncated_samples = len(train_df[train_df['word_count'] > SAFE_THRESHOLD])
truncation_pct = (truncated_samples / len(train_df)) * 100

print(f"\nSamples exceeding {SAFE_THRESHOLD} words (High Truncation Risk):")
print(f"{truncated_samples} samples ({truncation_pct:.2f}% of Train Set)")

# Quality Check: Extremely short codes (Junk data)
junk_samples = len(train_df[train_df['word_count'] < 5])
print(f"Extremely short samples (< 5 words, potential junk): {junk_samples}")

# ==========================================
# CELL 5: VISUAL SANITY CHECK
# ==========================================
print("5. Visual Inspection of Samples...\n")

# Print 1 random Clean sample safely
clean_samples = train_df[train_df['label'] == 0]
if len(clean_samples) > 0:
    clean_sample = clean_samples.sample(1).iloc[0]
    print("--- RANDOM CLEAN CODE (Label 0) ---")
    print(f"Code snippet (first 300 chars):\n{str(clean_sample['func_code'])[:300]}...\n")

# Print 1 random Vulnerable sample safely
vul_samples = train_df[train_df['label'] == 1]
if len(vul_samples) > 0:
    vul_sample = vul_samples.sample(1).iloc[0]
    print("--- RANDOM VULNERABLE CODE (Label 1) ---")
    print(f"CWE Type: {vul_sample.get('cwe_label', 'Unknown')}")
    print(f"Code snippet (first 300 chars):\n{str(vul_sample['func_code'])[:300]}...\n")

