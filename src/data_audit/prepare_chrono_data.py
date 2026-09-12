import pandas as pd
import numpy as np
import os
import re

def extract_year(cve_str):
    if pd.isna(cve_str):
        return np.nan
    match = re.search(r'CVE-(\d{4})', str(cve_str))
    if match:
        return int(match.group(1))
    return np.nan

def main():
    data_dir = r"d:\x-locvul\data\processed"
    
    print("1. Loading full datasets")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_dfs = [
        pd.read_csv(os.path.join(data_dir, f)) for f in ["test_bigvul.csv", "test_primevul.csv", "test_paired.csv"]
    ]
    
    full_df = pd.concat([train_df, val_df] + test_dfs, ignore_index=True)
    print(f"Total samples: {len(full_df)}")
    
    # Extract year
    full_df['year'] = full_df['cve'].apply(extract_year)
    
    print("2. Mapping missing years using commit_id")
    # Find all known years for each commit
    commit_years = full_df[full_df['year'].notna()].groupby('commit_id')['year'].first()
    
    # Fill missing years
    full_df['year'] = full_df.apply(
        lambda row: commit_years.get(row['commit_id'], np.nan) if pd.isna(row['year']) else row['year'],
        axis=1
    )
    
    print(f"Total samples with known year: {full_df['year'].notna().sum()} / {len(full_df)}")
    
    print("3. Splitting by year")
    # Logic E5: 
    # Year <= 2019 or NaN -> Train
    # Year == 2020 -> Val
    # Year >= 2021 -> Test
    
    train_mask = (full_df['year'] <= 2019) | (full_df['year'].isna())
    val_mask = (full_df['year'] == 2020)
    test_mask = (full_df['year'] >= 2021)
    
    train_chrono = full_df[train_mask].drop(columns=['year'])
    val_chrono = full_df[val_mask].drop(columns=['year'])
    test_chrono = full_df[test_mask].drop(columns=['year'])
    
    print(f"\nFinal Chronological Split:")
    print(f"Train: {len(train_chrono)} samples (Target 1: {len(train_chrono[train_chrono.target==1])})")
    print(f"Val:   {len(val_chrono)} samples (Target 1: {len(val_chrono[val_chrono.target==1])})")
    print(f"Test:  {len(test_chrono)} samples (Target 1: {len(test_chrono[test_chrono.target==1])})")
    
    print("\n4. Saving chronological datasets")
    train_chrono.to_csv(os.path.join(data_dir, "train_chrono.csv"), index=False)
    val_chrono.to_csv(os.path.join(data_dir, "val_chrono.csv"), index=False)
    test_chrono.to_csv(os.path.join(data_dir, "test_chrono.csv"), index=False)
    
    print("E5 Data Preparation Complete!")

if __name__ == "__main__":
    main()
