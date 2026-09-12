import pandas as pd

boot_df = pd.read_csv('results/e1/e1_all_bootstrap.csv')
boot_dict = {}
for _, row in boot_df.iterrows():
    boot_dict[row['Run']] = {
        'PR-AUC (Mean)': row['PR-AUC (Mean)'],
        'PR-AUC (95% CI)': row['PR-AUC (95% CI)'],
        'MCC (Mean)': row['MCC (Mean)'],
        'F1-score (Mean)': row['F1-score (Mean)']
    }

t1 = pd.read_csv('outputs/e1/primevul_seed_metrics.csv')

for idx, row in t1.iterrows():
    run_id = row['Run']
    if run_id in boot_dict:
        # Update point estimates
        t1.at[idx, 'PR-AUC'] = f"{boot_dict[run_id]['PR-AUC (Mean)']:.4f}"
        t1.at[idx, 'MCC'] = f"{boot_dict[run_id]['MCC (Mean)']:.4f}"
        t1.at[idx, 'F1-score'] = f"{boot_dict[run_id]['F1-score (Mean)']:.4f}"
        
        # Update CI for PR-AUC
        t1.at[idx, 'PR-AUC (95% CI)'] = boot_dict[run_id]['PR-AUC (95% CI)']

t1.to_csv('outputs/e1/primevul_seed_metrics.csv', index=False)
print("Table 1 completely updated with all bootstrap point estimates.")
