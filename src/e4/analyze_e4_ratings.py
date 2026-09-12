import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import wilcoxon

EXCEL_A_LLM = "./outputs/e4/e4_rater_A.csv"
EXCEL_B_LLM = "./outputs/e4/e4_rater_B.csv"
MASTER_KEY = "./results/e4/master_key.csv"

def compute_irr(dfA, dfB):
    print("\nInter-Rater Reliability")
    metrics = {
        'Score_RootCause_1_5': 'quadratic',
        'Score_Evidence_1_5': 'quadratic',
        'Score_CWE_1_5': 'quadratic',
        'Score_Repair_1_5': 'quadratic',
        'Score_UnsupportedClaim_0_1': None 
    }
    
    irr_results = []
    
    for metric, weight in metrics.items():
        if metric in dfA.columns and metric in dfB.columns:
            valid_idx = dfA[metric].notna() & dfB[metric].notna()
            a_vals = dfA.loc[valid_idx, metric].astype(int)
            b_vals = dfB.loc[valid_idx, metric].astype(int)
            
            if len(a_vals) > 0:
                kappa = cohen_kappa_score(a_vals, b_vals, weights=weight)
                weight_str = f"({weight} weights)" if weight else "(unweighted)"
                print(f"{metric} {weight_str}: {kappa:.3f}")
                irr_results.append({
                    'Metric': metric,
                    'Weight_Type': weight_str,
                    'Cohen_Kappa': kappa
                })
                
    pd.DataFrame(irr_results).to_csv("./outputs/e4/E4_IRR_Metrics.csv", index=False)

def compute_means(merged_df):
    print("\nMean Scores per Condition")
    metrics = ['Score_RootCause_1_5', 'Score_Evidence_1_5', 'Score_CWE_1_5', 'Score_Repair_1_5', 'Score_UnsupportedClaim_0_1']
    
    # Calculate average between A and B for each row
    for metric in metrics:
        merged_df[f"{metric}_avg"] = (merged_df[f"{metric}_A"] + merged_df[f"{metric}_B"]) / 2
        
    avg_cols = [f"{m}_avg" for m in metrics]
    means_by_cond = merged_df.groupby('Condition')[avg_cols].mean()
    print(means_by_cond.round(3))
    means_by_cond.round(3).to_csv("./outputs/e4/E4_Condition_Means.csv")
    merged_df.to_csv("./outputs/e4/E4_Merged_Average_Scores.csv", index=False)
    return merged_df

def paired_test(df, cond1, cond2, metrics, results_list):
    print(f"\nStatistical Test: {cond1} vs {cond2}")
    df_cond1 = df[df['Condition'] == cond1].sort_values('original_idx')
    df_cond2 = df[df['Condition'] == cond2].sort_values('original_idx')
    
    common_idx = set(df_cond1['original_idx']).intersection(set(df_cond2['original_idx']))
    
    c1 = df_cond1[df_cond1['original_idx'].isin(common_idx)].set_index('original_idx')
    c2 = df_cond2[df_cond2['original_idx'].isin(common_idx)].set_index('original_idx')
    
    for metric in metrics:
        col = f"{metric}_avg"
        diff = c1[col] - c2[col]
        # if all diffs are 0, wilcoxon raises an error
        if np.all(diff == 0):
            print(f"{metric}: p-value = 1.0 (Identical scores)")
            results_list.append({
                'Comparison': f"{cond1} vs {cond2}",
                'Metric': metric,
                'Mean_Diff': 0.0,
                'P_Value': 1.0
            })
            continue
            
        res = wilcoxon(c1[col], c2[col], zero_method='zsplit')
        mean_diff = diff.mean()
        print(f"{metric}: diff = {mean_diff:+.3f}, p-value = {res.pvalue:.4f}")
        results_list.append({
            'Comparison': f"{cond1} vs {cond2}",
            'Metric': metric,
            'Mean_Diff': mean_diff,
            'P_Value': res.pvalue
        })


def main():
    try:
        dfA = pd.read_csv(EXCEL_A_LLM)
        dfB = pd.read_csv(EXCEL_B_LLM)
        master = pd.read_csv(MASTER_KEY)
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return
        
    dfA = dfA.sort_values('blind order')
    dfB = dfB.sort_values('blind order')
    
    # 1. Compute IRR
    compute_irr(dfA, dfB)
    
    # 2. Merge back with master to get original Case IDs and Conditions
    dfA = dfA.merge(master, left_on='blind order', right_on='Review_ID', suffixes=('', '_m'))
    dfB = dfB.merge(master, left_on='blind order', right_on='Review_ID', suffixes=('', '_m'))
    
    # Merge A and B together for average scores
    metrics_cols = ['Score_RootCause_1_5', 'Score_Evidence_1_5', 'Score_CWE_1_5', 'Score_Repair_1_5', 'Score_UnsupportedClaim_0_1']
    merged_df = dfA[['original_idx', 'Condition'] + metrics_cols].copy()
    merged_df.columns = ['original_idx', 'Condition'] + [f"{c}_A" for c in metrics_cols]
    
    for c in metrics_cols:
        merged_df[f"{c}_B"] = dfB[c].values
        
    # 3. Compute Means
    merged_df = compute_means(merged_df)
    
    # 4. Paired differences and specific comparisons
    print("\n--- Specific Comparisons ---")
    metrics_to_test = ['Score_RootCause_1_5', 'Score_Evidence_1_5', 'Score_CWE_1_5']
    
    wilcoxon_results = []
    
    print("\n1. line+CWE (A3) vs line-only (A2)")
    paired_test(merged_df, 'A3_FunctionLineCWE', 'A2_FunctionLine', metrics_to_test, wilcoxon_results)
    
    print("\n2. Predicted evidence (A2) vs Oracle evidence (A4)")
    paired_test(merged_df, 'A2_FunctionLine', 'A4_Oracle', metrics_to_test, wilcoxon_results)
    
    print("\n3. Predicted evidence (A3) vs Oracle evidence (A4)")
    paired_test(merged_df, 'A3_FunctionLineCWE', 'A4_Oracle', metrics_to_test, wilcoxon_results)

    pd.DataFrame(wilcoxon_results).to_csv("./outputs/e4/E4_Wilcoxon_Tests.csv", index=False)


if __name__ == "__main__":
    main()
