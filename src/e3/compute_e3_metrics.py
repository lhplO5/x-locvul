import pandas as pd
import numpy as np
import ast
import os
import glob
from tqdm import tqdm

def normalize_code_line(line):
    return " ".join(str(line).split())

def is_intrinsic(pred_line, func_lines_normalized):
    pred_norm = normalize_code_line(pred_line)
    if not pred_norm:
        return False
    return pred_norm in func_lines_normalized

def get_rank(preds, gt_raw):
    gt_lines = [normalize_code_line(g) for g in str(gt_raw).split('/~/') if g.strip()]
    if not gt_lines:
        gt_lines = [normalize_code_line(g) for g in str(gt_raw).split('\n') if g.strip()]
        
    preds_norm = [normalize_code_line(p) for p in preds]
    if not preds_norm or not gt_lines:
        return -1
        
    for i, p in enumerate(preds_norm):
        if not p: continue
        for gt in gt_lines:
            if p == gt or p in gt or gt in p:
                return i + 1
    return -1

def is_rescue_successful(rescued_line, gt_raw):
    rescued_norm = normalize_code_line(rescued_line)
    gt_lines = [normalize_code_line(g) for g in str(gt_raw).split('/~/') if g.strip()]
    if not gt_lines:
        gt_lines = [normalize_code_line(g) for g in str(gt_raw).split('\n') if g.strip()]
        
    for gt in gt_lines:
        if rescued_norm == gt or rescued_norm in gt or gt in rescued_norm:
            return True
    return False

def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

def compute_dataset_metrics(df_indices, results_dict):
    metrics = {}
    N = len(df_indices)
    
    # Accumulators for rates
    sum_raw_preds = 0
    sum_hallucinated = 0
    sum_fuzzy_rescued = 0
    sum_semantic_rescued = 0
    
    # Store metrics for each strategy
    strat_metrics = {k: {"top1": 0, "top5": 0, "top10": 0, "mrr_sum": 0} for k in ['e31', 'e32', 'e33', 'e34']}
    
    for idx in df_indices:
        res = results_dict[idx]
        
        sum_raw_preds += res['num_raw']
        sum_hallucinated += res['num_halluc']
        sum_fuzzy_rescued += res['num_fuzzy_resc']
        sum_semantic_rescued += res['num_sem_resc']
        
        for k in ['e31', 'e32', 'e33', 'e34']:
            rank = res[f'{k}_rank']
            if rank != -1:
                if rank <= 1: strat_metrics[k]['top1'] += 1
                if rank <= 5: strat_metrics[k]['top5'] += 1
                if rank <= 10: strat_metrics[k]['top10'] += 1
                strat_metrics[k]['mrr_sum'] += 1.0 / rank

    # Compile results
    glhr = (sum_hallucinated / sum_raw_preds) if sum_raw_preds > 0 else 0
    prr_fuzzy = (sum_fuzzy_rescued / sum_hallucinated) if sum_hallucinated > 0 else 0
    prr_sem = (sum_semantic_rescued / sum_hallucinated) if sum_hallucinated > 0 else 0
    
    out = {
        'GLHR': glhr,
        'PRR_Fuzzy': prr_fuzzy,
        'PRR_Semantic': prr_sem
    }
    
    for k in ['e31', 'e32', 'e33', 'e34']:
        out[f'{k}_Top1'] = strat_metrics[k]['top1'] / N
        out[f'{k}_Top5'] = strat_metrics[k]['top5'] / N
        out[f'{k}_Top10'] = strat_metrics[k]['top10'] / N
        out[f'{k}_MRR'] = strat_metrics[k]['mrr_sum'] / N
        
    return out

def main():
    results_dir = "./results/e3"
    outputs_dir = "./outputs/e3"
    os.makedirs(outputs_dir, exist_ok=True)
    seed_files = glob.glob(os.path.join(results_dir, "e3_ablation_seed_*.csv"))
    if not seed_files:
        print("No seed files found")
        return
        
    all_results = [] # list of lists, [seed_idx][sample_idx]
    
    print("Parsing files")
    for f in seed_files:
        df = pd.read_csv(f)
        seed_res = []
        for _, row in df.iterrows():
            func = str(row['func_code'])
            gt = str(row['flaw_line'])
            func_lines_norm = set([normalize_code_line(l) for l in func.split('\n') if l.strip()])
            
            r_e31 = safe_eval(row['e31_raw_preds'])
            r_e32 = safe_eval(row['e32_exact_preds'])
            r_e33 = safe_eval(row['e33_fuzzy_preds'])
            r_e34 = safe_eval(row.get('e34_semantic_preds', row.get('e34_sem_preds', '[]'))) # fallback
            
            num_raw = len(r_e31)
            num_halluc = 0
            for p in r_e31:
                if not is_intrinsic(p, func_lines_norm):
                    num_halluc += 1
            
            # Count rescues
            num_fuzzy_resc = 0
            for p in r_e33:
                if p not in r_e32 and is_rescue_successful(p, gt):
                    num_fuzzy_resc += 1
                    
            num_sem_resc = 0
            for p in r_e34:
                if p not in r_e32 and is_rescue_successful(p, gt):
                    num_sem_resc += 1
            
            res = {
                'num_raw': num_raw,
                'num_halluc': num_halluc,
                'num_fuzzy_resc': num_fuzzy_resc,
                'num_sem_resc': num_sem_resc,
                'e31_rank': get_rank(r_e31, gt),
                'e32_rank': get_rank(r_e32, gt),
                'e33_rank': get_rank(r_e33, gt),
                'e34_rank': get_rank(r_e34, gt),
            }
            seed_res.append(res)
        all_results.append(seed_res)
        
    num_samples = len(all_results[0])
    num_seeds = len(all_results)
    print(f"Loaded {num_seeds} seeds, {num_samples} samples each")
    
    ensemble_results = []
    for i in range(num_samples):
        avg_res = {
            'num_raw': np.mean([all_results[s][i]['num_raw'] for s in range(num_seeds)]),
            'num_halluc': np.mean([all_results[s][i]['num_halluc'] for s in range(num_seeds)]),
            'num_fuzzy_resc': np.mean([all_results[s][i]['num_fuzzy_resc'] for s in range(num_seeds)]),
            'num_sem_resc': np.mean([all_results[s][i]['num_sem_resc'] for s in range(num_seeds)]),
        }
        
        for k in ['e31', 'e32', 'e33', 'e34']:
            pass
    
    B = 1000
    np.random.seed(42)
    boot_metrics = []
    
    print("Running Hierarchical Bootstrap")
    for _ in tqdm(range(B)):
        indices = np.random.choice(num_samples, num_samples, replace=True)
        # Compute metric for each seed
        seed_vals = []
        for s in range(num_seeds):
            seed_vals.append(compute_dataset_metrics(indices, all_results[s]))
            
        # Average across seeds for this bootstrap iteration
        avg_vals = {}
        for k in seed_vals[0].keys():
            avg_vals[k] = np.mean([sv[k] for sv in seed_vals])
        boot_metrics.append(avg_vals)
        
    df_boot = pd.DataFrame(boot_metrics)
    
    # Calculate IFA (Median and IQR)
    ifa_dict = {}
    for k in ['e31', 'e32', 'e33', 'e34']:
        all_hits = []
        for s in range(num_seeds):
            for i in range(num_samples):
                r = all_results[s][i][f'{k}_rank']
                if r != -1:
                    all_hits.append(r - 1)
        if len(all_hits) > 0:
            ifa_dict[k] = {
                'Median': np.median(all_hits),
                'Q1': np.percentile(all_hits, 25),
                'Q3': np.percentile(all_hits, 75)
            }
        else:
            ifa_dict[k] = {'Median': '-', 'Q1': '-', 'Q3': '-'}
            
    # Print Results
    print("\nFINAL E3 METRICS")
    summary_rows = []
    
    strategies = [
        ('e31', 'E3.1_Raw_Generation'),
        ('e32', 'E3.2_Exact_Match'),
        ('e33', 'E3.3_Fuzzy_Match'),
        ('e34', 'E3.4_Semantic_Proj')
    ]
    
    for k, name in strategies:
        row = {'Strategy': name}
        if k == 'e31':
            row['GLHR(%)'] = "-"
            row['PRR(%)'] = "-"
        elif k == 'e32':
            mean_glhr = df_boot['GLHR'].mean()
            ci_glhr = np.percentile(df_boot['GLHR'], [2.5, 97.5])
            row['GLHR(%)'] = f"{mean_glhr:.4f} [{ci_glhr[0]:.4f}, {ci_glhr[1]:.4f}]"
            row['PRR(%)'] = "-"
        elif k == 'e33':
            row['GLHR(%)'] = "-"
            mean_prr = df_boot['PRR_Fuzzy'].mean()
            ci_prr = np.percentile(df_boot['PRR_Fuzzy'], [2.5, 97.5])
            row['PRR(%)'] = f"{mean_prr:.4f} [{ci_prr[0]:.4f}, {ci_prr[1]:.4f}]"
        elif k == 'e34':
            row['GLHR(%)'] = "-"
            mean_prr = df_boot['PRR_Semantic'].mean()
            ci_prr = np.percentile(df_boot['PRR_Semantic'], [2.5, 97.5])
            row['PRR(%)'] = f"{mean_prr:.4f} [{ci_prr[0]:.4f}, {ci_prr[1]:.4f}]"
            
        for metric in ['Top1', 'Top5', 'Top10']:
            mean_v = df_boot[f'{k}_{metric}'].mean()
            ci_v = np.percentile(df_boot[f'{k}_{metric}'], [2.5, 97.5])
            row[metric] = f"{mean_v:.4f} [{ci_v[0]:.4f}, {ci_v[1]:.4f}]"
            
        mean_mrr = df_boot[f'{k}_MRR'].mean()
        ci_mrr = np.percentile(df_boot[f'{k}_MRR'], [2.5, 97.5])
        row['MRR'] = f"{mean_mrr:.4f} [{ci_mrr[0]:.4f}, {ci_mrr[1]:.4f}]"
        
        med = ifa_dict[k]['Median']
        q1 = ifa_dict[k]['Q1']
        q3 = ifa_dict[k]['Q3']
        row['IFA (Median, IQR)'] = f"{med} (IQR: {q1}-{q3})" if med != '-' else "-"
        
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(outputs_dir, 'table_e3_final.csv'), index=False)
    print(df_summary.to_string())
    
    # Paired comparisons
    print("\nPAIRED COMPARISONS")
    
    diff_sem_fuz_top1 = df_boot['e34_Top1'] - df_boot['e33_Top1']
    diff_sem_fuz_mrr = df_boot['e34_MRR'] - df_boot['e33_MRR']
    
    diff_fuz_raw_top1 = df_boot['e33_Top1'] - df_boot['e31_Top1']
    diff_fuz_raw_mrr = df_boot['e33_MRR'] - df_boot['e31_MRR']
    
    paired_rows = []
    
    def add_paired(name, diff_arr):
        mean = diff_arr.mean()
        ci = np.percentile(diff_arr, [2.5, 97.5])
        paired_rows.append({
            'Comparison': name,
            'Diff Mean': round(mean, 4),
            '95% CI': f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        })
        print(f"{name}: {mean:.4f} 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
    add_paired("Semantic vs Fuzzy (Top-1)", diff_sem_fuz_top1)
    add_paired("Semantic vs Fuzzy (MRR)", diff_sem_fuz_mrr)
    add_paired("Fuzzy vs Raw (Top-1)", diff_fuz_raw_top1)
    add_paired("Fuzzy vs Raw (MRR)", diff_fuz_raw_mrr)
    
    pd.DataFrame(paired_rows).to_csv(os.path.join(outputs_dir, 'e3_paired_comparison.csv'), index=False)

if __name__ == "__main__":
    main()
