import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef, f1_score

def evaluate_metrics(y_true, y_probs, threshold=0.5):
    preds = (y_probs > threshold).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    mcc = matthews_corrcoef(y_true, preds)
    f1 = f1_score(y_true, preds)
    return pr_auc, mcc, f1

def bootstrap_single_model(y_true, probs, n_iterations=1000, seed=42):
    rng = np.random.RandomState(seed)
    N = len(y_true)
    prauc_list, mcc_list, f1_list = [], [], []
    
    for i in range(n_iterations):
        indices = rng.choice(N, size=N, replace=True)
        y_true_boot = y_true[indices]
        probs_boot = np.mean(probs[:, indices], axis=0)
        
        pr_auc, mcc, f1 = evaluate_metrics(y_true_boot, probs_boot)
        prauc_list.append(pr_auc)
        mcc_list.append(mcc)
        f1_list.append(f1)
        
    def get_stats(data_list):
        mean_val = np.mean(data_list)
        ci_lower = np.percentile(data_list, 2.5)
        ci_upper = np.percentile(data_list, 97.5)
        return mean_val, ci_lower, ci_upper

    return {
        "PR-AUC": get_stats(prauc_list),
        "MCC": get_stats(mcc_list),
        "F1": get_stats(f1_list)
    }

print("Loading ground truth")
test_pv_df = pd.read_json("data/manifests/test_primevul_frozen.jsonl", lines=True)
test_pv_df = test_pv_df.dropna(subset=['func_code', 'label'])
y_true = test_pv_df['label'].astype(int).values

models = ["E1.1", "E1.2", "E1.3", "E1.4", "E1.5", "E1.6"]
results = {}

for m in models:
    print(f"Processing {m}")
    try:
        data = np.load(f"results/e1/cache_{m}.npz", allow_pickle=True)
        probs = data['pv']
        stats = bootstrap_single_model(y_true, probs, n_iterations=1000)
        results[m] = stats
    except Exception as e:
        print(f"Failed for {m}: {e}")

csv_str = "Run,PR-AUC (Mean),PR-AUC (95% CI),MCC (Mean),MCC (95% CI),F1-score (Mean),F1-score (95% CI)\\n"
for m, stats in results.items():
    p_mean, p_l, p_u = stats["PR-AUC"]
    m_mean, m_l, m_u = stats["MCC"]
    f_mean, f_l, f_u = stats["F1"]
    
    p_ci = f"[{p_l:.4f}, {p_u:.4f}]"
    m_ci = f"[{m_l:.4f}, {m_u:.4f}]"
    f_ci = f"[{f_l:.4f}, {f_u:.4f}]"
    
    csv_str += f"{m},{p_mean:.4f},\"{p_ci}\",{m_mean:.4f},\"{m_ci}\",{f_mean:.4f},\"{f_ci}\"\n"

with open("results/e1/e1_all_bootstrap.csv", "w", encoding="utf-8") as f:
    f.write(csv_str)
print("Saved to results/e1/e1_all_bootstrap.csv")
