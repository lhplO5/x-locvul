import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef, f1_score

def evaluate_metrics(y_true, y_probs, threshold=0.5):
    preds = (y_probs > threshold).astype(int)
    
    # Calculate PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Calculate MCC and F1
    mcc = matthews_corrcoef(y_true, preds)
    f1 = f1_score(y_true, preds)
    
    return pr_auc, mcc, f1

def hierarchical_bootstrap(y_true, probs1, probs2, n_iterations=1000, seed=42):
    """
    y_true: shape (N,)
    probs1: shape (5, N) for Model 1 (e.g., E1.3)
    probs2: shape (5, N) for Model 2 (e.g., E1.5)
    """
    rng = np.random.RandomState(seed)
    N = len(y_true)
    
    diff_prauc = []
    diff_mcc = []
    diff_f1 = []
    
    m1_prauc_list, m1_mcc_list, m1_f1_list = [], [], []
    m2_prauc_list, m2_mcc_list, m2_f1_list = [], [], []
    
    for i in range(n_iterations):
        indices = rng.choice(N, size=N, replace=True)
        y_true_boot = y_true[indices]
        
        probs1_boot = np.mean(probs1[:, indices], axis=0)
        probs2_boot = np.mean(probs2[:, indices], axis=0)
        
        # Evaluate Model 1
        m1_prauc, m1_mcc, m1_f1 = evaluate_metrics(y_true_boot, probs1_boot)
        m1_prauc_list.append(m1_prauc)
        m1_mcc_list.append(m1_mcc)
        m1_f1_list.append(m1_f1)
        
        # Evaluate Model 2
        m2_prauc, m2_mcc, m2_f1 = evaluate_metrics(y_true_boot, probs2_boot)
        m2_prauc_list.append(m2_prauc)
        m2_mcc_list.append(m2_mcc)
        m2_f1_list.append(m2_f1)
        
        # Paired Differences
        diff_prauc.append(m1_prauc - m2_prauc)
        diff_mcc.append(m1_mcc - m2_mcc)
        diff_f1.append(m1_f1 - m2_f1)
        
        if (i+1) % 100 == 0:
            print(f"Bootstrap Iteration {i+1}/{n_iterations}")

    def get_stats(data_list):
        mean_val = np.mean(data_list)
        ci_lower = np.percentile(data_list, 2.5)
        ci_upper = np.percentile(data_list, 97.5)
        return mean_val, ci_lower, ci_upper

    return {
        "E1.3": {
            "PR-AUC": get_stats(m1_prauc_list),
            "MCC": get_stats(m1_mcc_list),
            "F1": get_stats(m1_f1_list)
        },
        "E1.5": {
            "PR-AUC": get_stats(m2_prauc_list),
            "MCC": get_stats(m2_mcc_list),
            "F1": get_stats(m2_f1_list)
        },
        "Difference (E1.3 - E1.5)": {
            "PR-AUC": get_stats(diff_prauc),
            "MCC": get_stats(diff_mcc),
            "F1": get_stats(diff_f1)
        }
    }

print("1. Loading ground truth")
test_pv_df = pd.read_json("data/manifests/test_primevul_frozen.jsonl", lines=True)
test_pv_df = test_pv_df.dropna(subset=['func_code', 'label'])
y_true = test_pv_df['label'].astype(int).values

print(f"Loaded {len(y_true)} test samples.")

print("2. Loading Model predictions")
data1 = np.load('results/cache_E1.3.npz', allow_pickle=True)
probs1 = data1['pv']

data2 = np.load('results/cache_E1.5.npz', allow_pickle=True)
probs2 = data2['pv']

print(f"E1.3 probs shape: {probs1.shape}")
print(f"E1.5 probs shape: {probs2.shape}")

print("3. Running Paired Hierarchical Bootstrap")
stats = hierarchical_bootstrap(y_true, probs1, probs2, n_iterations=1000)

print("\nBOOTSTRAP RESULTS")
for model, metrics in stats.items():
    print(f"\n[{model}]")
    for metric, (mean_val, ci_lower, ci_upper) in metrics.items():
        print(f"  {metric}: {mean_val:.4f} [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")
