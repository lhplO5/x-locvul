import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef, f1_score,
    precision_score, recall_score, accuracy_score, confusion_matrix,
    roc_auc_score, brier_score_loss
)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

DATA_DIR = "./data/manifests"

print("1. Loading datasets...")
train_df = pd.read_json(f"{DATA_DIR}/train_frozen.jsonl", lines=True)
val_df = pd.read_json(f"{DATA_DIR}/val_frozen.jsonl", lines=True)
test_pv_df = pd.read_json(f"{DATA_DIR}/test_primevul_frozen.jsonl", lines=True)

# Drop missing values if any
train_df = train_df.dropna(subset=['func_code', 'label'])
val_df = val_df.dropna(subset=['func_code', 'label'])
test_pv_df = test_pv_df.dropna(subset=['func_code', 'label'])

X_train_raw = train_df['func_code'].astype(str)
y_train = train_df['label'].astype(int)

X_val_raw = val_df['func_code'].astype(str)
y_val = val_df['label'].astype(int)

X_test_pv_raw = test_pv_df['func_code'].astype(str)
y_test_pv = test_pv_df['label'].astype(int)

print("2. Extracting Lexical Features (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train_raw)
X_val = vectorizer.transform(X_val_raw)
X_test_pv = vectorizer.transform(X_test_pv_raw)
print(f"TF-IDF Feature shape: {X_train.shape}")

def evaluate_model(y_true, y_probs, threshold=0.5):
    preds = (y_probs > threshold).astype(int)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    accuracy = accuracy_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds)
    
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    roc_auc = roc_auc_score(y_true, y_probs)
    brier = brier_score_loss(y_true, y_probs)
    ece = expected_calibration_error(y_true, y_probs)
    
    return {
        "pr_auc": pr_auc,
        "mcc": mcc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece": ece
    }

print("\n3. Training E1.0 - Majority Baseline (Sanity Floor)...")
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# For DummyClassifier, predict_proba returns 1.0 for the majority class and 0.0 for others
dummy_probs_val = dummy_clf.predict_proba(X_val)[:, 1]
dummy_metrics_val = evaluate_model(y_val, dummy_probs_val)
print("Majority Baseline - Validation Metrics:")
for k, v in dummy_metrics_val.items():
    print(f"  {k}: {v:.4f}")

print("\n4. Training E1.0 - Lexical/Logistic Baseline...")
# class_weight='balanced' to handle the 98/2 class imbalance
log_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_clf.fit(X_train, y_train)

log_probs_val = log_clf.predict_proba(X_val)[:, 1]
log_metrics_val = evaluate_model(y_val, log_probs_val)
print("Logistic Regression - Validation Metrics:")
for k, v in log_metrics_val.items():
    print(f"  {k}: {v:.4f}")

log_probs_test = log_clf.predict_proba(X_test_pv)[:, 1]
log_metrics_test = evaluate_model(y_test_pv, log_probs_test)
print("\nLogistic Regression - Test (PrimeVul) Metrics:")
for k, v in log_metrics_test.items():
    print(f"  {k}: {v:.4f}")

print("\n[OK] E1.0 Baseline execution completed.")
