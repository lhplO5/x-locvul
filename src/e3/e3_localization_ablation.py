import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    RobertaModel
)

DATA_DIR = "./data/linevul"
OUTPUT_DIR = "./results/e3"
SEEDS = [10, 42, 100, 420, 1000]
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_size", type=int, default=10, help="Beam size for generation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    return parser.parse_args()

unixcoder_tokenizer = None
unixcoder_model = None

def init_unixcoder():
    global unixcoder_tokenizer, unixcoder_model
    if unixcoder_model is None:
        print("Loading UniXCoder for Semantic Projection (E3.4)")
        unixcoder_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base", use_fast=False)
        unixcoder_model = RobertaModel.from_pretrained("microsoft/unixcoder-base").to(device)
        unixcoder_model.eval()

def get_unixcoder_embedding(texts):
    """Get normalized embeddings from UniXCoder."""
    tokens = unixcoder_tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = unixcoder_model(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=1)

def semantic_projection(generated_line, actual_lines):
    """Project a hallucinated line onto the closest real line via cosine similarity."""
    if not actual_lines:
        return generated_line
    all_texts = [generated_line] + actual_lines
    all_embs = get_unixcoder_embedding(all_texts)
    gen_emb = all_embs[0:1]
    actual_embs = all_embs[1:]
    similarities = torch.matmul(gen_emb, actual_embs.T).squeeze(0)
    best_idx = torch.argmax(similarities).item()
    return actual_lines[best_idx]

def levenshtein_distance(s1, s2):
    """Compute edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def fuzzy_match(generated_line, actual_lines):
    """Find the closest actual line using Levenshtein distance."""
    if not actual_lines:
        return generated_line
    gen_norm = normalize_code_line(generated_line)
    best_line = actual_lines[0]
    best_dist = float('inf')
    for line in actual_lines:
        dist = levenshtein_distance(gen_norm, normalize_code_line(line))
        if dist < best_dist:
            best_dist = dist
            best_line = line
    return best_line

def lcs_length(x, y):
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l_score(prediction, reference):
    """Compute ROUGE-L F1 score between two strings (token-level)."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0
    recall = lcs / len(ref_tokens) if ref_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def normalize_code_line(line):
    """Strip and collapse whitespace for fair comparison."""
    return " ".join(str(line).split())

def is_intrinsic(pred_line, func_lines_normalized):
    """Check if predicted line EXACTLY matches any line in the function."""
    pred_norm = normalize_code_line(pred_line)
    if not pred_norm:
        return False
    return pred_norm in func_lines_normalized

def is_rescue_successful(rescued_line, gt_raw):
    """Check if a rescued line matches any ground-truth flaw line."""
    rescued_norm = normalize_code_line(rescued_line)
    gt_lines = [normalize_code_line(g) for g in str(gt_raw).split('\n') if g.strip()]
    for gt in gt_lines:
        if rescued_norm == gt or rescued_norm in gt or gt in rescued_norm:
            return True
    return False

def calculate_metrics(predictions, ground_truths):
    top1, top3, top5, top10 = 0, 0, 0, 0
    mrr_sum = 0
    ifa_sum = 0
    ifa_count = 0

    for preds, gt_raw in zip(predictions, ground_truths):
        gt_lines = [normalize_code_line(g) for g in str(gt_raw).split('\n') if g.strip()]
        preds_norm = [normalize_code_line(p) for p in preds]

        if not preds_norm or not gt_lines:
            continue

        rank = -1
        for i, p in enumerate(preds_norm):
            if not p:
                continue
            for gt in gt_lines:
                if p == gt or p in gt or gt in p:
                    rank = i + 1
                    break
            if rank != -1:
                break

        if rank != -1:
            if rank <= 1: top1 += 1
            if rank <= 3: top3 += 1
            if rank <= 5: top5 += 1
            if rank <= 10: top10 += 1
            mrr_sum += 1.0 / rank
            ifa_sum += (rank - 1)
            ifa_count += 1

    n = len(ground_truths)
    if n == 0:
        return {"Top-1": 0, "Top-5": 0, "Top-10": 0, "MRR": 0, "IFA": 0}

    avg_ifa = round(ifa_sum / ifa_count, 4) if ifa_count > 0 else float('inf')

    return {
        "Top-1": round(top1 / n, 4),
        "Top-5": round(top5 / n, 4),
        "Top-10": round(top10 / n, 4),
        "MRR": round(mrr_sum / n, 4),
        "IFA": avg_ifa
    }

def main():
    args = get_args()

    print("Loading test data")
    test_df = pd.read_csv(f"{DATA_DIR}/test_cleaned.csv")
    test_df = test_df[test_df['target'] == 1].copy()
    test_df = test_df.dropna(subset=['func_code', 'flaw_line']).reset_index(drop=True)

    funcs = test_df['func_code'].tolist()
    gts = test_df['flaw_line'].tolist()
    print(f"Test samples (Target=1): {len(funcs)}")

    init_unixcoder()
    
    all_seed_summaries = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"RUNNING INFERENCE FOR SEED: {seed} (Beam Size: {args.beam_size})")
        print(f"{'='*70}")
        
        model_path = f"./models/e3_locator_seed_{seed}_best"
        if not os.path.exists(model_path):
            print(f"ERROR: Best model not found at {model_path}. Skipping seed {seed}...")
            continue

        print(f"Loading CodeT5 model from {model_path}")
        codet5_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=False)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        model.eval()

        results_e31_raw = []
        results_e32_filtered = []
        results_e33_fuzzy = []
        results_e34_semantic = []

        total_preds = 0
        hallucination_count = 0
        fuzzy_rescue_success = 0
        semantic_rescue_success = 0
        rouge_scores = []

        for i in tqdm(range(0, len(funcs), args.batch_size), desc=f"Seed {seed}"):
            batch_funcs = funcs[i:i+args.batch_size]

            inputs = codet5_tokenizer(
                batch_funcs, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=args.beam_size,
                    num_return_sequences=args.beam_size,
                    early_stopping=True
                )

            decoded_preds = codet5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j in range(len(batch_funcs)):
                func_code = batch_funcs[j]
                gt_raw = gts[i + j]
                func_lines = [line.strip() for line in func_code.split('\n') if line.strip()]
                func_lines_norm_set = set(normalize_code_line(l) for l in func_lines)
                raw_preds = decoded_preds[j*args.beam_size : (j+1)*args.beam_size]

                results_e31_raw.append(raw_preds.copy())
                if raw_preds:
                    rouge_scores.append(rouge_l_score(
                        normalize_code_line(raw_preds[0]),
                        normalize_code_line(str(gt_raw))
                    ))

                preds_e32 = []
                for p in raw_preds:
                    total_preds += 1
                    if is_intrinsic(p, func_lines_norm_set):
                        preds_e32.append(p)
                    else:
                        hallucination_count += 1
                results_e32_filtered.append(preds_e32)

                preds_e33 = []
                for p in raw_preds:
                    if is_intrinsic(p, func_lines_norm_set):
                        preds_e33.append(p)
                    else:
                        rescued = fuzzy_match(p, func_lines)
                        preds_e33.append(rescued)
                        if is_rescue_successful(rescued, gt_raw):
                            fuzzy_rescue_success += 1
                # Deduplicate
                seen = set()
                preds_e33_dedup = []
                for p in preds_e33:
                    p_norm = normalize_code_line(p)
                    if p_norm not in seen:
                        seen.add(p_norm)
                        preds_e33_dedup.append(p)
                results_e33_fuzzy.append(preds_e33_dedup)

                preds_e34 = []
                for p in raw_preds:
                    if is_intrinsic(p, func_lines_norm_set):
                        preds_e34.append(p)
                    else:
                        rescued = semantic_projection(p, func_lines)
                        preds_e34.append(rescued)
                        if is_rescue_successful(rescued, gt_raw):
                            semantic_rescue_success += 1
                # Deduplicate
                seen = set()
                preds_e34_dedup = []
                for p in preds_e34:
                    p_norm = normalize_code_line(p)
                    if p_norm not in seen:
                        seen.add(p_norm)
                        preds_e34_dedup.append(p)
                results_e34_semantic.append(preds_e34_dedup)

        # Compute metrics for this seed
        avg_rouge = round(np.mean(rouge_scores), 4) if rouge_scores else 0
        metrics_e31 = calculate_metrics(results_e31_raw, gts)
        
        glhr = round(hallucination_count / total_preds * 100, 2) if total_preds > 0 else 0
        metrics_e32 = calculate_metrics(results_e32_filtered, gts)
        
        prr_fuzzy = round(fuzzy_rescue_success / hallucination_count * 100, 2) if hallucination_count > 0 else 0
        metrics_e33 = calculate_metrics(results_e33_fuzzy, gts)
        
        prr_semantic = round(semantic_rescue_success / hallucination_count * 100, 2) if hallucination_count > 0 else 0
        metrics_e34 = calculate_metrics(results_e34_semantic, gts)

        # Build summary dataframe for this seed
        summary = pd.DataFrame([
            {"Strategy": "E3.1_Raw_Generation", "ROUGE-L": avg_rouge, "GLHR(%)": 0, "PRR(%)": 0,
             "Top-1": metrics_e31["Top-1"], "Top-5": metrics_e31["Top-5"], "Top-10": metrics_e31["Top-10"],
             "MRR": metrics_e31["MRR"], "IFA": metrics_e31["IFA"]},
            {"Strategy": "E3.2_Exact_Match", "ROUGE-L": 0, "GLHR(%)": glhr, "PRR(%)": 0,
             "Top-1": metrics_e32["Top-1"], "Top-5": metrics_e32["Top-5"], "Top-10": metrics_e32["Top-10"],
             "MRR": metrics_e32["MRR"], "IFA": metrics_e32["IFA"]},
            {"Strategy": "E3.3_Fuzzy_Match", "ROUGE-L": 0, "GLHR(%)": 0, "PRR(%)": prr_fuzzy,
             "Top-1": metrics_e33["Top-1"], "Top-5": metrics_e33["Top-5"], "Top-10": metrics_e33["Top-10"],
             "MRR": metrics_e33["MRR"], "IFA": metrics_e33["IFA"]},
            {"Strategy": "E3.4_Semantic_Proj", "ROUGE-L": 0, "GLHR(%)": 0, "PRR(%)": prr_semantic,
             "Top-1": metrics_e34["Top-1"], "Top-5": metrics_e34["Top-5"], "Top-10": metrics_e34["Top-10"],
             "MRR": metrics_e34["MRR"], "IFA": metrics_e34["IFA"]}
        ])
        
        # Save individual seed metrics and details
        summary_path = f"{OUTPUT_DIR}/e3_metrics_seed_{seed}.csv"
        summary.to_csv(summary_path, index=False)
        
        detail_path = f"{OUTPUT_DIR}/e3_ablation_seed_{seed}.csv"
        pd.DataFrame({
            "func_code": funcs,
            "flaw_line": gts,
            "e31_raw_preds": [str(x) for x in results_e31_raw],
            "e32_exact_preds": [str(x) for x in results_e32_filtered],
            "e33_fuzzy_preds": [str(x) for x in results_e33_fuzzy],
            "e34_semantic_preds": [str(x) for x in results_e34_semantic]
        }).to_csv(detail_path, index=False)
        
        all_seed_summaries.append(summary)

    if all_seed_summaries:
        print("\n" + "="*70)
        print("  FINAL AVERAGE METRICS (ACROSS 5 SEEDS)")
        print("="*70)
        
        # Concat all seed dataframes and group by Strategy to compute mean
        merged_df = pd.concat(all_seed_summaries)
        avg_df = merged_df.groupby("Strategy").mean().reset_index()
        
        # Sort strategy to match the original order
        strategy_order = ["E3.1_Raw_Generation", "E3.2_Exact_Match", "E3.3_Fuzzy_Match", "E3.4_Semantic_Proj"]
        avg_df["Strategy"] = pd.Categorical(avg_df["Strategy"], categories=strategy_order, ordered=True)
        avg_df = avg_df.sort_values("Strategy").reset_index(drop=True)
        
        avg_df = avg_df.round(4)
        
        display_df = avg_df.copy().astype(object)
        for idx in range(len(display_df)):
            if display_df.loc[idx, "Strategy"] == "E3.1_Raw_Generation":
                display_df.loc[idx, "GLHR(%)"] = "-"
                display_df.loc[idx, "PRR(%)"] = "-"
            else:
                display_df.loc[idx, "ROUGE-L"] = "-"
                
            if display_df.loc[idx, "Strategy"] == "E3.2_Exact_Match":
                display_df.loc[idx, "PRR(%)"] = "-"
                
            if display_df.loc[idx, "Strategy"] in ["E3.3_Fuzzy_Match", "E3.4_Semantic_Proj"]:
                display_df.loc[idx, "GLHR(%)"] = "-"

        print(display_df.to_string(index=False))
        
        avg_path = f"{OUTPUT_DIR}/e3_metrics_AVERAGE_5_SEEDS.csv"
        display_df.to_csv(avg_path, index=False)
        print(f"\nFinal average metrics saved to {avg_path}")

if __name__ == "__main__":
    main()
