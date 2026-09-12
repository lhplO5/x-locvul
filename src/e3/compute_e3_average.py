import pandas as pd
import glob
import os

def main():
    results_dir = r"./results/e3"
    
    metric_files = glob.glob(os.path.join(results_dir, "e3_metrics_seed_*.csv"))
    if len(metric_files) < 5:
        print(f"Found {len(metric_files)} file metrics. Waiting for SCP to finish...")
        return
        
    all_seed_summaries = []
    for f in metric_files:
        df = pd.read_csv(f)
        all_seed_summaries.append(df)
        
    print("\n" + "="*70)
    print("FINAL AVERAGE METRICS (ACROSS 5 SEEDS) - CALCULATED LOCALLY")
    print("="*70)
    
    merged_df = pd.concat(all_seed_summaries)
    avg_df = merged_df.groupby("Strategy").mean().reset_index()
    
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
    
    avg_path = os.path.join(results_dir, "e3_metrics_AVERAGE_5_SEEDS.csv")
    display_df.to_csv(avg_path, index=False)
    print(f"\nSaved average metrics to: {avg_path}")

if __name__ == '__main__':
    main()
