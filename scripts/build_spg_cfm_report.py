import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "per_seed_external.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    metrics = [
        "aug_f1",
        "spg_cfm_train_mse_mean",
        "spg_cfm_train_cosine_mean",
        "spg_cfm_alignment_to_spg_mean",
        "spg_cfm_generated_direction_pairwise_cosine_mean",
        "spg_cfm_effective_aug_multiplier",
        "bridge_success_rate",
        "safe_clip_rate",
        "augmentation_build_time_sec",
        "generation_time_per_aug_sample_ms",
    ]

    available_metrics = [m for m in metrics if m in df.columns]

    print(f"--- SPG-CFM Report: {results_dir.name} ---")
    
    # 1. Performance and Diagnostics Summary
    agg = df.groupby("method")[["aug_f1"] + available_metrics].mean(numeric_only=True).round(6)
    print("\n[Average Metrics by Method]")
    print(agg.to_string())

    # 2. Speed Audit vs wDBA and U5 (csta_topk_uniform_top5)
    if "augmentation_build_time_sec" in df.columns:
        print("\n[Speed Audit]")
        speed_df = df.groupby("method")["augmentation_build_time_sec"].mean().reset_index()
        wdba_time = speed_df.loc[speed_df["method"] == "wdba_sameclass", "augmentation_build_time_sec"]
        u5_time = speed_df.loc[speed_df["method"] == "csta_topk_uniform_top5", "augmentation_build_time_sec"]
        
        wdba_val = wdba_time.values[0] if len(wdba_time) > 0 else np.nan
        u5_val = u5_time.values[0] if len(u5_time) > 0 else np.nan
        
        for _, row in speed_df.iterrows():
            method = row["method"]
            time_sec = row["augmentation_build_time_sec"]
            rel_wdba = time_sec / wdba_val if pd.notna(wdba_val) else np.nan
            rel_u5 = time_sec / u5_val if pd.notna(u5_val) else np.nan
            print(f"{method:35s}: {time_sec:8.2f}s | vs wDBA: {rel_wdba:5.2f}x | vs U5: {rel_u5:5.2f}x")

    # 3. Method-Dataset Breakdown for F1
    print("\n[F1 by Dataset and Method]")
    pivot = df.pivot_table(index="dataset", columns="method", values="aug_f1", aggfunc="mean").round(4)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
