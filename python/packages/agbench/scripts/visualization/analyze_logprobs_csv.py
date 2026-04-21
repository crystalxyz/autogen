#!/usr/bin/env python3
"""Analyze logprobs CSV to compare correct vs incorrect task distributions.

Usage:
    python scripts/analyze_logprobs_csv.py reports/logprobs.csv
    python scripts/analyze_logprobs_csv.py reports/logprobs.csv -o reports/logprobs_analysis.png
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd


METRICS = [
    # Whole-sequence
    ("mean_logprob", "Mean Log-Probability"),
    ("mean_prob", "Mean Token Probability"),
    ("mean_entropy", "Mean Entropy (top-k)"),
    ("high_conf_ratio", "High Confidence Ratio (p>0.9)"),
    ("perplexity", "Perplexity"),
    ("std_logprob", "Std Dev of Log-Probability"),
    ("min_logprob", "Min Log-Probability"),
    ("p10_logprob", "10th Percentile Log-Probability"),
    ("low_conf_ratio", "Low Confidence Ratio (p<0.5)"),
    # First-token signals
    ("ft_entropy", "First-Token Entropy"),
    ("ft_top1_prob", "First-Token Top-1 Probability"),
    ("ft_margin", "First-Token Margin (top1 - top2)"),
    ("ft_topk_spread", "First-Token Top-k Spread"),
    ("ft_temp_sensitivity", "First-Token Temp Sensitivity"),
    ("ft_hedging_max_prob", "First-Token Hedging Max Prob"),
    # Short-generation signals
    ("fn_mean_entropy", "First-N Mean Entropy"),
    ("fn_entropy_var", "First-N Entropy Variance"),
    ("fn_min_prob", "First-N Min Probability"),
    ("fn_perplexity", "First-N Perplexity"),
    ("fn_mean_margin", "First-N Mean Margin"),
    ("fn_min_margin", "First-N Min Margin"),
    # Aggregated
    ("fn_entropy_slope", "First-N Entropy Slope"),
    ("full_entropy_slope", "Full Entropy Slope"),
]


def print_summary(df: pd.DataFrame):
    """Print per-task and aggregate summaries."""
    print("=" * 100)
    print(f"{'Task':<18} {'Rep':<5} {'Pass':<6} {'MeanLP':>8} {'MeanProb':>9} "
          f"{'Entropy':>8} {'HiConf%':>8} {'LoConf%':>8} {'PPL':>8} {'#Tok':>6}")
    print("-" * 100)
    for _, row in df.sort_values(["task_id", "rep"]).iterrows():
        entropy_str = f"{row['mean_entropy']:>8.4f}" if pd.notna(row['mean_entropy']) else f"{'N/A':>8}"
        print(f"{row['task_id']:<18} {row['rep']:<5} {'YES' if row['correct'] else 'NO':<6} "
              f"{row['mean_logprob']:>8.4f} {row['mean_prob']:>9.4f} "
              f"{entropy_str} {row['high_conf_ratio']:>7.1%} {row['low_conf_ratio']:>7.1%} "
              f"{row['perplexity']:>8.2f} {row['num_tokens']:>6.0f}")

    # Aggregate
    print("\n" + "=" * 100)
    print("AGGREGATE COMPARISON")
    print("=" * 100)

    for label, group in [("CORRECT", df[df["correct"] == 1]), ("INCORRECT", df[df["correct"] == 0])]:
        if group.empty:
            print(f"\n{label}: no samples")
            continue
        print(f"\n{label} (n={len(group)}):")
        stat_cols = ["mean_logprob", "std_logprob", "mean_prob", "mean_entropy",
                     "high_conf_ratio", "low_conf_ratio", "perplexity", "min_logprob",
                     "p10_logprob", "num_tokens",
                     "ft_entropy", "ft_top1_prob", "ft_margin", "ft_topk_spread",
                     "ft_temp_sensitivity", "ft_hedging_max_prob",
                     "fn_mean_entropy", "fn_entropy_var", "fn_min_prob",
                     "fn_perplexity", "fn_mean_margin", "fn_min_margin",
                     "fn_entropy_slope", "full_entropy_slope"]
        for col in stat_cols:
            vals = group[col].dropna()
            if not vals.empty:
                print(f"  {col:<22}: mean={vals.mean():>9.4f}  std={vals.std():>9.4f}  "
                      f"min={vals.min():>9.4f}  max={vals.max():>9.4f}")

    # Statistical tests
    print("\n" + "=" * 100)
    print("STATISTICAL TESTS (Welch's t-test)")
    print("=" * 100)
    correct = df[df["correct"] == 1]
    incorrect = df[df["correct"] == 0]

    if len(correct) >= 2 and len(incorrect) >= 2:
        from scipy import stats
        test_cols = ["mean_logprob", "mean_prob", "mean_entropy", "high_conf_ratio",
                     "perplexity", "std_logprob", "low_conf_ratio",
                     "ft_entropy", "ft_top1_prob", "ft_margin", "ft_topk_spread",
                     "ft_temp_sensitivity", "ft_hedging_max_prob",
                     "fn_mean_entropy", "fn_entropy_var", "fn_min_prob",
                     "fn_perplexity", "fn_mean_margin", "fn_min_margin",
                     "fn_entropy_slope", "full_entropy_slope"]
        for col in test_cols:
            c_vals = correct[col].dropna()
            i_vals = incorrect[col].dropna()
            if len(c_vals) >= 2 and len(i_vals) >= 2:
                t_stat, p_val = stats.ttest_ind(c_vals, i_vals, equal_var=False)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {col:<22}: t={t_stat:>7.3f}  p={p_val:>8.5f} {sig}")
    else:
        print("  Not enough samples for t-tests (need >= 2 per group)")


def plot_analysis(df: pd.DataFrame, output_path: str):
    """Generate comparison charts."""
    import matplotlib.pyplot as plt

    correct = df[df["correct"] == 1]
    incorrect = df[df["correct"] == 0]

    n_metrics = len(METRICS)
    n_cols = 4
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    fig.suptitle("Logprobs Analysis: Correct vs Incorrect Tasks", fontsize=14, y=0.98)

    # Hide unused axes
    for ax in axes.flat[n_metrics:]:
        ax.set_visible(False)

    for ax, (metric, title) in zip(axes.flat, METRICS):
        c_vals = correct[metric].dropna().values
        i_vals = incorrect[metric].dropna().values

        positions = []
        data = []
        labels = []
        colors = []
        if len(c_vals) > 0:
            positions.append(1)
            data.append(c_vals)
            labels.append(f"Correct\n(n={len(c_vals)})")
            colors.append("#2ecc71")
        if len(i_vals) > 0:
            positions.append(2)
            data.append(i_vals)
            labels.append(f"Incorrect\n(n={len(i_vals)})")
            colors.append("#e74c3c")

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for pos, vals, color in zip(positions, data, colors):
                jitter = np.random.normal(0, 0.04, len(vals))
                ax.scatter([pos + j for j in jitter], vals, alpha=0.6, s=30,
                           color=color, edgecolors="black", linewidth=0.5, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved chart to {output_path}")

    # Per-task scatter: mean_logprob vs correctness
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        color = "#2ecc71" if row["correct"] else "#e74c3c"
        ax2.scatter(row["mean_logprob"], row["mean_entropy"] if pd.notna(row["mean_entropy"]) else 0,
                    c=color, s=60, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax2.annotate(row["task_id"].replace("HumanEval_", ""),
                     (row["mean_logprob"], row["mean_entropy"] if pd.notna(row["mean_entropy"]) else 0),
                     fontsize=7, ha="center", va="bottom")

    ax2.set_xlabel("Mean Log-Probability")
    ax2.set_ylabel("Mean Entropy")
    ax2.set_title("Mean Log-Probability vs Entropy (green=correct, red=incorrect)")
    ax2.grid(alpha=0.3)

    scatter_path = output_path.replace(".png", "_scatter.png")
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"Saved scatter to {scatter_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze logprobs CSV.")
    parser.add_argument("csv_path", help="Path to logprobs CSV file")
    parser.add_argument("-o", "--output", default="reports/logprobs_analysis.png", help="Output chart path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")
    print(f"  Correct: {(df['correct'] == 1).sum()}, Incorrect: {(df['correct'] == 0).sum()}")
    print(f"  Tasks: {df['task_id'].nunique()}, Runs: {df['run'].nunique()}\n")

    print_summary(df)
    plot_analysis(df, args.output)


if __name__ == "__main__":
    main()
