#!/usr/bin/env python3
"""Visualize logprobs distributions for correct vs incorrect tasks.

Usage:
    python scripts/visualize_logprobs.py reports/csv/logprobs_1.7b_nothink_v2.csv
    python scripts/visualize_logprobs.py reports/csv/logprobs_1.7b_nothink_v2.csv -o reports/logprobs_viz.png
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METRICS = [
    # Whole-sequence
    ("mean_logprob", "Mean Log-Probability"),
    ("mean_entropy", "Mean Entropy (top-k)"),
    ("mean_prob", "Mean Token Probability"),
    ("high_conf_ratio", "High Confidence Ratio (p>0.9)"),
    ("perplexity", "Perplexity"),
    ("std_logprob", "Std Dev of Log-Probability"),
    # Tail metrics
    ("tail_10pct_mean_logprob", "Tail 10% Mean Log-Prob"),
    ("tail_5pct_mean_logprob", "Tail 5% Mean Log-Prob"),
    ("tail_1pct_mean_logprob", "Tail 1% Mean Log-Prob"),
    # Entropy spikes
    ("entropy_spike_05_ratio", "Entropy Spike Ratio (>0.5)"),
    ("entropy_spike_10_ratio", "Entropy Spike Ratio (>1.0)"),
    ("entropy_spike_15_ratio", "Entropy Spike Ratio (>1.5)"),
    # First-token signals
    ("ft_entropy", "First-Token Entropy"),
    ("ft_top1_prob", "First-Token Top-1 Prob"),
    ("ft_margin", "First-Token Margin (top1-top2)"),
    ("ft_topk_spread", "First-Token Top-k Spread"),
    ("ft_temp_sensitivity", "First-Token Temp Sensitivity"),
    ("ft_hedging_max_prob", "First-Token Hedging Max Prob"),
    # Short-generation signals (first 20 tokens)
    ("fn_mean_entropy", "First-20 Mean Entropy"),
    ("fn_entropy_var", "First-20 Entropy Variance"),
    ("fn_min_prob", "First-20 Min Probability"),
    ("fn_perplexity", "First-20 Perplexity"),
    ("fn_mean_margin", "First-20 Mean Margin"),
    ("fn_min_margin", "First-20 Min Margin"),
    # Aggregated / derived
    ("fn_entropy_slope", "First-20 Entropy Slope"),
    ("full_entropy_slope", "Full Entropy Slope"),
]


def main():
    parser = argparse.ArgumentParser(description="Visualize logprobs distributions.")
    parser.add_argument("csv_path", help="Path to logprobs summary CSV")
    parser.add_argument("-o", "--output", default="reports/logprobs_viz.png",
                        help="Output image path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    n_correct = (df["correct"] == 1).sum()
    n_incorrect = (df["correct"] == 0).sum()
    print(f"Loaded {len(df)} rows: {n_correct} correct, {n_incorrect} incorrect")

    correct_df = df[df["correct"] == 1]
    incorrect_df = df[df["correct"] == 0]

    n_metrics = len(METRICS)
    n_cols = 4
    n_rows = math.ceil(n_metrics / n_cols)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))

    # Hide unused axes
    for ax in axes.flat[n_metrics:]:
        ax.set_visible(False)

    for ax, (metric, title) in zip(axes.flat, METRICS):
        c_vals = correct_df[metric].dropna()
        i_vals = incorrect_df[metric].dropna()

        if c_vals.empty and i_vals.empty:
            ax.set_visible(False)
            continue

        all_vals = pd.concat([c_vals, i_vals])
        lo = all_vals.min()
        hi = all_vals.max()
        pad = (hi - lo) * 0.05 if hi > lo else 0.1
        bins = np.linspace(lo - pad, hi + pad, 30)

        if len(c_vals) > 0:
            ax.hist(c_vals, bins=bins, alpha=0.6, color="#2ecc71", edgecolor="white",
                    linewidth=0.5, label=f"Pass (n={len(c_vals)})",
                    weights=np.ones(len(c_vals)) / len(c_vals) * 100)
        if len(i_vals) > 0:
            ax.hist(i_vals, bins=bins, alpha=0.6, color="#e74c3c", edgecolor="white",
                    linewidth=0.5, label=f"Fail (n={len(i_vals)})",
                    weights=np.ones(len(i_vals)) / len(i_vals) * 100)

        # Add vertical lines for medians
        if len(c_vals) > 0:
            ax.axvline(c_vals.median(), color="#1a9641", linestyle="--", linewidth=1.2, alpha=0.8)
        if len(i_vals) > 0:
            ax.axvline(i_vals.median(), color="#b2182b", linestyle="--", linewidth=1.2, alpha=0.8)

        # Significance annotation
        sig_text = ""
        if len(c_vals) >= 2 and len(i_vals) >= 2:
            _, p_val = stats.ttest_ind(c_vals, i_vals, equal_var=False)
            if p_val < 0.001:
                sig_text = "***"
            elif p_val < 0.01:
                sig_text = "**"
            elif p_val < 0.05:
                sig_text = "*"
            if sig_text:
                ax.text(0.97, 0.95, f"p<{'0.001' if p_val < 0.001 else '0.01' if p_val < 0.01 else '0.05'} {sig_text}",
                        transform=ax.transAxes, fontsize=7, fontweight="bold",
                        va="top", ha="right", color="#333",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.3))

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel("% of group", fontsize=7)
        ax.legend(fontsize=6.5, loc="upper right" if not sig_text else "upper left")
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Logprobs Signal Distributions: Pass vs Fail (1.7B NoThink)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
