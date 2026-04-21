#!/usr/bin/env python3
"""Box + strip plot comparing E2E latency distributions across setups."""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from token_usage_stat import collect_step_latencies

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

SETUP_DIRS = {
    "Two-Turn\n(0.6B-NT -> 4B-T)": os.path.join(BASE_DIR, "outputs", "two-turn-0.6nt-4t"),
    "Two-Turn\n(0.6B-NT -> 8B-T)": os.path.join(BASE_DIR, "outputs", "two-turn-0.6nt-8t"),
    "Two-Turn\n(1.7B-NT -> 4B-T)": os.path.join(BASE_DIR, "outputs", "two-turn-1.7nt-4t"),
    "Two-Turn\n(1.7B-NT -> 8B-T)": os.path.join(BASE_DIR, "outputs", "two-turn-1.7nt-8t"),
}


def compute_e2e_latencies(setup_dirs: dict[str, str]) -> dict[str, np.ndarray]:
    """Compute per-task E2E latency (sum of all steps) for each setup."""
    result = {}
    for label, path in setup_dirs.items():
        print(f"Processing {label} from {path}...")
        df = collect_step_latencies(path)
        if df.empty:
            print(f"  No data found, skipping.")
            continue
        e2e = df.groupby(["task", "trial"])["latency_s"].sum().values
        result[label] = e2e
        print(f"  {len(e2e)} tasks, mean={e2e.mean():.2f}s, median={np.median(e2e):.2f}s")
    return result


def plot_e2e_boxplot():
    e2e_data = compute_e2e_latencies(SETUP_DIRS)
    if not e2e_data:
        print("No data to plot.")
        return

    Y_MAX = 300
    labels = list(e2e_data.keys())
    raw_data = [e2e_data[l] for l in labels]  # unclipped, for stats
    data = [np.clip(d, None, Y_MAX) for d in raw_data]  # clipped, for plotting

    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ["#5B9BD5", "#3B82F6", "#A855F7", "#EC4899"]
    positions = list(range(1, len(data) + 1))

    # Violin plot for distribution shape
    vp = ax.violinplot(data, positions=positions, showextrema=False, widths=0.7)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.2)
        body.set_edgecolor(color)
        body.set_linewidth(1.2)

    # Box plot overlay (narrow)
    bp = ax.boxplot(
        data,
        positions=positions,
        tick_labels=labels,
        patch_artist=True,
        widths=0.25,
        showmeans=True,
        showfliers=False,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6, zorder=5),
        medianprops=dict(color="white", linewidth=2, zorder=4),
        whiskerprops=dict(color="gray", linewidth=1.5, linestyle="--"),
        capprops=dict(color="gray", linewidth=1.5),
        boxprops=dict(linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")

    # Strip plot (jittered individual points)
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            positions[i] + jitter,
            vals,
            s=14,
            color=color,
            alpha=0.45,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    # Annotate stats below x-axis labels (use raw unclipped data for stats)
    for i, (label, vals) in enumerate(zip(labels, raw_data)):
        mean = vals.mean()
        median = np.median(vals)
        p90 = np.percentile(vals, 90)
        n = len(vals)
        stats_text = f"n={n} | mean={mean:.1f}s\nmed={median:.1f}s | p90={p90:.1f}s"
        ax.annotate(
            stats_text,
            xy=(positions[i], 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -52),
            textcoords="offset points",
            fontsize=7.5,
            va="top",
            ha="center",
            fontweight="bold",
            color=colors[i],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.1, edgecolor=colors[i], linewidth=1),
        )

    ax.set_ylim(-5, Y_MAX + 10)
    ax.set_ylabel("E2E Latency per Task (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("E2E Latency Distribution — Two-Turn Setups (HumanEval)", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(bottom=0.22)
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "reports", "e2e_latency_boxplot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    plot_e2e_boxplot()
