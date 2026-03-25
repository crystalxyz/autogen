#!/usr/bin/env python3
"""Stacked histogram of E2E latency split by outcome (pass turn 1, pass turn 2, fail)
for all two_turn_nc runs. Each setup gets a column of 3 vertically-stacked histograms."""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

SETUP_DIRS = {
    "0.6B-NT -> 4B-T": os.path.join(BASE_DIR, "outputs", "two-turn-nc-0.6nt-4t"),
    "0.8B-NT -> 8B-T": os.path.join(BASE_DIR, "outputs", "two-turn-nc-0.8nt-8t"),
    "1.7B-NT -> 4B-T": os.path.join(BASE_DIR, "outputs", "two-turn-nc-1.7nt-4t"),
    "1.7B-NT -> 8B-T": os.path.join(BASE_DIR, "outputs", "two-turn-nc-1.7nt-8t"),
}

OUTCOME_LABELS = ["Pass (Turn 1)", "Pass (Turn 2)", "Fail"]
OUTCOME_COLORS = ["#22C55E", "#F59E0B", "#EF4444"]  # green, amber, red


def load_latencies_by_outcome(output_dir: str) -> dict[str, list[float]]:
    """Return {outcome_label: [latencies]} from result.json."""
    results_dir = os.path.join(output_dir, "Results")
    if not os.path.isdir(results_dir):
        return {}
    scenario_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not scenario_dirs:
        return {}
    result_path = os.path.join(results_dir, scenario_dirs[0], "result.json")
    if not os.path.exists(result_path):
        return {}

    with open(result_path) as f:
        data = json.load(f)

    buckets = {label: [] for label in OUTCOME_LABELS}
    for task in data["tasks"].values():
        for rep in task["repetitions"].values():
            if rep["status"] != "completed":
                continue
            t = rep["elapsed_time"]
            if rep["success"] and rep["turns"] <= 2:
                buckets["Pass (Turn 1)"].append(t)
            elif rep["success"]:
                buckets["Pass (Turn 2)"].append(t)
            else:
                buckets["Fail"].append(t)
    return buckets


def plot():
    plt.style.use("seaborn-v0_8-whitegrid")

    n_setups = len(SETUP_DIRS)
    n_outcomes = len(OUTCOME_LABELS)
    X_MAX = 300
    BINS = np.linspace(0, X_MAX, 61)  # 5s bins

    # 3 rows (outcomes) x 4 cols (setups), sharing x within columns and y within rows
    fig, axes = plt.subplots(
        n_outcomes, n_setups,
        figsize=(4.5 * n_setups, 2.5 * n_outcomes),
        sharex=True,
        sharey="row",
    )

    for col_idx, (setup_label, setup_path) in enumerate(SETUP_DIRS.items()):
        buckets = load_latencies_by_outcome(setup_path)

        for row_idx, (lbl, color) in enumerate(zip(OUTCOME_LABELS, OUTCOME_COLORS)):
            ax = axes[row_idx][col_idx]
            vals = np.array(buckets.get(lbl, []))
            vals_clipped = np.clip(vals, None, X_MAX) if len(vals) > 0 else np.array([])

            if len(vals_clipped) > 0:
                ax.hist(
                    vals_clipped,
                    bins=BINS,
                    color=color,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.5,
                )
                # Stats text in top-right corner
                mean = vals.mean()
                median = np.median(vals)
                p90 = np.percentile(vals, 90)
                n = len(vals)
                stats = f"n={n}  mean={mean:.1f}s\nmed={median:.1f}s  p90={p90:.1f}s"
                ax.text(
                    0.97, 0.92, stats,
                    transform=ax.transAxes,
                    fontsize=7,
                    fontweight="bold",
                    va="top",
                    ha="right",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1,
                              edgecolor=color, linewidth=0.8),
                )
            else:
                ax.text(
                    0.5, 0.5, "n=0",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="center",
                    color="gray",
                )

            # Row labels on leftmost column
            if col_idx == 0:
                ax.set_ylabel(lbl, fontsize=10, fontweight="bold", color=color)

            # Column titles on top row
            if row_idx == 0:
                ax.set_title(f"Two-Turn NC\n({setup_label})", fontsize=10, fontweight="bold")

            # X label on bottom row
            if row_idx == n_outcomes - 1:
                ax.set_xlabel("E2E Latency (s)", fontsize=9)

            ax.set_xlim(0, X_MAX)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        "E2E Latency Distribution by Outcome — Two-Turn NC (HumanEval)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "reports", "e2e_latency_by_outcome_nc.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    plot()
