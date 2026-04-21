#!/usr/bin/env python3
"""Combined P50/P90/Mean Pareto frontiers for four-turn-nc setups."""

import csv
import os

import matplotlib.pyplot as plt
from adjustText import adjust_text

INPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "csv", "four_turn_nc_stats.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "pareto")

XLIM_MAX = 60


def load_data():
    setups = []
    with open(INPUT_CSV) as f:
        for row in csv.DictReader(f):
            setups.append({
                "setup": row["setup"].replace("four-turn-nc-", ""),
                "accuracy": float(row["accuracy"]) * 100,
                "p50": float(row["p50_latency"]),
                "p90": float(row["p90_latency"]),
                "mean": float(row["mean_latency"]),
            })
    return setups


def pareto_frontier(points, x_key, y_key):
    sorted_idx = sorted(range(len(points)), key=lambda i: points[i][x_key])
    frontier = []
    best_y = -float("inf")
    for i in sorted_idx:
        if points[i][y_key] > best_y:
            frontier.append(i)
            best_y = points[i][y_key]
    return frontier


def main():
    setups = load_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))

    metrics = [
        ("p50", "P50", "#2ca02c", "o"),
        ("p90", "P90", "#E24A33", "s"),
        ("mean", "Mean", "#4878CF", "D"),
    ]

    all_texts, all_xs, all_ys = [], [], []

    for x_key, label, color, marker in metrics:
        frontier_idx = pareto_frontier(setups, x_key, "accuracy")
        f_in = [i for i in frontier_idx if setups[i][x_key] <= XLIM_MAX]
        fx = [setups[i][x_key] for i in f_in]
        fy = [setups[i]["accuracy"] for i in f_in]
        ax.plot(fx, fy, color=color, linewidth=2, linestyle="--", alpha=0.7, zorder=2)
        ax.scatter(fx, fy, s=70, zorder=4, color=color, edgecolors="white",
                   linewidths=0.5, marker=marker, label=f"{label} Frontier")

        for i in f_in:
            all_texts.append(ax.text(
                setups[i][x_key], setups[i]["accuracy"], setups[i]["setup"],
                fontsize=6.5, fontweight="bold", color=color, ha="center", va="center",
            ))
            all_xs.append(setups[i][x_key])
            all_ys.append(setups[i]["accuracy"])

    adjust_text(
        all_texts, x=all_xs, y=all_ys, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4, alpha=0.5),
        force_points=(2.0, 2.0), force_text=(1.5, 1.5), expand=(2.0, 2.5),
        iterations=200,
    )
    for t in all_texts:
        t.set_clip_on(True)

    ax.set_xlabel("Latency (s)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Four-Turn No-Context: Pareto Frontiers (P50 / P90 / Mean)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, XLIM_MAX)
    ymin = min(all_ys) - 2 if all_ys else 80
    ax.set_ylim(ymin, 100)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "pareto_4t_combined.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
