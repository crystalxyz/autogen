#!/usr/bin/env python3
"""Plot P90 and mean latency vs accuracy with three per-tier Pareto frontiers (1T/2T/3T)."""

import csv
import os

import matplotlib.pyplot as plt
from adjustText import adjust_text

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
ONE_TURN_CSV = os.path.join(REPORTS_DIR, "csv", "one_turn_baselines.csv")
TWO_TURN_CSV = os.path.join(REPORTS_DIR, "csv", "two_turn_nc_stats.csv")
THREE_TURN_CSV = os.path.join(REPORTS_DIR, "csv", "three_turn_nc_stats.csv")
OUTPUT_DIR = os.path.join(REPORTS_DIR, "pareto")


def load_one_turn():
    setups = []
    with open(ONE_TURN_CSV) as f:
        for row in csv.DictReader(f):
            setups.append({
                "setup": row["setup"].replace("one-turn-", ""),
                "accuracy": float(row["success_rate"]) * 100,
                "p50": float(row["latency_p50"]),
                "p90": float(row["latency_p90"]),
                "mean": float(row["latency_mean"]),
            })
    return setups


def load_two_turn():
    setups = []
    with open(TWO_TURN_CSV) as f:
        for row in csv.DictReader(f):
            setups.append({
                "setup": row["setup"].replace("two-turn-nc-", ""),
                "accuracy": float(row["accuracy"]) * 100,
                "p50": float(row["p50_latency"]),
                "p90": float(row["p90_latency"]),
                "mean": float(row["mean_latency"]),
            })
    return setups


def load_three_turn():
    setups = []
    with open(THREE_TURN_CSV) as f:
        for row in csv.DictReader(f):
            setups.append({
                "setup": row["setup"].replace("three-turn-nc-", "").replace("three-turn-", ""),
                "accuracy": float(row["accuracy"]) * 100,
                "p50": float(row["p50_latency"]),
                "p90": float(row["p90_latency"]),
                "mean": float(row["mean_latency"]),
            })
    return setups


def pareto_frontier(points, x_key, y_key):
    """Return indices of Pareto-optimal points (minimize x, maximize y)."""
    sorted_idx = sorted(range(len(points)), key=lambda i: points[i][x_key])
    frontier = []
    best_y = -float("inf")
    for i in sorted_idx:
        if points[i][y_key] > best_y:
            frontier.append(i)
            best_y = points[i][y_key]
    return frontier


def plot_three_frontiers(ax, data_1t, data_2t, data_3t, x_key, xlabel, title):
    series = [
        (data_1t, "1-Turn", "#2ca02c", "o"),
        (data_2t, "2-Turn", "#4878CF", "s"),
        (data_3t, "3-Turn", "#E24A33", "D"),
    ]

    all_texts = []
    all_xs = []
    all_ys = []

    for data, label, color, marker in series:
        frontier_idx = pareto_frontier(data, x_key, "accuracy")
        frontier_set = set(frontier_idx)

        # Non-frontier scatter (faded)
        nf_xs = [data[i][x_key] for i in range(len(data)) if i not in frontier_set]
        nf_ys = [data[i]["accuracy"] for i in range(len(data)) if i not in frontier_set]
        ax.scatter(nf_xs, nf_ys, s=30, zorder=2, color=color, alpha=0.2,
                   edgecolors="white", linewidths=0.3, marker=marker)

        # Frontier scatter + line
        fx = [data[i][x_key] for i in frontier_idx]
        fy = [data[i]["accuracy"] for i in frontier_idx]
        ax.plot(fx, fy, color=color, linewidth=2, linestyle="--", alpha=0.7, zorder=3)
        ax.scatter(fx, fy, s=80, zorder=4, color=color, edgecolors="white",
                   linewidths=0.5, marker=marker, label=f"{label} Frontier")

        # Labels for frontier points
        for i in frontier_idx:
            t = ax.text(
                data[i][x_key], data[i]["accuracy"], data[i]["setup"],
                fontsize=6.5, fontweight="bold", color=color, ha="center", va="center",
            )
            all_texts.append(t)
            all_xs.append(data[i][x_key])
            all_ys.append(data[i]["accuracy"])

    adjust_text(
        all_texts, x=all_xs, y=all_ys, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4, alpha=0.5),
        force_points=(2.0, 2.0), force_text=(1.5, 1.5), expand=(2.0, 2.5),
        iterations=300,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 101)


def main():
    data_1t = load_one_turn()
    data_2t = load_two_turn()
    data_3t = load_three_turn()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    panels = [
        ("p50", "P50 Latency (s)", "pareto_p50_1t_2t_3t.png"),
        ("p90", "P90 Latency (s)", "pareto_p90_1t_2t_3t.png"),
        ("mean", "Mean Latency (s)", "pareto_mean_1t_2t_3t.png"),
    ]
    for x_key, xlabel, filename in panels:
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_three_frontiers(ax, data_1t, data_2t, data_3t, x_key, xlabel,
                             f"HumanEval: {xlabel} vs Accuracy — 1T / 2T / 3T Pareto Frontiers")
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
