#!/usr/bin/env python3
"""Plot overall Pareto frontiers combining 1T/2T/3T/4T data."""

import csv
import os

import matplotlib.pyplot as plt
from adjustText import adjust_text

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
ONE_TURN_CSV = os.path.join(REPORTS_DIR, "csv", "one_turn_baselines.csv")
TWO_TURN_CSV = os.path.join(REPORTS_DIR, "csv", "two_turn_nc_stats.csv")
THREE_TURN_CSV = os.path.join(REPORTS_DIR, "csv", "three_turn_nc_stats.csv")
FOUR_TURN_CSV = os.path.join(REPORTS_DIR, "csv", "four_turn_nc_stats.csv")
OUTPUT_DIR = os.path.join(REPORTS_DIR, "pareto")

XLIM_MAX = 40


def load_all():
    points = []
    with open(ONE_TURN_CSV) as f:
        for row in csv.DictReader(f):
            setup = row["setup"].replace("one-turn-", "")
            if setup == "0.6b-nothink":
                continue
            points.append({
                "setup": setup,
                "tier": "1T",
                "accuracy": float(row["success_rate"]) * 100,
                "p50": float(row["latency_p50"]),
                "p90": float(row["latency_p90"]),
                "mean": float(row["latency_mean"]),
            })
    with open(TWO_TURN_CSV) as f:
        for row in csv.DictReader(f):
            points.append({
                "setup": row["setup"].replace("two-turn-nc-", ""),
                "tier": "2T",
                "accuracy": float(row["accuracy"]) * 100,
                "p50": float(row["p50_latency"]),
                "p90": float(row["p90_latency"]),
                "mean": float(row["mean_latency"]),
            })
    with open(THREE_TURN_CSV) as f:
        for row in csv.DictReader(f):
            points.append({
                "setup": row["setup"].replace("three-turn-nc-", "").replace("three-turn-", ""),
                "tier": "3T",
                "accuracy": float(row["accuracy"]) * 100,
                "p50": float(row["p50_latency"]),
                "p90": float(row["p90_latency"]),
                "mean": float(row["mean_latency"]),
            })
    with open(FOUR_TURN_CSV) as f:
        for row in csv.DictReader(f):
            points.append({
                "setup": row["setup"].replace("four-turn-nc-", ""),
                "tier": "4T",
                "accuracy": float(row["accuracy"]) * 100,
                "p50": float(row["p50_latency"]),
                "p90": float(row["p90_latency"]),
                "mean": float(row["mean_latency"]),
            })
    return points


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


TIER_STYLE = {
    "1T": ("#2ca02c", "o"),
    "2T": ("#4878CF", "s"),
    "3T": ("#E24A33", "D"),
    "4T": ("#9467bd", "^"),
}


def plot_single(points, x_key, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    frontier_idx = pareto_frontier(points, x_key, "accuracy")
    frontier_set = set(frontier_idx)

    def in_bounds(i):
        return points[i][x_key] <= XLIM_MAX

    # Non-frontier scatter, colored by tier but faded
    for tier, (color, marker) in TIER_STYLE.items():
        xs = [points[i][x_key] for i in range(len(points))
              if i not in frontier_set and points[i]["tier"] == tier and in_bounds(i)]
        ys = [points[i]["accuracy"] for i in range(len(points))
              if i not in frontier_set and points[i]["tier"] == tier and in_bounds(i)]
        ax.scatter(xs, ys, s=25, color=color, alpha=0.15, edgecolors="white",
                   linewidths=0.3, marker=marker, zorder=2)

    # Frontier line (in bounds)
    f_in = [i for i in frontier_idx if in_bounds(i)]
    fx = [points[i][x_key] for i in f_in]
    fy = [points[i]["accuracy"] for i in f_in]
    ax.plot(fx, fy, color="#333333", linewidth=2, linestyle="--", alpha=0.5, zorder=3)

    # Frontier scatter, colored by tier
    for tier, (color, marker) in TIER_STYLE.items():
        idxs = [i for i in f_in if points[i]["tier"] == tier]
        if idxs:
            ax.scatter(
                [points[i][x_key] for i in idxs],
                [points[i]["accuracy"] for i in idxs],
                s=80, zorder=5, color=color, edgecolors="white", linewidths=0.5,
                marker=marker, label=tier,
            )

    # Labels for frontier points
    texts, txs, tys = [], [], []
    for i in f_in:
        color = TIER_STYLE[points[i]["tier"]][0]
        t = ax.text(
            points[i][x_key], points[i]["accuracy"], points[i]["setup"],
            fontsize=8, fontweight="bold", color=color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, linewidth=0.6, alpha=0.9),
            zorder=10,
        )
        texts.append(t)
        txs.append(points[i][x_key])
        tys.append(points[i]["accuracy"])

    adjust_text(
        texts, x=txs, y=tys, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4, alpha=0.5),
        force_points=(2.0, 2.0), force_text=(1.5, 1.5), expand=(2.0, 2.5),
        iterations=300,
    )
    for t in texts:
        t.set_clip_on(True)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, XLIM_MAX)
    ax.set_ylim(60, 100)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


def main():
    points = load_all()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    panels = [
        ("p50", "P50 Latency (s)", "Overall Pareto Frontier — P50 Latency (1T / 2T / 3T / 4T)", "pareto_overall_p50.png"),
        ("p90", "P90 Latency (s)", "Overall Pareto Frontier — P90 Latency (1T / 2T / 3T / 4T)", "pareto_overall_p90.png"),
        ("mean", "Mean Latency (s)", "Overall Pareto Frontier — Mean Latency (1T / 2T / 3T / 4T)", "pareto_overall_mean.png"),
    ]

    for x_key, xlabel, title, filename in panels:
        plot_single(points, x_key, xlabel, title, os.path.join(OUTPUT_DIR, filename))


if __name__ == "__main__":
    main()
