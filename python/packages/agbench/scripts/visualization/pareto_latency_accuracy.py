#!/usr/bin/env python3
"""Plot Pareto frontiers of latency vs accuracy for two-turn-nc setups."""

import csv
import os

import matplotlib.pyplot as plt
from adjustText import adjust_text

INPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "csv", "two_turn_nc_stats.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "pareto")

XLIM_MAX = 60  # Drop outliers beyond this
YLIM_LO = 80


def load_data():
    setups = []
    with open(INPUT_CSV) as f:
        for row in csv.DictReader(f):
            setups.append({
                "setup": row["setup"].replace("two-turn-nc-", ""),
                "accuracy": float(row["accuracy"]) * 100,
                "mean": float(row["mean_latency"]),
                "p50": float(row["p50_latency"]),
                "p90": float(row["p90_latency"]),
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


def plot_panel(ax, setups, x_key, xlabel):
    frontier_idx = pareto_frontier(setups, x_key, "accuracy")
    frontier_set = set(frontier_idx)

    # Filter to in-bounds points
    def in_bounds(i):
        return setups[i][x_key] <= XLIM_MAX and setups[i]["accuracy"] >= YLIM_LO

    # Non-frontier scatter (gray)
    nf = [i for i in range(len(setups)) if i not in frontier_set and in_bounds(i)]
    ax.scatter([setups[i][x_key] for i in nf], [setups[i]["accuracy"] for i in nf],
               s=40, zorder=2, color="#cccccc", edgecolors="white", linewidths=0.5)

    # Frontier scatter + line (in bounds only)
    f_in = [i for i in frontier_idx if in_bounds(i)]
    fx = [setups[i][x_key] for i in f_in]
    fy = [setups[i]["accuracy"] for i in f_in]
    ax.plot(fx, fy, color="#E24A33", linewidth=2, linestyle="--", alpha=0.7, zorder=3)
    ax.scatter(fx, fy, s=80, zorder=4, color="#E24A33", edgecolors="white", linewidths=0.5)

    # Labels
    texts, txs, tys = [], [], []
    for i in range(len(setups)):
        if not in_bounds(i):
            continue
        x, y = setups[i][x_key], setups[i]["accuracy"]
        if i in frontier_set:
            texts.append(ax.text(x, y, setups[i]["setup"],
                                 fontsize=7.5, fontweight="bold", color="#E24A33", ha="center", va="center"))
        else:
            texts.append(ax.text(x, y, setups[i]["setup"],
                                 fontsize=6.5, fontweight="normal", color="#bbbbbb", ha="center", va="center"))
        txs.append(x)
        tys.append(y)

    adjust_text(texts, x=txs, y=tys, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                force_points=(1.5, 1.5), force_text=(1.0, 1.0), expand=(1.5, 1.8))
    for t in texts:
        t.set_clip_on(True)

    # Auto-fit xlim to data with padding
    all_x_in = [setups[i][x_key] for i in range(len(setups)) if in_bounds(i)]
    xmax = max(all_x_in) * 1.1 if all_x_in else XLIM_MAX
    ax.set_xlim(0, xmax)
    ax.set_ylim(YLIM_LO, 100)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_combined(setups, output_dir):
    """Plot all three Pareto frontiers (p50, p90, mean) on a single axis."""
    fig, ax = plt.subplots(figsize=(14, 9))

    metrics = [
        ("p50", "P50", "#2ca02c", "o"),
        ("p90", "P90", "#E24A33", "s"),
        ("mean", "Mean", "#4878CF", "D"),
    ]

    all_texts, all_xs, all_ys = [], [], []

    for x_key, label, color, marker in metrics:
        frontier_idx = pareto_frontier(setups, x_key, "accuracy")

        # Filter to in-bounds
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
    ax.set_title("Two-Turn No-Context: Pareto Frontiers (P50 / P90 / Mean)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, XLIM_MAX)
    # Auto-fit ylim to data with padding
    if all_ys:
        ymin = min(all_ys) - 2
    else:
        ymin = YLIM_LO
    ax.set_ylim(ymin, 100)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "pareto_combined.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main():
    setups = load_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Three-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    panels = [
        (axes[0], "p50", "P50 Latency (s)"),
        (axes[1], "p90", "P90 Latency (s)"),
        (axes[2], "mean", "Mean Latency (s)"),
    ]

    for ax, x_key, xlabel in panels:
        plot_panel(ax, setups, x_key, xlabel)

    fig.suptitle("Two-Turn No-Context: Latency vs Accuracy (Pareto Frontier)", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "pareto_latency_accuracy.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # Combined single-plot
    plot_combined(setups, OUTPUT_DIR)


if __name__ == "__main__":
    main()
