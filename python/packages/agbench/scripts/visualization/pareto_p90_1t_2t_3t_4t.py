#!/usr/bin/env python3
"""Plot P90/P50/mean latency vs accuracy with four per-tier Pareto frontiers (1T/2T/3T/4T).

Also computes a combined Pareto frontier across all tiers to check if 4T contributes new points.
"""

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


EXCLUDE_SETUPS = {"0.6b-nothink"}


def load_one_turn():
    setups = []
    with open(ONE_TURN_CSV) as f:
        for row in csv.DictReader(f):
            name = row["setup"].replace("one-turn-", "")
            if name in EXCLUDE_SETUPS:
                continue
            setups.append({
                "setup": name,
                "accuracy": float(row["success_rate"]) * 100,
                "p50": float(row["latency_p50"]),
                "p90": float(row["latency_p90"]),
                "mean": float(row["latency_mean"]),
                "tier": "1T",
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
                "tier": "2T",
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
                "tier": "3T",
            })
    return setups


def load_four_turn():
    setups = []
    with open(FOUR_TURN_CSV) as f:
        for row in csv.DictReader(f):
            setups.append({
                "setup": row["setup"].replace("four-turn-nc-", ""),
                "accuracy": float(row["accuracy"]) * 100,
                "p50": float(row["p50_latency"]),
                "p90": float(row["p90_latency"]),
                "mean": float(row["mean_latency"]),
                "tier": "4T",
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


def plot_frontiers(ax, tier_data_list, x_key, xlabel, title):
    """Plot per-tier and combined Pareto frontiers."""
    series = [
        (tier_data_list[0], "1-Turn", "#2ca02c", "o"),
        (tier_data_list[1], "2-Turn", "#4878CF", "s"),
        (tier_data_list[2], "3-Turn", "#E24A33", "D"),
        (tier_data_list[3], "4-Turn", "#9467bd", "^"),
    ]

    all_texts = []
    all_xs = []
    all_ys = []

    # Compute combined frontier
    combined = []
    for data, label, _, _ in series:
        for i, pt in enumerate(data):
            combined.append({**pt, "_src_label": label})
    combined_frontier_idx = set(pareto_frontier(combined, x_key, "accuracy"))
    combined_frontier_tiers = set()
    for i in combined_frontier_idx:
        combined_frontier_tiers.add((combined[i]["setup"], combined[i]["tier"]))

    for data, label, color, marker in series:
        frontier_idx = pareto_frontier(data, x_key, "accuracy")
        frontier_set = set(frontier_idx)

        # Check which frontier points are also on combined frontier
        on_combined = set()
        for i in frontier_idx:
            if (data[i]["setup"], data[i]["tier"]) in combined_frontier_tiers:
                on_combined.add(i)

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

        # Labels for frontier points (bold+star if on combined frontier)
        for i in frontier_idx:
            suffix = " *" if i in on_combined else ""
            t = ax.text(
                data[i][x_key], data[i]["accuracy"], data[i]["setup"] + suffix,
                fontsize=6.5, fontweight="bold", color=color, ha="center", va="center",
            )
            all_texts.append(t)
            all_xs.append(data[i][x_key])
            all_ys.append(data[i]["accuracy"])

    # Draw combined frontier as a gray step line
    combined_pts = [combined[i] for i in sorted(combined_frontier_idx, key=lambda i: combined[i][x_key])]
    if combined_pts:
        cx = [p[x_key] for p in combined_pts]
        cy = [p["accuracy"] for p in combined_pts]
        ax.plot(cx, cy, color="gray", linewidth=2.5, linestyle="-", alpha=0.4, zorder=1,
                label="Combined Frontier")

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
    ax.set_ylim(60, 101)


def print_frontier_analysis(data_1t, data_2t, data_3t, data_4t, x_key):
    """Print which 4T points land on the combined Pareto frontier."""
    combined = []
    for tier_label, data in [("1T", data_1t), ("2T", data_2t), ("3T", data_3t), ("4T", data_4t)]:
        for pt in data:
            combined.append({**pt, "_tier": tier_label})

    frontier_idx = pareto_frontier(combined, x_key, "accuracy")
    print(f"\n--- Combined Pareto frontier ({x_key}) ---")
    for i in frontier_idx:
        p = combined[i]
        print(f"  [{p['_tier']}] {p['setup']:<45} acc={p['accuracy']:.1f}%  {x_key}={p[x_key]:.2f}s")

    four_t_on_frontier = [i for i in frontier_idx if combined[i]["_tier"] == "4T"]
    if four_t_on_frontier:
        print(f"\n  => 4T contributes {len(four_t_on_frontier)} point(s) to the combined frontier!")
    else:
        print(f"\n  => 4T does NOT improve the combined Pareto frontier for {x_key}.")


def main():
    data_1t = load_one_turn()
    data_2t = load_two_turn()
    data_3t = load_three_turn()
    data_4t = load_four_turn()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    panels = [
        ("p50", "P50 Latency (s)", "pareto_p50_1t_2t_3t_4t.png"),
        ("p90", "P90 Latency (s)", "pareto_p90_1t_2t_3t_4t.png"),
        ("mean", "Mean Latency (s)", "pareto_mean_1t_2t_3t_4t.png"),
    ]
    for x_key, xlabel, filename in panels:
        print_frontier_analysis(data_1t, data_2t, data_3t, data_4t, x_key)

        fig, ax = plt.subplots(figsize=(14, 9))
        plot_frontiers(ax, [data_1t, data_2t, data_3t, data_4t], x_key, xlabel,
                       f"HumanEval: {xlabel} vs Accuracy — 1T / 2T / 3T / 4T Pareto Frontiers")
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
