#!/usr/bin/env python3
"""Overlay reflection configs onto the 1T/2T/3T/4T Pareto frontier.

Loads:
  reports/csv/one_turn_baselines.csv
  reports/csv/two_turn_nc_stats.csv
  reports/csv/three_turn_nc_stats.csv
  reports/csv/four_turn_nc_stats.csv
  reports/csv/reflection_pareto_data.csv

Produces Pareto plots (p50 / p90 / mean) with reflection as a separate tier,
and prints a summary of frontier composition so we can see whether reflection
contributes points to the Pareto frontier.
"""
from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
from adjustText import adjust_text

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
CSV_DIR = os.path.join(REPORTS_DIR, "csv")
ONE_TURN_CSV = os.path.join(CSV_DIR, "one_turn_baselines.csv")
TWO_TURN_CSV = os.path.join(CSV_DIR, "two_turn_nc_stats.csv")
THREE_TURN_CSV = os.path.join(CSV_DIR, "three_turn_nc_stats.csv")
FOUR_TURN_CSV = os.path.join(CSV_DIR, "four_turn_nc_stats.csv")
REFLECTION_CSV = os.path.join(CSV_DIR, "reflection_pareto_data.csv")
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
    with open(REFLECTION_CSV) as f:
        for row in csv.DictReader(f):
            points.append({
                "setup": row["config"],
                "tier": "Reflect",
                "accuracy": float(row["accuracy_pct"]),
                "p50": float(row["latency_p50_s"]),
                "p90": float(row["latency_p90_s"]),
                "mean": float(row["latency_mean_s"]),
            })
    return points


def pareto_frontier(points, x_key, y_key):
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
    "Reflect": ("#ff7f0e", "*"),
}


def plot_single(points, x_key, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    frontier_idx = pareto_frontier(points, x_key, "accuracy")
    frontier_set = set(frontier_idx)

    def in_bounds(i):
        return points[i][x_key] <= XLIM_MAX

    for tier, (color, marker) in TIER_STYLE.items():
        xs = [points[i][x_key] for i in range(len(points))
              if i not in frontier_set and points[i]["tier"] == tier and in_bounds(i)]
        ys = [points[i]["accuracy"] for i in range(len(points))
              if i not in frontier_set and points[i]["tier"] == tier and in_bounds(i)]
        alpha = 0.35 if tier == "Reflect" else 0.15
        size = 45 if tier == "Reflect" else 25
        ax.scatter(xs, ys, s=size, color=color, alpha=alpha, edgecolors="white",
                   linewidths=0.3, marker=marker, zorder=2)

    f_in = [i for i in frontier_idx if in_bounds(i)]
    fx = [points[i][x_key] for i in f_in]
    fy = [points[i]["accuracy"] for i in f_in]
    ax.plot(fx, fy, color="#333333", linewidth=2, linestyle="--", alpha=0.5, zorder=3)

    for tier, (color, marker) in TIER_STYLE.items():
        idxs = [i for i in f_in if points[i]["tier"] == tier]
        if idxs:
            ax.scatter(
                [points[i][x_key] for i in idxs],
                [points[i]["accuracy"] for i in idxs],
                s=120 if tier == "Reflect" else 80,
                zorder=5, color=color, edgecolors="white", linewidths=0.5,
                marker=marker, label=tier,
            )

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


def summarize_frontier(points, x_key, label):
    fidx = pareto_frontier(points, x_key, "accuracy")
    print(f"\n=== Pareto frontier ({label}, no latency cap) ===")
    print(f"{'tier':<8} {'setup':<35} {x_key:>10}  {'acc%':>6}")
    for i in fidx:
        p = points[i]
        print(f"{p['tier']:<8} {p['setup']:<35} {p[x_key]:>10.2f}  {p['accuracy']:>6.2f}")
    reflect_count = sum(1 for i in fidx if points[i]["tier"] == "Reflect")
    print(f"Reflection points on frontier: {reflect_count}/{len(fidx)}")


def main():
    points = load_all()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    panels = [
        ("p50", "P50 Latency (s)", "Pareto Frontier with Reflection — P50 Latency", "pareto_with_reflection_p50.png"),
        ("p90", "P90 Latency (s)", "Pareto Frontier with Reflection — P90 Latency", "pareto_with_reflection_p90.png"),
        ("mean", "Mean Latency (s)", "Pareto Frontier with Reflection — Mean Latency", "pareto_with_reflection_mean.png"),
    ]
    for x_key, xlabel, title, filename in panels:
        plot_single(points, x_key, xlabel, title, os.path.join(OUTPUT_DIR, filename))

    for x_key, xlabel, _, _ in panels:
        summarize_frontier(points, x_key, xlabel)


if __name__ == "__main__":
    main()
