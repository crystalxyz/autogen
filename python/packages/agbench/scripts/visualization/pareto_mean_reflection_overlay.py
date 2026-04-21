#!/usr/bin/env python3
"""Mean-latency Pareto frontier (1T/2T/3T/4T) with reflection points overlaid in a distinct color.

Base frontier is computed over 1T/2T/3T/4T only (same logic as pareto_overall.py).
Reflection points are plotted on top in a highlight color so we can visually
see where they land relative to the non-reflection frontier.
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
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pareto_overall_mean_with_reflection_overlay.png")

XLIM_MIN = 1.5
XLIM_MAX = 50

TIER_STYLE = {
    "1T": ("#2ca02c", "o"),
    "2T": ("#4878CF", "s"),
    "3T": ("#E24A33", "D"),
    "4T": ("#9467bd", "^"),
}
REFLECT_COLOR = "#ff7f0e"
REFLECT_MARKER = "*"


def load_base():
    points = []
    with open(ONE_TURN_CSV) as f:
        for row in csv.DictReader(f):
            setup = row["setup"].replace("one-turn-", "")
            if setup == "0.6b-nothink":
                continue
            points.append({
                "setup": setup, "tier": "1T",
                "accuracy": float(row["success_rate"]) * 100,
                "mean": float(row["latency_mean"]),
            })
    with open(TWO_TURN_CSV) as f:
        for row in csv.DictReader(f):
            points.append({
                "setup": row["setup"].replace("two-turn-nc-", ""), "tier": "2T",
                "accuracy": float(row["accuracy"]) * 100,
                "mean": float(row["mean_latency"]),
            })
    with open(THREE_TURN_CSV) as f:
        for row in csv.DictReader(f):
            points.append({
                "setup": row["setup"].replace("three-turn-nc-", "").replace("three-turn-", ""), "tier": "3T",
                "accuracy": float(row["accuracy"]) * 100,
                "mean": float(row["mean_latency"]),
            })
    with open(FOUR_TURN_CSV) as f:
        for row in csv.DictReader(f):
            points.append({
                "setup": row["setup"].replace("four-turn-nc-", ""), "tier": "4T",
                "accuracy": float(row["accuracy"]) * 100,
                "mean": float(row["mean_latency"]),
            })
    return points


def load_reflection():
    refs = []
    with open(REFLECTION_CSV) as f:
        for row in csv.DictReader(f):
            refs.append({
                "setup": row["config"], "tier": "Reflect",
                "accuracy": float(row["accuracy_pct"]),
                "mean": float(row["latency_mean_s"]),
            })
    return refs


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
    base = load_base()
    refl = load_reflection()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    frontier_idx = pareto_frontier(base, "mean", "accuracy")
    frontier_set = set(frontier_idx)

    def in_bounds(x):
        return XLIM_MIN <= x <= XLIM_MAX

    # Non-frontier base points (faded)
    for tier, (color, marker) in TIER_STYLE.items():
        xs = [p["mean"] for i, p in enumerate(base)
              if i not in frontier_set and p["tier"] == tier and in_bounds(p["mean"])]
        ys = [p["accuracy"] for i, p in enumerate(base)
              if i not in frontier_set and p["tier"] == tier and in_bounds(p["mean"])]
        ax.scatter(xs, ys, s=25, color=color, alpha=0.15, edgecolors="white",
                   linewidths=0.3, marker=marker, zorder=2)

    # Frontier line
    f_in = [i for i in frontier_idx if in_bounds(base[i]["mean"])]
    fx = [base[i]["mean"] for i in f_in]
    fy = [base[i]["accuracy"] for i in f_in]
    ax.plot(fx, fy, color="#333333", linewidth=2, linestyle="--", alpha=0.5, zorder=3,
            label="Pareto frontier (1T/2T/3T/4T)")

    # Frontier scatter per tier
    for tier, (color, marker) in TIER_STYLE.items():
        idxs = [i for i in f_in if base[i]["tier"] == tier]
        if idxs:
            ax.scatter(
                [base[i]["mean"] for i in idxs],
                [base[i]["accuracy"] for i in idxs],
                s=80, zorder=5, color=color, edgecolors="white", linewidths=0.5,
                marker=marker, label=tier,
            )

    # Frontier text labels
    texts, txs, tys = [], [], []
    for i in f_in:
        color = TIER_STYLE[base[i]["tier"]][0]
        t = ax.text(
            base[i]["mean"], base[i]["accuracy"], base[i]["setup"],
            fontsize=8, fontweight="bold", color=color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, linewidth=0.6, alpha=0.9),
            zorder=10,
        )
        texts.append(t)
        txs.append(base[i]["mean"])
        tys.append(base[i]["accuracy"])

    # Reflection overlay — distinct color, larger marker
    refl_in = [r for r in refl if in_bounds(r["mean"])]
    rx = [r["mean"] for r in refl_in]
    ry = [r["accuracy"] for r in refl_in]
    ax.scatter(rx, ry, s=150, color=REFLECT_COLOR, marker=REFLECT_MARKER,
               edgecolors="black", linewidths=0.8, alpha=0.9, zorder=6,
               label="Reflection")

    for r in refl_in:
        t = ax.text(
            r["mean"], r["accuracy"], r["setup"],
            fontsize=7, fontweight="bold", color=REFLECT_COLOR, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=REFLECT_COLOR, linewidth=0.7, alpha=0.88),
            zorder=11,
        )
        texts.append(t)
        txs.append(r["mean"])
        tys.append(r["accuracy"])

    ax.set_xscale("log")
    ax.set_xlim(XLIM_MIN, XLIM_MAX)
    ax.set_ylim(40, 102)

    adjust_text(
        texts, x=txs, y=tys, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5, alpha=0.6),
        force_points=(3.0, 3.0), force_text=(2.0, 2.0), expand=(2.5, 3.0),
        only_move={"points": "xy", "text": "xy"},
        iterations=600,
    )
    for t in texts:
        t.set_clip_on(False)

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel("Mean Latency (s, log scale)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Overall Pareto Frontier — Mean Latency\n(1T/2T/3T/4T frontier + reflection overlay)",
                 fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.09)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
