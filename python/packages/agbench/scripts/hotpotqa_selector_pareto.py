"""Latency vs accuracy Pareto plot for HotpotQA SelectorGroupChat runs.

Input:  reports/csv/hotpotqa_selector_breakdown.csv
Output: reports/figures/hotpotqa_selector_pareto.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/xz957/autogen/python/packages/agbench")
CSV_IN = ROOT / "reports/csv/hotpotqa_selector_breakdown.csv"
FIG_OUT = ROOT / "reports/figures/hotpotqa_selector_pareto.png"


def pareto_front(points):
    """Max accuracy, min latency. Return sorted frontier points."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front = []
    best_acc = -1.0
    for lat, acc, label in pts:
        if acc > best_acc:
            front.append((lat, acc, label))
            best_acc = acc
    return front


def main():
    rows = list(csv.DictReader(CSV_IN.open()))
    points = [(float(r["avg_e2e_s"]), float(r["success_rate"]), r["run"]) for r in rows]

    front = pareto_front(points)
    front_set = {p[2] for p in front}

    colors = {"nt": "#1f77b4", "t": "#d62728"}
    markers = {"1.7b": "o", "4b": "s", "8b": "^", "14b": "D"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for lat, acc, label in points:
        size, mode = label.rsplit("-", 1)
        on_front = label in front_set
        ax.scatter(
            lat,
            acc,
            s=160 if on_front else 100,
            c=colors[mode],
            marker=markers[size],
            edgecolors="black" if on_front else "gray",
            linewidths=2 if on_front else 0.8,
            zorder=3,
            label=None,
        )
        dx, dy = 6, 0.005
        ax.annotate(
            label,
            (lat, acc),
            xytext=(lat + dx, acc + dy),
            fontsize=9,
            fontweight="bold" if on_front else "normal",
        )

    # frontier line
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax.plot(fx, fy, "--", color="black", lw=1.5, alpha=0.6, zorder=2, label="Pareto frontier")

    # Legend: mode + size
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["nt"], markersize=10, label="non-thinking"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["t"], markersize=10, label="thinking"),
    ]
    for size, mk in markers.items():
        legend_handles.append(
            Line2D([0], [0], marker=mk, color="w", markerfacecolor="gray", markersize=10, label=size)
        )
    legend_handles.append(Line2D([0], [0], linestyle="--", color="black", lw=1.5, label="Pareto frontier"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Avg end-to-end latency per rep (s)")
    ax.set_ylabel("Accuracy (success rate)")
    ax.set_title("HotpotQA SelectorGroupChat — accuracy vs latency\n(100 questions × 3 reps, Qwen3 {1.7b,4b,8b,14b} × {thinking, non-thinking})")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.15, 0.43)
    ax.set_xlim(0, max(p[0] for p in points) * 1.1)

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=130)
    print(f"Wrote {FIG_OUT}")
    print("Pareto frontier:")
    for lat, acc, label in front:
        print(f"  {label:<8} lat={lat:6.1f}s  acc={acc:.3f}")


if __name__ == "__main__":
    main()
