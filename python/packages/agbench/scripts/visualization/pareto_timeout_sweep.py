#!/usr/bin/env python3
"""5-panel Pareto frontier (mean latency vs accuracy) for timeout cutoffs 20/40/60/80/100s.

For each timeout T: any rep with elapsed_time > T is reclassified as fail.
Mean latency = mean of min(elapsed_time, T) over ALL reps (fail + pass).
Pareto frontier is recomputed per timeout using 1T/2T/3T/4T raw result.json files.
"""
from __future__ import annotations

import glob
import json
import os
import statistics

import matplotlib.pyplot as plt

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", "pareto",
    "pareto_timeout_sweep.png",
)
TIMEOUTS = [20, 60, 100, 200, float("inf")]

TIER_PREFIXES = {
    "1T": "one-turn-",
    "2T": "two-turn-nc-",
    "3T": ("three-turn-nc-", "three-turn-"),
    "4T": "four-turn-nc-",
}


def tier_of(name: str) -> str | None:
    for tier, prefix in TIER_PREFIXES.items():
        if isinstance(prefix, tuple):
            if any(name.startswith(p) for p in prefix):
                return tier
        elif name.startswith(prefix):
            return tier
    return None


def short_name(name: str, tier: str) -> str:
    prefix = TIER_PREFIXES[tier]
    if isinstance(prefix, tuple):
        for p in prefix:
            if name.startswith(p):
                return name[len(p):]
        return name
    return name[len(prefix):]


def load_all_reps() -> dict[str, dict]:
    """Return {setup_name: {"tier": ..., "reps": [(elapsed_time, success), ...]}}"""
    configs: dict[str, dict] = {}
    run_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*")))
    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        tier = tier_of(name)
        if tier is None:
            continue
        result_files = glob.glob(os.path.join(run_dir, "Results", "*", "result.json"))
        if not result_files:
            continue
        # Skip backup/duplicate runs
        if name.endswith("-backup") or name.endswith("-dup"):
            continue
        with open(result_files[0]) as f:
            data = json.load(f)
        reps = []
        for task in data["tasks"].values():
            for rep in task["repetitions"].values():
                if rep["status"] == "completed":
                    reps.append((rep["elapsed_time"], rep.get("success", False)))
                else:
                    # non-completed rep → always a failure, infinite elapsed time
                    reps.append((float("inf"), False))
        if reps:
            configs[name] = {"tier": tier, "short": short_name(name, tier), "reps": reps}
    return configs


def compute_timeout_stats(reps: list, timeout: float) -> dict:
    total = len(reps)
    passed = sum(1 for t, s in reps if s and t <= timeout)
    mean_lat = statistics.mean(min(t, timeout) for t, _ in reps)
    return {"accuracy": passed / total * 100, "mean": mean_lat}


def pareto_frontier(points: list[dict], x_key: str, y_key: str) -> list[int]:
    sorted_idx = sorted(range(len(points)), key=lambda i: points[i][x_key])
    frontier, best_y = [], -float("inf")
    for i in sorted_idx:
        if points[i][y_key] > best_y:
            frontier.append(i)
            best_y = points[i][y_key]
    return frontier


TIMEOUT_COLORS = {
    20:           "#e41a1c",
    40:           "#ff7f00",
    60:           "#4daf4a",
    80:           "#377eb8",
    100:          "#984ea3",
    150:          "#a65628",
    200:          "#f781bf",
    float("inf"): "#000000",
}
TIMEOUT_LABELS = {
    20:           "Timeout ≤ 20s",
    40:           "Timeout ≤ 40s",
    60:           "Timeout ≤ 60s",
    80:           "Timeout ≤ 80s",
    100:          "Timeout ≤ 100s",
    150:          "Timeout ≤ 150s",
    200:          "Timeout ≤ 200s",
    float("inf"): "No timeout",
}


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    configs = load_all_reps()
    print(f"Loaded {len(configs)} configs")

    fig, ax = plt.subplots(figsize=(12, 8))

    for timeout in TIMEOUTS:
        color = TIMEOUT_COLORS[timeout]
        points = []
        for cfg in configs.values():
            stats = compute_timeout_stats(cfg["reps"], timeout)
            points.append({"mean": stats["mean"], "accuracy": stats["accuracy"]})

        fidx = pareto_frontier(points, "mean", "accuracy")
        f_pts = [points[i] for i in fidx if points[i]["accuracy"] >= 80]

        ax.plot([p["mean"] for p in f_pts], [p["accuracy"] for p in f_pts],
                color=color, linewidth=2.2, alpha=0.85,
                label=TIMEOUT_LABELS[timeout], zorder=4)
        ax.scatter([p["mean"] for p in f_pts], [p["accuracy"] for p in f_pts],
                   s=55, color=color, edgecolors="white", linewidths=0.6, zorder=5)

    ax.set_xlabel("Mean Latency (s)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        "Pareto Frontiers under Different Timeout Cutoffs\n"
        "(reps exceeding timeout → fail; latency capped at timeout)",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 15)
    ax.set_ylim(85, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
