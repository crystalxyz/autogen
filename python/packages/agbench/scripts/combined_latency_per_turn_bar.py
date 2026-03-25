# Note: run it from the agbench/ folder, change model log path if needed

#!/usr/bin/env python3
"""Generate a combined bar chart showing per-turn coder runtime and
repetition percentage across multiple model sizes (HumanEval)."""

import os
import re
import json
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# No output token limit
# models = {
#     "0.6b": "output_0.6b/human_eval/result.json",
#     "1.7b": "output_1.7b/human_eval/result.json",
#     "4b":   "output_4b/human_eval/result.json",
#     "8b":   "output_8b/human_eval/result.json",
# }

# le1000 output token limit
# models = {
#     "0.6b_le1000": "output_0.6b_1000output/Results/human_eval/result.json",
#     "1.7b_le1000": "output_1.7b_1000output/Results/human_eval/result.json",
#     "4b_le1000":   "output_4b_1000output/Results/human_eval/result.json",
#     "8b_le1000":   "output_8b_1000output/Results/human_eval/result.json",
# }

# eq1000 output token limit
models = {
    "0.6b_eq1000": "output_0.6b_eq1000output/Results/human_eval/result.json",
    "1.7b_eq1000": "output_1.7b_eq1000output/Results/human_eval/result.json",
    "4b_eq1000":   "output_4b_eq1000output/Results/human_eval/result.json",
    "8b_eq1000":   "output_8b_eq1000output/Results/human_eval/result.json",
}

# # All 12 runs
# models = {
#     "0.6b":        "output_0.6b/human_eval/result.json",
#     "0.6b_le1000": "output_0.6b_1000output/Results/human_eval/result.json",
#     "0.6b_eq1000": "output_0.6b_eq1000output/Results/human_eval/result.json",
#     "1.7b":        "output_1.7b/human_eval/result.json",
#     "1.7b_le1000": "output_1.7b_1000output/Results/human_eval/result.json",
#     "1.7b_eq1000": "output_1.7b_eq1000output/Results/human_eval/result.json",
#     "4b":          "output_4b/human_eval/result.json",
#     "4b_le1000":   "output_4b_1000output/Results/human_eval/result.json",
#     "4b_eq1000":   "output_4b_eq1000output/Results/human_eval/result.json",
#     "8b":          "output_8b/human_eval/result.json",
#     "8b_le1000":   "output_8b_1000output/Results/human_eval/result.json",
#     "8b_eq1000":   "output_8b_eq1000output/Results/human_eval/result.json",
# }

runtime_pattern = re.compile(r"\[runtime\]\s+name=(\S+)\s+seconds=([\d.]+)")
all_data = {}
total_nonzero = {}  # total reps with turns > 0 per model

for label, path in models.items():
    with open(path) as f:
        raw = f.read()
    raw = re.sub(r'(\d+\.\d+),(\d+)\s*\n', r'\1\n', raw)
    data = json.loads(raw)
    result_dir = os.path.dirname(os.path.abspath(path))

    scenario_dir = None
    for entry in os.listdir(result_dir):
        full = os.path.join(result_dir, entry)
        if os.path.isdir(full) and entry != "__pycache__":
            scenario_dir = full
            break

    steps_by_number = defaultdict(list)
    nonzero_count = 0
    for task_id, task in sorted(data["tasks"].items()):
        for rep_id, rep in task["repetitions"].items():
            if not rep.get("success", False):
                continue
            t = rep.get("turns")
            if not t or t == 0:
                continue
            nonzero_count += 1
            if scenario_dir:
                log_path = os.path.join(scenario_dir, task_id, str(rep_id), "console_log.txt")
                if os.path.isfile(log_path):
                    step_num = 0
                    with open(log_path) as lf:
                        for line in lf:
                            m = runtime_pattern.search(line)
                            if m and m.group(1) in ("coder", "executor"):
                                step_num += 1
                                steps_by_number[step_num].append(float(m.group(2)))
    all_data[label] = steps_by_number
    total_nonzero[label] = nonzero_count

all_steps = sorted(set().union(*(d.keys() for d in all_data.values())))
step_labels = [f"Coder {(s + 1) // 2}" if s % 2 == 1 else f"Executor {s // 2}" for s in all_steps]
model_labels = list(models.keys())
colors = plt.cm.tab20(np.linspace(0, 1, len(model_labels)))

fig, (ax_pct, ax_time) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

n_models = len(model_labels)
x = np.arange(len(all_steps))
bar_width = 0.8 / n_models

for i, (model, color) in enumerate(zip(model_labels, colors)):
    counts = []
    pcts = []
    means = []
    for s in all_steps:
        times = all_data[model].get(s, [])
        c = len(times)
        counts.append(c)
        pcts.append(100.0 * c / total_nonzero[model] if total_nonzero[model] > 0 else 0)
        means.append(np.mean(times) if times else 0)

    offset = (i - n_models / 2 + 0.5) * bar_width

    # Percentage subplot
    bars_p = ax_pct.bar(x + offset, pcts, bar_width, color=color, alpha=0.75,
                        edgecolor="black", linewidth=0.6, label=f"{model} (n={total_nonzero[model]})")
    for bar, val in zip(bars_p, pcts):
        if val > 0:
            ax_pct.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Avg time subplot
    bars_t = ax_time.bar(x + offset, means, bar_width, color=color, alpha=0.75,
                         edgecolor="black", linewidth=0.6, label=model)
    for bar, val in zip(bars_t, means):
        if val > 0:
            ax_time.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax_pct.set_ylabel("% of Successful Repetitions (turns > 0)", fontsize=12)
ax_pct.set_title("Proportion of Successful Tasks Reaching Each Agent Step", fontsize=13)
ax_pct.legend(fontsize=9)
ax_pct.grid(axis="y", alpha=0.25)

ax_time.set_ylabel("Avg Runtime (seconds)", fontsize=12)
ax_time.set_xlabel("Agent step (max 12)", fontsize=12)
ax_time.set_title("Average Runtime per Agent Step", fontsize=13)
ax_time.set_xticks(x)
ax_time.set_xticklabels(step_labels, fontsize=11)
ax_time.legend(fontsize=10)
ax_time.grid(axis="y", alpha=0.25)

fig.suptitle("Per-Step Agent Runtime & Count by Model Size – Successful Only (HumanEval)", fontsize=15, y=1.01)
fig.tight_layout()
out = "reports/combined_per_turn_bar_all.png"
os.makedirs("reports", exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
