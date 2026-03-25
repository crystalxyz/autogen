#!/usr/bin/env python3
"""Generate experiment report with diagrams from agbench output folders."""

import json
import re
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Collect data ──────────────────────────────────────────────────────────────

def load_result(path):
    with open(path) as f:
        content = f.read()
    # Fix known corruption in output_8b baseline
    content = content.replace('    "": 2.28,avg_turns_per_task', '    "avg_turns_per_task": 2.28,')
    content = re.sub(r',(\s*[}\]])', r'\\1', content)
    return json.loads(content)

def parse_folder_name(name):
    """Parse folder name into (model_size_str, token_budget, mode, budget_type)."""
    # Examples: output_0.6b, output_4b_1000output, output_8b_eq3000output, output_4b_1000output_nothink
    m = re.match(r'output_(\d+\.?\d*b)(?:_(eq)?(\d+)output)?(?:_(nothink))?$', name)
    if not m:
        return None
    model = m.group(1)
    budget_type = "eq" if m.group(2) else "max"
    tokens = int(m.group(3)) if m.group(3) else None
    mode = "nothink" if m.group(4) else "think"
    return model, tokens, mode, budget_type

MODEL_ORDER = ["0.6b", "1.7b", "4b", "8b"]
MODEL_SIZES_NUMERIC = {"0.6b": 0.6, "1.7b": 1.7, "4b": 4.0, "8b": 8.0}
TOKEN_BUDGETS = [1000, 3000, 5000]

records = []
raw_data = {}  # folder -> full parsed JSON (for per-task analysis)
for d in sorted(glob.glob(os.path.join(BASE_DIR, "output_*"))):
    folder = os.path.basename(d)
    parsed = parse_folder_name(folder)
    if not parsed:
        continue
    model, tokens, mode, budget_type = parsed
    rfiles = glob.glob(os.path.join(d, "**/result.json"), recursive=True)
    for rf in rfiles:
        try:
            data = load_result(rf)
        except Exception as e:
            print(f"WARN: failed to parse {rf}: {e}")
            continue
        s = data["stats"]
        raw_data[folder] = data
        records.append({
            "folder": folder,
            "model": model,
            "tokens": tokens,
            "mode": mode,
            "budget_type": budget_type,
            "success_rate": s["success_rate"],
            "avg_time": s["avg_time_per_task"],
            "avg_turns": s["avg_turns_per_task"],
            "total_tasks": s["total_tasks"],
        })

print(f"Loaded {len(records)} experiment results")

# ── Helper ────────────────────────────────────────────────────────────────────

COLORS = {"0.6b": "#e74c3c", "1.7b": "#e67e22", "4b": "#2ecc71", "8b": "#3498db"}
MARKERS = {"0.6b": "o", "1.7b": "s", "4b": "D", "8b": "^"}

def save(fig, name):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

# ── Figure 1: Accuracy vs Token Budget (think, max budget) ───────────────────

DEFAULT_X = 7000  # x-position for "unlimited" (default) on the right side

fig, ax = plt.subplots(figsize=(9, 5.5))
for model in MODEL_ORDER:
    xs, ys = [], []
    base = [r for r in records if r["model"] == model and r["tokens"] is None and r["mode"] == "think"]
    for tb in TOKEN_BUDGETS:
        pts = [r for r in records if r["model"] == model and r["tokens"] == tb
               and r["mode"] == "think" and r["budget_type"] == "max"]
        if pts:
            xs.append(tb)
            ys.append(pts[0]["success_rate"] * 100)
    if base:
        xs.append(DEFAULT_X)
        ys.append(base[0]["success_rate"] * 100)
        ax.axhline(y=base[0]["success_rate"] * 100, color=COLORS[model], alpha=0.15, linestyle="--", linewidth=0.8)
    ax.plot(xs, ys, marker=MARKERS[model], color=COLORS[model], label=model, linewidth=2, markersize=8)

ax.set_xlabel("Max Output Tokens", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy vs Max Output Tokens by Model Size\n(Think Mode)", fontsize=13, fontweight="bold")
ax.set_xticks(TOKEN_BUDGETS + [DEFAULT_X])
ax.set_xticklabels([str(t) for t in TOKEN_BUDGETS] + ["unlimited"])
ax.legend(title="Model Size", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(30, 100)
save(fig, "fig1_accuracy_vs_tokens.png")

# ── Figure 2: Latency vs Token Budget (think, max budget) ────────────────────

fig, ax = plt.subplots(figsize=(9, 5.5))
for model in MODEL_ORDER:
    xs, ys = [], []
    base = [r for r in records if r["model"] == model and r["tokens"] is None and r["mode"] == "think"]
    for tb in TOKEN_BUDGETS:
        pts = [r for r in records if r["model"] == model and r["tokens"] == tb
               and r["mode"] == "think" and r["budget_type"] == "max"]
        if pts:
            xs.append(tb)
            ys.append(pts[0]["avg_time"])
    if base:
        xs.append(DEFAULT_X)
        ys.append(base[0]["avg_time"])
    ax.plot(xs, ys, marker=MARKERS[model], color=COLORS[model], label=model, linewidth=2, markersize=8)

ax.set_xlabel("Max Output Tokens", fontsize=12)
ax.set_ylabel("Avg Time per Task (s)", fontsize=12)
ax.set_title("Latency vs Max Output Tokens by Model Size\n(Think Mode)", fontsize=13, fontweight="bold")
ax.set_xticks(TOKEN_BUDGETS + [DEFAULT_X])
ax.set_xticklabels([str(t) for t in TOKEN_BUDGETS] + ["unlimited"])
ax.legend(title="Model Size", fontsize=10)
ax.grid(True, alpha=0.3)
save(fig, "fig2_latency_vs_tokens.png")

# ── Figure 3: Think vs NoThink comparison ────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Accuracy
ax = axes[0]
nothink_tokens = [1000, 3000]
bar_width = 0.18
for i, model in enumerate(MODEL_ORDER):
    think_accs, nothink_accs = [], []
    for tb in nothink_tokens:
        t = [r for r in records if r["model"] == model and r["tokens"] == tb
             and r["mode"] == "think" and r["budget_type"] == "max"]
        nt = [r for r in records if r["model"] == model and r["tokens"] == tb
              and r["mode"] == "nothink" and r["budget_type"] == "max"]
        think_accs.append(t[0]["success_rate"] * 100 if t else 0)
        nothink_accs.append(nt[0]["success_rate"] * 100 if nt else 0)

    x = np.arange(len(nothink_tokens))
    offset = (i - 1.5) * bar_width
    bars_t = ax.bar(x + offset - bar_width/4, think_accs, bar_width * 0.45, color=COLORS[model], alpha=0.85, label=f"{model} think")
    bars_nt = ax.bar(x + offset + bar_width/4, nothink_accs, bar_width * 0.45, color=COLORS[model], alpha=0.4, label=f"{model} nothink", hatch="//")

ax.set_xlabel("Max Output Tokens", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Think vs NoThink: Accuracy", fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(len(nothink_tokens)))
ax.set_xticklabels([str(t) for t in nothink_tokens])
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(30, 100)

# Latency
ax = axes[1]
for i, model in enumerate(MODEL_ORDER):
    think_times, nothink_times = [], []
    for tb in nothink_tokens:
        t = [r for r in records if r["model"] == model and r["tokens"] == tb
             and r["mode"] == "think" and r["budget_type"] == "max"]
        nt = [r for r in records if r["model"] == model and r["tokens"] == tb
              and r["mode"] == "nothink" and r["budget_type"] == "max"]
        think_times.append(t[0]["avg_time"] if t else 0)
        nothink_times.append(nt[0]["avg_time"] if nt else 0)

    x = np.arange(len(nothink_tokens))
    offset = (i - 1.5) * bar_width
    ax.bar(x + offset - bar_width/4, think_times, bar_width * 0.45, color=COLORS[model], alpha=0.85, label=f"{model} think")
    ax.bar(x + offset + bar_width/4, nothink_times, bar_width * 0.45, color=COLORS[model], alpha=0.4, label=f"{model} nothink", hatch="//")

ax.set_xlabel("Max Output Tokens", fontsize=11)
ax.set_ylabel("Avg Time per Task (s)", fontsize=11)
ax.set_title("Think vs NoThink: Latency", fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(len(nothink_tokens)))
ax.set_xticklabels([str(t) for t in nothink_tokens])
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save(fig, "fig3_think_vs_nothink.png")

# ── Figure 4: Max vs Eq budget type comparison ──────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
eq_tokens = [1000, 3000]

# Accuracy
ax = axes[0]
for i, model in enumerate(MODEL_ORDER):
    max_accs, eq_accs = [], []
    for tb in eq_tokens:
        mx = [r for r in records if r["model"] == model and r["tokens"] == tb
              and r["mode"] == "think" and r["budget_type"] == "max"]
        eq = [r for r in records if r["model"] == model and r["tokens"] == tb
              and r["mode"] == "think" and r["budget_type"] == "eq"]
        max_accs.append(mx[0]["success_rate"] * 100 if mx else 0)
        eq_accs.append(eq[0]["success_rate"] * 100 if eq else 0)
    x = np.arange(len(eq_tokens))
    offset = (i - 1.5) * bar_width
    ax.bar(x + offset - bar_width/4, max_accs, bar_width * 0.45, color=COLORS[model], alpha=0.85, label=f"{model} max")
    ax.bar(x + offset + bar_width/4, eq_accs, bar_width * 0.45, color=COLORS[model], alpha=0.4, label=f"{model} eq", hatch="\\\\")

ax.set_xlabel("Token Budget", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Max vs Eq Budget: Accuracy", fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(len(eq_tokens)))
ax.set_xticklabels([str(t) for t in eq_tokens])
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(30, 100)

# Latency
ax = axes[1]
for i, model in enumerate(MODEL_ORDER):
    max_times, eq_times = [], []
    for tb in eq_tokens:
        mx = [r for r in records if r["model"] == model and r["tokens"] == tb
              and r["mode"] == "think" and r["budget_type"] == "max"]
        eq = [r for r in records if r["model"] == model and r["tokens"] == tb
              and r["mode"] == "think" and r["budget_type"] == "eq"]
        max_times.append(mx[0]["avg_time"] if mx else 0)
        eq_times.append(eq[0]["avg_time"] if eq else 0)
    x = np.arange(len(eq_tokens))
    offset = (i - 1.5) * bar_width
    ax.bar(x + offset - bar_width/4, max_times, bar_width * 0.45, color=COLORS[model], alpha=0.85, label=f"{model} max")
    ax.bar(x + offset + bar_width/4, eq_times, bar_width * 0.45, color=COLORS[model], alpha=0.4, label=f"{model} eq", hatch="\\\\")

ax.set_xlabel("Token Budget", fontsize=11)
ax.set_ylabel("Avg Time per Task (s)", fontsize=11)
ax.set_title("Max vs Eq Budget: Latency", fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(len(eq_tokens)))
ax.set_xticklabels([str(t) for t in eq_tokens])
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save(fig, "fig4_max_vs_eq_budget.png")

# ── Figure 5: Accuracy vs Latency scatter (Pareto frontier) ──────────────────

fig, ax = plt.subplots(figsize=(9, 6))
for r in records:
    if r["budget_type"] == "eq":
        continue  # skip eq to avoid clutter
    marker = MARKERS[r["model"]]
    color = COLORS[r["model"]]
    edge = "black" if r["mode"] == "nothink" else color
    fill = "none" if r["mode"] == "nothink" else color
    label_str = f'{r["model"]} {r["tokens"] or "def"}{"" if r["mode"]=="think" else " NT"}'
    ax.scatter(r["avg_time"], r["success_rate"] * 100, marker=marker, c=fill if fill != "none" else "white",
               edgecolors=edge, s=100, linewidths=1.5, zorder=5)
    ax.annotate(label_str, (r["avg_time"], r["success_rate"] * 100),
                textcoords="offset points", xytext=(5, 5), fontsize=6.5, alpha=0.8)

# Legend entries
for model in MODEL_ORDER:
    ax.scatter([], [], marker=MARKERS[model], c=COLORS[model], s=80, label=f"{model}")
ax.scatter([], [], marker="o", c="white", edgecolors="black", s=80, label="nothink (hollow)")
ax.scatter([], [], marker="o", c="gray", s=80, label="think (filled)")

ax.set_xlabel("Avg Time per Task (s)", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy vs Latency Trade-off\n(All Configurations)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(30, 100)
save(fig, "fig5_accuracy_vs_latency.png")

# ── Figure 6: Accuracy by Model Size (fixed token budgets) ───────────────────

fig, ax = plt.subplots(figsize=(9, 5.5))
x_positions = [MODEL_SIZES_NUMERIC[m] for m in MODEL_ORDER]
for tb in [None] + TOKEN_BUDGETS:
    ys = []
    for model in MODEL_ORDER:
        pts = [r for r in records if r["model"] == model and r["tokens"] == tb
               and r["mode"] == "think" and (r["budget_type"] == "max" or tb is None)]
        ys.append(pts[0]["success_rate"] * 100 if pts else np.nan)
    label = "default" if tb is None else f"{tb} tokens"
    style = "--" if tb is None else "-"
    ax.plot(x_positions, ys, marker="o", label=label, linewidth=2, markersize=8, linestyle=style)

ax.set_xlabel("Model Size (B parameters)", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Scaling: Accuracy vs Model Size\n(Think Mode, Max Budget)", fontsize=13, fontweight="bold")
ax.set_xscale("log")
ax.set_xticks(x_positions)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xticklabels(MODEL_ORDER)
ax.legend(title="Token Budget", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(30, 100)
save(fig, "fig6_scaling_accuracy.png")

# ── Figure 7: Turn distribution of successful tasks (stacked % bars) ─────────

TURN_BUCKETS = [2, 4, 6, 8, 10, 12]

def get_success_turn_distribution(data):
    """Count successful repetitions by turn bucket."""
    counts = {t: 0 for t in TURN_BUCKETS}
    for task in data["tasks"].values():
        for rep in task["repetitions"].values():
            if rep["success"]:
                turns = rep["turns"]
                if turns in counts:
                    counts[turns] += 1
                else:
                    # Clamp to nearest bucket
                    closest = min(TURN_BUCKETS, key=lambda b: abs(b - turns))
                    counts[closest] += 1
    return counts

# Select configs: think mode, max budget type (+ defaults)
fig7_configs = []
for model in MODEL_ORDER:
    for tb in TOKEN_BUDGETS + [None]:
        r = [rec for rec in records if rec["model"] == model and rec["tokens"] == tb
             and rec["mode"] == "think" and (rec["budget_type"] == "max" or tb is None)]
        if r:
            fig7_configs.append(r[0])

n_configs = len(fig7_configs)
fig, ax = plt.subplots(figsize=(max(12, n_configs * 0.9), 6))

# Compute distributions
distributions = []
bar_labels = []
for cfg in fig7_configs:
    data = raw_data[cfg["folder"]]
    dist = get_success_turn_distribution(data)
    distributions.append(dist)
    tok_str = str(cfg["tokens"]) if cfg["tokens"] else "unlim"
    bar_labels.append(f"{cfg['model']}\n{tok_str}")

# Stacked bar colors: light to dark greens for low to high turns
n_buckets = len(TURN_BUCKETS)
bucket_colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, n_buckets))

x = np.arange(n_configs)
width = 0.65

for i, dist in enumerate(distributions):
    total = sum(dist.values())
    if total == 0:
        continue
    bottom = 0
    for j, t in enumerate(TURN_BUCKETS):
        pct = dist[t] / total * 100
        bar = ax.bar(x[i], pct, width, bottom=bottom, color=bucket_colors[j],
                     edgecolor="white", linewidth=0.5,
                     label=f"{t} turns" if i == 0 else None)
        if pct > 4:
            ax.text(x[i], bottom + pct / 2, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=6.5, fontweight="bold")
        bottom += pct

# Add vertical separators between model groups
for sep in range(1, len(MODEL_ORDER)):
    sep_x = sep * (len(TOKEN_BUDGETS) + 1) - 0.5
    if sep_x < n_configs:
        ax.axvline(x=sep_x, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_ylabel("% of Successful Tasks", fontsize=12)
ax.set_xlabel("Model / Max Output Tokens", fontsize=12)
ax.set_title("Turn Distribution of Successful Tasks\n(Think Mode, Max Budget)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(title="Turns", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
plt.tight_layout()
save(fig, "fig7_turns_vs_tokens.png")

# ── Figure 8: 0.6B all-tasks turn distribution, think vs nothink ─────────────

def get_all_tasks_turn_distribution(data):
    """Count all repetitions (success + failed) by turn bucket, separated."""
    success_counts = {t: 0 for t in TURN_BUCKETS}
    failed_counts = {t: 0 for t in TURN_BUCKETS}
    for task in data["tasks"].values():
        for rep in task["repetitions"].values():
            turns = rep.get("turns")
            if turns is None or turns == 0:
                turns = TURN_BUCKETS[-1]  # treat missing/zero as max turns
            closest = min(TURN_BUCKETS, key=lambda b: abs(b - turns))
            if rep["success"]:
                success_counts[closest] += 1
            else:
                failed_counts[closest] += 1
    return success_counts, failed_counts

# Collect 0.6b configs: think and nothink, max budget only
fig8_configs = []
for mode in ["think", "nothink"]:
    for tb in TOKEN_BUDGETS + [None]:
        r = [rec for rec in records if rec["model"] == "0.6b" and rec["tokens"] == tb
             and rec["mode"] == mode and (rec["budget_type"] == "max" or tb is None)]
        if r:
            fig8_configs.append(r[0])

n_cfg8 = len(fig8_configs)
fig, ax = plt.subplots(figsize=(max(10, n_cfg8 * 1.2), 6))

x = np.arange(n_cfg8)
width = 0.65

# Colors: greens for success turns (light→dark), reds for failed turns (light→dark)
success_colors = plt.cm.Greens(np.linspace(0.25, 0.85, len(TURN_BUCKETS)))
failed_colors = plt.cm.Reds(np.linspace(0.25, 0.85, len(TURN_BUCKETS)))

bar_labels_8 = []
for i, cfg in enumerate(fig8_configs):
    data = raw_data[cfg["folder"]]
    s_counts, f_counts = get_all_tasks_turn_distribution(data)
    total = sum(s_counts.values()) + sum(f_counts.values())
    if total == 0:
        bar_labels_8.append("")
        continue

    tok_str = str(cfg["tokens"]) if cfg["tokens"] else "unlim"
    bar_labels_8.append(f"{cfg['mode']}\n{tok_str}")

    # Stack: success turns bottom-up, then failed turns on top
    bottom = 0
    for j, t in enumerate(TURN_BUCKETS):
        pct = s_counts[t] / total * 100
        ax.bar(x[i], pct, width, bottom=bottom, color=success_colors[j],
               edgecolor="white", linewidth=0.5,
               label=f"Pass @ {t}t" if i == 0 else None)
        if pct > 3:
            ax.text(x[i], bottom + pct / 2, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=6.5, fontweight="bold")
        bottom += pct
    for j, t in enumerate(TURN_BUCKETS):
        pct = f_counts[t] / total * 100
        ax.bar(x[i], pct, width, bottom=bottom, color=failed_colors[j],
               edgecolor="white", linewidth=0.5,
               label=f"Fail @ {t}t" if i == 0 else None)
        if pct > 3:
            ax.text(x[i], bottom + pct / 2, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white")
        bottom += pct

# Separator between think and nothink groups
n_think = len([c for c in fig8_configs if c["mode"] == "think"])
ax.axvline(x=n_think - 0.5, color="gray", linewidth=1.2, linestyle="--", alpha=0.5)
ax.text(n_think / 2 - 0.5, 102, "think", ha="center", fontsize=11, fontweight="bold", color="#2c3e50")
ax.text(n_think + (n_cfg8 - n_think) / 2 - 0.5, 102, "nothink", ha="center", fontsize=11, fontweight="bold", color="#2c3e50")

ax.set_xticks(x)
ax.set_xticklabels(bar_labels_8, fontsize=9)
ax.set_ylabel("% of All Tasks", fontsize=12)
ax.set_xlabel("Mode / Max Output Tokens", fontsize=12)
ax.set_title("0.6B Turn Distribution: All Tasks (Pass vs Fail)\nThink vs NoThink", fontsize=13, fontweight="bold")
ax.set_ylim(0, 108)
ax.legend(title="Outcome", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, ncol=2)
plt.tight_layout()
save(fig, "fig8_06b_think_nothink_turns.png")

# ── Generate summary table ───────────────────────────────────────────────────

print("\n" + "=" * 100)
print("EXPERIMENT SUMMARY TABLE")
print("=" * 100)
print(f"{'Config':<40} {'Accuracy':>10} {'Avg Time (s)':>14} {'Avg Turns':>10}")
print("-" * 100)
for r in sorted(records, key=lambda x: (MODEL_ORDER.index(x["model"]), x["tokens"] or 0, x["budget_type"], x["mode"])):
    tok_str = f"{r['tokens']}" if r['tokens'] else "default"
    config = f"{r['model']} | {tok_str:>5} {r['budget_type']:>3} | {r['mode']}"
    print(f"{config:<40} {r['success_rate']*100:>9.2f}% {r['avg_time']:>14.2f} {r['avg_turns']:>10.2f}")

print("\n" + "=" * 100)
print("KEY FINDINGS")
print("=" * 100)

# Best configs
best_acc = max(records, key=lambda x: x["success_rate"])
fastest = min(records, key=lambda x: x["avg_time"])
best_pareto_candidates = [r for r in records if r["success_rate"] >= 0.90]
if best_pareto_candidates:
    best_pareto = min(best_pareto_candidates, key=lambda x: x["avg_time"])
else:
    best_pareto = best_acc

print(f"\n1. BEST ACCURACY: {best_acc['folder']} — {best_acc['success_rate']*100:.2f}% (avg {best_acc['avg_time']:.1f}s)")
print(f"2. FASTEST:       {fastest['folder']} — {fastest['avg_time']:.2f}s (accuracy {fastest['success_rate']*100:.2f}%)")
print(f"3. BEST TRADE-OFF (>=90% acc, min latency): {best_pareto['folder']} — {best_pareto['success_rate']*100:.2f}% in {best_pareto['avg_time']:.2f}s")

# Think vs nothink summary
print(f"\n4. THINK vs NOTHINK (accuracy deltas at 1000 max tokens):")
for model in MODEL_ORDER:
    t = [r for r in records if r["model"] == model and r["tokens"] == 1000 and r["mode"] == "think" and r["budget_type"] == "max"]
    nt = [r for r in records if r["model"] == model and r["tokens"] == 1000 and r["mode"] == "nothink" and r["budget_type"] == "max"]
    if t and nt:
        delta = (t[0]["success_rate"] - nt[0]["success_rate"]) * 100
        speedup = t[0]["avg_time"] / nt[0]["avg_time"] if nt[0]["avg_time"] > 0 else 0
        print(f"   {model:>5}: think {t[0]['success_rate']*100:.1f}% vs nothink {nt[0]['success_rate']*100:.1f}% (delta {delta:+.1f}pp) | speedup {speedup:.1f}x")

print(f"\n5. SCALING EFFECT (default config):")
for model in MODEL_ORDER:
    base = [r for r in records if r["model"] == model and r["tokens"] is None and r["mode"] == "think"]
    if base:
        print(f"   {model:>5}: {base[0]['success_rate']*100:.2f}% accuracy, {base[0]['avg_time']:.1f}s avg time")

print(f"\nAll figures saved to: {REPORT_DIR}/")
