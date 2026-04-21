import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_elapsed_times(output_dir):
    results = []
    for rj in glob.glob(os.path.join(output_dir, 'Results', '*', 'result.json')):
        with open(rj) as f:
            data = json.load(f)
        for task_id, task_data in data.get('tasks', {}).items():
            for rep_id, rep_data in task_data.get('repetitions', {}).items():
                elapsed = rep_data.get('elapsed_time')
                success = rep_data.get('success', False)
                if elapsed is not None and elapsed != '':
                    results.append((float(elapsed), bool(success)))
    return results

def compute_accuracy_at_timeout(results, timeouts):
    total = len(results)
    if total == 0:
        return [0.0] * len(timeouts)
    return [sum(1 for e, s in results if s and e <= t) / total for t in timeouts]

configs = [
    ('outputs/one-turn-4b-nothink',            '4nt (1T)',              '#1565C0', 1.5),
    ('outputs/two-turn-nc-4nt-8nt',            '4nt→8nt (2T)',          '#E65100', 1.8),
    ('outputs/three-turn-nc-4nt-8nt-14nt',     '4nt→8nt→14nt (3T)',     '#2E7D32', 2.0),
    ('outputs/four-turn-nc-4nt-8nt-14nt-4t',   '4nt→8nt→14nt→4t (4T)',  '#C62828', 2.5),
]

timeouts = np.linspace(0.5, 100, 1000)

fig, ax = plt.subplots(figsize=(12, 7))

for output_dir, name, color, lw in configs:
    results = load_elapsed_times(output_dir)
    accs = compute_accuracy_at_timeout(results, timeouts)
    final_acc = sum(1 for _, s in results if s) / len(results)
    ax.plot(timeouts, [a*100 for a in accs], color=color, linewidth=lw,
            alpha=0.9, label=f'{name} — {final_acc*100:.1f}%')

for pct in [80, 90, 95, 98]:
    ax.axhline(y=pct, color='gray', linestyle=':', alpha=0.25, linewidth=0.8)
    ax.text(101, pct, f'{pct}%', fontsize=8, alpha=0.5, va='center')

ax.set_xlabel('Timeout Budget (seconds)', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Nothink Staircase: Accuracy vs Timeout (4nt prefix, +1 turn each step)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 100)
ax.set_ylim(0, 101)
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('reports/pareto/accuracy_vs_timeout_staircase.png', dpi=200, bbox_inches='tight')
print('Saved: accuracy_vs_timeout_staircase.png')

# Zoomed
fig2, ax2 = plt.subplots(figsize=(12, 7))

for output_dir, name, color, lw in configs:
    results = load_elapsed_times(output_dir)
    accs = compute_accuracy_at_timeout(results, timeouts)
    final_acc = sum(1 for _, s in results if s) / len(results)
    ax2.plot(timeouts, [a*100 for a in accs], color=color, linewidth=lw,
            alpha=0.9, label=f'{name} — {final_acc*100:.1f}%')

for pct in [80, 85, 90, 95, 98]:
    ax2.axhline(y=pct, color='gray', linestyle=':', alpha=0.25, linewidth=0.8)
    ax2.text(101, pct, f'{pct}%', fontsize=8, alpha=0.5, va='center')

ax2.set_xlabel('Timeout Budget (seconds)', fontsize=13)
ax2.set_ylabel('Accuracy (%)', fontsize=13)
ax2.set_title('Nothink Staircase (Zoomed): Accuracy vs Timeout', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, 50)
ax2.set_ylim(70, 100)
ax2.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('reports/pareto/accuracy_vs_timeout_staircase_zoomed.png', dpi=200, bbox_inches='tight')
print('Saved: accuracy_vs_timeout_staircase_zoomed.png')
