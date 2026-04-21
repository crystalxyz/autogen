import csv
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

frontier_configs = [
    ('outputs/one-turn-0.6b-nothink', '0.6b-nt', '1T'),
    ('outputs/one-turn-1.7b-nothink', '1.7b-nt', '1T'),
    ('outputs/one-turn-4b-nothink', '4b-nt', '1T'),
    ('outputs/two-turn-nc-4nt-8nt', '4nt→8nt', '2T'),
    ('outputs/two-turn-nc-4nt-14nt', '4nt→14nt', '2T'),
    ('outputs/two-turn-nc-8nt-4t', '8nt→4t', '2T'),
    ('outputs/two-turn-nc-14nt-4t', '14nt→4t', '2T'),
    ('outputs/two-turn-nc-4nt-14t', '4nt→14t', '2T'),
    ('outputs/two-turn-nc-8nt-14t', '8nt→14t', '2T'),
    ('outputs/three-turn-nc-4nt-8nt-14nt', '4nt→8nt→14nt', '3T'),
    ('outputs/three-turn-nc-4nt-14nt-4t', '4nt→14nt→4t', '3T'),
    ('outputs/three-turn-nc-8nt-4t-14t', '8nt→4t→14t', '3T'),
    ('outputs/three-turn-nc-14nt-4t-14t', '14nt→4t→14t', '3T'),
    ('outputs/four-turn-nc-4nt-8nt-14nt-4t', '4nt→8nt→14nt→4t', '4T'),
    ('outputs/four-turn-nc-4nt-8nt-14nt-14t', '4nt→8nt→14nt→14t', '4T'),
    ('outputs/four-turn-nc-4nt-8nt-4t-14t', '4nt→8nt→4t→14t', '4T'),
    ('outputs/four-turn-nc-4nt-14nt-4t-14t', '4nt→14nt→4t→14t', '4T'),
]

timeouts = np.linspace(0.5, 100, 1000)

all_data = []
for output_dir, name, typ in frontier_configs:
    results = load_elapsed_times(output_dir)
    if not results:
        continue
    accs = compute_accuracy_at_timeout(results, timeouts)
    final_acc = sum(1 for _, s in results if s) / len(results)
    all_data.append((name, typ, np.array(accs), final_acc))

# Compute envelope per type and overall
type_colors = {'1T': '#1565C0', '2T': '#E65100', '3T': '#2E7D32', '4T': '#C62828'}
type_labels = {'1T': '1-Turn', '2T': '2-Turn', '3T': '3-Turn', '4T': '4-Turn'}

# --- Plot: Envelope per turn count + overall ---
fig, ax = plt.subplots(figsize=(14, 8))

# Per-type envelopes
for typ in ['1T', '2T', '3T', '4T']:
    typ_accs = [accs for name, t, accs, _ in all_data if t == typ]
    if not typ_accs:
        continue
    envelope = np.maximum.reduce(typ_accs) * 100
    ax.plot(timeouts, envelope, color=type_colors[typ], linewidth=2.5, alpha=0.85,
            label=f'{type_labels[typ]} envelope')

# Overall envelope
all_accs = [accs for _, _, accs, _ in all_data]
overall_envelope = np.maximum.reduce(all_accs) * 100
ax.plot(timeouts, overall_envelope, color='black', linewidth=3, alpha=0.5,
        linestyle='--', label='Overall best')

# Find which config is best at key timeouts and annotate
key_timeouts = [2, 3, 5, 8, 10, 15, 20, 30, 50, 80]
for kt in key_timeouts:
    idx = np.argmin(np.abs(timeouts - kt))
    best_name = ''
    best_acc = 0
    best_typ = ''
    for name, typ, accs, _ in all_data:
        if accs[idx] > best_acc:
            best_acc = accs[idx]
            best_name = name
            best_typ = typ
    if best_acc > 0:
        ax.plot(kt, best_acc*100, 'ko', markersize=4, zorder=10)
        # Alternate label position
        y_off = 1.5 if kt % 6 < 3 else -2.0
        ax.annotate(f'{best_name}\n{best_acc*100:.1f}%',
                   (kt, best_acc*100), fontsize=6.5, fontweight='bold',
                   color=type_colors[best_typ],
                   xytext=(3, y_off), textcoords='offset fontsize',
                   ha='left', va='bottom' if y_off > 0 else 'top',
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5))

for pct in [90, 95, 98, 99]:
    ax.axhline(y=pct, color='gray', linestyle=':', alpha=0.25, linewidth=0.8)

ax.set_xlabel('Timeout Budget (seconds)', fontsize=13)
ax.set_ylabel('Best Achievable Accuracy (%)', fontsize=13)
ax.set_title('Accuracy vs Timeout Budget: Best Config Envelope by Turn Count', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 100)
ax.set_ylim(0, 101)
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('reports/pareto/accuracy_vs_timeout_envelope.png', dpi=200, bbox_inches='tight')
print('Saved: accuracy_vs_timeout_envelope.png')

# --- Zoomed version ---
fig2, ax2 = plt.subplots(figsize=(14, 8))

for typ in ['1T', '2T', '3T', '4T']:
    typ_accs = [accs for name, t, accs, _ in all_data if t == typ]
    if not typ_accs:
        continue
    envelope = np.maximum.reduce(typ_accs) * 100
    ax2.plot(timeouts, envelope, color=type_colors[typ], linewidth=2.5, alpha=0.85,
            label=f'{type_labels[typ]} envelope')

ax2.plot(timeouts, overall_envelope, color='black', linewidth=3, alpha=0.5,
        linestyle='--', label='Overall best')

# Annotate at key timeouts in zoomed view
for kt in key_timeouts:
    idx = np.argmin(np.abs(timeouts - kt))
    best_name = ''
    best_acc = 0
    best_typ = ''
    for name, typ, accs, _ in all_data:
        if accs[idx] > best_acc:
            best_acc = accs[idx]
            best_name = name
            best_typ = typ
    if best_acc > 0.84:
        ax2.plot(kt, best_acc*100, 'ko', markersize=5, zorder=10)
        y_off = 1.2 if kt % 6 < 3 else -1.5
        ax2.annotate(f'{best_name}\n{best_acc*100:.1f}%',
                   (kt, best_acc*100), fontsize=7, fontweight='bold',
                   color=type_colors[best_typ],
                   xytext=(3, y_off), textcoords='offset fontsize',
                   ha='left', va='bottom' if y_off > 0 else 'top',
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5))

for pct in [90, 95, 98, 99]:
    ax2.axhline(y=pct, color='gray', linestyle=':', alpha=0.25, linewidth=0.8)
    ax2.text(101, pct, f'{pct}%', fontsize=8, alpha=0.5, va='center')

ax2.set_xlabel('Timeout Budget (seconds)', fontsize=13)
ax2.set_ylabel('Best Achievable Accuracy (%)', fontsize=13)
ax2.set_title('Accuracy vs Timeout Budget (Zoomed): Best Config Envelope', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, 100)
ax2.set_ylim(85, 100.5)
ax2.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('reports/pareto/accuracy_vs_timeout_envelope_zoomed.png', dpi=200, bbox_inches='tight')
print('Saved: accuracy_vs_timeout_envelope_zoomed.png')

# --- Print the timeout frontier table ---
print('\n=== TIMEOUT FRONTIER: best config at each budget ===')
print(f'  {"Timeout":>8s}  {"Best Config":35s} {"Type":4s} {"Acc":>6s}')
print('  ' + '-'*60)
prev_name = ''
for kt in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
    idx = np.argmin(np.abs(timeouts - kt))
    best_name = ''
    best_acc = 0
    best_typ = ''
    for name, typ, accs, _ in all_data:
        if accs[idx] > best_acc:
            best_acc = accs[idx]
            best_name = name
            best_typ = typ
    marker = ' *' if best_name != prev_name else ''
    print(f'  {kt:>7.0f}s  {best_name:35s} {best_typ:4s} {best_acc*100:5.1f}%{marker}')
    prev_name = best_name
