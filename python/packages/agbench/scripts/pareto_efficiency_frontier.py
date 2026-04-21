import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Load all data
configs = []

with open('reports/csv/one_turn_baselines.csv') as f:
    for row in csv.DictReader(f):
        configs.append({
            'name': row['setup'].replace('one-turn-',''),
            'type': '1T',
            'acc': float(row['success_rate']),
            'mean': float(row['latency_mean']),
            'p50': float(row['latency_p50']),
            'p90': float(row['latency_p90']),
        })

with open('reports/csv/two_turn_nc_stats.csv') as f:
    for row in csv.DictReader(f):
        configs.append({
            'name': row['setup'].replace('two-turn-nc-',''),
            'type': '2T',
            'acc': float(row['accuracy']),
            'mean': float(row['mean_latency']),
            'p50': float(row['p50_latency']),
            'p90': float(row['p90_latency']),
        })

with open('reports/csv/three_turn_nc_stats.csv') as f:
    for row in csv.DictReader(f):
        configs.append({
            'name': row['setup'].replace('three-turn-nc-','').replace('three-turn-',''),
            'type': '3T',
            'acc': float(row['accuracy']),
            'mean': float(row['mean_latency']),
            'p50': float(row['p50_latency']),
            'p90': float(row['p90_latency']),
        })

with open('reports/csv/four_turn_nc_stats.csv') as f:
    for row in csv.DictReader(f):
        configs.append({
            'name': row['setup'].replace('four-turn-nc-',''),
            'type': '4T',
            'acc': float(row['accuracy']),
            'mean': float(row['mean_latency']),
            'p50': float(row['p50_latency']),
            'p90': float(row['p90_latency']),
        })

style = {
    '1T': {'color': '#1565C0', 'marker': 'o', 'size': 50, 'label': '1-Turn'},
    '2T': {'color': '#E65100', 'marker': 's', 'size': 40, 'label': '2-Turn'},
    '3T': {'color': '#2E7D32', 'marker': '^', 'size': 50, 'label': '3-Turn'},
    '4T': {'color': '#C62828', 'marker': 'D', 'size': 55, 'label': '4-Turn'},
}

def compute_frontier(configs, metric):
    sorted_c = sorted(configs, key=lambda x: x[metric])
    frontier = []
    max_acc = 0
    for c in sorted_c:
        if c['acc'] > max_acc:
            max_acc = c['acc']
            frontier.append(c)
    return frontier

# --- Zoomed mean latency plot (single large panel) ---
fig, ax = plt.subplots(figsize=(10, 7))
metric = 'mean'

frontier = compute_frontier(configs, metric)
frontier_set = set(id(c) for c in frontier)

zoomed = [c for c in configs if c['acc'] >= 0.85 and c[metric] <= 35]

for typ in ['1T', '2T', '3T', '4T']:
    s = style[typ]
    typ_configs = [c for c in zoomed if c['type'] == typ]
    
    non_f = [c for c in typ_configs if id(c) not in frontier_set]
    if non_f:
        ax.scatter([c[metric] for c in non_f], [c['acc']*100 for c in non_f],
                  color=s['color'], marker=s['marker'], s=s['size']*0.7,
                  alpha=0.15, zorder=2)
    
    on_f = [c for c in typ_configs if id(c) in frontier_set]
    if on_f:
        ax.scatter([c[metric] for c in on_f], [c['acc']*100 for c in on_f],
                  color=s['color'], marker=s['marker'], s=s['size']*2.2,
                  alpha=0.95, zorder=5, edgecolors='black', linewidths=0.6,
                  label=f'{s["label"]} ({len(on_f)} on frontier / {len(typ_configs)} total)')

# Frontier line
f_sorted = [c for c in sorted(frontier, key=lambda x: x[metric]) if c['acc'] >= 0.85 and c[metric] <= 35]
if f_sorted:
    ax.plot([c[metric] for c in f_sorted], [c['acc']*100 for c in f_sorted],
            color='black', linewidth=1.5, linestyle='--', alpha=0.4, zorder=4)

# Manual label offsets to avoid overlaps
label_offsets = {
    # (type, name): (x_off, y_off, ha, va)
    ('2T', '4nt-8nt'):          (1.5, -0.3, 'left', 'top'),
    ('3T', '4nt-8nt-14nt'):     (1.0, 0.3, 'left', 'bottom'),
    ('3T', '4nt-14nt-4t'):      (1.0, 0.5, 'left', 'bottom'),
    ('4T', '4nt-8nt-14nt-4t'):  (1.0, 0.3, 'left', 'bottom'),
    ('4T', '4nt-8nt-14nt-14t'): (1.0, -0.3, 'left', 'top'),
    ('4T', '4nt-8nt-4t-14t'):   (1.0, 0.4, 'left', 'bottom'),
    ('4T', '4nt-14nt-4t-14t'):  (-1.0, 0.4, 'right', 'bottom'),
}

for c in frontier:
    if c['acc'] < 0.85 or c[metric] > 35:
        continue
    key = (c['type'], c['name'])
    x_off, y_off, ha, va = label_offsets.get(key, (1.5, -0.5, 'left', 'top'))
    
    label = c['name']
    
    ax.annotate(f"{label}",
                (c[metric], c['acc']*100),
                fontsize=7.5, alpha=0.85, fontweight='bold',
                color=style[c['type']]['color'],
                xytext=(x_off, y_off), textcoords='offset fontsize',
                ha=ha, va=va,
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5))

ax.set_xlabel('Mean Latency (s)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Efficiency Frontier: Accuracy vs Mean Latency (1T/2T/3T/4T)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_ylim(85, 100.5)
ax.set_xlim(0, 35)
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('reports/pareto/pareto_efficiency_frontier_zoomed.png', dpi=200, bbox_inches='tight')
print('Saved zoomed')

# --- Full range 3-panel ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
metrics_list = [('mean', 'Mean Latency (s)'), ('p50', 'P50 Latency (s)'), ('p90', 'P90 Latency (s)')]

for ax, (met, xlabel) in zip(axes, metrics_list):
    frontier = compute_frontier(configs, met)
    frontier_set_m = set(id(c) for c in frontier)
    
    for typ in ['1T', '2T', '3T', '4T']:
        s = style[typ]
        typ_configs = [c for c in configs if c['type'] == typ]
        
        non_f = [c for c in typ_configs if id(c) not in frontier_set_m]
        if non_f:
            ax.scatter([c[met] for c in non_f], [c['acc']*100 for c in non_f],
                      color=s['color'], marker=s['marker'], s=s['size']*0.5,
                      alpha=0.12, zorder=2)
        
        on_f = [c for c in typ_configs if id(c) in frontier_set_m]
        if on_f:
            ax.scatter([c[met] for c in on_f], [c['acc']*100 for c in on_f],
                      color=s['color'], marker=s['marker'], s=s['size']*1.8,
                      alpha=0.9, zorder=5, edgecolors='black', linewidths=0.5,
                      label=f'{s["label"]} ({len(on_f)})')
    
    f_sorted = sorted(frontier, key=lambda x: x[met])
    ax.plot([c[met] for c in f_sorted], [c['acc']*100 for c in f_sorted],
            color='black', linewidth=1.2, linestyle='--', alpha=0.4, zorder=4)
    
    # Label frontier points
    for c in frontier:
        label = c['name']
        if len(label) > 22:
            label = label[:20] + '..'
        color = style[c['type']]['color']
        
        ax.annotate(f"{label}",
                    (c[met], c['acc']*100),
                    fontsize=5.5, alpha=0.8, color=color, fontweight='bold',
                    xytext=(4, -6), textcoords='offset points',
                    ha='left', va='top')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Accuracy vs {met.upper()} Latency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(30, 101)

plt.suptitle('Efficiency Frontier: All Configs (1T / 2T / 3T / 4T)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('reports/pareto/pareto_efficiency_frontier.png', dpi=200, bbox_inches='tight')
print('Saved full')
