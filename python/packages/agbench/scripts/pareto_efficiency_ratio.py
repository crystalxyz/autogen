import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

for c in configs:
    c['eff_mean'] = c['acc'] / c['mean']
    c['eff_p50'] = c['acc'] / c['p50']
    c['eff_p90'] = c['acc'] / c['p90']

style = {
    '1T': {'color': '#1565C0', 'marker': 'o', 'size': 60, 'label': '1-Turn'},
    '2T': {'color': '#E65100', 'marker': 's', 'size': 50, 'label': '2-Turn'},
    '3T': {'color': '#2E7D32', 'marker': '^', 'size': 55, 'label': '3-Turn'},
    '4T': {'color': '#C62828', 'marker': 'D', 'size': 65, 'label': '4-Turn'},
}

# --- Main plot: efficiency vs accuracy ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
eff_metrics = [('eff_mean', 'Acc / Mean Latency'), ('eff_p50', 'Acc / P50 Latency'), ('eff_p90', 'Acc / P90 Latency')]

for ax, (eff_key, ylabel) in zip(axes, eff_metrics):
    for typ in ['1T', '2T', '3T', '4T']:
        s = style[typ]
        tc = [c for c in configs if c['type'] == typ]
        ax.scatter([c['acc']*100 for c in tc], [c[eff_key] for c in tc],
                  color=s['color'], marker=s['marker'], s=s['size'],
                  alpha=0.6, zorder=3, edgecolors='black', linewidths=0.3,
                  label=s['label'])
    
    # Label top points
    top = sorted(configs, key=lambda x: -x[eff_key])[:5]
    for c in top:
        ax.annotate(f"{c['name']}",
                    (c['acc']*100, c[eff_key]),
                    fontsize=7, fontweight='bold',
                    color=style[c['type']]['color'],
                    xytext=(5, 5), textcoords='offset points',
                    ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5))
    
    # Also label best per accuracy tier
    tiers = [(0.90, 0.95), (0.95, 0.98), (0.98, 1.01)]
    for lo, hi in tiers:
        tier = [c for c in configs if lo <= c['acc'] < hi]
        if tier:
            best = max(tier, key=lambda x: x[eff_key])
            if best not in top:
                ax.annotate(f"{best['name']}",
                            (best['acc']*100, best[eff_key]),
                            fontsize=6.5, fontweight='bold',
                            color=style[best['type']]['color'],
                            xytext=(5, 3), textcoords='offset points',
                            ha='left', va='bottom',
                            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5))
    
    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_ylabel(f'Efficiency ({ylabel})', fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)

plt.suptitle('Efficiency (Accuracy / Latency) vs Accuracy — Higher is Better', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('reports/pareto/pareto_efficiency_ratio.png', dpi=200, bbox_inches='tight')
print('Saved: pareto_efficiency_ratio.png')

# --- Bar chart of best per tier ---
fig2, ax2 = plt.subplots(figsize=(12, 6))

tiers = [
    (0.0, 0.70, '<70%'),
    (0.70, 0.85, '70-85%'),
    (0.85, 0.90, '85-90%'),
    (0.90, 0.95, '90-95%'),
    (0.95, 0.98, '95-98%'),
    (0.98, 1.01, '98%+'),
]

bar_data = []
for lo, hi, label in tiers:
    tier = [c for c in configs if lo <= c['acc'] < hi]
    if tier:
        best = max(tier, key=lambda x: x['eff_mean'])
        bar_data.append((label, best))

x = np.arange(len(bar_data))
colors = [style[bd[1]['type']]['color'] for bd in bar_data]
effs = [bd[1]['eff_mean'] for bd in bar_data]

bars = ax2.bar(x, effs, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85, width=0.6)

for i, (tier_label, best) in enumerate(bar_data):
    ax2.text(i, effs[i] + 0.003, f"{best['type']}:{best['name']}\n{best['acc']*100:.1f}% / {best['mean']:.1f}s",
             ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels([bd[0] for bd in bar_data], fontsize=10)
ax2.set_xlabel('Accuracy Tier', fontsize=12)
ax2.set_ylabel('Efficiency (Accuracy / Mean Latency)', fontsize=12)
ax2.set_title('Most Efficient Config per Accuracy Tier', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.25, axis='y')
ax2.set_ylim(0, max(effs) * 1.35)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=style[t]['color'], edgecolor='black', label=style[t]['label']) 
                   for t in ['1T','2T','3T','4T']]
ax2.legend(handles=legend_elements, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('reports/pareto/pareto_efficiency_by_tier.png', dpi=200, bbox_inches='tight')
print('Saved: pareto_efficiency_by_tier.png')
