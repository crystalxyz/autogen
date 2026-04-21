import json
import os
import glob
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

def accuracy_at_timeout(results, t):
    total = len(results)
    if total == 0:
        return 0.0
    return sum(1 for e, s in results if s and e <= t) / total

# Discover all output directories
all_configs = []

# 1T
for d in sorted(glob.glob('outputs/one-turn-*')):
    name = os.path.basename(d).replace('one-turn-', '')
    if name.endswith('-5re'):  # skip variant runs
        continue
    results = load_elapsed_times(d)
    if results:
        all_configs.append((name, '1T', results))

# 2T
for d in sorted(glob.glob('outputs/two-turn-nc-*')):
    name = os.path.basename(d).replace('two-turn-nc-', '')
    results = load_elapsed_times(d)
    if results:
        all_configs.append((name, '2T', results))

# 3T
for d in sorted(glob.glob('outputs/three-turn-nc-*')) + sorted(glob.glob('outputs/three-turn-0*')):
    name = os.path.basename(d).replace('three-turn-nc-', '').replace('three-turn-', '')
    results = load_elapsed_times(d)
    if results:
        all_configs.append((name, '3T', results))

# 4T
for d in sorted(glob.glob('outputs/four-turn-nc-*')):
    name = os.path.basename(d).replace('four-turn-nc-', '')
    results = load_elapsed_times(d)
    if results:
        all_configs.append((name, '4T', results))

print(f'Loaded {len(all_configs)} configs')
for typ in ['1T', '2T', '3T', '4T']:
    n = sum(1 for _, t, _ in all_configs if t == typ)
    print(f'  {typ}: {n} configs')

timeouts = list(range(10, 125, 5))

lines = []
lines.append('Best Model Config per Timeout Budget')
lines.append('=' * 80)
lines.append(f'Generated from {len(all_configs)} total configs (all available runs)')
lines.append('')
lines.append('')

# Table 1: Best per type at each timeout
lines.append('=== Best Config per Turn Count at Each Timeout ===')
lines.append('')

for t in timeouts:
    lines.append(f'--- Timeout = {t}s ---')
    lines.append(f'  {"Type":4s}  {"Best Config":40s} {"Acc":>7s}  {"Final Acc":>9s}')
    lines.append(f'  ' + '-' * 65)
    
    overall_best_name = ''
    overall_best_acc = 0
    overall_best_typ = ''
    
    for typ in ['1T', '2T', '3T', '4T']:
        typ_configs = [(name, results) for name, tp, results in all_configs if tp == typ]
        if not typ_configs:
            continue
        best_name = ''
        best_acc = 0
        best_final = 0
        for name, results in typ_configs:
            acc = accuracy_at_timeout(results, t)
            final = sum(1 for _, s in results if s) / len(results)
            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_final = final
        
        marker = ''
        if best_acc > overall_best_acc:
            overall_best_acc = best_acc
            overall_best_name = best_name
            overall_best_typ = typ
        
        lines.append(f'  {typ:4s}  {best_name:40s} {best_acc*100:6.1f}%  {best_final*100:8.1f}%')
    
    lines.append(f'  >> OVERALL BEST: [{overall_best_typ}] {overall_best_name} at {overall_best_acc*100:.1f}%')
    lines.append('')

# Table 2: Compact summary
lines.append('')
lines.append('=== Summary Table ===')
lines.append('')
lines.append(f'  {"Timeout":>7s}  {"Best 1T":>25s}  {"Best 2T":>25s}  {"Best 3T":>25s}  {"Best 4T":>25s}  {"Overall Best":>30s}')
lines.append(f'  ' + '-' * 145)

for t in timeouts:
    row = f'  {t:>6.0f}s'
    overall_best = ('', '', 0)
    
    for typ in ['1T', '2T', '3T', '4T']:
        typ_configs = [(name, results) for name, tp, results in all_configs if tp == typ]
        best_name = ''
        best_acc = 0
        for name, results in typ_configs:
            acc = accuracy_at_timeout(results, t)
            if acc > best_acc:
                best_acc = acc
                best_name = name
        
        short = f'{best_name} {best_acc*100:.1f}%'
        row += f'  {short:>25s}'
        
        if best_acc > overall_best[2]:
            overall_best = (best_name, typ, best_acc)
    
    row += f'  [{overall_best[1]}] {overall_best[0]:>20s} {overall_best[2]*100:.1f}%'
    lines.append(row)

# Table 3: Which turn count wins at each timeout
lines.append('')
lines.append('')
lines.append('=== Winning Turn Count per Timeout ===')
lines.append('')

prev_winner = ''
for t in timeouts:
    best_per_type = {}
    for typ in ['1T', '2T', '3T', '4T']:
        typ_configs = [(name, results) for name, tp, results in all_configs if tp == typ]
        best_acc = 0
        best_name = ''
        for name, results in typ_configs:
            acc = accuracy_at_timeout(results, t)
            if acc > best_acc:
                best_acc = acc
                best_name = name
        best_per_type[typ] = (best_name, best_acc)
    
    winner_typ = max(best_per_type, key=lambda k: best_per_type[k][1])
    winner_name, winner_acc = best_per_type[winner_typ]
    
    # Show runner-up
    runner_typ = max([k for k in best_per_type if k != winner_typ], key=lambda k: best_per_type[k][1])
    runner_name, runner_acc = best_per_type[runner_typ]
    gap = (winner_acc - runner_acc) * 100
    
    change = ' << NEW LEADER' if winner_typ != prev_winner else ''
    prev_winner = winner_typ
    
    lines.append(f'  {t:>3.0f}s:  [{winner_typ}] {winner_name:35s} {winner_acc*100:5.1f}%   (runner-up: [{runner_typ}] {runner_name} {runner_acc*100:.1f}%, gap={gap:+.1f}%){change}')

output = '\n'.join(lines) + '\n'

with open('reports/best_config_per_timeout.txt', 'w') as f:
    f.write(output)

print(output)
print('\nSaved to reports/best_config_per_timeout.txt')
