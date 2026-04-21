#!/bin/bash
# Submit three-turn experiments derived from 1T/2T dominance analysis
# Total runs: 6
# Derivation: replace dominated 1T turns in 2T frontier points with dominating 2T cascades

set -e

echo "[1/6] Submitting 4nt -> 8nt -> 1.7t (targets ~94% at ~5s, replaces 14nt->1.7t)"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-4nt-8nt-1.7t.yaml" "outputs/three-turn-nc-4nt-8nt-1.7t"
echo "[2/6] Submitting 4nt -> 8nt -> 4t (targets ~97% at ~5-6s, replaces 8nt->4t / 14nt->4t)"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-4nt-8nt-4t.yaml" "outputs/three-turn-nc-4nt-8nt-4t"
echo "[3/6] Submitting 4nt -> 8nt -> 8t (targets ~95% at ~5s)"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-4nt-8nt-8t.yaml" "outputs/three-turn-nc-4nt-8nt-8t"
echo "[4/6] Submitting 4nt -> 8nt -> 14t (targets ~98% at ~5-7s, replaces 8nt->14t / 14nt->14t)"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-4nt-8nt-14t.yaml" "outputs/three-turn-nc-4nt-8nt-14t"
echo "[5/6] Submitting 1.7nt -> 4nt -> 4t (P90-optimal, tighter tail than 4nt->8nt prefix)"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-1.7nt-4nt-4t.yaml" "outputs/three-turn-nc-1.7nt-4nt-4t"
echo "[6/6] Submitting 4nt -> 14nt -> 4t (turn 2 substitution: 14t replaced by 14nt->4t)"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-4nt-14nt-4t.yaml" "outputs/three-turn-nc-4nt-14nt-4t"

echo "All 6 compositional three-turn jobs submitted."
