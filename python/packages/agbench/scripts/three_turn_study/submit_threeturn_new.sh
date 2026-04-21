#!/bin/bash
# Submit new three-turn experiments (pareto frontier analysis additions)
# Total runs: 6

set -e

echo "[1/6] Submitting 4nt -> 8nt -> 14nt (3 GPUs) — all-nothink cascade"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-4nt-8nt-14nt.yaml" "outputs/three-turn-nc-4nt-8nt-14nt"
echo "[2/6] Submitting 1.7nt -> 4nt -> 14nt (3 GPUs) — all-nothink cascade"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-1.7nt-4nt-14nt.yaml" "outputs/three-turn-nc-1.7nt-4nt-14nt"
echo "[3/6] Submitting 8nt -> 4t -> 14t (3 GPUs) — 8nt turn1 + think escalation"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-8nt-4t-14t.yaml" "outputs/three-turn-nc-8nt-4t-14t"
echo "[4/6] Submitting 8nt -> 8t -> 14t (3 GPUs) — 8nt turn1 + think escalation"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-8nt-8t-14t.yaml" "outputs/three-turn-nc-8nt-8t-14t"
echo "[5/6] Submitting 4nt -> 1.7t -> 8t (3 GPUs) — cheaper turn3"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-4nt-1.7t-8t.yaml" "outputs/three-turn-nc-4nt-1.7t-8t"
echo "[6/6] Submitting 8nt -> 1.7t -> 8t (3 GPUs) — cheaper turn3"
sbatch scripts/three_turn_study/start_threeturn.sh "scripts/configs/config-threeturn-nc-8nt-1.7t-8t.yaml" "outputs/three-turn-nc-8nt-1.7t-8t"

echo "All 6 new three-turn jobs submitted."
