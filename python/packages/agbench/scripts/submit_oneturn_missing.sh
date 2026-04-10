#!/bin/bash
# Submit all missing one-turn baseline experiments
# Missing: 0.6b-think, 1.7b-think, 4b-think, 8b-nothink
#
# Usage: bash scripts/submit_oneturn_missing.sh

set -e
cd "$(dirname "$0")/.."

echo "Submitting missing one-turn baseline runs..."

sbatch scripts/start.sh scripts/configs/config-oneturn-0.6b-think.yaml  outputs/one-turn-0.6b-think
sbatch scripts/start.sh scripts/configs/config-oneturn-1.7b-think.yaml  outputs/one-turn-1.7b-think
sbatch scripts/start.sh scripts/configs/config-oneturn-4b-think.yaml    outputs/one-turn-4b-think
sbatch scripts/start.sh scripts/configs/config-oneturn-8b-nothink.yaml  outputs/one-turn-8b-nothink

echo "All jobs submitted."
echo ""
echo "Monitor with: squeue -u \$USER"
