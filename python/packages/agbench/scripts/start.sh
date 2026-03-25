#!/bin/bash

#SBATCH --job-name=agbench
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=99:00:00
#SBATCH --partition=gupta
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:2

# Initialize conda for the shell
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate sglang-env

rm -rf "$2"

# Load environment variables (HF_TOKEN, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi
export HF_TOKEN="${HF_TOKEN:?Error: HF_TOKEN not set. Create python/packages/agbench/.env with HF_TOKEN=...}"

python scripts/run_bench.py --config "$1" --work-dir "$2"

# --nodelist=yosemite