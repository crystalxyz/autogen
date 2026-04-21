#!/bin/bash

#SBATCH --job-name=agbench-4turn
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=99:00:00
#SBATCH --partition=gupta
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:4

# Initialize conda for the shell
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate sglang-env

rm -rf "$2"

# Load environment variables (HF_TOKEN, etc.)
if [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]; then
    set -a; source "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi
export HF_TOKEN="${HF_TOKEN:?Error: HF_TOKEN not set. Create python/packages/agbench/.env with HF_TOKEN=...}"

python scripts/runners/run_bench.py --config "$1" --work-dir "$2"
