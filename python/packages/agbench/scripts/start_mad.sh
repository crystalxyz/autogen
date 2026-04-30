#!/bin/bash
# sbatch wrapper for HotpotQA Multi-Agent Debate with diverse models.
# This config defines 5 sglang servers (4 debaters + judge) on devices 0..4 —
# so the job needs 5 GPUs.
#
# Usage:
#   sbatch scripts/start_mad.sh scripts/configs/config-hotpotqa-mad-diverse.yaml outputs/hotpotqa-mad-diverse
#
# To run interactively (e.g. on an existing 5-GPU allocation):
#   bash scripts/start_mad.sh scripts/configs/config-hotpotqa-mad-diverse.yaml outputs/hotpotqa-mad-diverse

#SBATCH --job-name=agbench-mad
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --partition=gupta
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:5

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate sglang-env

rm -rf "$2"

# sbatch copies this script to /var/spool/slurmd, so $BASH_SOURCE no longer
# points at the repo. Use SLURM_SUBMIT_DIR (the directory you ran sbatch from)
# to locate the agbench package, falling back to PWD for interactive use.
PKG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PKG_DIR"

if [ -f "$PKG_DIR/.env" ]; then
    set -a; source "$PKG_DIR/.env"; set +a
fi
export HF_TOKEN="${HF_TOKEN:?Error: HF_TOKEN not set. Create python/packages/agbench/.env with HF_TOKEN=...}"

python scripts/runners/run_bench.py --config "$1" --work-dir "$2"
