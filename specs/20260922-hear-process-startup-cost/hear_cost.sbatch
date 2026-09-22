#!/bin/bash
#SBATCH --job-name=hear-cost
#SBATCH --partition=mit_preemptable
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/orcd/scratch/bcs/002/satra/hearperf/logs/%x-%j.out
set -euo pipefail
source /orcd/scratch/bcs/002/satra/triage_design_20260919/env_common.sh
export SENSELAB_REPO=/orcd/scratch/bcs/002/satra/senselab-yamnetnew
export CUDA_VISIBLE_DEVICES=""
W=/orcd/scratch/bcs/002/satra/hearperf
cd "$SENSELAB_REPO"
echo "=== host=$(hostname) exclusive=yes nproc=$(nproc)"
echo "=== loadavg_start=$(cat /proc/loadavg)"
git log --oneline -1
uv run --no-sync python -u "$W/probe_hear_cost.py" > "$W/cost-${SLURM_JOB_ID}.jsonl"
echo "=== loadavg_end=$(cat /proc/loadavg)"
echo "=== rows=$(wc -l < "$W/cost-${SLURM_JOB_ID}.jsonl")"
