#!/bin/bash
#SBATCH --job-name=yamnet-cost
#SBATCH --partition=mit_preemptable
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --output=/orcd/scratch/bcs/002/satra/yamnetperf/logs/%x-%j.out
set -euo pipefail
source /orcd/scratch/bcs/002/satra/triage_design_20260919/env_common.sh
export CUDA_VISIBLE_DEVICES=""
W=/orcd/scratch/bcs/002/satra/yamnetperf
YP=/orcd/scratch/bcs/002/satra/senselab-venvs/yamnet/bin/python
cd "$W"
echo "=== host=$(hostname) exclusive=yes"
echo "=== loadavg_start=$(cat /proc/loadavg)"
echo "=== nproc=$(nproc)"
"$YP" -u probe_yamnet_cost.py "$YP" > "$W/cost-${SLURM_JOB_ID}.jsonl"
echo "=== loadavg_end=$(cat /proc/loadavg)"
echo "=== rows=$(wc -l < "$W/cost-${SLURM_JOB_ID}.jsonl")"
