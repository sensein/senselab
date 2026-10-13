#!/bin/bash
#SBATCH --job-name=yperf-steps
#SBATCH --partition=mit_preemptable
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --output=/orcd/scratch/bcs/002/satra/yamnetperf/logs/%x-%j.out
set -euo pipefail
source /orcd/scratch/bcs/002/satra/triage_design_20260919/env_common.sh
export SENSELAB_REPO=/orcd/scratch/bcs/002/satra/senselab-yamnetperf
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
W=/orcd/scratch/bcs/002/satra/yamnetperf
cd "$SENSELAB_REPO"
echo "=== host=$(hostname) exclusive=yes nproc=$(nproc)"
echo "=== loadavg_start=$(cat /proc/loadavg)"
git log --oneline -1
uv run --no-sync python -u "$W/probe_preprocess_steps.py" \
    "$W/steps_manifest.jsonl" "$W/steps_out_${SLURM_JOB_ID}" 9 \
    > "$W/steps-${SLURM_JOB_ID}.jsonl"
echo "=== loadavg_end=$(cat /proc/loadavg)"
echo "=== rows=$(wc -l < "$W/steps-${SLURM_JOB_ID}.jsonl")"
