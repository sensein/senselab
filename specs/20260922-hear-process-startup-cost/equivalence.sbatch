#!/bin/bash
#SBATCH --job-name=hperf-equiv
#SBATCH --partition=mit_preemptable
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=01:30:00
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
uv run --no-sync python -u "$W/probe_equivalence.py" \
    "$W/steps_manifest.jsonl" fb3cb7a2b974721f3e17c35c369b8413e39d655f \
    > "$W/equiv-${SLURM_JOB_ID}.jsonl"
echo "=== loadavg_end=$(cat /proc/loadavg)"
cat "$W/equiv-${SLURM_JOB_ID}.jsonl"
