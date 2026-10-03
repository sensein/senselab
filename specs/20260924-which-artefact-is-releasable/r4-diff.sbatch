#!/bin/bash
#SBATCH --job-name=r4-diff
#SBATCH --partition=ou_bcs_normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --array=0-63
#SBATCH --output=/orcd/scratch/bcs/002/satra/triage_r4_20260924/logs/%x-%A_%a.out
#SBATCH --chdir=/orcd/scratch/bcs/002/satra/triage_r4_20260924
# One differential row per recording, read out of the replayed store, which carries both the old
# decision and the new one with the invalidation edge between them. Read-only over both trees.
set -euo pipefail
source /orcd/scratch/bcs/002/satra/triage_design_20260919/env_common.sh
export SENSELAB_REPO=/orcd/scratch/bcs/002/satra/senselab-r4
export CUDA_VISIBLE_DEVICES=""
R=/orcd/scratch/bcs/002/satra/triage_r4_20260924
M=/orcd/scratch/bcs/002/satra/triage_r3_20260923/replay_manifest.jsonl
cd "$SENSELAB_REPO"
uv run --no-sync python -u scripts/triage_replay_diff.py rows \
    "$M" \
    --slice-index "${SLURM_ARRAY_TASK_ID}" --slice-count 64 \
    --out-root "$R/out" \
    --log-dir "$R/diff"
echo "=== DONE $(date -Is)"
