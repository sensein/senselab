#!/bin/bash
#SBATCH --job-name=triage-r4-ou
#SBATCH --partition=ou_bcs_normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=40-139%128
#SBATCH --output=/orcd/scratch/bcs/002/satra/triage_r4_20260924/logs/%x-%A_%a.out
#SBATCH --chdir=/orcd/scratch/bcs/002/satra/triage_r4_20260924
# 62,548 recordings over 240 slices of ~261, the same manifest and the same sources r3 replayed.
# 240 and not the 400 r3 used: the first submission went to ou_bcs_normal, whose partition QOS caps
# submits at 256 whatever --qos says. That partition then ran 14 of 160 for twelve minutes against
# another user holding 230 jobs, so the array moved here, where r3 itself ran. %128 is the real
# ceiling: mit_preemptable MaxTRESPU is cpu=1024 and a task asks 8.
# The driver skips a store already carrying this configuration's replay marker, so a preempted
# slice is resumable by resubmission over the same roots.
set -euo pipefail
source /orcd/scratch/bcs/002/satra/triage_design_20260919/env_common.sh
export SENSELAB_REPO=/orcd/scratch/bcs/002/satra/senselab-r4
export TRIAGE_EXPECTED_COMMIT=84ebb3b5107b1176ef9f01f305156d578bf4ffb9
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
C=/orcd/scratch/bcs/002/satra/triage_design_20260919
R=/orcd/scratch/bcs/002/satra/triage_r4_20260924
M=/orcd/scratch/bcs/002/satra/triage_r3_20260923/replay_manifest.jsonl
sleep $(( (SLURM_ARRAY_TASK_ID * 7) % 120 ))
cd "$SENSELAB_REPO"
HEAD_SHA=$(git rev-parse HEAD)
if [ "$HEAD_SHA" != "$TRIAGE_EXPECTED_COMMIT" ]; then
  echo "FATAL: checkout is $HEAD_SHA, expected $TRIAGE_EXPECTED_COMMIT" >&2
  exit 74
fi
echo "=== host $(hostname) commit $HEAD_SHA slice ${SLURM_ARRAY_TASK_ID}/240 $(date -Is)"
# --hints is load-bearing: the hint is the one input the store does not hold, and routing, all
# three branches, REDACT and VERDICT read it. Without it every decision is made blind.
uv run --no-sync python -u scripts/extend_replay_decisions.py \
    "$M" \
    --slice-index "${SLURM_ARRAY_TASK_ID}" --slice-count 240 \
    --hints "$C/scope" \
    --out-root "$R/out" \
    --log-dir "$R/rows" \
    --commit "$HEAD_SHA"
echo "=== DONE $(date -Is)"
