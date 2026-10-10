#!/bin/bash
# REVIEW at prompt 12 / review_inputs 3 over the r18c rows select_rows.py kept. Prepared, not submitted.
#
#   N=$(( ($(wc -l < "$BASE/rows.jsonl") + 99) / 100 ))      # ~100 rows a task
#   SENSELAB_SHA=<40-hex pushed commit> R18C=<r18c corpus root> HINTS=<hints dir> \
#     sbatch --array=0-$((N - 1))%16 --export=ALL,SLICES=$N review.sbatch
#
# Followed by second_opinion.sbatch (afterok on this array), then scripts/extend_refold.py.

#SBATCH --job-name=r18c-review-p12
#SBATCH --partition=pi_satra,ou_bcs_low,mit_preemptable
#SBATCH --exclude=node2119,node2621,node3002,node1702
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/scratch/bcs/002/satra/r18c_review_p12/logs/%x-%A_%a.out

set -uo pipefail
BASE=/orcd/scratch/bcs/002/satra/r18c_review_p12
SHA="${SENSELAB_SHA:?SENSELAB_SHA must be the pinned commit}"
SLICES="${SLICES:?SLICES must be the array size}"
HINTS="${HINTS:?HINTS must be the hints directory}"
CHECKOUT="$BASE/senselab-$SHA"

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/orcd/scratch/bcs/002/satra/miniforge/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/orcd/data/satra/002/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export SENSELAB_RUN_ID="slurm-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
export SENSELAB_CACHE="$BASE/cache"
export OMP_NUM_THREADS=4

cd "$CHECKOUT" || exit 2
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 2; }

echo "=== review task ${SLURM_ARRAY_TASK_ID} of $SLICES on $(hostname) at $(date -Is)"
uv run --no-sync python scripts/extend_llm_review.py "$BASE/rows.jsonl" \
  --slice-index "$SLURM_ARRAY_TASK_ID" --slice-count "$SLICES" \
  --config "$BASE/review.yaml" --hints "$HINTS" --log-dir "$BASE" --no-refold
rc=$?
echo "=== exit $rc at $(date -Is)"
exit "$rc"
