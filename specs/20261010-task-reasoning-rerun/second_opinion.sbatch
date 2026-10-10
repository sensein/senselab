#!/bin/bash
# SECOND_OPINION at question set 4 over the same rows, after review.sbatch. Prepared, not submitted.
#
#   N=$(( ($(wc -l < "$BASE/rows.jsonl") + 399) / 400 ))     # ~400 rows a task
#   SENSELAB_SHA=<same commit as the review> HINTS=<hints dir> \
#     sbatch --dependency=afterok:<review array job id> --array=0-$((N - 1))%8 \
#       --export=ALL,SLICES=$N second_opinion.sbatch
#
# A row whose store has no REVIEW annotation is an error row; rerun the review slice, then this one.

#SBATCH --job-name=r18c-opinion-q4
#SBATCH --partition=pi_satra,ou_bcs_low,mit_preemptable
#SBATCH --exclude=node2119,node2621,node3002,node1702
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/scratch/bcs/002/satra/r18c_review_p12/logs/%x-%A_%a.out

set -uo pipefail
BASE=/orcd/scratch/bcs/002/satra/r18c_review_p12
SHA="${SENSELAB_SHA:?SENSELAB_SHA must be the pinned commit}"
SLICES="${SLICES:?SLICES must be the array size}"
HINTS="${HINTS:?HINTS must be the hints directory}"
CHECKOUT="$BASE/senselab-$SHA"
OLLAMA_ROOT=/orcd/data/satra/002/models/ollama-senselab

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/orcd/scratch/bcs/002/satra/miniforge/lib:${LD_LIBRARY_PATH:-}"
export SENSELAB_RUN_ID="slurm-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
export SENSELAB_CACHE="$BASE/cache"
export OMP_NUM_THREADS=8

cd "$CHECKOUT" || exit 2
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 2; }

echo "=== opinion task ${SLURM_ARRAY_TASK_ID} of $SLICES on $(hostname) at $(date -Is)"
uv run --no-sync python scripts/extend_second_opinion.py "$BASE/rows.jsonl" \
  --slice-index "$SLURM_ARRAY_TASK_ID" --slice-count "$SLICES" \
  --config "$BASE/second_opinion.yaml" --hints "$HINTS" --log-dir "$BASE/opinion" \
  --ollama-binary "$OLLAMA_ROOT/bin/ollama" \
  --ollama-models /orcd/scratch/bcs/002/satra/ollama-models-flash
rc=$?
echo "=== exit $rc at $(date -Is)"
exit "$rc"
