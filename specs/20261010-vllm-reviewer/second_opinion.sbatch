#!/bin/bash
# SECOND_OPINION (question set 6, clef:27b on the pinned Ollama) over the same rows, after review_vllm.sbatch.
# One H100 per array task, one Ollama server per task answering second_opinion.workers rows at once; the array
# spreads the rows over GPUs. Requeue-safe: a store already holding an ok opinion over its live annotation is present.
#
#   N=$(( ($(wc -l < "$BASE/rows.jsonl") + 399) / 400 ))      # ~400 rows a task
#   sbatch --dependency=afterok:<review array job id> --array=0-$((N - 1))%8 \
#     --output="$BASE/logs/%x-%A_%a.out" \
#     --export=ALL,BASE=$BASE,SENSELAB_SHA=<same commit as the review>,HINTS=<hints dir>,SLICES=$N \
#     second_opinion.sbatch
#
# A row whose store has no REVIEW annotation is an error row; rerun its review slice, then this one.

#SBATCH --job-name=opinion-q6
#SBATCH --partition=pi_satra,ou_bcs_low,mit_preemptable
#SBATCH --exclude=node2119,node2621,node3002,node1702
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue

set -uo pipefail
S=/orcd/scratch/bcs/002/satra
BASE="${BASE:?BASE must be the run directory}"
SHA="${SENSELAB_SHA:?SENSELAB_SHA must be the pinned commit}"
SLICES="${SLICES:?SLICES must be the array size}"
HINTS="${HINTS:?HINTS must be the hints directory}"
CHECKOUT="${CHECKOUT:-$BASE/senselab-$SHA}"
PY="${PY:-$S/senselab-r27/.venv/bin/python}"
OLLAMA="${OLLAMA:-$S/ollama-models-flash/ollama-0.35.1/bin/ollama}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$S/ollama-models-flash/models}"
VERIFIED="${VERIFIED:-/orcd/data/satra/002/models/ollama-senselab/verified}"

source "$S/triage_design_20260919/env_common.sh"
export SENSELAB_CACHE="$BASE/cache" SENSELAB_RUN_ID="slurm-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
export PYTHONPATH="$CHECKOUT/src" HF_HUB_OFFLINE=1 OLLAMA_VULKAN=0 OMP_NUM_THREADS=8

cd "$CHECKOUT" || exit 2
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 2; }
cat > "$BASE/second_opinion.yaml.tmp.${SLURM_ARRAY_TASK_ID:-0}" <<'EOF'
redaction:
  llm_check:
    enabled: true
    ref: 52f3f65bc7a02d555763bc923bd1d9094898219d
second_opinion:
  enabled: true
EOF
mv -f "$BASE/second_opinion.yaml.tmp.${SLURM_ARRAY_TASK_ID:-0}" "$BASE/second_opinion.yaml"

echo "=== opinion task ${SLURM_ARRAY_TASK_ID:-0} of $SLICES on $(hostname) at $(date -Is)"
"$PY" -u scripts/extend_second_opinion.py "$BASE/rows.jsonl" \
  --slice-index "${SLURM_ARRAY_TASK_ID:-0}" --slice-count "$SLICES" \
  --config "$BASE/second_opinion.yaml" --hints "$HINTS" --log-dir "$BASE/opinion" \
  --ollama-binary "$OLLAMA" --ollama-models "$OLLAMA_MODELS_DIR" --verified-dir "$VERIFIED"
rc=$?
echo "=== exit $rc at $(date -Is)"
exit "$rc"
