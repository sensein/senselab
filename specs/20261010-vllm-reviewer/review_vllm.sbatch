#!/bin/bash
# REVIEW (prompt 14) on the vLLM engine over a corpus's rows. One H100 per array task holds SERVERS vLLM servers;
# each is started and fed one request at a time by its own extend_llm_review.py process over its own shard.
# Requeue-safe: a row whose store holds a reading under this prompt and engine is skipped (present).
#
#   BASE=<run directory holding rows.jsonl>
#   N=$(( ($(wc -l < "$BASE/rows.jsonl") + 299) / 300 ))      # ~300 rows a task
#   mkdir -p "$BASE/logs"
#   sbatch --array=0-$((N - 1))%16 --output="$BASE/logs/%x-%A_%a.out" \
#     --export=ALL,BASE=$BASE,SENSELAB_SHA=<40-hex pushed commit>,HINTS=<hints dir>,SLICES=$N,SERVERS=<servers per card> \
#     review_vllm.sbatch
#
# CHECKOUT (default $BASE/senselab-$SENSELAB_SHA) is a clone at that commit; PY is the interpreter it runs under.
# Followed by second_opinion.sbatch (afterok on this array), then scripts/extend_refold.py.

#SBATCH --job-name=review-p14-vllm
#SBATCH --partition=pi_satra,ou_bcs_low,mit_preemptable
#SBATCH --exclude=node2119,node2621,node3002,node1702
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --requeue

set -uo pipefail
S=/orcd/scratch/bcs/002/satra
BASE="${BASE:?BASE must be the run directory}"
SHA="${SENSELAB_SHA:?SENSELAB_SHA must be the pinned commit}"
SLICES="${SLICES:?SLICES must be the array size}"
HINTS="${HINTS:?HINTS must be the hints directory}"
SERVERS="${SERVERS:-2}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
CHECKOUT="${CHECKOUT:-$BASE/senselab-$SHA}"
PY="${PY:-$S/senselab-r27/.venv/bin/python}"
JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"

source "$S/triage_design_20260919/env_common.sh"
export SENSELAB_CACHE="$BASE/cache" SENSELAB_RUN_ID="slurm-$JOB" PYTHONPATH="$CHECKOUT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export VLLM_CACHE_ROOT="$BASE/cache/vllm" TRITON_CACHE_DIR="$BASE/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor" FLASHINFER_WORKSPACE_BASE="$BASE/cache/flashinfer"
export SENSELAB_VLLM_LOG_DIR="$BASE/logs/vllm"
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/13.1.0 2>/dev/null && export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
mkdir -p "$BASE/logs" "$BASE/cache"

cd "$CHECKOUT" || exit 2
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 2; }
DRIVER_CUDA=$(nvidia-smi | grep -oE "CUDA Version: [0-9.]+" | awk '{print $3}')
awk -v d="$DRIVER_CUDA" 'BEGIN{exit !(d+0 >= 13.0)}' || { echo "driver CUDA $DRIVER_CUDA < 13.0 on $(hostname)"; exit 3; }

UTIL=$(awk -v n="$SERVERS" 'BEGIN{printf "%.2f", int(90 / n) / 100}')
CFG="$BASE/review-vllm-$SERVERS.yaml"
cat > "$CFG.tmp.$TASK" <<EOF
redaction:
  llm_check:
    enabled: true
    keep_worker_resident: true
    ref: 52f3f65bc7a02d555763bc923bd1d9094898219d
    engine: vllm
    vllm:
      gpu_memory_utilization: $UTIL
EOF
mv -f "$CFG.tmp.$TASK" "$CFG"

echo "=== review task $TASK of $SLICES, $SERVERS server(s) at $UTIL on $(hostname), driver CUDA $DRIVER_CUDA, $(date -Is)"
pids=()
for i in $(seq 0 $((SERVERS - 1))); do
  "$PY" -u scripts/extend_llm_review.py "$BASE/rows.jsonl" \
    --slice-index $((TASK * SERVERS + i)) --slice-count $((SLICES * SERVERS)) \
    --config "$CFG" --hints "$HINTS" --log-dir "$BASE" --no-refold \
    > "$BASE/logs/client-$JOB-$TASK-$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
echo "=== exit $rc at $(date -Is)"
exit "$rc"
