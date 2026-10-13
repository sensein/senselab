#!/bin/bash
# One slice of one r36 lane, run by a slice_pool worker:  bash r36_slice.sh LANE I N
# (pool.json: command = bash $W/r36_slice.sh LANE {i} {n}). Every driver resumes per recording, so a retried or
# taken-over slice redoes only what its earlier attempt did not finish. The lane variable is LANE, never STAGE
# (install_r34.py reads STAGE).
#   replay      extend_replay_decisions over manifest_replay (packaged config, reviewer off): exit 3 = needs_reread = success
#   bypass      REVIEW + SECOND_OPINION recorded without a model over review_nowords/rows.jsonl (CPU)
#   review      vLLM REVIEW over review/rows.jsonl (GPU, one server, one request at a time; --no-refold)
#   opinion     SECOND_OPINION (pinned Ollama) over review/rows.jsonl (GPU)
#   cohort      COHORT apply by participant (cohort_stage v4) over manifest_r36
#   refold      VERDICT re-decided over manifest_r36
#   task_audio  task-extent cuts over manifest_r36
#   rvec        recording vectors over $OUT into $RV
set -uo pipefail
LANE=${1:?lane}; I=${2:?slice index}; N=${3:?slice count}
case "$LANE" in review|opinion) export R36_GPU=1 ;; esac
source /orcd/scratch/bcs/002/satra/r18d_20261010/common_r36.sh
export SENSELAB_RUN_ID="slurm-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
echo "=== host $(hostname) commit $HEAD_SHA $LANE slice $I/$N attempt ${POOL_ATTEMPT:-?} $(date -Is)"
case "$LANE" in
replay)
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  $PY -u scripts/extend_replay_decisions.py $REPLAY_MANIFEST --slice-index "$I" --slice-count "$N" \
    --hints $HINTS --log-dir $R36/rows_replay --commit "$HEAD_SHA"
  rc=$?
  [ $rc -eq 3 ] && echo "=== NEEDS_REREAD (exit 3): review readings to re-read for some stems"
  ;;
bypass)
  NB=$R36/review_nowords
  export SENSELAB_CACHE=$NB/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
  export SENSELAB_VLLM_LOG_DIR="$NB/logs/vllm-$I-${POOL_ATTEMPT:-0}"
  $PY -u scripts/extend_llm_review.py $NB/rows.jsonl --slice-index "$I" --slice-count "$N" \
    --config "$CONFIG" --hints "$HINTS" --log-dir "$NB" --no-refold
  rc_review=$?
  $PY -u scripts/extend_second_opinion.py $NB/rows.jsonl --slice-index "$I" --slice-count "$N" \
    --config "$CONFIG" --hints "$HINTS" --log-dir "$NB/opinion" \
    --ollama-binary "$S/ollama-models-flash/ollama-0.35.1/bin/ollama" --ollama-models "$S/ollama-models-flash/models" \
    --verified-dir /orcd/data/satra/002/models/ollama-senselab/verified
  rc_opinion=$?
  if [ -e "$SENSELAB_VLLM_LOG_DIR" ] && [ -n "$(ls -A "$SENSELAB_VLLM_LOG_DIR" 2>/dev/null)" ]; then
    echo "SERVER: a vLLM server started for a no-words row: $SENSELAB_VLLM_LOG_DIR"; rc_review=9
  fi
  echo "=== review exit $rc_review opinion exit $rc_opinion"
  rc=$(( rc_review != 0 || rc_opinion != 0 ))
  ;;
review)
  B=$R36/review
  export SENSELAB_CACHE="$B/cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
  export VLLM_CACHE_ROOT="$B/cache/vllm" TRITON_CACHE_DIR="$B/cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$B/cache/inductor" FLASHINFER_WORKSPACE_BASE="$B/cache/flashinfer"
  export SENSELAB_VLLM_LOG_DIR="$B/logs/vllm-$I-${POOL_ATTEMPT:-0}"
  source /etc/profile.d/modules.sh 2>/dev/null
  module load cuda/13.1.0 2>/dev/null && export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  CFG=$B/review-vllm-1.yaml.$I.$$
  cat > "$CFG" <<YAML
redaction:
  llm_check:
    enabled: true
    keep_worker_resident: true
    ref: 52f3f65bc7a02d555763bc923bd1d9094898219d
    engine: vllm
    vllm:
      gpu_memory_utilization: 0.90
YAML
  $PY -u scripts/extend_llm_review.py $B/rows.jsonl --slice-index "$I" --slice-count "$N" \
    --config "$CFG" --hints "$HINTS" --log-dir "$B" --no-refold
  rc=$?
  rm -f "$CFG"
  ;;
opinion)
  B=$R36/review
  export SENSELAB_CACHE="$B/cache" HF_HUB_OFFLINE=1 OLLAMA_VULKAN=0 OMP_NUM_THREADS=8
  CFG=$B/second_opinion.yaml.$I.$$
  cat > "$CFG" <<YAML
redaction:
  llm_check:
    enabled: true
    ref: 52f3f65bc7a02d555763bc923bd1d9094898219d
second_opinion:
  enabled: true
YAML
  $PY -u scripts/extend_second_opinion.py $B/rows.jsonl --slice-index "$I" --slice-count "$N" \
    --config "$CFG" --hints "$HINTS" --log-dir "$B/opinion" \
    --ollama-binary "$S/ollama-models-flash/ollama-0.35.1/bin/ollama" --ollama-models "$S/ollama-models-flash/models" \
    --verified-dir /orcd/data/satra/002/models/ollama-senselab/verified
  rc=$?
  rm -f "$CFG"
  ;;
cohort)
  export OMP_NUM_THREADS=4
  test -s $R36/cohort/cohort_quantiles.json || { echo "FATAL: no r36 cohort quantiles (a copy of r34's)" >&2; exit 2; }
  $PY -u scripts/extend_cohort.py apply $MANIFEST --quantiles $R36/cohort/cohort_quantiles.json \
    --slice-index "$I" --slice-count "$N" --config $CONFIG --commit "$HEAD_SHA" \
    --log-dir $R36/rows_cohort --device cpu
  rc=$?
  ;;
refold)
  export OMP_NUM_THREADS=2
  $PY -u scripts/extend_refold.py $MANIFEST --slice-index "$I" --slice-count "$N" \
    --hints $HINTS --config $CONFIG --log-dir $R36/rows_refold --commit "$HEAD_SHA"
  rc=$?
  ;;
task_audio)
  export OMP_NUM_THREADS=2
  $PY -u scripts/extend_task_audio.py $MANIFEST --slice-index "$I" --slice-count "$N" \
    --config $CONFIG --log-dir $R36/rows_task_audio
  rc=$?
  ;;
rvec)
  umask 077
  mkdir -p $RV
  $PY scripts/triage_recording_vectors.py $OUT --out $RV --slice "$I" --slices "$N"
  rc=$?
  ;;
*) echo "unknown lane $LANE" >&2; exit 2 ;;
esac
echo "=== DONE $LANE slice $I/$N rc=$rc $(date -Is)"
exit $rc
