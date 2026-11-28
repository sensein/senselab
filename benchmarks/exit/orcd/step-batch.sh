#!/bin/bash
# Step batch: per-item time of one GPU backend's worker at explicit request sizes and under pydra's batch probe, on
# one typed H100, at the head SHA (`benchmarks/scale/batch_sizes.py`). TASK/BACKEND default to speaker_embeddings /
# speechbrain; ITEMS (default 256) records per size, SIZES (default 1,2,4,8,16,32,64), DURATIONS (record seconds,
# default 1,2,3,4).
# Submit after step-prepare.sh (placeholders, P-28):
#   sbatch --dependency=afterok:<prepare job> --partition="$GPU_PARTITION" --gres=gpu:h100:1 --cpus-per-task=8 \
#          --mem=64G --time=01:30:00 --job-name=orcd-batch --output="$WORK_ROOT/orcd-batch-%j.out" \
#          --export=ALL,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC"[,HF_TOKEN,TASK,BACKEND,ITEMS,SIZES,DURATIONS] \
#          step-batch.sh
# The first run at a SHA builds the backend env (kept under $WORK_ROOT/envs-<sha>) and downloads the model.
# Results in $WORK_ROOT/results/orcd-batch-<job>/: run.log, batch-sizes.json, verdict.json.
# PASS: the last line is one JSON verdict with "pass": true, exit 0: every output at every size equals its size-1
# output. The timings are the measurement; they are not judged.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
nvidia-smi -L | grep -q H100 || { echo "this allocation has no H100" >&2; exit 2; }
orcd_setup batch "gpu:h100" local
export SENSELAB_CACHE="$WORK/cache"
set +e
uv run --no-sync python benchmarks/scale/batch_sizes.py --profile "$PROFILE" --items "${ITEMS:-256}" \
  --sizes "${SIZES:-1,2,4,8,16,32,64}" --seconds "${DURATIONS:-1,2,3,4}" \
  --task "${TASK:-speaker_embeddings}" --backend "${BACKEND:-speechbrain}" --out "$OUT/batch-sizes.json" \
  2>&1 | tee "$OUT/run.log"
cp "$OUT/batch-sizes.json" "$WORK/verdict.json" 2>/dev/null
tail -n 1 "$WORK/verdict.json"
grep -q '"pass": true' "$WORK/verdict.json"
