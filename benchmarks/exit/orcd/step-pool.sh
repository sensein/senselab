#!/bin/bash
# Step pool (route-B S1b): a senselab run on pydra's warm pool, at the head SHA. `speaker_embeddings` (speechbrain)
# over ITEMS synthetic recordings (default 200), run by `benchmarks/scale/one_load.py` on the site profile's
# `gpu:h100` class (`gres: gpu:h100:1`, `max_workers: 1`, its partition). Without LOCAL, this CPU job is the
# coordinator and the pool's `SlurmSubstrate` submits the H100 worker as its own job (`--gres=gpu:h100:1`, no QOS);
# with LOCAL=1, run in a one-H100 allocation, the worker is a process of that allocation: the one-GPU scale case.
# Submit (placeholders, P-28):
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=16G --time=02:00:00 \
#          --job-name=orcd-pool --output="$WORK_ROOT/orcd-pool-%j.out" \
#          --export=ALL,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION"[,HF_TOKEN=...,WORKER_TIME_S=<s>,ITEMS=<n>] \
#          step-pool.sh
#   sbatch --partition="$GPU_PARTITION" --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
#          --job-name=orcd-pool-local --output="$WORK_ROOT/orcd-pool-local-%j.out" \
#          --export=ALL,LOCAL=1,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION"[,HF_TOKEN=...,ITEMS=<n>] \
#          step-pool.sh
# The first run at a SHA builds the backend env (kept under $WORK_ROOT/envs-<sha>) and downloads the model.
# Results in $WORK_ROOT/results/orcd-pool-<job>/: run.log, one-load.json, verdict.json.
# PASS: the last line is one JSON verdict with "pass": true, exit 0: every job done, one worker, one residency, one
# cold call (the load) and ITEMS-1 warm ones, read from the pool facts of each store entry.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
orcd_setup "pool${LOCAL:+-local}" "gpu:h100"
AUDIO="$WORK/audio"
ITEMS="${ITEMS:-200}"
uv run --no-sync python - "$AUDIO" "$ITEMS" <<'EOF'
import sys
from pathlib import Path

import numpy
import soundfile

root, items = Path(sys.argv[1]), int(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
for i in range(items):
    seconds = 1.0 + (i % 4)
    t = numpy.arange(int(16000 * seconds)) / 16000
    soundfile.write(root / f"r{i:05d}.wav", 0.1 * numpy.sin(2 * numpy.pi * (110 + 7 * i) * t), 16000)
EOF
set +e
uv run --no-sync python benchmarks/scale/one_load.py --profile "$PROFILE" --audio "$AUDIO" --items "$ITEMS" \
  --task speaker_embeddings --stack speechbrain ${LOCAL:+--local} --out "$OUT/one-load.json" 2>&1 | tee "$OUT/run.log"
cp "$OUT/one-load.json" "$WORK/verdict.json" 2>/dev/null
tail -n 1 "$WORK/verdict.json"
grep -q '"pass": true' "$WORK/verdict.json"
