#!/bin/bash
# Step 3: `SpeechToText` over the first 1000 LibriSpeech test-clean utterances (a BIDS tree), planned under
# `laptop` and under the site profile on this CPU node (one hash, differences only in placement), then run under the
# site profile with `Pool` on `slurm`; the profile places the node on its `gpu:h100` class. The coordinator is a CPU
# job; the workers are H100 Slurm jobs. Submit (placeholders, P-28):
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=32G --time=03:30:00 \
#          --job-name=orcd-step3 --output="$WORK_ROOT/orcd-step3-%j.out" \
#          --export=ALL,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION"[,HF_TOKEN=...,WORKER_TIME_S=<s>,FFMPEG_LIB=...] \
#          step-3.sh
# GPU_PARTITION must be a partition with a typed H100 on a healthy node; WORKER_TIME_S must
# fit that partition's time limit; the time fit below refuses before the run when the
# `gpu:h100` class asks more. Planning resolves the model ref (network, HF cache under $WORK_ROOT/hf); the first
# run at a SHA builds the backend env (10 to 20 minutes, kept under $WORK_ROOT/envs-<sha>).
# Expected wall: 60 to 120 minutes (the dataset fetch, a cold env build, model load, 1000 utterances at the batch the
# H100 measure gives, up to 4 workers); progress lines print every 60 s. Results in $WORK_ROOT/results/orcd-3-<job>/:
# plan.log, run.log, orcd-pool.json, verdict.json.
# PASS: the last line is one JSON verdict with "pass": true, exit 0. Checks: one plan hash with differences only in
# placement; the run's hash equals the plan's; 1000 items done, none failed, 1000 identities; the node placed on a
# `gpu:h100` class.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
orcd_setup 3 ""
W="$WORK/data"
mkdir -p "$W"
set +e
uv run --no-sync python benchmarks/exit/stt_identity.py plan --work "$W" --profile "$PROFILE" 2>&1 | tee "$OUT/plan.log"
PLAN=${PIPESTATUS[0]}
orcd_watched "$OUT/run.log" "runs:$W/cache/stt-*-pool-slurm/2x/runs/*" uv run --no-sync python benchmarks/exit/stt_identity.py run --work "$W" \
  --profile "$PROFILE" --executor pool-slurm --out "$OUT/orcd-pool.json"
RUN=$?
orcd_driver identity --out "$WORK/verdict.json" --plan-log "$OUT/plan.log" --plan-status "$PLAN" \
  --run-json "$OUT/orcd-pool.json" --run-status "$RUN" --items 1000 --require-class gpu:h100
exit $?
