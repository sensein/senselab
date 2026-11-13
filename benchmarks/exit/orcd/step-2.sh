#!/bin/bash
# Step 2, ORCD part: `signal.resample` (target_rate 8000) over the synthetic BIDS tree (n=1000, seed 0), planned
# under `laptop` and under the site profile on this node (one LogicalPlan hash; a field diff only inside `placement`),
# then run with `Pool` on `slurm` under the site profile, its output identities compared with the laptop run's.
# Coordinator on a CPU node; its workers are Slurm jobs of the profile's `cpu` class. Submit (placeholders, P-28):
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=16G --time=01:30:00 \
#          --job-name=orcd-step2 --output="$WORK_ROOT/orcd-step2-%j.out" \
#          --export=ALL,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION",LAPTOP_IDENTITIES="$SCRIPTS_DIR/laptop-in-process.json",EXPECT_HASH=<laptop plan hash>[,WORKER_TIME_S=<s>,FFMPEG_LIB=...] \
#          step-2.sh
# LAPTOP_IDENTITIES is the laptop in-process run's exit-identities file (copy it into SCRIPTS_DIR); EXPECT_HASH is its
# `hash`. Both are optional; without them the ORCD run is checked against its own plan only. LAPTOP_CACHE is a copy of
# that run's cache root (`<work>/cache/resample-laptop-in-process`, its `results/` and `runs/`): an item whose identity
# differs passes when its arrays agree under the run config's `samples` threshold (atol 1e-6; owner ruling 2026-11-10).
# Expected wall: 10 to 25 minutes (1000 items at about 0.04 s each across up to 32 four-CPU workers; queue wait for
# the workers is the unknown; a cold uv sync is 3 to 6 minutes). Progress lines (`progress: N/1000 items committed
# after S s`) print every 60 s from the run's events. Results in $WORK_ROOT/results/orcd-2-<job>/: plan.log, run.log,
# orcd-pool.json (hash, counts, wall, watch, placement, identities), verdict.json.
# PASS: every line before the last is `PASS <check>` and the last is one JSON verdict with "pass": true, exit 0. The
# checks: one plan hash with differences only in placement; the run's hash equals the plan's and EXPECT_HASH; 1000
# items done, none failed, 1000 identities; outputs agree with the laptop's under the `samples` threshold.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
orcd_setup 2 cpu,cpu:large
W="$WORK/data"
mkdir -p "$W"
set +e
uv run --no-sync python benchmarks/exit/resample_identity.py plan --work "$W" --profile "$PROFILE" 2>&1 | tee "$OUT/plan.log"
PLAN=${PIPESTATUS[0]}
orcd_watched "$OUT/run.log" senselab- uv run --no-sync python benchmarks/exit/resample_identity.py run --work "$W" \
  --profile "$PROFILE" --executor pool-slurm --out "$OUT/orcd-pool.json"
RUN=$?
CACHE=$(ls -d "$W"/cache/resample-*-pool-slurm 2>/dev/null | head -1)
orcd_driver identity --out "$WORK/verdict.json" --plan-log "$OUT/plan.log" --plan-status "$PLAN" \
  --run-json "$OUT/orcd-pool.json" --run-status "$RUN" --items 1000 \
  ${EXPECT_HASH:+--expect-hash "$EXPECT_HASH"} ${LAPTOP_IDENTITIES:+--laptop "$LAPTOP_IDENTITIES"} \
  ${LAPTOP_CACHE:+--laptop-cache "$LAPTOP_CACHE"} ${CACHE:+--cache "$CACHE"}
exit $?
