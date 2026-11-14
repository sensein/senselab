#!/bin/bash
# Step 5 (one host class per job): the whole `packages` suite, every tier, serial, at the head SHA, after a
# build of every shipped env lock. MODE=cpu runs on a CPU node, MODE=gpu on a typed H100 node (the `gpu` tests run only
# there). Tiers run one pytest each (logic, contract, limits, usecase, env; TIERS overrides, space-separated), so a
# failing tier does not hide the next and each tier has its own junit file. The model tiers read SENSELAB_TEST_PROFILE
# (this script sets it to the job's copy of the site profile) and build their envs under $WORK_ROOT/envs-<sha>. Never
# `-n`: every xdist worker would load the ML stack again.
# Submit (placeholders, P-28; one job per MODE):
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=8 --mem=48G --time=04:00:00 --job-name=orcd-step5-cpu \
#          --output="$WORK_ROOT/orcd-step5-cpu-%j.out" \
#          --export=ALL,MODE=cpu,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION"[,HF_TOKEN=...,FFMPEG_LIB=...] \
#          step-5.sh
#   sbatch --partition="$GPU_PARTITION" --gres=gpu:h100:1 --cpus-per-task=8 --mem=96G --time=04:00:00 --job-name=orcd-step5-gpu \
#          --output="$WORK_ROOT/orcd-step5-gpu-%j.out" --export=ALL,MODE=gpu,SHA=<40-hex>,... step-5.sh
# The 4 h ceiling of the group's partition is close to the expected wall: if the first job stops on its time limit,
# split it: TIERS="logic contract usecase" in one job and TIERS="limits env" (with the env builds) in another, and
# SKIP_ENVS=1 on the first. Expected wall: CPU 2 to 3.5 h (env builds cold: 5 envs, the vllm one the largest, 30 to 60
# minutes; limits+env on CPU measured 1476 s warm at 7e82d161; logic and contract a few minutes to tens of minutes;
# usecase unmeasured), GPU 1.5 to 3 h (limits+env measured 898 s warm). Progress: pytest -q prints a dot per test and
# the script a line per tier start and end.
# Results in $WORK_ROOT/results/orcd-5-<mode>-<job>/: envs.log, pytest-<tier>.log, junit-<tier>.xml, <tier>-verdict.json,
# envs-verdict.json, verdict.json (the combination).
# PASS: the last line is one JSON verdict with "pass": true over `envs` and every tier, exit 0; each tier: pytest
# exited 0, tests ran, none failed or errored. Skips are reported in the logs (-rs), not counted as failures.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
MODE="${MODE:?MODE is cpu or gpu}"
case "$MODE" in cpu | gpu) ;; *) echo "MODE is cpu or gpu" >&2; exit 2 ;; esac
if [ "$MODE" = gpu ]; then nvidia-smi -L | grep -q H100 || { echo "this allocation has no H100" >&2; exit 2; }; fi
orcd_setup "5-$MODE" "-"
TIERS="${TIERS:-logic contract limits usecase env}"
set +e
if [ "${SKIP_ENVS:-}" != 1 ]; then
  orcd_driver envs --profile "$PROFILE" --out "$OUT/envs-verdict.json" 2>&1 | tee "$OUT/envs.log"
  # The driver once crashed here and left no verdict, which the combined verdict did not notice: require the file.
  [ -s "$OUT/envs-verdict.json" ] || { echo "FAIL envs: the driver wrote no envs-verdict.json"; ENVS_MISSING=1; }
fi
for tier in $TIERS; do
  # A tier no test carries the marker of collects nothing and pytest exits 5; that is an empty tier (`usecase` has no
  # tests at fd513f0c), not a failed run, so it is reported and left out of the combined verdict.
  uv run --no-sync pytest packages -m "$tier" -p no:cacheprovider -q --collect-only >/dev/null 2>&1
  if [ $? -eq 5 ]; then echo "tier $tier: no test carries the marker, skipped"; continue; fi
  [ "$MODE" = gpu ] && nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed "s/^/tier $tier: gpu holder before start: /"
  echo "tier $tier: start $(date +%T)"
  uv run --no-sync pytest packages -m "$tier" -p no:cacheprovider -q -rs --durations=15 \
    --junitxml="$OUT/junit-$tier.xml" 2>&1 | tee "$OUT/pytest-$tier.log"
  STATUS=${PIPESTATUS[0]}
  echo "tier $tier: pytest exit $STATUS at $(date +%T)"
  orcd_driver suite --out "$OUT/$tier-verdict.json" --tier "$tier" --junit "$OUT/junit-$tier.xml" --status "$STATUS"
done
# On the GPU host a `gpu` test that skipped for want of a GPU means the allocation was not seen: that is a failure.
GPU_SKIPPED=0
if [ "$MODE" = gpu ] && grep -qs "needs a GPU" "$OUT"/pytest-*.log; then echo "FAIL gpu: a gpu test skipped for want of a GPU"; GPU_SKIPPED=1; fi
orcd_driver combine --out "$WORK/verdict.json" --dir "$OUT"
COMBINED=$?
[ "${ENVS_MISSING:-0}" = 0 ] && [ "$GPU_SKIPPED" = 0 ] || COMBINED=1
exit $COMBINED
