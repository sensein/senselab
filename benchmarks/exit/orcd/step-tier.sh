#!/bin/bash
# Step tier: one pytest over the execution suites at the head SHA, its pool jobs on a local substrate (processes of this
# allocation). TIER=cpu, on a CPU node: the logic, contract and limits tests that load no model, at most 120 s per test.
# TIER=gpu, on a typed H100 node: the gpu and model tests of the execution, pii_spans and api suites, at most 300 s per
# test, after `orcd_driver envs` builds every shipped env, whose verdict is part of the suite verdict. A test with its
# own bound keeps it: every test that starts a vLLM server carries `vllm_server(starts=n)`, which conftest turns into
# n * (vllm.startup_s + envs.server_start_margin_s) of the site profile. NO_PROGRESS_S (gpu default 240) is the copy's pool.no_progress_s, inside the per-test bound. Never `-n`.
# Submit after step-prepare.sh (placeholders, P-28):
#   sbatch --dependency=afterok:<prepare job> --partition="$CPU_PARTITION" --cpus-per-task=8 --mem=32G --time=03:00:00 \
#          --job-name=orcd-tier-cpu --output="$WORK_ROOT/orcd-tier-cpu-%j.out" \
#          --export=ALL,TIER=cpu,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC" \
#          step-tier.sh
#   sbatch --dependency=afterok:<prepare job> --partition="$GPU_PARTITION" --gres=gpu:h100:1 --cpus-per-task=8 --mem=96G \
#          --time=04:00:00 --job-name=orcd-tier-gpu --output="$WORK_ROOT/orcd-tier-gpu-%j.out" \
#          --export=ALL,TIER=gpu,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",HF_TOKEN \
#          step-tier.sh
# Results in $WORK_ROOT/results/orcd-tier-<tier>-<job>/: pytest.log, junit.xml, envs.log and envs-verdict.json (gpu).
# PASS: the last line is one JSON verdict with "pass": true, exit 0: pytest exited 0, tests ran, none failed or errored,
# and (gpu) every env built and imported.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
TIER="${TIER:?TIER is cpu or gpu}"
case "$TIER" in
  cpu)
    SUITES=(packages/senselab-core/tests/runtime packages/senselab-core/tests/lint packages/senselab-core/tests/testing
      packages/senselab/tests/api_test.py packages/senselab/tests/facade_test.py packages/senselab/tests/reference
      packages/senselab/tests/bench packages/senselab/tests/text/tasks/pii_spans)
    MARKS="(logic or contract or limits) and not gpu and not model and not large_model"
    TIMEOUT=120
    orcd_setup tier-cpu "" local
    ;;
  gpu)
    nvidia-smi -L | grep -q H100 || { echo "this allocation has no H100" >&2; exit 2; }
    SUITES=(packages/senselab-core/tests/runtime/execution packages/senselab/tests/text/tasks/pii_spans
      packages/senselab/tests/api_test.py)
    MARKS="gpu or model or large_model"
    TIMEOUT=300
    export NO_PROGRESS_S="${NO_PROGRESS_S:-240}"
    orcd_setup tier-gpu "gpu:h100,svc:h100" local
    ;;
  *) echo "TIER is cpu or gpu" >&2; exit 2 ;;
esac
export SENSELAB_CACHE="$WORK/cache"
set +e
ENVS=()
if [ "$TIER" = gpu ]; then
  orcd_driver envs --profile "$PROFILE" --out "$OUT/envs-verdict.json" 2>&1 | tee "$OUT/envs.log"
  ENVS=(--envs "$OUT/envs-verdict.json")
fi
uv run --no-sync pytest "${SUITES[@]}" -m "$MARKS" -p no:cacheprovider -q -rfEsx --durations=25 --timeout="$TIMEOUT" \
  --basetemp="$WORK/pytest-$TIER" -o log_cli=true --log-cli-level=WARNING \
  --junitxml="$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log"
STATUS=${PIPESTATUS[0]}
orcd_driver suite --out "$WORK/verdict.json" --tier "$TIER" --junit "$OUT/junit.xml" --status "$STATUS" "${ENVS[@]}"
