#!/bin/bash
# Step 6 (P-1): a fresh agent with no repository access plans a call from `describe()` output and a dataset path;
# `sl.plan(...).run()` runs it. Two phases, with the agent dispatched by the controller between them:
#   PHASE=prepare  lays a small synthetic BIDS tree, writes $WORK_ROOT/step6/agent-input.json (the instructions, the
#                  dataset path and `describe` of every task) and prints its path; the controller gives the agent
#                  exactly that file's `instructions`, `describe` and `dataset`, and saves the JSON it answers with
#                  as $WORK_ROOT/step6/call.json.
#   PHASE=run      validates the call (task name, params, model, dataset), plans it over the dataset under the site
#                  profile and runs it with `Pool` on `slurm`; a refused call is written as a result and fails the
#                  step with the error's type and message.
# Submit (placeholders, P-28; both phases are small CPU jobs):
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=2 --mem=8G --time=00:20:00 --job-name=orcd-step6-prep \
#          --output="$WORK_ROOT/orcd-step6-prep-%j.out" --export=ALL,PHASE=prepare,SHA=<40-hex>,WORK_ROOT=...,SCRIPTS_DIR=...,PROFILE_SRC=...,CPU_PARTITION=...,GPU_PARTITION=... step-6.sh
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=16G --time=02:00:00 --job-name=orcd-step6-run \
#          --output="$WORK_ROOT/orcd-step6-run-%j.out" --export=ALL,PHASE=run,... step-6.sh
# The dataset is 20 synthetic tones (CPU-runnable by every audio task but those that load models; a model task is a valid
# answer and then needs the `gpu:h100` class: add GPU_PARTITION and WORKER_TIME_S that fit it). Expected wall: prepare
# 3 to 6 minutes; run 5 to 20 minutes for a signal task, up to 60 for a model task with a cold env.
# Results in $WORK_ROOT/results/orcd-6-<phase>-<job>/: agent-input.json (prepare); agent-result.json and run.log (run).
# PASS (prepare): agent-input.json written, `describe` holds every task. PASS (run): the call validated, planned, ran,
# every item has an identity.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
PHASE="${PHASE:?PHASE is prepare or run}"
case "$PHASE" in prepare | run) ;; *) echo "PHASE is prepare or run" >&2; exit 2 ;; esac
orcd_setup "6-$PHASE" ""
STATE="$WORK_ROOT/step6"
mkdir -p "$STATE"
set +e
if [ "$PHASE" = prepare ]; then
  rm -rf "$STATE/data" "$STATE/agent-input.json" "$STATE/call.json"
  orcd_driver tree --out "$STATE/data/synthetic-0-20" --items 20 2>&1 | tee "$OUT/prepare.log"
  uv run --no-sync python benchmarks/exit/agent_plan.py prepare --dataset "$STATE/data/synthetic-0-20" --out "$STATE/agent-input.json" 2>&1 | tee -a "$OUT/prepare.log"
  STATUS=${PIPESTATUS[0]}
  cp "$STATE/agent-input.json" "$OUT/" 2>/dev/null
  orcd_driver agent-prepared --out "$WORK/verdict.json" --input "$STATE/agent-input.json" --status "$STATUS"
  exit $?
fi
[ -f "$STATE/call.json" ] || { echo "no $STATE/call.json: save the agent's answer there" >&2; exit 2; }
orcd_watched "$OUT/run.log" "runs:$WORK/data/cache/agent-*-pool-slurm/2x/runs/*" uv run --no-sync python benchmarks/exit/agent_plan.py run --call "$STATE/call.json" \
  --work "$WORK/data" --profile "$PROFILE" --executor pool-slurm --out "$OUT/agent-result.json"
STATUS=$?
orcd_driver agent-ran --out "$WORK/verdict.json" --result "$OUT/agent-result.json" --status "$STATUS"
exit $?
