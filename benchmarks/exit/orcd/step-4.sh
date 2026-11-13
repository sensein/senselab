#!/bin/bash
# Step 4: interrupt Step 3's run at a third of its items, then resume it. Two jobs over one run directory
# ($WORK_ROOT/step4, shared between them):
#   PHASE=start   plan and run `SpeechToText` (1000 utterances) with `Pool` on `slurm` in a child process; once a third
#                 of the items have a result event, cancel every Slurm job named by the run's prefix, SIGKILL the
#                 child (the coordinator), sweep what it submitted meanwhile, write start.json and exit 0. The
#                 interrupt is made by the driver (`interrupt_resume.py start`), not by a controller.
#   PHASE=resume  in a fresh Slurm job (its attempt id is its job id): `senselab.resume(<hash>)`; asserts executions
#                 are the items not committed before the interrupt (late hits reported apart), reuses are the rest,
#                 no committed item re-executed, and no model ref resolved.
# Submit start, then resume behind it (placeholders, P-28):
#   EXP=ALL,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION"
#   S=$(sbatch --parsable --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=32G --time=02:30:00 --job-name=orcd-step4-start \
#          --output="$WORK_ROOT/orcd-step4-start-%j.out" --export="$EXP,PHASE=start" step-4.sh)
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=32G --time=03:00:00 --job-name=orcd-step4-resume \
#          --dependency=afterany:$S --output="$WORK_ROOT/orcd-step4-resume-%j.out" --export="$EXP,PHASE=resume" step-4.sh
# Add WORKER_TIME_S=<s> (fits the GPU partition) and HF_TOKEN / FFMPEG_LIB as for Step 3. Run it at the same SHA as
# Step 3 so the backend env is built already. PHASE=start refuses an existing $WORK_ROOT/step4 (a second interrupt of
# the same plan would find results already stored); set FRESH=1 to remove it.
# Expected wall: start 25 to 60 minutes (model load, then about a third of the items; the interrupt itself is
# seconds); resume 40 to 80 minutes. Progress prints every 60 s in both. Results: $WORK_ROOT/results/orcd-4-<job>/
# start.json (hash, items, committed, cancelled jobs), resume.json (items, committed, executions, late, reuses),
# verdict.json.
# Both jobs run under the no-progress watchdog (lib.sh orcd_watched, STALL_S default 1800 s): exit 3 with a failing
# verdict when committed is unchanged that long and no worker job is in squeue.
# PASS: the last line of each job is one JSON verdict with "pass": true, exit 0. Start: the hash; committed at or past a
# third and below all. Resume: executions + late = items - committed; reuses = items - executions; committed at least
# the interrupt's; same hash. `account_resume` and `assert_no_resolution` raise otherwise, which fails the check.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
PHASE="${PHASE:?PHASE is start or resume}"
case "$PHASE" in start | resume) ;; *) echo "PHASE is start or resume" >&2; exit 2 ;; esac
orcd_setup "4-$PHASE" ""
STATE="$WORK_ROOT/step4"
if [ "$PHASE" = start ]; then
  [ "${FRESH:-}" = 1 ] && rm -rf "$STATE"
  [ ! -e "$STATE" ] || { echo "$STATE exists; set FRESH=1 to remove it" >&2; exit 2; }
  mkdir -p "$STATE"
  set +e
  orcd_watched "$OUT/start.log" senselab- uv run --no-sync python benchmarks/exit/interrupt_resume.py start \
    --work "$STATE" --profile "$PROFILE" --out "$STATE/start.json"
  STATUS=$?
  cp "$STATE/start.json" "$OUT/" 2>/dev/null
  orcd_driver interrupt --out "$WORK/verdict.json" --start-json "$STATE/start.json" --status "$STATUS"
  exit $?
fi
[ -f "$STATE/start.json" ] || { echo "no start.json under $STATE: the start job did not write one" >&2; exit 2; }
HASH=$(orcd_driver get --file "$STATE/start.json" --key hash)
PREFIX=$(orcd_driver prefix --work "$STATE" --profile "$PROFILE" --hash "$HASH")
[ -n "$PREFIX" ] || { echo "could not compute the run's Slurm job prefix" >&2; exit 2; }
set +e
: > "$OUT/resume.log"
orcd_watched "$OUT/resume.log" "$PREFIX" uv run --no-sync python benchmarks/exit/interrupt_resume.py resume "$HASH" \
  --work "$STATE" --profile "$PROFILE" --out "$OUT/resume.json"
STATUS=$?
cp "$STATE/start.json" "$OUT/" 2>/dev/null
orcd_driver resume --out "$WORK/verdict.json" --start-json "$STATE/start.json" --resume-json "$OUT/resume.json" --status "$STATUS"
exit $?
