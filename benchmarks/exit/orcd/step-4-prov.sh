#!/bin/bash
# Step 4, provenance clause: a post-hoc read of the Step 4 store of a start/resume pair. CPU only, no model, no worker
# submission. Run it at the SHA of the pair; it clones, `uv sync`s and only reads `<WORK_ROOT>/step4/cache/stt/2x/runs/
# <hash>/prov/`, `<WORK_ROOT>/step4/start.json` and the resume job's
# `$WORK_ROOT/results/orcd-4-resume-<RESUME_JOB>/resume.json`.
# Checks, for the run hash of start.json: the runtime's own `check_provenance` (ADR-010: every committed key has an
# execution or a Reuse; no writer records more Reuse of a key than the run has jobs with it); one key per item; the keys
# without an execution activity number at most pool.quantum_items per cancelled worker (a kill loses the claim in
# flight); resume-phase executions = items - committed at the interrupt - late start executions (resume.json "late");
# senselab:Reuse = the resume's reuses, reported split by the settled shape (the coordinator's) and the workers';
# exactly one plan entity. Each writer document is assigned to a phase as a whole, by its activities' times against the
# resume's first event; one that straddles it fails. The resume accounting's `executions` counts `finished` events, not
# executions, and is not compared. Information: items claimed twice within one epoch, from events.
# Submit alone (the resume has finished) or behind it with --dependency=afterok:<resume job> (placeholders, P-28):
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=2 --mem=8G --time=00:30:00 --job-name=orcd-step4-prov \
#          --output="$WORK_ROOT/orcd-step4-prov-%j.out" \
#          --export=ALL,SHA=<40-hex>,RESUME_JOB=<resume job id>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION" \
#          step-4-prov.sh
# Optional: HASH (default start.json's), FFMPEG_LIB. Expected wall 3 to 8 minutes. Results:
# $WORK_ROOT/results/orcd-4-prov-<job>/ (prov.log, verdict.json). A FAIL line states the count found, which is the finding.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
orcd_need RESUME_JOB
orcd_setup "4-prov" "none"
STATE="$WORK_ROOT/step4"
RESUME_JSON="$WORK_ROOT/results/orcd-4-resume-$RESUME_JOB/resume.json"
[ -f "$STATE/start.json" ] || { echo "no start.json under $STATE" >&2; exit 2; }
[ -f "$RESUME_JSON" ] || { echo "no resume accounting at $RESUME_JSON" >&2; exit 2; }
HASH="${HASH:-$(orcd_driver get --file "$STATE/start.json" --key hash)}"
QUANTUM=$(orcd_driver get --file "$PROFILE" --key pool.quantum_items)
set +e
orcd_driver prov --out "$WORK/verdict.json" --run-dir "$STATE/cache/stt/2x/runs/$HASH" --hash "$HASH" \
  --start-json "$STATE/start.json" --resume-json "$RESUME_JSON" --quantum-items "$QUANTUM" 2>&1 | tee "$OUT/prov.log"
exit ${PIPESTATUS[0]}
