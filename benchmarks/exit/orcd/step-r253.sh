#!/bin/bash
# R-253: does the per-item cost `a` (an item's wall minus its requests' walls: claim, bind, commit) vary with the
# item's job count? One unit, `signal.resample` (target_rate 8000) over 400 synthetic recordings on a local pool, probed
# by `ladder.run` once per grid point: `calibration.single_item_jobs` J in 2, 4, 8, 16, 32, 64 (4 repetitions each, a
# probe's single items then hold J jobs and its others 1 job), and `pool.quantum_items` 4 and 256 at J = 2. Each run has
# a SENSELAB_CACHE and measures root of its own. The report gives `a` per job count (n, median, quartiles, min, max)
# pooled over every run, and the least-squares slope of `a` on jobs with its standard error. CPU only. Submit
# (placeholders, P-28):
#   sbatch --partition="$CPU_PARTITION" --cpus-per-task=8 --mem=32G --time=01:00:00 --job-name=orcd-r253 \
#          --output="$WORK_ROOT/orcd-r253-%j.out" \
#          --export=ALL,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION"[,FFMPEG_LIB=...] \
#          step-r253.sh
# Expected wall: 15 to 30 minutes (26 runs of about 40 s on 8 workers: tree write, probe, production remainder).
# Results in $WORK_ROOT/results/orcd-r253-<job>/: r253.json (per run and pooled), run.log, verdict.json.
# PASS means the measurement completed: every run finished, each of J = 1, 2, 4, 8, 16, 32, 64 has probe items with an
# item wall. The finding is the printed `R253 a per job count` line and the slope with its standard error; the script
# sets no threshold on it (a flat `a` is a slope within about two standard errors of zero).
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
orcd_setup r253 "-"
mkdir -p "$WORK/r253"
set +e
orcd_driver r253 --out "$WORK/verdict.json" --profile "$PROFILE" --work "$WORK/r253" --items 400 --workers 8 --reps 4 2>&1 | tee "$OUT/run.log"
STATUS=${PIPESTATUS[0]}
cp "$WORK/r253/r253.json" "$OUT/" 2>/dev/null
exit "$STATUS"
