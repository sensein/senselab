#!/bin/bash
# Step prepare: one job, submitted first, that fills the shared uv cache at the head SHA (a clone and one `uv sync`,
# which builds the git dependencies) and checks the site profile (its variables expand and each class's time_s fits its
# partition). Every other job of the campaign is submitted with --dependency=afterok:<this job>, so their concurrent
# `uv sync`s read a warm cache instead of each building it under uv's cache lock.
# Submit (placeholders, P-28):
#   P=$(sbatch --parsable --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=8G --time=00:30:00 --job-name=orcd-prepare \
#          --output="$WORK_ROOT/orcd-prepare-%j.out" \
#          --export=ALL,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC"[,GROUP_UV_CACHE=...] \
#          step-prepare.sh)
# Results in $WORK_ROOT/results/orcd-prepare-<job>/: verdict.json.
# PASS: the last line is one JSON verdict with "pass": true, exit 0.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
orcd_setup prepare ""
orcd_driver get --file "$PROFILE" --key name
printf '{"pass": true, "failed": [], "checks": 1, "uv_cache": "%s"}\n' "$UV_CACHE_DIR" | tee "$WORK/verdict.json"
