#!/bin/bash
# Shared prologue of the step-<n>.sh scripts; sourced from $SCRIPTS_DIR, never run on its own.
#
# Every script needs, in its --export: SHA (40-hex commit, pushed), WORK_ROOT (this account's scratch), SCRIPTS_DIR (a
# scratch copy of this directory), PROFILE_SRC (the group's site profile file, copied to scratch; never committed),
# CPU_PARTITION and GPU_PARTITION (the partitions to run on, from the site's own planning).
# Optional: CHECKOUT_SRC (the campaign checkout, itself cloned from GitHub and pinned at SHA; default GitHub), FFMPEG_LIB,
# HF_TOKEN, WORKER_TIME_S (see below), SYNC_ARGS, STALL_S (the no-progress bound of orcd_watched, default 1800).
#
# orcd_setup NAME CHECKED_CLASSES: strict mode, a fresh clone pinned to SHA, `uv sync`, the site profile copied to
# $WORK/profile.yaml after a time fit against each checked class's partition (`sinfo`), and an EXIT trap that always
# prints one JSON verdict line and copies $OUT to $RESULTS. A script puts its files in $OUT and writes
# $WORK/verdict.json through the driver (benchmarks/exit/orcd_driver.py, from the checkout at SHA); if it dies before
# that, the trap writes a failing verdict.
set -eo pipefail
umask 027
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1

orcd_need() {
  local v
  for v in "$@"; do [ -n "${!v:-}" ] || { echo "$v is not set" >&2; exit 2; }; done
}

orcd_exit() {
  local rc=$?
  set +e
  if [ -n "${WORK:-}" ] && [ ! -f "$WORK/verdict.json" ]; then
    printf '{"pass": false, "failed": ["script stopped before its verdict (exit %s)"], "checks": 0}\n' "$rc" | tee "$WORK/verdict.json"
    rc=1
  fi
  if [ -n "${RESULTS:-}" ] && [ -d "${OUT:-/nonexistent}" ]; then
    mkdir -p "$RESULTS"
    cp -r "$OUT"/. "$RESULTS"/ 2>/dev/null
    cp "$WORK/verdict.json" "$RESULTS"/ 2>/dev/null
    echo "results: $RESULTS"
  fi
  exit "$rc"
}

# orcd_setup NAME CHECKED_CLASSES (comma-separated class names whose time_s is fitted to its partition; empty: all).
orcd_setup() {
  local name="$1" classes="${2:-}"
  orcd_need SHA WORK_ROOT SCRIPTS_DIR PROFILE_SRC CPU_PARTITION GPU_PARTITION
  [[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "SHA is not a 40-hex commit" >&2; exit 2; }
  [ -f "$PROFILE_SRC" ] || { echo "PROFILE_SRC is not a file" >&2; exit 2; }
  JOB="${SLURM_JOB_ID:-local}"
  WORK="$WORK_ROOT/work/orcd-$name-$JOB"
  RESULTS="$WORK_ROOT/results/orcd-$name-$JOB"
  OUT="$WORK/out"
  rm -rf "$WORK" "$RESULTS"
  mkdir -p "$OUT"
  trap orcd_exit EXIT
  export UV_CACHE_DIR="$WORK_ROOT/uv-cache" HF_HOME="$WORK_ROOT/hf" SENSELAB_ENV_ROOT="$WORK_ROOT/envs-$SHA"
  export SENSELAB_VENV_CACHE="$WORK_ROOT/venvs"
  export LD_LIBRARY_PATH="${FFMPEG_LIB:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  git clone -q "${CHECKOUT_SRC:-https://github.com/sensein/senselab}" "$WORK/checkout"
  git -C "$WORK/checkout" checkout -q --detach "$SHA"
  cd "$WORK/checkout"
  [ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA" >&2; exit 2; }
  uv sync ${SYNC_ARGS:---locked --all-extras --group dev --group benchmarks --group test}
  PROFILE="$WORK/profile.yaml"
  # WORKER_TIME_S caps every class's time_s in the job's copy of the profile (the group's file is not changed); the
  # time fit refuses, before anything is submitted, a checked class that still asks more than its partition allows.
  uv run --no-sync python "$WORK/checkout/benchmarks/exit/orcd_driver.py" profile --src "$PROFILE_SRC" --out "$PROFILE" \
    ${classes:+--classes "$classes"} ${WORKER_TIME_S:+--worker-time-s "$WORKER_TIME_S"}
  export SENSELAB_TEST_PROFILE="$PROFILE"
  # The nfs parametrisation of the lock and work-pool tests runs on this directory (scratch is NFS4); unset, it skips.
  export SENSELAB_WORK_POOL_NFS_DIR="${SENSELAB_WORK_POOL_NFS_DIR:-$WORK/nfs-probe}"
  mkdir -p "$SENSELAB_WORK_POOL_NFS_DIR"
  cd "$WORK/checkout"
}

# orcd_driver ARGS...: the driver, from the checkout.
orcd_driver() {
  uv run --no-sync python "$WORK/checkout/benchmarks/exit/orcd_driver.py" "$@"
}

# orcd_watched LOG SELECTOR COMMAND...: run COMMAND with its output teed to LOG under the no-progress watchdog
# (benchmarks/exit/stall.py). It is stopped, and the script exits 3 with a failing verdict, once the committed count of
# its last `progress:` line has not changed for STALL_S seconds (default 1800) while none of the run's Slurm jobs is
# queued or running. A `squeue` that fails is not read as "no job". SELECTOR names the run's jobs: `prefix:P` (jobs named
# P*, when the run's hash is known) or `runs:GLOB` (the run directories COMMAND creates; from the moment one exists
# the watchdog counts only the jobs `exitlib.slurm_prefix` gives it, another run's jobs under the same login do not
# count; until then any `senselab-` job does). Otherwise it returns COMMAND's status.
orcd_watched() {
  local log="$1" selector="$2" rc=0 where=()
  shift 2
  case "$selector" in
    prefix:*) where=(--prefix "${selector#prefix:}") ;;
    runs:*) where=(--runs-glob "${selector#runs:}" --profile "$PROFILE") ;;
    *) echo "orcd_watched: selector is prefix:P or runs:GLOB" >&2; exit 2 ;;
  esac
  orcd_driver watch --log "$log" "${where[@]}" --stall-s "${STALL_S:-1800}" --verdict "$WORK/verdict.json" -- "$@" || rc=$?
  if [ "$rc" = 3 ] && grep -q '"no progress' "$WORK/verdict.json" 2>/dev/null; then exit 3; fi
  return "$rc"
}
