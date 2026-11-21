#!/bin/bash
# Step 5, the lease cross-node check, at the head SHA, on one 3-node allocation of the shared filesystem:
# `probe_lease_cross_node.py` (a holder renews past stale_after while two nodes try to break it, then is SIGKILLed and
# one of them takes it within a bound; three distinct hosts). The pool's queue is pydra's, whose cross-node checks are
# its own (RFC 0005 W12). Each role runs as one task of `srun` on its own node. Submit (placeholders, P-28):
#   sbatch --partition="$CPU_PARTITION" --nodes=3 --ntasks-per-node=1 --cpus-per-task=8 --mem=16G --time=01:30:00 \
#          --job-name=orcd-crossnode --output="$WORK_ROOT/orcd-crossnode-%j.out" \
#          --export=ALL,SHA=<40-hex>,WORK_ROOT="$WORK_ROOT",SCRIPTS_DIR="$SCRIPTS_DIR",PROFILE_SRC="$PROFILE_SRC",CPU_PARTITION="$CPU_PARTITION",GPU_PARTITION="$GPU_PARTITION"[,FFMPEG_LIB=...] \
#          step-5-crossnode.sh
# Add --nodelist only if `orcd_submit.py --plan` says so (never exclude nodes by name). The probe directory is on
# $WORK_ROOT (the shared filesystem the profile's roots live on). Expected wall: about 25 minutes (lease: hold 360 s,
# break until 1200 s). Results in $WORK_ROOT/results/orcd-crossnode-<job>/: lease-verdict.json (the probe's own JSON),
# lease.log, verdict.json.
# PASS: the last line is one JSON verdict with "pass": true, exit 0: the probe verdict "pass": true over three distinct
# hosts. The probe's own checks are listed in its module docstring.
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR is not set}"
source "$SCRIPTS_DIR/lib.sh"
[ "${SLURM_NNODES:-0}" -ge 3 ] || { echo "this allocation has ${SLURM_NNODES:-0} nodes, the probes need 3" >&2; exit 2; }
orcd_setup crossnode "-"
CK="$WORK/checkout"
export PYTHONPATH="$CK/packages/senselab-core/src:$CK/packages/senselab/src:$CK/packages/senselab-worker/src"
PY="$CK/.venv/bin/python"
LEASE="$WORK/probe-lease"
mkdir -p "$LEASE"
STALE=120
FRACTION=0.25
SLACK=75
set +e
# Lease: node 0 holds, nodes 1 and 2 try to break. SIGKILL of the holder is its designed end, so exit statuses are not read.
echo "lease probe start $(date +%T)"
srun --nodes=3 --ntasks=3 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --kill-on-bad-exit=0 bash -c '
  case "$SLURM_PROCID" in
    0) exec "$0" "$1" hold "$2" --stale "$3" --fraction "$4" --hold 360 ;;
    1) exec "$0" "$1" break "$2" --stale "$3" --fraction "$4" --name B --until 1200 ;;
    2) exec "$0" "$1" break "$2" --stale "$3" --fraction "$4" --name C --until 1200 ;;
  esac' "$PY" "$CK/scripts/probe_lease_cross_node.py" "$LEASE" "$STALE" "$FRACTION" 2>&1 | tee "$OUT/lease.log"
"$PY" "$CK/scripts/probe_lease_cross_node.py" verdict "$LEASE" --stale "$STALE" --slack "$SLACK" | tee "$OUT/lease.verdict.raw"
tail -n 1 "$OUT/lease.verdict.raw" > "$OUT/lease-verdict.json"
orcd_driver crossnode --out "$WORK/verdict.json" --dir "$OUT"
exit $?
