#!/bin/bash
#SBATCH --job-name=work-pool-race
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:20:00
# Cross-node race for senselab.utils.work_pool. The caller names the partition and the layout, e.g.
#   sbatch -p <partition> -N 2 --ntasks-per-node 8 --export=ALL,POOL=<shared dir>,PY=<python with senselab> race.sbatch
# Submit from the repository root (or set RACE_HARNESS to this directory).
# Leg 1: every task claims and completes 2000 items. Leg 2: lease_s=5 and every third task SIGKILLs itself
# while holding a lease, so reaping races claiming. Each leg ends with: race items=N done=N double=0 lost=0
set -uo pipefail
POOL=${POOL:?set POOL to a directory every node sees}
PY=${PY:-python3}
ITEMS=${ITEMS:-2000}
HERE=${RACE_HARNESS:-$SLURM_SUBMIT_DIR/specs/20261012-cluster-exec/harness}
RUN=$POOL/race-$SLURM_JOB_ID
mkdir -p "$RUN"
# A submitting shell that runs inside another job exports its memory setting, which srun refuses beside ours.
[ -n "${SLURM_MEM_PER_CPU:-}" ] && unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
rc=0
echo "== nodes $SLURM_JOB_NUM_NODES tasks $SLURM_NTASKS"

echo "== primitives"
$PY "$HERE/race.py" prim-setup "$RUN/prim" --names 400
srun --kill-on-bad-exit=0 $PY "$HERE/race.py" prim "$RUN/prim" --names 400 --start $(( $(date +%s) + 10 ))
$PY "$HERE/race.py" prim-check "$RUN/prim" --names 400 || rc=1

echo "== leg 1: claim contention"
$PY "$HERE/race.py" publish "$RUN/leg1" --items "$ITEMS" --lease-s 60 --heartbeat-s 5
srun --kill-on-bad-exit=0 $PY "$HERE/race.py" work "$RUN/leg1" --items "$ITEMS" --start $(( $(date +%s) + 10 ))
$PY "$HERE/race.py" check "$RUN/leg1" --items "$ITEMS" || rc=1

echo "== leg 2: lease_s=5, a third of the workers killed mid-item"
$PY "$HERE/race.py" publish "$RUN/leg2" --items "$ITEMS" --lease-s 5 --heartbeat-s 1
srun --kill-on-bad-exit=0 $PY "$HERE/race.py" work "$RUN/leg2" --items "$ITEMS" --start $(( $(date +%s) + 10 )) \
    --hold 0.02 --kill-third
$PY "$HERE/race.py" check "$RUN/leg2" --items "$ITEMS" || rc=1

[ "$rc" = 0 ] && rm -rf "$RUN"
exit $rc
