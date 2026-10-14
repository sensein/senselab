#!/bin/bash
#SBATCH --job-name=work-pool-smoke
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
# Live smoke of the work_pool controller. The caller names the partition of this job and supplies the classes
# file (whose first class lists the partitions to try), e.g.
#   sbatch -p <partition> --export=ALL,POOL=<shared dir>,CLASSES=<classes.yaml>,PY=<python with senselab> smoke.sbatch
# Submit from the repository root (or set SMOKE_HARNESS to this directory).
set -uo pipefail
POOL=${POOL:?set POOL to a directory every node sees}
CLASSES=${CLASSES:?set CLASSES to the resource classes file}
PY=${PY:-python3}
HERE=${SMOKE_HARNESS:-$SLURM_SUBMIT_DIR/specs/20261012-cluster-exec/harness}
RUN=$POOL/smoke-$SLURM_JOB_ID
# A submitting shell that runs inside another job exports its memory setting; keep it out of the workers' jobs.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
mkdir -p "$RUN-logs"
$PY "$HERE/smoke.py" "$RUN" "$CLASSES" --items "${ITEMS:-20}" --seconds "${SECONDS_PER_ITEM:-60}" --tick 30 \
    --timeout "${SMOKE_TIMEOUT:-1500}" --log-dir "$RUN-logs"
