#!/bin/bash
#SBATCH --job-name=pool-worker
#SBATCH --partition=pi_satra,ou_bcs_low
#SBATCH --exclude=node2119,node2621,node3002,node1702,node3804
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# One slice_pool worker. Submit with `slice_pool.py submit POOL --workers W -- <sbatch args>` (partition, gres, cpus,
# mem, time on the command line override the defaults above); the per-slice command lives in POOL/pool.json.
#   POOL_TOOL  the slice_pool.py to run (default: the run tree's pool/slice_pool.py)
#   POOL_PY    the interpreter (default: the r28 venv python; slice_pool.py is stdlib only)
set -uo pipefail
POOL=${1:?usage: sbatch worker.sbatch POOL_DIR}
TOOL=${POOL_TOOL:-/orcd/scratch/bcs/002/satra/r18d_20261010/pool/slice_pool.py}
PY=${POOL_PY:-/orcd/scratch/bcs/002/satra/senselab-r28/.venv/bin/python}
echo "=== worker ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $(date -Is)"
exec "$PY" -u "$TOOL" work "$POOL"
