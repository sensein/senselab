#!/bin/bash
# Replay the in-family branch over the finished corpus run, one side of the change per submission.
#
#   REPLAY_SIDE=old sbatch --array=0-63 replay.sbatch
#   REPLAY_SIDE=new sbatch --array=0-63 replay.sbatch

#SBATCH --job-name=counts-replay
#SBATCH --partition=mit_preemptable
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=${ORCD_GROUP_SCRATCH}/counts-replay/logs/%x-%A_%a.out

set -uo pipefail
BASE=${ORCD_GROUP_SCRATCH}/counts-replay
SIDE="${REPLAY_SIDE:?REPLAY_SIDE must be old or new}"

export PYTHONPATH="$BASE/$SIDE"
export REPLAY_ROWS=${ORCD_GROUP_SCRATCH}/triage_design_20260919/run/rows
export REPLAY_OUT="$BASE/out/$SIDE"
export REPLAY_HINTS=${ORCD_GROUP_SCRATCH}/triage_design_20260919/scope
export HF_HOME=${ORCD_DATA}/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${ORCD_GROUP_SCRATCH}/miniforge/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

PY=${ORCD_GROUP_SCRATCH}/senselab-corpus/.venv/bin/python
echo "=== side $SIDE task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT} on $(hostname) at $(date -Is)"
"$PY" "$BASE/corpus-replay.py"
rc=$?
echo "=== exit $rc at $(date -Is)"
exit "$rc"
