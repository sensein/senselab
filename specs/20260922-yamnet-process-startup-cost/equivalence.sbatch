#!/bin/bash
#SBATCH --job-name=yperf-equiv
#SBATCH --partition=mit_preemptable
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=${ORCD_GROUP_SCRATCH}/yamnetperf/logs/%x-%j.out
set -euo pipefail
source ${ORCD_GROUP_SCRATCH}/triage_design_20260919/env_common.sh
export SENSELAB_REPO=${ORCD_GROUP_SCRATCH}/senselab-yamnetnew
export CUDA_VISIBLE_DEVICES=""
W=${ORCD_GROUP_SCRATCH}/yamnetperf
cd "$SENSELAB_REPO"
echo "=== host=$(hostname) exclusive=yes"
echo "=== loadavg_start=$(cat /proc/loadavg)"
git log --oneline -1
uv run --no-sync python -u "$W/probe_equivalence.py" \
    "$W/steps_manifest.jsonl" 884d481ca92cd0ed67e7e75706b1a4476f813ab7 \
    > "$W/equiv-${SLURM_JOB_ID}.jsonl"
echo "=== loadavg_end=$(cat /proc/loadavg)"
cat "$W/equiv-${SLURM_JOB_ID}.jsonl"
