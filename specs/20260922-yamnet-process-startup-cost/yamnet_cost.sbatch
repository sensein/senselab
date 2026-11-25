#!/bin/bash
#SBATCH --job-name=yamnet-cost
#SBATCH --partition=mit_preemptable
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --output=${ORCD_GROUP_SCRATCH}/yamnetperf/logs/%x-%j.out
set -euo pipefail
source ${ORCD_GROUP_SCRATCH}/triage_design_20260919/env_common.sh
export CUDA_VISIBLE_DEVICES=""
W=${ORCD_GROUP_SCRATCH}/yamnetperf
YP=${ORCD_GROUP_SCRATCH}/senselab-venvs/yamnet/bin/python
cd "$W"
echo "=== host=$(hostname) exclusive=yes"
echo "=== loadavg_start=$(cat /proc/loadavg)"
echo "=== nproc=$(nproc)"
"$YP" -u probe_yamnet_cost.py "$YP" > "$W/cost-${SLURM_JOB_ID}.jsonl"
echo "=== loadavg_end=$(cat /proc/loadavg)"
echo "=== rows=$(wc -l < "$W/cost-${SLURM_JOB_ID}.jsonl")"
