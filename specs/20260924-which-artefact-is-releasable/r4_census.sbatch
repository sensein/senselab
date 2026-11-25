#!/bin/bash
#SBATCH --job-name=r4-census
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:40:00
#SBATCH --array=0-31
#SBATCH --output=${ORCD_GROUP_SCRATCH}/axes_census/logs/%x-%A_%a.out
set -euo pipefail
source ${ORCD_GROUP_SCRATCH}/triage_design_20260919/env_common.sh
export SCAN_ROOT=${ORCD_GROUP_SCRATCH}/triage_r4_20260924/out
export SCAN_OUT=${ORCD_GROUP_SCRATCH}/axes_census/r4
python3 ${ORCD_GROUP_SCRATCH}/axes_census/axes_census.py
