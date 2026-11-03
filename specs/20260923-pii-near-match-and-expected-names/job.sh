#!/bin/bash
#SBATCH --job-name=pii-nearmatch
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --qos=normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=${ORCD_GROUP_SCRATCH}/pii_nearmatch_20260923/job-%j.out
#SBATCH --error=${ORCD_GROUP_SCRATCH}/pii_nearmatch_20260923/job-%j.err
set -euo pipefail
W=${ORCD_GROUP_SCRATCH}/pii_nearmatch_20260923
cd ${ORCD_GROUP_SCRATCH}/senselab-rerun
.venv/bin/python "$W/measure-near-match.py" \
  ${ORCD_GROUP_SCRATCH}/triage_rerun_20260923/out \
  "$W/tables.json" \
  --workers 48
.venv/bin/python "$W/report-near-match.py" "$W/tables.json" > "$W/report.txt"
echo done
