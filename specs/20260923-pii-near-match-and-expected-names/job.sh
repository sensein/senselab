#!/bin/bash
#SBATCH --job-name=pii-nearmatch
#SBATCH --partition=pi_satra
#SBATCH --qos=normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/orcd/scratch/bcs/002/satra/pii_nearmatch_20260923/job-%j.out
#SBATCH --error=/orcd/scratch/bcs/002/satra/pii_nearmatch_20260923/job-%j.err
set -euo pipefail
W=/orcd/scratch/bcs/002/satra/pii_nearmatch_20260923
cd /orcd/scratch/bcs/002/satra/senselab-rerun
.venv/bin/python "$W/measure-near-match.py" \
  /orcd/scratch/bcs/002/satra/triage_rerun_20260923/out \
  "$W/tables.json" \
  --workers 48
.venv/bin/python "$W/report-near-match.py" "$W/tables.json" > "$W/report.txt"
echo done
