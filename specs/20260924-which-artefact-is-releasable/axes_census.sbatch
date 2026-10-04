#!/bin/bash
#SBATCH --job-name=axes-census
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:40:00
#SBATCH --array=0-31
#SBATCH --output=/orcd/scratch/bcs/002/satra/axes_census/logs/%x-%A_%a.out
set -euo pipefail
source /orcd/scratch/bcs/002/satra/triage_design_20260919/env_common.sh
export SCAN_ROOT="${CENSUS_ROOT}"
export SCAN_OUT="${CENSUS_OUT}"
python3 /orcd/scratch/bcs/002/satra/axes_census/axes_census.py
