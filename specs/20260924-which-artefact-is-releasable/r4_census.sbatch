#!/bin/bash
#SBATCH --job-name=r4-census
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:40:00
#SBATCH --array=0-31
#SBATCH --output=/orcd/scratch/bcs/002/satra/axes_census/logs/%x-%A_%a.out
set -euo pipefail
source /orcd/scratch/bcs/002/satra/triage_design_20260919/env_common.sh
export SCAN_ROOT=/orcd/scratch/bcs/002/satra/triage_r4_20260924/out
export SCAN_OUT=/orcd/scratch/bcs/002/satra/axes_census/r4
python3 /orcd/scratch/bcs/002/satra/axes_census/axes_census.py
