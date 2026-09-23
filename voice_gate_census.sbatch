#!/bin/bash
#SBATCH --job-name=voice-gate-census
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/scratch/bcs/002/satra/voice_gate_census_20260923/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/bcs/002/satra/voice_gate_census_20260923/logs/%x-%j.err
set -euo pipefail
umask 027

B=/orcd/scratch/bcs/002/satra/voice_gate_census_20260923
CORPUS=/orcd/scratch/bcs/002/satra/triage_replay_20260922/out
PY=/orcd/scratch/orcd/013/satra/miniforge/bin/python3

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "corpus=$CORPUS (read-only)"
"$PY" -V

"$PY" -u "$B/voice_gate_census.py" \
  --corpus "$CORPUS" \
  --out "$B/results" \
  --workers 16

echo "finished=$(date -Is)"
