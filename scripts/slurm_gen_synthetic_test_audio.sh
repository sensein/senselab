#!/usr/bin/env bash
# One-off slurm job to generate the synthetic speaker-profile test fixtures.
# Submit with: sbatch scripts/slurm_gen_synthetic_test_audio.sh
#
# Runs the (CPU-bound, deterministic) SpeechT5 generator under uv. ~600 MB
# download on first run; ~5-10 min total on CPU.
#
#SBATCH --job-name=gen-synth-audio
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/gen_synth_audio_%j.out
#SBATCH --error=logs/gen_synth_audio_%j.err

set -euo pipefail

echo "================================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "================================================================"

cd /home/wilke18/senselab
mkdir -p logs

uv run python scripts/gen_synthetic_test_audio.py

echo "================================================================"
echo "Finished:   $(date)"
echo "Output:     src/tests/data_for_testing/synthetic/"
ls -la src/tests/data_for_testing/synthetic/ | head -50
echo "================================================================"
