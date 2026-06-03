#!/usr/bin/env bash
# Run the speaker-profile success-criteria smoke checks (SC-002/003/004/005) on a
# GPU compute node. These exercise the real ECAPA + ResNet models over the long
# synthetic passages, so they must NOT run on the login node.
#
# Submit with: sbatch scripts/slurm_speaker_profile_sc.sh
#
#SBATCH --job-name=sp-success-criteria
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sp_success_criteria_%j.out
#SBATCH --error=logs/sp_success_criteria_%j.err

set -euo pipefail

echo "================================================================"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "================================================================"

cd /orcd/home/002/wilke18/senselab
mkdir -p logs

# The repo conftest hard-aborts pytest when optional video extras are missing,
# so run the success-criteria test functions directly (bypassing collection).
# Embedding extraction auto-selects CUDA when a GPU is present.
uv run python - <<'PY'
import sys, time, traceback
import torch

print("cuda available:", torch.cuda.is_available(),
      "| device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), flush=True)

import src.tests.audio.workflows.speaker_profile.success_criteria_test as T

order = [
    "test_sc002_contamination_tolerance",
    "test_sc003_other_voice_detection_beats_false_positive",
    "test_sc004_target_only_false_flag_under_10pct",
    "test_sc005_clean_outranks_noisy_quality",
]
passed = failed = 0
for name in order:
    fn = getattr(T, name)
    t0 = time.time()
    try:
        fn()
        passed += 1
        print(f"PASS {name} ({time.time() - t0:.0f}s)", flush=True)
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"FAIL {name}: {exc!r} ({time.time() - t0:.0f}s)", flush=True)
        traceback.print_exc()
print(f"\n{passed} passed, {failed} failed", flush=True)
sys.exit(1 if failed else 0)
PY

echo "================================================================"
echo "Finished:   $(date)"
echo "================================================================"
