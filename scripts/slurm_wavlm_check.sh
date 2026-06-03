#!/usr/bin/env bash
# Verify the WavLM SV backend in the real 3-model consensus on a GPU node.
# WavLM had only unit-level / mocked coverage; this loads the real checkpoint,
# confirms it embeds at 512-D, and checks it discriminates target vs intruder.
#
# Submit with: sbatch scripts/slurm_wavlm_check.sh
#
#SBATCH --job-name=sp-wavlm-check
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sp_wavlm_check_%j.out
#SBATCH --error=logs/sp_wavlm_check_%j.err

set -euo pipefail

echo "================================================================"
echo "Job:        ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "Started:    $(date)"
echo "================================================================"

cd /orcd/home/002/wilke18/senselab
mkdir -p logs

uv run python - <<'PY'
import numpy as np
import torch

print("cuda:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), flush=True)

from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.build import ProfileInput, build_speaker_profile
from senselab.audio.workflows.speaker_profile.compare import compare_recording_to_profile
from senselab.audio.workflows.audio_analysis.embeddings import extract_per_window_embeddings
from src.tests.audio.workflows.speaker_profile.conftest import load_clip

MODELS = list(C.DEFAULT_EMBEDDING_MODELS)  # ECAPA + ResNet + WavLM
WAVLM = C.WAVLM_DEFAULT_CHECKPOINT
print("consensus models:", MODELS, flush=True)

# 1. Build the full 3-model profile from the long clean passages.
build_ids = ["sub-A-confident/ses-1/rainbow.flac", "sub-A-confident/ses-1/north-wind.flac"]
profile = build_speaker_profile(
    "sub-A", [ProfileInput(audio=load_clip(f), file_id=f) for f in build_ids], embedding_models=MODELS
)
print("\n== profile centroids (model -> dim) ==", flush=True)
for m, c in profile.centroids.items():
    print(f"  {m}: {len(c)}-D", flush=True)
assert WAVLM in profile.centroids, "WavLM produced no centroid!"
assert len(profile.centroids[WAVLM]) == 512, f"WavLM dim {len(profile.centroids[WAVLM])} != 512"
print("WavLM present in consensus, 512-D: OK", flush=True)

# 2. Length stability: WavLM embedding of the same clip at 1s vs 2s windows.
clip = load_clip("sub-A-confident/ses-1/grandfather.flac")
w1 = extract_per_window_embeddings(audio=clip, models=[WAVLM], window_s=1.0, hop_s=1.0)[WAVLM]
w2 = extract_per_window_embeddings(audio=clip, models=[WAVLM], window_s=2.0, hop_s=2.0)[WAVLM]
def _unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) or 1.0)
sim = float(_unit(w1[0].vector) @ _unit(w2[0].vector))
print(f"\nWavLM 1s-vs-2s window cos-sim (same clip start): {sim:.3f}", flush=True)

# 3. Discrimination: score target vs intruder with the full consensus.
def score(fid):
    det = extract_per_window_embeddings(audio=load_clip(fid), models=MODELS,
                                        window_s=C.DETECT_WINDOW_S, hop_s=C.DETECT_HOP_S)
    res = compare_recording_to_profile(det, profile.centroids, profile.calibration_band)
    wavlm_unc = [r.per_model.get(WAVLM) for r in res if r.per_model.get(WAVLM) is not None]
    flags = {}
    for r in res: flags[r.flag] = flags.get(r.flag, 0) + 1
    return (np.mean(wavlm_unc) if wavlm_unc else float("nan")), flags

t_unc, t_flags = score("sub-A-confident/ses-1/grandfather.flac")  # held-out target
i_unc, i_flags = score("speaker-B/clip-00.flac")                  # intruder
print(f"\nTARGET   grandfather: mean WavLM other-voice unc={t_unc:.2f}  flags={t_flags}", flush=True)
print(f"INTRUDER speaker-B:   mean WavLM other-voice unc={i_unc:.2f}  flags={i_flags}", flush=True)
assert i_unc > t_unc, f"WavLM did not discriminate (intruder {i_unc:.2f} <= target {t_unc:.2f})"
print("\nWavLM consensus discrimination: OK", flush=True)
PY

echo "================================================================"
echo "Finished:   $(date)"
echo "================================================================"
