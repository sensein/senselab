#!/usr/bin/env bash
# Characterization sweeps for the speaker-profile signals (GPU compute node).
# Produces curves (text tables) for:
#   1. Input-degradation robustness: as SNR drops on a pure-target recording,
#      how do SQUIM, target-quality, match-fraction, consistency, and the
#      false-other-voice rate move? (Tests whether the embedding-derived signals
#      track acoustic quality or stay flat — SV models are noise-robust by design.)
#   2. Contamination tolerance: centroid drift + target-vs-intruder similarity
#      vs. contamination fraction.
#   3. Threshold sensitivity (T028): detection vs. false-positive across the
#      other-voice cutoff (a mini-ROC) — characterize, do NOT lock in.
#
# Submit with: sbatch scripts/slurm_speaker_profile_sweep.sh
#
#SBATCH --job-name=sp-sweep
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/sp_sweep_%j.out
#SBATCH --error=logs/sp_sweep_%j.err

set -euo pipefail
echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) | $(date) ==="
cd /orcd/home/002/wilke18/senselab
mkdir -p logs

uv run python - <<'PY'
import numpy as np
import torch

print("cuda:", torch.cuda.is_available(), "|",
      (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), flush=True)

from senselab.audio.tasks.features_extraction import extract_features_from_audios
from senselab.audio.workflows.audio_analysis.embeddings import extract_per_window_embeddings
from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.build import ProfileInput, build_speaker_profile
from senselab.audio.workflows.speaker_profile.compare import compare_recording_to_profile, compute_target_quality
from src.tests.audio.workflows.speaker_profile.conftest import (
    add_noise_at_snr, compose_contamination, compose_other_voice, load_clip,
)

MODELS = list(C.DEFAULT_EMBEDDING_MODELS)
TARGET = "sub-A-confident/ses-1/grandfather.flac"
INTRUDER = "speaker-B/clip-00.flac"
BUILD = ["sub-A-confident/ses-1/rainbow.flac", "sub-A-confident/ses-1/north-wind.flac"]


def squim(audio):
    r = extract_features_from_audios([audio], opensmile=False, parselmouth=False,
                                     torchaudio=False, torchaudio_squim=True, sparc=False, ppgs=False)
    d = r[0] if r else {}
    sq = d.get("torchaudio_squim") or (d.get("result", {}) or {}).get("torchaudio_squim") or {}
    return {k: sq.get(k) for k in ("stoi", "pesq", "si_sdr")}


def detect(audio):
    return extract_per_window_embeddings(audio=audio, models=MODELS,
                                         window_s=C.DETECT_WINDOW_S, hop_s=C.DETECT_HOP_S)


def mean_unit(audio):
    out = {}
    for m, ws in detect(audio).items():
        v = [np.asarray(w.vector, float) for w in ws if w.vector.size]
        if v:
            mv = np.mean(np.stack(v), 0); n = np.linalg.norm(mv)
            out[m] = mv / (n or 1.0)
    return out


profile = build_speaker_profile("sub-A", [ProfileInput(audio=load_clip(f), file_id=f) for f in BUILD],
                                embedding_models=MODELS)
print(f"\nprofile confidence={profile.confidence} models={list(profile.centroids)}", flush=True)

# 1. DEGRADATION SWEEP --------------------------------------------------------
print("\n=== 1. DEGRADATION (pure target, varying SNR) ===", flush=True)
print(f"{'SNR_dB':>7} {'stoi':>6} {'pesq':>6} {'si_sdr':>7} | "
      f"{'tgt_qual':>8} {'match_fr':>8} {'consist':>8} {'false_OV':>8}", flush=True)
for snr in [None, 30, 20, 15, 10, 5, 0, -5]:
    audio = load_clip(TARGET) if snr is None else add_noise_at_snr(TARGET, snr_db=float(snr)).audio
    sq = squim(audio)
    res = compare_recording_to_profile(detect(audio), profile.centroids, profile.calibration_band)
    q = compute_target_quality(res, profile.confidence)
    scored = [r for r in res if r.flag != "unavailable"]
    false_ov = (sum(r.flag == "other_voice" for r in scored) / len(scored)) if scored else float("nan")
    lab = "clean" if snr is None else f"{snr}"
    def f(x): return f"{x:.2f}" if isinstance(x, (int, float)) else "  na"
    print(f"{lab:>7} {f(sq['stoi']):>6} {f(sq['pesq']):>6} {f(sq['si_sdr']):>7} | "
          f"{q.profile_target_quality:>8.2f} {q.profile_target_match_fraction:>8.2f} "
          f"{q.profile_mean_target_consistency:>8.2f} {false_ov:>8.2f}", flush=True)

# 2. CONTAMINATION SWEEP (build level) ---------------------------------------
print("\n=== 2. CONTAMINATION (build with x% intruder mixed into one file) ===", flush=True)
tgt_ref, int_ref = mean_unit(load_clip(TARGET)), mean_unit(load_clip(INTRUDER))
base = None
print(f"{'frac':>5} | per-model  centroid_drift_from_clean | mean(target_sim) mean(intruder_sim)", flush=True)
for frac in [0.0, 0.1, 0.2, 0.3]:
    contam = compose_contamination("sub-A-confident/ses-1/harvard-09.flac", INTRUDER, fraction=frac).audio
    inp = [ProfileInput(audio=load_clip(f), file_id=f) for f in BUILD]
    inp.append(ProfileInput(audio=contam, file_id="contam.flac"))
    p = build_speaker_profile("c", inp, embedding_models=MODELS)
    if base is None:
        base = {m: np.asarray(v, float) for m, v in p.centroids.items()}
    drift = {m: 1 - float(np.asarray(p.centroids[m], float) @ base[m]) for m in p.centroids}
    tsim = np.mean([float(np.asarray(p.centroids[m], float) @ tgt_ref[m]) for m in p.centroids if m in tgt_ref])
    isim = np.mean([float(np.asarray(p.centroids[m], float) @ int_ref[m]) for m in p.centroids if m in int_ref])
    drift_s = " ".join(f"{m.split('/')[-1][:10]}={d:.3f}" for m, d in drift.items())
    print(f"{frac:>5.2f} | {drift_s} | tgt={tsim:.3f} intr={isim:.3f}", flush=True)

# 3. THRESHOLD MINI-ROC (other-voice cutoff) on an overlay recording ---------
print("\n=== 3. THRESHOLD SENSITIVITY (overlay intruder on [6,10]s; vary cutoff) ===", flush=True)
overlay = compose_other_voice(TARGET, INTRUDER, [(6.0, 10.0)], intruder_gain=2.0).audio
det = detect(overlay)
print(f"{'cutoff':>7} {'detect_rate':>12} {'false_pos':>10}", flush=True)
for cut in [0.3, 0.4, 0.5, 0.6, 0.7]:
    res = compare_recording_to_profile(det, profile.centroids, profile.calibration_band, other_voice_threshold=cut)
    inr = [r for r in res if 6.0 <= 0.5 * (r.start + r.end) <= 10.0]
    out = [r for r in res if not (6.0 <= 0.5 * (r.start + r.end) <= 10.0)]
    dr = sum(r.flag == "other_voice" for r in inr) / len(inr) if inr else float("nan")
    fp = sum(r.flag == "other_voice" for r in out) / len(out) if out else float("nan")
    print(f"{cut:>7.2f} {dr:>12.2f} {fp:>10.2f}", flush=True)

print("\n=== sweep done ===", flush=True)
PY
echo "=== Finished $(date) ==="
