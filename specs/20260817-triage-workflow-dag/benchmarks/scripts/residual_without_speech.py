#!/usr/bin/env python3
"""Measure what FRCRN_SE_16K does to non-speech (cough/breath) recordings, and whether
``original - g*FRCRN(original)`` means anything there.

Runs FRCRN_SE_16K (senselab's ClearerVoice enhancement backend) over a fixed list of b2ai
recordings -- eight respiration/cough recordings across three subjects plus two Story-recall
(speech) comparators -- then calls `senselab.audio.tasks.speech_enhancement.residual.compute_residual`
directly for the lag search, gain fit, subtraction and band split, and runs one batched YAMNet call
over every original/enhanced/residual file.

This used to go through ``~/Downloads/buzz_separation_20260906/residual/subtract.py`` (imported
from its path at runtime) rather than duplicating its logic. That script now itself calls
``compute_residual`` rather than reimplementing the same math, so calling either would compute the
same numbers; this script now calls the library function directly, per the same one-implementation
rule, rather than going through a second script that also delegates to it.

Writes:
    - ``<out-dir>/<key>__original.wav``, ``__frcrn.wav``, ``__residual.wav`` (16 kHz PCM_16)
    - ``<out-dir>/subtract_reports.json``  (one report per recording, same shape as before)
    - ``<out-dir>/yamnet_by_file.json``    (max score per label, per file, over all windows)

Usage:
    uv run python residual_without_speech.py --out-dir ~/Downloads/frcrn_residual_no_speech_20260907
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

BIDS_ROOT = Path("~/Downloads/b2ai_v31_bids_07_01_v3").expanduser()
BANDS_HZ = [(0.0, 200.0), (200.0, 1000.0), (1000.0, 4000.0), (4000.0, 8000.0)]

# key, subject, task, is_speech, path relative to BIDS_ROOT
RECORDINGS: list[dict[str, Any]] = [
    dict(
        key="s1_cough1",
        subject="sub-17578482",
        task="Respiration-and-cough-Cough-1",
        is_speech=False,
        relpath=(
            "sub-17578482-8511-494a-8dfb-9113a7ffb63c/ses-A8D26790-951D-49A2-84A1-77FB52C5BD42/audio/"
            "sub-17578482-8511-494a-8dfb-9113a7ffb63c_ses-A8D26790-951D-49A2-84A1-77FB52C5BD42_"
            "task-Respiration-and-cough-Cough-1.wav"
        ),
    ),
    dict(
        key="s1_cough2",
        subject="sub-17578482",
        task="Respiration-and-cough-Cough-2",
        is_speech=False,
        relpath=(
            "sub-17578482-8511-494a-8dfb-9113a7ffb63c/ses-A8D26790-951D-49A2-84A1-77FB52C5BD42/audio/"
            "sub-17578482-8511-494a-8dfb-9113a7ffb63c_ses-A8D26790-951D-49A2-84A1-77FB52C5BD42_"
            "task-Respiration-and-cough-Cough-2.wav"
        ),
    ),
    dict(
        key="s1_fivebreaths1",
        subject="sub-17578482",
        task="Respiration-and-cough-FiveBreaths-1",
        is_speech=False,
        relpath=(
            "sub-17578482-8511-494a-8dfb-9113a7ffb63c/ses-A8D26790-951D-49A2-84A1-77FB52C5BD42/audio/"
            "sub-17578482-8511-494a-8dfb-9113a7ffb63c_ses-A8D26790-951D-49A2-84A1-77FB52C5BD42_"
            "task-Respiration-and-cough-FiveBreaths-1.wav"
        ),
    ),
    dict(
        key="s2_cough1",
        subject="sub-17cee767",
        task="Respiration-and-cough-Cough-1",
        is_speech=False,
        relpath=(
            "sub-17cee767-1864-457a-b2ec-446a058a81f8/ses-DA790C5A-93FF-432F-A5B6-418C19A4F2BA/audio/"
            "sub-17cee767-1864-457a-b2ec-446a058a81f8_ses-DA790C5A-93FF-432F-A5B6-418C19A4F2BA_"
            "task-Respiration-and-cough-Cough-1.wav"
        ),
    ),
    dict(
        key="s2_breath1",
        subject="sub-17cee767",
        task="Respiration-and-cough-Breath-1",
        is_speech=False,
        relpath=(
            "sub-17cee767-1864-457a-b2ec-446a058a81f8/ses-DA790C5A-93FF-432F-A5B6-418C19A4F2BA/audio/"
            "sub-17cee767-1864-457a-b2ec-446a058a81f8_ses-DA790C5A-93FF-432F-A5B6-418C19A4F2BA_"
            "task-Respiration-and-cough-Breath-1.wav"
        ),
    ),
    dict(
        key="s2_threequickbreaths1",
        subject="sub-17cee767",
        task="Respiration-and-cough-ThreeQuickBreaths-1",
        is_speech=False,
        relpath=(
            "sub-17cee767-1864-457a-b2ec-446a058a81f8/ses-DA790C5A-93FF-432F-A5B6-418C19A4F2BA/audio/"
            "sub-17cee767-1864-457a-b2ec-446a058a81f8_ses-DA790C5A-93FF-432F-A5B6-418C19A4F2BA_"
            "task-Respiration-and-cough-ThreeQuickBreaths-1.wav"
        ),
    ),
    dict(
        key="s3_hardcough",
        subject="sub-1f4ea26f",
        task="Respiration-and-cough-(v2)-HardCough",
        is_speech=False,
        relpath=(
            "sub-1f4ea26f-9764-4f89-a41a-66e248b9386f/ses-D987B8B0-A963-49B9-ABBE-2FD8FCC83E81/audio/"
            "sub-1f4ea26f-9764-4f89-a41a-66e248b9386f_ses-D987B8B0-A963-49B9-ABBE-2FD8FCC83E81_"
            "task-Respiration-and-cough-(v2)-HardCough.wav"
        ),
    ),
    dict(
        key="s3_breath",
        subject="sub-1f4ea26f",
        task="Respiration-and-cough-(v2)-Breath",
        is_speech=False,
        relpath=(
            "sub-1f4ea26f-9764-4f89-a41a-66e248b9386f/ses-D987B8B0-A963-49B9-ABBE-2FD8FCC83E81/audio/"
            "sub-1f4ea26f-9764-4f89-a41a-66e248b9386f_ses-D987B8B0-A963-49B9-ABBE-2FD8FCC83E81_"
            "task-Respiration-and-cough-(v2)-Breath.wav"
        ),
    ),
    dict(
        key="s1_storyrecall",
        subject="sub-17578482",
        task="Story-recall",
        is_speech=True,
        relpath=(
            "sub-17578482-8511-494a-8dfb-9113a7ffb63c/ses-A8D26790-951D-49A2-84A1-77FB52C5BD42/audio/"
            "sub-17578482-8511-494a-8dfb-9113a7ffb63c_ses-A8D26790-951D-49A2-84A1-77FB52C5BD42_"
            "task-Story-recall.wav"
        ),
    ),
    dict(
        key="s3_storyrecall",
        subject="sub-1f4ea26f",
        task="Story-recall-(v2)",
        is_speech=True,
        relpath=(
            "sub-1f4ea26f-9764-4f89-a41a-66e248b9386f/ses-D987B8B0-A963-49B9-ABBE-2FD8FCC83E81/audio/"
            "sub-1f4ea26f-9764-4f89-a41a-66e248b9386f_ses-D987B8B0-A963-49B9-ABBE-2FD8FCC83E81_"
            "task-Story-recall-(v2).wav"
        ),
    ),
]

WATCH_LABELS = [
    "Cough",
    "Breathing",
    "Sneeze",
    "Throat clearing",
    "Snoring",
    "Speech",
    "Buzz",
    "Hum",
    "Mains hum",
    "Noise",
]


def _dbfs_peak(x: np.ndarray) -> float:
    """Peak level in dBFS; ``-inf`` for a silent signal."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * np.log10(peak) if peak > 0 else float("-inf")


def _dbfs_rms(x: np.ndarray) -> float:
    """RMS level in dBFS; ``-inf`` for a silent signal."""
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


def _write_wav_pcm16(path: Path, x: np.ndarray, sr: int) -> dict[str, Any]:
    """Write ``x`` as 16-bit PCM, scaling down (and reporting the gain) only if it would clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    applied_gain_db = 0.0
    out = x
    if peak > 0.999:
        scale = 0.999 / peak
        out = x * scale
        applied_gain_db = 20 * np.log10(scale)
    sf.write(str(path), out, sr, subtype="PCM_16")
    return {"path": str(path), "input_peak": peak, "clip_avoidance_gain_db": applied_gain_db}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="Directory to write audio and JSON reports into.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.classification.api import classify_audios
    from senselab.audio.tasks.classification.label_scores import label_scores
    from senselab.audio.tasks.speech_enhancement.api import enhance_audios
    from senselab.audio.tasks.speech_enhancement.residual import band_energy_fractions, compute_residual
    from senselab.utils.data_structures import HFModel

    for rec in RECORDINGS:
        src = BIDS_ROOT / rec["relpath"]
        if not src.exists():
            raise FileNotFoundError(src)

    print(f"Loading {len(RECORDINGS)} recordings...", file=sys.stderr)
    audios = [Audio(filepath=str(BIDS_ROOT / rec["relpath"])) for rec in RECORDINGS]

    model = HFModel(path_or_uri="alibabasglab/FRCRN_SE_16K")
    print("Running FRCRN_SE_16K over all recordings in one call...", file=sys.stderr)
    enhanced = enhance_audios(audios, model=model)
    if len(enhanced) != len(RECORDINGS):
        raise RuntimeError(f"Expected {len(RECORDINGS)} enhanced outputs, got {len(enhanced)}")

    orig_paths: list[str] = []
    enh_paths: list[str] = []
    res_paths: list[str] = []
    reports: list[dict[str, Any]] = []

    for rec, enh in zip(RECORDINGS, enhanced):
        key = rec["key"]
        orig_out = out_dir / f"{key}__original.wav"
        enh_out = out_dir / f"{key}__frcrn.wav"

        shutil.copyfile(BIDS_ROOT / rec["relpath"], orig_out)
        enh.save_to_file(str(enh_out), subtype="PCM_16", out_of_range="warn")

        print(f"Computing residual for {key}...", file=sys.stderr)
        ref, sr_ref = sf.read(str(orig_out), dtype="float64", always_2d=True)
        ref = ref.mean(axis=1)
        sig, sr_sig = sf.read(str(enh_out), dtype="float64", always_2d=True)
        sig = sig.mean(axis=1)
        if sr_sig != sr_ref:
            raise RuntimeError(f"{key}: original ({sr_ref} Hz) and enhanced ({sr_sig} Hz) sampling rates differ")
        computation = compute_residual(ref, sig, sr_ref, max_lag_ms=200.0)

        res_out = out_dir / f"{key}__residual.wav"
        write_report = _write_wav_pcm16(res_out, computation.residual, sr_ref)

        # Same report shape `subtract.py`'s own JSON report used, so this reproduces the published
        # per-recording table directly against these field names -- built by calling the shared
        # library rather than re-deriving the numbers a second way.
        report: dict[str, Any] = {
            "input": str(orig_out),
            "streams": [str(enh_out)],
            "sampling_rate": sr_ref,
            "n_samples_aligned": len(computation.reference_aligned),
            "lag_samples_per_stream": {str(enh_out): computation.lag_samples},
            "lag_ms_per_stream": {str(enh_out): computation.lag_ms},
            "fitted_gain": computation.gain,
            "fitted_gain_db": computation.gain_db,
            "input_on_aligned_region": {
                "peak_dbfs": _dbfs_peak(computation.reference_aligned),
                "rms_dbfs": _dbfs_rms(computation.reference_aligned),
                "bands": band_energy_fractions(computation.reference_aligned, sr_ref, BANDS_HZ),
            },
            "direct": {
                "peak_dbfs": _dbfs_peak(computation.residual),
                "rms_dbfs": _dbfs_rms(computation.residual),
                "energy_fraction_of_input": computation.residual_energy_fraction,
                "energy_below_input_db": (
                    10.0 * np.log10(computation.residual_energy_fraction)
                    if computation.residual_energy_fraction > 0
                    else float("-inf")
                ),
                "bands": band_energy_fractions(computation.residual, sr_ref, BANDS_HZ),
                "write": write_report,
            },
            "enhanced_energy_fraction": computation.signal_energy_fraction,
            "correlation_input_vs_combined_stream": computation.correlation_signal,
            "correlation_residual_vs_input": computation.correlation_residual,
            "key": key,
            "subject": rec["subject"],
            "task": rec["task"],
            "is_speech": rec["is_speech"],
        }
        reports.append(report)

        orig_paths.append(str(orig_out))
        enh_paths.append(str(enh_out))
        res_paths.append(str(res_out))

    (out_dir / "subtract_reports.json").write_text(json.dumps(reports, indent=2))

    all_paths = orig_paths + enh_paths + res_paths
    all_roles = ["original"] * len(RECORDINGS) + ["enhanced"] * len(RECORDINGS) + ["residual"] * len(RECORDINGS)
    all_keys = [r["key"] for r in RECORDINGS] * 3

    print(f"Running one batched YAMNet call over {len(all_paths)} files...", file=sys.stderr)
    all_audios = [Audio(filepath=p) for p in all_paths]
    yam_results = classify_audios(all_audios, model="yamnet", top_k=521)

    def aggregate_max(windows: list[dict[str, Any]]) -> dict[str, float]:
        """Max score per label across every 0.96s window in the file."""
        best: dict[str, float] = {}
        for w in windows:
            for pair in label_scores(w):
                for label, score in pair.items():
                    if label not in best or score > best[label]:
                        best[label] = score
        return best

    yam_by_file: dict[str, dict[str, Any]] = {}
    for path, role, key, windows in zip(all_paths, all_roles, all_keys, yam_results):
        agg = aggregate_max(windows)
        top5 = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:5]
        watch = {lbl: agg.get(lbl, 0.0) for lbl in WATCH_LABELS}
        yam_by_file[path] = {
            "key": key,
            "role": role,
            "top5": top5,
            "watch": watch,
        }

    (out_dir / "yamnet_by_file.json").write_text(json.dumps(yam_by_file, indent=2))

    print(json.dumps({"n_recordings": len(RECORDINGS), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
