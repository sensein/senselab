#!/usr/bin/env python3
"""Measure what FRCRN_SE_16K does to non-speech (cough/breath) recordings, and whether
``original - g*FRCRN(original)`` means anything there.

Runs FRCRN_SE_16K (senselab's ClearerVoice enhancement backend) over a fixed list of b2ai
recordings -- eight respiration/cough recordings across three subjects plus two Story-recall
(speech) comparators -- then reuses the existing, already-validated alignment/gain-fit/subtraction
script (``~/Downloads/buzz_separation_20260906/residual/subtract.py``) to compute the residual for
each, and runs one batched YAMNet call over every original/enhanced/residual file.

Writes:
    - ``<out-dir>/<key>__original.wav``, ``__frcrn.wav``, ``__residual.wav`` (16 kHz PCM_16)
    - ``<out-dir>/subtract_reports.json``  (one subtract.py report per recording)
    - ``<out-dir>/yamnet_by_file.json``    (max score per label, per file, over all windows)

Usage:
    uv run python residual_without_speech.py --out-dir ~/Downloads/frcrn_residual_no_speech_20260907
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

BIDS_ROOT = Path("~/Downloads/b2ai_v31_bids_07_01_v3").expanduser()
SUBTRACT_PY = Path("~/Downloads/buzz_separation_20260906/residual/subtract.py").expanduser()

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


def load_subtract_module() -> types.ModuleType:
    """Import ``subtract.py`` from its actual path rather than duplicating its logic."""
    spec = importlib.util.spec_from_file_location("subtract", SUBTRACT_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SUBTRACT_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    from senselab.utils.data_structures import HFModel

    subtract = load_subtract_module()

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

        print(f"Subtracting for {key}...", file=sys.stderr)
        result = subtract.process(str(orig_out), [str(enh_out)])
        report = result["report"]

        res_out = out_dir / f"{key}__residual.wav"
        write_report = subtract.write_wav_pcm16(res_out, result["residual_direct"], result["sr"])
        report["direct"]["write"] = write_report
        report["key"] = key
        report["subject"] = rec["subject"]
        report["task"] = rec["task"]
        report["is_speech"] = rec["is_speech"]
        reports.append(report)

        orig_paths.append(str(orig_out))
        enh_paths.append(str(enh_out))
        res_paths.append(str(res_out))

    (out_dir / "subtract_reports.json").write_text(json.dumps(reports, indent=2))

    all_paths = orig_paths + enh_paths + res_paths
    all_roles = (
        ["original"] * len(RECORDINGS) + ["enhanced"] * len(RECORDINGS) + ["residual"] * len(RECORDINGS)
    )
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
