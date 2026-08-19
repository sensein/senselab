"""What survives enhancement, read by HeAR and YAMNet, across an SNR sweep.

Existing comparisons measured speech fidelity (SI-SDR) or element energy at one condition. This
measures whether the readers the triage design would use still detect each element after
enhancement, at every SNR -- which energy retention does not imply. See design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.audio.tasks.source_separation import separate_audios
from senselab.audio.tasks.speech_enhancement.api import enhance_audios
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel, SpeechBrainModel

# YAMNet carries 521 AudioSet labels. Anything less than all of them turns "outside the top k"
# into a zero, which reads identically to "the enhancer destroyed it".
YAMNET_ALL_LABELS = 521

# Verified events, from ground-truth-2026-08-18.md. Rows verified as onsets carry a nominal extent.
EVENTS: List[Dict[str, Any]] = [
    {
        "name": "mouth",
        "start": 0.893,
        "end": 1.095,
        "yamnet": ["Lip smacking", "Mouth", "Chewing, mastication"],
        "hear": [],
    },
    {"name": "breath_1", "start": 2.275, "end": 3.496, "yamnet": ["Breathing", "Exhalation"], "hear": ["Breathe"]},
    {"name": "breath_2", "start": 5.308, "end": 6.291, "yamnet": ["Breathing", "Exhalation"], "hear": ["Breathe"]},
    {"name": "cough_1", "start": 7.926, "end": 8.494, "yamnet": ["Cough"], "hear": ["Cough"]},
    {"name": "cough_2", "start": 9.610, "end": 10.250, "yamnet": ["Cough"], "hear": ["Cough"]},
    {"name": "speech", "start": 11.62, "end": 13.20, "yamnet": ["Speech"], "hear": ["Speech"]},
]

# Verified-empty stretches, so a rise here separates "preserved the element" from "invented one".
EMPTY: List[Tuple[float, float]] = [
    (0.0, 0.78),
    (1.0, 2.3),
    (3.5, 5.3),
    (6.3, 7.9),
    (8.5, 9.6),
    (10.25, 11.65),
    (13.2, 13.79),
]

# (row, backend, checkpoint, required input rate, backend parameters). Every enhancer refuses audio
# at another rate, so the input is resampled per row -- and a rate-matched unenhanced baseline is
# emitted for each rate present, so no enhanced row is compared against a reference at another rate.
# DriftSE's checkpoint is selected by a ``variant`` parameter, not by the repo id.
ENHANCERS: List[Tuple[str, Optional[Tuple[str, str]], int, Dict[str, Any]]] = [
    ("input@16k", None, 16000, {}),
    ("input@48k", None, 48000, {}),
    ("sepformer-wham16k-enh", ("speechbrain", "speechbrain/sepformer-wham16k-enhancement"), 16000, {}),
    ("sepformer-whamr16k", ("speechbrain", "speechbrain/sepformer-whamr16k"), 16000, {}),
    ("sepformer-dns4-16k-enh", ("speechbrain", "speechbrain/sepformer-dns4-16k-enhancement"), 16000, {}),
    ("metricgan-plus-voicebank", ("speechbrain", "speechbrain/metricgan-plus-voicebank"), 16000, {}),
    ("FRCRN_SE_16K", ("clearvoice", "alibabasglab/FRCRN_SE_16K"), 16000, {}),
    ("MossFormerGAN_SE_16K", ("clearvoice", "alibabasglab/MossFormerGAN_SE_16K"), 16000, {}),
    ("MossFormer2_SE_48K", ("clearvoice", "alibabasglab/MossFormer2_SE_48K"), 48000, {}),
    ("DriftSE_v1", ("driftse", "LIANGXU123/DriftSE"), 16000, {"variant": "distillhubert_three_layers_with_z"}),
    (
        "DriftSE_v2",
        ("driftse", "LIANGXU123/DriftSE"),
        16000,
        {"variant": "distillhubert_three_layers_pesq_sisdr_ccmse_with_z"},
    ),
]

# Separation is a form of enhancement for this question: each output channel is a candidate, and
# the old matrix's most interesting finding was a separator acting as an element filter. Scored per
# channel, so "cough survives in src1" is visible rather than averaged away.
#
# SpeechBrain separators (sepformer-whamr16k, sepformer-wsj02mix) are NOT here because senselab has
# no SpeechBrain separation backend -- separate_audios dispatches only to ClearVoice and unasdiff.
# The #569 guard tells a caller to use source_separation for those checkpoints, and that module
# cannot load them. Recorded so the absence is not read as a choice.
SEPARATORS: List[Tuple[str, str, int, Dict[str, Any]]] = [
    ("MossFormer2_SS_16K", "alibabasglab/MossFormer2_SS_16K", 16000, {}),
    ("unasdiff_speech_sound_Cough", "", 16000, {"mode": "speech_sound", "source_classes": ["Cough"]}),
    # The label-inert control: an unrelated class matched Cough to 0.4% at 60 steps, pre-#564.
    ("unasdiff_speech_sound_Keyboard", "", 16000, {"mode": "speech_sound", "source_classes": ["Computer_keyboard"]}),
]

SNRS: List[Optional[float]] = [None, 20.0, 10.0, 5.0, 0.0, -5.0]  # None = the recording as captured


def add_noise(waveform: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    """Return ``waveform`` with white noise at the requested SNR.

    Args:
        waveform: Channels-first samples.
        snr_db: Target signal-to-noise ratio in dB.
        seed: RNG seed, so a rerun reproduces the mixture.

    Returns:
        The noisy waveform, same shape and dtype.
    """
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(waveform.shape, generator=g, dtype=waveform.dtype)
    scale = (float((waveform**2).mean()) / (float((noise**2).mean()) * 10.0 ** (snr_db / 10.0))) ** 0.5
    return waveform + noise * scale


def build_model(kind: str, ident: str) -> SenselabModel:
    """Return the model spec ``enhance_audios`` needs for a backend.

    Args:
        kind: One of ``speechbrain``, ``clearvoice``, ``driftse``.
        ident: The checkpoint id.

    Returns:
        A senselab model spec.

    Raises:
        ValueError: On an unknown backend kind.
    """
    if kind == "speechbrain":
        return SpeechBrainModel(path_or_uri=ident, revision="main")
    if kind in ("clearvoice", "driftse"):
        return HFModel(path_or_uri=ident, revision="main")
    raise ValueError(f"unknown backend kind {kind!r}")


def flatten(window: Dict[str, Any]) -> Dict[str, float]:
    """Return one window's ``label_scores`` as a flat mapping.

    Both readers emit a list of single-key dicts, descending by score.

    Args:
        window: One per-window result dict.

    Returns:
        Label to score.
    """
    return {k: float(v) for d in window.get("label_scores", []) for k, v in d.items()}


def peak_over(windows: List[Dict[str, Any]], labels: List[str], start: float, end: float) -> Dict[str, float]:
    """Return each label's highest score over the windows overlapping ``[start, end)``.

    Args:
        windows: Per-window results for one recording.
        labels: Labels to report.
        start: Interval start, seconds.
        end: Interval end, seconds.

    Returns:
        Label to peak score; 0.0 where no window overlaps.
    """
    best = {label: 0.0 for label in labels}
    for w in windows:
        if float(w["end"]) <= start or float(w["start"]) >= end:
            continue
        flat = flatten(w)
        for label in labels:
            best[label] = max(best[label], flat.get(label, 0.0))
    return best


def read(audio: Audio) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    """Run both readers over a whole recording.

    Whole-recording rather than per-event clips: HeAR refuses audio shorter than 2 s, and every
    verified event here is shorter than that.

    Args:
        audio: The recording to read.

    Returns:
        YAMNet windows, HeAR windows, and a status map naming any reader that failed.
    """
    status: Dict[str, str] = {}
    try:
        yam = classify_audios([audio], model="yamnet", top_k=YAMNET_ALL_LABELS)[0]
    except Exception as exc:
        yam, status["yamnet"] = [], f"error: {exc}"
    try:
        hear = detect_health_acoustic_events([audio])[0]
    except Exception as exc:
        hear, status["hear"] = [], f"error: {exc}"
    return yam, hear, status


def main() -> int:
    """Run the sweep and write results.json.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    device = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU
    base = Audio(filepath=args.audio)
    dur = base.waveform.shape[-1] / base.sampling_rate
    print(f"input={args.audio} sr={base.sampling_rate} dur={dur:.3f}s device={device.value}", flush=True)

    rows: List[Dict[str, Any]] = []
    for snr in SNRS:
        noisy = (
            base
            if snr is None
            else Audio(waveform=add_noise(base.waveform, snr, args.seed), sampling_rate=base.sampling_rate)
        )
        for name, spec, rate, params in ENHANCERS:
            print(f"[snr={snr} enhancer={name} rate={rate}]", flush=True)
            try:
                fed = noisy if noisy.sampling_rate == rate else resample_audios([noisy], rate)[0]
                out = (
                    fed
                    if spec is None
                    else enhance_audios([fed], model=build_model(*spec), device=device, parameters=params or None)[0]
                )
            except Exception as exc:
                rows.append(
                    {
                        "snr": snr,
                        "enhancer": name,
                        "scope": "*",
                        "reader": "*",
                        "label": "*",
                        "score": None,
                        "status": f"enhance failed: {exc}",
                    }
                )
                print(f"    enhance failed: {exc}", flush=True)
                args.out.write_text(json.dumps(rows, indent=2))
                continue

            yam, hear, status = read(out)
            for reader, windows in (("yamnet", yam), ("hear", hear)):
                if reader in status:
                    rows.append(
                        {
                            "snr": snr,
                            "enhancer": name,
                            "scope": "*",
                            "reader": reader,
                            "label": "*",
                            "score": None,
                            "status": status[reader],
                        }
                    )
                    continue
                for ev in EVENTS:
                    labels = ev[reader]
                    if not labels:
                        continue
                    for label, score in peak_over(windows, labels, ev["start"], ev["end"]).items():
                        rows.append(
                            {
                                "snr": snr,
                                "enhancer": name,
                                "scope": ev["name"],
                                "reader": reader,
                                "label": label,
                                "score": score,
                                "status": "ok",
                            }
                        )
                # the same labels over verified-empty audio: a rise here is an invented event
                empty_labels = sorted({lab for ev in EVENTS for lab in ev[reader]})
                worst = {lab: 0.0 for lab in empty_labels}
                for a, b in EMPTY:
                    for lab, sc in peak_over(windows, empty_labels, a, b).items():
                        worst[lab] = max(worst[lab], sc)
                for label, score in worst.items():
                    rows.append(
                        {
                            "snr": snr,
                            "enhancer": name,
                            "scope": "verified_empty",
                            "reader": reader,
                            "label": label,
                            "score": score,
                            "status": "ok",
                        }
                    )
            args.out.write_text(json.dumps(rows, indent=2))  # checkpoint after every cell

        for name, ident, rate, params in SEPARATORS:
            print(f"[snr={snr} separator={name} rate={rate}]", flush=True)
            try:
                fed = noisy if noisy.sampling_rate == rate else resample_audios([noisy], rate)[0]
                model = HFModel(path_or_uri=ident, revision="main") if ident else None
                sources = separate_audios([fed], model=model, n_sources=2, device=device, **params)[0]
            except Exception as exc:
                rows.append(
                    {
                        "snr": snr,
                        "enhancer": name,
                        "scope": "*",
                        "reader": "*",
                        "label": "*",
                        "score": None,
                        "status": f"separate failed: {exc}",
                    }
                )
                print(f"    separate failed: {exc}", flush=True)
                args.out.write_text(json.dumps(rows, indent=2))
                continue

            for idx, src in enumerate(sources):
                row_name = f"{name}#src{idx}"
                yam, hear, status = read(src)
                for reader, windows in (("yamnet", yam), ("hear", hear)):
                    if reader in status:
                        rows.append(
                            {
                                "snr": snr,
                                "enhancer": row_name,
                                "scope": "*",
                                "reader": reader,
                                "label": "*",
                                "score": None,
                                "status": status[reader],
                            }
                        )
                        continue
                    for ev in EVENTS:
                        if not ev[reader]:
                            continue
                        for label, score in peak_over(windows, ev[reader], ev["start"], ev["end"]).items():
                            rows.append(
                                {
                                    "snr": snr,
                                    "enhancer": row_name,
                                    "scope": ev["name"],
                                    "reader": reader,
                                    "label": label,
                                    "score": score,
                                    "status": "ok",
                                }
                            )
                    empty_labels = sorted({lab for ev in EVENTS for lab in ev[reader]})
                    worst = {lab: 0.0 for lab in empty_labels}
                    for a, b in EMPTY:
                        for lab, sc in peak_over(windows, empty_labels, a, b).items():
                            worst[lab] = max(worst[lab], sc)
                    for label, score in worst.items():
                        rows.append(
                            {
                                "snr": snr,
                                "enhancer": row_name,
                                "scope": "verified_empty",
                                "reader": reader,
                                "label": label,
                                "score": score,
                                "status": "ok",
                            }
                        )
                args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
