#!/usr/bin/env python
r"""Probe whether the scene classifiers normalize input level (T018, FR-013 to FR-017).

Runs each classifier over one recording at several known gains and derives an
amplitude-invariance verdict per classifier. Also probes digital silence, to capture the
fixed response a classifier saturates to below its floor.

**Never downloads.** A classifier whose checkpoint is not already cached is skipped with a
message naming what is missing (constitution VI). The sweep is a diagnostic, not a reason
to pull gigabytes.

Usage:
    uv run python scripts/probe_classifier_levels.py \\
        --input src/tests/data_for_testing/<clip>.wav \\
        --gains-db -40 -20 -10 0 10 \\
        --out artifacts/level_probe/

Output: ``<out>/level-verdicts.json`` per ``contracts/level-verdicts.md``.

Interpreting the result:
    Both classifiers are expected to report ``level_sensitive``. That is a measured
    finding, not a hypothesis — a ``self_normalizing`` verdict means either the probe is
    wrong or the model changed, and both warrant investigation rather than a quiet
    threshold edit downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# Mechanisms established by the code audit, recorded alongside each verdict so the
# empirical result can be checked against the implementation (FR-016).
_MECHANISMS: dict[str, dict[str, str]] = {
    "ast": {
        "floor_mechanism": (
            "fixed dataset-level affine normalization (AudioSet mean/std constants, not "
            "per-example statistics); log(float32 eps) floor; the 2**15 pre-scale is "
            "commented out upstream"
        ),
        "mechanism_source": (
            "transformers/models/audio_spectrogram_transformer/"
            "feature_extraction_audio_spectrogram_transformer.py:75-77,113,156"
        ),
        "notes": (
            "Because the constants are global, the normalization cannot cancel a "
            "per-recording level offset: a gain becomes a rigid shift of every input bin."
        ),
    },
    "yamnet": {
        "floor_mechanism": (
            "log(mel + 0.001) on a magnitude mel, no normalization op in the graph; the "
            "collapse to silence is a learned absolute-level decision, not the log offset"
        ),
        "mechanism_source": "tfhub yamnet/1 saved_model graph: AddV2 const 0.001 feeding Log; no Square op",
        "notes": (
            "The silence collapse fires well above the level at which the log offset "
            "starts flooring bins, so it cannot be tuned away by changing the offset. It is "
            "monotone and source-independent, which makes it a usable level tripwire."
        ),
    },
}

_WINDOWS = {"ast": (10.24, 10.24), "yamnet": (0.96, 0.48)}

DEFAULT_AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _ast_cached(model_id: str) -> bool:
    """True when the checkpoint is on disk, without contacting the Hub."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 — any failure means "not usable offline"
        return False


def _yamnet_cached() -> bool:
    """True when a TF-Hub module directory looks present locally."""
    import os
    import tempfile

    roots = [os.environ.get("TFHUB_CACHE_DIR"), str(Path(tempfile.gettempdir()) / "tfhub_modules")]
    for root in filter(None, roots):
        base = Path(root)
        if base.is_dir() and any(p.joinpath("saved_model.pb").exists() for p in base.iterdir() if p.is_dir()):
            return True
    return False


def _classify(audio: Any, classifier: str, top_k: int) -> list[Any]:  # noqa: ANN401
    """Run one classifier over one Audio, returning per-window dicts."""
    from senselab.audio.tasks.classification import classify_audios
    from senselab.audio.workflows.audio_analysis.harvesters import classification_windows
    from senselab.audio.workflows.audio_analysis.sound_sources import AUDIOSET_SCORE_FUNCTION

    win, hop = _WINDOWS[classifier]
    kwargs: dict[str, Any] = {"win_length": win, "hop_length": hop, "top_k": top_k}
    if classifier == "ast":
        from senselab.utils.data_structures import HFModel

        model: Any = HFModel(path_or_uri=DEFAULT_AST_MODEL)
        kwargs["function_to_apply"] = AUDIOSET_SCORE_FUNCTION
    else:
        model = "yamnet"
    return classification_windows(classify_audios([audio], model, **kwargs))


def _gained_audio(audio: Any, gain_db: float) -> Any:  # noqa: ANN401
    """Return a copy of ``audio`` scaled by ``gain_db``, in float — never clipped."""
    import torch

    from senselab.audio.data_structures import Audio
    from senselab.audio.workflows.audio_analysis.level import apply_gain_db

    scaled = apply_gain_db(audio.waveform.squeeze().numpy(), gain_db)
    return Audio(waveform=torch.tensor(scaled, dtype=torch.float32).unsqueeze(0), sampling_rate=audio.sampling_rate)


def _silence_like(audio: Any) -> Any:  # noqa: ANN401
    """Digital silence at the same duration, for floor-signature detection."""
    import torch

    from senselab.audio.data_structures import Audio

    return Audio(waveform=torch.zeros_like(audio.waveform), sampling_rate=audio.sampling_rate)


def probe(
    input_path: Path,
    gains_db: Sequence[float],
    *,
    classifiers: Sequence[str],
    top_k: int,
    include_silence: bool,
) -> dict[str, Any]:
    """Run the sweep and return the verdict document."""
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.classification.level_probe import verdict_from_sweep
    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
    from senselab.utils.tasks.cached_inference import audio_signature

    audio = Audio(filepath=str(input_path))
    if audio.waveform.shape[0] > 1:
        audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]

    verdicts: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for classifier in classifiers:
        available = _ast_cached(DEFAULT_AST_MODEL) if classifier == "ast" else _yamnet_cached()
        if not available:
            reason = (
                f"{DEFAULT_AST_MODEL} is not in the local HuggingFace cache"
                if classifier == "ast"
                else "no TF-Hub module directory found locally"
            )
            print(f"SKIP {classifier}: {reason}. Not downloading (constitution VI).", file=sys.stderr)
            skipped.append({"classifier": classifier, "reason": reason})
            continue

        per_gain: dict[float, list[Any]] = {}
        for gain in gains_db:
            per_gain[float(gain)] = _classify(_gained_audio(audio, float(gain)), classifier, top_k)
            print(f"  {classifier} @ {gain:+.1f} dB: {len(per_gain[float(gain)])} windows", file=sys.stderr)

        silence_windows = _classify(_silence_like(audio), classifier, top_k) if include_silence else []
        meta = _MECHANISMS.get(classifier, {})
        verdict = verdict_from_sweep(
            DEFAULT_AST_MODEL if classifier == "ast" else classifier,
            window_length_s=_WINDOWS[classifier][0],
            per_gain=per_gain,
            silence_windows=silence_windows,
            floor_mechanism=meta.get("floor_mechanism", ""),
            mechanism_source=meta.get("mechanism_source", ""),
            notes=meta.get("notes", ""),
        )
        verdict.require_corroboration()
        verdicts.append(verdict.to_json())
        print(f"  -> {classifier}: {verdict.verdict}", file=sys.stderr)

    return {
        "probe_version": "1",
        "clip": str(input_path),
        "clip_signature": audio_signature(audio),
        "gains_db": [float(g) for g in gains_db],
        "top_k": top_k,
        "verdicts": verdicts,
        "skipped": skipped,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the sweep, write the verdict document."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Probe clip.")
    parser.add_argument(
        "--gains-db",
        type=float,
        nargs="+",
        default=[-40.0, -20.0, -10.0, 0.0, 10.0],
        help="Gain points in dB. Must include 0 and span at least 30 dB (SC-005).",
    )
    parser.add_argument("--classifiers", nargs="+", default=["ast", "yamnet"], choices=["ast", "yamnet"])
    parser.add_argument("--out", type=Path, default=Path("artifacts/level_probe"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-include-silence-probe",
        dest="include_silence",
        action="store_false",
        help="Skip the digital-silence probe that captures each classifier's floor signature.",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"input not found: {args.input}")

    from senselab.audio.tasks.classification.level_probe import validate_gain_range

    try:
        validate_gain_range(args.gains_db)
    except ValueError as exc:
        parser.error(str(exc))

    doc = probe(
        args.input,
        args.gains_db,
        classifiers=args.classifiers,
        top_k=args.top_k,
        include_silence=args.include_silence,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "level-verdicts.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    if not doc["verdicts"]:
        print("no verdicts produced — every classifier was skipped", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
