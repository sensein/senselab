#!/usr/bin/env python3
"""Synthetic degradation suite generator (spec T039, SC-001/006).

Takes a clean reference clip and emits variants with *localized, known* degradations
plus a machine-readable manifest of the injected spans — the ground truth the
adaptive loop is scored against (SC-001: are injected spans proposed as regions and
either improved or correctly explained?).

Variants (all spans recorded in manifest.json):
- ``noise``:   additive white noise burst over one span
- ``clip``:    hard digital clipping over one span
- ``lowpass``: telephone-band (~3.4 kHz) filtering over one span
- ``silence``: the span is zeroed (tests presence + FR-004 interactions)
- ``music``:   additive synthetic "music" (chord of sines) — hallucination bait (SC-006)

Usage:
    uv run python scripts/make_degradation_suite.py tutorial_audio_files/audio_48khz_mono_16bits.wav \
        --out artifacts/degradation_suite
    # then run analyze_audio.py + scripts/adaptive_loop.py on each variant and check
    # SC-001 via src/tests/audio/workflows/audio_analysis/adaptive/adaptive_e2e_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Generate the suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/degradation_suite"))
    parser.add_argument(
        "--span-start-fraction", type=float, default=0.4, help="Injected span start (fraction of duration)"
    )
    parser.add_argument("--span-length-s", type=float, default=3.0)
    parser.add_argument("--noise-std", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(args.audio), dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    duration = len(wav) / sr
    start = round(args.span_start_fraction * duration, 3)
    end = round(min(duration, start + args.span_length_s), 3)
    lo, hi = int(start * sr), int(end * sr)
    rng = np.random.default_rng(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": str(args.audio),
        "sample_rate": sr,
        "duration_s": round(duration, 3),
        "seed": args.seed,
        "variants": {},
    }

    def _emit(name: str, y: "np.ndarray", kind: str) -> None:
        path = args.out / f"{args.audio.stem}__{name}.wav"
        sf.write(str(path), np.clip(y, -1.0, 1.0), sr)
        manifest["variants"][name] = {  # type: ignore[index]
            "path": str(path),
            "kind": kind,
            "injected_span": [start, end],
        }
        print(f"  {name}: {path}  span=[{start}, {end}]")

    noise = wav.copy()
    noise[lo:hi] += rng.normal(0, args.noise_std, hi - lo).astype("float32")
    _emit("noise", noise, "additive_noise")

    clip = wav.copy()
    clip[lo:hi] = np.clip(clip[lo:hi] * 8.0, -0.98, 0.98)
    _emit("clip", clip, "hard_clipping")

    from scipy.signal import butter, sosfilt

    sos = butter(6, 3400, btype="low", fs=sr, output="sos")
    lowpass = wav.copy()
    lowpass[lo:hi] = sosfilt(sos, lowpass[lo:hi]).astype("float32")
    _emit("lowpass", lowpass, "band_limited")

    silence = wav.copy()
    silence[lo:hi] = rng.normal(0, 1e-4, hi - lo).astype("float32")
    _emit("silence", silence, "zeroed_span")

    t = np.arange(hi - lo) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6)) / 3.0
    music = wav.copy()
    music[lo:hi] += (0.15 * chord).astype("float32")
    _emit("music", music, "additive_music")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
