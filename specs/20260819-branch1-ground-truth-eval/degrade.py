"""Write the degraded copies the robustness probe runs on.

Parameters are stated here rather than chosen by a library default: see measurement.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from ground_truth import WAV

OUT = Path(__file__).parent / "raw" / "audio"
SEED = 20260819

WHITE_SNRS = [20.0, 10.0, 5.0, 0.0]
PINK_SNRS = [10.0, 0.0]
REVERB_T60 = [0.3, 0.7]
REVERB_DRR_DB = 0.0


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * scale, n=n)
    return shaped / np.std(shaped)


def _add_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_rms = float(np.sqrt(np.mean(signal**2)))
    noise = noise / float(np.sqrt(np.mean(noise**2)))
    target_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    return signal + noise * target_rms


def _rir(sr: int, t60: float, drr_db: float, rng: np.random.Generator) -> np.ndarray:
    n = int(1.5 * t60 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-3.0 * np.log(10.0) * t / t60)
    tail[0] = 0.0
    tail /= float(np.sqrt(np.sum(tail**2)))
    rir = np.zeros(n)
    rir[0] = 1.0
    return rir + tail * (10.0 ** (-drr_db / 20.0))


def _reverb(signal: np.ndarray, sr: int, t60: float, rng: np.random.Generator) -> np.ndarray:
    rir = _rir(sr, t60, REVERB_DRR_DB, rng)
    wet = np.convolve(signal, rir, mode="full")[: len(signal)]
    dry_rms = float(np.sqrt(np.mean(signal**2)))
    wet_rms = float(np.sqrt(np.mean(wet**2)))
    return wet * (dry_rms / wet_rms)


def main() -> None:
    """Write every variant and a manifest recording what was done to each."""
    OUT.mkdir(parents=True, exist_ok=True)
    clean, sr = sf.read(WAV, dtype="float64", always_2d=False)
    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    rng = np.random.default_rng(SEED)

    variants: Dict[str, np.ndarray] = {"clean": clean}
    for snr in WHITE_SNRS:
        variants[f"white_snr{snr:g}"] = _add_noise(clean, rng.standard_normal(len(clean)), snr)
    for snr in PINK_SNRS:
        variants[f"pink_snr{snr:g}"] = _add_noise(clean, _pink(len(clean), rng), snr)
    for t60 in REVERB_T60:
        variants[f"reverb_t60_{t60:g}"] = _reverb(clean, sr, t60, rng)

    manifest: List[Dict[str, object]] = []
    for name, wave in variants.items():
        peak = float(np.max(np.abs(wave)))
        # Headroom scaling, recorded rather than silent: a variant whose peak exceeds full scale
        # would be clipped by the container and the classifier would respond to the clipping.
        gain = 1.0 if peak <= 0.99 else 0.99 / peak
        out = wave * gain
        path = OUT / f"{name}.wav"
        sf.write(str(path), out.astype(np.float32), sr, subtype="FLOAT")
        manifest.append(
            {
                "name": name,
                "path": str(path),
                "sampling_rate": sr,
                "n_samples": int(len(out)),
                "peak_before_gain": peak,
                "headroom_gain": gain,
                "rms_db": float(20.0 * np.log10(np.sqrt(np.mean(out**2)))),
            }
        )
        print(f"{name:18s} peak={peak:.4f} gain={gain:.4f} rms_db={manifest[-1]['rms_db']:.2f}")

    meta = {
        "seed": SEED,
        "source": WAV,
        "white_snrs_db": WHITE_SNRS,
        "pink_snrs_db": PINK_SNRS,
        "reverb_t60_s": REVERB_T60,
        "reverb_drr_db": REVERB_DRR_DB,
        "snr_definition": "full-file RMS of signal over full-file RMS of noise",
        "variants": manifest,
    }
    (OUT.parent / "degradations.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
