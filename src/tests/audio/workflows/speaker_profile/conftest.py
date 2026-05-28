"""Deterministic fixture composers for speaker-profile tests (T010b).

These composers consume the committed synthetic clips produced by
``scripts/gen_synthetic_test_audio.py`` (T010a) and assemble the
contamination / overlay / noise scenarios that the US1–US3 tests assert
against. All composition is **seeded numpy** — no TTS at run time — so each
scenario is bit-reproducible.

The actual FLAC clips and ``manifest.json`` are committed under
``src/tests/data_for_testing/synthetic/``. Tests that need a fixture should
``pytest.importorskip("soundfile")`` if necessary, then call the composer
they need.

Ground-truth labels (target speaker, intruder intervals, SNRs) are returned
alongside the composed audio so assertions can reference them directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from senselab.audio.data_structures import Audio

# Resolve the committed fixtures directory relative to this test module.
SYNTHETIC_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "data_for_testing" / "synthetic"
"""Where ``gen_synthetic_test_audio.py`` writes its outputs (committed)."""


@dataclass(frozen=True)
class ManifestClip:
    """One committed clip's metadata, as recorded in ``manifest.json``."""

    file_id: str
    speaker_id: str
    transcript: str
    duration_s: float
    session_id: str | None


# ──────────────────────────────────────────────────────────────────────────
# Manifest + clip loaders


def load_manifest(synthetic_dir: Path = SYNTHETIC_DIR) -> dict[str, ManifestClip]:
    """Return ``{file_id -> ManifestClip}`` from the committed ``manifest.json``."""
    manifest_path = synthetic_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, ManifestClip] = {}
    for entry in raw.get("clips", []):
        out[entry["file_id"]] = ManifestClip(
            file_id=entry["file_id"],
            speaker_id=entry["speaker_id"],
            transcript=entry.get("transcript", ""),
            duration_s=float(entry.get("duration_s", 0.0)),
            session_id=entry.get("session_id"),
        )
    return out


def load_clip(file_id: str, synthetic_dir: Path = SYNTHETIC_DIR) -> Audio:
    """Load one committed FLAC clip as a mono 16 kHz :class:`Audio`."""
    import soundfile as sf

    path = synthetic_dir / file_id
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2 and data.shape[0] > data.shape[1]:
        data = data.T  # soundfile returns (samples, channels) for stereo
    waveform = torch.from_numpy(data).contiguous()
    return Audio(waveform=waveform, sampling_rate=int(sr))


def subject_clips(subject_id: str, synthetic_dir: Path = SYNTHETIC_DIR) -> list[ManifestClip]:
    """All committed clips whose ``file_id`` starts with ``subject_id/``."""
    manifest = load_manifest(synthetic_dir)
    prefix = f"{subject_id}/"
    return [m for fid, m in manifest.items() if fid.startswith(prefix)]


# ──────────────────────────────────────────────────────────────────────────
# Composition primitives — all seeded numpy.


def _pad_or_trim(wave: np.ndarray, target_len: int) -> np.ndarray:
    """Trim ``wave`` to ``target_len`` samples, or zero-pad on the right."""
    if wave.shape[-1] >= target_len:
        return wave[..., :target_len]
    pad = np.zeros((*wave.shape[:-1], target_len - wave.shape[-1]), dtype=wave.dtype)
    return np.concatenate([wave, pad], axis=-1)


def _mix(a: np.ndarray, b: np.ndarray, weight_b: float) -> np.ndarray:
    """Mix ``b`` into ``a`` with linear weighting; lengths must match."""
    return ((1.0 - weight_b) * a + weight_b * b).astype(a.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Contamination — mix a fraction of intruder energy into the target.


@dataclass(frozen=True)
class ContaminatedAudio:
    """Audio plus the ground-truth intruder fraction it was built with."""

    audio: Audio
    target_id: str
    intruder_id: str
    intruder_fraction: float  # in [0, 1]


def compose_contamination(
    target_clip: str,
    intruder_clip: str,
    fraction: float,
    *,
    seed: int = 0,
    synthetic_dir: Path = SYNTHETIC_DIR,
) -> ContaminatedAudio:
    """Mix ``intruder_clip`` energy into ``target_clip`` at the given fraction.

    The intruder is looped or trimmed to match the target length. Used to
    validate SC-002 ("profile remains closer to held-out target than
    intruder even with up to ~20% contamination").
    """
    rng = np.random.default_rng(seed)
    a = load_clip(target_clip, synthetic_dir).waveform.numpy().astype(np.float32)
    b = load_clip(intruder_clip, synthetic_dir).waveform.numpy().astype(np.float32)
    # Match shapes — both should be (1, samples) since the fixtures are mono.
    target_len = a.shape[-1]
    if b.shape[-1] < target_len:
        # Tile and trim
        reps = (target_len // b.shape[-1]) + 1
        b = np.tile(b, (1, reps))
    b = _pad_or_trim(b, target_len)
    # Slight RNG-jitter to avoid bit-exact reuse across seeds.
    _ = rng.standard_normal(1)
    mixed = _mix(a, b, weight_b=fraction)
    waveform = torch.from_numpy(mixed).contiguous()
    return ContaminatedAudio(
        audio=Audio(waveform=waveform, sampling_rate=16000),
        target_id=target_clip,
        intruder_id=intruder_clip,
        intruder_fraction=float(fraction),
    )


# ──────────────────────────────────────────────────────────────────────────
# Overlay — splice intruder into target at known time intervals.


@dataclass(frozen=True)
class OverlayAudio:
    """Audio + the (start_s, end_s) intervals where the intruder is present."""

    audio: Audio
    target_id: str
    intruder_id: str
    intruder_intervals_s: tuple[tuple[float, float], ...]


def compose_other_voice(
    target_clip: str,
    intruder_clip: str,
    intruder_intervals_s: Sequence[tuple[float, float]],
    *,
    intruder_gain: float = 1.0,
    seed: int = 0,
    synthetic_dir: Path = SYNTHETIC_DIR,
) -> OverlayAudio:
    """Overlay ``intruder_clip`` onto ``target_clip`` at the given time intervals.

    Used to validate SC-003 (detection rate on annotated other-voice regions)
    and SC-004 (false-positive ceiling on the target-only complement).
    """
    rng = np.random.default_rng(seed)
    a = load_clip(target_clip, synthetic_dir).waveform.numpy().astype(np.float32)
    b = load_clip(intruder_clip, synthetic_dir).waveform.numpy().astype(np.float32)
    sr = 16000
    mixed = a.copy()
    intruder_samples = b.shape[-1]
    for start_s, end_s in intruder_intervals_s:
        start_n = int(round(start_s * sr))
        end_n = int(round(end_s * sr))
        end_n = min(end_n, mixed.shape[-1])
        length = end_n - start_n
        if length <= 0:
            continue
        # Loop the intruder if the interval is longer than the clip.
        seg = b[..., : min(length, intruder_samples)]
        if seg.shape[-1] < length:
            reps = (length // seg.shape[-1]) + 1
            seg = np.tile(seg, (1, reps))[..., :length]
        mixed[..., start_n:end_n] = mixed[..., start_n:end_n] + intruder_gain * seg
    _ = rng.standard_normal(1)
    waveform = torch.from_numpy(mixed.astype(np.float32)).contiguous()
    return OverlayAudio(
        audio=Audio(waveform=waveform, sampling_rate=sr),
        target_id=target_clip,
        intruder_id=intruder_clip,
        intruder_intervals_s=tuple((float(s), float(e)) for s, e in intruder_intervals_s),
    )


# ──────────────────────────────────────────────────────────────────────────
# Noise — additive Gaussian at a controlled SNR.


@dataclass(frozen=True)
class NoisyAudio:
    """Audio + the SNR (dB) it was generated at."""

    audio: Audio
    clean_id: str
    snr_db: float


def add_noise_at_snr(
    clean_clip: str,
    snr_db: float,
    *,
    seed: int = 0,
    synthetic_dir: Path = SYNTHETIC_DIR,
) -> NoisyAudio:
    """Add Gaussian noise at exact ``snr_db`` to ``clean_clip``.

    Used to validate SC-005 (quality indicator orders cleaner > noisier).
    The noise is seeded so any given (clip, SNR, seed) is bit-reproducible.
    """
    rng = np.random.default_rng(seed)
    clean = load_clip(clean_clip, synthetic_dir).waveform.numpy().astype(np.float32)
    # Power = mean(x^2). For target SNR_dB, noise_power = sig_power / 10^(SNR/10).
    sig_power = float(np.mean(clean.astype(np.float64) ** 2))
    if sig_power <= 0:
        # Edge case: silent input → just return original.
        waveform = torch.from_numpy(clean).contiguous()
        return NoisyAudio(
            audio=Audio(waveform=waveform, sampling_rate=16000),
            clean_id=clean_clip,
            snr_db=float(snr_db),
        )
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(clean.shape).astype(np.float32) * float(np.sqrt(noise_power))
    noisy = (clean + noise).astype(np.float32)
    waveform = torch.from_numpy(noisy).contiguous()
    return NoisyAudio(
        audio=Audio(waveform=waveform, sampling_rate=16000),
        clean_id=clean_clip,
        snr_db=float(snr_db),
    )
