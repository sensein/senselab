"""The quantised spectrogram the review page draws: a coarse time-by-band grid of a few levels.

The grid is ``frames`` time bins over the whole recording by ``bands`` log-spaced frequency bands,
each cell one of ``levels`` levels, packed two bits per cell and base64-encoded. Settings are in
``data/review_page.yaml``.
"""

from __future__ import annotations

import base64
from functools import cache
from importlib import resources
from typing import Any

import numpy as np
import yaml


@cache
def settings() -> dict[str, Any]:
    """The page settings.

    Returns:
        ``data/review_page.yaml``, parsed.
    """
    text = resources.files("senselab.audio.workflows.triage.data").joinpath("review_page.yaml").read_text()
    return dict(yaml.safe_load(text))


def _band_edges(bands: int, f_min: float, f_max: float) -> np.ndarray:
    return np.geomspace(f_min, f_max, bands + 1)


def band_powers(samples: np.ndarray, sampling_hz: float) -> np.ndarray:
    """Mean power per analysis frame and band.

    Args:
        samples: The waveform, mono or ``(channels, n)``; channels are averaged.
        sampling_hz: Its sampling rate.

    Returns:
        ``(n_frames, bands)`` mean power; one zero frame where the waveform is shorter than a window.
    """
    spec = settings()["spectrogram"]
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0) if x.shape[0] < x.shape[-1] else x.mean(axis=1)
    window = max(16, int(round(float(spec["window_s"]) * sampling_hz)))
    n_fft = 1 << (window - 1).bit_length()
    hop = max(1, window // 2)
    bands = int(spec["bands"])
    if x.size < window:
        return np.zeros((1, bands))
    n_frames = 1 + (x.size - window) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, window)[::hop][:n_frames]
    power = np.abs(np.fft.rfft(frames * np.hanning(window), n=n_fft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_hz)
    f_max = min(float(spec["f_max_hz"]), sampling_hz / 2.0)
    edges = _band_edges(bands, min(float(spec["f_min_hz"]), f_max / 2.0), f_max)
    out = np.zeros((power.shape[0], bands))
    for b in range(bands):
        inside = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not inside.any():
            inside = np.zeros_like(inside)
            inside[int(np.argmin(np.abs(freqs - np.sqrt(edges[b] * edges[b + 1]))))] = True
        out[:, b] = power[:, inside].mean(axis=1)
    return out


def quantised_levels(samples: np.ndarray, sampling_hz: float) -> np.ndarray:
    """The recording's spectrogram as ``(frames, bands)`` integer levels.

    Args:
        samples: The waveform.
        sampling_hz: Its sampling rate.

    Returns:
        ``uint8`` levels in ``[0, levels)``, time-major; all zero for a silent or empty waveform.
    """
    spec = settings()["spectrogram"]
    frames, levels = int(spec["frames"]), int(spec["levels"])
    power = band_powers(samples, sampling_hz)
    groups = np.array_split(np.arange(power.shape[0]), frames) if power.shape[0] >= frames else None
    if groups is None:
        index = np.minimum((np.arange(frames) * power.shape[0]) // frames, power.shape[0] - 1)
        binned = power[index]
    else:
        binned = np.stack([power[g].mean(axis=0) for g in groups])
    db = 10.0 * np.log10(binned + 1e-12)
    low = float(np.percentile(db, float(spec["floor_percentile"])))
    high = float(db.max())
    if high - low < 1e-6:
        return np.zeros((frames, binned.shape[1]), dtype=np.uint8)
    scaled = np.floor((db - low) / (high - low) * levels)
    return np.clip(scaled, 0, levels - 1).astype(np.uint8)


def pack(levels: np.ndarray) -> str:
    """Pack two-bit levels four to a byte, time-major, and base64-encode them.

    Args:
        levels: ``(frames, bands)`` levels in ``[0, 4)``.

    Returns:
        The base64 text.
    """
    flat = np.asarray(levels, dtype=np.uint8).reshape(-1)
    padded = np.concatenate([flat, np.zeros((-flat.size) % 4, dtype=np.uint8)]).reshape(-1, 4)
    packed = padded[:, 0] | (padded[:, 1] << 2) | (padded[:, 2] << 4) | (padded[:, 3] << 6)
    return base64.b64encode(packed.astype(np.uint8).tobytes()).decode("ascii")


def unpack(text: str, frames: int, bands: int) -> np.ndarray:
    """The inverse of :func:`pack`.

    Args:
        text: The base64 text.
        frames: Time bins.
        bands: Frequency bands.

    Returns:
        ``(frames, bands)`` levels.
    """
    raw = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
    cells = np.stack([(raw >> shift) & 3 for shift in (0, 2, 4, 6)], axis=1).reshape(-1)
    return cells[: frames * bands].reshape(frames, bands)
