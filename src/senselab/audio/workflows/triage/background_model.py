"""The recording-level background model and acquisition faults, read off every recording.

A stationary floor per band, estimated outside the active regions and checked against the residual;
mains hum on the residual; impulses on the plain stream's sample envelope; regions of activity over
the floor; and the faults: shutoff, dropouts, discontinuities. No reading here knows the declared
task. Every parameter is in ``data/background_model.yaml``; the design is
``specs/20261007-task-events-in-background/design.md``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

BACKGROUND_MODEL_PATH = Path(__file__).parent / "data" / "background_model.yaml"

Signal = tuple[np.ndarray, int]
Run = tuple[int, int]
Span = tuple[float, float]


@functools.cache
def background_model_parameters() -> dict[str, Any]:
    """The parameters of ``data/background_model.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(BACKGROUND_MODEL_PATH.read_text()) or {})


def runs_of(mask: np.ndarray) -> list[Run]:
    """The runs of True in a mask, as ``(first, end)`` index pairs, end exclusive."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def bridge(runs: Sequence[Run], gap_frames: int) -> list[Run]:
    """Runs joined across every gap shorter than ``gap_frames``."""
    out: list[Run] = []
    for first, end in runs:
        if out and first - out[-1][1] < gap_frames:
            out[-1] = (out[-1][0], end)
        else:
            out.append((first, end))
    return out


def floor_db(level: np.ndarray, valid: np.ndarray, p: dict[str, Any]) -> float:
    """The quietest stretch of a level series outside the excluded frames.

    Args:
        level: The frame level, dB.
        valid: Which frames the floor may be read from.
        p: Parameters carrying ``floor_smooth_s``, ``hop_s`` and ``digital_floor_dbfs``.

    Returns:
        The floor, in dB.
    """
    width = max(1, int(round(p["floor_smooth_s"] / p["hop_s"])))
    values = np.where(valid & (level >= p["digital_floor_dbfs"]), level, np.nan)
    if np.isfinite(values).sum() < width:
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if finite.size else float(np.min(level))
    kernel = np.ones(width)
    sums = np.convolve(np.nan_to_num(values), kernel, mode="valid")
    counts = np.convolve(np.isfinite(values).astype(float), kernel, mode="valid")
    means = np.where(counts >= width, sums / np.maximum(counts, 1), np.nan)
    return float(np.nanmin(means)) if np.isfinite(means).any() else float(np.nanmin(values))


def shutoff_runs(level: np.ndarray, p: dict[str, Any]) -> list[Run]:
    """Runs where every band dropped abruptly to a flat level well under the rest of the recording.

    Args:
        level: The broadband frame level, dB.
        p: Parameters carrying ``hop_s``, ``digital_floor_dbfs``, ``floor_smooth_s``,
            ``phonation_db`` and the ``shutoff`` section.

    Returns:
        The shutoff runs, in frame indices.
    """
    q = p["shutoff"]
    hop = p["hop_s"]
    digital = level < p["digital_floor_dbfs"]
    width = max(2, int(round(q["flat_window_s"] / hop)))
    flat = np.zeros(len(level), dtype=bool)
    if len(level) >= width:
        windows = np.lib.stride_tricks.sliding_window_view(level, width)
        spread = np.percentile(windows, 90, axis=1) - np.percentile(windows, 10, axis=1)
        for k in np.flatnonzero(spread <= q["flat_db"]):
            flat[k : k + width] = True
    live = floor_db(level, ~(flat | digital), p)
    dead = digital | (flat & (level <= live - q["below_floor_db"]))
    active = np.flatnonzero(level >= live + p["phonation_db"])
    if active.size == 0:
        return []
    drop_n = max(1, int(round(q["drop_s"] / hop)))
    found: list[Run] = []
    for first, end in bridge(runs_of(dead), int(round(q["bridge_s"] / hop)) + 1):
        if (end - first) * hop < q["min_s"] or first <= active[0]:
            continue
        before = level[max(0, first - drop_n) : first]
        if before.size and before.max() - np.median(level[first:end]) >= q["drop_db"]:
            found.append((first, end))
    return found


@dataclass(frozen=True)
class BandFrames:
    """The per-band level series every reading is taken over.

    Attributes:
        times_s: Each frame's centre.
        band_db: The level per band, dB re full scale, frames by bands.
        level_db: The broadband level over the band edges, dB.
    """

    times_s: np.ndarray
    band_db: np.ndarray
    level_db: np.ndarray


def band_frames(signal: Signal, p: dict[str, Any]) -> BandFrames:
    """The band levels of one stream.

    Args:
        signal: The samples and their rate.
        p: The parameters.

    Returns:
        The frames.
    """
    samples, rate = signal
    frame, hop = int(p["frame_s"] * rate), max(1, int(p["hop_s"] * rate))
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    window = np.hanning(frame)
    windows = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop] * window
    nfft = int(2 ** np.ceil(np.log2(frame)))
    power = np.abs(np.fft.rfft(windows, n=nfft, axis=1)) ** 2 / max(float(np.sum(window**2)), 1e-12)
    freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
    edges = [e for e in p["band_edges_hz"] if e <= rate / 2.0]
    bands = [(freqs >= lo) & (freqs < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    band_power = np.stack([power[:, b].sum(axis=1) for b in bands], axis=1) / frame
    whole = (freqs >= edges[0]) & (freqs < edges[-1])
    times = (np.arange(len(windows)) * hop + frame / 2) / rate
    return BandFrames(
        times,
        10.0 * np.log10(band_power + 1e-12),
        10.0 * np.log10(power[:, whole].sum(axis=1) / frame + 1e-12),
    )


def active_mask(frames: BandFrames, floor: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Frames where enough bands stand the margin over their floor."""
    q = p["activity"]
    above = frames.band_db >= floor[None, :] + q["margin_db"]
    return np.asarray(above.mean(axis=1) >= q["band_fraction"])


@dataclass(frozen=True)
class Floor:
    """The stationary floor per band.

    Attributes:
        band_db: The floor per band, dB.
        source: ``quiet_frames`` (the recording's own), ``residual`` (the task fills the file), or
            ``digital`` (nothing over digital silence).
        quiet_s: Seconds of quiet frames the own floor was read over.
        own_db: The recording's own quiet-frame floor, recorded alongside whatever was chosen.
        residual_db: The residual's floor, None where the residual is absent.
    """

    band_db: np.ndarray
    source: str
    quiet_s: float
    own_db: np.ndarray
    residual_db: np.ndarray | None


def floor_of(frames: BandFrames, residual: BandFrames | None, p: dict[str, Any]) -> Floor:
    """The floor, read iteratively outside the active regions and checked against the residual.

    Args:
        frames: The plain stream's band frames.
        residual: The residual stream's, or None.
        p: The parameters.

    Returns:
        The floor.
    """
    q = p["floor"]
    hop = p["hop_s"]
    live = frames.level_db >= p["digital_floor_dbfs"]
    bands = frames.band_db.shape[1]
    if not live.any():
        silent = np.full(bands, float(p["digital_floor_dbfs"]))
        return Floor(silent, "digital", 0.0, silent, None)
    own = np.percentile(frames.band_db[live], q["initial_percentile"], axis=0)
    pad = int(round(q["pad_s"] / hop))
    quiet = live
    for _ in range(int(q["iterations"])):
        active = active_mask(frames, own, p)
        if pad:
            active = np.convolve(active.astype(float), np.ones(2 * pad + 1), mode="same") > 0
        candidate = live & ~active
        if candidate.sum() * hop < q["quiet_min_s"]:
            break
        quiet = candidate
        own = np.percentile(frames.band_db[quiet], q["quiet_percentile"], axis=0)
    residual_db = None
    if residual is not None:
        r_live = residual.level_db >= p["digital_floor_dbfs"]
        if r_live.any():
            residual_db = np.percentile(residual.band_db[r_live], q["residual_percentile"], axis=0)
    quiet_s = float(quiet.sum() * hop)
    fills = residual_db is not None and np.mean(own - residual_db >= q["residual_gap_db"]) >= 0.5
    if residual_db is not None and (fills or quiet_s < q["quiet_min_s"]):
        return Floor(residual_db, "residual", quiet_s, own, residual_db)
    return Floor(own, "quiet_frames", quiet_s, own, residual_db)
