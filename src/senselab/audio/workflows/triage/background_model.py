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


@dataclass(frozen=True)
class Impulse:
    """One short broadband transient: a click, knock or handling noise.

    Attributes:
        peak_s: Where its envelope peaks.
        start_s: Where it rises out of its background.
        end_s: Where it falls back into it.
        peak_db: Its envelope peak over its background.
        attack_ms: Rise time from start to peak.
    """

    peak_s: float
    start_s: float
    end_s: float
    peak_db: float
    attack_ms: float

    def record(self) -> dict[str, float]:
        """The impulse, for the store."""
        return {
            "peak_s": round(self.peak_s, 4),
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "peak_db": round(self.peak_db, 2),
            "attack_ms": round(self.attack_ms, 2),
        }


def impulses_of(signal: Signal, p: dict[str, Any]) -> list[Impulse]:
    """The impulses of a signal: fast to rise and soon over, on a pre-emphasised sample envelope.

    Args:
        signal: The samples and their rate.
        p: The parameters.

    Returns:
        The impulses, in time order.
    """
    from scipy.ndimage import median_filter  # noqa: PLC0415

    q = p["impulse"]
    samples, rate = signal
    x = np.asarray(samples, dtype=np.float64)
    if x.size < 2:
        return []
    y = x[1:] - 0.97 * x[:-1]
    step = max(1, int(rate / 1000))
    width = max(step, int(q["envelope_ms"] * rate / 1000))
    count = (len(y) - width) // step + 1
    if count < 3:
        return []
    squares = np.cumsum(np.concatenate([[0.0], y * y]))
    starts = np.arange(count) * step
    env = 10.0 * np.log10((squares[starts + width] - squares[starts]) / width + 1e-20)
    background = median_filter(env, size=max(3, int(q["background_ms"])), mode="nearest")
    excess = env - background
    ms = step * 1000.0 / rate
    found: list[Impulse] = []
    taken = np.zeros(count, dtype=bool)
    for i in np.argsort(-excess):
        if excess[i] < q["peak_db"]:
            break
        if taken[i]:
            continue
        a = i
        while a > 0 and excess[a - 1] >= q["onset_db"]:
            a -= 1
        b = i
        while b < count - 1 and excess[b + 1] >= q["onset_db"]:
            b += 1
        taken[a : b + 1] = True
        if (i - a) * ms <= q["attack_max_ms"] and (b - a + 1) * ms <= q["duration_max_ms"]:
            half = width / 2.0
            found.append(
                Impulse(
                    (starts[i] + half) / rate,
                    (starts[a] + half) / rate,
                    (starts[b] + half) / rate,
                    float(excess[i]),
                    (i - a) * ms,
                )
            )
    return sorted(found, key=lambda imp: imp.peak_s)


@dataclass(frozen=True)
class Region:
    """One region of activity over the floor; what it is, the branches decide.

    Attributes:
        start_s: Its start.
        end_s: Its end.
        peak_db: Its highest broadband level over the broadband floor.
        bands_fraction: The largest share of bands over their floor in any of its frames.
        onset_s: From its start to where it first comes within ``peak_window_db`` of its peak.
        offset_s: From where it last does to its end.
    """

    start_s: float
    end_s: float
    peak_db: float
    bands_fraction: float
    onset_s: float
    offset_s: float

    def record(self) -> dict[str, float]:
        """The region, for the store."""
        return {key: round(float(value), 3) for key, value in self.__dict__.items()}


def regions_of(frames: BandFrames, floor: np.ndarray, impulses: Sequence[Impulse], p: dict[str, Any]) -> list[Region]:
    """The regions of activity: bridged runs of active frames, less those an impulse explains.

    Args:
        frames: The plain stream's band frames.
        floor: The floor per band.
        impulses: The impulses, whose own frames do not make a region.
        p: The parameters.

    Returns:
        The regions, in time order.
    """
    q = p["activity"]
    hop = p["hop_s"]
    mask = active_mask(frames, floor, p)
    for imp in impulses:
        mask[(frames.times_s >= imp.start_s - hop) & (frames.times_s <= imp.end_s + hop)] = False
    above = (frames.band_db >= floor[None, :] + q["margin_db"]).mean(axis=1)
    broadband = 10.0 * np.log10(np.sum(10.0 ** (floor / 10.0)) + 1e-12)
    out: list[Region] = []
    for first, end in bridge(runs_of(mask), int(round(q["bridge_s"] / hop)) + 1):
        if (end - first) * hop < q["min_s"]:
            continue
        level = frames.level_db[first:end]
        near = np.flatnonzero(level >= level.max() - q["peak_window_db"])
        out.append(
            Region(
                float(frames.times_s[first] - hop / 2),
                float(frames.times_s[end - 1] + hop / 2),
                float(level.max() - broadband),
                float(above[first:end].max()),
                float(near[0] * hop),
                float((end - first - 1 - near[-1]) * hop),
            )
        )
    return out


@dataclass(frozen=True)
class Hum:
    """Mains lines in the residual.

    Attributes:
        lines: Mains fundamental to how many of its multiples stand over their neighbourhood.
        mains_hz: The fundamentals with enough lines, empty where none.
    """

    lines: dict[str, int]
    mains_hz: tuple[float, ...]

    def record(self) -> dict[str, Any]:
        """The reading, for the store."""
        return {"fired": bool(self.mains_hz), "lines": dict(self.lines), "mains_hz": list(self.mains_hz)}


def hum_of(residual: Signal | None, p: dict[str, Any]) -> Hum:
    """The mains lines the residual carries.

    Args:
        residual: The residual stream, or None where it is absent.
        p: The parameters.

    Returns:
        The hum reading.
    """
    q = p["hum"]
    lines: dict[str, int] = {}
    if residual is not None and len(residual[0]) >= residual[1]:
        from scipy.signal import welch  # noqa: PLC0415

        samples, rate = residual
        freqs, psd = welch(np.asarray(samples, dtype=np.float64), fs=rate, nperseg=int(rate))
        db = 10.0 * np.log10(psd + 1e-20)
        lo, hi = q["neighbourhood_hz"]
        for m in q["mains_hz"]:
            count = 0
            for k in range(1, int(q["harmonics_max"]) + 1):
                at = np.abs(freqs - k * m) <= 0.5
                around = (np.abs(freqs - k * m) >= lo) & (np.abs(freqs - k * m) <= hi)
                if at.any() and around.any() and db[at].max() - np.median(db[around]) >= q["line_db"]:
                    count += 1
            lines[f"{m:g}"] = count
    return Hum(lines, tuple(float(m) for m in q["mains_hz"] if lines.get(f"{m:g}", 0) >= q["lines_min"]))
