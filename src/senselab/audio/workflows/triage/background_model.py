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
