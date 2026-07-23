"""Background sound-source categorization for the presence axis.

Maps the AudioSet class scores emitted by AST and YAMNet into four coarse
categories — ``speech``, ``people`` (non-speech human sounds), ``machine``
(vehicles / engines / tools / appliances), and ``environment`` (nature /
ambient / animals / music / background) — and reports the per-bucket relative
mass of each plus the dominant category.

The mapping is a checked-in, versioned JSON (``data/audioset_source_map.json``)
authored by walking each AudioSet class to its top-level ontology ancestor, so
every class the classifiers can emit resolves to exactly one category (SC-003).
This is additive to the presence rows and independent of the existing top-1
``speech_presence_labels`` / YAMNet-veto uses, which are unaffected.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Optional

from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import classification_windows
from senselab.utils.data_structures.logging import logger

SOURCE_CATEGORIES = ("speech", "people", "machine", "environment")
_MAP_RESOURCE = "audioset_source_map.json"

# Classes seen at runtime that are missing from the map — warn once each.
_warned_unmapped: set[str] = set()


@lru_cache(maxsize=1)
def load_source_category_map() -> dict[str, Any]:
    """Load and cache the versioned AudioSet→category map from package data."""
    data_pkg = resources.files("senselab.audio.workflows.audio_analysis").joinpath("data", _MAP_RESOURCE)
    doc = json.loads(data_pkg.read_text(encoding="utf-8"))
    if "map" not in doc or "default" not in doc:
        raise ValueError(f"{_MAP_RESOURCE} is missing required 'map'/'default' keys")
    return doc


def _category_for(label: str, mapping: dict[str, str], default: str) -> str:
    """Resolve one AudioSet display name to a category, warning once if unmapped."""
    cat = mapping.get(label)
    if cat is None:
        if label not in _warned_unmapped:
            _warned_unmapped.add(label)
            logger.warning(f"sound_sources: AudioSet class {label!r} not in category map; using default {default!r}")
        return default
    return cat


def _native_grid(block: dict[str, Any]) -> tuple[float, float]:
    """Recover the (win_length, hop_length) a classifier ran with from its first window."""
    windows = classification_windows(block.get("result"))
    if not windows or not isinstance(windows[0], dict):
        return 1.0, 1.0
    w = windows[0]
    win_length = float(w.get("win_length", 0) or 0) or float(w.get("end", 0) - w.get("start", 0))
    hop_length = float(w.get("hop_length", 0) or 0) or win_length
    if win_length <= 0:
        win_length = 1.0
    if hop_length <= 0:
        hop_length = win_length
    return win_length, hop_length


def _window_category_masses(window: Any, mapping: dict[str, str], default: str) -> Optional[dict[str, float]]:  # noqa: ANN401
    """Sum one classification window's scores into the four category masses (normalized)."""
    if not isinstance(window, dict):
        return None
    labels = window.get("labels") or []
    scores = window.get("scores") or []
    if not labels or not scores:
        return None
    masses = {c: 0.0 for c in SOURCE_CATEGORIES}
    total = 0.0
    for label, score in zip(labels, scores):
        s = max(float(score), 0.0)
        masses[_category_for(str(label), mapping, default)] += s
        total += s
    if total <= 0:
        return None
    return {c: masses[c] / total for c in SOURCE_CATEGORIES}


def _classifier_masses_by_bucket(
    block: dict[str, Any],
    grid: BucketGrid,
    duration_s: float,
    mapping: dict[str, str],
    default: str,
) -> dict[tuple[float, float], dict[str, float]]:
    """Per-bucket category masses for one classifier block (AST or YAMNet)."""
    windows = classification_windows(block.get("result"))
    if not windows:
        return {}
    _win_len, hop = _native_grid(block)
    out: dict[tuple[float, float], dict[str, float]] = {}
    for b_start, b_end, _idx in grid.iter_buckets(duration_s):
        center = 0.5 * (b_start + b_end)
        win_idx = max(0, int(round(center / hop))) if hop > 0 else 0
        if win_idx >= len(windows):
            win_idx = len(windows) - 1
        masses = _window_category_masses(windows[win_idx], mapping, default)
        if masses is not None:
            out[(round(b_start, 6), round(b_end, 6))] = masses
    return out


def harvest_source_categories(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
) -> list[dict[str, Any]]:
    """Return one source-category dict per presence bucket on ``grid``.

    Combines AST + YAMNet (mean of whichever classifiers are available) into the
    four category masses, normalizes them to sum ~1, and reports the dominant
    category. Each dict carries ``start``, ``end``, ``src_speech``, ``src_people``,
    ``src_machine``, ``src_environment``, ``src_dominant`` (all ``None`` when both
    classifiers are absent — FR-023), and a ``_raw`` block for ``model_votes``.
    """
    duration_s = float(pass_summary.get("duration_s", 0.0) or 0.0)
    if duration_s <= 0:
        return []

    doc = load_source_category_map()
    mapping: dict[str, str] = doc["map"]
    default: str = doc["default"]

    per_classifier: dict[str, dict[tuple[float, float], dict[str, float]]] = {}
    for key in ("ast", "yamnet"):
        block = pass_summary.get(key)
        if isinstance(block, dict) and block.get("status") == "ok":
            per_classifier[key] = _classifier_masses_by_bucket(block, grid, duration_s, mapping, default)

    out: list[dict[str, Any]] = []
    for b_start, b_end, _idx in grid.iter_buckets(duration_s):
        bkey = (round(b_start, 6), round(b_end, 6))
        available = [m[bkey] for m in per_classifier.values() if bkey in m]
        if not available:
            out.append(
                {
                    "start": b_start,
                    "end": b_end,
                    "src_speech": None,
                    "src_people": None,
                    "src_machine": None,
                    "src_environment": None,
                    "src_dominant": None,
                    "_raw": {},
                }
            )
            continue
        combined = {c: sum(a[c] for a in available) / len(available) for c in SOURCE_CATEGORIES}
        total = sum(combined.values()) or 1.0
        combined = {c: combined[c] / total for c in SOURCE_CATEGORIES}
        dominant = max(SOURCE_CATEGORIES, key=lambda c: combined[c])
        out.append(
            {
                "start": b_start,
                "end": b_end,
                "src_speech": combined["speech"],
                "src_people": combined["people"],
                "src_machine": combined["machine"],
                "src_environment": combined["environment"],
                "src_dominant": dominant,
                "_raw": {"classifiers": sorted(per_classifier.keys())},
            }
        )
    return out
