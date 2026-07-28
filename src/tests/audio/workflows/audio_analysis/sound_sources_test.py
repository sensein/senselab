"""Tests for the sound-source categorizer (feature 20260722-175022, US2).

Covers SC-003 (complete, non-overlapping category coverage of the classifier
vocabularies), mass normalization + dominant selection, a background-machine
scenario, and the null-safe path when no classifier ran (FR-023).
"""

from __future__ import annotations

import json
from importlib import resources

from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.sound_sources import (
    SOURCE_CATEGORIES,
    harvest_source_categories,
    load_source_category_map,
)


def _classification_block(windows: list[dict]) -> dict:
    """Wrap per-window dicts as an AST/YAMNet pass_summary block."""
    return {"status": "ok", "result": [windows], "cache_key": "cls_k"}


def _window(start: float, end: float, labels: list[str], scores: list[float]) -> dict:
    """Build one classification window dict."""
    return {
        "start": start,
        "end": end,
        "labels": labels,
        "scores": scores,
        "win_length": end - start,
        "hop_length": 0.5,
    }


def test_category_map_covers_all_classifier_classes() -> None:
    """SC-003: every AST (527) and YAMNet (521) class maps to exactly one of the 4 categories."""
    doc = load_source_category_map()
    mapping = doc["map"]

    # Every value is a valid category (non-overlapping: dict → exactly one each).
    assert set(mapping.values()) <= set(SOURCE_CATEGORIES)
    for name, cat in mapping.items():
        assert cat in SOURCE_CATEGORIES, f"{name!r} → invalid category {cat!r}"

    # AST coverage: the map was authored over the full 527-class AST vocabulary.
    assert len(mapping) == 527

    # YAMNet coverage: the vendored 521-class list must all be present as keys.
    yam_res = resources.files("senselab.audio.workflows.audio_analysis").joinpath("data", "yamnet_class_names.json")
    yam = json.loads(yam_res.read_text(encoding="utf-8"))["names"]
    assert len(yam) == 521
    missing = [n for n in yam if n not in mapping]
    assert not missing, f"YAMNet classes missing from map: {missing[:10]}"


def test_masses_sum_to_one_and_dominant_is_argmax() -> None:
    """Per-bucket masses normalize to ~1 and src_dominant is their argmax."""
    windows = [_window(0.0, 0.5, ["Speech", "Vehicle", "Wind"], [0.7, 0.2, 0.1])]
    rows = harvest_source_categories(
        pass_summary={"duration_s": 0.5, "ast": _classification_block(windows)},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert rows
    r = rows[0]
    total = r["src_speech"] + r["src_people"] + r["src_machine"] + r["src_environment"]
    assert abs(total - 1.0) < 1e-6
    assert r["src_dominant"] == "speech"
    assert r["src_speech"] > r["src_machine"] > r["src_environment"]


def test_background_machine_window_dominant_machine() -> None:
    """A window dominated by vehicle/engine classes → src_machine dominant."""
    windows = [_window(0.0, 0.5, ["Engine", "Vehicle", "Speech"], [0.6, 0.3, 0.1])]
    rows = harvest_source_categories(
        pass_summary={"duration_s": 0.5, "yamnet": _classification_block(windows)},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert rows[0]["src_dominant"] == "machine"


def test_ast_and_yamnet_averaged() -> None:
    """When both classifiers run, masses are the mean of the two distributions."""
    ast_win = [_window(0.0, 0.5, ["Speech"], [1.0])]
    yam_win = [_window(0.0, 0.5, ["Vehicle"], [1.0])]
    rows = harvest_source_categories(
        pass_summary={
            "duration_s": 0.5,
            "ast": _classification_block(ast_win),
            "yamnet": _classification_block(yam_win),
        },
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    r = rows[0]
    assert abs(r["src_speech"] - 0.5) < 1e-6
    assert abs(r["src_machine"] - 0.5) < 1e-6


def test_null_when_no_classifier() -> None:
    """FR-023 / T017: no AST/YAMNet → all src_* columns null."""
    rows = harvest_source_categories(
        pass_summary={"duration_s": 1.0},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert rows
    for r in rows:
        assert r["src_speech"] is None
        assert r["src_dominant"] is None
