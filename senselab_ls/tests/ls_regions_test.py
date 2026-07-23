"""Unit tests for the Label Studio region builders (no senselab import needed)."""

from __future__ import annotations

from types import SimpleNamespace

from senselab_ls.common.ls_regions import (
    DEFAULT_SPEAKER_LABEL,
    diarization_to_ls,
    ls_label_region,
    new_region_id,
)


def test_new_region_id_is_zero_padded() -> None:
    """Region ids are the prefix plus a 4-digit zero-padded index."""
    assert new_region_id("diarization", 3) == "diarization_0003"


def test_ls_label_region_shape() -> None:
    """A labels region carries the expected keys and float-coerced timings."""
    region = ls_label_region(
        region_id="r0",
        from_name="diarization",
        start=1,
        end=2,
        label="SPEAKER_00",
    )
    assert region["type"] == "labels"
    assert region["from_name"] == "diarization"
    assert region["to_name"] == "audio"
    assert region["value"] == {"start": 1.0, "end": 2.0, "labels": ["SPEAKER_00"]}
    assert "score" not in region


def test_ls_label_region_includes_score_when_given() -> None:
    """A score is attached only when provided."""
    region = ls_label_region(
        region_id="r0",
        from_name="diarization",
        start=0.0,
        end=1.0,
        label="SPEAKER_00",
        score=0.75,
    )
    assert region["score"] == 0.75


def test_diarization_to_ls_from_objects() -> None:
    """ScriptLine-like objects convert to one labels region each."""
    segments = [
        SimpleNamespace(start=0.0, end=1.5, speaker="SPEAKER_00"),
        SimpleNamespace(start=1.5, end=3.0, speaker="SPEAKER_01"),
    ]
    regions = diarization_to_ls(segments, "diarization")
    assert [r["value"]["labels"][0] for r in regions] == ["SPEAKER_00", "SPEAKER_01"]
    assert regions[0]["id"] == "diarization_0000"
    assert regions[1]["value"]["start"] == 1.5


def test_diarization_to_ls_from_dicts_and_defaults() -> None:
    """Dict segments work, missing speaker falls back, untimed segments are dropped."""
    segments = [
        {"start": 0.0, "end": 1.0},  # no speaker -> default label
        {"start": None, "end": 2.0, "speaker": "SPEAKER_01"},  # untimed -> skipped
    ]
    regions = diarization_to_ls(segments, "diarization")
    assert len(regions) == 1
    assert regions[0]["value"]["labels"][0] == DEFAULT_SPEAKER_LABEL


def test_diarization_to_ls_unwraps_nested_shape() -> None:
    """The raw List[List[ScriptLine]] shape is unwrapped to its first element."""
    nested = [[SimpleNamespace(start=0.0, end=1.0, speaker="SPEAKER_00")]]
    regions = diarization_to_ls(nested, "diarization")
    assert len(regions) == 1


def test_diarization_to_ls_empty() -> None:
    """Empty or falsy input yields no regions."""
    assert diarization_to_ls([], "diarization") == []
    assert diarization_to_ls(None, "diarization") == []
