"""The speaker gate's reading: the dominant voice against the time a second voice holds the extent alone (r37 C1)."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.triage.speaker_share import speakers_within, subtract_intervals


def test_overlap_only_secondary_time_leaves_the_exclusive_share_whole() -> None:
    """The 8 labelled false fires: every second of the secondary speaker overlaps the dominant one."""
    segments = [("a", "S0", (0.0, 10.0)), ("b", "S1", (2.0, 4.0)), ("c", "S1", (6.0, 7.0))]
    reading = speakers_within((0.0, 10.0), segments, exclusive_min_s=1.0)
    assert reading["share"] == pytest.approx(10.0 / 13.0)
    assert reading["secondary_exclusive_s"] == 0.0 and reading["secondary_exclusive_spans"] == []
    assert reading["exclusive_share"] == 1.0


def test_a_second_voice_alone_counts_against_the_dominant_one() -> None:
    """Two seconds the second voice holds alone in a ten-second extent: 8 / (8 + 2)."""
    segments = [("a", "S0", (0.0, 4.0)), ("b", "S0", (6.0, 10.0)), ("c", "S1", (3.0, 6.5))]
    reading = speakers_within((0.0, 10.0), segments, exclusive_min_s=1.0)
    assert reading["secondary_exclusive_s"] == pytest.approx(2.0)
    assert reading["secondary_exclusive_spans"] == [[4.0, 6.0]]
    assert reading["exclusive_share"] == pytest.approx(0.8)


def test_exclusive_time_under_the_floor_holds_no_turn() -> None:
    """Half a second alone: recorded, and counted as none."""
    segments = [("a", "S0", (0.0, 5.0)), ("b", "S1", (5.0, 5.5))]
    reading = speakers_within((0.0, 6.0), segments, exclusive_min_s=1.0)
    assert reading["secondary_exclusive_s"] == pytest.approx(0.5)
    assert reading["exclusive_share"] == 1.0 and reading["exclusive_min_s"] == 1.0


def test_only_the_extent_is_read() -> None:
    """A second voice before the extent is not inside it."""
    segments = [("a", "S1", (0.0, 3.0)), ("b", "S0", (3.0, 9.0))]
    reading = speakers_within((3.0, 9.0), segments, exclusive_min_s=1.0)
    assert reading["speaker_labels"] == ["S0"] and reading["exclusive_share"] == 1.0
    assert reading["sources"] == ["b"]


def test_no_segment_reads_none() -> None:
    """Nothing attributed: no share at all, which the gate reads as undetermined."""
    reading = speakers_within((0.0, 5.0), [], exclusive_min_s=1.0)
    assert reading["share"] is None and reading["exclusive_share"] is None


def test_subtracting_intervals() -> None:
    """The arithmetic the exclusive pieces rest on."""
    assert subtract_intervals([(0.0, 10.0)], [(2.0, 3.0), (5.0, 12.0)]) == [(0.0, 2.0), (3.0, 5.0)]
