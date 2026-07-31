"""Every L1 signal carries its units, resolution, and the reduction that produced it.

Without units L2 cannot know whether 0.7 is a probability, a dB value or a within-file rank —
which is exactly how a rank came to be aggregated as though it were a probability. Without the
reduction recorded, a saturating reduction is invisible in the output and findable only by
rendering a figure and looking at it, which is how six such defects were actually found.
"""

from __future__ import annotations

import json

import pytest

from senselab.audio.workflows.audio_analysis.signal import (
    UNITS,
    SignalProvenance,
    measurement,
)


def test_a_measurement_carries_its_units() -> None:
    """The load-bearing field: 0.7 means nothing without it."""
    m = measurement(-23.4, units="LUFS", signal="acoustic_loudness", model="opensmile/eGeMAPSv02")
    assert m["value"] == pytest.approx(-23.4)
    assert m["provenance"]["units"] == "LUFS"


def test_units_must_be_a_recognised_kind() -> None:
    """A free-text unit cannot be checked, so a typo would silently disable the check."""
    with pytest.raises(ValueError, match="units"):
        measurement(0.5, units="loudness-ish", signal="x", model="y")


def test_a_probability_out_of_range_is_refused() -> None:
    """Declaring probability and then reporting 1.4 is the error the declaration exists to catch."""
    with pytest.raises(ValueError, match="probability"):
        measurement(1.4, units="probability", signal="x", model="y")


def test_a_dB_value_is_not_range_checked_as_a_probability() -> None:
    """-70 is a perfectly ordinary loudness and must not trip the probability guard."""
    assert measurement(-70.0, units="LUFS", signal="x", model="y")["value"] == pytest.approx(-70.0)


def test_the_reduction_is_recorded() -> None:
    """A saturating reduction must be visible in the output, not only in a plot.

    ``max`` over speaker channels produced a posterior of exactly 1.0000 in all 1070 buckets of
    a real recording; had the reduction been named in the provenance it would have been
    apparent without rendering anything.
    """
    m = measurement(
        0.9,
        units="probability",
        signal="frame_segmentation",
        model="pyannote/segmentation-3.0",
        reduction="noisy_or_over_speaker_channels",
    )
    assert m["provenance"]["reduction"] == "noisy_or_over_speaker_channels"


def test_resolution_and_window_are_recorded_separately() -> None:
    """Hop and window are separate fields because they differ.

    openSMILE HNR steps every 10 ms over a 60 ms window, so a consumer that assumed hop equals
    window would treat overlapping frames as independent observations.
    """
    m = measurement(
        11.2,
        units="dB",
        signal="acoustic_hnr",
        model="opensmile/eGeMAPSv02",
        resolution_s=0.010,
        window_s=0.060,
    )
    p = m["provenance"]
    assert p["resolution_s"] == pytest.approx(0.010)
    assert p["window_s"] == pytest.approx(0.060)


def test_a_failed_signal_is_representable() -> None:
    """A model that ran and failed must be distinguishable from one never configured."""
    m = measurement(None, units="probability", signal="yamnet", model="google/yamnet", status="failed")
    assert m["value"] is None
    assert m["provenance"]["status"] == "failed"


def test_a_null_value_with_ok_status_is_refused() -> None:
    """Measured-and-the-answer-is-nothing is not a state; it is a missing status."""
    with pytest.raises(ValueError, match="status"):
        measurement(None, units="probability", signal="x", model="y")


def test_the_provenance_serialises() -> None:
    """It travels with the signal into parquet and JSON, so it must round-trip."""
    m = measurement(0.5, units="probability", signal="x", model="y")
    assert json.loads(json.dumps(m)) == m


def test_provenance_is_frozen() -> None:
    """A consumer must not be able to relabel a signal's units after the fact."""
    p = SignalProvenance(signal="x", model="y", units="probability")
    with pytest.raises(Exception):
        p.units = "dB"  # type: ignore[misc]


def test_every_declared_unit_has_a_documented_meaning() -> None:
    """A unit nobody can interpret is no better than no unit at all."""
    for unit in UNITS:
        assert UNITS[unit], f"{unit} has no description"
