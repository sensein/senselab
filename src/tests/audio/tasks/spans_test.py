"""Span proposal: the gate, the peak-anchored onset, the range-relative offset."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from senselab.audio.tasks.spans import NoContrast, Span, propose_spans

SR = 16000


def _envelope(events: list[tuple[float, float, float]], seconds: float = 14.0, floor: float = -55.0) -> np.ndarray:
    env = np.full(int(seconds * SR), floor)
    for start, end, peak in events:
        env[int(start * SR) : int(end * SR)] = peak
    return env


def _propose(env: np.ndarray, floor: np.ndarray) -> list[Span] | NoContrast:
    """Run propose_spans with the config's measured values; the fixture supplies the literals."""
    return propose_spans(
        env,
        floor,
        SR,
        k_db=18.0,
        onset_drop_db=15.0,
        offset_fraction=0.7,
        hangover_ms=120,
        min_duration_ms=50,
        min_separation_ms=150,
    )


class TestGate:
    """Only a peak that clears k_db over the local floor proposes a span."""

    def test_a_peak_below_k_is_not_proposed(self) -> None:
        """Of one 10 dB peak and one 35 dB peak, only the one clearing 18 dB becomes a span."""
        env = _envelope([(2.0, 2.5, -45.0), (8.0, 8.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert not isinstance(out, NoContrast)
        assert len(out) == 1, "the 10 dB event must be gated out, not merely narrowed"
        (span,) = out
        assert span.start == pytest.approx(8.0, abs=0.05)
        assert span.end == pytest.approx(8.5, abs=0.05)

    def test_a_peak_above_k_is_proposed(self) -> None:
        """A 35 dB event over the floor clears the 18 dB gate."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        assert len(out) == 1 and isinstance(out[0], Span)

    def test_two_events_far_apart_stay_two_spans(self) -> None:
        """Distant events do not merge."""
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        assert len(out) == 2


class TestNoContrast:
    """An unmeasurable recording is a distinct value, not a quiet one."""

    def test_no_peak_anywhere_is_no_contrast_not_an_empty_list(self) -> None:
        """A flat envelope 1 dB over its floor yields NoContrast naming the gate."""
        env = np.full(int(3.0 * SR), -30.0)
        out = _propose(env, np.full_like(env, -29.0))
        assert isinstance(out, NoContrast)
        assert "18" in out.reason


class TestKIsRequired:
    """No rule parameter has a default."""

    def test_k_has_no_default(self) -> None:
        """Calling without the rule parameters is a TypeError, not a silent default."""
        env = _envelope([(2.0, 2.5, -20.0)])
        with pytest.raises(TypeError):
            propose_spans(env, np.full_like(env, -55.0), SR)  # type: ignore[call-arg]


class TestSpanCarriesItsContrast:
    """A span reports how far its peak stood above the floor."""

    def test_peak_over_floor_travels_with_the_span(self) -> None:
        """A -20 dB event over a -55 dB floor reports 35 dB of contrast."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        (span,) = out
        assert span.peak_over_floor_db == pytest.approx(35.0, abs=0.5)


class TestTheMergeRate:
    """A span records how many proposals it absorbed, so several events in one span are legible."""

    def test_an_unmerged_span_absorbed_one_proposal(self) -> None:
        """One proposal in, one span out: the count is 1, not 0 — a span is its own proposal."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        (span,) = out
        assert span.merged_proposals == 1

    def test_two_overlapping_proposals_report_two(self) -> None:
        """Two peaks over one shoulder propose two spans covering it; the survivor says so."""
        env = _envelope([(2.0, 2.2, -20.0), (2.2, 2.4, -25.0), (2.4, 2.6, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0].merged_proposals == 2

    def test_two_separated_proposals_each_report_one(self) -> None:
        """Nothing was absorbed, and neither span claims otherwise."""
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        assert [span.merged_proposals for span in out] == [1, 1]


class TestOnsetIsAlsoFloorRelative:
    """Onset walks back past a dip that is still far above the floor, not just close to the peak.

    Peak-anchored alone (walk back only while within onset_drop_db of the peak) is the fitted,
    benchmarked rule and stays the primary criterion; this only covers the case the benchmark did
    not exercise: an event whose envelope dips internally more than onset_drop_db below its own
    peak while remaining well above the local floor (a gain-curve settling artifact on a short
    burst, or ordinary two-syllable amplitude modulation, produce the same shape).
    """

    def test_an_onset_ramp_more_than_onset_drop_db_below_the_peak_is_included(self) -> None:
        """A single peak, so offset/hangover/merge cannot be doing this — isolates the onset walk.

        The onset ramp sits 16 dB below the peak (> onset_drop_db=15, so peak-anchored alone stops
        at its far edge, t=2.05) but 19 dB above the floor (>= k_db=18, so the floor-relative clause
        walks through it to the true onset at t=2.0).
        """
        env = _envelope([(2.05, 2.5, -20.0)])
        env[int(2.0 * SR) : int(2.05 * SR)] = -36.0
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        (span,) = out
        assert span.merged_proposals == 1, "one peak, one proposal -- offset/merge are not involved"
        assert span.start == pytest.approx(2.0, abs=0.005), "peak-anchored alone would stop at 2.05"

    def test_a_gap_at_the_floor_still_stays_two_spans(self) -> None:
        """The floor-relative clause extends the walk; it does not make it walk through real silence.

        The gap is wider than ``hangover_ms`` (150 ms vs. 120 ms), so the offset side's own hangover
        mechanism cannot be what keeps these separate — this isolates the onset change specifically.
        """
        env = _envelope([(2.0, 2.5, -20.0)])
        env[int(2.2 * SR) : int(2.35 * SR)] = -55.0  # genuinely back at the floor, not just a dip
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        assert len(out) == 2, "a gap that reaches the floor is two events, not one bridged span"


class TestAnUnmeasurableEnvelope:
    """The envelope carries NaN where the filtered signal had no dB value; a span extent may not."""

    def test_scattered_unmeasurable_samples_do_not_stretch_the_span_to_the_end(self) -> None:
        """The hangover asks whether the envelope stayed low; a NaN is not evidence that it rose."""
        env = _envelope([(2.0, 2.5, -20.0)])
        env[int(2.5 * SR) :: SR // 20] = np.nan
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        (span,) = out
        assert span.end == pytest.approx(2.5, abs=0.15), "the offset must close on the measured samples"

    def test_no_span_extent_or_contrast_is_nan(self) -> None:
        """A NaN reaching an extent becomes a bar of unknown width and a dB label reading nan."""
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        env[int(1.0 * SR) : int(1.2 * SR)] = np.nan
        env[int(2.5 * SR) : int(2.7 * SR)] = np.nan
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        for span in out:
            assert np.isfinite([span.start, span.end, span.peak_over_floor_db]).all()

    def test_an_envelope_with_no_measurable_sample_is_no_contrast(self) -> None:
        """Nothing was measured, so nothing rose above anything; the reason may not read "nan"."""
        env = np.full(int(3.0 * SR), np.nan)
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, NoContrast)
        assert "nan" not in out.reason.lower()

    def test_an_unmeasurable_envelope_raises_no_warning(self) -> None:
        """NoContrast is the answer, not a RuntimeWarning from an empty reduction."""
        env = np.full(int(3.0 * SR), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert isinstance(_propose(env, np.full_like(env, -55.0)), NoContrast)
