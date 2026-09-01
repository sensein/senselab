"""Span proposal: the gate, and the symmetric, floor-relative onset/offset walk."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from senselab.audio.tasks.spans import NoContrast, Span, group_extents_into_runs, propose_spans

SR = 16000


def _envelope(events: list[tuple[float, float, float]], seconds: float = 14.0, floor: float = -55.0) -> np.ndarray:
    env = np.full(int(seconds * SR), floor)
    for start, end, peak in events:
        env[int(start * SR) : int(end * SR)] = peak
    return env


def _propose(env: np.ndarray, floor: float) -> list[Span] | NoContrast:
    """Run propose_spans with the config's measured values; the fixture supplies the literals."""
    return propose_spans(
        env,
        floor,
        SR,
        k_db=18.0,
        floor_margin_db=12.0,
        transition_window_ms=300,
        min_duration_ms=50,
        min_separation_ms=150,
    )


class TestGate:
    """Only a peak that clears k_db over the local floor proposes a span."""

    def test_a_peak_below_k_is_not_proposed(self) -> None:
        """Of one 10 dB peak and one 35 dB peak, only the one clearing 18 dB becomes a span."""
        env = _envelope([(2.0, 2.5, -45.0), (8.0, 8.5, -20.0)])
        out = _propose(env, -55.0)
        assert not isinstance(out, NoContrast)
        assert len(out) == 1, "the 10 dB event must be gated out, not merely narrowed"
        (span,) = out
        assert span.start == pytest.approx(8.0, abs=0.05)
        assert span.end == pytest.approx(8.5, abs=0.05)

    def test_a_peak_above_k_is_proposed(self) -> None:
        """A 35 dB event over the floor clears the 18 dB gate."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        assert len(out) == 1 and isinstance(out[0], Span)

    def test_two_events_far_apart_stay_two_spans(self) -> None:
        """Distant events do not merge."""
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        assert len(out) == 2


class TestNoContrast:
    """An unmeasurable recording is a distinct value, not a quiet one."""

    def test_no_peak_anywhere_is_no_contrast_not_an_empty_list(self) -> None:
        """A flat envelope 1 dB over its floor yields NoContrast naming the gate."""
        env = np.full(int(3.0 * SR), -30.0)
        out = _propose(env, -29.0)
        assert isinstance(out, NoContrast)
        assert "18" in out.reason


class TestKIsRequired:
    """No rule parameter has a default."""

    def test_k_has_no_default(self) -> None:
        """Calling without the rule parameters is a TypeError, not a silent default."""
        env = _envelope([(2.0, 2.5, -20.0)])
        with pytest.raises(TypeError):
            propose_spans(env, -55.0, SR)  # type: ignore[call-arg]


class TestSpanCarriesItsContrast:
    """A span reports how far its peak stood above the floor."""

    def test_peak_over_floor_travels_with_the_span(self) -> None:
        """A -20 dB event over a -55 dB floor reports 35 dB of contrast."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        (span,) = out
        assert span.peak_over_floor_db == pytest.approx(35.0, abs=0.5)


class TestTheMergeRate:
    """A span records how many proposals it absorbed, so several events in one span are legible."""

    def test_an_unmerged_span_absorbed_one_proposal(self) -> None:
        """One proposal in, one span out: the count is 1, not 0 — a span is its own proposal."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        (span,) = out
        assert span.merged_proposals == 1

    def test_two_close_crossings_report_two(self) -> None:
        """Two threshold-crossings nearer than min_separation_ms are one proposal, absorbed as two."""
        env = _envelope([(2.0, 2.2, -20.0), (2.21, 2.4, -20.0)])
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0].merged_proposals == 2

    def test_two_separated_proposals_each_report_one(self) -> None:
        """Nothing was absorbed, and neither span claims otherwise."""
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        assert [span.merged_proposals for span in out] == [1, 1]


class TestWalkStopIsFloorRelativeAndSymmetric:
    """Onset and offset walk by the identical rule -- within floor_margin_db, sustained transition_window_ms.

    Replaces a peak-anchored onset (walk back while within a fixed drop of the peak) paired with a
    floor-fraction offset (walk forward to a threshold fixed once at the peak's own floor value): that
    pair's asymmetry let a peak's stale threshold outlive the walk's own progress across a scene. There
    is no more "distance below the peak" concept at all -- only distance above the floor.
    """

    def test_onset_and_offset_walk_the_same_distance_for_a_symmetric_event(self) -> None:
        """A symmetric envelope around one peak produces a symmetric span -- the rule is identical both ways."""
        env = np.full(int(8.0 * SR), -55.0)
        center = int(4.0 * SR)
        half_plateau = int(0.1 * SR)
        ramp_len = int(0.5 * SR)
        env[center - half_plateau : center + half_plateau] = -20.0
        ramp = np.linspace(-20.0, -55.0, ramp_len)
        env[center - half_plateau - ramp_len : center - half_plateau] = ramp[::-1]
        env[center + half_plateau : center + half_plateau + ramp_len] = ramp
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        (span,) = out
        onset_samples = center - int(span.start * SR)
        offset_samples = int(span.end * SR) - center
        assert onset_samples == pytest.approx(offset_samples, abs=2)

    def test_a_dip_that_stays_above_the_threshold_does_not_end_the_walk(self) -> None:
        """The threshold is floor(-55) + floor_margin_db(12) = -43; a -40 dB dip never crosses it."""
        env = _envelope([(2.0, 2.5, -20.0)])
        env[int(2.2 * SR) : int(2.3 * SR)] = -40.0
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        assert len(out) == 1
        (span,) = out
        assert span.start == pytest.approx(2.0, abs=0.01)
        assert span.end == pytest.approx(2.5, abs=0.01)

    def test_a_brief_dip_below_threshold_does_not_end_the_walk(self) -> None:
        """50 ms below threshold is not a sustained transition; transition_window_ms is 300 ms."""
        env = _envelope([(2.0, 2.5, -20.0)])
        env[int(2.2 * SR) : int(2.25 * SR)] = -50.0
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        assert len(out) == 1
        (span,) = out
        assert span.start == pytest.approx(2.0, abs=0.01)
        assert span.end == pytest.approx(2.5, abs=0.01)

    def test_a_sustained_return_to_the_floor_ends_the_walk_leaving_two_spans(self) -> None:
        """A 350 ms gap at the floor, longer than transition_window_ms, is a genuine transition."""
        env = _envelope([(2.0, 2.5, -20.0), (2.85, 3.35, -20.0)])
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        assert len(out) == 2


class TestAnUnmeasurableEnvelope:
    """The envelope carries NaN where the filtered signal had no dB value; a span extent may not."""

    def test_scattered_unmeasurable_samples_do_not_stretch_the_span_to_the_end(self) -> None:
        """The hangover asks whether the envelope stayed low; a NaN is not evidence that it rose."""
        env = _envelope([(2.0, 2.5, -20.0)])
        env[int(2.5 * SR) :: SR // 20] = np.nan
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        (span,) = out
        assert span.end == pytest.approx(2.5, abs=0.15), "the offset must close on the measured samples"

    def test_no_span_extent_or_contrast_is_nan(self) -> None:
        """A NaN reaching an extent becomes a bar of unknown width and a dB label reading nan."""
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        env[int(1.0 * SR) : int(1.2 * SR)] = np.nan
        env[int(2.5 * SR) : int(2.7 * SR)] = np.nan
        out = _propose(env, -55.0)
        assert isinstance(out, list)
        for span in out:
            assert np.isfinite([span.start, span.end, span.peak_over_floor_db]).all()

    def test_an_envelope_with_no_measurable_sample_is_no_contrast(self) -> None:
        """Nothing was measured, so nothing rose above anything; the reason may not read "nan"."""
        env = np.full(int(3.0 * SR), np.nan)
        out = _propose(env, -55.0)
        assert isinstance(out, NoContrast)
        assert "nan" not in out.reason.lower()

    def test_an_unmeasurable_envelope_raises_no_warning(self) -> None:
        """NoContrast is the answer, not a RuntimeWarning from an empty reduction."""
        env = np.full(int(3.0 * SR), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert isinstance(_propose(env, -55.0), NoContrast)


class TestGroupExtentsIntoRuns:
    """The generic word/extent grouper shared by PREPROCESS's ASR spans and SPEECH's own spans."""

    def test_a_gap_over_the_threshold_starts_a_new_run(self) -> None:
        """Two extents 200 ms apart, with a 150 ms gap threshold, stay two runs."""
        runs = group_extents_into_runs([(0.0, 0.5), (0.7, 1.0)], gap_ms=150.0)
        assert runs == [(0.0, 0.5, [0]), (0.7, 1.0, [1])]

    def test_a_gap_under_the_threshold_merges_into_one_run(self) -> None:
        """Two extents 100 ms apart, with a 150 ms gap threshold, merge into one run."""
        runs = group_extents_into_runs([(0.0, 0.5), (0.6, 1.0)], gap_ms=150.0)
        assert runs == [(0.0, 1.0, [0, 1])]

    def test_input_order_does_not_matter(self) -> None:
        """Extents are sorted by start before grouping; member indices still refer to the input."""
        runs = group_extents_into_runs([(0.6, 1.0), (0.0, 0.5)], gap_ms=150.0)
        assert runs == [(0.0, 1.0, [1, 0])]

    def test_no_extents_is_no_runs(self) -> None:
        """An empty input produces an empty output, not an error."""
        assert group_extents_into_runs([], gap_ms=150.0) == []
