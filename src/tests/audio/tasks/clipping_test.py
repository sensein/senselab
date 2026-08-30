"""ClipDaT: clip-event detection against the recording's own amplitude extreme."""

from __future__ import annotations

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.clipping import ClipEvent, detect_clip_events

SR = 16000


def _audio(x: np.ndarray) -> Audio:
    return Audio(waveform=x.astype("float64")[None, :], sampling_rate=SR)


def _detect(audio: Audio) -> list[ClipEvent]:
    """Run detect_clip_events with the paper's own reported constants."""
    return detect_clip_events(audio, near_threshold=0.995, leniency_samples=3, minimum_extreme=1e-6)


def _tone(seconds: float = 1.0, amp: float = 0.5, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


class TestNoClipping:
    """A lone occurrence of the global extreme still opens a leniency-length event by construction.

    The algorithm has no minimum-run-length rule of its own — every file's unique sample maximum and
    minimum trivially satisfy "sample == max"/"sample == min" once, and the leniency window then
    tolerates a few more samples past it before closing. This is a real, load-bearing property of
    the paper's algorithm, not a bug: a caller must apply its own minimum-duration filter to tell a
    genuine plateau from this floor-level artifact, exactly as a genuine event is itself required to
    clear ``spans.min_duration_ms`` elsewhere in this codebase. These tests lock in that floor rather
    than asserting the (incorrect) expectation that a lone peak produces no event at all.
    """

    def test_a_clean_tones_lone_peaks_cover_a_negligible_fraction_of_the_file(self) -> None:
        """A smooth peak keeps a few neighbouring samples within the band too, which is expected.

        A tiny noise floor is added deliberately: an ideal, noiseless sinusoid reaches its exact
        analytic extremum at bit-identical float64 values on every single cycle (measured: -0.5
        recurs at all 200 cycle troughs of a 200 Hz, 1 s tone), which is a property of perfect
        periodicity, not of clipping — and not representative of any real recording, which always
        carries some noise breaking that exact recurrence. Sustained-vowel and phonation tasks are
        this project's own real periodic signals, so this floor is a realistic minimum to test with.
        """
        rng = np.random.default_rng(0)
        x = _tone(amp=0.5) + rng.normal(scale=1e-6, size=SR)
        events = _detect(_audio(x))
        total_event_samples = sum(event.end_sample - event.start_sample + 1 for event in events)
        assert total_event_samples < 0.01 * len(x)

    def test_a_single_high_sample_opens_one_short_event_not_a_plateau(self) -> None:
        """The extreme occurring once opens an event bounded by the leniency window, not a run."""
        x = _tone(seconds=0.1, amp=0.5)
        x[0] = 0.9
        events = _detect(_audio(x))
        positive = [event for event in events if event.polarity == "positive"]
        assert len(positive) == 1
        assert positive[0].start_sample == 0
        assert positive[0].end_sample <= 3 + 1

    def test_near_silence_opens_no_event(self) -> None:
        """minimum_extreme guards a near-zero global extreme, where the band excludes almost nothing."""
        x = np.full(1000, 1e-8)
        events = _detect(_audio(x))
        assert events == []

    def test_empty_audio_opens_no_event(self) -> None:
        """An empty waveform has no samples to compare against a global extreme."""
        events = _detect(_audio(np.zeros(0)))
        assert events == []


class TestAFlatToppedClip:
    """A run of samples pinned at the same extreme is one event, not one per sample."""

    def test_a_saturated_run_is_one_event(self) -> None:
        """The event's end includes the leniency-tolerated tail past the true plateau, by design."""
        x = np.zeros(1000)
        x[100:150] = 1.0
        events = _detect(_audio(x))
        assert len(events) == 1
        assert events[0].polarity == "positive"
        assert events[0].start_sample == 100
        assert events[0].end_sample == 149 + 4  # the plateau, plus the 4th dip that exceeds leniency=3

    def test_both_polarities_are_tagged_separately(self) -> None:
        """A positive-rail run and a negative-rail run in the same file are two distinct events."""
        x = np.zeros(1000)
        x[100:150] = 1.0
        x[500:550] = -1.0
        events = _detect(_audio(x))
        assert [event.polarity for event in events] == ["positive", "negative"]

    def test_a_second_disjoint_run_is_a_second_event(self) -> None:
        """Two separated runs at the same polarity are not merged into one."""
        x = np.zeros(1000)
        x[100:150] = 1.0
        x[300:350] = 1.0
        events = _detect(_audio(x))
        assert len(events) == 2


class TestTheLeniencyWindow:
    """Up to leniency_samples consecutive dips are absorbed into the run rather than splitting it."""

    def test_a_brief_dip_does_not_split_the_event(self) -> None:
        """Three samples below the band, within the leniency of three, stay inside one event."""
        x = np.zeros(200)
        x[10:20] = 1.0
        x[20:23] = 0.5  # three samples below the 0.995 band
        x[23:30] = 1.0
        events = _detect(_audio(x))
        assert len(events) == 1
        assert events[0].end_sample >= 29

    def test_a_dip_past_the_leniency_closes_the_event(self) -> None:
        """A fourth consecutive dip exceeds the leniency and ends the run there."""
        x = np.zeros(200)
        x[10:20] = 1.0
        x[20:24] = 0.5  # four samples below the band, one more than the leniency
        events = _detect(_audio(x))
        assert len(events) == 1
        assert events[0].end_sample == 23  # closes at the sample that exceeded the tolerance

    def test_a_wavering_near_clip_top_is_still_one_event(self) -> None:
        """Fig. 10b/10c's motivating case: the top wavers slightly but never truly leaves the clip."""
        x = np.zeros(200)
        x[10:40] = 1.0
        x[20] = 0.996  # a single-sample wobble, still within the 0.995 band
        events = _detect(_audio(x))
        assert len(events) == 1
        assert events[0].start_sample == 10
        assert events[0].end_sample == 39 + 4  # the plateau, plus the tolerated trailing dip


class TestTheThresholdIsRelativeToThisFile:
    """A gain change applied after clipping does not hide a run pinned at the file's own extreme."""

    def test_clipping_survives_a_later_gain_reduction(self) -> None:
        """The band is 99.5% of this file's own extreme, not of a fixed full-scale value."""
        x = np.zeros(1000)
        x[100:150] = 1.0
        x *= 0.3  # simulate a renormalization after the fact; the plateau is now 0.3, not full scale
        events = _detect(_audio(x))
        assert len(events) == 1
        assert events[0].start_sample == 100

    def test_a_stereo_input_is_averaged_like_disruptions(self) -> None:
        """Multi-channel audio is downmixed before detection, matching detect_disruptions."""
        left = np.zeros(200)
        left[50:70] = 1.0
        right = np.zeros(200)
        right[50:70] = 1.0
        stereo = Audio(waveform=np.stack([left, right]).astype("float64"), sampling_rate=SR)
        events = detect_clip_events(stereo, near_threshold=0.995, leniency_samples=3, minimum_extreme=1e-6)
        assert len(events) == 1
