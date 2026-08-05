"""The speaker axis's three composition terms, each measurable on its own.

Pure functions over plain data, so the axis's composition can be checked without running a model —
which is the point: the change-detection composition these replace could only be judged from a full
run, and it read 0.666 on a clean two-speaker conversation whose per-speaker presence doubt was 0.168.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.attribution import (
    per_speaker_attribution_doubt,
    target_activity_doubt,
    word_location_doubt,
)

BUCKETS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]


def test_unanimous_models_carry_no_attribution_doubt() -> None:
    """Every model placing the same speaker here is the case that must read zero."""
    clusters = {"pyannote": "C0", "sortformer": "C0", "emb/ecapa": "C0"}
    assert per_speaker_attribution_doubt(clusters) == pytest.approx(0.0)


def test_an_even_split_between_two_speakers_saturates() -> None:
    """Two models each, on different speakers: a 50/50 share is maximal doubt for both."""
    clusters = {"a": "C0", "b": "C0", "c": "C1", "d": "C1"}
    assert per_speaker_attribution_doubt(clusters) == pytest.approx(1.0)


def test_the_doubt_is_the_max_over_speakers_not_the_mean() -> None:
    """A confidently-placed speaker must not hide doubt about another one.

    C0 is held by 3 of 5 (H(0.6) = 0.971) and C1 by 1 of 5 (H(0.2) = 0.722). Max is 0.971; mean is
    0.846. The asymmetry is what makes the two distinguishable.
    """
    clusters = {"a": "C0", "b": "C0", "c": "C0", "d": "C1", "e": "SIL"}
    doubt = per_speaker_attribution_doubt(clusters)
    assert doubt == pytest.approx(0.9710, abs=1e-3), "must be the max over speakers"
    assert doubt != pytest.approx(0.8463, abs=1e-3), "must not be the mean over speakers"


def test_models_reporting_silence_stay_in_the_denominator() -> None:
    """A lone detection among silent models must not read as certain."""
    clusters = {"a": "C0", "b": "SIL", "c": "SIL", "d": "SIL"}
    # share 0.25 -> H(0.25) = 0.8113
    assert per_speaker_attribution_doubt(clusters) == pytest.approx(0.8113, abs=1e-3)


def test_no_speaker_present_is_no_claim() -> None:
    """All models silent, or none reporting: None rather than 0.0."""
    assert per_speaker_attribution_doubt({"a": "SIL", "b": "SIL"}) is None
    assert per_speaker_attribution_doubt({}) is None


def test_word_location_doubt_is_coverage_weighted() -> None:
    """A bucket's location doubt is the coverage-weighted mean over the words reaching it."""
    words = [
        {"start": 0.0, "end": 0.1, "temporal_confidence": 0.5},
        {"start": 0.1, "end": 0.2, "temporal_confidence": 1.0},
    ]
    out = word_location_doubt(words, BUCKETS)
    assert out[(0.0, 0.1)] == pytest.approx(0.5)
    assert out[(0.1, 0.2)] == pytest.approx(0.0)


def test_a_bucket_no_word_reaches_has_no_location_doubt() -> None:
    """None, not 0.0: nothing was said there, so nothing localises it."""
    words = [{"start": 0.0, "end": 0.1, "temporal_confidence": 0.5}]
    assert word_location_doubt(words, BUCKETS)[(0.2, 0.3)] is None


def test_a_word_without_a_temporal_confidence_is_skipped() -> None:
    """An unmeasured word contributes nothing rather than counting as fully confident."""
    words = [{"start": 0.0, "end": 0.1, "temporal_confidence": None}]
    assert word_location_doubt(words, BUCKETS)[(0.0, 0.1)] is None


def test_target_active_contributes_no_doubt() -> None:
    """Where the mask is confident the target is active, the attribution question is simply live."""
    regions = [{"start": 0.0, "end": 0.3, "state": "target_active", "uncertainty": 0.24}]
    out = target_activity_doubt(regions, BUCKETS)
    assert out[(0.0, 0.1)] == (None, "target_active")


def test_indeterminate_contributes_its_uncertainty() -> None:
    """Not knowing whether the target was active is not knowing whether anyone is here."""
    regions = [{"start": 0.0, "end": 0.3, "state": "indeterminate", "uncertainty": 1.0}]
    doubt, state = target_activity_doubt(regions, BUCKETS)[(0.0, 0.1)]
    assert doubt == pytest.approx(1.0)
    assert state == "indeterminate"


def test_target_free_is_reported_as_a_state_for_the_caller_to_null() -> None:
    """The function reports the state; the caller turns target_free into no claim at all."""
    regions = [{"start": 0.0, "end": 0.3, "state": "target_free", "uncertainty": 0.05}]
    assert target_activity_doubt(regions, BUCKETS)[(0.0, 0.1)][1] == "target_free"


def test_a_bucket_takes_the_region_it_overlaps_most() -> None:
    """Regions are coarse and a bucket can straddle two; the dominant one wins, deterministically."""
    regions = [
        {"start": 0.0, "end": 0.12, "state": "indeterminate", "uncertainty": 1.0},
        {"start": 0.12, "end": 0.3, "state": "target_active", "uncertainty": 0.0},
    ]
    out = target_activity_doubt(regions, BUCKETS)
    assert out[(0.1, 0.2)][1] == "target_active", "0.08 s of target_active beats 0.02 s indeterminate"


def test_a_bucket_no_region_covers_has_no_state() -> None:
    """No mask region here means the mask said nothing, which is not 'target active'."""
    regions = [{"start": 1.0, "end": 2.0, "state": "target_active", "uncertainty": 0.0}]
    assert target_activity_doubt(regions, BUCKETS)[(0.0, 0.1)] == (None, None)
