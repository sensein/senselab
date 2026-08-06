"""The speaker axis's three composition terms, each measurable on its own.

Pure functions over plain data, so the axis's composition can be checked without running a model —
which is the point: the change-detection composition these replace could only be judged from a full
run, and it read 0.666 on a clean two-speaker conversation whose per-speaker presence doubt was 0.168.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.attribution import (
    speaker_assignment_doubt,
    target_activity_doubt,
    word_coverage,
)

BUCKETS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]


def test_unanimous_models_carry_no_attribution_doubt() -> None:
    """Every model placing the same speaker here is the case that must read zero."""
    clusters = {"pyannote": "C0", "sortformer": "C0", "emb/ecapa": "C0"}
    assert speaker_assignment_doubt(clusters) == pytest.approx(0.0)


def test_an_even_split_between_two_speakers_saturates() -> None:
    """Two models each, on different speakers: an even split across the answers is maximal doubt."""
    clusters = {"a": "C0", "b": "C0", "c": "C1", "d": "C1"}
    assert speaker_assignment_doubt(clusters) == pytest.approx(1.0)


def test_no_speaker_is_privileged_when_no_target_is_given() -> None:
    """The doubt is over *all* the answers the models gave, not the worst single speaker.

    Absent a target embedding the question is "do we know who is talking", so electing one speaker and
    reporting its doubt answers a question nobody asked. Three models say C0, one says C1, one says
    silence: the spread over those three answers is H([0.6, 0.2, 0.2]) / log2(3) = 0.8650. The old
    ``max`` over per-speaker binary entropies gave 0.9710 — C0's own H(0.6) — which is a statement
    about C0, not about the assignment.
    """
    clusters = {"a": "C0", "b": "C0", "c": "C0", "d": "C1", "e": "SIL"}
    doubt = speaker_assignment_doubt(clusters)
    assert doubt == pytest.approx(0.8650, abs=1e-3), "must be the spread over every answer given"
    assert doubt != pytest.approx(0.9710, abs=1e-3), "must not elect a single speaker"


def test_two_answers_are_unchanged_by_dropping_the_max() -> None:
    """Binary entropy is symmetric, so ``H(p) = H(1-p)``: with two answers the two forms coincide.

    Which is why this correction is invisible on a two-speaker recording and still wrong in principle —
    the ``max`` only started electing a speaker once three or more answers were in play.
    """
    assert speaker_assignment_doubt({"a": "C0", "b": "C0", "c": "C1"}) == pytest.approx(0.9183, abs=1e-3)


def test_a_target_restores_the_binary_question() -> None:
    """Given a reference embedding the question *is* about one speaker, and this is that measure.

    The hook exists so the two modes stay distinguishable rather than one silently standing in for the
    other. C1 is held by 1 of 5 -> H(0.2) = 0.7219, regardless of how the other four models split.
    """
    clusters = {"a": "C0", "b": "C0", "c": "C0", "d": "C1", "e": "SIL"}
    assert speaker_assignment_doubt(clusters, target="C1") == pytest.approx(0.7219, abs=1e-3)


def test_models_reporting_silence_stay_in_the_denominator() -> None:
    """A lone detection among silent models must not read as certain."""
    clusters = {"a": "C0", "b": "SIL", "c": "SIL", "d": "SIL"}
    # share 0.25 -> H(0.25) = 0.8113
    assert speaker_assignment_doubt(clusters) == pytest.approx(0.8113, abs=1e-3)


def test_no_speaker_present_is_no_claim() -> None:
    """All models silent, or none reporting: None rather than 0.0."""
    assert speaker_assignment_doubt({"a": "SIL", "b": "SIL"}) is None
    assert speaker_assignment_doubt({}) is None


def test_word_coverage_is_the_fraction_of_the_bucket_words_occupy() -> None:
    """A proportion in ``[0, 1]``, in the bucket's own units — no threshold here."""
    words = [{"start": 0.0, "end": 0.05}, {"start": 0.1, "end": 0.2}]
    out = word_coverage(words, BUCKETS)
    assert out[(0.0, 0.1)] == pytest.approx(0.5)
    assert out[(0.1, 0.2)] == pytest.approx(1.0)


def test_a_bucket_no_word_reaches_has_zero_coverage() -> None:
    """Zero is a measurement here, not an imputed value: no word occupies any of the bucket.

    This is the one the speaker axis gates on. A bucket with no words has no speech to attribute, so
    the axis makes no claim about who is speaking there rather than reporting doubt.
    """
    words = [{"start": 0.0, "end": 0.1}]
    assert word_coverage(words, BUCKETS)[(0.2, 0.3)] == pytest.approx(0.0)


def test_overlapping_words_cannot_push_coverage_over_one() -> None:
    """Two recognizers' words overlapping the same span is still one span of speech."""
    words = [{"start": 0.0, "end": 0.1}, {"start": 0.02, "end": 0.09}]
    assert word_coverage(words, BUCKETS)[(0.0, 0.1)] == pytest.approx(1.0)


def test_a_word_with_an_unusable_span_is_skipped() -> None:
    """A malformed word contributes nothing rather than crashing the harvest."""
    words: list[dict[str, Any]] = [{"start": None, "end": 0.1}, {"start": 0.0, "end": 0.05}]
    assert word_coverage(words, BUCKETS)[(0.0, 0.1)] == pytest.approx(0.5)


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
