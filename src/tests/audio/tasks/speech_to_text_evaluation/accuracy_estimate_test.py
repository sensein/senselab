"""Tests for the two-recognizer transcript accuracy estimate."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from senselab.audio.tasks.speech_to_text_evaluation.accuracy_estimate import (
    CALIBRATION_PATH,
    estimate_transcript_accuracy,
    load_calibration,
    predict_wer,
)
from senselab.utils.data_structures import ScriptLine


def _line(words: list[tuple[str, float, float, float | None]]) -> ScriptLine:
    chunks = [ScriptLine(text=t, start=s, end=e, score=p) for t, s, e, p in words]
    return ScriptLine(text=" ".join(t for t, *_ in words), start=words[0][1], end=words[-1][2], chunks=chunks)


PRIMARY = _line(
    [
        ("So", 0.0, 0.3, 0.9),
        ("[UM]", 0.4, 0.6, 0.6),
        ("I", 0.7, 0.8, 0.95),
        ("think,", 0.8, 1.1, 0.9),
        ("uh,", 1.2, 1.4, 0.7),
        ("we", 1.5, 1.6, 0.9),
        ("should", 1.6, 1.9, 0.8),
        ("go.", 1.9, 2.2, 0.3),
        ("Yes", 3.0, 3.3, 0.9),
        ("please", 3.3, 3.8, 0.9),
    ]
)
SECONDARY = _line(
    [
        ("So", 0.0, 0.3, None),
        ("I", 0.7, 0.8, None),
        ("think", 0.8, 1.1, None),
        ("we", 1.5, 1.6, None),
        ("should", 1.6, 1.9, None),
        ("gone", 1.9, 2.2, None),
        ("yes", 3.0, 3.3, None),
        ("please", 3.3, 3.8, None),
    ]
)


def test_fillers_are_removed_from_both_sides_before_comparison() -> None:
    """Bracketed and plain fillers do not count as disagreements; one real substitution remains."""
    out = estimate_transcript_accuracy(PRIMARY, SECONDARY)
    assert out.overall.n_primary_words == 8
    assert out.overall.n_secondary_words == 8
    assert out.overall.inter_system_wer == pytest.approx(1 / 8)
    assert out.overall.inter_system_disagreement == pytest.approx(1 / 8)


def test_keeping_fillers_counts_them_as_deletions() -> None:
    """With ``exclude_fillers=False`` plain fillers stay (bracketed markers are always dropped)."""
    out = estimate_transcript_accuracy(PRIMARY, SECONDARY, exclude_fillers=False)
    assert out.overall.n_primary_words == 9
    assert out.overall.inter_system_wer == pytest.approx(2 / 8), "the primary's 'uh' is an insertion"
    assert out.overall.inter_system_disagreement == pytest.approx(2 / 9)


def test_confidence_summaries_use_the_primary_lexical_words() -> None:
    """Mean/min/low fraction are over the primary's scored words that survive normalisation."""
    out = estimate_transcript_accuracy(PRIMARY, SECONDARY)
    scores = [0.9, 0.95, 0.9, 0.9, 0.8, 0.3, 0.9, 0.9]
    threshold = load_calibration()["low_confidence_threshold"]
    assert out.overall.mean_confidence == pytest.approx(sum(scores) / len(scores))
    assert out.overall.min_confidence == pytest.approx(0.3)
    assert out.overall.low_confidence_fraction == pytest.approx(sum(s < threshold for s in scores) / len(scores))


def test_segments_split_words_by_midpoint() -> None:
    """Each span gets its own measurements; the disagreement lands in the first span only."""
    out = estimate_transcript_accuracy(PRIMARY, SECONDARY, segments=[(0.0, 2.5), (2.5, 4.0)])
    first, second = out.segments
    assert (first.start, first.end) == (0.0, 2.5)
    assert first.n_primary_words == 6 and first.inter_system_wer == pytest.approx(1 / 6)
    assert second.n_primary_words == 2 and second.inter_system_wer == 0.0
    assert second.estimated_wer is not None and first.estimated_wer is not None
    assert first.estimated_wer > second.estimated_wer


def test_empty_span_has_no_estimate() -> None:
    """A span with no words reports no WER rather than zero."""
    out = estimate_transcript_accuracy(PRIMARY, SECONDARY, segments=[(10.0, 20.0)])
    span = out.segments[0]
    assert span.n_primary_words == 0
    assert span.inter_system_wer is None and span.inter_system_disagreement is None
    assert span.estimated_wer is None


def test_segments_need_word_timestamps() -> None:
    """Spans cannot be assigned without word times."""
    text_only = ScriptLine(text="so I think we should go")
    with pytest.raises(ValueError, match="secondary"):
        estimate_transcript_accuracy(PRIMARY, text_only, segments=[(0.0, 1.0)])
    overall = estimate_transcript_accuracy(PRIMARY, text_only)
    assert overall.overall.n_secondary_words == 6


def test_interval_narrows_with_more_words_and_contains_the_estimate() -> None:
    """The calibrated interval is centred on the estimate and shrinks as 1/sqrt(n)."""
    profile = load_calibration()
    est10, lo10, hi10 = predict_wer(0.2, 0.85, 10, profile)
    est1000, lo1000, hi1000 = predict_wer(0.2, 0.85, 1000, profile)
    assert est10 == pytest.approx(est1000)
    assert est10 is not None and est1000 is not None
    assert lo10 is not None and hi10 is not None and lo1000 is not None and hi1000 is not None
    assert lo1000 <= est1000 <= hi1000
    assert (hi10 - est10) == pytest.approx((hi1000 - est1000) * math.sqrt(100))


def test_more_disagreement_and_lower_confidence_raise_the_estimate() -> None:
    """The packaged map is monotone in both measurements."""
    profile = load_calibration()
    assert predict_wer(0.4, 0.9, 100, profile)[0] > predict_wer(0.1, 0.9, 100, profile)[0]  # type: ignore[operator]
    assert predict_wer(0.2, 0.6, 100, profile)[0] > predict_wer(0.2, 0.95, 100, profile)[0]  # type: ignore[operator]


def test_packaged_profile_carries_its_derivation() -> None:
    """The profile names its data and fit, per the data/ convention."""
    profile = load_calibration()
    assert Path(CALIBRATION_PATH).exists()
    for key in ("name", "version", "derivation", "low_confidence_threshold", "model", "interval"):
        assert key in profile
