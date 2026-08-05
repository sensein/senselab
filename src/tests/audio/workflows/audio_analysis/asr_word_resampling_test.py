"""D-27: the asr axis is a gridded resampling of the two-part word confidence.

The defect this replaces, measured on ``english_conversation_higgs_audio_v2_20260805-034348``:
``harvest_asr_votes`` bucketed each model's words by time with ``fully_contained=True`` and took
pairwise WER over the bucket texts. Qwen3-ASR and CrisperWhisper produced **word-identical**
transcripts, yet bucket 0.5-1.5 held "you did that without" against "you did that" — a timing
difference reading as textual disagreement. Mean pairwise WER across buckets was 0.0751 while the
axis reported 0.4266.

Under D-27 the word carries the two doubts separately and the axis resamples them: doubt mass is
``1 - existence_confidence`` and how far it reaches is set by the word's temporal uncertainty. A
word every model agrees on but times loosely is a wide, low-mass contribution; a word they disagree
about is a narrow, high-mass one. The old scheme could not distinguish those, and they call for
different interventions.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.asr import resample_word_doubt

BUCKETS = [(round(i * 0.5, 6), round(i * 0.5 + 1.0, 6)) for i in range(6)]


def _word(text: str, start: float, end: float, existence: float, temporal: float | None) -> dict[str, Any]:
    return {
        "text": text,
        "start": start,
        "end": end,
        "existence_confidence": existence,
        "temporal_confidence": temporal,
    }


def test_a_bucket_no_word_reaches_is_absent_not_certain() -> None:
    """Absence of a claim is not a claim of absence — the rule this codebase turns on."""
    out = resample_word_doubt([_word("hi", 0.1, 0.3, 1.0, 1.0)], BUCKETS)
    assert out[(0.0, 1.0)] == pytest.approx(0.0), "a confident word leaves no doubt where it is"
    assert out.get((2.5, 3.5)) is None, "no word reaches here, so there is nothing to report"


def test_a_doubtful_word_puts_its_doubt_where_it_is() -> None:
    """Existence doubt is the mass; a well-localised word deposits it locally."""
    out = resample_word_doubt([_word("maybe", 2.6, 2.9, 0.25, 1.0)], BUCKETS)
    assert out[(2.5, 3.5)] == pytest.approx(0.75), out
    assert out.get((0.0, 1.0)) is None, "a well-localised word does not reach four buckets away"


def test_identical_text_timed_differently_is_not_textual_disagreement() -> None:
    """The measured defect, stated as a test.

    Two models agree the word was said, so existence doubt is zero, and no amount of disagreement
    about *when* may manufacture doubt about *what*. Under the bucketed-WER scheme this exact case
    produced a full WER in whichever buckets the two readings fell into differently.
    """
    agreed_but_smeared = _word("without", 1.2, 1.6, existence=1.0, temporal=0.1)
    out = resample_word_doubt([agreed_but_smeared], BUCKETS)
    assert all(v == pytest.approx(0.0) for v in out.values() if v is not None), out


def test_a_poorly_localised_word_smears_the_same_mass_wider() -> None:
    """Temporal uncertainty sets the reach, not the amount.

    Same existence doubt, two localisations: the loose one must touch strictly more buckets, and
    neither may invent doubt the word did not carry.
    """
    tight = resample_word_doubt([_word("what", 2.6, 2.9, 0.5, 1.0)], BUCKETS)
    loose = resample_word_doubt([_word("what", 2.6, 2.9, 0.5, 0.0)], BUCKETS)

    tight_touched = {k for k, v in tight.items() if v is not None}
    loose_touched = {k for k, v in loose.items() if v is not None}
    assert loose_touched > tight_touched, f"loose={loose_touched} tight={tight_touched}"
    assert max(v for v in loose.values() if v is not None) <= 0.5 + 1e-9, "reach must not inflate mass"


def test_unmeasured_localisation_does_not_smear() -> None:
    """``temporal_confidence`` is ``None`` for a lone timing source — unmeasured, not zero.

    Treating it as zero confidence would spread a single-witness word across the recording on the
    strength of a measurement nobody made.
    """
    out = resample_word_doubt([_word("solo", 2.6, 2.9, 0.5, None)], BUCKETS)
    touched = {k for k, v in out.items() if v is not None}
    assert touched == {(2.0, 3.0), (2.5, 3.5)}, touched


def test_two_words_in_one_bucket_combine_by_coverage() -> None:
    """A bucket holding one doubtful word among confident ones is partly doubtful, not fully."""
    words = [
        _word("sure", 2.55, 2.75, existence=1.0, temporal=1.0),
        _word("iffy", 2.80, 3.00, existence=0.0, temporal=1.0),
    ]
    out = resample_word_doubt(words, BUCKETS)
    value = out[(2.5, 3.5)]
    assert value is not None and 0.0 < value < 1.0, f"expected a blend, got {value}"


def test_no_words_at_all_reports_nothing() -> None:
    """A run with no transcript has an unmeasured asr axis, not a confident one."""
    assert all(v is None for v in resample_word_doubt([], BUCKETS).values())
