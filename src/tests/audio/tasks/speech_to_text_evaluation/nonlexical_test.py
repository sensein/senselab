"""Non-lexical marker handling in transcript normalization (CrisperWhisper fillers).

Some ASR models annotate non-speech events and fillers inline — CrisperWhisper emits
``[cough]``, ``[UH]``, ``[LAUGH]`` and similar. Those markers must not be compared against
models that stay silent in the same span.

Stripping only the brackets is *worse than doing nothing*: ``[cough]`` then normalizes to
the ordinary word ``cough``, which a WER comparison scores as a substitution against a model
that transcribed nothing there. The disagreement is then about annotation convention rather
than about what was said, and it inflates the utterance uncertainty axis with a difference
carrying no signal.
"""

from __future__ import annotations

import pytest

from senselab.audio.tasks.speech_to_text_evaluation.utils import (
    normalize_transcript_for_wer,
    strip_nonlexical_tokens,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[cough]", ""),
        ("[UH] hello", "hello"),
        ("hello [LAUGH] world", "hello world"),
        ("<unk> hello", "hello"),
        ("[cough] [cough] there", "there"),
        ("no markers here", "no markers here"),
        ("", ""),
    ],
)
def test_bracketed_markers_are_removed(raw: str, expected: str) -> None:
    """Square and angle brackets, with their contents."""
    assert strip_nonlexical_tokens(raw) == expected


def test_parentheses_are_left_alone() -> None:
    """Some conventions use parentheses for genuine but uncertain speech.

    Stripping them would delete transcript content rather than annotation, so the pattern
    deliberately covers only square and angle brackets.
    """
    assert strip_nonlexical_tokens("(maybe) hello") == "(maybe) hello"


def test_the_word_inside_the_marker_does_not_survive() -> None:
    """The specific bug: brackets removed but the word kept.

    ``[cough]`` becoming ``cough`` is what turns an annotation into a phantom word.
    """
    assert "cough" not in normalize_transcript_for_wer("[cough] hello")


def test_annotated_and_unannotated_transcripts_agree() -> None:
    """The property that matters for the utterance axis.

    Two models that heard the same speech must not disagree because one annotates
    non-speech events and the other does not.
    """
    annotated = "[cough] [cough] There is [cough] something going on"
    plain = "There is something going on"
    assert normalize_transcript_for_wer(annotated) == normalize_transcript_for_wer(plain)


def test_wer_between_annotated_and_plain_is_zero() -> None:
    """End to end: measured 0.6 before this fix, from three phantom substitutions."""
    from senselab.audio.tasks.speech_to_text_evaluation.utils import calculate_wer

    a = normalize_transcript_for_wer("[cough] [cough] There is [cough] something going on")
    b = normalize_transcript_for_wer("There is something going on")
    assert calculate_wer(b, a) == pytest.approx(0.0)


def test_real_disagreement_still_registers() -> None:
    """The fix must not mask genuine transcription differences."""
    from senselab.audio.tasks.speech_to_text_evaluation.utils import calculate_wer

    a = normalize_transcript_for_wer("[cough] there is something going on")
    b = normalize_transcript_for_wer("there is nothing going on")
    assert calculate_wer(b, a) > 0.0


def test_a_transcript_that_is_only_markers_normalizes_to_empty() -> None:
    """A span where a model heard only non-speech contributes no words."""
    assert normalize_transcript_for_wer("[cough] [breath] [UH]") == ""


def test_ordinary_normalization_is_unchanged() -> None:
    """Punctuation and case handling must survive the addition."""
    assert normalize_transcript_for_wer("First.") == normalize_transcript_for_wer("first!")
