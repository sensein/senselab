"""This module implements some utilities for evaluating a transcription."""

import re

try:
    import jiwer

    JIWER_AVAILABLE = True
except ModuleNotFoundError:
    JIWER_AVAILABLE = False

# TODO: add more metrics which take into account the meaning/intention

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s']")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NONLEXICAL_PATTERN = re.compile(r"[\[<][^\]>]*[\]>]")
"""Bracketed non-lexical markers: ``[cough]``, ``[UH]``, ``<unk>``, ``[LAUGH]``.

Only square and angle brackets. Parentheses are left alone because some conventions use
them for genuine speech (uncertain or overlapping words), so stripping them would delete
transcript content rather than annotation.
"""


def strip_nonlexical_tokens(text: str) -> str:
    """Remove bracketed non-lexical markers and their contents.

    ASR models that annotate fillers and non-speech events — CrisperWhisper emits
    ``[cough]``, ``[UH]``, ``[LAUGH]`` and similar — must not have those markers compared
    against models that stay silent in the same span. Removing the brackets alone is worse
    than doing nothing: ``[cough]`` then normalizes to the ordinary word ``cough``, which a
    WER comparison counts as a substitution error against a model that transcribed nothing
    there. The disagreement is then about annotation *convention*, not about what was said,
    and it inflates the asr uncertainty axis with a difference that carries no signal.

    The same applies to grapheme-to-phoneme conversion: ``[cough]`` yields phonemes for a
    word nobody spoke, and those get aligned against real acoustics.

    Args:
        text (str): raw transcript text.

    Returns:
        str: text with bracketed markers removed (may be empty).
    """
    if not text:
        return ""
    return _WHITESPACE_PATTERN.sub(" ", _NONLEXICAL_PATTERN.sub(" ", text)).strip()


def normalize_transcript_for_wer(text: str) -> str:
    """Lowercase, drop non-lexical markers, strip punctuation, collapse whitespace.

    The shared surface-normalization applied before WER-style comparisons so
    that ``"first."`` vs ``"first!"`` and ``"I"`` vs ``"i"`` don't count as
    errors (moved here from the audio-analysis workflow — architecture-review
    T049 — so task- and workflow-level WER share one definition).

    Non-lexical markers are removed first, so a model that annotates ``[cough]`` is not
    scored as having said the word "cough" against one that annotates nothing — see
    :func:`strip_nonlexical_tokens`.

    Args:
        text (str): raw transcript text.

    Returns:
        str: normalized text (may be empty).
    """
    if not text:
        return ""
    cleaned = _PUNCTUATION_PATTERN.sub(" ", strip_nonlexical_tokens(text).lower())
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate the Word Error Rate (WER) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The WER score.

    Examples:
        >>> calculate_wer("hello world", "hello duck")
        0.5
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    return jiwer.wer(reference, hypothesis)


def calculate_mer(reference: str, hypothesis: str) -> float:
    """Calculate the Match Error Rate (MER) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The MER score.

    Examples:
        >>> calculate_mer("hello world", "hello duck")
        0.5
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )
    return jiwer.mer(reference, hypothesis)


def calculate_wil(reference: str, hypothesis: str) -> float:
    """Calculate the Word Information Lost (WIL) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The WIL score.

    Examples:
        >>> calculate_wil("hello world", "hello duck")
        0.75
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )
    return jiwer.wil(reference, hypothesis)


def calculate_wip(reference: str, hypothesis: str) -> float:
    """Calculate the Word Information Preserved (WIP) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The WIP score.

    Examples:
        >>> calculate_wip("hello world", "hello duck")
        0.25
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )
    return jiwer.wip(reference, hypothesis)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate the Character Error Rate (CER) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The CER score.

    Examples:
        >>> calculate_cer("hello world", "hello duck")
        0.45454545454545453
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )
    cer_value = jiwer.cer(reference, hypothesis)
    if isinstance(cer_value, dict):
        return float(cer_value.get("cer", 0.0))  # Extract CER if returned in a dictionary
    return float(cer_value)  # Ensure output is always a float
