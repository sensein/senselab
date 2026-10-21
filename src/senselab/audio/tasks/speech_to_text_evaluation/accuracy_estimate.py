"""Reference-free accuracy estimate for a transcript from a second recognizer and word confidences.

``estimate_transcript_accuracy`` compares a primary transcript (e.g. CrisperWhisper) with a
secondary one (e.g. Qwen3-ASR) and reads the primary's word confidences, overall and per time span,
then maps those measurements to an estimated word error rate of the primary through a calibration
profile fitted on public conversational speech with references
(``data/accuracy_calibration.yaml``; derivation in ``specs/20261017-104000-asr-apple-silicon/``).

Both transcripts are normalised the same way before comparison: bracketed markers dropped,
lowercased, punctuation stripped, and (by default) hesitation fillers removed, because one
recognizer transcribes fillers verbatim and the other omits them inconsistently.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel

from senselab.audio.tasks.speech_to_text_evaluation.utils import normalize_transcript_for_wer
from senselab.utils.data_structures import ScriptLine

CALIBRATION_PATH = Path(__file__).parent / "data" / "accuracy_calibration.yaml"

FILLER_WORDS = frozenset({"uh", "um", "uhm", "umm", "er", "erm", "ah", "eh", "ehm", "hm", "hmm", "mm", "mhm"})
"""Hesitation fillers removed from both sides when ``exclude_fillers`` is set (after normalisation)."""


class AccuracyEstimate(BaseModel):
    """Measurements and the calibrated WER estimate for one time span (or the whole transcript).

    Attributes:
        start: Span start in seconds, or ``None`` for the whole transcript.
        end: Span end in seconds, or ``None`` for the whole transcript.
        n_primary_words: Normalised primary words in the span.
        n_secondary_words: Normalised secondary words in the span.
        inter_system_wer: WER of the primary scored against the secondary taken as reference;
            ``None`` when the secondary has no words in the span.
        inter_system_disagreement: Word edit distance between the two, divided by the longer of the
            two word counts, in [0, 1]; ``None`` when neither has words. The calibration's input.
        mean_confidence: Mean primary word confidence over the span's scored words.
        min_confidence: Lowest primary word confidence in the span.
        low_confidence_fraction: Fraction of scored primary words below the profile's
            ``low_confidence_threshold``.
        estimated_wer: Calibrated estimate of the primary's WER against a reference.
        estimated_wer_low: Lower end of the calibrated interval.
        estimated_wer_high: Upper end of the calibrated interval.
    """

    start: Optional[float] = None
    end: Optional[float] = None
    n_primary_words: int
    n_secondary_words: int
    inter_system_wer: Optional[float] = None
    inter_system_disagreement: Optional[float] = None
    mean_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    low_confidence_fraction: Optional[float] = None
    estimated_wer: Optional[float] = None
    estimated_wer_low: Optional[float] = None
    estimated_wer_high: Optional[float] = None


class TranscriptAccuracyEstimate(BaseModel):
    """Overall and per-span accuracy estimates for one primary/secondary transcript pair.

    Attributes:
        overall: The estimate over the whole transcript.
        segments: One estimate per requested span, in the order given.
        calibration: The calibration profile's ``name`` and ``version``.
        interval_coverage: Nominal coverage of the ``estimated_wer_low``..``estimated_wer_high`` interval.
    """

    overall: AccuracyEstimate
    segments: List[AccuracyEstimate]
    calibration: str
    interval_coverage: float


@lru_cache(maxsize=4)
def load_calibration(path: Optional[str] = None) -> Dict:
    """Read an accuracy calibration profile.

    Args:
        path: A profile YAML; ``None`` reads the packaged ``data/accuracy_calibration.yaml``.

    Returns:
        The profile mapping (``model``, ``low_confidence_threshold``, ``interval``, ...).
    """
    with open(path or CALIBRATION_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _words(line: ScriptLine) -> List[Tuple[str, Optional[float], Optional[float], Optional[float]]]:
    """Return ``(text, start, end, score)`` for each word of a line, falling back to its text."""
    if line.chunks:
        return [(c.text or "", c.start, c.end, c.score) for c in line.chunks]
    return [(w, None, None, None) for w in (line.text or "").split()]


def _normalise(text: str, exclude_fillers: bool) -> List[str]:
    tokens = normalize_transcript_for_wer(text).split()
    if exclude_fillers:
        tokens = [t for t in tokens if t not in FILLER_WORDS]
    return tokens


def _in_span(start: Optional[float], end: Optional[float], span: Tuple[float, float]) -> bool:
    if start is None or end is None:
        return False
    mid = (start + end) / 2.0
    return span[0] <= mid < span[1]


def _edits(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Word-level Levenshtein distance between two token lists."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp in enumerate(hypothesis, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp))
        previous = current
    return previous[-1]


def _wer(reference: Sequence[str], hypothesis: Sequence[str]) -> Optional[float]:
    """Word error rate over token lists; ``None`` for an empty reference."""
    if not reference:
        return None
    return _edits(reference, hypothesis) / len(reference)


def _features(
    primary: List[Tuple[str, Optional[float], Optional[float], Optional[float]]],
    secondary: List[Tuple[str, Optional[float], Optional[float], Optional[float]]],
    exclude_fillers: bool,
    low_threshold: float,
) -> Dict:
    prim_tokens: List[str] = []
    scores: List[float] = []
    for text, _, _, score in primary:
        tokens = _normalise(text, exclude_fillers)
        prim_tokens.extend(tokens)
        if tokens and score is not None:
            scores.append(float(score))
    sec_tokens: List[str] = []
    for text, _, _, _ in secondary:
        sec_tokens.extend(_normalise(text, exclude_fillers))
    edits = _edits(sec_tokens, prim_tokens)
    longer = max(len(prim_tokens), len(sec_tokens))
    return {
        "n_primary_words": len(prim_tokens),
        "n_secondary_words": len(sec_tokens),
        "inter_system_wer": edits / len(sec_tokens) if sec_tokens else None,
        "inter_system_disagreement": edits / longer if longer else None,
        "mean_confidence": (sum(scores) / len(scores)) if scores else None,
        "min_confidence": min(scores) if scores else None,
        "low_confidence_fraction": (sum(s < low_threshold for s in scores) / len(scores)) if scores else None,
    }


def predict_wer(
    disagreement: Optional[float],
    mean_confidence: Optional[float],
    n_words: int,
    profile: Dict,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Map measurements to ``(estimated_wer, low, high)`` with a calibration profile.

    The profile's ``model`` holds a linear map on the inter-system disagreement and
    ``1 - mean_confidence`` (``confidence_free`` when no confidence is available); the interval
    half-width is the profile's length-standardised residual quantile divided by ``sqrt(n_words)``.

    Args:
        disagreement: The inter-system disagreement (edits over the longer word count).
        mean_confidence: Mean primary word confidence, or ``None``.
        n_words: Normalised primary words the measurements cover.
        profile: A calibration profile from :func:`load_calibration`.

    Returns:
        The estimate and its interval, each clipped at 0; all ``None`` without a disagreement.
    """
    if disagreement is None or n_words <= 0:
        return None, None, None
    if mean_confidence is not None:
        coef = profile["model"]["with_confidence"]
        estimate = (
            coef["intercept"]
            + coef["disagreement"] * disagreement
            + coef["one_minus_confidence"] * (1.0 - mean_confidence)
        )
        quantile = coef["residual_quantile"]
    else:
        coef = profile["model"]["confidence_free"]
        estimate = coef["intercept"] + coef["disagreement"] * disagreement
        quantile = coef["residual_quantile"]
    half = quantile / math.sqrt(n_words)
    estimate = max(0.0, estimate)
    return estimate, max(0.0, estimate - half), estimate + half


def estimate_transcript_accuracy(
    primary: ScriptLine,
    secondary: ScriptLine,
    *,
    segments: Optional[Sequence[Tuple[float, float]]] = None,
    exclude_fillers: bool = True,
    calibration_path: Optional[str] = None,
) -> TranscriptAccuracyEstimate:
    """Estimate the primary transcript's accuracy from a second recognizer and its word confidences.

    Words are assigned to a span by their midpoint; spans therefore need word timestamps on both
    lines (``chunks`` with ``start``/``end``). The whole-transcript estimate uses every word.

    Args:
        primary: The transcript whose accuracy is estimated; word ``chunks`` carry ``score`` as the
            word confidence when available.
        secondary: An independent transcript of the same audio.
        segments: Optional ``(start, end)`` spans in seconds, e.g. speaker turns.
        exclude_fillers: Remove :data:`FILLER_WORDS` from both sides before comparing.
        calibration_path: A calibration profile other than the packaged one.

    Returns:
        A :class:`TranscriptAccuracyEstimate` with the overall and per-span estimates.

    Raises:
        ValueError: Spans were requested and a line has no word timestamps.
    """
    profile = load_calibration(calibration_path)
    low_threshold = float(profile["low_confidence_threshold"])
    prim_words = _words(primary)
    sec_words = _words(secondary)

    def _estimate(prim: List, sec: List, span: Optional[Tuple[float, float]]) -> AccuracyEstimate:
        feats = _features(prim, sec, exclude_fillers, low_threshold)
        est, low, high = predict_wer(
            feats["inter_system_disagreement"], feats["mean_confidence"], feats["n_primary_words"], profile
        )
        return AccuracyEstimate(
            start=span[0] if span else None,
            end=span[1] if span else None,
            estimated_wer=est,
            estimated_wer_low=low,
            estimated_wer_high=high,
            **feats,
        )

    per_span: List[AccuracyEstimate] = []
    if segments:
        for line, name in ((primary, "primary"), (secondary, "secondary")):
            if not line.chunks or any(c.start is None or c.end is None for c in line.chunks):
                raise ValueError(f"The {name} transcript has no word timestamps; spans cannot be assigned.")
        for span in segments:
            span_t = (float(span[0]), float(span[1]))
            per_span.append(
                _estimate(
                    [w for w in prim_words if _in_span(w[1], w[2], span_t)],
                    [w for w in sec_words if _in_span(w[1], w[2], span_t)],
                    span_t,
                )
            )
    return TranscriptAccuracyEstimate(
        overall=_estimate(prim_words, sec_words, None),
        segments=per_span,
        calibration=f"{profile['name']}@{profile['version']}",
        interval_coverage=float(profile["interval"]["coverage"]),
    )
