"""Tests for ``api._compute_detection_confidence``.

Exercises the in-host scoring plumbing: cross-detector and cross-ASR-model
agreement weighting, and ``None`` vs ``0.0`` propagation for the "detectors
didn't run" vs "ran and found nothing" distinction.

Moved here from ``audio/workflows/audio_analysis/pii_test.py`` alongside the
module under test (see plan-b Task 1). The integration tests that used to
live in this file for ``detect_pii_in_pass`` moved again in plan-b Task 6,
to ``src/tests/audio/workflows/pii_adapter_test.py`` alongside the workflow
adapter that now owns that function -- this file keeps only what actually
tests the standalone task API.
"""

import pytest

from senselab.text.tasks.pii_detection.api import (
    PiiSpan,
    _compute_detection_confidence,
)

# ── _compute_detection_confidence — unit-level ──────────────────────


def _span(*, text: str, category: str, source: str, asr_model: str, score: float) -> PiiSpan:
    """Shorthand for a fully-populated PiiSpan."""
    return PiiSpan(text=text, category=category, source=source, asr_model=asr_model, score=score)


def test_confidence_empty_spans_returns_zero() -> None:
    """No spans → confident negative. ``None`` is reserved for "detectors didn't run"."""
    assert _compute_detection_confidence([], n_asr_models=3, n_detectors_run=2) == 0.0


def test_confidence_single_detector_single_asr_halves_score() -> None:
    """One detector × one of one ASR → score × 0.5 × 1.0."""
    spans = [
        _span(text="John Doe", category="PERSON", source="presidio", asr_model="whisper", score=0.8),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=2)
    # 0.8 * (1/2) * (1/1) = 0.4
    assert conf == pytest.approx(0.4)


def test_confidence_two_detectors_agreeing_doubles_factor() -> None:
    """Presidio + GLiNER on the same span → detector factor 1.0 instead of 0.5."""
    spans = [
        _span(text="John Doe", category="PERSON", source="presidio", asr_model="whisper", score=0.8),
        _span(text="John Doe", category="PERSON", source="gliner/person", asr_model="whisper", score=0.9),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=2)
    # max_score=0.9 * detector_agreement=1.0 * asr_agreement=1.0 = 0.9
    assert conf == pytest.approx(0.9)


def test_confidence_cross_asr_corroboration_scales_with_fraction() -> None:
    """Two of three ASRs flag the same finding → asr_agreement = 2/3."""
    spans = [
        _span(text="John Doe", category="PERSON", source="presidio", asr_model="whisper", score=0.8),
        _span(text="John Doe", category="PERSON", source="presidio", asr_model="canary", score=0.7),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=3, n_detectors_run=2)
    # max_score=0.8 * detector_agreement=0.5 * asr_agreement=(2/3) ≈ 0.267
    assert conf == pytest.approx(0.8 * 0.5 * (2 / 3))


def test_confidence_max_across_independent_findings() -> None:
    """Multiple distinct findings → return the strongest, not the sum."""
    spans = [
        _span(text="John", category="PERSON", source="presidio", asr_model="whisper", score=0.4),
        _span(text="John", category="PERSON", source="gliner/person", asr_model="whisper", score=0.4),
        _span(text="555-1234", category="PHONE_NUMBER", source="presidio", asr_model="whisper", score=0.95),
    ]
    # Phone-number finding wins: 0.95 * 0.5 * 1.0 = 0.475
    # Person finding: 0.4 * 1.0 * 1.0 = 0.4
    conf = _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=2)
    assert conf == pytest.approx(0.475)


def test_confidence_whitespace_only_text_dropped() -> None:
    """Spans whose normalized text is empty after stripping are ignored."""
    spans = [
        _span(text="   ", category="PERSON", source="presidio", asr_model="whisper", score=0.9),
        _span(text="real", category="PERSON", source="presidio", asr_model="whisper", score=0.3),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=2)
    # Only the "real" span counts: 0.3 * 0.5 * 1.0 = 0.15
    assert conf == pytest.approx(0.15)


def test_confidence_missing_per_span_score_treated_as_zero() -> None:
    """A span without a confidence score contributes max_score=0 to its group.

    Doesn't crash; just lowers the per-finding contribution. Matters when
    a detector emits no score (legacy spaCy NER pattern, possible future
    backends).
    """
    spans = [
        PiiSpan(text="John", category="PERSON", source="presidio", asr_model="whisper", score=None),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=2)
    # max_score collapses to 0.0 → overall 0.0
    assert conf == 0.0
