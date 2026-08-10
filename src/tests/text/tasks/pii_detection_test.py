"""Characterisation tests for PII detection.

Written before the move from workflows/audio_analysis so the move can be shown
to change nothing but the import path.
"""

from senselab.text.tasks.pii_detection.api import PiiSpan, _compute_detection_confidence


def test_confidence_is_zero_when_no_spans() -> None:
    """'Detectors ran, found nothing' is 0.0. 'Detectors did not run' is None,
    and is carried on the report, not here."""
    assert _compute_detection_confidence([], n_asr_models=2) == 0.0


def test_two_detectors_agreeing_beats_one_detector_alone() -> None:
    """Cross-detector agreement is the strongest 'real entity vs hallucination'
    signal available at this layer, so it must dominate an equal raw score."""
    both = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9),
        PiiSpan(text="Jane Doe", category="PERSON", source="gliner/name", asr_model="w", score=0.9),
    ]
    one = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9),
    ]
    assert _compute_detection_confidence(both, n_asr_models=1) > _compute_detection_confidence(one, n_asr_models=1)


def test_cross_source_agreement_raises_confidence() -> None:
    """A span only one transcript contains is the prototypical ASR hallucination."""
    in_both = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="a", score=0.9),
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="b", score=0.9),
    ]
    in_one = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="a", score=0.9),
    ]
    assert _compute_detection_confidence(in_both, n_asr_models=2) > _compute_detection_confidence(
        in_one, n_asr_models=2
    )
