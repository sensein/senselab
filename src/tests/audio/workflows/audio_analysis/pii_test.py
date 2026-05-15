"""Tests for ``pii.detect_pii_in_pass`` and ``pii._compute_detection_confidence``.

Mocks ``detect_pii_via_subprocess`` (the subprocess dispatch layer) so this
file exercises the in-host plumbing: report construction, cross-detector
and cross-ASR-model agreement weighting, and ``None`` propagation for the
"detectors didn't run" case.
"""

import pytest

from senselab.audio.workflows.audio_analysis import pii as pii_module
from senselab.audio.workflows.audio_analysis.pii import (
    PiiSpan,
    _compute_detection_confidence,
    detect_pii_in_pass,
)

# ── _compute_detection_confidence — unit-level ──────────────────────


def _span(*, text: str, category: str, source: str, asr_model: str, score: float) -> PiiSpan:
    """Shorthand for a fully-populated PiiSpan."""
    return PiiSpan(text=text, category=category, source=source, asr_model=asr_model, score=score)


def test_confidence_empty_spans_returns_zero() -> None:
    """No spans → confident negative. ``None`` is reserved for "detectors didn't run"."""
    assert _compute_detection_confidence([], n_asr_models=3) == 0.0


def test_confidence_single_detector_single_asr_halves_score() -> None:
    """One detector × one of one ASR → score × 0.5 × 1.0."""
    spans = [
        _span(text="John Doe", category="PERSON", source="presidio", asr_model="whisper", score=0.8),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=1)
    # 0.8 * (1/2) * (1/1) = 0.4
    assert conf == pytest.approx(0.4)


def test_confidence_two_detectors_agreeing_doubles_factor() -> None:
    """Presidio + GLiNER on the same span → detector factor 1.0 instead of 0.5."""
    spans = [
        _span(text="John Doe", category="PERSON", source="presidio", asr_model="whisper", score=0.8),
        _span(text="John Doe", category="PERSON", source="gliner/person", asr_model="whisper", score=0.9),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=1)
    # max_score=0.9 * detector_agreement=1.0 * asr_agreement=1.0 = 0.9
    assert conf == pytest.approx(0.9)


def test_confidence_cross_asr_corroboration_scales_with_fraction() -> None:
    """Two of three ASRs flag the same finding → asr_agreement = 2/3."""
    spans = [
        _span(text="John Doe", category="PERSON", source="presidio", asr_model="whisper", score=0.8),
        _span(text="John Doe", category="PERSON", source="presidio", asr_model="canary", score=0.7),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=3)
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
    conf = _compute_detection_confidence(spans, n_asr_models=1)
    assert conf == pytest.approx(0.475)


def test_confidence_whitespace_only_text_dropped() -> None:
    """Spans whose normalized text is empty after stripping are ignored."""
    spans = [
        _span(text="   ", category="PERSON", source="presidio", asr_model="whisper", score=0.9),
        _span(text="real", category="PERSON", source="presidio", asr_model="whisper", score=0.3),
    ]
    conf = _compute_detection_confidence(spans, n_asr_models=1)
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
    conf = _compute_detection_confidence(spans, n_asr_models=1)
    # max_score collapses to 0.0 → overall 0.0
    assert conf == 0.0


# ── detect_pii_in_pass — integration with mocked subprocess ─────────


def _mock_subprocess_result(
    spans_by_asr: dict[str, list[dict]],
    detectors_used: list[str] | None = None,
    failures: dict[str, str] | None = None,
) -> dict:
    return {
        "spans_by_asr": spans_by_asr,
        "detectors_used": list(detectors_used) if detectors_used is not None else ["presidio", "gliner"],
        "failures": failures or {},
    }


def test_detect_pii_in_pass_populates_detection_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: confidence computed and stored on the report."""

    def fake_subprocess(transcripts_by_asr: dict[str, str], **_: object) -> dict:
        return _mock_subprocess_result(
            spans_by_asr={
                "whisper": [
                    {"text": "John Doe", "category": "PERSON", "source": "presidio", "score": 0.85},
                ],
                "canary": [
                    {"text": "John Doe", "category": "PERSON", "source": "gliner/person", "score": 0.9},
                ],
            },
        )

    monkeypatch.setattr(pii_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        pass_label="raw_16k",
        asr_resolved={
            "whisper": [{"text": "Hi I am John Doe."}],
            "canary": [{"text": "Hi I am John Doe."}],
        },
    )
    # max_score=0.9 × detector_agreement=1.0 (both Presidio + GLiNER) ×
    # asr_agreement=1.0 (2/2 ASRs) = 0.9.
    assert report.detection_confidence == pytest.approx(0.9)
    assert report.detector_used == "presidio,gliner"


def test_detect_pii_in_pass_detection_confidence_none_when_subprocess_finds_no_detectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both detectors failed to load → detector_used None → confidence None."""

    def fake_subprocess(*_: object, **__: object) -> dict:
        return _mock_subprocess_result(
            spans_by_asr={"whisper": []},
            detectors_used=[],
            failures={"presidio": "ImportError: ...", "gliner": "ImportError: ..."},
        )

    monkeypatch.setattr(pii_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        pass_label="raw_16k",
        asr_resolved={"whisper": [{"text": "Some text."}]},
    )
    assert report.detector_used is None
    assert report.detection_confidence is None


def test_detect_pii_in_pass_detection_confidence_none_when_subprocess_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess crash → report still produced, confidence stays None."""

    def fake_subprocess(*_: object, **__: object) -> dict:
        raise RuntimeError("subprocess venv build failed")

    monkeypatch.setattr(pii_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        pass_label="raw_16k",
        asr_resolved={"whisper": [{"text": "Some text."}]},
    )
    assert report.detector_used is None
    assert report.detection_confidence is None
    assert "pii_subprocess" in report.failures


def test_detect_pii_in_pass_detection_confidence_zero_when_detectors_ran_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detectors ran, no spans → 0.0, not None. Distinct signal from "didn't run"."""

    def fake_subprocess(*_: object, **__: object) -> dict:
        return _mock_subprocess_result(spans_by_asr={"whisper": []})

    monkeypatch.setattr(pii_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        pass_label="raw_16k",
        asr_resolved={"whisper": [{"text": "Some text."}]},
    )
    assert report.detector_used == "presidio,gliner"
    assert report.detection_confidence == 0.0


def test_detect_pii_in_pass_disabled_path_yields_none_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``detectors=[]`` short-circuit: no subprocess spawn, confidence None."""

    def fake_subprocess(*_: object, **__: object) -> dict:
        raise AssertionError("must not call subprocess when detectors=[]")

    monkeypatch.setattr(pii_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        pass_label="raw_16k",
        asr_resolved={"whisper": [{"text": "Some text."}]},
        detectors=[],
    )
    assert report.detector_used is None
    assert report.detection_confidence is None
    assert "pii_disabled" in report.failures
