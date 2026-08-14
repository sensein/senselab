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


# ── The scan/decide split ───────────────────────────────────────────


def test_a_scan_carries_evidence_and_no_verdict() -> None:
    """``PiiScan`` must not have grown a verdict field.

    The split only holds if the evidence type stays free of one. A ``contains_pii`` or a
    confidence on ``PiiScan`` would mean the scan had decided something, and every caller
    wanting a different rule would be arguing with a value already computed.
    """
    from senselab.text.tasks.pii_detection.api import PiiScan

    scan = PiiScan()
    for decided in ("contains_pii", "detection_confidence", "detector_used"):
        assert not hasattr(scan, decided), f"PiiScan must not carry {decided!r} — that is a decision"
    assert (scan.spans, scan.detectors_used, scan.failures) == ([], [], {})


def test_deciding_runs_no_detectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """``decide_pii`` must be pure aggregation — no subprocess, no venv, no model load.

    If deciding could re-run detection, "apply a different rule to the same evidence" would
    silently cost another scan, and two rules applied to one recording could disagree
    because they saw different detector output rather than because the rules differ.
    """
    import subprocess

    from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan, decide_pii

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decide_pii must not spawn a subprocess")),
    )

    scan = PiiScan(
        spans=[PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="0", score=0.9)],
        detectors_used=["presidio"],
    )
    report = decide_pii(scan)
    assert report.contains_pii is True
    assert report.detector_used == "presidio"


def test_one_scan_can_yield_different_verdicts_under_different_rules() -> None:
    """The same evidence, two corroboration rules, two answers — which is the whole point.

    A single uncorroborated hit from one of two detectors counts under the permissive rule
    and does not under the strict one. That the decision can change while the scan is
    untouched is what makes the two functions genuinely separable rather than a cosmetic
    split.
    """
    from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan, decide_pii

    scan = PiiScan(
        spans=[PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="0", score=0.9)],
        detectors_used=["presidio", "gliner"],
    )

    strict = decide_pii(scan, require_cross_source_corroboration=True)
    permissive = decide_pii(scan, require_cross_source_corroboration=False)

    assert strict.contains_pii is False, "one of two detectors is not corroboration"
    assert permissive.contains_pii is True
    assert strict.spans == permissive.spans, "the evidence must be identical; only the rule changed"


def test_a_scan_that_never_ran_decides_to_none_not_zero() -> None:
    """A failed scan and a clean scan must stay distinguishable.

    An empty ``detectors_used`` is the first; ``detection_confidence`` therefore has to be
    ``None`` rather than ``0.0``, or a caller reads a failed scan as an all-clear.
    """
    from senselab.text.tasks.pii_detection.api import PiiScan, decide_pii

    report = decide_pii(PiiScan(failures={"pii_subprocess": "venv build failed"}))
    assert report.contains_pii is False
    assert report.detection_confidence is None
    assert report.detector_used is None
    assert report.failures == {"pii_subprocess": "venv build failed"}


def test_detect_pii_is_exactly_the_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    """``detect_pii`` must add no behaviour of its own beyond scan-then-decide.

    A convenience wrapper that quietly differs from its parts is worse than no wrapper:
    the split would be documented but not real.
    """
    from senselab.text.tasks.pii_detection import api

    scan = api.PiiScan(
        spans=[api.PiiSpan(text="Jane", category="PERSON", source="presidio", asr_model="0", score=0.8)],
        detectors_used=["presidio", "gliner"],
    )
    monkeypatch.setattr(api, "scan_for_pii", lambda *a, **k: scan)

    for rule in (True, False):
        composed = api.detect_pii("some text", require_cross_source_corroboration=rule)
        by_hand = api.decide_pii(scan, require_cross_source_corroboration=rule)
        assert composed == by_hand
