"""Characterisation tests for PII detection.

Written before the move from workflows/audio_analysis so the move can be shown
to change nothing but the import path.
"""

import pytest

from senselab.text.tasks.pii_detection.api import (
    PiiReport,
    PiiSpan,
    _compute_detection_confidence,
    detect_pii,
    flatten_script_line,
)
from senselab.utils.data_structures import ScriptLine


def test_confidence_is_zero_when_no_spans() -> None:
    """Detectors that ran and found nothing score 0.0.

    "Detectors did not run" is None instead, and is carried on the report rather than here.
    """
    assert _compute_detection_confidence([], n_asr_models=2) == 0.0


def test_two_detectors_agreeing_beats_one_detector_alone() -> None:
    """Cross-detector agreement must dominate an equal raw score.

    It is the strongest "real entity vs hallucination" signal available at this layer.
    """
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


# ── flatten_script_line ─────────────────────────────────────────────


def test_flatten_plain_script_line() -> None:
    """A flat line with no chunks returns its own text unchanged."""
    assert flatten_script_line(ScriptLine(text="my name is Jane")) == "my name is Jane"


def test_flatten_joins_nested_chunks_depth_first() -> None:
    """A word-level ASR result and a segment-level one must scan identically.

    Whisper returns nested chunks; MMS alignment returns them too. If nesting changed what
    got scanned, PII coverage would silently depend on the backend.

    The parent carries a ``speaker`` (not just ``chunks``) because ``ScriptLine`` requires at
    least one of ``text``/``speaker`` at every level, including containers with no text of
    their own — a bare ``chunks=[...]`` line is not constructible.
    """
    line = ScriptLine(
        text=None,
        speaker="spk1",
        chunks=[
            ScriptLine(text="my name is"),
            ScriptLine(text="Jane Doe"),
        ],
    )
    assert flatten_script_line(line) == "my name is Jane Doe"


def test_flatten_ignores_a_speaker_only_line() -> None:
    """Diarization ScriptLines carry a speaker and no text.

    They contribute nothing rather than raising — a mixed list is a normal input.
    """
    assert flatten_script_line(ScriptLine(speaker="spk1")) == ""


def test_flatten_drops_whitespace_only_entries() -> None:
    """A whitespace-only chunk contributes nothing to the flattened text.

    Same ``speaker``-on-the-container requirement as the depth-first test above.
    """
    line = ScriptLine(speaker="spk1", chunks=[ScriptLine(text="  "), ScriptLine(text="Jane")])
    assert flatten_script_line(line) == "Jane"


# ── detect_pii ───────────────────────────────────────────────────────


def test_detect_pii_with_no_detectors_short_circuits() -> None:
    """``detectors=[]`` means "the caller deliberately turned this off".

    It must be distinguishable from "the check failed" and from "the check found nothing" —
    an auditor reading the report needs all three apart.
    """
    report = detect_pii("my name is Jane Doe", detectors=[])
    assert isinstance(report, PiiReport)  # narrows the scalar/list union for mypy and for the reader
    assert report.detector_used is None
    assert report.contains_pii is False
    assert report.detection_confidence is None
    assert "pii_disabled" in report.failures


def test_detect_pii_on_empty_text_does_not_spawn_a_subprocess() -> None:
    """A whitespace-only input never reaches ``detect_pii_via_subprocess``."""
    report = detect_pii("   ")
    assert isinstance(report, PiiReport)
    assert report.n_spans == 0
    assert report.detector_used is None


def test_detect_pii_returns_one_report_per_input_in_order() -> None:
    """A sequence input returns one report per element, same length and order."""
    reports = detect_pii(["", "  ", ""], detectors=[])
    assert isinstance(reports, list)
    assert len(reports) == 3


def test_detect_pii_single_input_returns_a_bare_report() -> None:
    """A scalar input returns a bare report, not a one-element list."""
    assert not isinstance(detect_pii("", detectors=[]), list)
