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
    assert _compute_detection_confidence([], n_asr_models=2, n_detectors_run=2) == 0.0


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
    assert _compute_detection_confidence(both, n_asr_models=1, n_detectors_run=2) > _compute_detection_confidence(
        one, n_asr_models=1, n_detectors_run=2
    )


def test_cross_source_agreement_raises_confidence() -> None:
    """A span only one transcript contains is the prototypical ASR hallucination."""
    in_both = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="a", score=0.9),
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="b", score=0.9),
    ]
    in_one = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="a", score=0.9),
    ]
    assert _compute_detection_confidence(in_both, n_asr_models=2, n_detectors_run=2) > _compute_detection_confidence(
        in_one, n_asr_models=2, n_detectors_run=2
    )


def test_agreement_denominator_is_detectors_that_ran_not_detectors_that_exist() -> None:
    """A Presidio-only finding when GLiNER never ran is the best available evidence.

    Dividing by the number of known detectors caps it at 0.5 as though a second
    detector had declined to confirm it — when in fact none was asked.
    """
    spans = [PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9)]
    assert _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=1) == pytest.approx(0.9)


def test_a_third_detector_does_not_rescale_two_detector_agreement() -> None:
    """Adding a third detector to the module must not silently rescale published confidences.

    Two detectors agreeing out of two that ran is still full agreement.
    """
    two = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.8),
        PiiSpan(text="Jane Doe", category="PERSON", source="gliner/name", asr_model="w", score=0.8),
    ]
    assert _compute_detection_confidence(two, n_asr_models=1, n_detectors_run=2) == pytest.approx(0.8)


def test_partial_agreement_among_three_scores_between() -> None:
    """Two of three detectors agreeing scores strictly between one-of-three and full agreement."""
    three_ran_two_agree = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9),
        PiiSpan(text="Jane Doe", category="PERSON", source="gliner/name", asr_model="w", score=0.9),
    ]
    all_three = three_ran_two_agree + [
        PiiSpan(text="Jane Doe", category="PERSON", source="rules/gazetteer", asr_model="w", score=0.9),
    ]
    partial = _compute_detection_confidence(three_ran_two_agree, n_asr_models=1, n_detectors_run=3)
    full = _compute_detection_confidence(all_three, n_asr_models=1, n_detectors_run=3)
    assert 0.0 < partial < full == pytest.approx(0.9)


def test_denominator_never_divides_by_zero() -> None:
    """A defined score beats a crash even for an input the caller should never send.

    ``n_detectors_run=0`` cannot reach here in practice — the caller short-circuits
    before calling this function — but a ``ZeroDivisionError`` in a scoring function
    is a bad failure mode regardless.
    """
    spans = [PiiSpan(text="x", category="PERSON", source="presidio", asr_model="w", score=0.5)]
    assert _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=0) >= 0.0


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


def test_detectors_at_different_granularity_still_corroborate() -> None:
    """Presidio's PERSON and the rules cascade's NAME are the same finding.

    Keying agreement on the raw category made the cascade structurally unable to
    corroborate anything while still counting toward the denominator, so adding it as a
    third detector pushed a finding Presidio and GLiNER both agreed on from 2/2 down to
    2/3. Adding a detector must not make a well-corroborated finding look less certain.
    """
    from senselab.text.tasks.pii_detection.api import corroboration_family

    spans = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9),
        PiiSpan(text="Jane Doe", category="NAME", source="rules/gazetteer", asr_model="w", score=0.9),
    ]
    assert corroboration_family("PERSON") == corroboration_family("NAME")
    # Two of two detectors agree on the family, so the score must reflect full agreement
    # rather than the zero it would score if PERSON and NAME were treated as unrelated.
    agreed = _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=2)
    only_one = _compute_detection_confidence(spans[:1], n_asr_models=1, n_detectors_run=2)
    assert agreed > only_one, f"agreement across granularity did not help: {agreed} vs {only_one}"


def test_an_unmapped_category_is_not_folded_into_a_family() -> None:
    """A label the family map has not seen agrees only with itself.

    The reduction is fine -> coarse and deterministic; guessing which family an unknown
    label belongs to would silently merge unrelated findings.
    """
    from senselab.text.tasks.pii_detection.api import corroboration_family

    assert corroboration_family("SOME_NEW_ENTITY") == "SOME_NEW_ENTITY"
    assert corroboration_family("SOME_NEW_ENTITY") != corroboration_family("PERSON")


def _fake_rules_subprocess(transcripts_by_asr: dict, **_: object) -> dict:
    """Stand-in for ``detect_pii_via_subprocess``: one email finding per non-empty input.

    Patched over the real subprocess dispatch so the scan tests exercise this module's
    span materialisation without standing up the detector venv.
    """
    return {
        "spans_by_asr": {
            index: [{"text": "alice@example.com", "category": "EMAIL_ADDRESS", "source": "rules/email", "score": 0.9}]
            for index in transcripts_by_asr
        },
        "detectors_used": ["rules"],
        "failures": {},
    }


def test_a_finding_is_a_script_line() -> None:
    """An unlocated finding is a ScriptLine with start and end still None."""
    from senselab.text.tasks.pii_detection.api import PiiSpan
    from senselab.utils.data_structures import ScriptLine

    span = PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9)
    assert isinstance(span, ScriptLine), "a finding is a timed piece of text, not a parallel type"
    assert span.start is None and span.end is None, "unlocated until something locates it"


def test_a_located_finding_reports_its_extent_and_speaker_natively() -> None:
    """Location is the inherited start/end/speaker, with no parallel start_s name."""
    from senselab.text.tasks.pii_detection.api import PiiSpan

    span = PiiSpan(
        text="Jane Doe",
        category="PERSON",
        source="presidio",
        asr_model="w",
        start=11.9,
        end=12.4,
        speaker="SPEAKER_00",
    )
    assert (span.start, span.end) == (11.9, 12.4)
    assert span.speaker == "SPEAKER_00"
    assert not hasattr(span, "start_s"), "no parallel name for a field that already exists"


def test_the_timing_provenance_lives_on_the_finding() -> None:
    """A finding records which producer timed it, via the inherited timestamp_model."""
    from senselab.text.tasks.pii_detection.api import PiiSpan

    span = PiiSpan(
        text="Jane",
        category="PERSON",
        source="presidio",
        asr_model="w",
        start=1.0,
        end=1.3,
        timestamp_model="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    assert span.timestamp_model == "Qwen/Qwen3-ForcedAligner-0.6B", (
        "two recognizers timed by one aligner are not independent witnesses"
    )


def test_scanning_a_script_line_carries_its_timing_onto_every_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every finding from a scanned line inherits that line's extent and speaker."""
    from senselab.text.tasks.pii_detection import api as pii_api_module
    from senselab.text.tasks.pii_detection.api import scan_for_pii
    from senselab.utils.data_structures import ScriptLine

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", _fake_rules_subprocess)
    line = ScriptLine(text="call alice@example.com", speaker="SPEAKER_01", start=3.0, end=4.5)
    scan = scan_for_pii(line, detectors=["rules"])
    assert not isinstance(scan, list), "a scalar input yields a scalar scan"
    assert scan.spans, "rules detector found nothing"
    assert all(sp.start == 3.0 and sp.end == 4.5 for sp in scan.spans)
    assert all(sp.speaker == "SPEAKER_01" for sp in scan.spans)


def test_scanning_a_bare_string_leaves_the_finding_unlocated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare string carries no timing, so its findings claim none."""
    from senselab.text.tasks.pii_detection import api as pii_api_module
    from senselab.text.tasks.pii_detection.api import scan_for_pii

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", _fake_rules_subprocess)
    scan = scan_for_pii("call alice@example.com", detectors=["rules"])
    assert not isinstance(scan, list), "a scalar input yields a scalar scan"
    assert scan.spans
    assert all(sp.start is None for sp in scan.spans), "a bare string has no timing to claim"
