"""The workflow adapter preserves the contract audio_analysis already depends on.

Mocks ``detect_pii_via_subprocess`` at the point ``senselab.text.tasks.pii_detection.api``
resolves it (not a copy on this module) -- the adapter reaches the subprocess dispatch
through ``detect_pii``, so patching anywhere else would not reach the call site.

Also proves the layering the move exists for: the task layer must come back clean of
"pass"/"perturbation" vocabulary and must not import from ``senselab.audio.workflows``,
or the boundary this task drew will drift back one function at a time.
"""

import json
import pathlib

import pytest

from senselab.audio.workflows.audio_analysis.pii import (
    PiiPassReport,
    detect_pii_in_pass,
    report_to_dict,
)
from senselab.text.tasks import pii_detection
from senselab.text.tasks.pii_detection import api as pii_api_module

# ── layering: the whole reason this module exists ───────────────────


def test_task_layer_has_no_workflow_vocabulary() -> None:
    """The workflow words "pass" and "perturbation" must not appear in the standalone task API.

    Scans every source file in ``text/tasks/pii_detection`` rather than asserting on one
    function, so the vocabulary can't drift back in through a different function later.
    """
    pkg_dir = pathlib.Path(pii_detection.__file__).parent
    py_files = sorted(pkg_dir.rglob("*.py"))
    assert py_files, "expected to find source files under text/tasks/pii_detection"
    for path in py_files:
        text = path.read_text()
        assert "perturbation" not in text, f"{path} still mentions 'perturbation' -- workflow vocabulary leaked back"
        assert "senselab.audio.workflows" not in text, f"{path} imports from senselab.audio.workflows"


# ── the adapter's own contract, from the task-6 brief ────────────────


def test_adapter_returns_a_pass_report_carrying_the_perturbation() -> None:
    """The adapter re-attaches ``perturbation``, because the workflow keys artifacts on it.

    ``perturbation`` is workflow vocabulary that the task API deliberately does not carry.
    """
    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"openai/whisper-tiny": [{"text": "hello"}]},
        detectors=[],
    )
    assert isinstance(report, PiiPassReport)
    assert report.perturbation == "raw"


def test_adapter_report_to_dict_is_json_serializable() -> None:
    """``report_to_dict`` must produce a plain, JSON-round-trippable dict."""
    report = detect_pii_in_pass(perturbation="raw", asr_resolved={"m": [{"text": "hello"}]}, detectors=[])
    json.dumps(report_to_dict(report))


def test_cross_asr_corroboration_survives_the_move() -> None:
    """A span in only one of several ASR transcripts is the prototypical hallucination.

    The workflow relies on this gate; the task API's own corroboration is per-input, so
    the adapter must keep doing it across inputs.
    """
    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"a": [{"text": ""}], "b": [{"text": ""}]},
        detectors=[],
    )
    assert report.contains_pii is False


# ── report_to_dict shape, incl. the re-attached per-span perturbation ─


def test_report_to_dict_stamps_perturbation_onto_every_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every span dict in the serialized report carries ``perturbation``.

    ``PiiSpan`` itself carries no such field (it moved to the task layer's ``PiiReport``
    without it); ``report_to_dict`` re-attaches it uniformly from ``report.perturbation``
    at serialization time so the on-disk shape (``pii.json``) is unchanged.
    """

    def fake_subprocess(transcripts_by_asr: dict[str, str], **_: object) -> dict:
        return {
            "spans_by_asr": {"0": [{"text": "John Doe", "category": "PERSON", "source": "presidio", "score": 0.9}]},
            "detectors_used": ["presidio"],
            "failures": {},
        }

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(perturbation="enhanced", asr_resolved={"whisper": [{"text": "Hi John Doe."}]})
    payload = report_to_dict(report)
    assert payload["spans"]
    assert all(s["perturbation"] == "enhanced" for s in payload["spans"])
    assert all(s["asr_model"] == "whisper" for s in payload["spans"])


# ── detect_pii_in_pass — integration with mocked subprocess ──────────
#
# Moved here from src/tests/text/tasks/pii_detection_api_test.py (plan-b Task 6):
# detect_pii_in_pass now lives in this module, not the task layer.


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
                "0": [{"text": "John Doe", "category": "PERSON", "source": "presidio", "score": 0.85}],
                "1": [{"text": "John Doe", "category": "PERSON", "source": "gliner/person", "score": 0.9}],
            },
        )

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={
            "whisper": [{"text": "Hi I am John Doe."}],
            "canary": [{"text": "Hi I am John Doe."}],
        },
    )
    # max_score=0.9 x detector_agreement=1.0 (both Presidio + GLiNER, pooled across the two
    # ASR transcripts) x asr_agreement=1.0 (2/2 ASRs) = 0.9.
    assert report.detection_confidence == pytest.approx(0.9)
    assert report.detector_used == "presidio,gliner"
    assert report.contains_pii is True
    assert {s.asr_model for s in report.spans} == {"whisper", "canary"}


def test_detect_pii_in_pass_detection_confidence_none_when_subprocess_finds_no_detectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both detectors failed to load -> detector_used None -> confidence None."""

    def fake_subprocess(*_: object, **__: object) -> dict:
        return _mock_subprocess_result(
            spans_by_asr={"0": []},
            detectors_used=[],
            failures={"presidio": "ImportError: ...", "gliner": "ImportError: ..."},
        )

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"whisper": [{"text": "Some text."}]},
    )
    assert report.detector_used is None
    assert report.detection_confidence is None


def test_detect_pii_in_pass_detection_confidence_none_when_subprocess_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess crash -> report still produced, confidence stays None."""

    def fake_subprocess(*_: object, **__: object) -> dict:
        raise RuntimeError("subprocess venv build failed")

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"whisper": [{"text": "Some text."}]},
    )
    assert report.detector_used is None
    assert report.detection_confidence is None
    assert "pii_subprocess" in report.failures


def test_detect_pii_in_pass_detection_confidence_zero_when_detectors_ran_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detectors ran, no spans -> 0.0, not None. Distinct signal from "didn't run"."""

    def fake_subprocess(*_: object, **__: object) -> dict:
        return _mock_subprocess_result(spans_by_asr={"0": []})

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"whisper": [{"text": "Some text."}]},
    )
    assert report.detector_used == "presidio,gliner"
    assert report.detection_confidence == 0.0
    assert report.contains_pii is False


def test_detect_pii_in_pass_disabled_path_yields_none_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``detectors=[]`` short-circuit: no subprocess spawn, confidence None."""

    def fake_subprocess(*_: object, **__: object) -> dict:
        raise AssertionError("must not call subprocess when detectors=[]")

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"whisper": [{"text": "Some text."}]},
        detectors=[],
    )
    assert report.detector_used is None
    assert report.detection_confidence is None
    assert "pii_disabled" in report.failures


def test_uncorroborated_single_asr_span_does_not_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """A span only one of >= 2 ASR transcripts produced is the classic hallucination case.

    With two ASR backends available and only one flagging the entity, cross-ASR
    corroboration must withhold ``contains_pii``, even though a span exists.
    """

    def fake_subprocess(transcripts_by_asr: dict[str, str], **_: object) -> dict:
        return _mock_subprocess_result(
            spans_by_asr={
                "0": [{"text": "John Doe", "category": "PERSON", "source": "presidio", "score": 0.9}],
                "1": [],
            },
        )

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"whisper": [{"text": "Hi John Doe."}], "canary": [{"text": "Hi there."}]},
    )
    assert report.n_spans == 1
    assert report.contains_pii is False


def test_single_asr_backend_any_hit_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """With only one ASR backend in the pass, corroboration cannot apply -- any hit counts."""

    def fake_subprocess(transcripts_by_asr: dict[str, str], **_: object) -> dict:
        return _mock_subprocess_result(
            spans_by_asr={"0": [{"text": "John Doe", "category": "PERSON", "source": "presidio", "score": 0.9}]},
        )

    monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)

    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"whisper": [{"text": "Hi John Doe."}]},
    )
    assert report.contains_pii is True
