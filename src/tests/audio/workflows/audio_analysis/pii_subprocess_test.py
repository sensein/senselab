"""Tests for the subprocess-venv PII dispatcher.

All ``subprocess.run`` calls are mocked. The real worker (Presidio +
GLiNER inside the isolated Python-3.13 venv) is exercised only by
out-of-band runs against a real machine — exercising it here would
require building the actual venv on every test run.

Coverage focuses on the dispatch surface: detector selection (the
``detectors`` arg), JSON-request construction, response parsing, and
the empty-detector short-circuit.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from senselab.audio.workflows.audio_analysis import pii_subprocess
from senselab.audio.workflows.audio_analysis.pii_subprocess import (
    _DEFAULT_GLINER_LABELS,
    _DEFAULT_GLINER_MODEL,
    _GLINER_TO_PRESIDIO_CATEGORY,
    _PRESIDIO_PII_ENTITIES,
    DETECTOR_GLINER,
    DETECTOR_PRESIDIO,
    detect_pii_via_subprocess,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fake_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Skip the real venv build; return a tmp path as the resolved venv dir."""
    venv_dir = tmp_path / "pii-detection"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(pii_subprocess, "ensure_venv", lambda *_, **__: venv_dir)
    return venv_dir


class _SubprocessRecorder:
    """Records calls + returns canned JSON output. Mirrors subprocess.run's signature."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response_json = json.dumps(response)
        self.input_hook: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None

    def __call__(
        self,
        argv: list[str],
        input: str = "",  # noqa: A002 — subprocess.run uses this kw name
        capture_output: bool = False,
        text: bool = False,
        timeout: int = 0,
        env: Optional[dict[str, str]] = None,
        **_: object,
    ) -> subprocess.CompletedProcess:
        parsed_input = json.loads(input) if input else {}
        self.calls.append({"argv": argv, "input": parsed_input, "timeout": timeout})
        response = self._response_json
        if self.input_hook is not None:
            response = json.dumps(self.input_hook(parsed_input))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=response, stderr="")


# ── Default-detectors behavior ──────────────────────────────────────


def test_default_runs_both_presidio_and_gliner(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an explicit ``detectors`` argument both detectors are requested."""
    recorder = _SubprocessRecorder(
        {"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["presidio", "gliner"]}
    )
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample transcript."})

    sent = recorder.calls[0]["input"]
    assert set(sent["detectors"]) == {DETECTOR_PRESIDIO, DETECTOR_GLINER}


def test_explicit_presidio_only_skips_gliner(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``detectors=["presidio"]`` keeps GLiNER out of the worker request."""
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["presidio"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."}, detectors=[DETECTOR_PRESIDIO])

    sent = recorder.calls[0]["input"]
    assert sent["detectors"] == [DETECTOR_PRESIDIO]
    assert DETECTOR_GLINER not in sent["detectors"]


def test_explicit_gliner_only_skips_presidio(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``detectors=["gliner"]`` keeps Presidio out of the worker request."""
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["gliner"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."}, detectors=[DETECTOR_GLINER])

    sent = recorder.calls[0]["input"]
    assert sent["detectors"] == [DETECTOR_GLINER]


def test_empty_detector_list_short_circuits_without_spawning_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``detectors=[]`` returns immediately — no venv build, no subprocess.

    This is the explicit-disable signal; ``ensure_venv`` is never even
    consulted, so a host that's never built the venv before pays nothing
    for opting out.
    """
    monkeypatch.setattr(
        pii_subprocess,
        "ensure_venv",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("must not build venv when detectors=[]")),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("must not spawn subprocess when detectors=[]")),
    )

    result = detect_pii_via_subprocess({"whisper": "Some text.", "canary": "Other text."}, detectors=[])

    assert result["detectors_used"] == []
    assert result["failures"] == {}
    # Same shape as a successful run — empty list per ASR model.
    assert result["spans_by_asr"] == {"whisper": [], "canary": []}


def test_unknown_detector_name_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecognized detector names fail fast at the dispatch layer."""
    monkeypatch.setattr(
        pii_subprocess,
        "ensure_venv",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("must not build venv with bad detector")),
    )

    with pytest.raises(ValueError, match="Unknown PII detector"):
        detect_pii_via_subprocess({"whisper": "x"}, detectors=["regex", "presidio"])


def test_detectors_argument_dedupes_preserving_order(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated entries in ``detectors`` are collapsed but order is kept."""
    recorder = _SubprocessRecorder(
        {"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["gliner", "presidio"]}
    )
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess(
        {"whisper": "Sample."},
        detectors=[DETECTOR_GLINER, DETECTOR_PRESIDIO, DETECTOR_GLINER],
    )

    sent = recorder.calls[0]["input"]
    assert sent["detectors"] == [DETECTOR_GLINER, DETECTOR_PRESIDIO]


# ── Argument plumbing ──────────────────────────────────────────────


def test_request_carries_defaults_for_threshold_and_gliner_settings(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default thresholds + GLiNER model id + labels flow through unchanged."""
    recorder = _SubprocessRecorder(
        {"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["presidio", "gliner"]}
    )
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."})

    sent = recorder.calls[0]["input"]
    assert sent["presidio_score_threshold"] == 0.4
    assert sent["gliner_model"] == _DEFAULT_GLINER_MODEL
    assert sent["gliner_threshold"] == 0.5
    assert sent["gliner_labels"] == list(_DEFAULT_GLINER_LABELS)
    assert sent["presidio_entities"] == list(_PRESIDIO_PII_ENTITIES)
    # The full GLiNER → Presidio category map ships every call.
    assert sent["gliner_label_map"] == _GLINER_TO_PRESIDIO_CATEGORY


def test_caller_overrides_for_gliner_model_and_labels_flow_through(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``gliner_model`` and ``gliner_labels`` reach the worker as supplied."""
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["gliner"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess(
        {"whisper": "Sample."},
        gliner_model="urchade/gliner_multi-v2.1",
        gliner_labels=["person", "email"],
        gliner_threshold=0.7,
    )

    sent = recorder.calls[0]["input"]
    assert sent["gliner_model"] == "urchade/gliner_multi-v2.1"
    assert sent["gliner_labels"] == ["person", "email"]
    assert sent["gliner_threshold"] == 0.7


# ── Response parsing ───────────────────────────────────────────────


def test_response_passes_through_spans_failures_and_detectors_used(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker's three response fields are returned unmodified."""
    canned = {
        "spans_by_asr": {
            "whisper": [
                {"text": "John Doe", "category": "PERSON", "source": "presidio", "score": 0.85},
                {"text": "John Doe", "category": "PERSON", "source": "gliner/person", "score": 0.92},
            ],
            "canary": [],
        },
        "failures": {"presidio": "config glitch"},
        "detectors_used": ["gliner"],
    }
    monkeypatch.setattr(subprocess, "run", _SubprocessRecorder(canned))

    result = detect_pii_via_subprocess({"whisper": "x", "canary": "y"})

    assert result["spans_by_asr"] == canned["spans_by_asr"]
    assert result["failures"] == canned["failures"]
    assert result["detectors_used"] == canned["detectors_used"]


def test_known_detectors_constant_matches_aliases() -> None:
    """The frozenset and the alias constants must agree — guards against drift."""
    from senselab.audio.workflows.audio_analysis.pii_subprocess import _KNOWN_DETECTORS

    assert _KNOWN_DETECTORS == {DETECTOR_PRESIDIO, DETECTOR_GLINER}
