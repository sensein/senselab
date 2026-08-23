"""Tests for the subprocess-venv PII dispatcher.

All ``subprocess.run`` calls are mocked. The real worker (Presidio +
GLiNER inside the isolated Python-3.13 venv) is exercised only by
out-of-band runs against a real machine — exercising it here would
require building the actual venv on every test run.

Coverage focuses on the dispatch surface: detector selection (the
``detectors`` arg), JSON-request construction, response parsing, and
the empty-detector short-circuit.

Moved here from ``audio/workflows/audio_analysis/pii_subprocess_test.py``
alongside the module under test (see plan-b Task 1).
"""

import io
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from senselab.text.tasks.pii_detection import subprocess_backend as pii_subprocess
from senselab.text.tasks.pii_detection.subprocess_backend import (
    _DEFAULT_GLINER_LABELS,
    _DEFAULT_GLINER_MODEL,
    _GLINER_TO_PRESIDIO_CATEGORY,
    _PRESIDIO_PII_ENTITIES,
    DETECTOR_GLINER,
    DETECTOR_LLM,
    DETECTOR_PRESIDIO,
    DETECTOR_RULES,
    detect_pii_via_subprocess,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fake_resolve_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake the GLiNER ref resolution every gliner-enabled test triggers.

    detect_pii_via_subprocess resolves the GLiNER ref to a commit SHA before staging;
    faking it here keeps every test in this module independent of network reachability
    or this host's local HF cache state. Autouse because most tests below exercise the
    default detector set (both presidio and gliner), so opting in per-test would just
    repeat this line everywhere it matters.
    """
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)


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


def test_default_runs_presidio_gliner_and_rules(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an explicit ``detectors`` argument all three detectors are requested."""
    recorder = _SubprocessRecorder(
        {"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["presidio", "gliner", "rules"]}
    )
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample transcript."})

    sent = recorder.calls[0]["input"]
    assert set(sent["detectors"]) == {DETECTOR_PRESIDIO, DETECTOR_GLINER, DETECTOR_RULES}


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
    # A resolved commit SHA, never the mutable "main" ref (see _fake_resolve_revision).
    assert sent["gliner_revision"] == "f" * 40
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
    from senselab.text.tasks.pii_detection.subprocess_backend import _KNOWN_DETECTORS

    assert _KNOWN_DETECTORS == {DETECTOR_PRESIDIO, DETECTOR_GLINER, DETECTOR_RULES, DETECTOR_LLM}


def test_the_llm_detector_is_known_but_not_default() -> None:
    """The gap between the two sets is the whole point of the opt-in.

    Collapsing them either way is a real defect: making ``llm`` default-on ties a scan's
    result to whether a server happened to be listening, and dropping it from
    ``_KNOWN_DETECTORS`` would reject it by name and leave it out of the agreement
    denominator on the runs where a caller did enable it.
    """
    from senselab.text.tasks.pii_detection.subprocess_backend import _DEFAULT_DETECTORS, _KNOWN_DETECTORS

    assert _DEFAULT_DETECTORS == {DETECTOR_PRESIDIO, DETECTOR_GLINER, DETECTOR_RULES}
    assert _KNOWN_DETECTORS - _DEFAULT_DETECTORS == {DETECTOR_LLM}


def test_the_llm_detector_is_not_sent_to_the_worker(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It runs host-side, so naming it must not put it in the worker's payload.

    Leaving it in would have the worker report an unknown detector it cannot load,
    turning an opt-in host-side scan into a spurious venv failure.
    """
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["presidio"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."}, detectors=[DETECTOR_PRESIDIO, DETECTOR_LLM])

    assert recorder.calls[0]["input"]["detectors"] == [DETECTOR_PRESIDIO]


def test_gliner_only_still_ships_the_cascade_source_for_windowing(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GLiNER needs ``_gliner_chunks`` from rules.py even when the rules detector is off.

    Without the source the worker falls back to one whole-text pass, which is the
    truncation this windowing exists to remove — and it would fail silently, since a
    truncated scan still returns spans.
    """
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["gliner"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."}, detectors=[DETECTOR_GLINER])

    sent = recorder.calls[0]["input"]
    assert sent["rules_source"], "GLiNER-only must still carry the cascade source"
    assert "_gliner_chunks" in sent["rules_source"]


def test_presidio_only_does_not_pay_for_the_cascade_source(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The file read stays opt-in: neither detector that needs it is running here."""
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["presidio"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."}, detectors=[DETECTOR_PRESIDIO])

    assert recorder.calls[0]["input"]["rules_source"] is None


# ── GLiNER loads from the staged snapshot, never through the Hub ─────


class _OfflineHubGliner:
    """Stands in for the venv's ``GLiNER``, refusing a repo id the way offline mode does.

    ``gliner``'s own ``from_pretrained`` reaches the Hub tree-listing API even when every file is
    cached, and that call raises under ``HF_HUB_OFFLINE=1`` — which is what the parent sets for this
    worker. So the stub raises on anything that is not an existing directory: a test that passes has
    demonstrated the load needed no Hub call.
    """

    loaded_from: list[str] = []
    loaded_kwargs: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id: object, **kwargs: object) -> "_OfflineHubGliner":
        """Load from a local directory; anything else is a Hub call, which fails offline."""
        if not Path(str(model_id)).is_dir():
            raise RuntimeError(f"OfflineModeIsEnabled: cannot reach the Hub for {model_id!r}")
        cls.loaded_from.append(str(model_id))
        cls.loaded_kwargs.append(dict(kwargs))
        return cls()

    def predict_entities(self, text: str, labels: list, threshold: float = 0.5) -> list:
        """No findings; these tests are about the load, not the detections."""
        return []


@pytest.fixture
def worker_runner(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> Callable[..., dict[str, Any]]:
    """Run the worker script in-process with the venv-only imports stubbed.

    The script under test is the same string the subprocess executes, reading its request from stdin
    and answering on stdout, so what is exercised is the worker's own load path rather than a
    paraphrase of it. ``gliner`` and ``torch`` are the two venv-only imports its GLiNER branch makes.
    """

    def _run(payload: dict[str, Any], **stubs: Any) -> dict[str, Any]:  # noqa: ANN401
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        try:
            exec(  # noqa: S102 — executing the worker string is the point of this harness
                compile(pii_subprocess._PII_WORKER_SCRIPT, "<pii-worker>", "exec"), {"__name__": "__main__"}
            )
        finally:
            for name, previous in saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        return json.loads(capsys.readouterr().out)

    return _run


def _gliner_stub_modules() -> dict[str, Any]:
    """A fake ``gliner`` (offline-strict) and a fake ``torch`` (no CUDA), as sys.modules entries."""
    _OfflineHubGliner.loaded_from = []
    _OfflineHubGliner.loaded_kwargs = []
    gliner_module = types.ModuleType("gliner")
    gliner_module.GLiNER = _OfflineHubGliner  # type: ignore[attr-defined]
    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    return {"gliner": gliner_module, "torch": torch_module}


def _gliner_worker_payload(load_path: Optional[str], sha: str = "b" * 40) -> dict[str, Any]:
    """A GLiNER-only worker request, optionally carrying the staged snapshot path."""
    return {
        "transcripts": {"whisper": "Sample transcript."},
        "detectors": [DETECTOR_GLINER],
        "presidio_entities": [],
        "presidio_score_threshold": 0.4,
        "gliner_model": _DEFAULT_GLINER_MODEL,
        "gliner_model_path": load_path,
        "gliner_revision": sha,
        "gliner_labels": ["person"],
        "gliner_threshold": 0.5,
        "gliner_label_map": {},
        "rules_source": None,
    }


def test_the_worker_loads_gliner_from_the_staged_snapshot_directory(
    worker_runner: Callable[..., dict[str, Any]], tmp_path: Path
) -> None:
    """GLiNER is loaded from the staged path, so offline mode needs no Hub call."""
    snapshot = tmp_path / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    output = worker_runner(_gliner_worker_payload(str(snapshot)), **_gliner_stub_modules())
    assert output["detectors_used"] == [DETECTOR_GLINER], f"gliner did not load: {output.get('failures')}"
    assert output["failures"] == {}
    assert _OfflineHubGliner.loaded_from == [str(snapshot)], "the loader must receive a local path, not a repo id"


def test_the_worker_records_the_commit_it_loaded_from(
    worker_runner: Callable[..., dict[str, Any]], tmp_path: Path
) -> None:
    """Loading from a path must not cost the provenance: the SHA is still recorded."""
    snapshot = tmp_path / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    output = worker_runner(_gliner_worker_payload(str(snapshot)), **_gliner_stub_modules())
    assert output["gliner_revision"] == "b" * 40


def test_the_worker_falls_back_to_the_repo_id_when_nothing_was_staged(
    worker_runner: Callable[..., dict[str, Any]],
) -> None:
    """A parent that could not stage leaves the child online, so the repo id is still the load path."""
    output = worker_runner(_gliner_worker_payload(None), **_gliner_stub_modules())
    assert output["failures"], "the stub refuses a repo id, which is what an offline Hub call does"
    assert _OfflineHubGliner.loaded_from == []


def test_the_parent_sends_the_staged_snapshot_path_for_gliner(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The payload carries the staged directory, and still carries the resolved SHA."""
    snapshot = tmp_path / "snapshots" / ("f" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setattr("senselab.utils.dependencies.resolve_model", lambda *a, **k: ("f" * 40, snapshot))
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["gliner"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."}, detectors=[DETECTOR_GLINER])

    sent = recorder.calls[0]["input"]
    assert sent["gliner_model_path"] == str(snapshot)
    assert Path(sent["gliner_model_path"]).is_dir(), "the worker is given a directory it can open"
    assert sent["gliner_revision"] == "f" * 40, "the resolved commit still travels, for the provenance"
    assert sent["gliner_model"] == _DEFAULT_GLINER_MODEL, "the repo id is still named"


def test_the_parent_sends_no_snapshot_path_when_staging_fails(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unstageable model leaves the child online rather than pointed at a path to nothing."""

    def _fail(*_a: object, **_k: object) -> tuple[str, Path]:
        raise RuntimeError("not cached and the Hub is unreachable")

    monkeypatch.setattr("senselab.utils.dependencies.resolve_model", _fail)
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["gliner"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."}, detectors=[DETECTOR_GLINER])

    assert recorder.calls[0]["input"]["gliner_model_path"] is None


def test_presidio_only_sends_no_gliner_snapshot_path(fake_venv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is staged when GLiNER is not requested, so nothing is pointed at either."""
    recorder = _SubprocessRecorder({"spans_by_asr": {"whisper": []}, "failures": {}, "detectors_used": ["presidio"]})
    monkeypatch.setattr(subprocess, "run", recorder)

    detect_pii_via_subprocess({"whisper": "Sample."}, detectors=[DETECTOR_PRESIDIO])

    assert recorder.calls[0]["input"]["gliner_model_path"] is None
