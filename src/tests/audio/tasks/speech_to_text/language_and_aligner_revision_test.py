"""``language`` reaches the Qwen3-ASR and CrisperWhisper workers; the aligner revision is a SHA."""

from __future__ import annotations

import pytest
import torch

import senselab.audio.tasks.speech_to_text.api as api_mod
import senselab.audio.tasks.speech_to_text.crisperwhisper as cw
import senselab.audio.tasks.speech_to_text.qwen as qwen_mod
from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, Language

ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
TAG_SHA = "a" * 40
MAIN_SHA = "b" * 40


def _audio() -> Audio:
    return Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)


def _stub_qwen(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], list[tuple[str, str]], list[dict]]:
    calls: list[dict] = []
    resolved: list[tuple[str, str]] = []
    staged: list[dict] = []

    def _resolve(repo: str, ref: str = "main", **k: object) -> str:
        resolved.append((repo, ref))
        return {"v1": TAG_SHA}.get(ref, MAIN_SHA)

    def _serve(identity: tuple, **k: object) -> dict:
        calls.append({"identity": identity, **k})
        return {"results": [{"text": "hi", "chunks": [{"text": "hi", "start": 0.0, "end": 0.5}]}]}

    def _env(repo: str, revision: str = "main", *, also: object = None, **k: object) -> dict:
        staged.append({"repo": repo, "also": also})
        return {}

    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", _resolve)
    monkeypatch.setattr(qwen_mod, "hf_subprocess_env", _env)
    monkeypatch.setattr(qwen_mod, "ensure_venv", lambda *a, **k: "/fake/venv")
    monkeypatch.setattr(qwen_mod, "venv_python", lambda *a, **k: "/fake/venv/bin/python")
    monkeypatch.setattr(qwen_mod, "serve_in_venv", _serve)
    return calls, resolved, staged


def test_aligner_revision_is_resolved_to_a_sha_before_the_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller's aligner ref is resolved once; the worker payload and identity carry the SHA."""
    calls, resolved, staged = _stub_qwen(monkeypatch)
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B", revision="main")
    qwen_mod.QwenASR.transcribe_with_qwen(
        [_audio()], model=model, device=DeviceType.CPU, return_timestamps=True, aligner_revision="v1"
    )

    assert (ALIGNER, "v1") in resolved
    assert (ALIGNER, "main") not in resolved
    assert calls[0]["init"]["forced_aligner_revision"] == TAG_SHA
    assert TAG_SHA in calls[0]["identity"]
    assert staged[0]["also"] == [(ALIGNER, "v1")]


def test_default_aligner_ref_is_main_and_still_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an aligner revision the aligner is still loaded by a resolved SHA, never by ``main``."""
    calls, _, _ = _stub_qwen(monkeypatch)
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B", revision="main")
    qwen_mod.QwenASR.transcribe_with_qwen([_audio()], model=model, device=DeviceType.CPU)
    assert calls[0]["init"]["forced_aligner_revision"] == MAIN_SHA


def test_align_with_qwen_resolves_the_caller_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone aligner worker payload carries the SHA of the caller's ref."""
    _, resolved, staged = _stub_qwen(monkeypatch)
    sent: dict = {}

    def _run(*a: object, **k: object) -> None:
        import json

        sent.update(json.loads(str(k["input"])))

    monkeypatch.setattr(qwen_mod.subprocess, "run", _run)
    monkeypatch.setattr(qwen_mod, "parse_subprocess_result", lambda *a, **k: {"results": []})
    qwen_mod.QwenASR.align_with_qwen([(_audio(), qwen_mod.ScriptLine(text="hi"), None)], aligner_revision="v1")

    assert (ALIGNER, "v1") in resolved
    assert sent["aligner_revision"] == TAG_SHA


def test_language_reaches_the_qwen_worker_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """``transcribe_audios(language=...)`` sends the English language name in every Qwen request."""
    calls, _, _ = _stub_qwen(monkeypatch)
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B", revision="main")
    api_mod.transcribe_audios([_audio()], model, language=Language(language_code="de"), device=DeviceType.CPU)
    assert calls[0]["request"]["language"] == "German"

    api_mod.transcribe_audios([_audio()], model, device=DeviceType.CPU)
    assert calls[1]["request"]["language"] is None


def test_aligner_revision_kwarg_is_forwarded_by_transcribe_audios(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public entry point passes ``aligner_revision`` through to the Qwen backend."""
    seen: dict = {}

    def _fake(**k: object) -> list:
        seen.update(k)
        return []

    monkeypatch.setattr(api_mod.QwenASR, "transcribe_with_qwen", _fake)
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: MAIN_SHA)
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B", revision="main")
    api_mod.transcribe_audios([_audio()], model, aligner_revision="v1")
    assert seen["aligner_revision"] == "v1"


def test_language_reaches_the_crisperwhisper_worker_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """``transcribe_audios(language=...)`` sends the ISO 639-1 code in every CrisperWhisper request."""
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: MAIN_SHA)
    monkeypatch.setattr(cw, "ensure_venv", lambda *a, **k: "/fake/venv")
    monkeypatch.setattr(cw, "venv_python", lambda *a, **k: "/fake/venv/bin/python")
    monkeypatch.setattr(cw, "resolve_model", lambda *a, **k: (MAIN_SHA, "/fake/snapshot"))
    monkeypatch.setattr(cw, "hf_subprocess_env", lambda *a, **k: {})
    requests: list[dict] = []

    def _serve(*a: object, **k: object) -> dict:
        requests.append(k["request"])  # type: ignore[arg-type]
        return {"results": [{"text": "a", "words": []}]}

    monkeypatch.setattr(cw, "serve_in_venv", _serve)
    model: HFModel = HFModel(path_or_uri="nyralabs/CrisperWhisper2.0_turbo", revision="main")
    api_mod.transcribe_audios([_audio()], model, language=Language(language_code="deu"), device=DeviceType.CPU)
    assert requests[0]["language"] == "de"
