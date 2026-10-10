"""The decision-model runner: a pinned store or nothing, and answers read back as probabilities.

``specs/20261003-clef-second-opinion/design.md`` is the design.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from senselab.text.tasks.decision_model import (
    QUESTIONS,
    OllamaPin,
    PinMismatchError,
    ask_decisions,
    ask_second_opinion,
    read_answers,
    verify_pin,
)
from senselab.text.tasks.decision_model.second_opinion import MORE_THAN_ONE, opinion_state

WEIGHTS = b"not really a gguf, but bytes with a digest"
CONFIG = b'{"model_format": "gguf"}'
SYSTEM = b"Classify the context using the supplied schema."


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _store(root: Path, *, weights: bytes = WEIGHTS, size: int | None = None, layer: str | None = None) -> Path:
    """An Ollama model store holding one manifest, ``clef:27b``, its weights blob and a system layer."""
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    (blobs / _digest(weights).replace(":", "-")).write_bytes(weights)
    (blobs / _digest(SYSTEM).replace(":", "-")).write_bytes(SYSTEM)
    manifest = {
        "schemaVersion": 2,
        "config": {"digest": _digest(CONFIG), "size": len(CONFIG)},
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": layer or _digest(weights),
                "size": len(weights) if size is None else size,
            },
            {"mediaType": "application/vnd.ollama.image.system", "digest": _digest(SYSTEM), "size": len(SYSTEM)},
        ],
    }
    path = root / "manifests" / "registry.ollama.ai" / "library" / "clef" / "27b"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _manifest(root: Path) -> Path:
    return root / "manifests" / "registry.ollama.ai" / "library" / "clef" / "27b"


def _pin(root: Path, **overrides: str) -> OllamaPin:
    """The pin of the store as written, with any digest replaced."""
    held = {
        "blob_digest": _digest(WEIGHTS),
        "config_digest": _digest(CONFIG),
        "manifest_digest": _digest(_manifest(root).read_bytes()),
        **overrides,
    }
    return OllamaPin(name="clef", tag="27b", **held)


PIN_DIGEST = _digest(WEIGHTS)


class TestThePin:
    """A store serves only when its manifest and its blob carry the pinned digests."""

    def test_the_pinned_store_verifies_and_names_its_blob(self, tmp_path: Path) -> None:
        """The good path returns the weights' path."""
        store = _store(tmp_path / "models")
        assert verify_pin(store, _pin(store)).read_bytes() == WEIGHTS

    def test_a_retagged_manifest_with_another_config_is_refused(self, tmp_path: Path) -> None:
        """The tag moved: same name, another image."""
        store = _store(tmp_path / "models")
        with pytest.raises(PinMismatchError, match="manifest config"):
            verify_pin(store, _pin(store, config_digest=_digest(b"other")))

    def test_a_manifest_edited_in_place_is_refused(self, tmp_path: Path) -> None:
        """Another system prompt or parameters layer is another model, under the same weights."""
        store = _store(tmp_path / "models")
        pin = _pin(store)
        _manifest(store).write_text(_manifest(store).read_text(encoding="utf-8") + " ", encoding="utf-8")
        with pytest.raises(PinMismatchError, match="manifest is"):
            verify_pin(store, pin)

    def test_a_tampered_small_layer_is_refused(self, tmp_path: Path) -> None:
        """Every layer the manifest names is hashed, not only the weights."""
        store = _store(tmp_path / "models")
        (store / "blobs" / _digest(SYSTEM).replace(":", "-")).write_bytes(b"Answer whatever you like.")
        with pytest.raises(PinMismatchError, match="hashes to"):
            verify_pin(store, _pin(store))

    def test_a_manifest_naming_other_weights_is_refused(self, tmp_path: Path) -> None:
        """The manifest's model layer is not the pinned blob."""
        store = _store(tmp_path / "models", layer=_digest(b"other weights"))
        with pytest.raises(PinMismatchError, match="model layer"):
            verify_pin(store, _pin(store))

    def test_a_truncated_blob_is_refused_before_it_is_hashed(self, tmp_path: Path) -> None:
        """Size is checked first, so a partial copy fails cheaply."""
        store = _store(tmp_path / "models", size=len(WEIGHTS) + 1)
        with pytest.raises(PinMismatchError, match="bytes"):
            verify_pin(store, _pin(store))

    def test_a_blob_whose_content_differs_is_refused(self, tmp_path: Path) -> None:
        """Same size, other bytes: only the full hash catches it."""
        store = _store(tmp_path / "models")
        blob = store / "blobs" / PIN_DIGEST.replace(":", "-")
        blob.write_bytes(b"X" * len(WEIGHTS))
        with pytest.raises(PinMismatchError, match="hashes to"):
            verify_pin(store, _pin(store))

    def test_a_missing_manifest_is_refused(self, tmp_path: Path) -> None:
        """No pull was ever made into this store."""
        (tmp_path / "models").mkdir()
        with pytest.raises(PinMismatchError, match="no readable manifest"):
            verify_pin(tmp_path / "models", OllamaPin("clef", "27b", PIN_DIGEST, PIN_DIGEST, PIN_DIGEST))

    def test_a_completed_hash_is_remembered_and_a_changed_file_is_hashed_again(self, tmp_path: Path) -> None:
        """The marker is keyed on size and mtime, so rewriting the blob invalidates it."""
        store = _store(tmp_path / "models")
        verified = tmp_path / "verified"
        pin = _pin(store)
        verify_pin(store, pin, verified_dir=verified)
        assert len(list(verified.iterdir())) == 1
        verify_pin(store, pin, verified_dir=verified)
        blob = store / "blobs" / PIN_DIGEST.replace(":", "-")
        stat = blob.stat()
        blob.write_bytes(b"X" * len(WEIGHTS))
        os.utime(blob, (stat.st_atime, stat.st_mtime + 10))
        with pytest.raises(PinMismatchError, match="hashes to"):
            verify_pin(store, pin, verified_dir=verified)


def _answers(other: float = 0.02, instructions: float = 0.03, identifier: float = 0.05, free: float = 0.9) -> dict:
    return {
        "other_voice": {
            "choice": MORE_THAN_ONE if other > 0.5 else "one",
            "probabilities": {"one": 1 - other, MORE_THAN_ONE: other, "unclear": 0.0},
        },
        "instructions_spoken": {"noul": instructions},
        "policy_identifier_present": {"noul": identifier},
        "masked_text_free_of_identifiers": {"noul": free},
        "off_task_speech": {"choice": "some", "probabilities": {"none": 0.25, "some": 0.5, "extensive": 0.25}},
    }


class TestTheAnswers:
    """Each answer is a probability of yes; other_voice is the probability of more_than_one."""

    def test_every_question_is_read_as_a_probability(self) -> None:
        """The three probabilities, keyed by question."""
        opinion = read_answers(_answers(other=0.85, identifier=0.99))
        assert opinion.probabilities == {
            "other_voice": 0.85,
            "instructions_spoken": 0.03,
            "policy_identifier_present": 0.99,
            "masked_text_free_of_identifiers": 0.9,
            "off_task_speech": 0.75,
        }
        assert opinion.choices == {"other_voice": MORE_THAN_ONE, "off_task_speech": "some"}
        assert opinion.class_probabilities["off_task_speech"] == {"none": 0.25, "some": 0.5, "extensive": 0.25}

    def test_an_off_task_answer_without_its_classes_is_an_error(self) -> None:
        """A choice whose probabilities name none of its classes carries no confidence."""
        answers = _answers()
        answers["off_task_speech"] = {"choice": "some", "probabilities": {"maybe": 1.0}}
        with pytest.raises(ValueError, match="off_task_speech"):
            read_answers(answers)
        del answers["off_task_speech"]
        with pytest.raises(ValueError, match="off_task_speech"):
            read_answers(answers)

    def test_a_masked_text_answer_without_a_probability_is_an_error(self) -> None:
        """The new boolean question is required like the others."""
        answers = _answers()
        answers["masked_text_free_of_identifiers"] = {"noul": None}
        with pytest.raises(ValueError, match="masked_text_free_of_identifiers"):
            read_answers(answers)

    def test_the_policy_is_the_reviewers_and_the_nature_comes_from_task_guidance(self) -> None:
        """Both prompts read one policy text and one task-guidance file."""
        from senselab.text.tasks.decision_model.second_opinion import QUESTION_SET_VERSION
        from senselab.text.tasks.pii_detection.redaction_review import redaction_policy_text, task_nature_description

        state = opinion_state("dog cat", {"task": "animal-fluency"}, "dog cat")
        assert state["redaction_policy"] == redaction_policy_text()
        assert state["task_nature"] == task_nature_description("animal-fluency") != ""
        assert "state.redaction_policy" in QUESTIONS["policy_identifier_present"]["instructions"]
        assert "state.masked_text" in QUESTIONS["masked_text_free_of_identifiers"]["instructions"]
        assert QUESTIONS["off_task_speech"]["type"] == "choice" and QUESTION_SET_VERSION == 4

    def test_an_unanswered_question_is_an_error_not_a_zero(self) -> None:
        """A missing answer must not read as a confident no."""
        answers = _answers()
        del answers["policy_identifier_present"]
        with pytest.raises(ValueError, match="policy_identifier_present"):
            read_answers(answers)

    def test_a_choice_without_class_probabilities_is_an_error(self) -> None:
        """The choice alone carries no confidence."""
        answers = _answers()
        answers["other_voice"] = {"choice": "one"}
        with pytest.raises(ValueError, match="class probabilities"):
            read_answers(answers)

    def test_the_state_separates_the_instructions_from_the_stimulus(self) -> None:
        """The stimulus the participant is asked to say is not the instructions."""
        from senselab.text.tasks.pii_detection.redaction_review import redaction_policy_text

        state = opinion_state(
            "the rainbow passage",
            {"task": "reading", "instructions": "Read aloud.", "asked_to_say": "When the sunlight"},
            "the [PERSON] passage",
        )
        assert state == {
            "task": "reading",
            "task_nature": "",
            "speech_type": "",
            "instructions": "Read aloud.",
            "stimulus": "When the sunlight",
            "task_content": "",
            "redaction_policy": redaction_policy_text(),
            "transcript": "the rainbow passage",
            "masked_text": "the [PERSON] passage",
        }

    def test_the_question_set_is_sent_whole(self) -> None:
        """Ask receives the state and every question."""
        seen: list[Any] = []

        def ask(state: Any, questions: Any) -> dict:  # noqa: ANN401
            seen.append((state, questions))
            return _answers()

        ask_second_opinion(ask, "hello", {"task": "free-speech"}, "[PERSON]")
        assert seen[0][1] is QUESTIONS and seen[0][0]["transcript"] == "hello"
        assert seen[0][0]["masked_text"] == "[PERSON]"


class _Recorder(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 — the http.server name
        length = int(self.headers["Content-Length"])
        type(self).bodies.append({"path": self.path, **json.loads(self.rfile.read(length))})
        payload = json.dumps({"answers": _answers()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # noqa: ANN401
        return


@pytest.fixture
def recorder() -> Iterator[str]:
    """A loopback server that records each request and answers every question."""
    _Recorder.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_a_request_is_deterministic_by_construction(recorder: str) -> None:
    """Temperature 0 and the seed travel with every request, to the decision endpoint."""
    answers = ask_decisions(recorder, "clef:27b", {"transcript": "hi"}, QUESTIONS, seed=7)
    assert set(answers) == set(QUESTIONS)
    body = _Recorder.bodies[0]
    assert body["path"] == "/v1/systemone"
    assert body["model"] == "clef:27b"
    assert body["options"] == {"temperature": 0, "seed": 7}


class _FakeProcess:
    """Stands in for ``ollama serve``: records the environment it was started with."""

    started: list[dict[str, str]] = []

    def __init__(self, args: list[str], *, env: dict[str, str], **kw: Any) -> None:  # noqa: ANN401
        type(self).started.append(dict(env))
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.mark.parametrize("num_parallel", [1, 4])
def test_the_server_answers_as_many_requests_as_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_parallel: int
) -> None:
    """``num_parallel`` reaches the server as OLLAMA_NUM_PARALLEL."""
    from contextlib import nullcontext

    from senselab.text.tasks.decision_model import ollama

    _FakeProcess.started = []
    monkeypatch.setattr(ollama, "verify_pin", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(ollama.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(ollama.urllib.request, "urlopen", lambda *a, **kw: nullcontext())
    pin = OllamaPin(name="clef", tag="27b", blob_digest="b", config_digest="c", manifest_digest="m")
    with ollama.OllamaServer(tmp_path / "ollama", tmp_path, pin, num_parallel=num_parallel, require_gpu=False):
        pass
    assert _FakeProcess.started[0]["OLLAMA_NUM_PARALLEL"] == str(num_parallel)


def test_the_server_refuses_vulkan_unless_the_caller_asks_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OLLAMA_VULKAN is 0 by default; a value in the caller's environment wins."""
    from contextlib import nullcontext

    from senselab.text.tasks.decision_model import ollama

    _FakeProcess.started = []
    monkeypatch.setattr(ollama, "verify_pin", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(ollama.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(ollama.urllib.request, "urlopen", lambda *a, **kw: nullcontext())
    pin = OllamaPin(name="clef", tag="27b", blob_digest="b", config_digest="c", manifest_digest="m")
    monkeypatch.delenv("OLLAMA_VULKAN", raising=False)
    with ollama.OllamaServer(tmp_path / "ollama", tmp_path, pin, require_gpu=False):
        pass
    monkeypatch.setenv("OLLAMA_VULKAN", "1")
    with ollama.OllamaServer(tmp_path / "ollama", tmp_path, pin, require_gpu=False):
        pass
    assert [env["OLLAMA_VULKAN"] for env in _FakeProcess.started] == ["0", "1"]


class _FakeOllama(BaseHTTPRequestHandler):
    """``ollama serve``'s version, load and process-list routes, answering as the test sets them."""

    load_status = 200
    loaded: list[dict[str, Any]] = []
    loads: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ANN401
        pass

    def _send(self, status: int, payload: Any) -> None:  # noqa: ANN401
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/version":
            self._send(200, {"version": "0.35.0"})
        elif self.path == "/api/ps":
            self._send(200, {"models": type(self).loaded})
        else:
            self._send(404, {})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        type(self).loads.append({"path": self.path, **json.loads(self.rfile.read(length))})
        status = type(self).load_status
        self._send(status, {} if status == 200 else {"error": "llama-server process has terminated"})


@pytest.fixture
def fake_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A served fake, with the server's process, port and pin check stood in for."""
    from senselab.text.tasks.decision_model import ollama

    _FakeOllama.load_status, _FakeOllama.loaded, _FakeOllama.loads = 200, [], []
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(ollama, "verify_pin", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(ollama.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(ollama, "_free_port", lambda: server.server_address[1])
    try:
        yield ollama
    finally:
        server.shutdown()


def _resident(size: int, vram: int) -> list[dict[str, Any]]:
    return [{"name": "clef:27b", "model": "clef:27b", "size": size, "size_vram": vram}]


def _served_pin() -> OllamaPin:
    return OllamaPin(name="clef", tag="27b", blob_digest="b", config_digest="c", manifest_digest="m")


def test_a_model_wholly_on_the_gpu_is_served(fake_ollama: Any, tmp_path: Path) -> None:  # noqa: ANN401
    """The model is loaded on entry by one decision request, with the server's keep-alive."""
    _FakeOllama.loaded = _resident(10, 10)
    with fake_ollama.OllamaServer(tmp_path / "ollama", tmp_path, _served_pin()) as server:
        assert server.model == "clef:27b"
    assert _FakeOllama.loads == [
        {"path": "/v1/systemone", "model": "clef:27b", **fake_ollama.WARM_UP, "keep_alive": "60m"}
    ]


def test_a_model_that_does_not_load_is_refused_with_the_log_tail(
    fake_ollama: Any,  # noqa: ANN401
    tmp_path: Path,
) -> None:
    """A failed load names the server log's last line."""
    _FakeOllama.load_status = 500
    log = tmp_path / "ollama.log"
    log.write_text("starting\nerror loading model: vk::PhysicalDevice::createDevice: ErrorInitializationFailed\n")
    with pytest.raises(fake_ollama.ServerUnusableError, match="ErrorInitializationFailed"):
        with fake_ollama.OllamaServer(tmp_path / "ollama", tmp_path, _served_pin(), log_path=log, discovery_wait_s=0):
            pass


class _DiscoveringProcess(_FakeProcess):
    """A fake ``ollama serve`` whose log reports the devices its discovery settled on."""

    lines: tuple[str, ...] = ()

    def __init__(self, args: list[str], *, env: dict[str, str], **kw: Any) -> None:  # noqa: ANN401
        super().__init__(args, env=env, **kw)
        handle = kw.get("stdout")
        assert handle is not None
        for line in type(self).lines:
            handle.write((line + "\n").encode("utf-8"))


_CPU_FALLBACK = (
    'time=2026-10-04T10:00:30 level=WARN source=runner.go:584 msg="llama-server GPU discovery watchdog timed out"',
    'time=2026-10-04T10:00:31 level=INFO source=types.go:130 msg="inference compute" id=cpu library=cpu '
    'compute="" name=cpu description=cpu total="1007.7 GiB" available="950.1 GiB"',
)
_CUDA_FOUND = (
    'time=2026-10-04T10:00:02 level=INFO source=types.go:130 msg="inference compute" id=GPU-1a2b library=CUDA '
    'compute=9.0 name=CUDA0 description="NVIDIA H100 80GB HBM3" total="79.2 GiB" available="78.6 GiB"',
)


@pytest.mark.parametrize(("lines", "refused"), [(_CPU_FALLBACK, True), (_CUDA_FOUND, False)])
def test_a_discovery_that_fell_back_to_the_cpu_is_refused_before_the_load(
    fake_ollama: Any,  # noqa: ANN401
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lines: tuple[str, ...],
    refused: bool,
) -> None:
    """2026-10-04, an H100 node: the watchdog timed out and discovery settled on the CPU; refuse at once."""
    _DiscoveringProcess.lines = lines
    monkeypatch.setattr(fake_ollama.subprocess, "Popen", _DiscoveringProcess)
    _FakeOllama.loaded = _resident(10, 10)
    log = tmp_path / "ollama.log"
    log.write_text('msg="inference compute" id=GPU-old library=CUDA\n')
    server = fake_ollama.OllamaServer(tmp_path / "ollama", tmp_path, _served_pin(), log_path=log, discovery_wait_s=5)
    if refused:
        with pytest.raises(fake_ollama.ServerUnusableError, match="fell back to the CPU"):
            with server:
                pass
        assert _FakeOllama.loads == [], "no load is attempted"
    else:
        with server:
            pass
        assert server.discovered_libraries() == ["cuda"]


@pytest.mark.parametrize(
    "loaded, reason",
    [([], "is not loaded"), (_resident(10, 6), "size_vram 6 of size 10"), (_resident(0, 0), "size_vram 0 of size 0")],
)
def test_a_model_not_wholly_on_the_gpu_is_refused(
    fake_ollama: Any,  # noqa: ANN401
    tmp_path: Path,
    loaded: list[dict[str, Any]],
    reason: str,
) -> None:
    """Partly offloaded to the CPU, or absent from the process list: not served."""
    _FakeOllama.loaded = loaded
    with pytest.raises(fake_ollama.ServerUnusableError, match=reason):
        with fake_ollama.OllamaServer(tmp_path / "ollama", tmp_path, _served_pin()):
            pass


def test_without_require_gpu_nothing_is_loaded_on_entry(fake_ollama: Any, tmp_path: Path) -> None:  # noqa: ANN401
    """The check is the caller's to waive."""
    with fake_ollama.OllamaServer(tmp_path / "ollama", tmp_path, _served_pin(), require_gpu=False):
        pass
    assert _FakeOllama.loads == []
