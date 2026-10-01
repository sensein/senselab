"""The decision-model runner: a pinned store or nothing, and answers read back as probabilities.

``specs/20261001-nimble-second-opinion/design.md`` is the design.
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
    """An Ollama model store holding one manifest, ``nimble:9b``, its weights blob and a system layer."""
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
    path = root / "manifests" / "registry.ollama.ai" / "library" / "nimble" / "9b"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _manifest(root: Path) -> Path:
    return root / "manifests" / "registry.ollama.ai" / "library" / "nimble" / "9b"


def _pin(root: Path, **overrides: str) -> OllamaPin:
    """The pin of the store as written, with any digest replaced."""
    held = {
        "blob_digest": _digest(WEIGHTS),
        "config_digest": _digest(CONFIG),
        "manifest_digest": _digest(_manifest(root).read_bytes()),
        **overrides,
    }
    return OllamaPin(name="nimble", tag="9b", **held)


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
            verify_pin(tmp_path / "models", OllamaPin("nimble", "9b", PIN_DIGEST, PIN_DIGEST, PIN_DIGEST))

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


def _answers(other: float = 0.02, instructions: float = 0.03, diagnosis: float = 0.01, harbor: float = 0.05) -> dict:
    return {
        "other_voice": {
            "choice": MORE_THAN_ONE if other > 0.5 else "one",
            "probabilities": {"one": 1 - other, MORE_THAN_ONE: other, "unclear": 0.0},
        },
        "instructions_spoken": {"noul": instructions},
        "named_diagnosis": {"noul": diagnosis},
        "safe_harbor_identifier_present": {"noul": harbor},
    }


class TestTheAnswers:
    """Each answer is a probability of yes; other_voice is the probability of more_than_one."""

    def test_every_question_is_read_as_a_probability(self) -> None:
        """The four probabilities, keyed by question."""
        opinion = read_answers(_answers(other=0.85, diagnosis=0.99))
        assert opinion.probabilities == {
            "other_voice": 0.85,
            "instructions_spoken": 0.03,
            "named_diagnosis": 0.99,
            "safe_harbor_identifier_present": 0.05,
        }
        assert opinion.other_voice_choice == MORE_THAN_ONE

    def test_an_unanswered_question_is_an_error_not_a_zero(self) -> None:
        """A missing answer must not read as a confident no."""
        answers = _answers()
        del answers["named_diagnosis"]
        with pytest.raises(ValueError, match="named_diagnosis"):
            read_answers(answers)

    def test_a_choice_without_class_probabilities_is_an_error(self) -> None:
        """The choice alone carries no confidence."""
        answers = _answers()
        answers["other_voice"] = {"choice": "one"}
        with pytest.raises(ValueError, match="class probabilities"):
            read_answers(answers)

    def test_the_state_separates_the_instructions_from_the_stimulus(self) -> None:
        """The stimulus the participant is asked to say is not the instructions."""
        state = opinion_state(
            "the rainbow passage",
            {"task": "reading", "instructions": "Read aloud.", "asked_to_say": "When the sunlight"},
        )
        assert state == {
            "task": "reading",
            "speech_type": "",
            "instructions": "Read aloud.",
            "stimulus": "When the sunlight",
            "transcript": "the rainbow passage",
        }

    def test_the_question_set_is_sent_whole(self) -> None:
        """Ask receives the state and every question."""
        seen: list[Any] = []

        def ask(state: Any, questions: Any) -> dict:  # noqa: ANN401
            seen.append((state, questions))
            return _answers()

        ask_second_opinion(ask, "hello", {"task": "free-speech"})
        assert seen[0][1] is QUESTIONS and seen[0][0]["transcript"] == "hello"


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
    answers = ask_decisions(recorder, "nimble:9b", {"transcript": "hi"}, QUESTIONS, seed=7)
    assert set(answers) == set(QUESTIONS)
    body = _Recorder.bodies[0]
    assert body["path"] == "/v1/systemone"
    assert body["model"] == "nimble:9b"
    assert body["options"] == {"temperature": 0, "seed": 7}
