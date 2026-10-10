"""The reviewer on a vLLM server: the engine's identity, the cache key, and the worker against a mocked server."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes import review as review_node
from senselab.text.tasks.pii_detection import redaction_review as rr
from senselab.text.tasks.pii_detection import redaction_review_vllm as rv

SHA = "c" * 40

_ANSWER = [
    "REASONING:",
    "fine.\nREDACTION:",
    "not_applicable\nORIGINAL:",
    "clean\nSPEAKERS:",
    "one\nPHRASES_INSTEAD_OF_ITEMS:",
    "occasional\nPHRASE_QUOTES:",
    '["hello"]\nCONDITIONS:',
    "[]\nINSTRUCTIONS_SPOKEN:",
    "[]\nPROPOSAL:",
    "[]",
]

_FAKE_SERVER = r'''
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
assert argv[0] == "serve"
model_path = argv[1]
opts = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1) if argv[i].startswith("--") and not argv[i + 1].startswith("--")}
port = int(opts["--port"])
log = os.environ["FAKE_VLLM_LOG"]
answer = json.loads(os.environ["FAKE_VLLM_ANSWER"])
with open(log + ".pid", "w") as handle:
    handle.write(str(os.getpid()))
with open(log + ".argv", "w") as handle:
    json.dump(argv, handle)
state = {"inflight": 0, "most": 0}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.reply({})
        if self.path == "/version":
            return self.reply({"version": os.environ.get("FAKE_VLLM_VERSION", "0.31.0")})
        if self.path == "/v1/models":
            return self.reply({"data": [{"id": opts["--served-model-name"], "root": model_path, "max_model_len": int(opts["--max-model-len"])}]})
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            state["inflight"] += 1
            state["most"] = max(state["most"], state["inflight"])
        with open(log, "a") as handle:
            handle.write(json.dumps({**body, "most_inflight": state["most"]}) + "\n")
        self.reply(
            {
                "choices": [{"text": "unused", "token_ids": answer, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(answer)},
            }
        )
        with lock:
            state["inflight"] -= 1


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def _tokenizer(model_dir: Path) -> list[int]:
    """A word-level tokenizer whose vocabulary spells the canned answer; returns the answer's ids plus eos."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for token in _ANSWER:
        vocab.setdefault(token, len(vocab))
    words = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    words.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=words, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>")
    fast.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    fast.save_pretrained(str(model_dir))
    (model_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 0]}))
    return [vocab[token] for token in _ANSWER] + [1]


def _executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def mocked_vllm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """A fake vLLM venv: the host interpreter as its python, and a stdlib HTTP server as its ``vllm``."""
    model_dir = tmp_path / "snapshots" / SHA
    model_dir.mkdir(parents=True)
    answer = _tokenizer(model_dir)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _executable(venv / "bin" / "python", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (tmp_path / "fake_vllm.py").write_text(_FAKE_SERVER)
    _executable(venv / "bin" / "vllm", f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake_vllm.py"}" "$@"\n')
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_VLLM_LOG", str(log))
    monkeypatch.setenv("FAKE_VLLM_ANSWER", json.dumps(answer))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(rv, "ensure_venv", lambda *a, **k: venv)
    monkeypatch.setattr(rv, "hf_subprocess_env", lambda *a, **k: dict(os.environ))
    monkeypatch.setattr(rr, "_staged_snapshot", lambda repo, revision: str(model_dir))
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: SHA)
    rr.shutdown_review_worker()
    yield {"log": log, "answer": answer, "model_dir": model_dir}
    rr.shutdown_review_worker()


def _settings() -> dict[str, Any]:
    return dict(load_triage_config().require("redaction.llm_check.vllm"))


class TestTheEngineIdentity:
    """The engine is named in full, and the cache key changes with every part of it."""

    def test_the_vllm_identity_carries_the_version_the_arguments_and_one_request_at_a_time(self) -> None:
        """The packaged arguments are the benchmark's, in a fixed order."""
        identity = rv.engine_identity("vllm", _settings())
        assert identity["name"] == "vllm" and identity["version"] == "0.31.0" and identity["concurrency"] == 1
        assert identity["server_args"] == [
            "--max-model-len",
            "12288",
            "--gpu-memory-utilization",
            "0.9",
            "--kv-cache-dtype",
            "auto",
            "--generation-config",
            "vllm",
            "--limit-mm-per-prompt",
            '{"audio":0,"image":0}',
            "--enable-prefix-caching",
        ]
        assert rv.engine_identity("transformers") == {"name": "transformers", "concurrency": 1}
        with pytest.raises(ValueError):
            rv.engine_identity("sglang")

    def test_the_cache_key_names_the_engine(self) -> None:
        """Same engine, same key; another engine or any other server argument, another key."""
        base = {
            "model_id": "m/x",
            "max_new_tokens": 1024,
            "max_iterations": 3,
            "engine": "vllm",
            "vllm": _settings(),
        }
        key = review_node.review_cache_key("a b", None, {"task": "t"}, base, SHA)
        assert key == review_node.review_cache_key("a b", None, {"task": "t"}, dict(base), SHA)
        assert key != review_node.review_cache_key("a b", None, {"task": "t"}, {**base, "engine": "transformers"}, SHA)
        shared = {**base, "vllm": {**_settings(), "gpu_memory_utilization": 0.3}}
        assert key != review_node.review_cache_key("a b", None, {"task": "t"}, shared, SHA)


class TestTheWorkerAgainstAMockedServer:
    """The worker templates token ids, asks one request at a time, decodes the ids and stops its server."""

    def test_a_review_round_trips_and_records_the_engine(self, mocked_vllm: dict[str, Any]) -> None:
        """Greedy, the checkpoint's stop ids, the caller's ceiling; the answer parses; the engine is recorded."""
        engine = rv.engine_identity("vllm", _settings())
        first = rr.review_transcript("hello", context={"item_set": True}, engine=engine, max_new_tokens=64)
        assert first.available, first.failure
        assert first.phrases_instead_of_items == "occasional" and first.phrase_quotes == ["hello"]
        assert first.engine == engine and first.revision == SHA and first.load_s > 0
        assert rr.review_payload(first)["engine"] == engine
        second = rr.review_transcript("hello", context={"item_set": True}, engine=engine, max_new_tokens=64)
        assert second.available and second.load_s == 0.0 and second.reasoning == first.reasoning
        sent = [json.loads(line) for line in mocked_vllm["log"].read_text().splitlines()]
        assert len(sent) == 2 and sent[0]["prompt"] == sent[1]["prompt"]
        assert all(isinstance(token, int) for token in sent[0]["prompt"])
        assert sent[0]["temperature"] == 0.0 and sent[0]["max_tokens"] == 64
        assert sent[0]["stop_token_ids"] == [1, 0] and sent[0]["return_token_ids"] is True
        assert max(request["most_inflight"] for request in sent) == 1
        argv = json.loads(Path(str(mocked_vllm["log"]) + ".argv").read_text())
        assert argv[argv.index("--max-model-len") + 1] == "12288" and "--enable-prefix-caching" in argv

    def test_an_over_long_request_fails_alone_and_the_server_stops_with_the_worker(
        self, mocked_vllm: dict[str, Any]
    ) -> None:
        """A request past max_model_len is an absent reading; the next one is served; shutdown ends the server."""
        engine = rv.engine_identity("vllm", {**_settings(), "max_model_len": 4000})
        too_long = rr.review_transcript("hello", engine=engine, max_new_tokens=3000)
        assert not too_long.available and "max_model_len" in str(too_long.failure)
        fits = rr.review_transcript("hello", engine=engine, max_new_tokens=64)
        assert fits.available and fits.load_s == 0.0, fits.failure
        pid = int(Path(str(mocked_vllm["log"]) + ".pid").read_text())
        rr.shutdown_review_worker()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            pytest.fail("the fake vLLM server outlived its worker")

    def test_a_server_of_another_version_is_refused(
        self, mocked_vllm: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reading would name an engine it was not served by; the worker is not used."""
        monkeypatch.setenv("FAKE_VLLM_VERSION", "0.30.0")
        result = rr.review_transcript("hello", engine=rv.engine_identity("vllm", _settings()), max_new_tokens=64)
        assert not result.available and "version" in str(result.failure)
