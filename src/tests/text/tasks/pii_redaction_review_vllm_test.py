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

from senselab.text.tasks.pii_detection import redaction_review as rr
from senselab.text.tasks.pii_detection import redaction_review_vllm as rv

pytestmark = pytest.mark.usefixtures("triage_profile")  # these tests run triage nodes on the fixture profile

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

_FAKE_SERVER = r"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
assert argv[0] == "serve"
model_path = argv[1]
opts = {
    argv[i]: argv[i + 1]
    for i in range(2, len(argv) - 1)
    if argv[i].startswith("--") and not argv[i + 1].startswith("--")
}
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
            model = {"id": opts["--served-model-name"], "root": model_path}
            return self.reply({"data": [{**model, "max_model_len": int(opts["--max-model-len"])}]})
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
"""


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


VLLM_SETTINGS = {
    "max_model_len": 12288,
    "gpu_memory_utilization": 0.90,
    "enable_prefix_caching": True,
    "kv_cache_dtype": "auto",
    "generation_config": "vllm",
    "limit_mm_per_prompt": {"image": 0, "audio": 0},
}
"""The vLLM server arguments under test: a caller's settings (triage's ``redaction.llm_check.vllm``), written out."""


def _settings() -> dict[str, Any]:
    return dict(VLLM_SETTINGS)


@pytest.fixture(scope="module")
def policy(tmp_path_factory: pytest.TempPathFactory) -> rr.ReviewPolicy:
    """A review policy of the test's own: the reviewer takes the caller's (plan step 6b), not a dataset's."""
    root = Path(tmp_path_factory.mktemp("policy"))
    (root / "review_policy.txt").write_text("TEST POLICY. Remove names.\n\n")
    (root / "task_guidance.yaml").write_text("families: []\nnatures: []\n")
    return rr.load_review_policy(root / "review_policy.txt", root / "task_guidance.yaml")


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


class TestTheWorkerAgainstAMockedServer:
    """The worker templates token ids, asks one request at a time, decodes the ids and stops its server."""

    def test_a_review_round_trips_and_records_the_engine(
        self, mocked_vllm: dict[str, Any], policy: rr.ReviewPolicy
    ) -> None:
        """Greedy, the checkpoint's stop ids, the caller's ceiling; the answer parses; the engine is recorded."""
        engine = rv.engine_identity("vllm", _settings())
        first = rr.review_transcript(
            "hello", policy=policy, context={"item_set": True}, engine=engine, max_new_tokens=64
        )
        assert first.available, first.failure
        assert first.phrases_instead_of_items == "occasional" and first.phrase_quotes == ["hello"]
        assert first.engine == engine and first.revision == SHA and first.load_s > 0
        assert rr.review_payload(first)["engine"] == engine
        second = rr.review_transcript(
            "hello", policy=policy, context={"item_set": True}, engine=engine, max_new_tokens=64
        )
        assert second.available and second.load_s == 0.0 and second.reasoning == first.reasoning
        sent = [json.loads(line) for line in mocked_vllm["log"].read_text().splitlines()]
        assert len(sent) == 2 and sent[0]["prompt"] == sent[1]["prompt"]
        assert all(isinstance(token, int) for token in sent[0]["prompt"])
        assert sent[0]["temperature"] == 0.0 and sent[0]["max_tokens"] == 64
        assert sent[0]["stop_token_ids"] == [1, 0] and sent[0]["return_token_ids"] is True
        assert max(request["most_inflight"] for request in sent) == 1
        argv = json.loads(Path(str(mocked_vllm["log"]) + ".argv").read_text())
        assert argv[argv.index("--max-model-len") + 1] == "12288" and "--enable-prefix-caching" in argv

    @pytest.mark.parametrize(
        ("mask", "order", "named"),
        [
            ("1", "PCI_BUS_ID", ["NVIDIA H100 80GB HBM3"]),
            ("1", None, None),
            ("GPU-0000aaaa-11", None, ["NVIDIA A100-SXM4-80GB"]),
            (None, None, ["NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"]),
        ],
        ids=["index-pci-order", "index-default-order", "uuid", "unset"],
    )
    def test_the_answer_records_the_vllm_version_and_the_machine(
        self,
        mocked_vllm: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        policy: rr.ReviewPolicy,
        mask: str | None,
        order: str | None,
        named: list[str] | None,
    ) -> None:
        """The vLLM that served it, and the devices the worker's CUDA_VISIBLE_DEVICES names, read off nvidia-smi.

        The worker beside the server touches no CUDA, so its devices come from its mask and one ``nvidia-smi``
        table, and its torch is the version installed where it runs, read without importing torch. On this node of
        two models an index is CUDA's only under ``CUDA_DEVICE_ORDER=PCI_BUS_ID``; otherwise it names nothing.
        """
        from importlib.metadata import version

        from senselab.utils.reproduce import host

        calls: list[Any] = []

        def _run(argv: Any) -> tuple[int, str]:  # noqa: ANN401
            calls.append(argv)
            assert argv[0] == "nvidia-smi"
            return (
                0,
                "0, GPU-0000aaaa-11, NVIDIA A100-SXM4-80GB, 570.86.15\n"
                "1, GPU-1111bbbb-22, NVIDIA H100 80GB HBM3, 570.86.15\n",
            )

        monkeypatch.setattr(host, "_run", _run)
        if mask is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        if order is None:
            monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        else:
            monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
        engine = rv.engine_identity("vllm", _settings())
        result = rr.review_transcript("hello", policy=policy, context={"item_set": True}, engine=engine)
        assert result.available, result.failure
        assert result.runtime["vllm"] == "0.31.0" == engine["version"]
        assert result.runtime["cuda_visible_devices"] == mask
        assert result.runtime["gpus"] == named
        assert result.runtime["torch"] == version("torch"), "the worker's python is this one, torch installed"
        assert result.runtime["node_gpus"] == ["NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"]
        assert result.runtime["nvidia_driver"] == "570.86.15"
        assert len(calls) == 1, "one nvidia-smi query per worker start"
        assert rr.review_payload(result)["runtime"] == result.runtime

    def test_an_over_long_request_fails_alone_and_the_server_stops_with_the_worker(
        self, mocked_vllm: dict[str, Any], policy: rr.ReviewPolicy
    ) -> None:
        """A request past max_model_len is an absent reading; the next one is served; shutdown ends the server."""
        engine = rv.engine_identity("vllm", {**_settings(), "max_model_len": 4000})
        too_long = rr.review_transcript("hello", policy=policy, engine=engine, max_new_tokens=3000)
        assert not too_long.available and "max_model_len" in str(too_long.failure)
        fits = rr.review_transcript("hello", policy=policy, engine=engine, max_new_tokens=64)
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
        self, mocked_vllm: dict[str, Any], monkeypatch: pytest.MonkeyPatch, policy: rr.ReviewPolicy
    ) -> None:
        """The reading would name an engine it was not served by; the worker is not used."""
        monkeypatch.setenv("FAKE_VLLM_VERSION", "0.30.0")
        result = rr.review_transcript(
            "hello", policy=policy, engine=rv.engine_identity("vllm", _settings()), max_new_tokens=64
        )
        assert not result.available and "version" in str(result.failure)
