"""The redaction reviewer served by a pinned vLLM server, one request at a time.

The worker is a subprocess of :data:`VLLM_VENV`, an isolated venv :func:`~senselab.utils.subprocess_venv.ensure_venv`
builds from its committed lock. It loads the checkpoint's tokenizer, starts one ``vllm serve`` over the staged
snapshot, and answers the same line protocol as the transformers worker in
:mod:`~senselab.text.tasks.pii_detection.redaction_review`: each request is templated into token ids, sent to the
server's OpenAI-compatible completions endpoint with greedy decoding, the checkpoint's own stop ids and the
caller's ``max_new_tokens``, and the returned token ids are decoded the way the transformers worker decodes its
own. Only one request is in flight per server.

Servers sharing one GPU start one at a time, under a lock file named by the host and the visible devices.

The engine's identity -- the vLLM version, the server arguments and ``concurrency`` 1 -- is in every reading's
cache key and provenance (:func:`engine_identity`). The design is in
``specs/20261010-vllm-reviewer/design.md``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, venv_python

VLLM_VENV = "pii-review-vllm"
VLLM_PYTHON = "3.12"
VLLM_VERSION = "0.31.0"
VLLM_REQUIREMENTS = [f"vllm=={VLLM_VERSION}"]
VLLM_MAX_CUDA = (13, 0)

ENGINE_TRANSFORMERS = "transformers"
ENGINE_VLLM = "vllm"
ENGINES = (ENGINE_TRANSFORMERS, ENGINE_VLLM)
"""The reviewer's engines: the in-process transformers worker, or a vLLM server."""

CONCURRENCY = 1
"""Requests in flight per server; the engine's reproducibility holds only at one."""

SERVED_MODEL_NAME = "reviewer"

SERVER_FLAGS = (
    ("max_model_len", "--max-model-len"),
    ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ("kv_cache_dtype", "--kv-cache-dtype"),
    ("generation_config", "--generation-config"),
    ("limit_mm_per_prompt", "--limit-mm-per-prompt"),
)
"""The ``redaction.llm_check.vllm`` keys passed as valued server flags, in order."""

SWITCHES = (("enable_prefix_caching", "--enable-prefix-caching", "--no-enable-prefix-caching"),)
"""The keys passed as on/off server switches."""


def server_args(settings: Mapping[str, Any]) -> list[str]:
    """The ``vllm serve`` arguments a configuration names, after the model path.

    Args:
        settings: The ``redaction.llm_check.vllm`` mapping.

    Returns:
        The arguments, in a fixed order; a mapping value is written as compact JSON with sorted keys.

    Raises:
        KeyError: If a key is missing.
    """
    args: list[str] = []
    for key, flag in SERVER_FLAGS:
        value = settings[key]
        if isinstance(value, Mapping):
            value = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        args += [flag, str(value)]
    for key, on, off in SWITCHES:
        args.append(on if settings[key] else off)
    return args


def engine_identity(engine: str, vllm_settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What a reading's engine is, as its cache key and provenance carry it.

    Args:
        engine: One of :data:`ENGINES`.
        vllm_settings: The ``redaction.llm_check.vllm`` mapping, read for the vLLM engine.

    Returns:
        ``name`` and ``concurrency``; for vLLM also ``version`` and ``server_args``.

    Raises:
        ValueError: On an unknown engine.
    """
    if engine == ENGINE_TRANSFORMERS:
        return {"name": ENGINE_TRANSFORMERS, "concurrency": CONCURRENCY}
    if engine == ENGINE_VLLM:
        return {
            "name": ENGINE_VLLM,
            "version": VLLM_VERSION,
            "server_args": server_args(vllm_settings or {}),
            "concurrency": CONCURRENCY,
        }
    raise ValueError(f"unknown reviewer engine {engine!r}; known: {ENGINES}")


def _start_lock_path() -> str:
    """The node-local lock file serialising server start-up on this host's visible GPUs."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "-") or "none"
    return str(Path(tempfile.gettempdir()) / f"senselab-vllm-start-{socket.gethostname()}-{devices}.lock")


SERVER_LOG_DIR_ENV = "SENSELAB_VLLM_LOG_DIR"
"""Where each server's own output is written (mode 600), where set; otherwise it joins the worker's stderr."""


def _server_log_path() -> str | None:
    """The server's log file under :data:`SERVER_LOG_DIR_ENV`, named by host and process, or None."""
    directory = os.environ.get(SERVER_LOG_DIR_ENV)
    if not directory:
        return None
    Path(directory).mkdir(parents=True, exist_ok=True)
    return str(Path(directory) / f"vllm-{socket.gethostname()}-{os.getpid()}.log")


# Load payload (stdin), sent once:
#   {"model_id": str, "model_path": str, "revision": str, "vllm": str, "server_args": [str],
#    "served_model_name": str, "start_lock": str, "startup_timeout_s": int, "server_log": str | None}
# Load reply: {"ready": True, "revision": str, "load_s": float, "engine": {...}}
# Review request and reply: as the transformers worker's. A request the server cannot take (a prompt that
# with its max_new_tokens exceeds max_model_len) is answered {"error": {...}, "recoverable": True} and the
# worker carries on; any other error ends it, and the server with it.
_VLLM_WORKER_SCRIPT = r"""
import fcntl, importlib.metadata, json, os, signal, socket, subprocess, sys, time, urllib.error, urllib.request

MARKER = "%s"
_replies = sys.stdout
sys.stdout = sys.stderr
SERVER = None


def emit(payload):
    _replies.write(MARKER + json.dumps(payload) + "\n")
    _replies.flush()


def stop_server():
    global SERVER
    server, SERVER = SERVER, None
    if server is None or server.poll() is not None:
        return
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=60)
    except Exception:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except Exception:
            pass


def on_term(signum, frame):
    stop_server()
    sys.exit(128 + signum)


def die_with_parent():
    os.setsid()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)
    except Exception:
        pass


def call(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def healthy(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def free_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def start(args):
    global SERVER
    from transformers import AutoTokenizer

    model_path = args["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generation = json.load(open(os.path.join(model_path, "generation_config.json")))
    eos = generation.get("eos_token_id")
    stop_ids = [int(token) for token in (eos if isinstance(eos, list) else [eos]) if token is not None]
    deadline = time.monotonic() + float(args["startup_timeout_s"])
    with open(args["start_lock"], "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        port = free_port()
        base = "http://127.0.0.1:%%d" %% port
        command = [
            args["vllm"], "serve", model_path, "--served-model-name", args["served_model_name"],
            "--host", "127.0.0.1", "--port", str(port), "--disable-uvicorn-access-log", *args["server_args"],
        ]
        sink = sys.stderr
        if args.get("server_log"):
            sink = os.fdopen(os.open(args["server_log"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "w")
        SERVER = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.STDOUT, preexec_fn=die_with_parent
        )
        while not healthy(base):
            if SERVER.poll() is not None:
                raise RuntimeError("vllm serve exited with %%s before it was healthy" %% SERVER.returncode)
            if time.monotonic() > deadline:
                raise TimeoutError("vllm serve was not healthy within %%ss" %% args["startup_timeout_s"])
            time.sleep(2)
    version = (call(base + "/version") or {}).get("version")
    model = ((call(base + "/v1/models") or {}).get("data") or [{}])[0]
    root = os.path.basename(os.path.normpath(str(model.get("root") or model_path)))
    engine = {
        "name": "vllm",
        "version": version or importlib.metadata.version("vllm"),
        "server_args": list(args["server_args"]),
        "concurrency": 1,
        "max_model_len": model.get("max_model_len"),
    }
    return tokenizer, stop_ids, base, engine, root


def review(state, request):
    tokenizer, stop_ids, base, engine, _root = state
    messages = [{"role": "user", "content": request["prompt"] + request["text"]}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
    ids = [int(token) for token in ids["input_ids"]]
    limit = engine.get("max_model_len")
    wanted = int(request["max_new_tokens"])
    if limit and len(ids) + wanted > int(limit):
        return {
            "error": {
                "type": "ContextLengthError",
                "message": "prompt of %%d tokens plus max_new_tokens %%d exceeds max_model_len %%d" %% (len(ids), wanted, limit),
            },
            "recoverable": True,
        }
    body = {
        "model": request["served_model_name"],
        "prompt": ids,
        "max_tokens": wanted,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_tokens": 0,
        "repetition_penalty": 1.0,
        "stop_token_ids": stop_ids,
        "skip_special_tokens": True,
        "return_token_ids": True,
        "stream": False,
    }
    began = time.monotonic()
    answer = call(base + "/v1/completions", body, timeout=float(request["timeout_s"]))
    choice = (answer.get("choices") or [{}])[0]
    usage = answer.get("usage") or {}
    if usage.get("prompt_tokens") is not None and int(usage["prompt_tokens"]) != len(ids):
        raise RuntimeError("the server read %%s prompt tokens, not %%d" %% (usage["prompt_tokens"], len(ids)))
    tokens = choice.get("token_ids")
    completion = tokenizer.decode(tokens, skip_special_tokens=True) if tokens is not None else str(choice.get("text") or "")
    return {
        "completion": completion,
        "generate_s": round(time.monotonic() - began, 3),
        "input_tokens": len(ids),
        "output_tokens": int(usage.get("completion_tokens") or (len(tokens) if tokens is not None else 0)),
        "finish_reason": choice.get("finish_reason"),
        "peak_reserved_mib": 0,
        "resident_mib": 0,
    }


def main():
    signal.signal(signal.SIGTERM, on_term)
    started = time.monotonic()
    args = json.loads(sys.stdin.readline())
    state = start(args)
    emit({"ready": True, "revision": state[4], "load_s": round(time.monotonic() - started, 3), "engine": state[3]})
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("stop"):
            return
        request.setdefault("served_model_name", args["served_model_name"])
        request.setdefault("timeout_s", 1800)
        emit(review(state, request))


try:
    main()
except Exception as exc:
    emit({"error": {"type": type(exc).__name__, "message": str(exc)}})
    stop_server()
    sys.exit(1)
finally:
    stop_server()
"""


def worker_script(marker: str) -> str:
    """The vLLM worker's source, with the reply marker filled in.

    Args:
        marker: The reply-line prefix the host reads.

    Returns:
        The script.
    """
    return _VLLM_WORKER_SCRIPT % marker


def vllm_executable(venv_dir: Path) -> str:
    """The ``vllm`` console script of a venv.

    Args:
        venv_dir: The venv.

    Returns:
        Its path, beside the venv's interpreter.
    """
    return str(Path(venv_python(venv_dir)).parent / "vllm")


def build_worker_env(model_id: str, revision: str, venv_dir: Path) -> dict[str, str]:
    """The environment a vLLM worker runs in: offline over the staged commit, its venv's tools first on PATH.

    Args:
        model_id: The HuggingFace repo.
        revision: The resolved 40-hex commit.
        venv_dir: The vLLM venv.

    Returns:
        The environment.
    """
    env = hf_subprocess_env(model_id, revision, base_env=_clean_subprocess_env())
    env["PATH"] = os.pathsep.join([str(Path(venv_python(venv_dir)).parent), env.get("PATH", "")])
    return env


def start_vllm_worker(worker: Any, timeout_s: int) -> None:  # noqa: ANN401 — the review module's worker type
    """Start a :class:`~senselab.text.tasks.pii_detection.redaction_review._ReviewWorker` on the vLLM engine.

    Builds the venv from its lock, stages the commit, spawns the worker and waits for its server.

    Args:
        worker: The worker, carrying ``model_id``, ``revision`` and the expected ``engine`` identity.
        timeout_s: Wall-clock ceiling on start-up, the server's included.

    Raises:
        RuntimeError: Through the worker's own error type, if the commit could not be staged, the worker
            did not report ready, or the server it reports is not the engine expected.
    """
    from senselab.text.tasks.pii_detection import redaction_review as review
    from senselab.utils.model_revision import resolve_revision

    worker.revision = resolve_revision(worker.model_id, str(worker.revision))
    venv_dir = ensure_venv(VLLM_VENV, VLLM_REQUIREMENTS, python_version=VLLM_PYTHON, max_cuda_version=VLLM_MAX_CUDA)
    model_path = review._staged_snapshot(worker.model_id, str(worker.revision))
    if model_path is None:
        raise review.ReviewWorkerError(f"vllm reviewer: {worker.model_id}@{worker.revision} could not be staged")
    worker._process = subprocess.Popen(  # noqa: S603 — the interpreter is this repo's own venv
        [venv_python(venv_dir), "-c", worker_script(review._WORKER_MARKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=build_worker_env(worker.model_id, str(worker.revision), venv_dir),
    )
    threading.Thread(target=worker._pump_replies, daemon=True).start()
    threading.Thread(target=worker._pump_noise, daemon=True).start()
    began = time.monotonic()
    expected: Mapping[str, Any] = worker.engine
    worker._send(
        {
            "model_id": worker.model_id,
            "model_path": model_path,
            "revision": worker.revision,
            "vllm": vllm_executable(venv_dir),
            "server_args": list(expected["server_args"]),
            "served_model_name": SERVED_MODEL_NAME,
            "start_lock": _start_lock_path(),
            "startup_timeout_s": int(timeout_s),
            "server_log": _server_log_path(),
        }
    )
    reply = worker._await(timeout_s)
    live = dict(reply.get("engine") or {})
    loaded: Optional[str] = reply.get("revision")
    mismatch = [
        key for key in ("name", "version", "server_args", "concurrency") if live.get(key) != expected.get(key)
    ]
    if loaded != worker.revision:
        mismatch.append("revision")
    if mismatch:
        worker.close()
        raise review.ReviewWorkerError(
            f"vllm reviewer: the server is not the engine expected ({', '.join(mismatch)} differ: "
            f"{ {key: live.get(key) for key in mismatch if key != 'revision'} }, revision {loaded})"
        )
    worker.engine = {key: live[key] for key in ("name", "version", "server_args", "concurrency")}
    worker.engine_details = live
    worker.load_s = round(time.monotonic() - began, 3)
