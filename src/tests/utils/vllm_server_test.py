"""The vLLM server and its clients against a fake ``vllm``: lifecycle, args, logs and request shapes."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from senselab.utils import vllm_server as vs

SHA = "a" * 40
CANARY = "CANARY-7f3a"

# A stand-in for ``vllm serve``: /health, /version and /v1/models through http.server. It forks a grandchild
# that listens on a second port, prints one access line per request (never the body, unless asked to log
# requests) and records its start and ready times. Each POST is recorded in requests.jsonl (a JSON body as is; a
# multipart body as its form fields and, per file, its name, type, size and digest).
_FAKE = r"""
import hashlib, json, os, signal, sys, time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, HTTPServer

argv = sys.argv[1:]
assert argv[0] == "serve"
model_dir, flags = argv[1], argv[2:]
out = os.environ["FAKE_VLLM_DIR"]


def value(flag, default=None):
    return flags[flags.index(flag) + 1] if flag in flags else default


def event(name):
    with open(os.path.join(out, "events.log"), "a") as handle:
        handle.write("%s %d %.6f\n" % (name, os.getpid(), time.time()))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        print("%s %s" % (self.command, self.path), flush=True)

    def reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        broken = value("--fake-broken", "")
        if broken and self.path == broken.split(":")[0]:
            mode = broken.split(":")[1]
            if mode == "http500":
                return self.send_error(500)
            body = b"not json" if mode == "notjson" else b"[1, 2]"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if self.path == "/health":
            return self.reply({})
        if self.path == "/version":
            return self.reply({"version": value("--fake-version", "0.31.0")})
        if self.path == "/v1/models":
            return self.reply({"data": [{"id": model_dir, "root": value("--fake-root", model_dir)}]})
        self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if "--enable-log-requests" in flags:
            print(body.decode(), flush=True)
        kind = self.headers.get("Content-Type", "")
        record = {"path": self.path, "content_type": kind.split(";")[0]}
        if kind.startswith("multipart/form-data"):
            form = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + kind.encode() + b"\r\n\r\n" + body)
            record["form"], record["files"] = {}, {}
            for part in form.iter_parts():
                name = part.get_param("name", header="content-disposition")
                data = part.get_payload(decode=True)
                if part.get_filename() is None:
                    record["form"][name] = data.decode()
                else:
                    record["files"][name] = {
                        "filename": part.get_filename(),
                        "type": part.get_content_type(),
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
        else:
            record["json"] = json.loads(body)
        with open(os.path.join(out, "requests.jsonl"), "a") as handle:
            handle.write(json.dumps(record) + "\n")
        if self.path == "/v1/chat/completions":
            return self.reply({"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]})
        if self.path == "/v1/audio/transcriptions":
            return self.reply({"text": "ok"})
        self.reply({"choices": [{"index": 0, "text": "ok", "token_ids": [7, 8]}]})


event("start")
if "--fake-ignore-term" in flags:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if "--fake-nofit" in flags:
    print("ValueError: the KV cache cannot hold max_model_len tokens", file=sys.stderr, flush=True)
    sys.exit(1)
side = HTTPServer(("127.0.0.1", 0), Handler)
if os.fork() == 0:
    side.serve_forever()
    os._exit(0)
with open(os.path.join(out, "%d.json" % os.getpid()), "w") as handle:
    json.dump(
        {
            "pgid": os.getpgid(0),
            "ports": [int(value("--port")), side.server_port],
            "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        handle,
    )
side.server_close()
time.sleep(float(value("--fake-slow", "0")))
main = HTTPServer(("127.0.0.1", int(value("--port"))), Handler)
event("ready")
main.serve_forever()
"""

# Holds a server until it is killed.
_HOLDER = r"""
import sys, time
from senselab.utils.vllm_server import VllmServer

exe, model, locks = sys.argv[1:4]
with VllmServer(exe, model, server_args=[], devices=["0"], start_lock_dir=locks, expected_version="0.31.0"):
    print("up", flush=True)
    time.sleep(600)
"""

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A fake ``vllm`` executable, a snapshot dir named by its commit, and the fake's record dir."""
    out = tmp_path / "fake"
    out.mkdir()
    script = tmp_path / "fake_vllm.py"
    script.write_text(_FAKE)
    exe = tmp_path / "bin" / "vllm"
    exe.parent.mkdir()
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "hub" / "snapshots" / SHA
    model.mkdir(parents=True)
    monkeypatch.setenv("FAKE_VLLM_DIR", str(out))
    return {"exe": exe, "model": model, "out": out, "locks": tmp_path / "locks"}


def _server(fake: dict[str, Path], *args: str, **kwargs: Any) -> vs.VllmServer:  # noqa: ANN401
    return vs.VllmServer(
        fake["exe"],
        fake["model"],
        server_args=list(args),
        devices=["0"],
        start_lock_dir=fake["locks"],
        expected_version="0.31.0",
        **kwargs,
    )


def _infos(fake: dict[str, Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(fake["out"].glob("*.json"))]


def _refuses(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except OSError:
        return True
    return False


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _gone(info: dict[str, Any], timeout_s: float = 30.0) -> bool:
    """Whether the server's group has no process left and none of its ports accepts, within the timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _group_alive(info["pgid"]) and all(_refuses(port) for port in info["ports"]):
            return True
        time.sleep(0.2)
    return False


def test_two_servers_start_one_at_a_time_under_the_lock(fake: dict[str, Path]) -> None:
    """The second server starts only once the first is healthy, then both serve side by side."""
    both_up = threading.Barrier(2)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            with _server(fake, "--fake-slow", "1"):
                both_up.wait(timeout=60)
        except BaseException as exc:  # noqa: BLE001 — reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert errors == []
    events = [line.split() for line in (fake["out"] / "events.log").read_text().splitlines()]
    starts = {pid: float(at) for name, pid, at in events if name == "start"}
    readies = {pid: float(at) for name, pid, at in events if name == "ready"}
    first, second = sorted(starts, key=starts.__getitem__)
    assert starts[second] >= readies[first]
    assert (fake["locks"] / f"vllm-start-{socket.gethostname()}-0.lock").is_file()
    assert all(_gone(info) for info in _infos(fake))


def test_a_server_that_exits_before_health_is_exited(fake: dict[str, Path]) -> None:
    """A server that refuses its KV cache exits before it is healthy: the fit failure."""
    with pytest.raises(vs.VllmStartError) as caught:
        with _server(fake, "--fake-nofit"):
            pass
    assert caught.value.kind == "exited"


def test_a_slow_server_is_timeout(fake: dict[str, Path]) -> None:
    """A server not healthy within the start-up timeout is stopped, its whole group with it."""
    began = time.monotonic()
    with pytest.raises(vs.VllmStartError) as caught:
        with _server(fake, "--fake-slow", "60", startup_timeout_s=1):
            pass
    assert caught.value.kind == "timeout" and time.monotonic() - began < 30
    [info] = _infos(fake)
    assert _gone(info)


def test_a_version_mismatch_is_version(fake: dict[str, Path]) -> None:
    """A server of another version is refused; so is one serving a snapshot other than ``model_dir``'s commit."""
    cases = [
        (("--fake-version", "0.30.0"), "version"),
        (("--fake-root", str(fake["model"].parent / ("b" * 40))), "snapshot"),
    ]
    for args, kind in cases:
        with pytest.raises(vs.VllmStartError) as caught:
            with _server(fake, *args):
                pass
        assert caught.value.kind == kind
    assert all(_gone(info) for info in _infos(fake))


def _hold(fake: dict[str, Path], *, own_session: bool) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    """Start a process that holds a server; return it once the server is up, with the server's record."""
    src = Path(vs.__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])}
    holder = subprocess.Popen(  # noqa: S603 — this interpreter
        [sys.executable, "-c", _HOLDER, str(fake["exe"]), str(fake["model"]), str(fake["locks"])],
        stdout=subprocess.PIPE,
        env=env,
        start_new_session=own_session,
    )
    assert holder.stdout is not None
    for line in holder.stdout:
        if line.strip() == b"up":
            break
    else:
        holder.wait()
        pytest.fail("the owner exited before its server was up")
    [info] = _infos(fake)
    assert _group_alive(info["pgid"]) and not any(_refuses(port) for port in info["ports"])
    return holder, info


def test_the_server_dies_with_its_owner(fake: dict[str, Path]) -> None:
    """Killing the owner with SIGKILL takes the server and its grandchild down; both ports refuse."""
    holder, info = _hold(fake, own_session=False)
    holder.kill()
    holder.wait()
    assert _gone(info)


def test_the_server_dies_when_the_owners_group_is_killed(fake: dict[str, Path]) -> None:
    """A SIGKILL to the owner's whole process group does not take the watchdog with it; the server goes too."""
    holder, info = _hold(fake, own_session=True)
    try:
        os.killpg(holder.pid, signal.SIGKILL)
    finally:
        holder.wait()
    assert _gone(info)


def test_devices_set_the_visible_devices_only_when_given(
    fake: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit devices become the server's ``CUDA_VISIBLE_DEVICES``; without them the inherited value stays."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    host = socket.gethostname()
    for devices, seen, lock in [(["0", "1"], "0,1", "0-1"), (None, "3", "3")]:
        for path in fake["out"].glob("*.json"):
            path.unlink()
        server = vs.VllmServer(
            fake["exe"],
            fake["model"],
            server_args=[],
            devices=devices,
            start_lock_dir=fake["locks"],
            expected_version="0.31.0",
        )
        assert server.lock_path.name == f"vllm-start-{host}-{lock}.lock"
        with server:
            [info] = _infos(fake)
        assert info["devices"] == seen


def test_a_broken_version_or_models_probe_is_version(fake: dict[str, Path]) -> None:
    """An HTTP error, a body that is not JSON or JSON of the wrong shape on either probe is a ``version`` error."""
    for path in ("/version", "/v1/models"):
        for mode in ("http500", "notjson", "list"):
            with pytest.raises(vs.VllmStartError) as caught:
                with _server(fake, "--fake-broken", f"{path}:{mode}"):
                    pass
            assert caught.value.kind == "version", (path, mode, caught.value)
    assert all(_gone(info) for info in _infos(fake))


def test_a_server_that_ignores_sigterm_is_killed(fake: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """When the group outlives SIGTERM past the stop timeout, SIGKILL ends it."""
    monkeypatch.setattr(vs, "STOP_TIMEOUT_S", 1.0)
    with _server(fake, "--fake-ignore-term"):
        [info] = _infos(fake)
    assert _gone(info, timeout_s=5)


def test_server_args_are_ordered_and_identity_is_stable() -> None:
    """The argv order is fixed whatever the mapping order; the identity is equal for equal inputs."""
    settings = {
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"video": 1, "image": 4, "audio": 2},
        "generation_config": "vllm",
        "kv_cache_dtype": "auto",
        "gpu_memory_utilization": 0.45,
        "max_num_seqs": 8,
        "max_model_len": 12288,
    }
    args = vs.server_args(settings)
    assert args == [
        "--max-model-len",
        "12288",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.45",
        "--kv-cache-dtype",
        "auto",
        "--generation-config",
        "vllm",
        "--limit-mm-per-prompt",
        '{"audio":2,"image":4,"video":1}',
        "--no-enable-prefix-caching",
    ]
    assert vs.server_args({**settings, "enable_prefix_caching": True})[-1] == "--enable-prefix-caching"
    with pytest.raises(KeyError):
        vs.server_args({key: value for key, value in settings.items() if key != "max_num_seqs"})

    identity = vs.engine_identity("0.31.0", args, concurrency=8, batch_size=16, servers_per_card=2)
    assert identity == {
        "name": "vllm",
        "version": "0.31.0",
        "server_args": args,
        "concurrency": 8,
        "batch_size": 16,
        "servers_per_card": 2,
    }
    again = vs.engine_identity("0.31.0", list(args), concurrency=8, batch_size=16, servers_per_card=2)
    assert json.dumps(again, sort_keys=True) == json.dumps(identity, sort_keys=True)
    assert identity != vs.engine_identity("0.31.0", args, concurrency=4, batch_size=16, servers_per_card=2)
    args.append("--extra")
    assert "--extra" not in identity["server_args"]


def test_utilisation_rule() -> None:
    """``floor(90 / n) / 100`` of the card per server; at least one server."""
    for servers, expected in [(1, 0.9), (2, 0.45), (3, 0.3), (4, 0.22), (7, 0.12)]:
        assert vs.utilisation(servers) == expected
    with pytest.raises(ValueError):
        vs.utilisation(0)


def test_request_content_is_not_logged(fake: dict[str, Path], tmp_path: Path) -> None:
    """The server log is mode 600 and holds access lines, never a request body; request logging is refused."""
    logs = tmp_path / "logs"
    with _server(fake, log_dir=logs) as server:
        request = urllib.request.Request(
            f"{server.url}/v1/completions",
            data=json.dumps({"prompt": CANARY}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with _NO_PROXY.open(request, timeout=30) as response:
            assert response.status == 200
    [log] = list(logs.iterdir())
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    text = log.read_text()
    assert "POST /v1/completions" in text and CANARY not in text
    for flag in ("--enable-log-requests", "--enable-log-requests=true", "--enable-log-outputs"):
        with pytest.raises(ValueError):
            _server(fake, flag)


def _requests(fake: dict[str, Path]) -> list[dict[str, Any]]:
    path = fake["out"] / "requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _media(tmp_path: Path) -> dict[str, Path]:
    """Tiny local image, video and audio files in one directory."""
    media = tmp_path / "media"
    media.mkdir()
    files = {
        "image": (media / "frame.png", b"\x89PNG\r\n\x1a\n" + bytes(16)),
        "video": (media / "clip.mp4", b"\x00\x00\x00\x18ftypmp42" + bytes(16)),
        "audio": (media / "voice.wav", b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(16)),
    }
    for path, data in files.values():
        path.write_bytes(data)
    return {"dir": media, **{kind: path for kind, (path, _) in files.items()}}


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def test_chat_sends_image_video_and_audio_parts(fake: dict[str, Path], tmp_path: Path) -> None:
    """Text, image, video and audio parts reach the server in order: local files inline, or as file URLs."""
    media = _media(tmp_path)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Describe the clip."},
        {
            "role": "user",
            "content": [
                vs.content_part("text", "What happens?"),
                vs.content_part("image", media["image"]),
                vs.content_part("image", str(media["image"])),
                vs.content_part("video", media["video"]),
                vs.content_part("audio", media["audio"]),
            ],
        },
        {
            "role": "user",
            "content": [
                vs.content_part("video", media["video"], local_media_dir=media["dir"]),
                vs.content_part("image", "https://example.org/frame.png"),
            ],
        },
    ]
    with _server(fake) as server:
        assert server.url is not None
        reply = vs.chat(server.url, "m", messages, sampling={"temperature": 0.0, "max_tokens": 8, "seed": 3})
    assert reply["choices"][0]["message"]["content"] == "ok"
    [request] = _requests(fake)
    assert request["path"] == "/v1/chat/completions" and request["content_type"] == "application/json"
    body = request["json"]
    sent = body["messages"]
    shape = [[part["type"] for part in m["content"]] if isinstance(m["content"], list) else "text" for m in sent]
    assert shape == ["text", ["text", "image_url", "image_url", "video_url", "input_audio"], ["video_url", "image_url"]]
    parts, local = sent[1]["content"], sent[2]["content"]
    assert parts[0] == {"type": "text", "text": "What happens?"}
    image_url = f"data:image/png;base64,{_b64(media['image'])}"
    assert parts[1]["image_url"]["url"] == parts[2]["image_url"]["url"] == image_url
    assert parts[3]["video_url"]["url"] == f"data:video/mp4;base64,{_b64(media['video'])}"
    assert parts[4]["input_audio"] == {"data": _b64(media["audio"]), "format": "wav"}
    assert local[0]["video_url"]["url"] == media["video"].resolve().as_uri()
    assert local[1]["image_url"]["url"] == "https://example.org/frame.png"
    assert {key: body[key] for key in ("model", "temperature", "max_tokens", "seed", "stream")} == {
        "model": "m",
        "temperature": 0.0,
        "max_tokens": 8,
        "seed": 3,
        "stream": False,
    }
    elsewhere = tmp_path / "elsewhere.mp4"
    elsewhere.write_bytes(b"")
    refused = [
        lambda: vs.content_part("video", elsewhere, local_media_dir=media["dir"]),
        lambda: vs.content_part("audio", "https://example.org/voice.wav"),
        lambda: vs.content_part("image", tmp_path / "frame.unknownext"),
        lambda: vs.content_part("smell", "x"),
        lambda: vs.chat("http://127.0.0.1:9", "m", messages, sampling={"messages": []}),
    ]
    for call in refused:
        with pytest.raises(ValueError):
            call()
    assert len(_requests(fake)) == 1


def test_transcribe_posts_multipart_audio(fake: dict[str, Path], tmp_path: Path) -> None:
    """The audio file and the form fields go up as multipart; language and prompt only when given."""
    media = _media(tmp_path)
    with _server(fake) as server:
        assert server.url is not None
        reply = vs.transcribe(
            server.url,
            "m",
            media["audio"],
            language="en",
            prompt="A child and a parent.",
            sampling={"temperature": 0.2, "response_format": "json", "use_beam_search": False},
        )
        bare = vs.transcribe(server.url, "m", media["audio"], sampling={})
    assert reply == bare == {"text": "ok"}
    first, second = _requests(fake)
    assert first["path"] == second["path"] == "/v1/audio/transcriptions"
    assert first["content_type"] == "multipart/form-data"
    assert first["form"] == {
        "model": "m",
        "language": "en",
        "prompt": "A child and a parent.",
        "temperature": "0.2",
        "response_format": "json",
        "use_beam_search": "false",
    }
    assert second["form"] == {"model": "m"}
    [(name, upload)] = first["files"].items()
    assert name == "file" and upload["filename"] == "voice.wav" and upload["type"].startswith("audio/")
    data = media["audio"].read_bytes()
    assert upload["size"] == len(data) and upload["sha256"] == hashlib.sha256(data).hexdigest()
    for sampling in ({"file": "x"}, {"model": "x"}, {"language": "x"}, {"prompt": "x"}):
        with pytest.raises(ValueError):
            vs.transcribe("http://127.0.0.1:9", "m", media["audio"], sampling=sampling)
    with pytest.raises(TypeError):
        vs.transcribe("http://127.0.0.1:9", "m", media["audio"], sampling={"timestamp_granularities": ["word"]})
    assert len(_requests(fake)) == 2


def test_complete_sends_token_ids(fake: dict[str, Path]) -> None:
    """The prompt goes up as token ids with the sampling settings; the parsed reply comes back."""
    sampling = {"max_tokens": 4, "temperature": 0.0, "stop_token_ids": [2], "return_token_ids": True}
    with _server(fake) as server:
        assert server.url is not None
        reply = vs.complete(server.url, "m", [101, 7, 0], sampling=sampling)
    assert reply["choices"][0]["token_ids"] == [7, 8]
    [request] = _requests(fake)
    assert request["path"] == "/v1/completions"
    assert request["json"] == {"model": "m", "prompt": [101, 7, 0], **sampling, "stream": False}
    bad: list[list[Any]] = [[], [1, "2"], [1, -1], [True]]
    for ids in bad:
        with pytest.raises(ValueError):
            vs.complete("http://127.0.0.1:9", "m", ids, sampling={})
    with pytest.raises(ValueError):
        vs.complete("http://127.0.0.1:9", "m", [1], sampling={"prompt": "x"})


def test_clients_bypass_the_proxy_on_loopback(
    fake: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every proxy variable pointing at a closed port, all three clients still reach the loopback server."""
    media = _media(tmp_path)
    with _server(fake) as server:
        assert server.url is not None
        for variable in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.setenv(variable, "http://127.0.0.1:9")
        for no_proxy in ("127.0.0.1", None):
            for variable in ("no_proxy", "NO_PROXY"):
                if no_proxy is None:
                    monkeypatch.delenv(variable, raising=False)
                else:
                    monkeypatch.setenv(variable, no_proxy)
            text: list[dict[str, Any]] = [{"role": "user", "content": [vs.content_part("text", "hi")]}]
            assert vs.chat(server.url, "m", text, sampling={})["choices"]
            assert vs.transcribe(server.url, "m", media["audio"], sampling={}) == {"text": "ok"}
            assert vs.complete(server.url, "m", [1], sampling={})["choices"]
    paths = ["/v1/chat/completions", "/v1/audio/transcriptions", "/v1/completions"]
    assert [request["path"] for request in _requests(fake)] == paths * 2


def test_the_vllm_server_lock_is_committed_and_matches_its_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed lock was compiled from the registered requirements, and the venv is built from it."""
    from senselab.utils import subprocess_venv, venv_lock

    assert (vs.VLLM_VENV, vs.VLLM_REQUIREMENTS, vs.VLLM_PYTHON, vs.VLLM_MAX_CUDA) == (
        "vllm-server",
        ["vllm==0.31.0"],
        "3.12",
        (13, 0),
    )
    path = venv_lock.lock_path(vs.VLLM_VENV)
    lock = venv_lock.parse_lock(path)
    assert lock.requirements_sha256 == venv_lock.requirements_sha256(
        vs.VLLM_REQUIREMENTS, vs.VLLM_PYTHON, vs.VLLM_MAX_CUDA
    )
    assert lock.python == "3.12" and lock.header["torch-index"] == "cu130"
    assert lock.header["exclude-newer"] == "2026-10-10T00:00:00Z"
    assert any(line.startswith("vllm==0.31.0 ") for line in path.read_text().splitlines())
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def ensure_venv(*args: Any, **kwargs: Any) -> Path:  # noqa: ANN401
        calls.append((args, kwargs))
        return tmp_path / "venv"

    monkeypatch.setattr(subprocess_venv, "ensure_venv", ensure_venv)
    assert vs.ensure_vllm_venv() == tmp_path / "venv" / "bin" / "vllm"
    assert calls == [(("vllm-server", ["vllm==0.31.0"]), {"python_version": "3.12", "max_cuda_version": (13, 0)})]
