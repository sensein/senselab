"""SENSELAB_BLOCK_NETWORK: a worker started from the shared env builders cannot reach the network."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from senselab.utils import network_guard
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env

_PROBE = r"""
import json, os, socket, sys
out = {"guard": bool(getattr(socket, "senselab_network_guard", False))}
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(("1.1.1.1", 443))
    out["connect"] = "connected"
except Exception as e:
    out["connect"] = type(e).__name__ + ": " + str(e)
finally:
    s.close()
try:
    socket.getaddrinfo("huggingface.co", 443)
    out["lookup"] = "resolved"
except Exception as e:
    out["lookup"] = type(e).__name__ + ": " + str(e)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 0))
server.listen(1)
client = socket.create_connection(server.getsockname(), timeout=2)
out["loopback"] = "ok"
client.close(); server.close()
a, b = socket.socketpair(socket.AF_UNIX)
a.sendall(b"x"); out["unix"] = b.recv(1).decode()
a.close(); b.close()
keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OTEL_SDK_DISABLED", "PYANNOTE_METRICS_ENABLED")
keys += ("HF_HUB_DISABLE_TELEMETRY",)
out["env"] = {k: os.environ.get(k) for k in keys}
print(json.dumps(out))
"""


def _run(env: dict) -> dict:
    result = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture
def blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request blocking in this process's environment."""
    monkeypatch.setenv(network_guard.BLOCK_NETWORK_ENV, "1")


def test_worker_refuses_non_loopback_connect_and_lookup(blocked: None) -> None:
    """The guard is installed before ``-c`` code runs: 1.1.1.1:443 and DNS refused, loopback and AF_UNIX open."""
    out = _run(_clean_subprocess_env())
    assert out["guard"] is True
    assert out["connect"].startswith("PermissionError: SENSELAB_BLOCK_NETWORK"), out["connect"]
    assert "SENSELAB_BLOCK_NETWORK" in out["lookup"] and out["lookup"].startswith("gaierror"), out["lookup"]
    assert out["loopback"] == "ok"
    assert out["unix"] == "x"
    assert out["env"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OTEL_SDK_DISABLED": "true",
        "PYANNOTE_METRICS_ENABLED": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def test_hf_subprocess_env_carries_the_guard(blocked: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The HF worker env builder applies the guard too, even with a caller-supplied base env."""
    monkeypatch.setattr("senselab.utils.dependencies.resolve_model", lambda *a, **k: ("f" * 40, Path("/tmp")))
    env = hf_subprocess_env("org/model", "main", base_env={"PATH": os.environ.get("PATH", "")})
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(network_guard.GUARD_DIR)
    assert env["OTEL_SDK_DISABLED"] == "true"


def test_unset_flag_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the flag the env builders add no guard and the worker keeps its sockets."""
    monkeypatch.delenv(network_guard.BLOCK_NETWORK_ENV, raising=False)
    env = _clean_subprocess_env()
    assert str(network_guard.GUARD_DIR) not in env.get("PYTHONPATH", "")
    assert network_guard.BLOCK_NETWORK_ENV not in env
    probe = "import socket; print(getattr(socket, 'senselab_network_guard', False))"
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=60)
    assert result.stdout.strip() == "False"


def test_guard_dir_is_prepended_once() -> None:
    """Applying the guard twice leaves one PYTHONPATH entry."""
    env = {network_guard.BLOCK_NETWORK_ENV: "true", "PYTHONPATH": "/elsewhere"}
    network_guard.apply_network_guard(env)
    network_guard.apply_network_guard(env)
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts == [str(network_guard.GUARD_DIR), "/elsewhere"]


@pytest.mark.parametrize("value,expected", [("1", True), ("TRUE", True), ("on", True), ("0", False), ("", False)])
def test_flag_values(value: str, expected: bool) -> None:
    """Only true-ish values enable blocking."""
    assert network_guard.network_blocked({network_guard.BLOCK_NETWORK_ENV: value}) is expected
