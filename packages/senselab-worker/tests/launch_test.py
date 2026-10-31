"""The worker entry point, run as a real process."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from senselab_worker.codec import encode

TESTS = Path(__file__).parent


def _launch(tmp_path: Path, backend: str, *messages: dict) -> subprocess.CompletedProcess[bytes]:
    stdin = b"".join(encode(m, array_dir=tmp_path) for m in messages)
    env = {**os.environ, "PYTHONPATH": str(TESTS)}
    return subprocess.run(
        [sys.executable, "-m", "senselab_worker.launch", backend],
        input=stdin,
        capture_output=True,
        timeout=60,
        env=env,
    )


def _keys(stdout: bytes) -> list[str]:
    return [next(iter(json.loads(line))) for line in stdout.splitlines()]


def _session(tmp_path: Path) -> list[dict]:
    init = {"init": {"array_dir": str(tmp_path)}}
    return [init, {"id": "r", "items": [{"id": "a", "inputs": {}}], "params": {}}, {"stop": True}]


@pytest.mark.logic
def test_backend_prints_at_import_never_reach_the_protocol_stream(tmp_path: Path) -> None:
    """Writes to `sys.stdout` or fd 1 at import time or in `load` go to stderr."""
    proc = _launch(tmp_path, "printing_backend_fixture", *_session(tmp_path))
    assert proc.returncode == 0 and _keys(proc.stdout) == ["hello", "ready", "id"]
    for noise in (b"noise at import", b"noise on fd 1 at import", b"noise from load"):
        assert noise in proc.stderr


@pytest.mark.logic
def test_fake_runs_through_the_launcher(tmp_path: Path) -> None:
    """The fake backend serves a session through the launcher."""
    proc = _launch(tmp_path, "senselab_worker.fake", *_session(tmp_path))
    assert proc.returncode == 0
    assert "value" in json.loads(proc.stdout.splitlines()[-1])["results"][0]["outputs"]


@pytest.mark.logic
@pytest.mark.parametrize(
    ("backend", "error_type"), [("no_such_backend", "ModuleNotFoundError"), ("json", "AttributeError")]
)
def test_unusable_backend_is_fatal_after_hello(tmp_path: Path, backend: str, error_type: str) -> None:
    """A backend that cannot be imported, or lacks `load` and `handle`, is a fatal load error."""
    proc = _launch(tmp_path, backend, {"init": {"array_dir": str(tmp_path)}})
    out = [json.loads(line) for line in proc.stdout.splitlines()]
    assert proc.returncode != 0 and _keys(proc.stdout) == ["hello", "error"]
    assert out[1]["fatal"] is True and out[1]["error"]["type"] == error_type


@pytest.mark.logic
def test_a8_backend_children_inherit_dev_null_not_the_channel(tmp_path: Path) -> None:
    """The probe's cause: fd 0 is `/dev/null`, fd 1 is stderr, and only fds 0-2 are inheritable."""
    proc = _launch(tmp_path, "channel_probe_backend_fixture", *_session(tmp_path))
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.splitlines()[-1])["results"][0]["outputs"]
    assert report == {
        "fd0_is_devnull": True,
        "fd1_is_fd2": True,
        "sys_stdin_is_devnull": True,
        "sys_stdout_is_stderr": True,
        "inheritable": [0, 1, 2],
    }
