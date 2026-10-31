"""The worker entry point, run as a real process."""

import itertools
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest
from senselab_worker.codec import Kind, encode

TESTS = Path(__file__).parent


def _launch(tmp_path: Path, backend: str, *messages: dict) -> subprocess.CompletedProcess[bytes]:
    kinds: tuple[Kind, ...] = ("init", "stop")
    default: Kind = "request"
    stdin = b"".join(encode(m, next((k for k in kinds if k in m), default), array_dir=tmp_path) for m in messages)
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
    assert proc.returncode == 0 and _keys(proc.stdout) == ["hello", "ready", "id", "id"]
    for noise in (b"noise at import", b"noise on fd 1 at import", b"noise from load"):
        assert noise in proc.stderr


@pytest.mark.logic
def test_fake_runs_through_the_launcher(tmp_path: Path) -> None:
    """The fake backend serves a session through the launcher."""
    proc = _launch(tmp_path, "senselab_worker.fake", *_session(tmp_path))
    assert proc.returncode == 0
    *_, result, done = proc.stdout.splitlines()
    assert "value" in json.loads(result)["outputs"] and json.loads(done) == {"id": "r", "done": 1}


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
    report = json.loads(proc.stdout.splitlines()[-2])["outputs"]
    assert report == {
        "fd0_is_devnull": True,
        "fd1_is_fd2": True,
        "sys_stdin_is_devnull": True,
        "sys_stdout_is_stderr": True,
        "inheritable": [0, 1, 2],
    }


@pytest.mark.logic
def test_closed_fd0_at_start_reads_dev_null() -> None:
    """A worker started with fd 0 closed says hello and exits 0 at the end of its (empty) channel."""
    script = 'exec "$0" -m senselab_worker.launch senselab_worker.fake <&-'
    proc = subprocess.run(["/bin/sh", "-c", script, sys.executable], capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert _keys(proc.stdout) == ["hello"]


PLANTED = b"\xde\xff planted\n"
SITECUSTOMIZE = """
import os
held = [open(os.environ["PLANT"], "r+b") for _ in range(3)]
with open(os.environ["PLANT"] + ".fds", "w") as f:
    f.write(" ".join(str(h.fileno()) for h in held))
"""


def _bounded() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))


@pytest.mark.contract
@pytest.mark.parametrize(
    "closed", [c for n in range(4) for c in itertools.combinations((0, 1, 2), n)], ids=lambda c: f"closed{list(c)}"
)
def test_files_opened_at_interpreter_start_never_become_the_channel(tmp_path: Path, closed: tuple[int, ...]) -> None:
    """With any of fds 0-2 closed, files a `sitecustomize` holds open on them are neither read nor written."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    plant = tmp_path / "planted.bin"
    plant.write_bytes(PLANTED)
    redirects = " ".join(f"{fd}>&-" if fd else "<&-" for fd in closed)
    script = f'exec "$0" -m senselab_worker.launch senselab_worker.fake {redirects}'
    env = {**os.environ, "PYTHONPATH": str(site), "PLANT": str(plant)}
    proc = subprocess.run(
        ["/bin/sh", "-c", script, sys.executable],
        input=b"",
        capture_output=True,
        timeout=60,
        env=env,
        preexec_fn=_bounded,
    )
    held = {int(fd) for fd in (tmp_path / "planted.bin.fds").read_text().split()}
    assert set(closed) <= held, "the plant did not land on the closed descriptors"
    assert proc.returncode == 0, proc.stderr
    assert plant.read_bytes() == PLANTED
    if 1 not in closed:
        assert _keys(proc.stdout) == ["hello"]
