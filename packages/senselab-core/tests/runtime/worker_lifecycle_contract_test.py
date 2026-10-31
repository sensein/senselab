"""Every fixture backend, on every shutdown path, leaves no process and no file behind, within the bound."""

import os
import sys
import threading
import time
from pathlib import Path

import numpy
import pytest
from backends.subdir_backend_fixture import ARRAY_LIKE
from lifecycle_probes_fixture import alive, gate, group_alive, listing, recorded_pids
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import SHUTDOWN_BOUND_S, WorkerClient, WorkerLost

BACKENDS = [
    "backends.forking_backend_fixture",
    "backends.stdin_reading_backend_fixture",
    "backends.subdir_backend_fixture",
    "backends.slow_reader_backend_fixture",
]
PATHS = ["normal", "timeout", "crash", "close_from_another_thread"]
TIMEOUT_S = 2.0
ITEMS = [{"id": "a", "inputs": {"x": numpy.arange(16, dtype="<f4")}}, {"id": "b", "inputs": {}}]


def _close_during_run(client: WorkerClient, g: object) -> list[BaseException]:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            client.run(ITEMS, {"mode": "gate"})
        except BaseException as exc:  # noqa: BLE001 - the test inspects whatever the run raised
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    g.entered()  # type: ignore[attr-defined]
    client.close()
    thread.join(SHUTDOWN_BOUND_S)
    assert not thread.is_alive(), "the run outlived close() from another thread"
    return errors


def _signals_only_while_the_leader_is_held(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Wrap `os.killpg`; return the (pgid, signal) pairs sent after the group leader had been reaped."""
    late: list[tuple[int, int]] = []
    real = os.killpg

    def checked(pgid: int, sig: int) -> None:
        if sig != 0 and not worker_client._leader_held(pgid):
            late.append((pgid, sig))
        real(pgid, sig)

    monkeypatch.setattr(os, "killpg", checked)
    return late


@pytest.mark.contract
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_shutdown_leaves_no_process_and_no_file(
    backend: str, path: str, tmp_path: Path, backends_importable: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After `close()`: no process remains, the call met its bound and the parent is unchanged.

    The group was signalled only while its leader was unreaped. The one exception is a directory the
    backend made in the array directory: it is not the client's, so it stays there, and nothing else
    does.
    """
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "theirs.txt").write_text("t")
    before = listing(parent)
    pid_file = tmp_path / "pids"
    late = _signals_only_while_the_leader_is_held(monkeypatch)
    with gate() as g:
        client = WorkerClient(
            sys.executable,
            backend,
            init={"pid_file": str(pid_file), "gate_port": g.port},
            array_parent=parent,
            request_timeout=TIMEOUT_S if path == "timeout" else None,
        )
        pgid = client.hello["pid"]
        assert all("outputs" in r for r in client.run(ITEMS, {}))
        bound = SHUTDOWN_BOUND_S
        t0 = time.monotonic()
        if path == "timeout":
            bound += TIMEOUT_S
            with pytest.raises(TimeoutError):
                client.run(ITEMS, {"mode": "hang"})
        elif path == "crash":
            with pytest.raises(WorkerLost):
                client.run(ITEMS, {"mode": "crash"})
        elif path == "close_from_another_thread":
            errors = _close_during_run(client, g)
            assert [type(e) for e in errors] == [WorkerLost]
        client.close()
        elapsed = time.monotonic() - t0
    assert elapsed < bound
    assert not group_alive(pgid)
    assert late == [], "a signal went to the group after its leader was reaped"
    assert not any(map(alive, recorded_pids(pid_file)))
    made = backend == "backends.subdir_backend_fixture"
    name = client.array_dir.name
    left = {name, f"{name}/{ARRAY_LIKE}", f"{name}/{ARRAY_LIKE}/scratch.bin"} if made else set()
    assert listing(parent) == before | left
