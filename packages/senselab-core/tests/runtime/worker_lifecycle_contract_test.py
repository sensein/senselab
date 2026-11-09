"""Every fixture backend, on every shutdown path, leaves no process and no file behind, within the bound."""

import os
import threading
from collections.abc import Callable
from pathlib import Path

import numpy
import pytest
from backends.subdir_backend_fixture import ARRAY_LIKE
from lease_clock_fixture import Called, ManualClock
from lifecycle_probes_fixture import gate, listing, recorded_pids
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import ShutdownGraces, WorkerClient, WorkerLost
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.processes import alive, group_alive

PYTHON = source_tree_python()

BACKENDS = [
    "backends.forking_backend_fixture",
    "backends.stdin_reading_backend_fixture",
    "backends.subdir_backend_fixture",
    "backends.slow_reader_backend_fixture",
]
PATHS = ["normal", "timeout", "crash", "close_from_another_thread"]
TIMEOUT_S = 8.0
STEP_S = 1.0
"""How far the test advances the clock each time a stop in progress is parked on it."""
UNREACHED_S = 2.0**20
GRACES = ShutdownGraces(stop=UNREACHED_S, kill=UNREACHED_S)
"""Graces no test reaches: a group signalled is waited for until it is gone."""
ITEMS = [{"id": "a", "inputs": {"x": numpy.arange(16, dtype="<f4")}}, {"id": "b", "inputs": {}}]


def _close_during_run(client: WorkerClient, g: object, hang_s: float) -> list[BaseException]:
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
    thread.join(hang_s)
    assert not thread.is_alive(), "the run outlived close() from another thread"
    return errors


def _signals_only_while_the_leader_is_held(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Wrap `os.killpg`; return the (pgid, signal) pairs sent after the group leader had been reaped."""
    late: list[tuple[int, int]] = []
    real = os.killpg

    def checked(pgid: int, sig: int) -> None:
        if sig != 0:
            probe_exit_state = threading.Event()
            probe_exit_state.set()
            try:
                held = worker_client._leader_held(pgid, probe_exit_state, worker_client.LEADER_PROBE_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - a probe failure must not stop the client's kill
                held = None
            if held is False:
                late.append((pgid, sig))
        real(pgid, sig)

    monkeypatch.setattr(os, "killpg", checked)
    return late


def _shutdown_leaves_nothing(
    backend: str, path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_clock: Callable[..., ManualClock]
) -> None:
    """Run `path` against `backend` and assert nothing is left behind.

    A stop that needed a grace to pass would park on the clock, which only the timeout path advances, and fail at
    the clock's hang bound.
    """
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "theirs.txt").write_text("t")
    before = listing(parent)
    pid_file = tmp_path / "pids"
    late = _signals_only_while_the_leader_is_held(monkeypatch)
    clock = lease_clock(spends="park" if path == "timeout" else False)
    with gate() as g:
        client = WorkerClient(
            PYTHON,
            backend,
            init={"pid_file": str(pid_file), "gate_port": g.port},
            array_parent=parent,
            request_timeout=TIMEOUT_S if path == "timeout" else None,
            graces=GRACES,
            clock=clock,
        )
        pgid = client.hello["pid"]
        assert all("outputs" in r for r in client.run(ITEMS, {}))
        if path == "timeout":
            call = Called(lambda: client.run(ITEMS, {"mode": "hang"}), hang_s=clock.hang_s)
            g.entered()
            assert clock.parked(call.thread)
            clock.advance_to(TIMEOUT_S)
            clock.drive(call.thread, STEP_S)
            with pytest.raises(TimeoutError):
                call.result()
        elif path == "crash":
            with pytest.raises(WorkerLost):
                client.run(ITEMS, {"mode": "crash"})
        elif path == "close_from_another_thread":
            errors = _close_during_run(client, g, clock.hang_s)
            assert [type(e) for e in errors] == [WorkerLost]
        client.close()
    assert not group_alive(pgid)
    assert late == [], "a signal went to the group after its leader was reaped"
    assert not any(map(alive, recorded_pids(pid_file)))
    made = backend == "backends.subdir_backend_fixture"
    name = client.array_dir.name
    left = {name, f"{name}/{ARRAY_LIKE}", f"{name}/{ARRAY_LIKE}/scratch.bin"} if made else set()
    assert listing(parent) == before | left


@pytest.mark.contract
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_shutdown_leaves_no_process_and_no_file(  # noqa: PLR0913 - the case, its fixtures and the clock
    backend: str,
    path: str,
    tmp_path: Path,
    backends_importable: None,
    monkeypatch: pytest.MonkeyPatch,
    lease_clock: Callable[..., ManualClock],
) -> None:
    """After `close()`: no process remains and the parent is unchanged.

    The group was signalled only while its leader was unreaped. The one exception is a directory the
    backend made in the array directory: it is not the client's, so it stays there, and nothing else
    does.
    """
    _shutdown_leaves_nothing(backend, path, tmp_path, monkeypatch, lease_clock)
