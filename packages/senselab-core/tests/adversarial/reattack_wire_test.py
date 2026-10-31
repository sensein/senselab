# ruff: noqa: ANN401
"""H8c re-attack findings on the worker lifecycle (XW-1..XW-4), rebuilt from reviews/phase1-reattack/wire.md.

Each `test_xw<n>_*` rebuilds its finding's probe and fails at 99a6d5f8; the parametrised and generated tests
are contracts over the class the finding belongs to. Probes that could hang at the base run in the shutdown
schedule driver, which bounds each case.
"""

import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy
import pytest
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import SHUTDOWN_BOUND_S, ProtocolMismatch, WorkerClient, WorkerLost

RUNTIME_TESTS = Path(__file__).resolve().parents[1] / "runtime"
FAKE = "senselab_worker.fake"
EXIT_CRASH = 70
WAIT_S = 10.0


@pytest.fixture
def schedule(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The shutdown schedule driver module, importable with the runtime test helpers."""
    monkeypatch.syspath_prepend(str(RUNTIME_TESTS))
    import shutdown_schedule_fixture

    return shutdown_schedule_fixture


@pytest.fixture
def parent(tmp_path: Path) -> Iterator[Path]:
    """A parent for the client's array directory, which must hold nothing after the test."""
    path = tmp_path / "arrays"
    path.mkdir()
    yield path
    assert list(path.iterdir()) == []


@pytest.mark.contract
def test_xw1_sigint_inside_shutdown_still_kills_the_group(tmp_path: Path, schedule: Any) -> None:
    """The probe: SIGINT while the group of a SIGTERM-trapping worker is being stopped left the group running."""
    case = {"path": "timeout", "boundary": "module:_members_gone", "kind": "sigint", "occurrence": 1}
    [report] = schedule.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] == "KeyboardInterrupt"
    assert not report["group_alive"], "the group was running when KeyboardInterrupt surfaced"
    assert schedule.violations(report) == []


@pytest.mark.contract
def test_xw2_close_from_a_signal_handler_inside_close_returns(tmp_path: Path, schedule: Any) -> None:
    """The probe: a SIGUSR1 handler calling `close()` while `close()` stopped the group deadlocked on its lock."""
    case = {"path": "close", "boundary": "module:_members_gone", "kind": "handler_close", "occurrence": 1}
    [report] = schedule.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] is None
    assert report["elapsed"] < SHUTDOWN_BOUND_S
    assert schedule.violations(report) == []


@pytest.mark.logic
def test_xw3_close_from_another_thread_while_encoding_is_worker_lost(
    parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe: `close()` from another thread while `run` encoded made `run` raise `ValueError`, a content fault."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=parent)
    real = worker_client.to_wire
    closer = threading.Thread(target=client.close, daemon=True)

    def encode_while_closing(*args: Any, **kwargs: Any) -> Any:
        if not closer.is_alive() and closer.ident is None:
            closer.start()
            end = time.monotonic() + WAIT_S
            while client.returncode is None and time.monotonic() < end:
                time.sleep(0.01)
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_client, "to_wire", encode_while_closing)
    outcome: list[BaseException] = []

    def call() -> None:
        try:
            client.run([{"id": "a", "inputs": {"x": numpy.arange(4.0)}}], {})
        except BaseException as exc:  # noqa: BLE001 - the test inspects whatever surfaced
            outcome.append(exc)

    runner = threading.Thread(target=call, daemon=True)
    runner.start()
    runner.join(WAIT_S + SHUTDOWN_BOUND_S)
    closer.join(SHUTDOWN_BOUND_S)
    assert not runner.is_alive() and not closer.is_alive()
    assert [type(e) for e in outcome] == [WorkerLost], outcome


def _interpreter(directory: Path, body: str) -> Path:
    path = directory / "interpreter"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _await_exit(client: WorkerClient) -> None:
    end = time.monotonic() + WAIT_S
    while client.alive:
        assert time.monotonic() < end, "the worker did not exit"
        time.sleep(0.01)


def _late_notification(monkeypatch: pytest.MonkeyPatch) -> None:
    real = worker_client._wait_exit

    def late(pid: int) -> Any:
        time.sleep(0.3)
        return real(pid)

    monkeypatch.setattr(worker_client, "_wait_exit", late)


@pytest.mark.logic
@pytest.mark.parametrize("how", ["idle_sigkill", "crash_mid_request", "late_exit_notification", "exit_before_hello"])
def test_xw4_every_death_carries_its_exit_code(parent: Path, monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    """The probes: the exit code was lost when the worker died idle, before its hello, or before kqueue registered.

    Every way the worker can die surfaces as a typed fault naming the code the worker exited with.
    """
    if how == "exit_before_hello":
        with pytest.raises(ProtocolMismatch) as mismatch:
            WorkerClient(_interpreter(parent.parent, "exit 7"), FAKE, init={}, array_parent=parent)
        assert any("worker exit code 7;" in note for note in mismatch.value.__notes__)
        return
    if how == "late_exit_notification":
        _late_notification(monkeypatch)
    init = {} if how == "idle_sigkill" else {"faults": {"crash_after": 0}}
    client = WorkerClient(sys.executable, FAKE, init=init, array_parent=parent)
    try:
        expected = EXIT_CRASH
        if how == "idle_sigkill":
            os.kill(client.hello["pid"], signal.SIGKILL)
            _await_exit(client)
            expected = -signal.SIGKILL
        with pytest.raises(WorkerLost) as lost:
            client.run([{"id": "a", "inputs": {}}], {})
        assert lost.value.returncode == expected and f"exit code {expected}" in str(lost.value)
        assert client.returncode == expected
    finally:
        client.close()
