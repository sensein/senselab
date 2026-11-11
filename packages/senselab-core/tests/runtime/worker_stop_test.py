"""A worker that times out or is closed is stopped in order, each grace read on the test clock."""

import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from lease_clock_fixture import Called, ManualClock
from lifecycle_probes_fixture import Gate, gate, recorded_pids
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import (
    ShutdownGraces,
    WorkerClient,
    WorkerTimeout,
)
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.processes import alive, group_alive, state
from senselab_worker.fake import FAULT_KEY
from senselab_worker.faults import error_record

FAKE = "senselab_worker.fake"
PYTHON = source_tree_python()
pytestmark = pytest.mark.usefixtures("no_leaked_processes")

REQUEST_TIMEOUT_S = 8.0
"""A request timeout on the test clock, which a test reaches by advancing it."""


STOP_STEP_S = 1.0
"""How far the test advances the clock each time a stop in progress is parked on it."""


UNREACHED_S = 2.0**20
"""A grace no test advances the clock to: what it bounds, a real death, is waited for in real time."""


GRACES = ShutdownGraces(stop=UNREACHED_S, kill=UNREACHED_S)
"""Graces that are never reached: a group signalled is waited for until it is gone."""


REACHABLE = ShutdownGraces(
    stop=2 * STOP_STEP_S, term=STOP_STEP_S, kill=2 * STOP_STEP_S, join=STOP_STEP_S, exit_code_wait=STOP_STEP_S
)
"""Graces the test clock reaches: each is passed by `drive` advancing `STOP_STEP_S` a few times."""


def _waiting(clock: ManualClock, call: Called) -> None:
    """Assert `call` is parked on `clock`, then still parked after the clock passed half of `REQUEST_TIMEOUT_S`."""
    assert clock.parked(call.thread), "the call ended before it waited"
    clock.advance(REQUEST_TIMEOUT_S / 2)
    assert clock.parked(call.thread), "the call gave up before its timeout"


def _timed_out(clock: ManualClock, call: Called) -> BaseException:
    """Advance `clock` to the timeout of a call parked on it, then drive its stop; return what the call raised."""
    _waiting(clock, call)
    clock.advance_to(REQUEST_TIMEOUT_S)
    clock.drive(call.thread, STOP_STEP_S)
    with pytest.raises(TimeoutError) as info:
        call.result()
    return info.value


def _hung_forking_client(  # noqa: PLR0913 - the clock, where the worker reports, and the variants of the request
    clock: ManualClock,
    pid_file: Path,
    g: Gate,
    array_parent: Path | None = None,
    mode: str = "hang",
    graces: ShutdownGraces = GRACES,
) -> tuple[WorkerClient, Called]:
    """A client over the forking backend, in a `mode` request that has recorded its grandchildren; it and the call."""
    client = WorkerClient(
        PYTHON,
        "backends.forking_backend_fixture",
        init={"pid_file": str(pid_file), "gate_port": g.port},
        array_parent=array_parent,
        request_timeout=REQUEST_TIMEOUT_S,
        graces=graces,
        clock=clock,
    )
    call = Called(lambda: client.run([{"id": "a", "inputs": {}}], {"mode": mode}), hang_s=clock.hang_s)
    g.entered()
    return client, call


def _escalation(sent: list[tuple[float, int, int]]) -> list[tuple[float, signal.Signals]]:
    """The signals a stop sent a group, with the clock's time at each, not counting the liveness probes."""
    return [(when, signal.Signals(sig)) for when, _, sig in sent if sig != 0]


@pytest.mark.logic
def test_timeout_is_resource_fault(
    array_parent: Path, lease_clock: Callable[..., ManualClock], slack_tests: dict[str, Any]
) -> None:
    """A load or request past its timeout is `WorkerTimeout`, a typed resource fault, and the worker is killed.

    The first client's graces are reachable on its clock, which the test drives. A lost wake in either half fails
    within `stop_test.hang_s` real seconds, not at the tier's budget.
    """
    hang_s = slack_tests["stop_test"]["hang_s"]
    clock = lease_clock(spends="park", hang_s=hang_s)
    client = WorkerClient(
        PYTHON,
        FAKE,
        init={},
        array_parent=array_parent,
        request_timeout=REQUEST_TIMEOUT_S,
        graces=REACHABLE,
        clock=clock,
    )
    call = Called(lambda: client.run([{"id": "a", "inputs": {}}], {FAULT_KEY: {"latency_s": 60}}), hang_s=clock.hang_s)
    error = _timed_out(clock, call)
    record = error_record(error)
    assert (record["fault"], record["gap"]) == ("resource", False) and isinstance(error, WorkerTimeout)
    assert not client.alive and error.__notes__
    client.close()
    with pytest.raises(WorkerTimeout):
        WorkerClient(
            PYTHON,
            FAKE,
            init={},
            array_parent=array_parent,
            load_timeout=0.0,
            graces=REACHABLE,
            clock=lease_clock(spends=False, hang_s=hang_s),
        )


@pytest.mark.contract
def test_forking_backend_leaves_no_survivor(
    tmp_path: Path,
    backends_importable: None,
    monkeypatch: pytest.MonkeyPatch,
    lease_clock: Callable[..., ManualClock],
) -> None:
    """A timed-out request and `close()` leave no grandchild running, by SIGTERM alone while they die of it."""
    pid_file = tmp_path / "pids"
    clock = lease_clock(spends="park")
    sent = _record_killpg(monkeypatch, clock)
    with gate() as g:
        client, call = _hung_forking_client(clock, pid_file, g)
        pgid = client.hello["pid"]
        _timed_out(clock, call)
    t = REQUEST_TIMEOUT_S
    assert _escalation(sent) == [(t, signal.SIGTERM), (t, signal.SIGCONT)]
    client.close()
    pids = recorded_pids(pid_file)
    assert len(pids) == 2 and not any(map(alive, pids))
    assert not group_alive(pgid)


@pytest.mark.contract
def test_a_worker_ignoring_sigterm_is_killed_when_the_stop_grace_ends(
    tmp_path: Path,
    backends_importable: None,
    monkeypatch: pytest.MonkeyPatch,
    lease_clock: Callable[..., ManualClock],
) -> None:
    """A timed-out worker that ignores SIGTERM gets SIGTERM and SIGCONT, then SIGKILL once `stop` has passed."""
    graces = ShutdownGraces(stop=2 * REQUEST_TIMEOUT_S, kill=UNREACHED_S)
    pid_file = tmp_path / "pids"
    clock = lease_clock(spends="park")
    sent = _record_killpg(monkeypatch, clock)
    with gate() as g:
        client, call = _hung_forking_client(clock, pid_file, g, mode="deaf", graces=graces)
        pgid = client.hello["pid"]
        _timed_out(clock, call)
    t = REQUEST_TIMEOUT_S
    assert _escalation(sent) == [(t, signal.SIGTERM), (t, signal.SIGCONT), (t + graces.stop, signal.SIGKILL)]
    client.close()
    assert not group_alive(pgid) and not any(map(alive, recorded_pids(pid_file)))


@pytest.mark.contract
def test_timeout_returns_though_a_grandchild_holds_stderr(
    tmp_path: Path, backends_importable: None, lease_clock: Callable[..., ManualClock]
) -> None:
    """A backend that leaves a grandchild on stderr and hangs: `run` alone returns and kills it.

    The grandchildren start in `load`, so they exist before the request's timeout starts.
    """
    pid_file = tmp_path / "pids"
    init = {"pid_file": str(pid_file), "spawn_at_load": True}
    clock = lease_clock(spends="park")
    client = WorkerClient(
        PYTHON,
        "backends.forking_backend_fixture",
        init=init,
        request_timeout=REQUEST_TIMEOUT_S,
        graces=GRACES,
        clock=clock,
    )
    assert len(recorded_pids(pid_file)) == 2
    call = Called(lambda: client.run([{"id": "a", "inputs": {}}], {"mode": "hang"}), hang_s=clock.hang_s)
    _timed_out(clock, call)
    assert not client.alive and not any(map(alive, recorded_pids(pid_file)))
    client.close()


def _wait_stopped(pid: int, hang_s: float) -> None:
    """Return once child `pid` has been stopped; kill it if it has not been by `hang_s`, so the wait fails."""
    guard = threading.Timer(hang_s, os.kill, (pid, signal.SIGKILL))
    guard.start()
    try:
        _, status = os.waitpid(pid, os.WUNTRACED)
    finally:
        guard.cancel()
    assert os.WIFSTOPPED(status), f"worker {pid} ended instead of stopping"


@pytest.mark.contract
def test_send_to_wedged_worker_respects_deadline(
    array_parent: Path, backends_importable: None, lease_clock: Callable[..., ManualClock]
) -> None:
    """An 8 MB request to a worker that has stopped reading times out at `request_timeout`, not before."""
    clock = lease_clock(spends="park")
    client = WorkerClient(
        PYTHON,
        "backends.slow_reader_backend_fixture",
        init={},
        array_parent=array_parent,
        request_timeout=REQUEST_TIMEOUT_S,
        graces=GRACES,
        clock=clock,
    )
    client.run([{"id": "a", "inputs": {}}], {"mode": "stop_reading"})
    _wait_stopped(client.hello["pid"], clock.hang_s)
    call = Called(lambda: client.run([{"id": "a", "inputs": {}}], {"blob": "x" * 8 * 2**20}), hang_s=clock.hang_s)
    _timed_out(clock, call)
    assert not group_alive(client.hello["pid"])
    client.close()


def _record_killpg(monkeypatch: pytest.MonkeyPatch, clock: ManualClock) -> list[tuple[float, int, int]]:
    """Record each signal sent to a group as (the clock's time, the group, the signal)."""
    sent: list[tuple[float, int, int]] = []
    real = os.killpg

    def record(pgid: int, sig: int) -> None:
        sent.append((clock.now(), pgid, sig))
        real(pgid, sig)

    monkeypatch.setattr(os, "killpg", record)
    return sent


@pytest.mark.logic
def test_late_close_after_the_leader_was_reaped_elsewhere_sends_no_signal(
    array_parent: Path, monkeypatch: pytest.MonkeyPatch, lease_clock: Callable[..., ManualClock]
) -> None:
    """Once the leader is reaped outside the client its pid may be reused, so `close()` signals nothing."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent)
    pid = client.hello["pid"]
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    sent = _record_killpg(monkeypatch, lease_clock())
    client.close()
    assert [s for s in sent if s[2] != 0] == []


@pytest.mark.logic
def test_idle_crash_keeps_the_leader_unreaped_until_close(array_parent: Path, tier_bound: float) -> None:
    """A leader that dies while idle stays a zombie, holding its pid and group id, until `close()` reaps it."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent)
    pid = client.hello["pid"]
    os.kill(pid, signal.SIGKILL)
    assert client.wait_exited(tier_bound)
    assert client.returncode is None and state(pid).startswith("Z")
    client.close()
    assert client.returncode == -signal.SIGKILL


@pytest.mark.contract
def test_graceful_close_leaves_sigterm_its_own_grace(
    array_parent: Path,
    backends_importable: None,
    monkeypatch: pytest.MonkeyPatch,
    lease_clock: Callable[..., ManualClock],
) -> None:
    """A worker that ignores the stop line gets no signal until `stop - term` has passed, then SIGTERM, not SIGKILL."""
    clock = lease_clock(spends="park")
    graces = ShutdownGraces(stop=UNREACHED_S, term=UNREACHED_S / 2, kill=UNREACHED_S)
    client = WorkerClient(
        PYTHON, "backends.slow_reader_backend_fixture", init={}, array_parent=array_parent, graces=graces, clock=clock
    )
    client.run([{"id": "a", "inputs": {}}], {"mode": "stop_reading"})
    _wait_stopped(client.hello["pid"], clock.hang_s)
    sent = _record_killpg(monkeypatch, clock)
    closing = Called(client.close, hang_s=clock.hang_s)
    assert clock.parked(closing.thread)
    clock.advance_to(graces.stop - graces.term - graces.stop / 8)
    assert clock.parked(closing.thread) and not sent, "a signal went before the stop line's share of the grace passed"
    clock.advance_to(graces.stop - graces.term)
    clock.drive(closing.thread, STOP_STEP_S)
    closing.result()
    signals = [(when, sig) for when, _, sig in sent if sig != 0]
    assert signals[0] == (graces.stop - graces.term, signal.SIGTERM)
    assert signal.SIGKILL not in [sig for _, sig in signals]


def _probe_fails(monkeypatch: pytest.MonkeyPatch, how: str) -> list[float]:
    """Route the leader probe through `ps`, as where `waitid` is missing; make every `ps` fail as `how`."""
    calls: list[float] = []
    real_run = subprocess.run

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        if argv[0] != worker_client._PS:
            return real_run(argv, **kwargs)
        timeout = kwargs["timeout"]
        assert isinstance(timeout, float) and timeout <= worker_client.LEADER_PROBE_TIMEOUT_S
        calls.append(timeout)
        if how == "missing":
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.delattr(os, "waitid", raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _close_after_an_idle_crash_with_failing_ps(
    array_parent: Path, monkeypatch: pytest.MonkeyPatch, how: str, clock: ManualClock, tier_bound: float
) -> None:
    """Crash an idle worker, make `ps` fail as `how`, close; assert nothing was signalled and no grace passed.

    The clock is never advanced, so a close that waited on a grace would park on it and fail at the hang bound.
    """
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent, clock=clock)
    pid = client.hello["pid"]
    os.kill(pid, signal.SIGKILL)
    assert client.wait_exited(tier_bound)
    calls = _probe_fails(monkeypatch, how)
    sent = _record_killpg(monkeypatch, clock)
    client.close()
    assert calls and [s for s in sent if s[2] != 0] == []
    assert client.returncode == -signal.SIGKILL and not alive(pid)


@pytest.mark.logic
def test_linux_without_waitid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux without `os.waitid`, the exit probe answers 'running' and the leader probe asks `ps`."""
    monkeypatch.setattr(sys, "platform", "linux")
    calls = _probe_fails(monkeypatch, "missing")
    exited = threading.Event()
    assert worker_client._exit_seen(os.getpid(), exited) is False
    exited.set()
    assert worker_client._leader_held(os.getpid(), exited, worker_client.LEADER_PROBE_TIMEOUT_S) is None
    assert len(calls) == 1


@pytest.mark.logic
@pytest.mark.parametrize("how", ["missing", "timeout"])
def test_unknown_leader_state_is_never_read_as_reaped(
    array_parent: Path,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    lease_clock: Callable[..., ManualClock],
    tier_bound: float,
) -> None:
    """With `ps` missing or timing out after an idle crash, `close()` returns, signals nothing and waits on no grace."""
    _close_after_an_idle_crash_with_failing_ps(array_parent, monkeypatch, how, lease_clock(spends=False), tier_bound)


@pytest.mark.contract
@pytest.mark.parametrize("how", ["missing", "timeout"])
def test_a_timed_out_worker_is_stopped_without_asking_ps(  # noqa: PLR0913 - the fixtures of a forking worker
    tmp_path: Path,
    array_parent: Path,
    backends_importable: None,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    lease_clock: Callable[..., ManualClock],
) -> None:
    """A leader whose exit has not been observed is held: a timed-out worker's group is stopped without `ps`."""
    pid_file = tmp_path / "pids"
    clock = lease_clock(spends="park")
    with gate() as g:
        client, call = _hung_forking_client(clock, pid_file, g, array_parent)
        pgid = client.hello["pid"]
        calls = _probe_fails(monkeypatch, how)
        _timed_out(clock, call)
    client.close()
    assert calls == [] and not group_alive(pgid) and not any(map(alive, recorded_pids(pid_file)))


GUARD_S = 0.2
"""The hang guard of the test that the guard fires: short, since that test waits it out in real time."""


@pytest.mark.logic
def test_a_wait_nobody_ends_fails_within_the_hang_guard(lease_clock: Callable[..., ManualClock]) -> None:
    """A wait on a clock nobody advances is a failure after the hang guard's real seconds, in the waiting thread."""
    clock = lease_clock(spends="park", hang_s=GUARD_S)
    parked = Called(lambda: clock.wait(threading.Event(), UNREACHED_S), hang_s=GUARD_S * 10)
    with pytest.raises(AssertionError, match="nobody advanced"):
        parked.result()
