# ruff: noqa: ANN401
"""H8c re-attack findings on the worker lifecycle (XW-1..XW-4) and the H9a review's (I-1, I-3, Ctrl-C).

The XW findings are rebuilt from reviews/phase1-reattack/wire.md, the I items from the H9a review. Each
`test_xw<n>_*` and `test_i<n>_*` rebuilds its probe; the parametrised and generated tests are contracts over the
class the finding belongs to. Probes that could hang run in the shutdown schedule driver, which bounds each case.
"""

import gc
import logging
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy
import pytest
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import (
    ESCALATE_HINT,
    SHUTDOWN_BOUND_S,
    WorkerClient,
    WorkerLost,
    _signals_deferred,
)

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
def probes(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The lifecycle probes the runtime tests share."""
    monkeypatch.syspath_prepend(str(RUNTIME_TESTS))
    import lifecycle_probes_fixture

    return lifecycle_probes_fixture


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
    assert not report["at_outcome"]["group_alive"], "the group was running when KeyboardInterrupt surfaced"
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
def test_xw4_every_death_carries_its_exit_code(
    parent: Path, monkeypatch: pytest.MonkeyPatch, probes: Any, how: str
) -> None:
    """The probes: the exit code was lost when the worker died idle, before its hello, or before kqueue registered.

    Every way the worker can die surfaces as a typed fault naming the code the worker exited with.
    """
    if how == "exit_before_hello":
        with pytest.raises(WorkerLost) as before:
            WorkerClient(probes.interpreter(parent.parent, "exit 7"), FAKE, init={}, array_parent=parent)
        assert before.value.returncode == 7
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


FORK_WAIT_S = 30.0


def _forked(action: Any) -> int:
    """Run `action` in a forked child that exits 0, or 1 if it raised; return the child's exit status."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        code = 1
        try:
            action()
            code = 0
        finally:
            os._exit(code)
    end = time.monotonic() + FORK_WAIT_S
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() > end:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise AssertionError("the forked child did not finish")
        time.sleep(0.01)


@pytest.mark.contract
@pytest.mark.parametrize("how", ["close", "collect", "run"])
def test_i1_a_forked_child_never_touches_the_parents_worker(parent: Path, how: str) -> None:
    """The probe: a forked child's `close()` stopped the parent's worker and deleted its array directory.

    In a forked child, closing, collecting or running the inherited client leaves the parent's worker, files and
    channel as they were.
    """
    box = [WorkerClient(sys.executable, FAKE, init={}, array_parent=parent)]
    try:

        def act() -> None:
            if how == "close":
                box[0].close()
            elif how == "collect":
                box.clear()
                gc.collect()
            else:
                with pytest.raises(WorkerLost):
                    box[0].run([{"id": "a", "inputs": {}}], {})

        assert _forked(act) == 0
        client = box[0]
        assert client.alive and client.array_dir.is_dir()
        assert "outputs" in client.run([{"id": "a", "inputs": {}}], {})[0]
    finally:
        box[0].close()


def _interrupting(monkeypatch: pytest.MonkeyPatch, *, times: int) -> None:
    """Make the first `times` calls of `_members_gone` raise `SystemExit`, as a signal handler might."""
    real = worker_client._members_gone
    calls = [0]

    def interrupted(*args: Any) -> bool:
        calls[0] += 1
        if calls[0] <= times:
            raise SystemExit(1)
        return bool(real(*args))

    monkeypatch.setattr(worker_client, "_members_gone", interrupted)


@pytest.mark.logic
def test_i3_an_interrupted_idle_close_still_reaps_and_removes_the_files(
    parent: Path, monkeypatch: pytest.MonkeyPatch, probes: Any
) -> None:
    """The probe: `SystemExit` inside an idle `close()` left the leader unreaped and the array directory behind."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=parent)
    pgid = client.hello["pid"]
    _interrupting(monkeypatch, times=1)
    with pytest.raises(SystemExit):
        client.close()
    assert client.returncode is not None and not client.array_dir.exists()
    assert not probes.group_alive(pgid)


@pytest.mark.logic
def test_i3_collecting_a_client_whose_shutdown_was_cut_short_finishes_it(
    parent: Path, monkeypatch: pytest.MonkeyPatch, probes: Any
) -> None:
    """A client whose close and finishing pass were both interrupted is finished when it is collected.

    Nothing may keep the interruption's traceback, which holds the client: the log record is suppressed.
    """
    box = [WorkerClient(sys.executable, FAKE, init={}, array_parent=parent)]
    pgid = box[0].hello["pid"]
    array_dir = box[0].array_dir
    _interrupting(monkeypatch, times=2)
    monkeypatch.setattr(worker_client.logger, "disabled", True)
    try:
        box[0].close()
    except SystemExit:
        pass
    else:
        raise AssertionError("close() was not interrupted")
    assert array_dir.exists()
    monkeypatch.undo()
    box.clear()
    gc.collect()
    assert not array_dir.exists() and not probes.group_alive(pgid)


@pytest.mark.logic
def test_ctrl_c_held_once_with_a_hint_and_escalated_by_a_second(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first SIGINT during shutdown is held and logs the hint; the second raises `KeyboardInterrupt` at once."""
    monkeypatch.setattr(worker_client, "_HINTED", threading.Event())
    with caplog.at_level(logging.WARNING), pytest.raises(KeyboardInterrupt):
        with _signals_deferred(escalate=True):
            signal.raise_signal(signal.SIGINT)
            assert caplog.messages.count(ESCALATE_HINT) == 1
            signal.raise_signal(signal.SIGINT)
            raise AssertionError("the second SIGINT did not interrupt")


def _bounded() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))


_THREE_SHUTDOWNS = """
import logging, signal
from senselab_core.runtime.worker_client import _signals_deferred
logging.basicConfig(level=logging.WARNING)
for _ in range(3):
    try:
        with _signals_deferred(escalate=True):
            signal.raise_signal(signal.SIGINT)
    except KeyboardInterrupt:
        pass
"""


@pytest.mark.logic
def test_l1_the_escalation_hint_is_logged_once_per_process() -> None:
    """Three shutdowns in one process, each interrupted once, log the hint once."""
    proc = subprocess.run(
        [sys.executable, "-c", _THREE_SHUTDOWNS], capture_output=True, text=True, timeout=WAIT_S, preexec_fn=_bounded
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stderr.count(ESCALATE_HINT) == 1


@pytest.mark.contract
def test_ctrl_c_twice_kills_reaps_and_releases_within_the_ctrl_c_bound(tmp_path: Path, schedule: Any) -> None:
    """With the real graces, a second SIGINT while a SIGTERM-ignoring group is stopped ends the close in about 1 s.

    The bound adds a load allowance and stays below the real STOP grace, so a held second SIGINT still fails.

    The group is SIGKILLed and reaped and the files are removed before `KeyboardInterrupt` surfaces.
    """
    case = {
        "path": "timeout",
        "boundary": "module:_members_gone",
        "kind": "double_sigint",
        "occurrence": 1,
        "real_graces": True,
    }
    [report] = schedule.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] == "KeyboardInterrupt"
    assert report["elapsed"] < schedule.ctrl_c_bound()
    assert report["at_outcome"] == schedule.CLEAN
    assert schedule.violations(report) == []
