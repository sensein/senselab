# ruff: noqa: ANN401
"""Shutdown under interruption: every injection at every boundary call ends typed, bounded and with no orphan."""

import gc
import logging
import multiprocessing as mp
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import lifecycle_probes_fixture
import numpy
import pytest
import shutdown_schedule_fixture
from senselab_core.runtime.worker_client import (
    DEFAULT_GRACES,
    ESCALATE_HINT,
    WorkerClient,
    WorkerLost,
)
from shutdown_schedule_fixture import (
    CASE_BOUND_S,
    KINDS,
    PATHS,
    REAL_GRACES,
    case_bound,
    counted,
    ctrl_c_bound,
    run_schedule,
    schedule,
    violations,
)


def _check(tmp_path: Path, kind: str, occurrences: range) -> list[dict[str, Any]]:
    todo = [case for case in schedule(counted(tmp_path), occurrences) if case["kind"] == kind]
    reports = run_schedule(tmp_path, todo)
    failed = {f"{r['path']} {r['boundary']}#{r['occurrence']}": v for r in reports if (v := violations(r))}
    assert failed == {}
    return reports


RAISING_KINDS = ("sigint", "handler_raises_base", "handler_raises_exception")
FIRST_CALL_KINDS = tuple(k for k in KINDS if k != "sigint")


def _check_first_calls(tmp_path: Path, kind: str) -> None:
    reports = _check(tmp_path, kind, range(1, 2))
    if kind in RAISING_KINDS:
        assert any(r["calls"]["client:_kill_group_now"] for r in reports)
        assert any(r["calls"]["client:_steps"] >= 2 and r["path"] == "close" for r in reports)


@pytest.mark.contract
def test_a_sigint_at_every_boundary_ends_typed_bounded_and_clean(tmp_path: Path) -> None:
    """A SIGINT at the first call of each boundary, on construction, a timed-out run and an idle close.

    An injection that raises inside the shutdown runs the SIGKILL fallback and a second pass of the steps.
    """
    _check_first_calls(tmp_path, "sigint")


@pytest.mark.limits
@pytest.mark.parametrize("kind", FIRST_CALL_KINDS)
def test_every_injection_at_the_first_call_of_every_boundary_ends_typed_bounded_and_clean(
    tmp_path: Path, kind: str
) -> None:
    """Each other kind of injection at the first call of each boundary, as the SIGINT case is checked."""
    _check_first_calls(tmp_path, kind)


@pytest.mark.limits
@pytest.mark.parametrize("kind", KINDS)
def test_every_injection_at_every_later_boundary_call_ends_typed_bounded_and_clean(tmp_path: Path, kind: str) -> None:
    """The same at every later call of each boundary: the SIGKILL `_signal_group`, the second `_members_gone`."""
    _check(tmp_path, kind, range(2, 100))


@pytest.mark.logic
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("real_graces", [False, True])
def test_every_case_bound_carries_load_slack_yet_fails_a_real_hang(path: str, real_graces: bool) -> None:
    """A shrunk-grace bound stays below the real STOP grace and a real-grace bound below the faulthandler timeout.

    A case that ignores the shrunk graces waits out at least the real STOP grace, and a deadlock ends at
    `CASE_BOUND_S`; both must exceed the bound, so the load allowance never hides either.
    """
    bound = case_bound({"path": path, "real_graces": real_graces})
    assert bound < (CASE_BOUND_S if real_graces else REAL_GRACES.stop)
    assert ctrl_c_bound() < REAL_GRACES.stop


RUNTIME_TESTS = Path(__file__).resolve().parents[1] / "runtime"


FAKE = "senselab_worker.fake"


T = TypeVar("T")


EXIT_CRASH = 70


WAIT_S = 10.0


@pytest.fixture
def parent(tmp_path: Path) -> Iterator[Path]:
    """A parent for the client's array directory, which must hold nothing after the test."""
    path = tmp_path / "arrays"
    path.mkdir()
    yield path
    assert list(path.iterdir()) == []


@pytest.mark.contract
def test_sigint_inside_shutdown_still_kills_the_group(tmp_path: Path) -> None:
    """SIGINT while the group of a SIGTERM-trapping worker is being stopped still ends with the group killed."""
    case = {"path": "timeout", "boundary": "module:_members_gone", "kind": "sigint", "occurrence": 1}
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] == "KeyboardInterrupt"
    assert not report["at_outcome"]["group_alive"], "the group was running when KeyboardInterrupt surfaced"
    assert shutdown_schedule_fixture.violations(report) == []


@pytest.mark.contract
def test_close_from_a_signal_handler_inside_close_returns(tmp_path: Path) -> None:
    """A signal handler that calls `close()` while `close()` stops the group returns instead of waiting on its lock."""
    case = {"path": "close", "boundary": "module:_members_gone", "kind": "handler_close", "occurrence": 1}
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] is None
    assert shutdown_schedule_fixture.violations(report) == []


@pytest.mark.limits
def test_close_from_a_signal_handler_inside_close_returns_within_the_shutdown_bound(tmp_path: Path) -> None:
    """The nested `close()` returns within the shrunk graces' bound."""
    case = {"path": "close", "boundary": "module:_members_gone", "kind": "handler_close", "occurrence": 1}
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] is None
    assert report["elapsed"] < shutdown_schedule_fixture.GRACES.bound


def _in_a_child(fn: Callable[..., T], *args: object) -> T:
    """Run the shutdown fixture's `fn(*args)` in a spawned child, where it may substitute a declared boundary."""
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(fn, *args).result(timeout=WAIT_S + 4 * DEFAULT_GRACES.bound)


@pytest.mark.logic
def test_close_from_another_thread_while_encoding_is_worker_lost(parent: Path) -> None:
    """`close()` from another thread while `run` encodes makes `run` raise `WorkerLost`, not a content fault."""
    assert _in_a_child(shutdown_schedule_fixture.close_while_encoding, str(parent), WAIT_S) == ["WorkerLost"]


def _await_exit(client: WorkerClient) -> None:
    end = time.monotonic() + WAIT_S
    while client.alive:
        assert time.monotonic() < end, "the worker did not exit"
        time.sleep(0.01)


@pytest.mark.logic
@pytest.mark.parametrize("how", ["idle_sigkill", "crash_mid_request", "late_exit_notification", "exit_before_hello"])
def test_every_death_carries_its_exit_code(parent: Path, how: str) -> None:
    """A worker that dies idle, mid-request, before its hello or before the exit watch registers raises `WorkerLost`.

    The fault names the code the worker exited with.
    """
    if how == "exit_before_hello":
        with pytest.raises(WorkerLost) as before:
            WorkerClient(
                lifecycle_probes_fixture.interpreter(parent.parent, "exit 7"), FAKE, init={}, array_parent=parent
            )
        assert before.value.returncode == 7
        return
    if how == "late_exit_notification":
        code, message, client_code = _in_a_child(
            shutdown_schedule_fixture.crash_with_a_late_exit_notification, str(parent)
        )
        assert code == EXIT_CRASH and f"exit code {EXIT_CRASH}" in message
        assert client_code == EXIT_CRASH
        return
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
def test_a_forked_child_never_touches_the_parents_worker(parent: Path, how: str) -> None:
    """In a forked child, closing, collecting or running the inherited client leaves the parent's worker alone.

    The parent's worker, files and channel are as they were.
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
    """Make the first `times` probes of whether the group has live members raise `SystemExit`, as a handler might.

    The probe is `os.listdir("/proc")` where `/proc` exists, else `os.killpg(pgid, 0)`.
    """
    real_killpg, real_listdir = os.killpg, os.listdir
    calls = [0]

    def interrupt() -> None:
        calls[0] += 1
        if calls[0] <= times:
            raise SystemExit(1)

    def killpg(pgid: int, sig: int) -> None:
        if sig == 0:
            interrupt()
        real_killpg(pgid, sig)

    def listdir(path: Any = ".") -> list[str]:  # noqa: ANN401
        if path == "/proc":
            interrupt()
        return real_listdir(path)

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(os, "listdir", listdir)


@pytest.mark.logic
def test_an_interrupted_idle_close_still_reaps_and_removes_the_files(
    parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SystemExit` inside an idle `close()` still reaps the leader and removes the array directory."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=parent)
    pgid = client.hello["pid"]
    _interrupting(monkeypatch, times=1)
    with pytest.raises(SystemExit):
        client.close()
    assert client.returncode is not None and not client.array_dir.exists()
    assert not lifecycle_probes_fixture.group_alive(pgid)


@pytest.mark.logic
def test_collecting_a_client_whose_shutdown_was_cut_short_finishes_it(
    parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client whose close and finishing pass were both interrupted is finished when it is collected.

    Nothing may keep the interruption's traceback, which holds the client: the log record is suppressed.
    """
    box = [WorkerClient(sys.executable, FAKE, init={}, array_parent=parent)]
    pgid = box[0].hello["pid"]
    array_dir = box[0].array_dir
    _interrupting(monkeypatch, times=2)
    monkeypatch.setattr(logging.getLogger("senselab_core.runtime.worker_client"), "disabled", True)
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
    assert not array_dir.exists() and not lifecycle_probes_fixture.group_alive(pgid)


def _bounded() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))


_HELD_THEN_ESCALATED = """
import logging, signal
from senselab_core.runtime.worker_client import ESCALATE_HINT, _signals_deferred
messages = []
class Keep(logging.Handler):
    def emit(self, record):
        messages.append(record.getMessage())
logging.getLogger().addHandler(Keep())
logging.getLogger().setLevel(logging.WARNING)
try:
    with _signals_deferred(escalate=True):
        signal.raise_signal(signal.SIGINT)
        assert messages.count(ESCALATE_HINT) == 1, messages
        signal.raise_signal(signal.SIGINT)
        raise SystemExit("the second SIGINT did not interrupt")
except KeyboardInterrupt:
    print("interrupted")
"""


@pytest.mark.logic
def test_ctrl_c_held_once_with_a_hint_and_escalated_by_a_second() -> None:
    """The first SIGINT during shutdown is held and logs the hint; the second raises `KeyboardInterrupt` at once."""
    proc = subprocess.run(
        [sys.executable, "-c", _HELD_THEN_ESCALATED],
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        preexec_fn=_bounded,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "interrupted"


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
def test_the_escalation_hint_is_logged_once_per_process() -> None:
    """Three shutdowns in one process, each interrupted once, log the hint once."""
    proc = subprocess.run(
        [sys.executable, "-c", _THREE_SHUTDOWNS],
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        preexec_fn=_bounded,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stderr.count(ESCALATE_HINT) == 1


@pytest.mark.contract
def test_ctrl_c_twice_kills_reaps_and_releases_within_the_ctrl_c_bound(tmp_path: Path) -> None:
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
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] == "KeyboardInterrupt"
    assert report["at_outcome"] == shutdown_schedule_fixture.CLEAN
    assert shutdown_schedule_fixture.violations(report) == []


@pytest.mark.limits
def test_ctrl_c_twice_ends_the_close_within_the_ctrl_c_bound(tmp_path: Path) -> None:
    """With the real graces, a second SIGINT ends the close within `ctrl_c_bound()`, below the real STOP grace."""
    case = {
        "path": "timeout",
        "boundary": "module:_members_gone",
        "kind": "double_sigint",
        "occurrence": 1,
        "real_graces": True,
    }
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] == "KeyboardInterrupt"
    assert report["elapsed"] < shutdown_schedule_fixture.ctrl_c_bound()


@pytest.mark.contract
@pytest.mark.parametrize("kind", ["handler_close", "close_same_thread", "close_other_thread"])
def test_close_during_the_spawn_leaves_no_group(tmp_path: Path, kind: str) -> None:
    """`close()` while the constructor spawns the worker ends in `WorkerLost` or a built client, never an orphan."""
    case = {"path": "init", "boundary": "client:_start", "kind": kind, "occurrence": 1}
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] in {None, "WorkerLost"}
    assert not (report["at_outcome"]["group_alive"] and not report["client_open"])
    assert shutdown_schedule_fixture.violations(report) == []


@pytest.mark.contract
def test_an_exception_in_the_kill_fallback_still_reaps_the_leader(tmp_path: Path) -> None:
    """A `RuntimeError` from a handler while a SIGTERM-ignoring group is stopped leaves the leader reaped."""
    case = {"path": "timeout", "boundary": "module:_members_gone", "kind": "handler_raises_exception", "occurrence": 1}
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["fired"] and report["calls"]["module:_signal_group"] >= 3
    assert report["at_outcome"]["reaped"] and not report["at_outcome"]["group_alive"]
    assert shutdown_schedule_fixture.violations(report) == []
