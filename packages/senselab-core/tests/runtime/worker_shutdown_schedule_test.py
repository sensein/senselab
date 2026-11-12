# ruff: noqa: ANN401
"""Shutdown under interruption: every injection at every boundary call ends typed, bounded and with no orphan."""

import gc
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import lifecycle_probes_fixture
import pytest
import shutdown_schedule_fixture
from bounded_probe_fixture import limit_file_size
from senselab_core.runtime.worker_client import (
    DEFAULT_GRACES,
    ESCALATE_HINT,
    WorkerClient,
    WorkerLost,
)
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.processes import group_alive
from senselab_worker.fake import FAULT_KEY
from shutdown_schedule_fixture import (
    CLEAN,
    KINDS,
    counted,
    run_schedule,
    schedule,
    violations,
)

PYTHON = source_tree_python()


def _check(tmp_path: Path, kind: str, occurrences: range) -> list[dict[str, Any]]:
    todo = [case for case in schedule(counted(tmp_path), occurrences) if case["kind"] == kind]
    reports = run_schedule(tmp_path, todo)
    failed = {f"{r['path']} {r['boundary']}#{r['occurrence']}": v for r in reports if (v := violations(r))}
    assert failed == {}
    return reports


@pytest.mark.logic
def test_a_case_whose_clock_tripped_its_guard_is_a_violation() -> None:
    """A clean report passes; the same report with a guard trip the stop path swallowed fails."""
    report: dict[str, Any] = {
        "fired": True,
        "path": "close",
        "kind": "sigint",
        "outcome": None,
        "elapsed": 0.0,
        "bound": 1.0,
        "at_outcome": CLEAN,
        "client_open": False,
        "abandoned": False,
        "after_close": CLEAN,
        "tripped": [],
    }
    report["outcome"] = next(iter(shutdown_schedule_fixture.TYPED["close"] - {None}), None)
    assert violations(report) == []
    assert any("guard" in v for v in violations({**report, "tripped": ["a wait outlived 30s"]}))


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
    assert report["nested_acquires"] == 0, "the nested close tried to take the shutdown lock"
    assert shutdown_schedule_fixture.violations(report) == []


def _in_a_child(fn: Callable[..., T], *args: object) -> T:
    """Run the shutdown fixture's `fn(*args)` in a spawned child, where it may substitute a declared boundary."""
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(fn, *args).result(timeout=WAIT_S + 4 * DEFAULT_GRACES.bound)


@pytest.mark.logic
def test_close_from_another_thread_while_encoding_is_worker_lost(parent: Path) -> None:
    """`close()` from another thread while `run` encodes makes `run` raise `WorkerLost`, not a content fault."""
    assert _in_a_child(shutdown_schedule_fixture.close_while_encoding, str(parent), WAIT_S) == ["WorkerLost"]


@pytest.mark.logic
@pytest.mark.parametrize("how", ["idle_sigkill", "crash_mid_request", "late_exit_notification", "exit_before_hello"])
def test_every_death_carries_its_exit_code(parent: Path, how: str, tier_bound: float) -> None:
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
    params = {} if how == "idle_sigkill" else {FAULT_KEY: {"crash_after": 0}}
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=parent)
    try:
        expected = EXIT_CRASH
        if how == "idle_sigkill":
            os.kill(client.hello["pid"], signal.SIGKILL)
            assert client.wait_exited(tier_bound), "the worker did not exit"
            expected = -signal.SIGKILL
        with pytest.raises(WorkerLost) as lost:
            client.run([{"id": "a", "inputs": {}}], params)
        assert lost.value.returncode == expected and f"exit code {expected}" in str(lost.value)
        assert client.returncode == expected
    finally:
        client.close()


def _forked(action: Any, hang_s: float) -> int:
    """Run `action` in a forked child that exits 0, or 1 if it raised; return its exit status; kill it at `hang_s`."""
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
    hang_guard = threading.Timer(hang_s, os.kill, (pid, signal.SIGKILL))
    hang_guard.start()
    try:
        status = os.waitpid(pid, 0)[1]
    finally:
        hang_guard.cancel()
    code = os.waitstatus_to_exitcode(status)
    assert code != -signal.SIGKILL, "the forked child did not finish"
    return code


@pytest.mark.contract
@pytest.mark.parametrize("how", ["close", "collect", "run"])
def test_a_forked_child_never_touches_the_parents_worker(parent: Path, how: str, tier_bound: float) -> None:
    """In a forked child, closing, collecting or running the inherited client leaves the parent's worker alone.

    The parent's worker, files and channel are as they were.
    """
    box = [WorkerClient(PYTHON, FAKE, init={}, array_parent=parent)]
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

        assert _forked(act, tier_bound) == 0
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
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=parent)
    pgid = client.hello["pid"]
    _interrupting(monkeypatch, times=1)
    with pytest.raises(SystemExit):
        client.close()
    assert client.returncode is not None and not client.array_dir.exists()
    assert not group_alive(pgid)


@pytest.mark.logic
def test_collecting_a_client_whose_shutdown_was_cut_short_finishes_it(
    parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client whose close and finishing pass were both interrupted is finished when it is collected.

    Nothing may keep the interruption's traceback, which holds the client: the log record is suppressed.
    """
    box = [WorkerClient(PYTHON, FAKE, init={}, array_parent=parent)]
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
    assert not array_dir.exists() and not group_alive(pgid)


_HELD_THEN_ESCALATED = """
import logging, signal, threading
from senselab_core.runtime.clock import SystemClock
from senselab_core.runtime.worker_client import ESCALATE_HINT, _signals_deferred
from senselab_core.runtime.worker_pipes import Lines
messages = []
class Keep(logging.Handler):
    def emit(self, record):
        messages.append(record.getMessage())
logging.getLogger().addHandler(Keep())
logging.getLogger().setLevel(logging.WARNING)
hinted = threading.Event()
def log_hint():
    logging.getLogger("hint").warning(ESCALATE_HINT)
    hinted.set()
lines = Lines(SystemClock(), on_hint=log_hint)
try:
    with _signals_deferred(escalate=True, request_hint=lines.request_hint):
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        raise SystemExit("the second SIGINT did not interrupt")
except KeyboardInterrupt:
    assert hinted.wait(10)
    lines.close()
    assert messages.count(ESCALATE_HINT) == 1, messages
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
        preexec_fn=limit_file_size,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "interrupted"


_SIGINT_INSIDE_A_LOGGING_SINK = """
import logging, logging.handlers, queue, signal, sys, threading
from signal_in_lock_fixture import deliver_in
from senselab_core.runtime.clock import SystemClock
from senselab_core.runtime.worker_client import ESCALATE_HINT, _signals_deferred
from senselab_core.runtime.worker_pipes import Lines
records = queue.Queue()
logging.getLogger().addHandler(logging.handlers.QueueHandler(records))
logging.getLogger().setLevel(logging.WARNING)
state = deliver_in(signal.SIGINT, lambda f: f.f_code.co_name == "put", (queue.__file__,))
hinted = threading.Event()
def log_hint():
    logging.getLogger("hint").warning(ESCALATE_HINT)
    hinted.set()
lines = Lines(SystemClock(), on_hint=log_hint)
try:
    with _signals_deferred(escalate=True, request_hint=lines.request_hint):
        logging.getLogger("work").warning("shutdown step")
except KeyboardInterrupt:
    pass
sys.settrace(None)
assert hinted.wait(10)
lines.close()
messages = []
while not records.empty():
    messages.append(records.get_nowait().getMessage())
print(state["fired"], messages.count(ESCALATE_HINT), flush=True)
"""


@pytest.mark.logic
def test_a_sigint_landing_inside_the_logging_sinks_lock_logs_the_hint_on_another_frame() -> None:
    """A SIGINT inside `Queue.put` of a `QueueHandler`, mutex held, is held and its hint is still logged once."""
    proc = subprocess.run(
        [sys.executable, "-c", _SIGINT_INSIDE_A_LOGGING_SINK],
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        preexec_fn=limit_file_size,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "True 1"


_THREE_SHUTDOWNS = """
import logging, signal, threading
from senselab_core.runtime.clock import SystemClock
from senselab_core.runtime.worker_client import ESCALATE_HINT, _signals_deferred
from senselab_core.runtime.worker_pipes import Lines
logging.basicConfig(level=logging.WARNING)
hinted = threading.Event()
def log_hint():
    logging.getLogger("hint").warning(ESCALATE_HINT)
    hinted.set()
lines = Lines(SystemClock(), on_hint=log_hint)
for _ in range(3):
    try:
        with _signals_deferred(escalate=True, request_hint=lines.request_hint):
            signal.raise_signal(signal.SIGINT)
    except KeyboardInterrupt:
        pass
assert hinted.wait(10)
lines.close()
"""


@pytest.mark.logic
def test_the_escalation_hint_is_logged_once_per_process() -> None:
    """Three shutdowns in one process, each interrupted once, log the hint once."""
    proc = subprocess.run(
        [sys.executable, "-c", _THREE_SHUTDOWNS],
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        preexec_fn=limit_file_size,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stderr.count(ESCALATE_HINT) == 1


@pytest.mark.contract
def test_ctrl_c_twice_kills_reaps_and_releases(tmp_path: Path) -> None:
    """A second SIGINT while a SIGTERM-ignoring group is stopped SIGKILLs the group before anything else is sent.

    The group is reaped and the files are removed before `KeyboardInterrupt` surfaces.
    """
    case = {
        "path": "timeout",
        "boundary": "module:_members_gone",
        "kind": "double_sigint",
        "occurrence": 1,
    }
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["fired"] and report["outcome"] == "KeyboardInterrupt"
    assert report["at_outcome"] == shutdown_schedule_fixture.CLEAN
    after = report["signals"][report["signals"].index("SIGINT2") + 1 :]
    assert after[:1] == ["SIGKILL"], f"the first signal sent after the second SIGINT was not SIGKILL: {after}"
    assert shutdown_schedule_fixture.violations(report) == []


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


@pytest.mark.contract
@pytest.mark.parametrize("boundary", ["client:_fail", "client:_shutdown", "client:_steps", "client:_acquire_shutdown"])
def test_a_close_on_another_thread_that_takes_the_timed_out_stop_is_driven(tmp_path: Path, boundary: str) -> None:
    """`close()` on another thread before the timed-out run takes the stop lock: that thread's waits are driven."""
    case = {"path": "timeout", "boundary": boundary, "kind": "close_other_thread", "occurrence": 1}
    [report] = shutdown_schedule_fixture.run_schedule(tmp_path, [case])
    assert report["signals"][:2] == ["SIGTERM", "SIGCONT"] and "SIGKILL" in report["signals"]
    assert shutdown_schedule_fixture.violations(report) == []
