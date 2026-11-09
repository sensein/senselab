"""Run shutdown schedules in a process of their own: `python shutdown_schedule_fixture.py <array_parent> <cases json>`.

A case names a `path`, a `boundary` (a function or method the client calls), an `occurrence` and a `kind`.
Every boundary is wrapped to count its calls on the main thread; at the `occurrence`-th call of the named one,
`kind` is injected before the call proceeds. Paths: `init` constructs the client under injection, and
`init_lost` does so over a worker that exits during load (its grandchildren left running); `timeout`
runs a request in `deaf` mode (SIGTERM ignored, so the group needs SIGKILL) past its timeout; `close` closes
the idle client. A case of kind `count` injects nothing and reports the calls each boundary got, from which
the schedule is generated. The client runs on a `ManualClock` that a driver thread advances only while the
client is parked in a shutdown wait (or, on the `timeout` path, once the worker has gone deaf), so a case's
`elapsed` is clock time and its bound is derived from the default graces. One JSON line per case reports what
surfaced, the state at that moment, the order of the signals sent to the group and after a final `close()`.
`faulthandler` ends the process with a traceback if a case outlives `CASE_BOUND_S`, so a deadlock fails the
case instead of hanging the test.
"""

import contextlib
import faulthandler
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy
from bounded_probe_fixture import limit_file_size
from lease_clock_fixture import ManualClock
from lifecycle_probes_fixture import gate, recorded_pids
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import DEFAULT_GRACES, WorkerClient, WorkerLost
from senselab_core.runtime.worker_pipes import Pump
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.processes import alive, group_alive
from senselab_worker.fake import FAULT_KEY

PYTHON = source_tree_python()

RUNTIME_TESTS = Path(__file__).parent
BACKEND = "backends.forking_backend_fixture"
FAKE = "senselab_worker.fake"
REQUEST_TIMEOUT_S = 0.2
GRACES = DEFAULT_GRACES
CASE_BOUND_S = 60.0
STEP_S = 1.0
"""How far the driver advances the clock each time the client is parked in a wait it may advance."""
CLOCK_POLL_S = 0.0005
"""The real seconds a parked thread of the case's clock waits between looks."""
OTHER_THREAD_JOIN_S = 0.05
PATHS = ("init", "init_lost", "timeout", "close")
RAISES = {
    "sigint": "KeyboardInterrupt",
    "double_sigint": "KeyboardInterrupt",
    "sigterm": "SystemExit",
    "handler_raises_base": "SystemExit",
    "handler_raises_exception": "RuntimeError",
}
KINDS = (*RAISES, "handler_close", "close_same_thread", "close_other_thread")
"""`sigint` raises SIGINT; `double_sigint` raises it twice; `sigterm` raises SIGTERM, whose handler closes the
client and exits; `handler_close`, `handler_raises_base` and `handler_raises_exception` raise SIGUSR1, whose
handler closes the client, raises `SystemExit` or raises `RuntimeError`; `close_same_thread` calls `close()`
where the boundary is reached; `close_other_thread` starts `close()` on another thread."""
BOUNDARIES = (
    "module:to_wire",
    "module:_leader_held",
    "module:_signal_group",
    "module:_members_gone",
    "module:_wait_exit",
    "client:_start",
    "client:_handshake",
    "client:_send",
    "client:_receive",
    "client:_fail",
    "client:_shutdown",
    "client:_steps",
    "client:_acquire_shutdown",
    "client:_stop",
    "client:_terminate",
    "client:_write_stop",
    "client:_retire_channel",
    "client:_kill_group_now",
    "client:_remove_arrays",
    "client:_release",
    "pump:finish",
    "pump:close",
    "popen:wait",
)
POLLED = {"module:_wait_exit": 1}
"""Boundaries polled in timed loops, whose later calls vary with timing: the most occurrences injected into each."""
ITEMS = [{"id": "a", "inputs": {}}]
TYPED: dict[str, set[str | None]] = {
    "init": {None, "WorkerLost"},
    "init_lost": {"WorkerLost"},
    "timeout": {"WorkerTimeout", "WorkerLost"},
    "close": {None},
}
"""Per path, the outcomes it may end in besides the injected exception."""
CLEAN = {"group_alive": False, "grandchild_alive": False, "reaped": True, "array_dir_gone": True}


class _Capturing(WorkerClient):
    """A client that hands itself to `sink` before it starts, so a case reaches one whose constructor raises."""

    def __init__(self, sink: list[WorkerClient], *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        sink.append(self)
        super().__init__(*args, **kwargs)


def _owner(boundary: str) -> tuple[Any, str]:
    if boundary not in BOUNDARIES:
        raise ValueError(f"{boundary!r} is not a declared shutdown boundary")
    where, name = boundary.split(":")
    return {"module": worker_client, "client": WorkerClient, "pump": Pump, "popen": subprocess.Popen}[where], name


def case_bound(case: dict[str, Any]) -> float:
    """How much clock time a case may take: request timeout, exit-code wait and two shutdown passes."""
    timeout = REQUEST_TIMEOUT_S if case["path"] == "timeout" else 0.0
    return timeout + GRACES.exit_code_wait + 2 * GRACES.bound


class _Injector:
    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.calls: dict[str, int] = dict.fromkeys(BOUNDARIES, 0)
        self.fired = False
        self.clients: list[WorkerClient] = []
        self.other: threading.Thread | None = None
        self.second_sigint = False
        self.events: list[str] = []
        self.nested_acquires: int | None = None

    def nested_close(self, client: WorkerClient) -> None:
        """Close from inside the stop, and record how many shutdown locks that close tried to take."""
        before = self.calls["client:_acquire_shutdown"]
        client.close()
        self.nested_acquires = self.calls["client:_acquire_shutdown"] - before

    def on_parked(self) -> bool:
        """Called by the driver while the client is parked: raise the pending second SIGINT; return whether it did."""
        if not self.second_sigint:
            return False
        self.second_sigint = False
        self.events.append("SIGINT2")
        signal.raise_signal(signal.SIGINT)
        return True

    def wrap(self, boundary: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def hooked(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if threading.current_thread() is threading.main_thread():
                self.calls[boundary] += 1
                if boundary == self.case["boundary"] and self.calls[boundary] == self.case["occurrence"]:
                    self.fired = True
                    self.inject()
            return real(*args, **kwargs)

        return hooked

    def inject(self) -> None:
        kind = self.case["kind"]
        client = self.clients[0]
        if kind == "sigint":
            signal.raise_signal(signal.SIGINT)
        elif kind == "double_sigint":
            signal.raise_signal(signal.SIGINT)
            self.second_sigint = True
        elif kind == "sigterm":
            signal.raise_signal(signal.SIGTERM)
        elif kind.startswith("handler_"):
            signal.raise_signal(signal.SIGUSR1)
        elif kind == "close_same_thread":
            self.nested_close(client)
        elif kind == "close_other_thread":
            self.other = threading.Thread(target=client.close, daemon=True)
            self.other.start()
            self.other.join(OTHER_THREAD_JOIN_S)


def _handlers(kind: str, injector: _Injector) -> None:
    clients = injector.clients

    def on_term(signum: int, frame: object) -> None:
        clients[0].close()
        raise SystemExit(128 + signum)

    def on_usr1(signum: int, frame: object) -> None:
        if kind == "handler_raises_base":
            raise SystemExit(128 + signum)
        if kind == "handler_raises_exception":
            raise RuntimeError("raised by a SIGUSR1 handler")
        injector.nested_close(clients[0])

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGUSR1, on_usr1)


def _state(client: WorkerClient, pid_file: Path) -> dict[str, Any]:
    pgid = client._proc.pid if client._proc is not None else None
    return {
        "group_alive": pgid is not None and group_alive(pgid),
        "grandchild_alive": any(map(alive, recorded_pids(pid_file))),
        "reaped": pgid is None or client.returncode is not None,
        "array_dir_gone": not client.array_dir.exists(),
    }


def _drive(clock: ManualClock, injector: _Injector, may: Callable[[], bool], done: threading.Event) -> None:
    """Until `done`, advance `clock` by `STEP_S` each time the main thread is parked on it and `may` allows."""
    main = threading.main_thread()
    while not done.is_set():
        if may() and clock.is_parked(main):
            if not injector.on_parked():
                clock.advance(STEP_S)
        else:
            done.wait(clock.poll_s)


def _perform(
    case: dict[str, Any], injector: _Injector, build: Callable[[int], None], port: int, clock: ManualClock
) -> tuple[str | None, bool, float]:
    """Run the case's path under injection; return what surfaced, whether a constructor finished and the clock time."""
    reals = {boundary: getattr(*_owner(boundary)) for boundary in BOUNDARIES}
    outcome, built, started = None, False, clock.now()
    try:
        constructing = case["path"].startswith("init")
        if not constructing:
            build(port)
            started = clock.now()
        for boundary, real in reals.items():
            setattr(*_owner(boundary), injector.wrap(boundary, real))
        if constructing:
            build(port)
            built = True
        elif case["path"] == "timeout":
            injector.clients[0].run(ITEMS, {"mode": "deaf"})
        else:
            injector.clients[0].close()
    except BaseException as exc:  # noqa: BLE001 - the case reports whatever surfaced
        outcome = type(exc).__name__
    finally:
        for boundary, real in reals.items():
            setattr(*_owner(boundary), real)
    return outcome, built, clock.now() - started


def run_case(case: dict[str, Any], parent: Path, index: int) -> dict[str, Any]:
    """Run one case and return its report."""
    pid_file = parent / f"pids-{index}"
    injector = _Injector(case)
    clock = ManualClock(poll_s=CLOCK_POLL_S, hang_s=CASE_BOUND_S / 2, spends="park")
    gate_open = threading.Event()
    done = threading.Event()
    killpg = os.killpg

    def record_killpg(pgid: int, sig: int) -> None:
        if sig != 0:
            injector.events.append(signal.Signals(sig).name)
        killpg(pgid, sig)

    def may_advance() -> bool:
        stopping = bool(injector.clients) and bool(getattr(injector.clients[0], "_shutting_down", None))
        return stopping or gate_open.is_set()

    def build(port: int = 0) -> None:
        timeout = REQUEST_TIMEOUT_S if case["path"] == "timeout" else None
        init = {"pid_file": str(pid_file), "exit_at_load": case["path"] == "init_lost", "gate_port": port}
        _Capturing(
            injector.clients,
            PYTHON,
            BACKEND,
            init=init,
            array_parent=parent,
            request_timeout=timeout,
            graces=GRACES,
            clock=clock,
        )

    _handlers(case["kind"], injector)
    driver = threading.Thread(target=_drive, args=(clock, injector, may_advance, done), daemon=True)
    driver.start()
    os.killpg = record_killpg  # type: ignore[assignment]
    with contextlib.ExitStack() as stack:
        port = 0
        if case["path"] == "timeout":
            server = stack.enter_context(gate())
            port = server.port
            threading.Thread(target=lambda: (server.entered(), gate_open.set()), daemon=True).start()
        outcome, built, elapsed = _perform(case, injector, build, port, clock)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    client = injector.clients[0]
    report = {
        **case,
        "fired": injector.fired,
        "calls": injector.calls,
        "outcome": outcome,
        "elapsed": elapsed,
        "bound": case_bound(case),
        "signals": list(injector.events),
        "nested_acquires": injector.nested_acquires,
        "abandoned": case["path"].startswith("init") and not built,
        "client_open": client.alive,
        "at_outcome": _state(client, pid_file),
    }
    if injector.other is not None:
        injector.other.join(case_bound(case))
        report["other_thread_done"] = not injector.other.is_alive()
    client.close()
    report["after_close"] = _state(client, pid_file)
    done.set()
    os.killpg = killpg
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return report


def schedule(counts: dict[str, dict[str, int]], occurrences: range) -> list[dict[str, Any]]:
    """Every (kind, path, boundary, occurrence) the counted runs reached, for occurrences in `occurrences`."""
    return [
        {"path": path, "boundary": boundary, "occurrence": n, "kind": kind}
        for kind in KINDS
        for path in PATHS
        for boundary in BOUNDARIES
        for n in occurrences
        if n <= min(counts[path][boundary], POLLED.get(boundary, n))
    ]


def run_schedule(tmp_path: Path, todo: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run `todo` in a driver process bounded per case; return its reports, failing if it died."""
    parent = tmp_path / "parent"
    parent.mkdir(exist_ok=True)
    existing = os.environ.get("PYTHONPATH")
    env = {**os.environ, "PYTHONPATH": str(RUNTIME_TESTS) + (f"{os.pathsep}{existing}" if existing else "")}
    proc = subprocess.run(
        [sys.executable, str(Path(__file__)), str(parent), json.dumps(todo)],
        capture_output=True,
        text=True,
        env=env,
        timeout=CASE_BOUND_S * len(todo) + 60,
        preexec_fn=limit_file_size,
        check=False,
    )
    reports = [json.loads(line) for line in proc.stdout.splitlines()]
    assert proc.returncode == 0 and len(reports) == len(todo), (
        f"the driver ended after {len(reports)} of {len(todo)} cases (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
    )
    assert [p.name for p in parent.iterdir() if not p.name.startswith("pids-")] == []
    return reports


def counted(tmp_path: Path) -> dict[str, dict[str, int]]:
    """Per path, the calls each boundary gets when nothing is injected: the fewer of two runs, by boundary."""
    todo = [{"path": path, "boundary": "", "occurrence": 0, "kind": "count"} for path in PATHS for _ in range(2)]
    calls: dict[str, dict[str, int]] = {}
    for report in run_schedule(tmp_path, todo):
        seen = calls.setdefault(report["path"], report["calls"])
        calls[report["path"]] = {b: min(n, report["calls"][b]) for b, n in seen.items()}
    return calls


def violations(report: dict[str, Any]) -> list[str]:
    """What a case's report breaks: reach, typed outcome, bound, no live group under a closed client, clean ends.

    A constructor that raised hands the caller nothing to close, so its state must already be clean.
    """
    found: list[str] = []
    if not report["fired"]:
        found.append("the boundary was never reached")
    allowed = TYPED[report["path"]] | {RAISES.get(report["kind"])}
    if report["kind"] == "handler_raises_exception":
        allowed |= {None}
    if report["outcome"] not in allowed:
        found.append(f"untyped outcome {report['outcome']}")
    if report["elapsed"] >= report["bound"]:
        found.append(f"took {report['elapsed']:.2f} s of clock time, bound {report['bound']:.2f} s")
    if report["at_outcome"]["group_alive"] and not report["client_open"]:
        found.append("the group outlived the client when the outcome surfaced")
    if report["abandoned"] and report["at_outcome"] != CLEAN:
        found.append(f"a raising constructor left {report['at_outcome']}")
    if report.get("other_thread_done") is False:
        found.append("close() on the other thread did not return")
    if report["after_close"] != CLEAN:
        found.append(f"after close: {report['after_close']}")
    return found


@contextlib.contextmanager
def _substituted(boundary: str, wrapper: Callable[[Callable[..., Any]], Callable[..., Any]]) -> Iterator[None]:
    """Replace the declared `boundary` with `wrapper(real)` for the block; only ever called in a child process."""
    owner, name = _owner(boundary)
    real = getattr(owner, name)
    setattr(owner, name, wrapper(real))
    try:
        yield
    finally:
        setattr(owner, name, real)


def close_while_encoding(parent: str, wait_s: float) -> list[str]:
    """In this child: `close()` on another thread while `run` encodes its inputs; return what `run` raised."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=Path(parent))
    closer = threading.Thread(target=client.close, daemon=True)

    def encoding(real: Callable[..., Any]) -> Callable[..., Any]:
        def encode(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if closer.ident is None:
                closer.start()
                end = time.monotonic() + wait_s
                while client.returncode is None and time.monotonic() < end:
                    time.sleep(0.01)
                assert client.returncode is not None, "close did not reap the worker"
            return real(*args, **kwargs)

        return encode

    outcome: list[str] = []

    def call() -> None:
        try:
            client.run([{"id": "a", "inputs": {"x": numpy.arange(4.0)}}], {})
        except BaseException as exc:  # noqa: BLE001 - the caller inspects whatever surfaced
            outcome.append(type(exc).__name__)

    with _substituted("module:to_wire", encoding):
        runner = threading.Thread(target=call, daemon=True)
        runner.start()
        runner.join(wait_s + client.graces.bound)
        closer.join(client.graces.bound)
    assert not runner.is_alive() and not closer.is_alive()
    return outcome


def crash_with_a_late_exit_notification(parent: str) -> tuple[int | None, str, int | None]:
    """In this child: a worker that crashes mid-request while its exit notification is held until `_fail` starts.

    Returns the `WorkerLost` exit code and message and the client's exit code.
    """
    released = threading.Event()

    def late(real: Callable[..., Any]) -> Callable[..., Any]:
        def wait(pid: int, *, block: bool = True) -> Any:  # noqa: ANN401
            if block:
                assert released.wait(CASE_BOUND_S), "the client never reached the exit-code wait"
            return real(pid, block=block)

        return wait

    def releasing(real: Callable[..., Any]) -> Callable[..., Any]:
        def fail(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            released.set()
            return real(*args, **kwargs)

        return fail

    with _substituted("module:_wait_exit", late), _substituted("client:_fail", releasing):
        client = WorkerClient(PYTHON, FAKE, init={}, array_parent=Path(parent))
        try:
            try:
                client.run([{"id": "a", "inputs": {}}], {FAULT_KEY: {"crash_after": 0}})
            except WorkerLost as lost:
                return lost.returncode, str(lost), client.returncode
            raise AssertionError("the crash did not surface as WorkerLost")
        finally:
            client.close()


def main(argv: list[str]) -> None:
    """Run the cases in `argv[2]` (a JSON list), printing one JSON report per line."""
    parent = Path(argv[1])
    for index, case in enumerate(json.loads(argv[2])):
        faulthandler.dump_traceback_later(CASE_BOUND_S, exit=True)
        print(json.dumps(run_case(case, parent, index)), flush=True)
        faulthandler.cancel_dump_traceback_later()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv)
