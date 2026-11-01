"""Run shutdown schedules in a process of their own: `python shutdown_schedule_fixture.py <array_parent> <cases json>`.

A case names a `path`, a `boundary` (a function or method the client calls), an `occurrence` and a `kind`.
Every boundary is wrapped to count its calls on the main thread; at the `occurrence`-th call of the named one,
`kind` is injected before the call proceeds. Paths: `init` constructs the client under injection, and
`init_lost` does so over a worker that exits during load (its grandchildren left running); `timeout`
runs a request in `deaf` mode (SIGTERM ignored, so the group needs SIGKILL) past its timeout; `close` closes
the idle client. A case of kind `count` injects nothing and reports the calls each boundary got, from which
the schedule is generated. The graces are shrunk to `GRACES` unless the case sets `real_graces`, and the
bound is derived from the graces in force. One JSON line per case reports what surfaced, the state at that
moment and after a final `close()`. `faulthandler` ends the process with a traceback if a case outlives
`CASE_BOUND_S`, so a deadlock fails the case instead of hanging the test.
"""

import contextlib
import faulthandler
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy
from lifecycle_probes_fixture import alive, group_alive, recorded_pids
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import DEFAULT_GRACES, ShutdownGraces, WorkerClient, WorkerLost, _Pump

RUNTIME_TESTS = Path(__file__).parent
BACKEND = "backends.forking_backend_fixture"
FAKE = "senselab_worker.fake"
REQUEST_TIMEOUT_S = 0.2
GRACES = ShutdownGraces(stop=0.3, term=0.1, kill=0.5, join=0.5, exit_code_wait=0.2)
REAL_GRACES = DEFAULT_GRACES
CASE_BOUND_S = 60.0
LOAD_ALLOWANCE_S = 5.0
"""Added to every derived case bound for interpreter spawn, imports and hello on a loaded host."""
CTRL_C_LOAD_ALLOWANCE_S = 3.0
"""Added to the derived Ctrl-C bound (request timeout plus the second SIGINT's delay) for the same."""
OTHER_THREAD_JOIN_S = 0.05
SECOND_SIGINT_AFTER_S = 0.2
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
ITEMS = [{"id": "a", "inputs": {}}]
TYPED: dict[str, set[str | None]] = {
    "init": {None, "WorkerLost"},
    "init_lost": {"WorkerLost"},
    "timeout": {"TimeoutError", "WorkerLost"},
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
    return {"module": worker_client, "client": WorkerClient, "pump": _Pump, "popen": subprocess.Popen}[where], name


def case_bound(case: dict[str, Any]) -> float:
    """How long a case may take: request timeout, exit-code wait, two shutdown passes and `LOAD_ALLOWANCE_S`."""
    graces = REAL_GRACES if case.get("real_graces") else GRACES
    timeout = REQUEST_TIMEOUT_S if case["path"] == "timeout" else 0.0
    return timeout + graces.exit_code_wait + 2 * graces.bound + LOAD_ALLOWANCE_S


def ctrl_c_bound() -> float:
    """How long the real-grace double Ctrl-C case may take: request timeout, second-SIGINT delay, allowance."""
    return REQUEST_TIMEOUT_S + SECOND_SIGINT_AFTER_S + CTRL_C_LOAD_ALLOWANCE_S


class _Injector:
    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.calls: dict[str, int] = dict.fromkeys(BOUNDARIES, 0)
        self.fired = False
        self.clients: list[WorkerClient] = []
        self.other: threading.Thread | None = None
        self.timer: threading.Timer | None = None

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
            if self.case.get("real_graces"):
                self.timer = threading.Timer(SECOND_SIGINT_AFTER_S, signal.raise_signal, (signal.SIGINT,))
                self.timer.start()
            else:
                signal.raise_signal(signal.SIGINT)
        elif kind == "sigterm":
            signal.raise_signal(signal.SIGTERM)
        elif kind.startswith("handler_"):
            signal.raise_signal(signal.SIGUSR1)
        elif kind == "close_same_thread":
            client.close()
        elif kind == "close_other_thread":
            self.other = threading.Thread(target=client.close, daemon=True)
            self.other.start()
            self.other.join(OTHER_THREAD_JOIN_S)


def _handlers(kind: str, clients: list[WorkerClient]) -> None:
    def on_term(signum: int, frame: object) -> None:
        clients[0].close()
        raise SystemExit(128 + signum)

    def on_usr1(signum: int, frame: object) -> None:
        if kind == "handler_raises_base":
            raise SystemExit(128 + signum)
        if kind == "handler_raises_exception":
            raise RuntimeError("raised by a SIGUSR1 handler")
        clients[0].close()

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


def run_case(case: dict[str, Any], parent: Path, index: int) -> dict[str, Any]:
    """Run one case and return its report."""
    graces = REAL_GRACES if case.get("real_graces") else GRACES
    pid_file = parent / f"pids-{index}"
    injector = _Injector(case)

    def build() -> None:
        timeout = REQUEST_TIMEOUT_S if case["path"] == "timeout" else None
        init = {"pid_file": str(pid_file), "exit_at_load": case["path"] == "init_lost"}
        _Capturing(
            injector.clients,
            sys.executable,
            BACKEND,
            init=init,
            array_parent=parent,
            request_timeout=timeout,
            graces=graces,
        )

    _handlers(case["kind"], injector.clients)
    reals = {boundary: getattr(*_owner(boundary)) for boundary in BOUNDARIES}
    outcome = None
    built = False
    started = time.monotonic()
    try:
        constructing = case["path"].startswith("init")
        if not constructing:
            build()
            started = time.monotonic()
        for boundary, real in reals.items():
            setattr(*_owner(boundary), injector.wrap(boundary, real))
        if constructing:
            build()
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
    elapsed = time.monotonic() - started
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if injector.timer is not None:
        injector.timer.cancel()
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
        "abandoned": case["path"].startswith("init") and not built,
        "client_open": client.alive,
        "at_outcome": _state(client, pid_file),
    }
    if injector.other is not None:
        injector.other.join(case_bound(case))
        report["other_thread_done"] = not injector.other.is_alive()
    client.close()
    report["after_close"] = _state(client, pid_file)
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
        if n <= counts[path][boundary]
    ]


def _bounded() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))


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
        preexec_fn=_bounded,
        check=False,
    )
    reports = [json.loads(line) for line in proc.stdout.splitlines()]
    assert proc.returncode == 0 and len(reports) == len(todo), (
        f"the driver ended after {len(reports)} of {len(todo)} cases (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
    )
    assert [p.name for p in parent.iterdir() if not p.name.startswith("pids-")] == []
    return reports


def counted(tmp_path: Path) -> dict[str, dict[str, int]]:
    """Per path, the calls each boundary gets when nothing is injected."""
    todo = [{"path": path, "boundary": "", "occurrence": 0, "kind": "count"} for path in PATHS]
    return {r["path"]: r["calls"] for r in run_schedule(tmp_path, todo)}


def violations(report: dict[str, Any]) -> list[str]:
    """What a case's report breaks: reach, typed outcome, bound, no live group under a closed client, clean ends.

    A constructor that raised hands the caller nothing to close, so its state must already be clean.
    """
    found = []
    if not report["fired"]:
        found.append("the boundary was never reached")
    allowed = TYPED[report["path"]] | {RAISES.get(report["kind"])}
    if report["kind"] == "handler_raises_exception":
        allowed |= {None}
    if report["outcome"] not in allowed:
        found.append(f"untyped outcome {report['outcome']}")
    if report["elapsed"] >= report["bound"]:
        found.append(f"took {report['elapsed']:.2f} s, bound {report['bound']:.2f} s")
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
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=Path(parent))
    closer = threading.Thread(target=client.close, daemon=True)

    def encoding(real: Callable[..., Any]) -> Callable[..., Any]:
        def encode(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if closer.ident is None:
                closer.start()
                end = time.monotonic() + wait_s
                while client.returncode is None and time.monotonic() < end:
                    time.sleep(0.01)
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
    """In this child: a worker that crashes mid-request while its exit notification is 0.3 s late.

    Returns the `WorkerLost` exit code and message and the client's exit code.
    """

    def late(real: Callable[..., Any]) -> Callable[..., Any]:
        def wait(pid: int) -> Any:  # noqa: ANN401
            time.sleep(0.3)
            return real(pid)

        return wait

    with _substituted("module:_wait_exit", late):
        client = WorkerClient(sys.executable, FAKE, init={"faults": {"crash_after": 0}}, array_parent=Path(parent))
        try:
            try:
                client.run([{"id": "a", "inputs": {}}], {})
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
