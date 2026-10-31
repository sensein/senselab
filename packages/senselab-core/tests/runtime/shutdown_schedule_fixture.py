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
from typing import Any, Callable

from lifecycle_probes_fixture import alive, group_alive, recorded_pids
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import WorkerClient, _Pump

RUNTIME_TESTS = Path(__file__).parent
BACKEND = "backends.forking_backend_fixture"
REQUEST_TIMEOUT_S = 0.2
GRACES = {"STOP_GRACE_S": 0.3, "TERM_GRACE_S": 0.1, "KILL_GRACE_S": 0.5, "JOIN_GRACE_S": 0.5, "EXIT_CODE_WAIT_S": 0.2}
REAL_GRACES = {name: getattr(worker_client, name) for name in GRACES}
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


def _owner(boundary: str) -> tuple[Any, str]:
    where, name = boundary.split(":")
    return {"module": worker_client, "client": WorkerClient, "pump": _Pump, "popen": subprocess.Popen}[where], name


def shutdown_bound(graces: dict[str, float]) -> float:
    """`SHUTDOWN_BOUND_S` under `graces`."""
    return graces["STOP_GRACE_S"] + graces["KILL_GRACE_S"] + graces["JOIN_GRACE_S"]


def case_bound(case: dict[str, Any]) -> float:
    """How long a case may take: request timeout, exit-code wait, two shutdown passes and `LOAD_ALLOWANCE_S`."""
    graces = REAL_GRACES if case.get("real_graces") else GRACES
    timeout = REQUEST_TIMEOUT_S if case["path"] == "timeout" else 0.0
    return timeout + graces["EXIT_CODE_WAIT_S"] + 2 * shutdown_bound(graces) + LOAD_ALLOWANCE_S


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
    for name, value in graces.items():
        setattr(worker_client, name, value)
    worker_client.SHUTDOWN_BOUND_S = shutdown_bound(graces)
    pid_file = parent / f"pids-{index}"
    injector = _Injector(case)
    real_init = WorkerClient.__init__

    def capture(self: WorkerClient, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        injector.clients.append(self)
        real_init(self, *args, **kwargs)

    def build() -> None:
        timeout = REQUEST_TIMEOUT_S if case["path"] == "timeout" else None
        init = {"pid_file": str(pid_file), "exit_at_load": case["path"] == "init_lost"}
        WorkerClient(sys.executable, BACKEND, init=init, array_parent=parent, request_timeout=timeout)

    _handlers(case["kind"], injector.clients)
    reals = {boundary: getattr(*_owner(boundary)) for boundary in BOUNDARIES}
    outcome = None
    built = False
    started = time.monotonic()
    try:
        WorkerClient.__init__ = capture  # type: ignore[method-assign]
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
        WorkerClient.__init__ = real_init  # type: ignore[method-assign]
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
