"""Run shutdown schedules in a process of their own: `python shutdown_schedule_fixture.py <array_parent> <cases json>`.

Each case starts a `WorkerClient` over the forking backend, wraps one boundary (a function or method the
run or the shutdown calls) so that its `occurrence`-th call on the main thread first injects `kind`, and
then takes `path`: `timeout` runs a request in `deaf` mode (SIGTERM ignored, so the group needs SIGKILL)
past its timeout; `close` closes the idle client. One JSON line per case reports what surfaced and the
state at that moment, then the state after a final `close()`. `faulthandler` ends the process with a
traceback if a case outlives `CASE_BOUND_S`, so a deadlock is a failed case, not a hung test.
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
from senselab_core.runtime.worker_client import SHUTDOWN_BOUND_S, WorkerClient, _Pump

RUNTIME_TESTS = Path(__file__).parent
BACKEND = "backends.forking_backend_fixture"
REQUEST_TIMEOUT_S = 0.3
STOP_GRACE_S = 0.6
TERM_GRACE_S = 0.3
CASE_BOUND_S = 2 * SHUTDOWN_BOUND_S + 10
OTHER_THREAD_JOIN_S = 0.05
PATHS = ("timeout", "close")
KINDS = ("sigint", "sigterm", "handler_close", "close_same_thread", "close_other_thread")
"""`sigint` raises SIGINT; `sigterm` raises SIGTERM, whose handler closes the client and exits;
`handler_close` raises SIGUSR1, whose handler closes the client; `close_same_thread` calls `close()`
where the boundary is reached; `close_other_thread` starts `close()` on another thread."""
RUN_BOUNDARIES = ("module:to_wire", "client:_send", "client:_receive")
SHUTDOWN_BOUNDARIES = (
    "client:_shutdown",
    "client:_terminate",
    "client:_write_stop",
    "client:_retire_channel",
    "module:_leader_held",
    "module:_signal_group",
    "module:_members_gone",
    "client:_remove_arrays",
    "client:_release",
    "pump:finish",
)
UNREACHED = frozenset({("timeout", "client:_write_stop")})
"""A host fault skips the stop line."""
ITEMS = [{"id": "a", "inputs": {}}]


def cases() -> list[dict[str, Any]]:
    """Every (kind, path, boundary) whose boundary the path reaches: run boundaries only on `timeout`."""
    out = []
    for kind in KINDS:
        for boundary in RUN_BOUNDARIES:
            out.append({"path": "timeout", "boundary": boundary, "kind": kind, "occurrence": 1})
        for path in PATHS:
            for boundary in SHUTDOWN_BOUNDARIES:
                if (path, boundary) not in UNREACHED:
                    out.append({"path": path, "boundary": boundary, "kind": kind, "occurrence": 1})
    return out


def _owner(boundary: str) -> tuple[Any, str]:
    where, name = boundary.split(":")
    return {"module": worker_client, "client": WorkerClient, "pump": _Pump}[where], name


class _Injector:
    def __init__(self, case: dict[str, Any], client: WorkerClient) -> None:
        self.case = case
        self.client = client
        self.calls = 0
        self.fired = False
        self.other: threading.Thread | None = None

    def wrap(self, real: Callable[..., Any]) -> Callable[..., Any]:
        def hooked(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if threading.current_thread() is threading.main_thread():
                self.calls += 1
                if self.calls == self.case["occurrence"]:
                    self.fired = True
                    self.inject()
            return real(*args, **kwargs)

        return hooked

    def inject(self) -> None:
        kind = self.case["kind"]
        if kind == "sigint":
            signal.raise_signal(signal.SIGINT)
        elif kind == "sigterm":
            signal.raise_signal(signal.SIGTERM)
        elif kind == "handler_close":
            signal.raise_signal(signal.SIGUSR1)
        elif kind == "close_same_thread":
            self.client.close()
        elif kind == "close_other_thread":
            self.other = threading.Thread(target=self.client.close, daemon=True)
            self.other.start()
            self.other.join(OTHER_THREAD_JOIN_S)


def run_case(case: dict[str, Any], parent: Path, index: int) -> dict[str, Any]:
    """Run one case and return its report."""
    pid_file = parent / f"pids-{index}"
    client = WorkerClient(
        sys.executable,
        BACKEND,
        init={"pid_file": str(pid_file)},
        array_parent=parent,
        request_timeout=REQUEST_TIMEOUT_S if case["path"] == "timeout" else None,
    )
    pgid = client.hello["pid"]
    injector = _Injector(case, client)

    def on_term(signum: int, frame: object) -> None:
        client.close()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGUSR1, lambda signum, frame: client.close())
    owner, name = _owner(case["boundary"])
    real = getattr(owner, name)
    setattr(owner, name, injector.wrap(real))
    outcome = None
    started = time.monotonic()
    try:
        if case["path"] == "timeout":
            client.run(ITEMS, {"mode": "deaf"})
        else:
            client.close()
    except BaseException as exc:  # noqa: BLE001 - the case reports whatever surfaced
        outcome = type(exc).__name__
    finally:
        setattr(owner, name, real)
    elapsed = time.monotonic() - started
    report = {
        **case,
        "fired": injector.fired,
        "outcome": outcome,
        "elapsed": elapsed,
        "group_alive": group_alive(pgid),
        "client_open": client.alive,
    }
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    if injector.other is not None:
        injector.other.join(SHUTDOWN_BOUND_S)
        report["other_thread_done"] = not injector.other.is_alive()
    client.close()
    report["after_close"] = {
        "group_alive": group_alive(pgid),
        "grandchild_alive": any(map(alive, recorded_pids(pid_file))),
        "reaped": client.returncode is not None,
        "array_dir_gone": not client.array_dir.exists(),
    }
    return report


TYPED: dict[str, set[str | None]] = {
    "timeout": {"TimeoutError", "WorkerLost", "KeyboardInterrupt", "SystemExit"},
    "close": {None, "KeyboardInterrupt", "SystemExit"},
}


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


def violations(report: dict[str, Any]) -> list[str]:
    """What a case's report breaks: typed outcome, bound, no live group under a closed client, a clean end."""
    found = []
    if not report["fired"]:
        found.append("the boundary was never reached")
    if report["outcome"] not in TYPED[report["path"]]:
        found.append(f"untyped outcome {report['outcome']}")
    bound = SHUTDOWN_BOUND_S + (REQUEST_TIMEOUT_S if report["path"] == "timeout" else 0.0)
    if report["elapsed"] >= bound:
        found.append(f"took {report['elapsed']:.1f} s, bound {bound:.1f} s")
    if report["group_alive"] and not report["client_open"]:
        found.append("the group outlived the client when the outcome surfaced")
    if report.get("other_thread_done") is False:
        found.append("close() on the other thread did not return")
    after = report["after_close"]
    if after != {"group_alive": False, "grandchild_alive": False, "reaped": True, "array_dir_gone": True}:
        found.append(f"after close: {after}")
    return found


def main(argv: list[str]) -> None:
    """Run the cases named in `argv[2]` (a JSON list, or `all`), printing one JSON report per line."""
    worker_client.STOP_GRACE_S = STOP_GRACE_S
    worker_client.TERM_GRACE_S = TERM_GRACE_S
    parent = Path(argv[1])
    todo = cases() if argv[2] == "all" else json.loads(argv[2])
    for index, case in enumerate(todo):
        faulthandler.dump_traceback_later(CASE_BOUND_S, exit=True)
        print(json.dumps(run_case(case, parent, index)), flush=True)
        faulthandler.cancel_dump_traceback_later()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv)
