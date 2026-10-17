"""Long-lived venv workers: one load per identity, restart after a crash, separation, shutdown."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from senselab.utils import venv_worker
from senselab.utils.venv_worker import (
    VenvWorkerTimeout,
    serve_in_venv,
    shutdown_venv_workers,
    venv_worker_events,
    venv_worker_stats,
)

_SCRIPT = r"""
import os
import time
from pathlib import Path


def load(init):
    log = Path(init["load_log"])
    with log.open("a") as handle:
        handle.write("%d\n" % os.getpid())
    print("a library printing to stdout must not corrupt the reply channel")
    if init.get("fail_load"):
        raise ValueError("bad checkpoint")
    return {"tag": init["tag"], "calls": 0}


def handle(state, request):
    state["calls"] += 1
    print("noise during a request")
    if request.get("crash"):
        os._exit(3)
    if request.get("raise"):
        raise ValueError("bad input " + request["raise"])
    if request.get("sleep"):
        time.sleep(request["sleep"])
    return {"tag": state["tag"], "x2": request["x"] * 2, "pid": os.getpid(), "calls": state["calls"]}
"""


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    shutdown_venv_workers()
    venv_worker_events(clear=True)
    yield
    shutdown_venv_workers()
    venv_worker_events(clear=True)


def _call(tmp_path: Path, request: Dict[str, Any], tag: str = "a", **overrides: Any) -> Dict[str, Any]:  # noqa: ANN401
    init = {"load_log": str(tmp_path / f"loads-{tag}.txt"), "tag": tag, **overrides.pop("init", {})}
    return serve_in_venv(
        ("test-venv", f"model-{tag}", "0" * 40, "cpu", "float32"),
        python=sys.executable,
        script=_SCRIPT,
        init=init,
        request=request,
        env=dict(os.environ),
        label=f"test worker {tag}",
        load_timeout_s=60,
        request_timeout_s=overrides.pop("request_timeout_s", 60),
    )


def _loads(tmp_path: Path, tag: str = "a") -> list[str]:
    log = tmp_path / f"loads-{tag}.txt"
    return log.read_text().split() if log.exists() else []


def test_one_load_serves_many_calls(tmp_path: Path) -> None:
    """N calls for one identity load once, in one process, and the state persists between them."""
    replies = [_call(tmp_path, {"x": i}) for i in range(5)]
    assert [r["x2"] for r in replies] == [0, 2, 4, 6, 8]
    assert len({r["pid"] for r in replies}) == 1
    assert [r["calls"] for r in replies] == [1, 2, 3, 4, 5]
    assert len(_loads(tmp_path)) == 1
    [stats] = venv_worker_stats()
    assert stats["served"] == 5 and stats["identity"][:2] == ["test-venv", "model-a"]
    assert "torch_threads" in stats["threads"] or "affinity" in stats["threads"]


def test_identities_are_served_by_separate_workers(tmp_path: Path) -> None:
    """A different model identity, or a different init, gets its own process and its own load."""
    a = _call(tmp_path, {"x": 1}, tag="a")
    b = _call(tmp_path, {"x": 1}, tag="b")
    assert a["tag"] == "a" and b["tag"] == "b" and a["pid"] != b["pid"]
    again = _call(tmp_path, {"x": 1}, tag="a")
    assert again["pid"] == a["pid"]
    assert len(_loads(tmp_path, "a")) == 1 and len(_loads(tmp_path, "b")) == 1


def test_a_crash_during_a_request_raises_and_the_next_call_restarts(tmp_path: Path) -> None:
    """A worker that dies mid-request fails that call, is recorded, and is replaced on the next one."""
    first = _call(tmp_path, {"x": 1})
    with pytest.raises(RuntimeError, match="exited during request"):
        _call(tmp_path, {"x": 1, "crash": True})
    after = _call(tmp_path, {"x": 2})
    assert after["pid"] != first["pid"] and after["x2"] == 4
    assert len(_loads(tmp_path)) == 2
    kinds = [e["event"] for e in venv_worker_events()]
    assert kinds == ["lost", "restart"]
    assert venv_worker_events()[0]["returncode"] == 3


def test_a_worker_killed_while_idle_is_restarted_with_an_event(tmp_path: Path) -> None:
    """A worker that died between calls is noticed before the next request and started again."""
    first = _call(tmp_path, {"x": 1})
    os.kill(first["pid"], 9)
    worker = next(iter(venv_worker._POOL.values()))
    worker._process.wait(timeout=10)  # type: ignore[union-attr]
    after = _call(tmp_path, {"x": 3})
    assert after["pid"] != first["pid"] and after["x2"] == 6
    assert [e["event"] for e in venv_worker_events()] == ["died_idle", "restart"]


def test_a_handler_error_raises_its_type_and_keeps_the_worker(tmp_path: Path) -> None:
    """A request the handler rejects is reported as the one-shot path did; the model stays loaded."""
    first = _call(tmp_path, {"x": 1})
    with pytest.raises(ValueError, match="bad input q"):
        _call(tmp_path, {"x": 1, "raise": "q"})
    after = _call(tmp_path, {"x": 1})
    assert after["pid"] == first["pid"] and len(_loads(tmp_path)) == 1
    assert venv_worker_events() == []


def test_a_load_failure_raises_its_type(tmp_path: Path) -> None:
    """A load that raises is the caller's error, not a hang or a generic failure."""
    with pytest.raises(ValueError, match="bad checkpoint"):
        _call(tmp_path, {"x": 1}, init={"fail_load": True})
    assert venv_worker_stats() == []


def test_a_request_past_its_ceiling_kills_the_worker(tmp_path: Path) -> None:
    """A timed-out worker is killed, so a late reply can never be read as the next call's."""
    first = _call(tmp_path, {"x": 1})
    with pytest.raises(VenvWorkerTimeout):
        _call(tmp_path, {"x": 1, "sleep": 5}, request_timeout_s=0.5)
    after = _call(tmp_path, {"x": 5})
    assert after["pid"] != first["pid"] and after["x2"] == 10


def test_shutdown_ends_every_worker(tmp_path: Path) -> None:
    """Shutdown stops the processes; a later call starts a fresh one."""
    a = _call(tmp_path, {"x": 1}, tag="a")
    b = _call(tmp_path, {"x": 1}, tag="b")
    processes = [w._process for w in venv_worker._POOL.values()]
    shutdown_venv_workers()
    assert venv_worker_stats() == []
    assert all(p is not None and p.poll() is not None for p in processes)
    again = _call(tmp_path, {"x": 1}, tag="a")
    assert again["pid"] not in (a["pid"], b["pid"])


def test_close_by_label_ends_only_that_label(tmp_path: Path) -> None:
    """Closing one label stops its worker and leaves the others; the next call for it starts fresh."""
    a = _call(tmp_path, {"x": 1}, tag="a")
    b = _call(tmp_path, {"x": 1}, tag="b")
    assert venv_worker.close_venv_workers("test worker a") == 1
    assert [s["identity"][1] for s in venv_worker_stats()] == ["model-b"]
    assert _call(tmp_path, {"x": 1}, tag="b")["pid"] == b["pid"]
    assert _call(tmp_path, {"x": 1}, tag="a")["pid"] != a["pid"]
    assert venv_worker.close_venv_workers("no such label") == 0


class _CountingLock:
    """A worker lock that counts the threads that have asked for it."""

    def __init__(self, inner: Any) -> None:  # noqa: ANN401 — a threading.Lock
        self.inner, self.asked = inner, 0

    def __enter__(self) -> None:
        self.asked += 1
        self.inner.acquire()

    def __exit__(self, *exc: object) -> None:
        self.inner.release()


def test_two_callers_of_a_dead_idle_worker_share_one_replacement(tmp_path: Path) -> None:
    """Two calls queued on a worker that died idle start one replacement; the second is served by it."""
    first = _call(tmp_path, {"x": 1})
    os.kill(first["pid"], 9)
    worker = next(iter(venv_worker._POOL.values()))
    worker._process.wait(timeout=10)  # type: ignore[union-attr]
    gate = _CountingLock(worker.lock)
    worker.lock = gate  # type: ignore[assignment]
    replies: list[Dict[str, Any]] = []
    errors: list[BaseException] = []

    def call(x: int) -> None:
        try:
            replies.append(_call(tmp_path, {"x": x}))
        except BaseException as exc:  # noqa: BLE001 — reported by the assertion below
            errors.append(exc)

    gate.inner.acquire()
    threads = [threading.Thread(target=call, args=(x,)) for x in (2, 3)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 30
    while gate.asked < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gate.asked == 2
    gate.inner.release()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert sorted(r["x2"] for r in replies) == [4, 6]
    assert len({r["pid"] for r in replies}) == 1 and first["pid"] not in {r["pid"] for r in replies}
    assert len(_loads(tmp_path)) == 2 and len(venv_worker._POOL) == 1
    assert [e["event"] for e in venv_worker_events()] == ["died_idle", "restart"]


def test_importing_venv_worker_pulls_in_neither_torch_nor_the_data_structures() -> None:
    """The module logs through ``logging.getLogger``, so a lean host can import it without senselab's heavy deps."""
    heavy = ("torch", "senselab.utils.data_structures")
    code = f"import sys, senselab.utils.venv_worker; print([m for m in {heavy!r} if m in sys.modules])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "[]"
