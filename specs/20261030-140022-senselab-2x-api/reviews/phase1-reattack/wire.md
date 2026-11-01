# H8c surface W findings (commit b42d3cf0, macOS arm64, Python 3.12.11)

Probe scripts: <scratch>/h8c-wire/probes/ (p1..p10, bk.py backend).
Each ran against a detached worktree of b42d3cf0 with `uv run`. The worktree is removed.

## XW-1 (high): a second SIGINT during shutdown orphans the worker's process group
Invariant: wire-protocol.md "Timeouts and shutdown": every host fault stops the group within SHUTDOWN_BOUND_S; no orphaned descendants.
Cause: `_shutdown` sets `_terminated = True` before `_terminate` runs, and `_terminate` only guards `except Exception`.
A `KeyboardInterrupt` raised inside `_terminate` (for example in `_members_gone`) leaves the group unkilled. The retry in `run()`'s `finally` then skips `_terminate`, because `_terminated` is already True. It does release the arrays.
Repro (probes/p1.py):
```
c = WorkerClient(sys.executable, "bk", init={"ignore_term": True}, request_timeout=1.0)   # backend traps SIGTERM
Timer(3.0, lambda: os.kill(os.getpid(), SIGINT)).start()
c.run([{"id":"a","inputs":{}}], {"mode":"sleep"})   # TimeoutError at 1 s, then shutdown begins; SIGINT at 3 s
```
Observed: KeyboardInterrupt at t=3.0 s. After it, `os.killpg(pid, 0)` succeeds: the group is still alive and keeps running. `c._closed` is True and the array dir is removed, so nothing will ever clean it up.
Expected: the group is SIGKILLed, or `_terminated` is cleared and the group is killed before the exception propagates.
Not a regression of a closed A-row (not found among the pinned tests).

## XW-2 (medium): `close()` re-entered on the same thread deadlocks (non-reentrant `_shutdown_lock`)
Invariant: `close()` "Safe to call twice", returns within SHUTDOWN_BOUND_S; client docstring.
Repro (probes/p2.py): a SIGUSR1 handler calls `c.close()`. The signal arrives while the main thread is inside `close()` and the worker traps SIGTERM, so `_terminate` is held up to 10 s.
Observed: the handler blocks forever on `with self._shutdown_lock` (worker_client.py:612, faulthandler traceback) and the main thread cannot proceed. Only a watchdog `os._exit` ended it, at 25 s against the 17 s bound.
Realistic trigger: a service whose SIGTERM handler calls `client.close()` receives a second SIGTERM mid-close.
Expected: an RLock, or a signal-safe flag.

## XW-3 (medium): close() from another thread during `run()`'s encode phase makes `run` raise ValueError, not WorkerLost
Invariant: "a run in flight then raises WorkerLost" (close() doc and spec); typed infrastructure faults only.
Repro (probes/p5.py): thread A runs an item with a 600 MB array (to_wire takes more than 1 s); thread B calls `c.close()` at 0.4 s.
Observed: A raises `ValueError: I/O operation on closed file`, from `self._proc.stdin.fileno()` in `_send` after `_terminate` closed stdin. The fault table classifies ValueError as content, so an infrastructure loss reads as bad input.
Related hazard: `_send` caches the integer fd across `os.write`. If close lands between `fileno()` and the write, the fd number can be recycled and a request line written to an unrelated descriptor. I did not reproduce the recycling.
Expected: `_send` checks `_closed` and `WorkerLost`, and maps ValueError/EBADF to WorkerLost.

## XW-4 (low): worker exit code is lost in two cases
Invariant: "EOF becomes WorkerLost with the return code".
a) Worker dead while idle (probes/p4.py: SIGKILL, then `run`): `_send` gets EPIPE and raises `WorkerLost("cannot write to the worker")` with `returncode=None`, although `_exit_status == [-9]` is known.
b) Worker dies before hello (probes/p3.py, `exit 7` wrapper): `ProtocolMismatch("... (wire, runtime) = None")`. It is not WorkerLost and carries no code. Spec text arguably allows the type ("none arrives"), but the exit code and stderr cause are dropped.
c) macOS kqueue race (probes/p8.py): a leader already exited more than about 10 ms before the reaper registers `NOTE_EXIT` gives `_wait_exit` -> None, so there is no code. 20 of 20 trials with a delay of 0.01 s or more returned None.

## XW-5 (low): the reader does not enforce canonical `.npy` or valid-Unicode strings
Spec "The file is canonical .npy ... dtype always <… or |…"; the `to_wire` Unicode rule.
- probes/p9.py: a sha-consistent big-endian (`>f4`) or Fortran-order file passes `from_wire`, and the array is returned `>f4` / F-contiguous.
- probes/p6.py: `parse_line` accepts `"\ud800"` lone surrogates in output keys and values. Only the writer refuses them, so a non-conforming worker hands the host strings that fail `.encode("utf-8")` downstream.
Only a non-conforming worker can trigger either.

## What held
- Depth envelope: inputs/params/outputs, with scalar, array and empty-list leaves, depths 50..69. The writer's refusal point matched the reader exactly. There was no fatal and no ProtocolError; the over-deep cases were per-item WireValueError (params: raised).
- Duplicate keys (including `a` vs `a`), NaN, Infinity, 1E400, trailing garbage, BOM and NUL are all refused by `parse_line`.
- 5 cycles each of ok, timeout, `sys.exit` in the backend, and external SIGKILL: no fd, thread or array-dir leak.
- Run concurrent with close from another thread: arrays released exactly once, no hang.
- Symlink, FIFO and directory array refs, and the TOCTOU on array swap (sha is computed over the bytes read), all fail only the affected item.
- Pump line-splitting at limit+1, partial last line discarded on the host, non-blocking send with slow reader: no defect found.
- `SystemExit` in `handle` kills the worker, but that is documented (pass-through).
