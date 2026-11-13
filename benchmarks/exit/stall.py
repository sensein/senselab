"""The no-progress watchdog every coordinator step runs under.

`watched` runs a command, copies its output to a log and to stdout, and reads the committed count from the log's last
`progress: N` line (`exitlib.progress_line`). The command is stopped, and `Watched.stalled` names why, once that count
has not changed for `stall_s` seconds and no Slurm job of the run is queued or running. A `squeue` that fails is not
an empty queue: it is no evidence, and the command is left to run.
"""

import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import exitlib

PROGRESS = re.compile(r"progress: (\d+)")
EXIT_STALLED = 3
"""The exit status of a command stopped for want of progress."""
TAIL_BYTES = 65536
"""How much of the end of the log is read for the last progress line."""


def last_committed(log: Path) -> int | None:
    """Return the count of the last `progress: N` line in the end of `log`, None when it has none."""
    if not log.is_file():
        return None
    with log.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(handle.tell() - TAIL_BYTES, 0))
        found = PROGRESS.findall(handle.read().decode("utf-8", errors="replace"))
    return int(found[-1]) if found else None


def has_live_jobs(prefix: str, timeout_s: float, run: Callable[[list[str], float], str] | None = None) -> bool | None:
    """Return whether this user has a live Slurm job named `prefix*`, None when `squeue` could not be read."""
    try:
        return bool(exitlib.slurm_job_ids(prefix, timeout_s, run or exitlib.squeue))
    except (subprocess.SubprocessError, OSError):
        return None


def run_prefixes(runs_glob: str, profile: Path) -> Callable[[], list[str]]:
    """Return a function giving the Slurm job-name prefix of each run directory matching `runs_glob`, as it stands.

    A run directory is named by its plan hash. The prefix is `exitlib.slurm_prefix`, the one the run's own jobs carry,
    so another run's jobs under the same login are not this run's.
    """
    loaded = exitlib.load_profile(str(profile))

    def prefixes() -> list[str]:
        return [
            exitlib.slurm_prefix(loaded, Path(r), Path(r).name)
            for r in sorted(glob.glob(runs_glob))
            if Path(r).is_dir()
        ]

    return prefixes


def _any_live(prefixes: Sequence[str], timeout_s: float, live: Callable[[str, float], bool | None]) -> bool | None:
    """Return True when a job of any prefix is live, else None when any listing failed, else False."""
    answers = [live(prefix, timeout_s) for prefix in prefixes]
    return True if True in answers else (None if None in answers else False)


@dataclass
class Stall:
    """The rule: the committed count unchanged for `stall_s` seconds, and the run's queue read and empty."""

    stall_s: float
    prefix: str
    began: float
    committed: int | None = None

    def observe(self, now: float, committed: int | None, live: Callable[[], bool | None]) -> str | None:
        """Return the verdict once the run has stalled, else None; `live` is asked only when the count has been idle."""
        if committed != self.committed:
            self.committed, self.began = committed, now
            return None
        idle = now - self.began
        if idle < self.stall_s or live() is not False:
            return None
        return (
            f"committed unchanged at {'none' if self.committed is None else self.committed} for {idle:.0f} s "
            f"and no job named {self.prefix} in squeue"
        )


@dataclass(frozen=True)
class Watched:
    """How a watched command ended: its exit status, and the stall that stopped it if one did."""

    status: int
    stalled: str | None


def _pump(stream: "subprocess.Popen[str]", log: Path) -> None:
    """Copy the lines of `stream`'s output to `log` and to stdout until it closes."""
    assert stream.stdout is not None  # noqa: S101 - opened with stdout=PIPE
    with log.open("a", encoding="utf-8") as sink:
        for line in stream.stdout:
            sink.write(line)
            sink.flush()
            sys.stdout.write(line)
            sys.stdout.flush()


def _stop(process: "subprocess.Popen[str]", grace_s: float) -> None:
    """Terminate `process`'s group, then kill it if it outlives `grace_s`."""
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def watched(  # noqa: PLR0913 - the command, its log, the rule, then the three clocks and probes
    argv: Sequence[str],
    log: Path,
    *,
    stall_s: float,
    poll_s: float,
    prefix: str,
    timeout_s: float,
    live: Callable[[str, float], bool | None] | None = None,
    prefixes: Callable[[], list[str]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    grace_s: float = 30.0,
) -> Watched:
    """Run `argv` with its output teed to `log`; stop it when `Stall` says the run has no progress."""
    process = subprocess.Popen(  # noqa: S603 - the caller's command
        list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
    )
    pump = threading.Thread(target=_pump, args=(process, log), name="stall-pump", daemon=True)
    pump.start()
    rule = Stall(stall_s, f"{prefix}*", clock())
    stalled = None
    while process.poll() is None and stalled is None:
        time.sleep(poll_s)
        current = (prefixes() if prefixes else []) or [prefix]
        rule.prefix = " or ".join(f"{p}*" for p in current)
        stalled = rule.observe(
            clock(), last_committed(log), lambda: _any_live(current, timeout_s, live or has_live_jobs)
        )
    if stalled is not None:
        print(f"WATCHDOG: {stalled}; stopping the command", file=sys.stderr, flush=True)
        _stop(process, grace_s)
    status = process.wait()
    pump.join()
    return Watched(EXIT_STALLED if stalled is not None else status, stalled)


def write_stall_verdict(out: Path, stalled: str) -> None:
    """Write the failing verdict of a stalled step to `out`, and print it."""
    document = {"pass": False, "failed": [f"no progress: {stalled}"], "checks": 1}
    out.write_text(json.dumps(document) + "\n", encoding="utf-8")
    print(json.dumps(document), flush=True)
