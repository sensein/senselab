"""Step 4: interrupt `SpeechToText` on a Slurm pool at a share of its items, then resume it and account the work.

`interrupt_resume.py start --work DIR --out FILE` plans the `stt` step under the profile on `pool-slurm` and runs it in
a child process; once the config's share of the items has a result event the coordinator cancels every Slurm job named
by the run's prefix, SIGKILLs the child, cancels again whatever the child had submitted meanwhile, writes the report to
`FILE`, prints the hash and the committed count and exits 0. The interrupt depends on no controller.
`resume HASH --work DIR --out FILE` (a fresh process, after the interrupt) resumes the run and asserts from its events:
executions are the items not committed before the interrupt, every other item a reuse, and no model ref was resolved.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import exitlib  # noqa: E402 - beside this script
from senselab_core.runtime.execution.events import read_events, run_dir_of  # noqa: E402
from senselab_core.runtime.execution.run import RESULTS_DIR  # noqa: E402
from senselab_core.runtime.execution.substrates import worker_id  # noqa: E402
from senselab_core.runtime.revision import cache_root  # noqa: E402
from senselab_core.runtime.store import ResultStore  # noqa: E402

STEP = "stt"
EXECUTOR = "pool-slurm"


def scancel(jobs: Sequence[str]) -> None:
    """Cancel the Slurm jobs `jobs`."""
    subprocess.run(["scancel", *jobs], check=True)


def interrupt(  # noqa: PLR0913 - the prefix and process, then the three effects and the bound
    prefix: str,
    pid: int,
    timeout_s: float,
    poll_s: float,
    deadline: float | None,
    jobs: Callable[[str, float], list[str]],
    cancel: Callable[[Sequence[str]], None],
    kill: Callable[[int], None],
) -> list[str]:
    """Cancel the jobs named `prefix*`, kill process group `pid`, then cancel until none of the prefix is live.

    The second round takes jobs the killed process submitted after the first listing. A job already cancelled that
    is still listed (completing) is neither cancelled again nor counted again.

    Returns:
        Every job id cancelled.

    Raises:
        exitlib.BoundExceeded: jobs of the prefix are still live at `deadline` (a `time.monotonic()` value).
    """
    cancelled: list[str] = []

    def sweep() -> list[str]:
        live = jobs(prefix, timeout_s)
        fresh = [j for j in live if j not in cancelled]
        if fresh:
            cancel(fresh)
            cancelled.extend(fresh)
        return live

    sweep()
    kill(pid)
    while sweep():
        if deadline is not None and time.monotonic() >= deadline:
            raise exitlib.BoundExceeded(f"jobs named {prefix}* are still live after the interrupt")
        time.sleep(poll_s)
    return cancelled


def _plan(work: Path, profile_argument: str, config: exitlib.Config) -> tuple[Any, Any]:  # noqa: ANN401
    """Return the profile and the placed plan of the step under `work`."""
    profile = exitlib.profile_for(EXECUTOR, exitlib.load_profile(profile_argument), config)
    exitlib.isolate(work, STEP)
    return profile, exitlib.planned(config.steps[STEP], exitlib.build_tree(config.steps[STEP], work), profile)


def run_child(work: Path, profile_argument: str, config: exitlib.Config) -> int:
    """Plan and run the step to its end; the process `start` spawns, killed at the interrupt."""
    _, placed = _plan(work, profile_argument, config)
    run, _ = exitlib.launch(placed, EXECUTOR, ResultStore(cache_root() / RESULTS_DIR))
    while not run.wait(timeout=config.watch_poll_s).done:
        pass
    return 0


def spawn_child(work: Path, profile_argument: str) -> subprocess.Popen[bytes]:
    """Start `run_child` as a process of its own, leading its own process group."""
    argv = [sys.executable, str(Path(__file__).resolve()), "_run", "--work", str(work), "--profile", profile_argument]
    return subprocess.Popen(argv, start_new_session=True)  # noqa: S603 - this interpreter and script


def start(
    work: Path,
    out: Path,
    profile_argument: str,
    config: exitlib.Config,
    spawn: Callable[[Path, str], subprocess.Popen[bytes]] = spawn_child,
) -> int:
    """Run the step in a child until its share of the items is committed, then interrupt it and report.

    Raises:
        RuntimeError: the run is done before the share was committed, or not started as a Slurm job.
        BoundExceeded: the wall bound passed before the share was committed.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the coordinator must be a Slurm job: submit this command, do not run it on a login node")
    profile, placed = _plan(work, profile_argument, config)
    run_dir = run_dir_of(placed.hash)
    items = len(placed.manifest.rows)
    target = exitlib.interrupt_target(items, config.interrupt_share)
    bounds, poll = config.bounds[EXECUTOR], config.watch_poll_s
    prefix = exitlib.slurm_prefix(profile, run_dir, placed.hash)
    timeout = exitlib.slurm_timeout(profile, poll)
    child = spawn(work, profile_argument)
    deadline = None if bounds.wall_s is None else time.monotonic() + bounds.wall_s
    committed = 0
    try:
        while (committed := len(exitlib.committed_jobs(read_events(run_dir)))) < target:
            if child.poll() is not None:
                raise RuntimeError(
                    f"the run ended (exit {child.returncode}) with {committed} of {items} items committed, short of "
                    f"{target}: nothing to interrupt; {exitlib.failure_summary(read_events(run_dir))}"
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise exitlib.BoundExceeded(f"fewer than {target} of {items} items committed within {bounds.wall_s} s")
            time.sleep(poll)
        cancelled = interrupt(
            prefix,
            child.pid,
            timeout,
            poll,
            deadline,
            lambda each, seconds: exitlib.slurm_job_ids(each, seconds, exitlib.squeue),
            scancel,
            lambda each: os.killpg(each, signal.SIGKILL),
        )
        child.wait()
    except BaseException:
        _reap(child, prefix, timeout)
        raise
    exitlib.write_json(
        out,
        {"hash": placed.hash, "items": items, "committed": committed, "coordinator": worker_id(), "workers": cancelled},
    )
    print(f"INTERRUPTED hash={placed.hash} committed={committed}/{items} cancelled={' '.join(cancelled)}")
    return 0


def _reap(child: subprocess.Popen[bytes], prefix: str, timeout_s: float) -> None:
    """Kill the child's process group and cancel the jobs of `prefix` after a failed start."""
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()
    live = exitlib.slurm_job_ids(prefix, timeout_s, exitlib.squeue)
    if live:
        scancel(live)


def refuse_failures(results: Any) -> None:  # noqa: ANN401 - a ResultSet
    """Raise `exitlib.ResumeMismatch` when `results.failures` (a list) is not empty."""
    if results.failures:
        raise exitlib.ResumeMismatch(f"{len(results.failures)} items failed after the resume")


def resume(plan_hash: str, work: Path, out: Path, profile_argument: str) -> int:
    """Resume the run of `plan_hash` and assert its accounting; write it to `out`.

    Raises:
        exitlib.ResumeMismatch: the events or the run manifest contradict the check.
    """
    import senselab  # noqa: PLC0415 - imported after the cache root is set

    exitlib.isolate(work, STEP)
    run_dir = run_dir_of(plan_hash)
    profile = exitlib.load_profile(profile_argument)
    prefix = exitlib.slurm_prefix(profile, run_dir, plan_hash)
    live = exitlib.slurm_job_ids(
        prefix, exitlib.slurm_timeout(profile, exitlib.load_config().watch_poll_s), exitlib.squeue
    )
    if live:
        raise exitlib.ResumeMismatch(
            f"jobs of the run are still live ({' '.join(live)}); cancel every job named {prefix}*"
        )
    before = read_events(run_dir)
    resolved = exitlib.resolution_snapshot(cache_root())
    with senselab.use(execution=profile):
        results = senselab.resume(plan_hash)
    after = read_events(run_dir)
    items = len(results)
    accounting = exitlib.account_resume(before, after, items)
    exitlib.assert_no_resolution(resolved, exitlib.resolution_snapshot(cache_root()))
    refuse_failures(results)
    document = {"hash": plan_hash, **vars(accounting)}
    exitlib.write_json(out, document)
    print(
        f"resume {plan_hash}: {accounting.executions} executions (items not committed: {items - accounting.committed}, "
        f"late hits {accounting.late}), {accounting.reuses} reuses (items committed: {accounting.committed}), "
        "no ref resolved"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the step's command line."""
    parser = argparse.ArgumentParser(prog="interrupt_resume.py")
    commands = parser.add_subparsers(dest="command", required=True)
    starting = commands.add_parser("start", help="run until the share is committed; print the jobs to scancel")
    running = commands.add_parser("_run", help="the process `start` spawns")
    resuming = commands.add_parser("resume", help="resume a run and assert its executions and reuses")
    resuming.add_argument("plan_hash")
    for each in (starting, running, resuming):
        each.add_argument("--work", type=Path, required=True)
        each.add_argument("--out", type=Path, required=each is not running)
        each.add_argument("--profile", required=True, help="a packaged execution profile, or a profile file")
    args = parser.parse_args(argv)
    if args.command == "_run":
        return run_child(args.work, args.profile, exitlib.load_config())
    if args.command == "start":
        return start(args.work, args.out, args.profile, exitlib.load_config())
    return resume(args.plan_hash, args.work, args.out, args.profile)


if __name__ == "__main__":
    sys.exit(main())
