"""The ``data`` group and the ``status``, ``explain``, ``coordinate`` and ``execute`` commands of ``senselab.cli``."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from senselab_core.kernel.placement import PlanPlacement
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.runtime.execution.epoch import StaleCoordinator
from senselab_core.runtime.execution.events import Event, runs_root
from senselab_core.runtime.execution.executor import QueueHeld
from senselab_core.runtime.execution.pool import ExternalWorkers, coordinate, run_store
from senselab_core.runtime.execution.pool_worker import serve_pool
from senselab_core.runtime.execution.run import PLACEMENTS_DIR, PLAN_FILE, Run, latest_placement
from senselab_core.runtime.explain import node_line
from senselab_core.runtime.fetch import DataManifest, FetchError, ManifestError, default_cache_root, fetch
from senselab_core.runtime.profiles import ExecutionProfile


def mount_data(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add the ``data`` command group to a parser's subparsers."""
    data = groups.add_parser("data", help="versioned datasets")
    commands = data.add_subparsers(dest="command", required=True)
    fetch_cmd = commands.add_parser("fetch", help="download and verify the files of a data manifest")
    fetch_cmd.add_argument("manifest", type=Path, help="path to a data-manifest@2.0 YAML file")
    fetch_cmd.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="cache directory (default: $SENSELAB_DATA_CACHE or ~/.cache/senselab/data)",
    )
    fetch_cmd.add_argument(
        "--verify", action="store_true", help="re-hash every cached file instead of checking its size"
    )
    data.set_defaults(run=run_data)


def run_data(args: argparse.Namespace) -> int:
    """Run a parsed ``data fetch``; return the exit code (0 on success, 1 on a reported error)."""
    try:
        manifest = DataManifest.load(args.manifest)
        paths = fetch(manifest, cache_root=args.cache or default_cache_root(), verify="full" if args.verify else "size")
    except (ManifestError, FetchError) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1
    except OSError as err:
        cause = " ".join(str(err).split())
        print(
            f"senselab: error: {cause}; check the paths exist and the cache directory is writable (--cache DIR)",
            file=sys.stderr,
        )
        return 1
    for name, path in paths.items():
        print(f"{name}\t{path}")
    return 0


def _run_directory(name: str) -> Path:
    """Return the run directory `name` names: a directory, or a plan hash or its unique prefix.

    Raises:
        FileNotFoundError: no run matches, or a prefix matches several.
    """
    if Path(name).is_dir():
        return Path(name)
    root = runs_root()
    listed = sorted(root.iterdir()) if root.is_dir() else []
    found = [p for p in listed if p.is_dir() and name and p.name.startswith(name)]
    if len(found) != 1:
        raise FileNotFoundError(f"{'no run' if not found else 'several runs'} match {name!r} under {root}")
    return found[0]


def _tail(text: str, lines: int = 10) -> str:
    return "\n".join(text.splitlines()[-lines:])


def run_status(args: argparse.Namespace) -> int:
    """Run a parsed ``status``: the jobs of a run by state, its faults and its broken envs, from its events."""
    try:
        run = Run(_run_directory(args.run_ref))
        status = run.status()
    except (OSError, ValueError) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1
    state = "cancelled" if status.cancelled else "done" if status.done else "not done"
    print(f"run {run.manifest().get('plan_hash', run.run_dir.name)}: {state}; last event {status.last_event_ns}")
    print("jobs: " + ", ".join(f"{k} {v}" for k, v in status.counts.items()))
    print("faults: " + (", ".join(f"{k} {v}" for k, v in sorted(status.faults.items())) or "none"))
    for broken in status.broken:
        print(f"broken env {broken.env}: {broken.failures} failures; {broken.fault.type}: {broken.fault.message}")
    for node, counts in status.nodes.items():
        print(f"node {node}: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    return 0


def _event_line(event: Event) -> str:
    element = "" if event.job is None else str(list(event.job.element))
    fault = "" if event.fault is None else f" fault {event.fault}"
    key = "" if event.key is None else f" key {event.key[:12]}"
    return f"  {event.t_ns} {event.kind} {event.node}{element} attempt {event.attempt} seq {event.seq}{key}{fault}"


def _placement(run_dir: Path, events: list[Event]) -> PlanPlacement:
    """Return the placement of the attempt that wrote the newest of `events`, else the latest attempt's."""
    for event in reversed(events):
        path = run_dir / PLACEMENTS_DIR / f"{event.attempt}.json"
        if path.is_file():
            return PlanPlacement.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return latest_placement(run_dir)


def run_explain(args: argparse.Namespace) -> int:
    """Run a parsed ``explain``: why a node or key failed, was skipped, hit the cache or was placed where it was."""
    try:
        run = Run(_run_directory(args.run_ref))
        document = json.loads((run.run_dir / PLAN_FILE).read_text(encoding="utf-8"))
        logical = LogicalPlan.model_validate(document["logical"])
        events = run.events()
    except (OSError, ValueError, KeyError) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1
    tasks = {n.id: n.task for n in logical.graph.nodes}
    matched = [e for e in events if e.node == args.target or (e.key is not None and e.key.startswith(args.target))]
    nodes = sorted({e.node for e in matched} | ({args.target} if args.target in tasks else set()))
    if not nodes:
        print(f"senselab: error: no node or key {args.target!r} in run {run.run_dir.name}", file=sys.stderr)
        return 1
    for node in nodes:
        shown = [e for e in matched if e.node == node] or [e for e in events if e.node == node]
        placement = _placement(run.run_dir, shown)
        print(node_line(node, tasks[node], placement) if node in placement.nodes else f"node {node}")
        print(f"events ({len(shown)}):")
        for event in shown:
            print(_event_line(event))
        for event in (e for e in shown if e.failure is not None):
            assert event.failure is not None
            print(f"fault {event.failure.fault}: {event.failure.type}")
            print(_tail(event.failure.message))
        for event in (e for e in shown if e.kind in ("blocked", "cancelled")):
            why = "an upstream job failed" if event.kind == "blocked" else "the run was cancelled before it was sent"
            print(f"{event.kind} {event.node}{list(event.job.element) if event.job else ''}: {why}")
    return 0


def run_coordinate(args: argparse.Namespace) -> int:
    """Run a parsed ``coordinate``: take the run's next epoch and send its jobs to its queues, for pool workers."""
    try:
        profile = ExecutionProfile.load(args.profile) if args.profile else None
        outcome = coordinate(_run_directory(args.run_ref), substrate=ExternalWorkers(), profile=profile)
    except (OSError, ValueError, StaleCoordinator, QueueHeld) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1
    blocked = sum(len(e) for e in outcome.blocked.values())
    failed = sum(r.failure is not None for refs in outcome.results.values() for r in refs)
    print(f"run {outcome.plan_hash}: {'done' if outcome.done else 'not done'}; {failed} failed, {blocked} blocked")
    return 0 if outcome.done else 1


def run_execute(args: argparse.Namespace) -> int:
    """Run a parsed ``execute``: serve a run's queues of the named resource classes as one pool worker."""
    pool = Path(args.pool)
    classes = [c for c in args.classes.split(",") if c]
    try:
        profile = ExecutionProfile.load(args.profile) if args.profile else ExecutionProfile.load()
        return serve_pool(pool, classes, profile=profile, store=run_store(pool.parent))
    except (OSError, ValueError) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1


def mount_runs(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add the ``status``, ``explain``, ``coordinate`` and ``execute`` commands to a parser's subparsers."""
    status = groups.add_parser("status", help="the jobs, faults and broken envs of a run")
    status.add_argument(
        "run_ref",
        metavar="run",
        help="a run directory, or a plan hash (or its unique prefix) under $SENSELAB_CACHE/2x/runs",
    )
    status.set_defaults(run=run_status)
    explain = groups.add_parser("explain", help="the events, fault and placement of a node or key of a run")
    explain.add_argument("run_ref", metavar="run", help="a run directory, or a plan hash (or its unique prefix)")
    explain.add_argument("target", help="a node id, or a key digest or its prefix")
    explain.set_defaults(run=run_explain)
    coordinating = groups.add_parser("coordinate", help="coordinate a run on its queues, as the next epoch")
    coordinating.add_argument("run_ref", metavar="run", help="a run directory, or a plan hash (or its unique prefix)")
    coordinating.add_argument("--profile", default=None, help="the execution profile the run was placed under")
    coordinating.set_defaults(run=run_coordinate)
    executing = groups.add_parser("execute", help="serve a run's queues as one pool worker")
    executing.add_argument("--pool", required=True, help="the run's queues directory, <run dir>/queues")
    executing.add_argument("--classes", required=True, help="the resource classes to serve, comma-separated")
    executing.add_argument("--profile", default=None, help="the execution profile the run was placed under")
    executing.set_defaults(run=run_execute)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``data`` group and the ``status``, ``explain``, ``coordinate`` and ``execute`` commands."""
    parser = argparse.ArgumentParser(prog="senselab")
    groups = parser.add_subparsers(dest="group", required=True)
    mount_data(groups)
    mount_runs(groups)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
