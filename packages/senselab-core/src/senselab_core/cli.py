"""The ``data`` group and the ``status``, ``plan``, ``explain`` and ``run`` commands."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from senselab_core.kernel.identity import ReusePolicy
from senselab_core.kernel.placement import PlanPlacement
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.events import UNPLACEABLE, Event, placements, runs_root
from senselab_core.runtime.execution.executor import FaultRecord, QueueHeld
from senselab_core.runtime.execution.provenance_check import check_provenance
from senselab_core.runtime.execution.run import (
    PLACEMENTS_DIR,
    PLAN_FILE,
    Run,
    latest_placement,
    load_plan,
    read_placement,
    stored_policy,
)
from senselab_core.runtime.explain import node_line
from senselab_core.runtime.fetch import DataManifest, FetchError, ManifestError, default_cache_root, fetch
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.site_placement import PlacementHookFailed, load_overrides


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
        "--profile", default=None, help="the execution profile whose retry policy and fetch timeout apply"
    )
    fetch_cmd.add_argument(
        "--verify", action="store_true", help="re-hash every cached file instead of checking its size"
    )
    data.set_defaults(run=run_data)


def run_data(args: argparse.Namespace) -> int:
    """Run a parsed ``data fetch``; return the exit code (0 on success, 1 on a reported error)."""
    try:
        manifest = DataManifest.load(args.manifest)
        profile = _profile(args.profile)
        paths = fetch(
            manifest,
            cache_root=args.cache or default_cache_root(),
            verify="full" if args.verify else "size",
            retry=profile.retry,
            network=profile.network,
            timeout_s=profile.timeouts.fetch_s,
        )
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


def _placement_lines(events: list[Event], resource_class: str | None = None) -> list[str]:
    """Return one line per class placement in `events` (of `resource_class` when given): values, layers, verdict."""
    lines = []
    for event in placements(events):
        found = event.placement
        if found is None or resource_class not in (None, found.resource_class):
            continue
        if found.closing is not None:
            asked = found.request
            lines.append(
                f"placement {found.resource_class} epoch {event.epoch}: closed; "
                f"wall_s {found.closing.wall_s} (requested {None if asked is None else asked.wall_s}), "
                f"gpu_hours {found.closing.gpu_hours} (requested {None if asked is None else asked.gpu_hours}), "
                f"device peak bytes {found.closing.device_peak_bytes} "
                f"(requested {None if asked is None else asked.device_memory_bytes})"
            )
            continue
        layers = ", ".join(f"{k} by {v}" for k, v in found.used.layers.items())
        note = "" if found.note is None else f"; {found.note}"
        lines.append(
            f"placement {found.resource_class} epoch {event.epoch}: {found.verdict}; partition {found.used.partition}, "
            f"time_s {found.used.time_s}, max_workers {found.used.max_workers}, {len(found.exclude)} excluded "
            f"({layers}){note}"
        )
    return lines


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
    if status.reclaimed:
        print(f"reclaimed claims: {status.reclaimed}")
    if status.quarantined:
        print(f"quarantined event lines: {status.quarantined}")
    for message in status.unplaceable:
        print(f"unplaceable: {' '.join(message.split())}")
    for broken in status.broken:
        print(f"broken env {broken.env}: {broken.failures} failures; {broken.fault.type}: {broken.fault.message}")
    for node, counts in status.nodes.items():
        print(f"node {node}: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    for line in _placement_lines(run.events()):
        print(line)
    if args.check_provenance:
        report = check_provenance(run.run_dir)
        print(
            f"provenance: {report.committed} committed keys, {report.executions} executions, {report.reuses} reuses: "
            + ("complete" if report.complete else "INCOMPLETE")
        )
        for problem in report.problems:
            print(f"provenance: {problem}")
        return 0 if report.complete else 1
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
            return read_placement(path)
    return latest_placement(run_dir)


def _failure_text(failure: FaultRecord) -> str:
    """Return a failure's lines for `explain`: an unplaceable plan's own headline first, then its class and tail."""
    head = f"unplaceable: {' '.join(failure.message.split())}\n" if failure.type == UNPLACEABLE else ""
    return f"{head}fault {failure.fault}: {failure.type}\n{_tail(failure.message)}"


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
    nodes = sorted({e.node for e in matched if e.node is not None} | ({args.target} if args.target in tasks else set()))
    if not nodes:
        print(f"senselab: error: no node or key {args.target!r} in run {run.run_dir.name}", file=sys.stderr)
        return 1
    for node in nodes:
        shown = [e for e in matched if e.node == node] or [e for e in events if e.node == node]
        try:
            placement = _placement(run.run_dir, shown)
        except (OSError, ValueError) as err:
            print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
            return 1
        print(node_line(node, tasks[node], placement) if node in placement.nodes else f"node {node}")
        if node in placement.nodes:
            for line in _placement_lines(events, placement.nodes[node].resource_class):
                print(line)
        print(f"events ({len(shown)}):")
        for event in shown:
            print(_event_line(event))
        for event in (e for e in shown if e.failure is not None):
            assert event.failure is not None
            print(_failure_text(event.failure))
        for event in (e for e in shown if e.kind in ("blocked", "cancelled")):
            why = "an upstream job failed" if event.kind == "blocked" else "the run was cancelled before it was sent"
            print(f"{event.kind} {event.node}{list(event.job.element) if event.job else ''}: {why}")
    return 0


def _profile(argument: str | None) -> ExecutionProfile:
    """Return the profile `argument` names: a packaged profile, a profile YAML file, or a whole profile (`.json`)."""
    if argument is None:
        return ExecutionProfile.load()
    if argument.endswith(".json") and Path(argument).is_file():
        return ExecutionProfile.model_validate_json(Path(argument).read_text(encoding="utf-8"))
    return ExecutionProfile.load(argument)


def _with_placement(placed: Plan, args: argparse.Namespace) -> Plan:
    """Return `placed` with the explicit placement values and the site's hook the command line gives.

    Raises:
        PlacementAnswerRefused: `--placement` names no class of the profile or holds a refused value.
    """
    return replace(placed, overrides=load_overrides(args.placement), hook=args.placement_hook)


def run_plan_request(args: argparse.Namespace) -> int:
    """Run a parsed ``plan``: print what each resource class of a run needs, as `placement-request@1.0` JSON.

    Nothing is computed, no model downloaded and no worker started.
    """
    try:
        profile = _profile(args.profile)
        placed = _with_placement(run_plan(_run_directory(args.run_ref), profile), args)
        document = placed.placement_request().document()
    except (OSError, ValueError) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


def run_plan(run_dir: Path, profile: ExecutionProfile) -> Plan:
    """Return the plan of the run in `run_dir` under its latest placement, `profile` and its policies.

    Raises:
        ValueError: the plan was placed under another profile than `profile`.
        FileNotFoundError: the run has no plan, or no claimed attempt.
    """
    stored = load_plan(run_dir)
    if stored.placement.profile_digest != profile.identity_digest():
        raise ValueError(f"plan {stored.hash} was placed under another execution profile; name the one it was")
    config = RunConfig(reuse=ReusePolicy(stored.logical.reuse), licenses=stored_policy(run_dir))
    return replace(stored, execution=profile, config=config)


def run_run(args: argparse.Namespace) -> int:
    """Run a parsed ``run``: run the plan of a run directory again, as a new attempt on a warm pool, and wait.

    The run uses the process's default session for the profile (`default_session`), whose substrate starts its
    workers; it is closed when the command exits.
    """
    try:
        profile = _profile(args.profile)
        run_dir = _run_directory(args.run_ref)
        placed = _with_placement(run_plan(run_dir, profile), args)
        run = placed.run()
        status = run.wait()
    except (OSError, ValueError, QueueHeld, PlacementHookFailed) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1
    counts = ", ".join(f"{n} {state}" for state, n in sorted(status.counts.items()))
    print(f"run {placed.hash}: {'done' if status.done else 'not done'}; {counts}")
    return 0 if status.done else 1


def _placement_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--placement", type=Path, default=None, help="a YAML file of explicit placement values per resource class"
    )
    parser.add_argument(
        "--placement-hook", default=None, help="the site's placement hook, `module:callable`; replaces the profile's"
    )


def mount_runs(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add the ``status``, ``explain``, ``coordinate`` and ``execute`` commands to a parser's subparsers."""
    status = groups.add_parser("status", help="the jobs, faults and broken envs of a run")
    status.add_argument(
        "run_ref",
        metavar="run",
        help="a run directory, or a plan hash (or its unique prefix) under $SENSELAB_CACHE/2x/runs",
    )
    status.add_argument(
        "--check-provenance",
        action="store_true",
        help=(
            "also check ADR-010: each committed key has an execution or a Reuse, and no writer records more Reuse "
            "of a key than the run has jobs with it; exit 1 if not"
        ),
    )
    status.set_defaults(run=run_status)
    planning = groups.add_parser("plan", help="what each resource class of a run needs, as placement-request@1.0")
    planning.add_argument("run_ref", metavar="run", help="a run directory, or a plan hash (or its unique prefix)")
    planning.add_argument("--json", action="store_true", help="print the request as JSON (the only output form)")
    planning.add_argument("--profile", default=None, help="the execution profile the run was placed under")
    _placement_arguments(planning)
    planning.set_defaults(run=run_plan_request)
    explain = groups.add_parser("explain", help="the events, fault and placement of a node or key of a run")
    explain.add_argument("run_ref", metavar="run", help="a run directory, or a plan hash (or its unique prefix)")
    explain.add_argument("target", help="a node id, or a key digest or its prefix")
    explain.set_defaults(run=run_explain)
    running = groups.add_parser("run", help="run a run's plan again as a new attempt on a warm pool, and wait")
    running.add_argument("run_ref", metavar="run", help="a run directory, or a plan hash (or its unique prefix)")
    running.add_argument("--profile", default=None, help="the execution profile the run was placed under")
    _placement_arguments(running)
    running.set_defaults(run=run_run)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``data`` group and the ``status``, ``plan``, ``explain`` and ``run`` commands."""
    parser = argparse.ArgumentParser(prog="senselab")
    groups = parser.add_subparsers(dest="group", required=True)
    mount_data(groups)
    mount_runs(groups)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
