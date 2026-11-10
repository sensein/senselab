"""Shared parts of the Phase 3 exit checks: scenarios, plan comparison, identities, bounded runs and resume accounting.

One module per step (`resample_identity.py`, `stt_identity.py`, `interrupt_resume.py`, `agent_plan.py`) sits over this
one. A run is isolated under `<work>/cache/<name>`, so no step reads another's results, and is bounded by the config
(`exit.yaml`): wall seconds and resident bytes of the coordinator and its workers, watched with the host's load.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from senselab_core.kernel.dataset import ManifestRow
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import RESULTS, Event, EventLog, read_events, run_dir_of, terminal_events
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.run import RESULTS_DIR, Run
from senselab_core.runtime.execution.substrates import (
    LIVE_STATES,
    SQUEUE,
    SQUEUE_FIELDS,
    SQUEUE_FORMAT,
    Slurm,
    substrate_of,
)
from senselab_core.runtime.hardware import host_token
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import MANIFEST_NAME, NAMESPACE, cache_root
from senselab_core.runtime.store import ResultStore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent / "scale")]
import bids_tree  # noqa: E402 - beside the scale ladder

CONFIG = HERE / "exit.yaml"
CONFIG_FORMAT = "exit-checks@1.0"
IDENTITIES_FORMAT = "exit-identities@1.0"
EXECUTORS = ("in-process", "pool-local", "pool-slurm")
KIB: Final = 1024  # invariant: `ps` reports resident set sizes in KiB


@dataclass(frozen=True)
class Step:
    """One scenario: a task class of `senselab`, its params, and the BIDS tree its records come from."""

    task: str
    params: Mapping[str, Any]
    tree: str
    items: int
    seed: int | None = None
    manifest: str | None = None


@dataclass(frozen=True)
class Bounds:
    """The wall seconds and resident bytes of a run; None leaves that bound to the allocation."""

    wall_s: float | None
    resident_bytes: int | None


@dataclass(frozen=True)
class Config:
    """`exit.yaml`: the scenarios, bounds per executor, local workers, watch cadence and interrupt share."""

    plan_profiles: tuple[str, ...]
    steps: Mapping[str, Step]
    bounds: Mapping[str, Bounds]
    pool_local_workers: int
    watch_poll_s: float
    progress_s: float
    interrupt_share: tuple[int, int]


def load_config(path: Path = CONFIG) -> Config:
    """Read `exit.yaml`.

    Raises:
        ValueError: the file is not of the format, or names an executor or step badly.
    """
    raw = load_yaml(path, "exit checks")
    if raw.get("format") != CONFIG_FORMAT:
        raise ValueError(f"{path.name} is not a {CONFIG_FORMAT} file")
    bounds = {name: Bounds(**b) for name, b in raw["bounds"].items()}
    if set(bounds) != set(EXECUTORS):
        raise ValueError(f"{path.name} bounds the executors {sorted(bounds)}, not {sorted(EXECUTORS)}")
    numerator, denominator = raw["interrupt_share"]
    if not 0 < numerator < denominator:
        raise ValueError(f"interrupt_share is a proper fraction, got {numerator}/{denominator}")
    return Config(
        plan_profiles=tuple(raw["plan_profiles"]),
        steps={name: Step(**step) for name, step in raw["steps"].items()},
        bounds=bounds,
        pool_local_workers=int(raw["pool_local_workers"]),
        watch_poll_s=float(raw["watch_poll_s"]),
        progress_s=float(raw["progress_s"]),
        interrupt_share=(int(numerator), int(denominator)),
    )


def load_profile(argument: str) -> ExecutionProfile:
    """Return the packaged execution profile `argument` names, or the profile file it is a path to."""
    return ExecutionProfile.load(argument)


def isolate(work: Path, name: str) -> Path:
    """Point this process, and the workers it starts, at the cache root `<work>/cache/<name>`; return it."""
    root = work / "cache" / name
    os.environ["SENSELAB_CACHE"] = str(root)
    return root


def build_tree(step: Step, work: Path) -> Path:
    """Return the BIDS tree of `step` under `work`, laid first if absent."""
    name = f"{step.tree}-{step.items}" if step.seed is None else f"{step.tree}-{step.seed}-{step.items}"
    root = work / "data" / name
    if (root / "dataset_description.json").is_file():
        return root
    if step.tree == "synthetic" and step.seed is not None:
        return bids_tree.synthetic(root, step.items, step.seed)
    if step.tree == "librispeech" and step.manifest is not None:
        return bids_tree.from_librispeech(root, ROOT / "benchmarks" / "manifests" / f"{step.manifest}.yaml", step.items)
    raise ValueError(f"tree {step.tree!r} with manifest {step.manifest!r} is not one this harness lays")


def planned(step: Step, root: Path, profile: ExecutionProfile) -> Plan:
    """Return `step`'s task mapped over the BIDS tree at `root`, planned under `profile`."""
    import senselab  # noqa: PLC0415 - imported after the cache root is set

    task = getattr(senselab, step.task)(**step.params)
    return senselab.plan(task, senselab.Dataset.bids(root), execution=profile)


@dataclass(frozen=True)
class Difference:
    """A field two documents give differently: its dotted path and both values."""

    path: str
    left: object
    right: object

    def render(self) -> str:
        """Return the difference as one line."""
        return f"{self.path}: {json.dumps(self.left, sort_keys=True)} != {json.dumps(self.right, sort_keys=True)}"


ABSENT: Final = "<absent>"  # invariant: the value a field has in the document that lacks it


def field_diff(left: object, right: object, path: str = "") -> list[Difference]:
    """Return every leaf field that differs between two JSON values, by dotted path; lists differ index by index."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        found: list[Difference] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key in left and key in right:
                found += field_diff(left[key], right[key], child)
            else:
                found.append(Difference(child, left.get(key, ABSENT), right.get(key, ABSENT)))
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [d for i, (a, b) in enumerate(zip(left, right, strict=True)) for d in field_diff(a, b, f"{path}[{i}]")]
    return [] if left == right else [Difference(path, left, right)]


def plan_document(placed: Plan) -> dict[str, Any]:
    """Return a plan as JSON: its hash, logical plan, manifest digest and placement."""
    return {
        "hash": placed.hash,
        "logical": placed.logical.model_dump(mode="json"),
        "manifest": placed.manifest.digest(),
        "placement": placed.placement.model_dump(mode="json"),
    }


class PlanIdentityError(AssertionError):
    """Two plans of one scenario differ in their hash or outside the part allowed to differ."""

    def __init__(self, outside: Sequence[Difference]) -> None:
        """Name every difference outside the allowed part."""
        super().__init__("plans differ outside placement:\n" + "\n".join(d.render() for d in outside))
        self.outside = tuple(outside)


def compare_plans(left: Plan, right: Plan, confined: str = "placement") -> list[Difference]:
    """Return every field two plans differ in, requiring one hash and every difference inside `confined`.

    Raises:
        PlanIdentityError: the hashes differ, or a field outside `confined` does.
    """
    found = field_diff(plan_document(left), plan_document(right))
    outside = [d for d in found if d.path.split(".")[0].split("[")[0] != confined]
    if left.hash != right.hash or outside:
        raise PlanIdentityError(outside)
    return found


def output_identities(events: Sequence[Event], rows: Sequence[ManifestRow]) -> dict[str, str]:
    """Return each record id's output identity from the terminal result events of a one-node run.

    Raises:
        ValueError: the events name more than one node.
    """
    terminal = {job: e for job, e in terminal_events(events).items() if e.kind in RESULTS and job[1]}
    if len({node for node, _ in terminal}) > 1:
        raise ValueError("output identities are read from a run of one node")
    return {rows[element[0]].id: e.output for (_, element), e in terminal.items() if e.output is not None}


def compare_identities(left: Mapping[str, str], right: Mapping[str, str]) -> list[str]:
    """Return a line for each record one side lacks and each record whose identity differs; none when equal."""
    lines = [f"{i}: only on the left" for i in sorted(left.keys() - right.keys())]
    lines += [f"{i}: only on the right" for i in sorted(right.keys() - left.keys())]
    lines += [f"{i}: {left[i]} != {right[i]}" for i in sorted(left.keys() & right.keys()) if left[i] != right[i]]
    return lines


def tree_resident(ps_text: str, root: int) -> int:
    """Return the resident bytes of process `root` and its descendants from `ps -A -o pid=,ppid=,rss=` output."""
    children: dict[int, list[int]] = defaultdict(list)
    rss: dict[int, int] = {}
    for line in filter(str.strip, ps_text.splitlines()):
        pid, parent, size = (int(field) for field in line.split())
        children[parent].append(pid)
        rss[pid] = size
    found, todo = {root}, [root]
    while todo:
        for child in children[todo.pop()]:
            if child not in found:
                found.add(child)
                todo.append(child)
    return KIB * sum(rss.get(pid, 0) for pid in found)


def process_resident(poll_s: float) -> int | None:
    """Return this process tree's resident bytes now; None when `ps` took longer than `poll_s`."""
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, check=True, timeout=poll_s
        ).stdout
    except subprocess.TimeoutExpired:
        return None
    return tree_resident(out, os.getpid())


class BoundExceeded(RuntimeError):
    """A run passed its wall or resident bound and was cancelled."""


class Watch:
    """Samples resident bytes and the one-minute load every `poll_s` seconds while its block runs.

    `exceeded` names the bound a sample passed; `peak_resident` and `peak_load` are the largest samples.
    """

    def __init__(
        self,
        bounds: Bounds,
        poll_s: float,
        *,
        resident: Callable[[], int | None] | None = None,
        load: Callable[[], float] = lambda: os.getloadavg()[0],
    ) -> None:
        """Watch against `bounds`, sampling with `resident` (this tree's, by default) and `load`."""
        self.bounds, self.poll_s, self._load = bounds, poll_s, load
        self._resident = resident if resident is not None else lambda: process_resident(poll_s)
        self.peak_resident = 0
        self.peak_load = 0.0
        self.skipped = 0
        self.exceeded: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="exit-watch", daemon=True)

    def sample(self) -> None:
        """Take one sample and record a passed bound."""
        resident, load = self._resident(), self._load()
        self.peak_load = max(self.peak_load, load)
        if resident is None:
            self.skipped += 1
            return
        self.peak_resident = max(self.peak_resident, resident)
        limit = self.bounds.resident_bytes
        if limit is not None and resident > limit and self.exceeded is None:
            self.exceeded = f"resident {resident} B passed the bound {limit} B"

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            self.sample()

    def __enter__(self) -> "Watch":
        """Start sampling."""
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop sampling after a last sample."""
        self._stop.set()
        self._thread.join()
        self.sample()

    def report(self) -> dict[str, Any]:
        """Return the peaks, the samples skipped as too slow, and the host's CPU count."""
        return {
            "peak_resident_bytes": self.peak_resident,
            "peak_load": self.peak_load,
            "skipped_samples": self.skipped,
            "cpus": os.cpu_count(),
        }


def progress_line(committed: int, items: int | None, elapsed_s: float) -> str:
    """Return the one line that says how many of `items` (when known) have a result after `elapsed_s` seconds."""
    return f"progress: {committed}{'' if items is None else f'/{items}'} items committed after {elapsed_s:.0f} s"


class Progress:
    """Prints `progress_line` for the run in `run_dir` every `every_s` seconds, and at the end, while its block runs."""

    def __init__(
        self, run_dir: Path, items: int | None, every_s: float, out: Callable[[str], None] | None = None
    ) -> None:
        """Report the results in the events of `run_dir` out of `items`, through `out` (flushed print by default)."""
        self.run_dir, self.items, self.every_s = run_dir, items, every_s
        self._out = out if out is not None else lambda line: print(line, flush=True)
        self._began = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="exit-progress", daemon=True)

    def tick(self) -> None:
        """Print the committed items now."""
        committed = len(committed_jobs(read_events(self.run_dir))) if self.run_dir.is_dir() else 0
        self._out(progress_line(committed, self.items, time.monotonic() - self._began))

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self.tick()

    def __enter__(self) -> "Progress":
        """Start reporting."""
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop reporting after a last line."""
        self._stop.set()
        self._thread.join()
        self.tick()


class Waitable(Protocol):
    """What `drive` needs of a run."""

    def wait(self, timeout: float | None = None) -> Any:  # noqa: ANN401 - a RunStatus
        """Return the run's status within `timeout` seconds."""

    def cancel(self) -> None:
        """Stop releasing jobs."""


def drive(run: Waitable, watch: Watch, bounds: Bounds, poll_s: float) -> Any:  # noqa: ANN401 - a RunStatus
    """Wait for `run` to be done and return its status; cancel it and raise when a bound is passed.

    Raises:
        BoundExceeded: the watch passed its resident bound, or the wall bound passed, or the coordinator stopped.
    """
    deadline = None if bounds.wall_s is None else time.monotonic() + bounds.wall_s
    while True:
        status = run.wait(timeout=poll_s)
        if status.done:
            return status
        reason = watch.exceeded
        if reason is None and deadline is not None and time.monotonic() >= deadline:
            reason = f"wall passed the bound {bounds.wall_s} s"
        if reason is None and status.interrupted:
            reason = "the coordinator stopped before the run was done"
        if reason is not None:
            run.cancel()
            raise BoundExceeded(reason)


def profile_for(executor: str, profile: ExecutionProfile, config: Config) -> ExecutionProfile:
    """Return the profile a run under `executor` is planned and run with.

    `in-process` and `pool-local` take a local-substrate profile, `pool-local` with `pool_local_workers` workers on each
    cpu class; `pool-slurm` takes a slurm one unchanged.

    Raises:
        ValueError: `executor` is unknown or does not fit the profile's substrate.
    """
    if executor not in EXECUTORS:
        raise ValueError(f"executor {executor!r} is none of {EXECUTORS}")
    local = profile.substrate.kind == "local"
    if local != (executor != "pool-slurm"):
        raise ValueError(
            f"{executor} does not run under the {profile.substrate.kind} substrate of profile {profile.name}"
        )
    if executor != "pool-local":
        return profile
    classes = {
        name: c.model_copy(update={"max_workers": config.pool_local_workers}) if c.device == "cpu" else c
        for name, c in profile.resource_classes.items()
    }
    return profile.model_copy(update={"resource_classes": classes})


def launch(placed: Plan, executor: str, store: ResultStore) -> tuple[Run, Callable[[], None]]:
    """Start `placed` on `executor`; return the run and what closes the executor after it.

    `in-process` and `pool-slurm` are the defaults of the plan's substrate, owned by the run; `pool-local` is a `Pool`
    over local workers of the plan's profile.
    """
    if executor != "pool-local" or placed.execution is None:
        return placed.run(store=store), lambda: None
    run_dir, profile = run_dir_of(placed.hash), placed.execution
    pool = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current),
        store=store,
        events=EventLog(run_dir, "a0", 1, epoch=0),
        profile=profile,
        substrate=substrate_of(profile, run_dir, placed.hash),
        epoch=0,
    )
    return placed.run(pool, store=store), pool.close


@dataclass(frozen=True)
class Execution:
    """A bounded run: its plan, status, events and watch report."""

    placed: Plan
    status: Any
    events: list[Event]
    watch: Mapping[str, Any]
    wall_s: float = 0.0
    counts: Mapping[str, int] = field(default_factory=dict)


def execute(placed: Plan, executor: str, config: Config) -> Execution:
    """Run `placed` on `executor` into a result store of this cache root, bounded by the config, and wait.

    Raises:
        FileExistsError: the plan's run directory exists, so its results would be reused.
        BoundExceeded: as `drive`.
    """
    run_dir = run_dir_of(placed.hash)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; remove the work directory to run afresh")
    store = ResultStore(cache_root() / RESULTS_DIR)
    bounds, began = config.bounds[executor], time.monotonic()
    with Watch(bounds, config.watch_poll_s) as watch, Progress(run_dir, len(placed.manifest.rows), config.progress_s):
        run, close = launch(placed, executor, store)
        try:
            status = drive(run, watch, bounds, config.watch_poll_s)
        finally:
            close()
    return Execution(
        placed, status, read_events(run_dir), watch.report(), time.monotonic() - began, dict(status.counts)
    )


def identities_document(step: str, base: ExecutionProfile, executor: str, done: Execution) -> dict[str, Any]:
    """Return the `exit-identities@1.0` document of a run: its plan hash, counts, watch and record id to output."""
    found = output_identities(done.events, done.placed.manifest.rows)
    return {
        "format": IDENTITIES_FORMAT,
        "step": step,
        "profile": base.name,
        "executor": executor,
        "host": host_token(),
        "hash": done.placed.hash,
        "items": len(done.placed.manifest.rows),
        "counts": dict(done.counts),
        "wall_s": done.wall_s,
        "watch": dict(done.watch),
        "placement": {
            n: {"backend": p.backend, "resource_class": p.resource_class}
            for n, p in done.placed.placement.nodes.items()
        },
        "identities": found,
    }


def write_json(path: Path, document: object) -> None:
    """Write `document` to `path` as sorted, indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")


class NeedsSecondProfile(ValueError):
    """A plan comparison with fewer than two distinct profiles."""


def plan_identity(step: str, work: Path, config: Config, profiles: Sequence[str] = ()) -> tuple[str, list[Difference]]:
    """Plan `step` under the config's profiles and `profiles` on this host; return the one hash and the differences.

    Raises:
        NeedsSecondProfile: fewer than two distinct profiles are named.
        PlanIdentityError: the hashes differ, or the plans differ outside `placement`.
    """
    names = list(dict.fromkeys([*config.plan_profiles, *profiles]))
    if len(names) < 2:  # noqa: PLR2004 - a comparison needs two profiles
        raise NeedsSecondProfile(f"a plan comparison needs at least two distinct profiles, got {names}")
    isolate(work, f"{step}-plan")
    root = build_tree(config.steps[step], work)
    plans = [planned(config.steps[step], root, load_profile(name)) for name in names]
    found: list[Difference] = []
    for other in plans[1:]:
        found += compare_plans(plans[0], other)
    return plans[0].hash, found


def run_identity(step: str, profile_argument: str, executor: str, work: Path, config: Config) -> dict[str, Any]:
    """Plan and run `step` under a profile and executor in a cache of its own; return its identities document.

    Raises:
        BoundExceeded, FileExistsError: as `execute`.
        RuntimeError: an item has no result, or a job failed.
    """
    base = load_profile(profile_argument)
    profile = profile_for(executor, base, config)
    isolate(work, f"{step}-{base.name}-{executor}")
    placed = planned(config.steps[step], build_tree(config.steps[step], work), profile)
    done = execute(placed, executor, config)
    document = identities_document(step, base, executor, done)
    if document["counts"].get("failed") or len(document["identities"]) != document["items"]:
        raise RuntimeError(f"{len(document['identities'])} of {document['items']} items have a result: {done.counts}")
    return document


def compare_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    """Return what separates two identities documents: another step or plan hash, and every identity that differs."""
    lines = [f"step: {left['step']} != {right['step']}"] if left["step"] != right["step"] else []
    lines += [f"hash: {left['hash']} != {right['hash']}"] if left["hash"] != right["hash"] else []
    return lines + compare_identities(left["identities"], right["identities"])


def identity_cli(step: str, argv: Sequence[str] | None = None) -> int:
    """Run the `plan`, `run` and `compare` commands of an identity step; return 0, or 1 on a failed check."""
    parser = argparse.ArgumentParser(prog=Path(sys.argv[0]).name)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="plan under every profile of the config; one hash, placement-only diff")
    planning.add_argument("--work", type=Path, required=True)
    planning.add_argument(
        "--profile",
        action="append",
        default=[],
        help="a packaged profile or a profile file planned beside the config's",
    )
    running = commands.add_parser("run", help="plan and run under a profile and executor; write the identities")
    running.add_argument("--work", type=Path, required=True)
    running.add_argument("--profile", required=True, help="a packaged execution profile, or a profile file")
    running.add_argument("--executor", required=True, choices=EXECUTORS)
    running.add_argument("--out", type=Path, required=True)
    comparing = commands.add_parser("compare", help="compare two identities files")
    comparing.add_argument("left", type=Path)
    comparing.add_argument("right", type=Path)
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "plan":
        try:
            plan_hash, found = plan_identity(step, args.work, config, args.profile)
        except PlanIdentityError as error:
            print(error)
            return 1
        except NeedsSecondProfile as error:
            parser.error(f"{error}; name a site profile with `plan --profile FILE`")
        print(
            f"{step}: one plan hash {plan_hash} under {', '.join([*config.plan_profiles, *args.profile])}; "
            f"{len(found)} differences, all in placement"
        )
        print("\n".join(d.render() for d in found))
        return 0
    if args.command == "run":
        document = run_identity(step, args.profile, args.executor, args.work, config)
        write_json(args.out, document)
        print(f"placed {document['placement']}; wrote {args.out}")
        return 0
    differences = compare_runs(*(json.loads(p.read_text(encoding="utf-8")) for p in (args.left, args.right)))
    print("\n".join(differences) if differences else "equal plan hash and output identities")
    return 1 if differences else 0


@dataclass(frozen=True)
class Accounting:
    """What a resume did: items committed before it, executions, late hits, and items served without executing."""

    items: int
    committed: int
    executions: int
    late: int
    reuses: int


class ResumeMismatch(AssertionError):
    """The events of a resume contradict executions = items not committed, reuses = items committed."""


def committed_jobs(events: Sequence[Event]) -> dict[tuple[str, tuple[int, ...]], str | None]:
    """Return each job with a result event (`finished`, `cache_hit`), with its key."""
    return {job: e.key for job, e in terminal_events(events).items() if e.kind in RESULTS and job[1]}


def failure_summary(events: Sequence[Event]) -> str:
    """Return each distinct failure message of `events` with its count, the most common first."""
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        if event.failure is not None:
            counts[event.failure.message.strip()] += 1
    if not counts:
        return "no failure events"
    return "failures: " + "; ".join(f"{n} x {message}" for message, n in sorted(counts.items(), key=lambda i: -i[1]))


def _judge_groups(
    groups: Mapping[object, Sequence[tuple[str, tuple[int, ...]]]],
    held: set[object],
    finished: Mapping[tuple[str, tuple[int, ...]], int],
) -> tuple[int, int, list[str]]:
    """Return the groups that executed once, those served wholly from the store, and each group that did wrong."""
    executions = late = 0
    problems = []
    for key, members in groups.items():
        count = sum(finished[job] for job in members)
        if key in held and count:
            problems.append(f"key {key} was committed before and executed again")
        elif count > 1:
            problems.append(f"key {key} executed {count} times")
        elif count == 1:
            executions += 1
        elif key not in held:
            late += 1
    return executions, late, problems


def account_resume(before: Sequence[Event], after: Sequence[Event], items: int) -> Accounting:
    """Account a resume from the events before it and after it.

    Only events of an epoch above every epoch before the resume are the resume's (a surviving worker of the
    interrupted epoch may still log). Jobs the resume still had to compute are grouped by key, as the run groups
    them (an in-run duplicate is an exact reuse of the first): a group whose key was committed before must not
    execute; any other executes exactly once, or
    is served wholly from the store when its first finished between the last event and the interrupt (a late hit).

    Raises:
        ResumeMismatch: the run was not interrupted part way, a committed job executed again, a group executed twice,
            or an item has no result.
    """
    top = max((e.epoch for e in before), default=-1)
    new = [e for e in after if e.epoch > top]
    committed = committed_jobs(before)
    keys = {(e.node, e.job.element): e.key for e in after if e.job is not None and e.key is not None}
    jobs = {(e.node, e.job.element) for e in after if e.job is not None and e.job.element}
    finished: dict[tuple[str, tuple[int, ...]], int] = defaultdict(int)
    for e in new:
        if e.kind == "finished" and e.job is not None:
            finished[(e.node, e.job.element)] += 1
    problems = []
    if not 0 < len(committed) < items:
        problems.append(
            f"{len(committed)} of {items} items were committed before the resume: it did not follow an interrupt"
        )
    problems += [f"job {job} was committed before and executed again" for job in sorted(committed) if finished[job]]
    groups: dict[object, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for job in jobs - committed.keys():
        groups[keys.get(job, job)].append(job)
    held: set[object] = {keys.get(job, job) for job in committed}
    executions, late, wrong = _judge_groups(groups, held, finished)
    problems += wrong
    results = committed_jobs(after)
    if len(results) != items:
        problems.append(f"{items - len(results)} of {items} items have no result after the resume")
    if problems:
        raise ResumeMismatch("\n".join(problems))
    return Accounting(items, len(committed), executions, late, items - executions)


def resolution_snapshot(cache: Path) -> dict[str, dict[str, str]]:
    """Return every run manifest of resolutions under the cache root `cache`, by path: the entries each holds."""
    found = {}
    for path in sorted((cache / NAMESPACE / "runs").glob(f"*/{MANIFEST_NAME}")):
        document = json.loads(path.read_text(encoding="utf-8"))
        found[str(path.relative_to(cache))] = document if isinstance(document, dict) else {}
    return found


def assert_no_resolution(before: Mapping[str, Mapping[str, str]], after: Mapping[str, Mapping[str, str]]) -> None:
    """Refuse a resume that resolved a model ref: no manifest under the cache root gained an entry or appeared.

    Raises:
        ResumeMismatch: a resolution was recorded; names the manifest and ref.
    """
    added = [f"{path}: {key}" for path, entries in after.items() for key in entries if key not in before.get(path, {})]
    if added:
        raise ResumeMismatch("the resume resolved refs: " + "; ".join(sorted(added)))


def slurm_prefix(profile: ExecutionProfile, run_dir: Path, plan_hash: str) -> str:
    """Return the start of the name of every Slurm job of the run in `run_dir`.

    Raises:
        TypeError: the profile's substrate is not Slurm.
    """
    substrate = substrate_of(profile, run_dir, plan_hash)
    if not isinstance(substrate, Slurm):
        raise TypeError(f"profile {profile.name} names a {type(substrate).__name__} substrate, not Slurm")
    return substrate.prefix


def slurm_timeout(profile: ExecutionProfile, fallback_s: float) -> float:
    """Return the profile's Slurm command timeout in seconds, else `fallback_s`."""
    commands = profile.substrate.commands
    return commands.timeout_s if commands is not None else fallback_s


def slurm_job_ids(prefix: str, timeout_s: float, run: Callable[[list[str], float], str]) -> list[str]:
    """Return the ids of this user's live Slurm jobs whose name starts with `prefix`, from one `squeue`."""
    out = run([SQUEUE, "--me", "--noheader", f"--states={LIVE_STATES}", SQUEUE_FORMAT], timeout_s)
    rows = [line.split("|", SQUEUE_FIELDS - 1) for line in out.splitlines() if line.count("|") == SQUEUE_FIELDS - 1]
    return [job for job, _, _, name in rows if name.startswith(prefix)]


def squeue(argv: list[str], timeout_s: float) -> str:
    """Run `argv` and return its output, within `timeout_s` seconds."""
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout_s).stdout


def interrupt_target(items: int, share: tuple[int, int]) -> int:
    """Return the committed items at which the run is to be interrupted: `share` of `items`, rounded up."""
    return math.ceil(items * share[0] / share[1])
