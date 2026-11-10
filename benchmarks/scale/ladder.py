"""The scale ladder: each task of a suite run at its scale points and measured per item (`scale-result@2.0`).

`uv run python benchmarks/scale/ladder.py run SUITE --execution PROFILE [--measures-root PATH]`

A suite (`scale-suite@1.1`, `suites/<name>.yaml`; `suites/fake.yaml` runs with `fake_stack/` on PYTHONPATH) names a
seed, the long signal's seconds and its tasks, and may name a table profile: `calibration:` overrides of
`length_strata`, `batched_sizes` and `probe_items`, and `point_wall_s`, the seconds of warm probe work a point's probe
may take (`Probe.plan`'s budget, in place of the profile's `probe_s`). Each task names the input port its data binds
to, its params, model and stack, its data (`synthetic` tones, `librispeech` utterances of a manifest, `gretel`
documents of a manifest, or `payloads`, JSON records for the fake stack) and its scale points: item counts, or `long`,
one recording of the long seconds. `PROFILE` is a packaged execution profile, or a profile file (a whole profile: copy
`slurm` or `laptop` and edit it; `substrate.kind: local` runs a cluster's classes as local workers of one allocation).
Each point is planned, run on a `Pool` over the profile's substrate into a store of its own, its pool probing into a
measures root of the point's own (so each unit is probed in the run, not read from another), and measured from its
events, its files, the samples each pool worker reported and the observations of every request. One `scale-result@2.0`
document per point is written under `$SENSELAB_CACHE/2x/benchmarks/scale/<suite>/<commit>/`: per unit the plan resolved
whether it was probed, its probe observations and its `Estimate` (unmeasured, with reasons, where the probe did not
measure), the per-item seconds (`request_cost`) and the calibration settings it ran under. With `--measures-root PATH`
every point's pool probes into and reads that one root instead, so a group calibrating its own resource lists `PATH` in
its profile's `calibration.measures_roots` and a unit measured there is read on an identity match. A unit already in
`PATH` is not probed again: its result entry says `probed` false and names no `probe_ids`, so a campaign whose figures
are evidence starts from an empty `PATH`. One line is printed per point, then each task's flatness against
`budgets.yaml` (`scale-budgets@1.0`). A point whose run directory already exists is refused, so no point reuses a run's
results, and every entry is planned before any point is measured, so a suite with an entry that does not plan is
refused, every such entry listed.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile
from senselab_core.kernel.dataset import ManifestRow
from senselab_core.kernel.formats import readable
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.runtime.bids import bids_dataset
from senselab_core.runtime.calibration import Observation
from senselab_core.runtime.call import packaged
from senselab_core.runtime.config import ConfigError, load_yaml
from senselab_core.runtime.datasets import from_records
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import Event, EventLog, read_events, run_dir_of
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet, probe_of
from senselab_core.runtime.execution.substrates import substrate_of, worker_id
from senselab_core.runtime.execution_settings import CalibrationSettings
from senselab_core.runtime.fetch import DataManifest, FetchError, ManifestError, default_cache_root, fetch
from senselab_core.runtime.hardware import host_token, memory_bytes
from senselab_core.runtime.measures import MeasureRoots, MeasureUnit
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.probe import Probe
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import NAMESPACE, cache_root
from senselab_core.runtime.sizing import Sample
from senselab_core.runtime.storage import StorePolicy
from senselab_core.runtime.store import ResultStore
from senselab_worker.units import NS_PER_S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent)]
import bids_tree  # noqa: E402 - beside this script
from batch_invariance import harness  # noqa: E402 - the suites' adapters

SUITE_FORMAT = "scale-suite@1.1"
"""The suite format this reads and writes; `@1.1` added `calibration` and `point_wall_s`, and any `@1.x` is read."""
RESULT_FORMAT = "scale-result@2.0"
BUDGETS_FORMAT = "scale-budgets@1.0"
LONG = "long"
FLAT_POINTS = (100, 10_000)
"""The points whose per-item costs the flatness rule compares (P-33)."""
TABLE_KEYS = frozenset({"length_strata", "batched_sizes", "probe_items"})
"""The calibration settings a suite's table profile may override: strata, batch sizes and item budget, no seconds."""
FAULT_MESSAGE_CHARS = 200
"""The characters of a failed job's message a result keeps: enough to name the failure, not to copy its input."""
RECORDING = {"synthetic": "Audio", "librispeech": "Audio"}
"""The graph input type of each data kind read from a BIDS tree; `payloads` take the port's declared type."""


@dataclass(frozen=True)
class Entry:
    """One task of a suite: its port, data, points, params, model, and its stack (the profile's choice when None)."""

    task: str
    port: str
    data: str
    points: tuple[int | str, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    model: str | None = None
    manifest: str | None = None
    stack: str | None = None


@dataclass(frozen=True)
class Suite:
    """A `scale-suite@1.1` file: name, seed, long seconds, tasks, table profile and per-point wall budget.

    `calibration` overrides `TABLE_KEYS` of the execution profile's calibration settings; `point_wall_s` is the seconds
    of warm probe work a point's probe may take, None for the profile's `probe_s`.
    """

    name: str
    seed: int
    long_s: int
    tasks: tuple[Entry, ...]
    calibration: Mapping[str, int] = field(default_factory=dict)
    point_wall_s: float | None = None

    @classmethod
    def load(cls, path: Path) -> "Suite":
        """Read a suite file.

        Raises:
            ValueError: the file is not of the format, an entry is malformed, or `calibration` names a key other than
                `TABLE_KEYS` (a seconds key among them).
        """
        raw = load_yaml(path, "scale suite")
        if not readable(SUITE_FORMAT, raw.get("format")):
            raise ValueError(f"{path.name} is not a {SUITE_FORMAT.split('.', maxsplit=1)[0]}.x file")
        tasks = tuple(Entry(**{**t, "points": tuple(t["points"])}) for t in raw["tasks"])
        bad = [p for t in tasks for p in t.points if p != LONG and not (isinstance(p, int) and p >= 1)]
        if bad:
            raise ValueError(f"{path.name}: a point is an item count or {LONG!r}, got {bad}")
        calibration = dict(raw.get("calibration") or {})
        if wrong := sorted(set(calibration) - TABLE_KEYS):
            raise ValueError(f"{path.name}: calibration overrides only {sorted(TABLE_KEYS)}, not {wrong}")
        wall = raw.get("point_wall_s")
        if wall is not None and not (isinstance(wall, int | float) and wall > 0):
            raise ValueError(f"{path.name}: point_wall_s is a positive number of seconds, got {wall!r}")
        return cls(str(raw["name"]), int(raw["seed"]), int(raw["long_s"]), tasks, calibration, wall)


def table_profile(profile: ExecutionProfile, suite: Suite) -> ExecutionProfile:
    """Return `profile` with the suite's table-profile overrides of its calibration settings; unchanged without any."""
    if not suite.calibration:
        return profile
    merged = CalibrationSettings.model_validate({**profile.calibration.model_dump(), **suite.calibration})
    return profile.model_copy(update={"calibration": merged})


def dataset_of(entry: Entry, point: int | str, suite: Suite, work: Path) -> tuple[Any, Path | None]:
    """Return the dataset of `entry` at `point`, and the BIDS root it is read from (None for payloads)."""
    count = 1 if point == LONG else int(point)
    if entry.data == "payloads":
        return from_records({"id": f"p{i}", "inputs": {"x": {"n": i}}} for i in range(count)), None
    if entry.data == "gretel" and point != LONG and entry.manifest is not None:
        manifest = DataManifest.load(ROOT / "benchmarks" / "manifests" / f"{entry.manifest}.yaml")
        items = harness().GretelPii().items(fetch(manifest, cache_root=default_cache_root()), count, work, None)
        return from_records({"id": i.id, "inputs": {"x": i.inputs["text"].content}} for i in items), None
    root = work / "data" / f"{entry.data}-{suite.seed}-{point}"
    if not (root / "dataset_description.json").is_file():
        if entry.data == "synthetic":
            bids_tree.synthetic(root, count, suite.seed, seconds=suite.long_s if point == LONG else None)
        elif entry.data == "librispeech" and point != LONG and entry.manifest is not None:
            bids_tree.from_librispeech(root, ROOT / "benchmarks" / "manifests" / f"{entry.manifest}.yaml", count)
        else:
            raise ValueError(f"{entry.task}: data {entry.data!r} has no point {point!r}")
    return bids_dataset(root), root


def pinned(model_id: str) -> str:
    """Return `model_id` at its catalogue revision, `<id>@<revision>`; the catalogue is the one source of the pin.

    Raises:
        UnknownModel: the catalogue holds no entry `model_id`.
        ValueError: `model_id` names a revision itself.
    """
    if "@" in model_id:
        raise ValueError(f"a suite names a model by its id alone, got {model_id!r}; the catalogue pins its revision")
    return f"{model_id}@{packaged().catalogue.entry(model_id).revision}"


def graph_of(entry: Entry) -> Graph:
    """Return the one-node graph mapping `entry`'s task over the records' input `x`.

    With a stack, the node names its backend's params (its defaults), so each stack's plan has a run of its own. A
    model is named by its id and pinned to its catalogue revision (`pinned`).
    """
    registry = packaged().registry
    spec = registry.spec(entry.task)
    kind = RECORDING.get(entry.data, spec.inputs[entry.port])
    chosen = [b.name for b in registry.backends(entry.task) if b.stack == entry.stack]
    node = Node(
        id="a",
        task=entry.task,
        task_version=spec.version,
        params=dict(entry.params),
        model=None if entry.model is None else pinned(entry.model),
        backend_params={name: {} for name in chosen},
    )
    return Graph(
        inputs={"x": f"list[{kind}]"},
        nodes=(node,),
        edges=(Edge(source="input.x", target=f"a.{entry.port}", kind="map"),),
        outputs={o: f"a.{o}" for o in spec.outputs},
    )


def _tally(*roots: Path) -> tuple[int, int]:
    """Return the regular files under `roots` and their bytes."""
    files = size = 0
    for root in roots:
        for directory, _, names in os.walk(root):
            for name in names:
                files, size = files + 1, size + os.lstat(os.path.join(directory, name)).st_size
    return files, size


def _outcomes(run_dir: Path) -> list[dict[str, Any]]:
    """Return the outcome record of each item the run's queues committed, with its `queue`'s name."""
    found = []
    for path in sorted((run_dir / RUN_QUEUES).glob("*/outcome/*")):
        if path.name.startswith("."):
            continue
        record = json.loads(path.read_bytes())
        if isinstance(record, dict):
            found.append({**record, "queue": path.parent.parent.name})
    return found


def _samples(outcomes: Sequence[Mapping[str, Any]]) -> list[Sample]:
    """Return the `Sample` each item outcome carries."""
    return [Sample.model_validate(o["measure"]) for o in outcomes if "measure" in o]


def _observations(outcomes: Sequence[Mapping[str, Any]]) -> list[Observation]:
    """Return every request `Observation` the item outcomes report."""
    return [Observation.model_validate(r) for o in outcomes for r in o.get("observations", ())]


def _own_requests(observations: Sequence[Observation]) -> list[Observation]:
    """Return the production requests, each once: a request spanning several items is in each one's observations.

    A request is told by everything it reported, so two items' requests that agree in every figure are one request.
    """
    owner: dict[tuple[object, ...], str] = {}
    found = []
    for o in observations:
        if o.probe is None:
            key = (o.wall_s, o.units, o.padded_units, o.batch, o.jobs, o.since_load, o.load_s, o.outcome)
            key += (o.device_peak, o.device_after, o.host_peak, o.host_after)
            if owner.setdefault(key, o.item) == o.item:
                found.append(o)
    return found


def production_cost(outcomes: Sequence[Mapping[str, Any]]) -> tuple[float, int] | None:
    """Return the span in seconds of the production items and the count of their jobs that gave a result, else None.

    Production items are those committed on a queue that is not a probe's. Both ends of the span are the workers' own
    clock (`Sample.ended_s`, taken just before the commit; the claim's start is `ended_s - wall_s - claim_s`), the
    latest end less the earliest start, so one allocation's clock covers both. Worker starts, model loads before the
    first claim, the wait after the last commit, the coordinator's lag in settling and probe items running beside
    production do not enter it.
    """
    mine = [o for o in outcomes if probe_of(o["queue"]) is None]
    samples = [Sample.model_validate(o["measure"]) for o in mine if "measure" in o]
    if not samples:
        return None
    began = min(s.ended_s - s.wall_s - s.claim_s for s in samples)
    return max(s.ended_s for s in samples) - began, sum(
        j.get("failure") is None for o in mine for j in o.get("jobs", ())
    )


def request_cost(observations: Sequence[Observation], outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a point's per-item figures.

    `per_item_s` is the throughput wall per production job: `production_cost`'s span over its job count, counted from
    the outcome records, so the coordinator's overhead between jobs stays in it. `worker_s_per_item` is worker seconds
    per job: the walls of the `ok` production requests that did not load the model over their jobs, each request counted
    once however many items it spanned and timed from its own start. With no production job `per_item_s` is None.
    """
    warm = [o for o in _own_requests(observations) if o.outcome == "ok" and o.load_s is None]
    warm_jobs = sum(o.jobs for o in warm)
    found = production_cost(outcomes)
    return {
        "per_item_s": None if found is None or not found[1] else found[0] / found[1],
        "per_item_basis": "production span on the worker clock, first claim to last commit, per production job"
        if found is not None and found[1]
        else "no production items",
        "worker_s_per_item": sum(o.wall_s for o in warm) / warm_jobs if warm_jobs else None,
    }


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


class SuiteRefused(ValueError):
    """Entries of a suite that do not plan, each named with its reason; nothing was measured."""


def placed_point(
    entry: Entry, point: int | str, suite: Suite, profile: ExecutionProfile, data: Path
) -> tuple[Plan, Path | None, ExecutionProfile]:
    """Return `entry` planned at `point` over its data under `data`, the BIDS root it is read from, and the profile."""
    dataset, root = dataset_of(entry, point, suite, data)
    if entry.stack is not None:
        stacks = {**profile.stack_preferences, entry.task: (entry.stack,)}
        profile = profile.model_copy(update={"stack_preferences": stacks})
    placed = plan(graph_of(entry), dataset, execution=profile, bind=None if root is None else {"x": "audio"})
    return placed, root, profile


def labels(placed: Sequence[tuple[str, str, str | None]]) -> list[str]:
    """Return the label of each entry placed as `(task, backend, model id)`.

    A label is `<task>~<backend>`, with `~<model>` when two entries share a task and backend; a result's file and
    store are named by its label and point.
    """
    pairs = [(task, backend) for task, backend, _ in placed]
    return [
        "~".join(
            [task, backend]
            + ([model.replace("/", "--")] if model is not None and pairs.count((task, backend)) > 1 else [])
        )
        for task, backend, model in placed
    ]


def check_suite(suite: Suite, profile: ExecutionProfile, data: Path) -> list[str]:
    """Fetch and plan every entry of `suite` at its smallest point, its data under `data`; return their labels.

    Nothing is measured.

    Raises:
        SuiteRefused: an entry's data cannot be had or it does not plan, or two entries share a label (so their
            results would share a path); each such entry is listed with the stage and its error.
    """
    bad: list[str] = []
    placed: list[tuple[str, str, str | None]] = []
    for index, entry in enumerate(suite.tasks):
        counts = [p for p in entry.points if p != LONG]
        point = min(counts) if counts else LONG
        name = f"entry {index} ({entry.task}{'' if entry.stack is None else f', stack {entry.stack}'})"
        try:
            dataset_of(entry, point, suite, data)
        except (OSError, ValueError, ConfigError, FetchError, ManifestError) as err:
            bad.append(f"{name}: data: {type(err).__name__}: {err}")
            continue
        try:
            plan_, _, _ = placed_point(entry, point, suite, profile, data)
        except Exception as err:  # noqa: BLE001 - every planning error of every entry is listed, then the suite refused
            bad.append(f"{name}: plan: {type(err).__name__}: {err}")
            continue
        model = plan_.logical.models["a"]
        placed.append((entry.task, plan_.placement.nodes["a"].backend, None if model is None else model.ref.id))
    found = labels(placed) if not bad else []
    bad += [
        f"entry {i}: label {label!r} is another entry's too" for i, label in enumerate(found) if found.count(label) > 1
    ]
    if bad:
        raise SuiteRefused(
            f"{suite.name}: {len(bad)} entries do not fetch or plan, nothing was measured:\n" + "\n".join(bad)
        )
    return found


def point_roots(profile: ExecutionProfile, directory: Path) -> MeasureRoots:
    """Return the one measures root of a point: `directory`, so each point probes afresh and reads no other's."""
    return MeasureRoots(
        [directory],
        StorePolicy.load_default(),
        profile.calibration.trust,
        os.geteuid(),
        stale_after_s=profile.lease.stale_after_s,
    )


def measure_point(  # noqa: PLR0913 - the entry and its point, the suite, the profile, its label and where results go
    entry: Entry,
    point: int | str,
    suite: Suite,
    profile: ExecutionProfile,
    label: str,
    out: Path,
    measures_root: Path | None = None,
) -> dict[str, Any]:
    """Plan, run and measure `entry` at `point` under `profile`; return its `scale-result@2.0` document.

    Its store is `<out>/<label>@<point>/store`, and its pool probes into the measures root beside it, with the
    suite's per-point wall budget as its probe seconds, so each point probes afresh and reads no other point's
    measure; with `measures_root` it probes into and reads that root instead.
    Its `units` hold, per unit the plan resolved, whether this run probed it, the probes it wrote and their
    observations, and its `Estimate` (`Probe.estimate`; unmeasured, with reasons, where the probe did not measure).
    `per_item_s` is `request_cost` over every request the run reported.

    Raises:
        FileExistsError: the point's run directory exists, so its results would be reused.
    """
    placed, root, profile = placed_point(entry, point, suite, profile, out.parent)
    run_dir = run_dir_of(placed.hash)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; remove it to measure {entry.task} at {point} afresh")
    store_root = out / f"{label}@{point}" / "store"
    store = ResultStore(store_root)
    roots = point_roots(profile, measures_root or out / f"{label}@{point}" / "measures")
    pool = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current),
        store=store,
        events=EventLog(run_dir, "a0", 1, epoch=0),
        profile=profile,
        substrate=substrate_of(profile, run_dir, placed.hash),
        epoch=0,
        roots=roots,
        probe_s=suite.point_wall_s,
    )
    began = time.monotonic()
    try:
        status = placed.run(pool, store=store).wait()
    finally:
        pool.close()
    resolved, probed = dict(pool.resolved), {d: u.probe for d, u in pool.probes.items()}
    wall = time.monotonic() - began
    outcomes = _outcomes(run_dir)
    samples = _samples(outcomes)
    events = read_events(run_dir)
    queued = {e.job: e.t_ns for e in events if e.kind == "queued" and e.job is not None}
    waits = [(e.t_ns - queued[e.job]) / NS_PER_S for e in events if e.kind == "claimed" and e.job in queued]
    workers = {s.job for s in samples}
    files, written = _tally(store_root, run_dir)
    entries, stored = _tally(store_root)
    node, model = placed.placement.nodes["a"], placed.logical.models["a"]
    items = len(placed.manifest.rows)
    return {
        "format": RESULT_FORMAT,
        "suite": suite.name,
        "task": entry.task,
        "label": label,
        "point": point,
        "items": items,
        "backend": node.backend,
        "model": None if model is None else {"id": model.ref.id, "revision": model.identity},
        "resource_class": node.resource_class,
        "batch_size": node.batch_size,
        "profile": profile.name,
        "profile_digest": profile.canonical_digest(),
        "probe_s": profile.probe_s if suite.point_wall_s is None else suite.point_wall_s,
        "calibration": profile.calibration.model_dump(mode="json"),
        "sha": commit(),
        "jobs": sorted(workers | {worker_id()}),
        "done": status.done,
        "counts": dict(status.counts),
        "wall_s": wall,
        **request_cost(_observations(outcomes), outcomes),
        "units": _units(roots, profile, resolved, probed),
        "faults": _faults(events, placed.manifest.rows),
        "host": host_token(),
        "backend_params": dict(node.backend_params),
        "input_s": None if root is None else sum(soundfile.info(p).duration for p in sorted(root.rglob("*_audio.*"))),
        "job_s": _median([s.wall_s / s.jobs for s in samples if not s.cold]),
        "queue_wait_s": _median(waits),
        "coordinator_s": wall - sum(s.wall_s for s in samples) / len(workers) if workers else None,
        "peak_rss_bytes": max((s.peak_memory_bytes for s in samples), default=memory_bytes()[1]),
        "files_written": files,
        "bytes_written": written,
        "store": {"entries": entries, "bytes": stored},
    }


def _units(
    roots: MeasureRoots, profile: ExecutionProfile, resolved: Mapping[str, MeasureUnit], probed: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Return each unit this point's plan resolved, in digest order, with how it was measured.

    `resolved` is the units by digest and `probed` the probe id this run gave each unit it probed. A unit carries
    `probed` (whether this run probed it), `probe_ids` (the probes this run wrote to `roots`, empty when its estimate
    was read from the root's earlier observations), the observations of those probes and its `Estimate` over every
    observation the root pools for it.
    """
    estimate = Probe.estimator(
        profile.calibration, records_of=roots.regime_records, memory_headroom=profile.sizing.memory_headroom
    )
    found = []
    for digest in sorted(resolved):
        unit = resolved[digest]
        wrote = [p for p in roots.probes(unit) if digest in probed and p.probe == probed[digest]]
        found.append(
            {
                "unit": unit.model_dump(mode="json"),
                "label": unit.label(),
                "probed": digest in probed,
                "probe_ids": [p.probe for p in wrote],
                "observations": [o.model_dump(mode="json") for p in wrote for o in p.observations],
                "estimate": estimate(unit, roots.observations(unit)).model_dump(mode="json"),
            }
        )
    return found


def _faults(events: Sequence[Event], rows: Sequence[ManifestRow]) -> list[dict[str, Any]]:
    """Return each failed job's record id, fault class, exception type and message, its last failure, by record."""
    last = {}
    for event in events:
        if event.kind == "failed" and event.job is not None and event.failure is not None and event.job.element:
            last[event.job.element[0]] = event.failure
    return [
        {"record": rows[i].id, "fault": f.fault, "type": f.type, "message": f.message[:FAULT_MESSAGE_CHARS]}
        for i, f in sorted(last.items())
    ]


def commit() -> str:
    """Return the 40-hex commit of this checkout."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


PER_ITEM: Mapping[str, Callable[[Mapping[str, Any]], float | None]] = {
    "per_item_s": lambda r: r["per_item_s"],
    "peak_rss_bytes": lambda r: r["peak_rss_bytes"],
    "files_per_item": lambda r: r["files_written"] / r["items"],
    "bytes_per_item": lambda r: r["bytes_written"] / r["items"],
}
"""What must stay flat with scale (P-33), read from a result: per-item time, peak memory, files and bytes written."""


def _verdict(value: float | None, bound: float | None, what: str) -> str:
    if value is None:
        return f"{what} n/a"
    return f"{what} unmeasured" if bound is None else (f"{what} ok" if value <= bound else f"{what} EXCEEDED ({bound})")


def flatness(results: Sequence[Mapping[str, Any]], budgets: Mapping[str, Any]) -> list[str]:
    """Return one line per entry (by its label, else its task) and measure: its flatness and its ceiling, by basis.

    Flatness is the value at 10⁴ over at 10² against `flat_ratio`; a point with no production items has no per-item
    seconds and enters no ratio. The ceiling reads the largest of those points the entry ran (the largest of its
    points when it ran neither); each line names the point its ceiling read. An unset budget reads unmeasured.
    """
    lines = []
    for label in sorted({r.get("label", r["task"]) for r in results}):
        at = {r["point"]: r for r in results if r.get("label", r["task"]) == label}
        ceilings = (budgets.get("per_item") or {}).get(label) or {}
        basis = next((p for p in reversed(FLAT_POINTS) if p in at), None)
        if basis is None:
            basis = max(at, key=lambda p: at[p]["items"])
        for name, read in PER_ITEM.items():
            values = {p: read(r) for p, r in at.items()}
            low, high = (values.get(p) for p in FLAT_POINTS)
            ratio = high / low if low and high is not None else None
            flat = _verdict(ratio, budgets.get("flat_ratio"), "flat")
            ceiling = _verdict(values.get(basis), ceilings.get(name), "ceiling")
            shown = "n/a" if ratio is None else f"{ratio:.3f}"
            lines.append(
                f"{label} {name}: at {FLAT_POINTS[1]} / at {FLAT_POINTS[0]} = {shown}, {flat}, {ceiling} (at {basis})"
            )
    return lines


def _profile(argument: str) -> ExecutionProfile:
    return ExecutionProfile.load(argument)


def run(suite_path: Path, profile: ExecutionProfile, measures_root: Path | None = None) -> list[dict[str, Any]]:
    """Measure every point of every task of the suite at `suite_path`; write and return the results.

    Every entry is planned first (`check_suite`), so a suite with an entry that does not plan measures nothing.
    """
    suite = Suite.load(suite_path)
    profile = table_profile(profile, suite)
    out = cache_root() / NAMESPACE / "benchmarks" / "scale" / suite.name / commit()
    found = check_suite(suite, profile, out.parent)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for entry, label in zip(suite.tasks, found, strict=True):
        for point in entry.points:
            result = measure_point(entry, point, suite, profile, label, out, measures_root=measures_root)
            (out / f"{label}@{point}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
            per = "n/a" if result["per_item_s"] is None else f"{result['per_item_s']:.4f}"
            measured = sum(u["estimate"]["status"] == "measured" for u in result["units"])
            print(
                f"{result['sha']} {','.join(result['jobs'])} {label}@{point}: {result['items']} items, "
                f"{measured} of {len(result['units'])} units measured, {per} s/item, "
                f"peak {result['peak_rss_bytes']} B, "
                f"{result['files_written']} files, {result['bytes_written']} B written, done {result['done']}"
            )
            results.append(result)
    budgets = load_yaml(HERE / "budgets.yaml", "scale budgets")
    if budgets.get("format") != BUDGETS_FORMAT:
        raise ValueError(f"budgets.yaml is not a {BUDGETS_FORMAT} file")
    for line in flatness(results, budgets):
        print(line)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run `ladder.py run SUITE --execution PROFILE [--measures-root PATH]`; return 0."""
    parser = argparse.ArgumentParser(prog="ladder.py")
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="measure every point of a scale suite")
    running.add_argument("suite", type=Path, help="a scale-suite@1.x file")
    running.add_argument(
        "--execution", required=True, help="a packaged profile name, or a path to a whole profile file"
    )
    running.add_argument("--measures-root", type=Path, help="one measures root every point probes into and reads")
    args = parser.parse_args(argv)
    run(args.suite, _profile(args.execution), args.measures_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
