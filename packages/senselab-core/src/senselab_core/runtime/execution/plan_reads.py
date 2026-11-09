"""What a placed plan says of itself before it runs: its nodes' units and lengths, and what the store would serve."""

import functools
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from senselab_core.kernel.identity import ExecutionContext, InputIdentity
from senselab_core.runtime.calibration import Estimate, LengthProfile
from senselab_core.runtime.call import RunConfig, env_license, input_binding, input_decoder, packaged
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.batching import BucketGrid, bucket_edges, keyed_formation
from senselab_core.runtime.execution.in_process import (
    fragment_jobs,
    job_units,
    node_call,
    node_jobs,
    placement_of_record,
    plan_nodes,
)
from senselab_core.runtime.execution.job import Job, held_decoder, job_context
from senselab_core.runtime.execution.run import SourcesUnrecorded, port_types, recorded_sources
from senselab_core.runtime.execution.workers import service_of
from senselab_core.runtime.keys import Placement, request_components
from senselab_core.runtime.measures import MeasureUnit
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import EnvStates, UnitResolver, Unresolved


@dataclass(frozen=True)
class ItemCounts:
    """What a resource class's jobs will do: `servable` from the store (`ResultStore.admits`), the rest computed.

    `computed` jobs are those the store would not serve. `undetermined` jobs cannot be told at plan time, for the
    reasons in `why`; `later` names the nodes whose jobs wait on an upstream result.
    """

    servable: int = 0
    computed: int = 0
    undetermined: int = 0
    why: tuple[str, ...] = ()
    later: tuple[str, ...] = ()


def _admitted(store: ResultStore, key: str, context: ExecutionContext, run: RunConfig, backend: str) -> bool:
    return store.admits(key, context, run, backend=backend) is not None


def job_asks(  # noqa: PLR0913 - the jobs and their keys, the env, and what the predicate reads
    jobs: Sequence[Job],
    placement: Placement,
    keys: Sequence[str],
    env: EnvHandle,
    *,
    profile: ExecutionProfile,
    store: ResultStore,
    run: RunConfig,
    grid: BucketGrid,
) -> list[Callable[[], bool]]:
    """Return, per job, the call that asks `ResultStore.admits` whether the store would serve it in its context."""
    backend = placement.backend
    expected = service_of(backend, jobs[0].call.model) if jobs else None
    code = env_license(backend)
    asks: list[Callable[[], bool]] = []
    for job, key in zip(jobs, keys, strict=True):
        picked, _ = keyed_formation(job, backend, grid)
        keyed = replace(placement, **request_components(backend, picked))
        context = job_context(keyed, env, expected, profile, job, code)
        asks.append(functools.partial(_admitted, store, key, context, run, backend.name))
    return asks


def servable_jobs(  # noqa: PLR0913 - the fragment's jobs and keys, the env, and what the predicate reads
    jobs: Sequence[Job],
    placement: Placement,
    keys: Sequence[str],
    envs: EnvStates,
    *,
    profile: ExecutionProfile,
    store: ResultStore,
    run: RunConfig,
) -> list[bool]:
    """Return whether the store would serve each job (`ResultStore.admits`, on `pool.admit_threads` threads).

    Every job is computed where its env is not known: not built, not readable, or on a slurm node.
    """
    env = envs.get(placement.backend)
    if isinstance(env, Unresolved):
        return [False] * len(jobs)
    asks = job_asks(jobs, placement, keys, env, profile=profile, store=store, run=run, grid=BucketGrid.load_default())
    with ThreadPoolExecutor(max_workers=profile.pool.admit_threads) as threads:
        return list(threads.map(lambda ask: ask(), asks))


def count_items(  # noqa: PLR0913 - the plan and its values, the store and run, the profile and the env reads
    placed: Plan,
    values: Mapping[InputIdentity, object],
    *,
    store: ResultStore,
    run: RunConfig,
    profile: ExecutionProfile,
    envs: EnvStates | None = None,
) -> dict[str, ItemCounts]:
    """Return, per resource class, the jobs of `placed` the store would serve under `run` and the ones computed.

    The one servable predicate is `ResultStore.admits`, asked of each job's key digest and the context it would run
    in (`job_context`), on up to `pool.admit_threads` threads. Nothing is read from a result, built or started: a job
    whose env is not built or not readable, whose context is not knowable before a worker starts (a slurm node), or
    whose key waits on an upstream result, is undetermined.

    Raises:
        As `ResultStore.admits`, `fragment_jobs` and `service_of`.
    """
    states = envs if envs is not None else EnvStates(profile)
    counts: dict[str, ItemCounts] = {}
    planned = plan_nodes(placed, values)
    grid = BucketGrid.load_default()
    asked: list[tuple[str, Callable[[], bool]]] = []
    for node, record in placed.placement.nodes.items():
        held = counts.get(record.resource_class, ItemCounts())
        if node not in planned:
            counts[record.resource_class] = replace(held, later=(*held.later, node))
            continue
        known = planned[node]
        jobs, placement = fragment_jobs(known.fragment, placed, values)
        backend = placement.backend
        env = states.get(backend)
        if isinstance(env, Unresolved):
            counts[record.resource_class] = replace(
                held, undetermined=held.undetermined + len(jobs), why=(*held.why, env.reason)
            )
            continue
        for ask in job_asks(jobs, placement, known.keys, env, profile=profile, store=store, run=run, grid=grid):
            asked.append((record.resource_class, ask))
    with ThreadPoolExecutor(max_workers=profile.pool.admit_threads) as pool:
        answers = list(pool.map(lambda a: a[1](), asked))
    for (name, _), servable in zip(asked, answers, strict=True):
        held = counts.get(name, ItemCounts())
        counts[name] = replace(held, servable=held.servable + servable, computed=held.computed + (not servable))
    return counts


def plan_decoder(placed: Plan, node: str) -> str | None | Unresolved:
    """Return the decoder `node`'s file-backed inputs are read by, from the plan's recorded sources; no file is read.

    A path source of a port with a reader is read by that reader's decoder, a JSON value by none, and a
    `{"memory": ...}` source is as the value `placed.live` holds. A node fed by both kinds has two units; one fed by a
    value this process does not hold, or with no recorded sources, has an unknown decoder.
    """
    types = port_types(placed)
    try:
        rows, shared = recorded_sources(placed)
    except SourcesUnrecorded as err:
        return Unresolved(str(err))
    fed = {
        e.source.split(".", 1)[1]
        for e in placed.logical.graph.edges
        if e.target.split(".", 1)[0] == node and e.source.startswith("input.")
    }
    found: set[str | None] = set()
    for port in fed:
        for row in rows:
            source = row[port] if port in row else shared.get(port)
            if isinstance(source, Mapping) and set(source) == {"memory"}:
                if source["memory"] not in placed.live:
                    return Unresolved("its inputs are held in memory by the process that planned the run")
                found.add(held_decoder([placed.live[source["memory"]]]))
            elif isinstance(source, str) and input_binding(types[port]) == "path":
                found.add(input_decoder(types[port]))
            else:
                found.add(None)
    if len(found) > 1:
        return Unresolved("its jobs read both files and in-memory inputs, which are two units")
    return next(iter(found), None)


def unit_of_node(placed: Plan, resolver: UnitResolver) -> Callable[[str], MeasureUnit | Unresolved]:
    """Return what resolves each node of `placed` to its unit: its call, placement and the decoder its sources give."""
    registry = packaged().registry

    def of(node: str) -> MeasureUnit | Unresolved:
        call = node_call(placed, node, registry)
        placement = placement_of_record(placed.placement.nodes[node], call.task, registry)
        decoder = plan_decoder(placed, node)
        return decoder if isinstance(decoder, Unresolved) else resolver.unit(call, placement, decoder)

    return of


def node_work(
    placed: Plan, node: str, values: Mapping[InputIdentity, object] | None, estimate: Estimate
) -> LengthProfile | None:
    """Return the input units of `node`'s jobs in the unit of `estimate`; None when they are not known at plan time."""
    jobs = None if values is None else node_jobs(placed, node, values)
    return None if jobs is None else LengthProfile(units=tuple(job_units(j, estimate.unit_kind) for j in jobs))


def node_edges(placed: Plan, node: str, estimate: Estimate) -> tuple[float, ...]:
    """Return the bucket edges, in the units of `estimate`, `node`'s jobs are cut on; none unless bucketed."""
    registry = packaged().registry
    call = node_call(placed, node, registry)
    backend = placement_of_record(placed.placement.nodes[node], call.task, registry).backend
    return bucket_edges(backend, estimate.unit_kind)
