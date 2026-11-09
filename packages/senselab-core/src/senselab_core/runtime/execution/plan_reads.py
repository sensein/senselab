"""What a placed plan says of itself before it runs: its nodes' units and lengths, and what the store would serve."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.calibration import Estimate, LengthProfile
from senselab_core.runtime.call import RunConfig, env_license, packaged
from senselab_core.runtime.envs import EnvLockError, ManagedPythonMissing
from senselab_core.runtime.execution.batching import BucketGrid, bucket_edges, keyed_formation
from senselab_core.runtime.execution.in_process import (
    fragment_jobs,
    job_units,
    node_call,
    node_decoder,
    node_jobs,
    placement_of_record,
    plan_nodes,
)
from senselab_core.runtime.execution.job import job_context
from senselab_core.runtime.execution.workers import service_of
from senselab_core.runtime.keys import request_components
from senselab_core.runtime.measures import MeasureUnit
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import UnitResolver, Unresolved


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


def count_items(
    placed: Plan,
    values: Mapping[InputIdentity, object],
    *,
    store: ResultStore,
    run: RunConfig,
    profile: ExecutionProfile,
) -> dict[str, ItemCounts]:
    """Return, per resource class, the jobs of `placed` the store would serve under `run` and the ones computed.

    The one servable predicate is `ResultStore.admits`, asked of each job's key digest and the context it would run
    in (`job_context`). Nothing is read from a result, built or started: a job whose env is not built, or whose key
    waits on an upstream result, is undetermined.

    Raises:
        As `ResultStore.admits`, `fragment_jobs` and `service_of`.
    """
    counts: dict[str, ItemCounts] = {}
    planned = plan_nodes(placed, values)
    grid = BucketGrid.load_default()
    for node, record in placed.placement.nodes.items():
        held = counts.get(record.resource_class, ItemCounts())
        if node not in planned:
            counts[record.resource_class] = replace(held, later=(*held.later, node))
            continue
        known = planned[node]
        jobs, placement = fragment_jobs(known.fragment, placed, values)
        backend = placement.backend
        try:
            env = packaged().env_read(backend, profile)
        except (EnvLockError, ManagedPythonMissing) as err:
            env, reason = None, f"the env of backend {backend.name!r} cannot be read: {err}"
        else:
            reason = f"the env of backend {backend.name!r} is not built"
        if env is None:
            counts[record.resource_class] = replace(
                held, undetermined=held.undetermined + len(jobs), why=(*held.why, reason)
            )
            continue
        expected = service_of(backend, jobs[0].call.model) if jobs else None
        code = env_license(backend)
        servable = computed = 0
        for job, key in zip(jobs, known.keys, strict=True):
            picked, _ = keyed_formation(job, backend, grid)
            keyed = replace(placement, **request_components(backend, picked))
            context = job_context(keyed, env, expected, profile, job, code)
            if store.admits(key, context, run, backend=backend.name) is None:
                computed += 1
            else:
                servable += 1
        counts[record.resource_class] = replace(
            held, servable=held.servable + servable, computed=held.computed + computed
        )
    return counts


def unit_of_node(
    placed: Plan, resolver: UnitResolver, values: Mapping[InputIdentity, object] | None
) -> Callable[[str], MeasureUnit | Unresolved]:
    """Return what resolves each node of `placed` to its unit: its call, placement and the decoder of its jobs.

    Without `values` the decoder is unknown and taken as None, which is the unit of in-memory inputs.
    """
    registry = packaged().registry

    def of(node: str) -> MeasureUnit | Unresolved:
        call = node_call(placed, node, registry)
        placement = placement_of_record(placed.placement.nodes[node], call.task, registry)
        decoder = None if values is None else node_decoder(placed, node, values)
        if isinstance(decoder, Unresolved):
            return decoder
        return resolver.unit(call, placement, decoder=decoder)

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
    return bucket_edges(
        placement_of_record(placed.placement.nodes[node], call.task, registry).backend, estimate.unit_kind
    )
