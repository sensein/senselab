"""`explain`: a placed plan as text, one line per node after its hash and env builds."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from senselab_core.kernel.license import LicenseSet
from senselab_core.kernel.placement import NodePlacement, PlanPlacement
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.runtime.calibration import Estimate, LengthProfile
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.measures import resolve_roots
from senselab_core.runtime.placement import HOUR_S
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import cache_root
from senselab_core.runtime.site_placement import ClassNeeds, CurrentValues, UnmeasuredFigure
from senselab_core.runtime.sizing import Grant, Sizing, size

if TYPE_CHECKING:
    from senselab_core.runtime.execution.plan_reads import ItemCounts


def _licenses(found: LicenseSet) -> str:
    return ", ".join(f"{s} {'no record' if r is None else r.expression}" for s, r in found.records.items()) or "none"


def node_line(node_id: str, task: str, placement: PlanPlacement) -> str:
    """Return one node's backend, stack, env, device, batch, resource class, partition, licenses and reason."""
    placed: NodePlacement = placement.nodes[node_id]
    return (
        f"node {node_id} ({task}): backend {placed.backend!r} {placed.backend_version} (stack {placed.stack!r}, "
        f"env {placed.env}, device {placed.device}, batch {placed.batch_size}, resource class "
        f"{placed.resource_class}, partition {placed.partition or 'none'}); licenses: "
        f"{_licenses(placement.licenses[node_id])}; {placed.why}"
    )


def _source(estimate: Estimate) -> str:
    found = estimate.source
    return f"{len(found.probe_ids)} probes, {found.observations} observations"


def _labelled(value: float | None, unit: str, source: str | None, why: str) -> str:
    return f"unmeasured ({why})" if value is None or source is None else f"{value:.3g} {unit} (measured: {source})"


@dataclass(frozen=True)
class NodeSized:
    """What `size` gave a node: its grant, its sizing (None when unmeasured) and why a time figure may be missing."""

    estimate: Estimate | None
    grant: Grant
    sized: Sizing | None
    why: str


def sized_node(  # noqa: PLR0913 - the node's placement, the profile, the measures read, and what is known of it
    placed: NodePlacement,
    profile: ExecutionProfile,
    estimate: Estimate | None,
    work: LengthProfile | None,
    hardware: HardwareInfo | None,
    *,
    why_no_estimate: str,
    bucket_edges: Sequence[float] = (),
    why_no_lengths: str | None = None,
) -> NodeSized:
    """Return the grant and `size`'s sizing of a node, purely; `sized` is None without an estimate or its lengths."""
    resource = profile.resource_classes[placed.resource_class]
    cuda = resource.device == "cuda"
    devices = len(hardware.gpus) if hardware is not None and cuda and profile.substrate.kind == "local" else None
    grant = Grant.seen(resource, profile.pool, hardware, devices=devices)
    if estimate is None:
        return NodeSized(estimate, grant, None, why_no_estimate)
    if work is None:
        return NodeSized(estimate, grant, None, why_no_lengths or "lengths unknown until the inputs are read")
    sized = size(estimate, grant, profile.sizing, work, calibration=profile.calibration, bucket_edges=bucket_edges)
    return NodeSized(estimate, grant, sized, "a length is unknown or outside the measured range")


def node_sizing(  # noqa: PLR0913 - the node's placement, the profile, the measures read, and what is known of it
    placed: NodePlacement,
    profile: ExecutionProfile,
    estimate: Estimate | None,
    work: LengthProfile | None,
    hardware: HardwareInfo | None,
    *,
    why_no_estimate: str,
    unit_label: str | None = None,
    bucket_edges: Sequence[float] = (),
    why_no_lengths: str | None = None,
) -> list[str]:
    """Return the lines `size` gives a node: wall time, GPU hours and peak memory, each measured or unmeasured.

    `estimate` is what `MeasureSource.get` returned for the node's unit, None when there is none; then
    `why_no_estimate` is the reason printed, `unit_label` the unit's digest label when it resolved (never an
    identifier), `bucket_edges` the grid a bucketed formation cuts on, and `why_no_lengths` why `work` is None when
    the inputs could not be read (None: they are read when the run starts). Every figure is a field of the node's
    `Sizing`. Nothing here probes.
    """
    found = sized_node(
        placed,
        profile,
        estimate,
        work,
        hardware,
        why_no_estimate=why_no_estimate,
        bucket_edges=bucket_edges,
        why_no_lengths=why_no_lengths,
    )
    grant, sized, why = found.grant, found.sized, found.why
    cuda = profile.resource_classes[placed.resource_class].device == "cuda"
    source = None if estimate is None else _source(estimate)
    wall = None if sized is None else sized.wall_s
    gpu = wall * sized.workers / HOUR_S if wall is not None and sized is not None and cuda else None
    batch_peak = None if sized is None or not sized.measured.time else sized.batch_peak
    peak_why = "memory unmeasured" if sized is not None and sized.batch_peak is None else why
    kind = "device" if cuda else "host"
    figures = [
        f"wall {_labelled(wall, 's', source, why)}",
        *([f"gpu {_labelled(gpu, 'h', source, why)}"] if cuda else []),
        f"peak {kind} memory {_labelled(batch_peak, 'bytes', source, peak_why)}",
    ]
    lines = [f"  unit {unit_label}" if unit_label else "  unit unresolved", "  sizing: " + "; ".join(figures)]
    if estimate is not None and estimate.warmup_unsettled:
        lines.append(
            "  warm-up: requests 1 to 3 after a load differ from the later median beyond the spread; "
            "calibration.warmup_requests may be too short for this unit"
        )
    cap = grant.memory_bytes
    if sized is not None and sized.batch_one_peak is not None and cap is not None and sized.batch_one_peak > cap:
        lines.append(
            f"  needs {sized.batch_one_peak} bytes of {kind} memory at batch 1 for its longest job; "
            f"class {placed.resource_class} caps it at {cap} bytes"
        )
    return lines


def _worker_seconds(found: NodeSized, profile: ExecutionProfile) -> float | None:
    """Return the longest a worker of a timed node needs: the load plus its largest item at the upper prediction."""
    sized, estimate = found.sized, found.estimate
    if sized is None or estimate is None or estimate.single is None or not sized.measured.time:
        return None
    upper = 1.0 + profile.calibration.confidence * (estimate.single.spread or 0.0)
    items = [
        sum(m or 0.0 for m in sized.predicted_s[lo:hi]) * upper + (estimate.item_s or 0.0) for lo, hi in sized.cuts
    ]
    return (estimate.load_s or 0.0) + max(items, default=0.0)


def class_needs(name: str, profile: ExecutionProfile, current: CurrentValues, nodes: Sequence[NodeSized]) -> ClassNeeds:
    """Return what class `name` needs from its nodes' sizings: workers, worker time, wall time, GPU hours and memory.

    A figure no measure supports is None and named in `unmeasured` with the reason; an unmeasured class runs
    `calibration.unmeasured_workers` workers. The worker time is held within the class's `time_s`.
    """
    resource = profile.resource_classes[name]
    cuda = resource.device == "cuda"
    open_ = [n for n in nodes if n.sized is None or not n.sized.measured.time]
    why = open_[0].why if open_ else ""
    missing: list[UnmeasuredFigure] = []

    def need(figure: str, because: str) -> None:
        missing.append(UnmeasuredFigure(figure=figure, why=because))

    workers = profile.calibration.unmeasured_workers
    wall = per_worker = gpu = floor = memory = None
    if not open_:
        workers = max((n.sized.workers for n in nodes if n.sized is not None), default=0)
        times = [_worker_seconds(n, profile) for n in nodes]
        if None in times:
            need("time_per_worker_s", "a node's request fit is unmeasured")
        else:
            per_worker = max((t for t in times if t is not None), default=None)
            per_worker = (
                per_worker if per_worker is None or resource.time_s is None else min(per_worker, resource.time_s)
            )
        walls = [n.sized.wall_s for n in nodes if n.sized is not None]
        wall = None if None in walls else sum(w for w in walls if w is not None)
        gpu = (
            None
            if wall is None or not cuda
            else sum((n.sized.wall_s or 0.0) * n.sized.workers / HOUR_S for n in nodes if n.sized is not None)
        )
    else:
        for figure in ("workers", "time_per_worker_s", "wall_s", *(("gpu_hours",) if cuda else ())):
            need(figure, why)
        loads = [
            n.estimate.load_s + (n.estimate.item_s or 0.0)
            for n in nodes
            if n.estimate is not None and n.estimate.load_s is not None
        ]
        floor = max(loads, default=None)
    if cuda:
        peaks = [n.sized.batch_peak if n.sized is not None else None for n in nodes]
        if peaks and None not in peaks:
            memory = max(p for p in peaks if p is not None)
        else:
            memory = resource.memory_cap_bytes
            need("device_memory_bytes", "memory unmeasured" + ("; the class's cap is given" if memory else ""))
    return ClassNeeds(
        resource_class=name,
        device=resource.device,
        gpu_type=resource.gpu_type,
        device_memory_bytes=memory,
        cpus=resource.cpus,
        mem_bytes=resource.mem_bytes,
        workers=workers,
        workers_max=current.max_workers,
        time_per_worker_s=per_worker,
        time_floor_s=floor if per_worker is None else None,
        wall_s=wall,
        gpu_hours=gpu,
        unmeasured=tuple(missing),
        current=current,
    )


def item_lines(counts: Mapping[str, "ItemCounts"]) -> list[str]:
    """Return one line per resource class: its jobs the store would serve, those computed, and any it cannot tell."""
    lines = []
    for name, found in sorted(counts.items()):
        parts = [f"{found.servable} servable", f"{found.computed} computed"]
        if found.undetermined:
            parts.append(f"{found.undetermined} undetermined ({'; '.join(found.why)})")
        if found.later:
            parts.append(f"nodes {', '.join(found.later)} wait on an upstream result")
        lines.append(f"class {name}: " + ", ".join(parts))
    return lines


def undetermined_lines(placement: PlanPlacement, why: str) -> list[str]:
    """Return one line per resource class of `placement` whose servable and computed jobs cannot be told, and why."""
    classes = sorted({p.resource_class for p in placement.nodes.values()})
    return [f"class {name}: servable and computed undetermined ({why})" for name in classes]


def waits(profile: ExecutionProfile) -> list[str]:
    """Return where measures are written and the waits a run may incur on another run (lease takeover, peer probe)."""
    roots = resolve_roots(profile.calibration.measures_roots, cache_root())
    return [
        f"measures are written to the first writable of: {', '.join(str(r) for r in roots) or 'none'}",
        f"waits on another run: a lease is taken over after {profile.lease.stale_after_s:g} s; a peer's probe of the "
        f"same unit is waited for up to {profile.peer_wait_s:g} s",
    ]


def explain(
    plan_hash: str,
    logical: LogicalPlan,
    placement: PlanPlacement,
    *,
    sizing: Mapping[str, Sequence[str]] | None = None,
    footer: Sequence[str] = (),
) -> str:
    """Return the plan hash and envs to build, then per node in graph order its placement, licenses and sizing.

    `sizing` gives each node's lines after its own (`node_sizing`); `footer` closes the text (`waits`).
    """
    builds = ", ".join(placement.env_builds) or "none"
    lines = [
        f"plan {plan_hash} under profile {placement.profile_digest[:12]}",
        f"envs to build: {builds}",
    ]
    for n in logical.graph.nodes:
        if n.id in placement.nodes:
            lines += [node_line(n.id, n.task, placement), *(sizing or {}).get(n.id, ())]
    return "\n".join([*lines, *footer])
