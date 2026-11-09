"""`explain`: a placed plan as text, one line per node after its hash, env builds and cost."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from senselab_core.kernel.license import LicenseSet
from senselab_core.kernel.placement import Cost, NodePlacement, PlanPlacement
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.runtime.calibration import Estimate, LengthProfile
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.measures import resolve_roots
from senselab_core.runtime.placement import _HOUR_S
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import cache_root
from senselab_core.runtime.sizing import Grant, size

if TYPE_CHECKING:
    from senselab_core.runtime.execution.plan_reads import ItemCounts


def _cost(cost: Cost) -> str:
    return f"download {'unmeasured' if cost.download_bytes is None else f'{cost.download_bytes:g} bytes'}"


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
    if found.commit is not None:
        return f"packaged row {found.commit[:12]}"
    return f"{len(found.probe_ids)} probes, {found.observations} observations"


def _labelled(value: float | None, unit: str, source: str | None, why: str) -> str:
    return f"unmeasured ({why})" if value is None or source is None else f"{value:.3g} {unit} (measured: {source})"


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
) -> list[str]:
    """Return the lines `size` gives a node: wall time, GPU hours and peak memory, each measured or unmeasured.

    `estimate` is what `MeasureSource.get` returned for the node's unit, None when there is none; then
    `why_no_estimate` is the reason printed, `unit_label` the unit's digest label when it resolved (never an
    identifier) and `bucket_edges` the grid a bucketed formation cuts on. Every figure is a field of the node's
    `Sizing`. Nothing here probes.
    """
    resource = profile.resource_classes[placed.resource_class]
    cuda = resource.device == "cuda"
    devices = len(hardware.gpus) if hardware is not None and cuda and profile.substrate.kind == "local" else None
    grant = Grant.seen(resource, profile.pool, hardware, devices=devices)
    if estimate is None:
        why, sized = why_no_estimate, None
    elif work is None:
        why, sized = "lengths unknown until the inputs are read", None
    else:
        why = "a length is unknown or outside the measured range"
        sized = size(estimate, grant, profile.sizing, work, calibration=profile.calibration, bucket_edges=bucket_edges)
    source = None if estimate is None else _source(estimate)
    wall = None if sized is None else sized.wall_s
    gpu = wall * sized.workers / _HOUR_S if wall is not None and sized is not None and cuda else None
    batch_peak = None if sized is None or not sized.measured.time else sized.batch_peak
    peak_why = "memory unmeasured" if sized is not None and sized.batch_peak is None else why
    kind = "device" if cuda else "host"
    figures = [
        f"wall {_labelled(wall, 's', source, why)}",
        *([f"gpu {_labelled(gpu, 'h', source, why)}"] if cuda else []),
        f"peak {kind} memory {_labelled(batch_peak, 'bytes', source, peak_why)}",
    ]
    lines = [f"  unit {unit_label}" if unit_label else "  unit unresolved", "  sizing: " + "; ".join(figures)]
    cap = grant.memory_bytes
    if sized is not None and sized.batch_one_peak is not None and cap is not None and sized.batch_one_peak > cap:
        lines.append(
            f"  needs {sized.batch_one_peak} bytes of {kind} memory at batch 1 for its longest job; "
            f"class {placed.resource_class} caps it at {cap} bytes"
        )
    return lines


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
    """Return the plan hash, envs to build and cost, then per node in graph order its placement, licenses and cost.

    `sizing` gives each node's lines after its own (`node_sizing`); `footer` closes the text (`waits`).
    """
    builds = ", ".join(placement.env_builds) or "none"
    lines = [
        f"plan {plan_hash} under profile {placement.profile_digest[:12]}",
        f"envs to build: {builds}",
        f"cost: {_cost(placement.cost)}",
    ]
    for n in logical.graph.nodes:
        if n.id in placement.nodes:
            lines += [node_line(n.id, n.task, placement), *(sizing or {}).get(n.id, ())]
    return "\n".join([*lines, *footer])
