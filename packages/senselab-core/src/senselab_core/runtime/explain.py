"""`explain`: a placed plan as text, one line per node after its hash, env builds and cost."""

from collections.abc import Mapping, Sequence

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


def _figure(value: float | None, unit: str) -> str:
    return "unmeasured" if value is None else f"{value:g} {unit}"


def _cost(cost: Cost) -> str:
    figures = (
        f"wall {_figure(cost.wall_s, 's')}, gpu {_figure(cost.gpu_hours, 'h')}, "
        f"download {_figure(cost.download_bytes, 'bytes')}"
    )
    return figures + (f"; unmeasured: {', '.join(cost.unmeasured)}" if cost.unmeasured else "")


def _licenses(found: LicenseSet) -> str:
    return ", ".join(f"{s} {'no record' if r is None else r.expression}" for s, r in found.records.items()) or "none"


def node_line(node_id: str, task: str, placement: PlanPlacement) -> str:
    """Return one node's backend, stack, env, device, batch, resource class, partition, licenses, cost and reason."""
    placed: NodePlacement = placement.nodes[node_id]
    cost = "unmeasured" if node_id in placement.cost.unmeasured else "measured"
    return (
        f"node {node_id} ({task}): backend {placed.backend!r} {placed.backend_version} (stack {placed.stack!r}, "
        f"env {placed.env}, device {placed.device}, batch {placed.batch_size}, resource class "
        f"{placed.resource_class}, partition {placed.partition or 'none'}); licenses: "
        f"{_licenses(placement.licenses[node_id])}; cost {cost}; {placed.why}"
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
    unit: bool,
) -> list[str]:
    """Return the lines `size` gives a node: wall time, GPU hours and peak memory, each measured or unmeasured.

    `estimate` is what `MeasureSource.get` returned for the node's unit, None when `unit` is False (no unit can be
    named before the env is built and the hardware seen) or no estimate exists. Nothing here probes.
    """
    resource = profile.resource_classes[placed.resource_class]
    cuda = resource.device == "cuda"
    devices = len(hardware.gpus) if hardware is not None and cuda and profile.substrate.kind == "local" else None
    host = hardware.memory_bytes if hardware is not None and not cuda and resource.mem_bytes is None else None
    grant = Grant.of(resource, profile.pool, host, on_device=cuda, devices=devices)
    if estimate is None:
        why = "no estimate for its unit" if unit else "no measure unit before the env is built and the hardware seen"
        sized = None
    elif work is None:
        why, sized = "lengths unknown until the inputs are read", None
    else:
        why = "a length is unknown or outside the measured range"
        sized = size(estimate, grant, profile.sizing, work, calibration=profile.calibration)
    source = None if estimate is None else _source(estimate)
    wall = None if sized is None else sized.wall_s
    gpu = wall * sized.workers / _HOUR_S if wall is not None and sized is not None and cuda else None
    fit = None if sized is None else sized.memory
    longest = 0.0 if work is None else max((u for u in work.units if u is not None), default=0.0)
    known = sized is not None and fit is not None and sized.measured.time
    peak = fit.resident + fit.m0 + fit.m1 * sized.batch_at(longest) * longest if known and sized and fit else None
    peak_why = "memory unmeasured" if sized is not None and fit is None else why
    kind = "device" if cuda else "host"
    figures = [
        f"wall {_labelled(wall, 's', source, why)}",
        *([f"gpu {_labelled(gpu, 'h', source, why)}"] if cuda else []),
        f"peak {kind} memory {_labelled(peak, 'bytes', source, peak_why)}",
    ]
    lines = ["  sizing: " + "; ".join(figures)]
    cap = grant.memory_bytes
    if sized is not None and sized.batch_one_peak is not None and cap is not None and sized.batch_one_peak > cap:
        lines.append(
            f"  needs {sized.batch_one_peak} bytes of {kind} memory at batch 1 for its longest job; "
            f"class {placed.resource_class} caps it at {cap} bytes"
        )
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
