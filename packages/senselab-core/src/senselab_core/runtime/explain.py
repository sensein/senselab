"""`explain`: a placed plan as text, one line per node after its hash, env builds and cost."""

from senselab_core.kernel.license import LicenseSet
from senselab_core.kernel.placement import Cost, NodePlacement, PlanPlacement
from senselab_core.kernel.plan import LogicalPlan


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


def _node(node_id: str, task: str, placement: PlanPlacement) -> str:
    placed: NodePlacement = placement.nodes[node_id]
    cost = "unmeasured" if node_id in placement.cost.unmeasured else "measured"
    return (
        f"node {node_id} ({task}): backend {placed.backend!r} {placed.backend_version} (stack {placed.stack!r}, "
        f"env {placed.env}, device {placed.device}, batch {placed.batch_size}, resource class "
        f"{placed.resource_class}, partition {placed.partition or 'none'}); licenses: "
        f"{_licenses(placement.licenses[node_id])}; cost {cost}; {placed.why}"
    )


def explain(plan_hash: str, logical: LogicalPlan, placement: PlanPlacement) -> str:
    """Return the plan hash, envs to build and cost, then per node in graph order its placement, licenses and cost."""
    builds = ", ".join(placement.env_builds) or "none"
    lines = [
        f"plan {plan_hash} under profile {placement.profile_digest[:12]}",
        f"envs to build: {builds}",
        f"cost: {_cost(placement.cost)}",
    ]
    lines += [_node(n.id, n.task, placement) for n in logical.graph.nodes if n.id in placement.nodes]
    return "\n".join(lines)
