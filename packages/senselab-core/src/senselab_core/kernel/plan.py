"""The logical plan `logical-plan@1.0`: a graph with every resolution it runs under, and its hash, the run id.

A plan holds the graph, each node's pinned model, each node's profile slices keyed by identity, the digest of the
input manifest, the input binding and the reuse policy. It holds nothing about where or how it runs.
"""

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal

from pydantic import AfterValidator, BeforeValidator, PlainSerializer, StringConstraints, model_validator
from senselab_worker.digest import identity_digest

from senselab_core.kernel._base import FrozenMapping, NonEmptyStr, thawed
from senselab_core.kernel.formats import LOGICAL_PLAN_FORMAT, FormatTag, Formatted, later_minor
from senselab_core.kernel.graph import Graph, NodeId, PortName, graph_digest
from senselab_core.kernel.identity import ProfileSliceIdentity, ResolvedModel, ReusePolicy

PLAN_HASH_TAG = "logical-plan@1"
"""The identity tag of a logical plan's canonical JSON."""
PROFILE_SLICE_TAG = "profile-slice@1"
"""The identity tag of a profile slice's value."""


def slice_identity(value: object) -> ProfileSliceIdentity:
    """Return `"profile:" + identity_digest("profile-slice@1", value)` for a slice's JSON value."""
    return "profile:" + identity_digest(PROFILE_SLICE_TAG, thawed(value))


def _frozen(v: Mapping[str, Any]) -> Mapping[str, Any]:
    return FrozenMapping(v)


def _policy_name(v: object) -> object:
    return v.value if isinstance(v, ReusePolicy) else v


ReuseName = Annotated[Literal["exact", "equivalent", "tolerant"], BeforeValidator(_policy_name)]
"""A `ReusePolicy` by its value; given a `ReusePolicy`, its value is held."""

Slices = Annotated[Mapping[str, ProfileSliceIdentity], AfterValidator(_frozen), PlainSerializer(dict)]


class LogicalPlan(Formatted):
    """A `logical-plan@1.0` document: the graph and every resolution of it.

    `models` gives each node's pinned model (None for a node without one) and `profile_slices` each node's declared
    slices by name; both have exactly the graph's node ids. `manifest_digest` is `InputManifest.digest()` of the
    records the graph runs over, `bind` maps graph inputs to the record inputs they read, and `reuse` is the value of
    the widest `ReusePolicy` under which a stored result serves a node.
    """

    format: FormatTag = LOGICAL_PLAN_FORMAT
    port_value: ClassVar[bool] = False
    graph: Graph
    models: Annotated[Mapping[NodeId, ResolvedModel | None], AfterValidator(_frozen), PlainSerializer(thawed)]
    profile_slices: Annotated[Mapping[NodeId, Slices], AfterValidator(_frozen), PlainSerializer(thawed)]
    manifest_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    bind: Annotated[Mapping[PortName, NonEmptyStr], AfterValidator(_frozen), PlainSerializer(dict)] = FrozenMapping({})
    reuse: ReuseName

    @model_validator(mode="after")
    def _check_nodes(self) -> "LogicalPlan":
        ids = {n.id for n in self.graph.nodes}
        for name, given in (("models", self.models), ("profile_slices", self.profile_slices)):
            if set(given) != ids:
                raise ValueError(f"{name} names nodes {sorted(given)}; the graph has {sorted(ids)}")
        unknown = sorted(set(self.bind) - set(self.graph.inputs))
        if unknown:
            raise ValueError(f"bind names {unknown}, which are not graph inputs")
        return self


def plan_hash(plan: LogicalPlan) -> str:
    """Return `identity_digest("logical-plan@1", fields)`, the run id, over every field but `format`.

    The graph enters as the `graph_digest` of its nodes without their `model` references, and each model as its
    `key_identity()` (hub repo and commit, or the weights digest), so neither the reference text, the source path nor
    the license reaches the hash.

    Raises:
        ValueError: if the plan, or its graph, declares a later minor than this reader's.
        UnpinnedModel: if a model is not pinned.
    """
    if later_minor(LOGICAL_PLAN_FORMAT, plan.format):
        raise ValueError(
            f"a {plan.format} plan has fields a {LOGICAL_PLAN_FORMAT} reader does not hold; it has no hash"
        )
    nodes = tuple(n.model_copy(update={"model": None}) for n in plan.graph.nodes)
    bare = plan.graph.model_copy(update={"nodes": nodes})
    pins = {n: None if m is None else m.key_identity().model_dump(mode="json") for n, m in plan.models.items()}
    form: dict[str, Any] = plan.model_dump(mode="json", exclude={"format", "graph", "models"})
    return identity_digest(PLAN_HASH_TAG, {**form, "models": pins, "graph": graph_digest(bare)})
