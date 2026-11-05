"""The placement `placement@1.0`: where each node of a logical plan runs, the licenses it carries and its cost.

A placement is never part of an identity: a plan placed on another site keeps its hash.
"""

from collections.abc import Mapping
from typing import Annotated, ClassVar

from pydantic import AfterValidator, Field, PlainSerializer, StringConstraints, model_validator

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, thawed
from senselab_core.kernel.formats import PLACEMENT_FORMAT, FormatTag, Formatted
from senselab_core.kernel.graph import NodeId
from senselab_core.kernel.identity import Params
from senselab_core.kernel.license import LicenseSet

ResourceClassName = Annotated[str, StringConstraints(pattern=r"^(cpu|mps|gpu)(:[a-z0-9_.-]+)?$|^svc:[a-z0-9_.-]+$")]
"""A resource class name: `cpu`, `mps`, `gpu` or `gpu:<type>`, or `svc:<id>`, each optionally `:<variant>`."""

Figure = Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
"""A cost figure; None when it is not measured."""


def _frozen(v: Mapping[str, object]) -> Mapping[str, object]:
    return FrozenMapping(v)


class NodePlacement(Frozen):
    """Where one node runs: its backend and params, stack, env, device, batch size, resource class and partition.

    `env` is `host` or `<kind>:<name>`; `env_root` is the directory its env is built under (None for a host env);
    `partition` is the partition as the profile names it (None off a scheduler); `why` is the reason for the choice.
    """

    backend: NonEmptyStr
    backend_version: NonEmptyStr
    backend_params: Params
    stack: NonEmptyStr
    env: NonEmptyStr
    device: NonEmptyStr
    batch_size: Annotated[int, Field(ge=1)]
    resource_class: ResourceClassName
    partition: NonEmptyStr | None = None
    env_root: NonEmptyStr | None = None
    why: str = ""


class Cost(Frozen):
    """A plan's measured cost: wall seconds, GPU hours and bytes to download.

    A figure is None when a node or model it needs has no resource table, or an input it needs is not known;
    `unmeasured` names every node and model without a table.
    """

    wall_s: Figure = None
    gpu_hours: Figure = None
    download_bytes: Annotated[int, Field(ge=0)] | None = None
    unmeasured: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _check_unmeasured(self) -> "Cost":
        if list(self.unmeasured) != sorted(set(self.unmeasured)):
            raise ValueError("unmeasured names each node or model once, sorted")
        return self


class PlanPlacement(Formatted):
    """A `placement@1.0` document: each node's placement and license set, the envs to build and the cost.

    `profile_digest` is the canonical digest of the execution profile placed under; `nodes` and `licenses` name the
    same node ids; `env_builds` names the envs, as `<kind>:<name>`, not yet built.
    """

    format: FormatTag = PLACEMENT_FORMAT
    port_value: ClassVar[bool] = False
    profile_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    nodes: Annotated[Mapping[NodeId, NodePlacement], AfterValidator(_frozen), PlainSerializer(thawed)]
    env_builds: tuple[NonEmptyStr, ...] = ()
    licenses: Annotated[Mapping[NodeId, LicenseSet], AfterValidator(_frozen), PlainSerializer(thawed)]
    cost: Cost

    @model_validator(mode="after")
    def _check_nodes(self) -> "PlanPlacement":
        if set(self.licenses) != set(self.nodes):
            raise ValueError(f"licenses name nodes {sorted(self.licenses)}; the placement has {sorted(self.nodes)}")
        return self
