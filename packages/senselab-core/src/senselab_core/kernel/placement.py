"""The placement `placement@3.0`: where each node of a logical plan runs and the licenses it carries.

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


def _frozen(v: Mapping[str, object]) -> Mapping[str, object]:
    return FrozenMapping(v)


class NodePlacement(Frozen):
    """Where one node runs: its backend and params, stack, env, device, batch size, resource class and partition.

    `env` is `host` or `<kind>:<name>`, built under the directory the worker's own profile gives; `partition` is the
    partition as the profile names it (None off a scheduler); `why` is the reason for the choice.
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
    why: str = ""


class PlanPlacement(Formatted):
    """A `placement@3.0` document: each node's placement and license set and the envs to build.

    `profile_digest` is the canonical digest of the execution profile placed under; `nodes` and `licenses` name the
    same node ids; `env_builds` names the envs, as `<kind>:<name>`, not yet built.
    """

    format: FormatTag = PLACEMENT_FORMAT
    port_value: ClassVar[bool] = False
    profile_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    nodes: Annotated[Mapping[NodeId, NodePlacement], AfterValidator(_frozen), PlainSerializer(thawed)]
    env_builds: tuple[NonEmptyStr, ...] = ()
    licenses: Annotated[Mapping[NodeId, LicenseSet], AfterValidator(_frozen), PlainSerializer(thawed)]

    @model_validator(mode="after")
    def _check_nodes(self) -> "PlanPlacement":
        if set(self.licenses) != set(self.nodes):
            raise ValueError(f"licenses name nodes {sorted(self.licenses)}; the placement has {sorted(self.nodes)}")
        return self
