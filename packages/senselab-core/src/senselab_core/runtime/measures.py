"""What a measure is of, and where a unit's estimate is read from.

A `MeasureUnit` names the task, backend, parameters, model, env, service and hardware a measure holds for; its digest
is the key every measures root and packaged table is looked up by. `MeasureSource` reads a unit's `Estimate` from the
measures roots, then the packaged table.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Protocol

from pydantic import Field
from senselab_worker.digest import identity_digest

from senselab_core.kernel._base import Frozen, NonEmptyStr, StrMap
from senselab_core.kernel.formats import MEASURE_OBSERVATIONS_FORMAT, MEASURE_UNIT_TAG
from senselab_core.kernel.identity import Attestation, ModelIdentity, Params
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.calibration import Estimate, HardwareClass, Observation

Formation = Annotated[str, Field(pattern=r"^(mixed|single|bucket:\S+)$")]
"""How a backend forms batches: `mixed`, `single`, or `bucket:<grid id>`."""

OBSERVATION_FORMAT = MEASURE_OBSERVATIONS_FORMAT.split(".")[0]
"""The observation format an identity names, as `<name>@<major>`: an observation of another cannot be pooled."""


class MeasureEnv(Frozen):
    """The env's fingerprinted context fields that decide cost: its lock and wheels, driver, platform and numerics."""

    env_lock: NonEmptyStr
    wheel_set: NonEmptyStr
    driver: StrMap
    platform: NonEmptyStr
    decoder: NonEmptyStr | None
    numerics: StrMap


class MeasureUnit(Frozen):
    """What a measure is of. The input lengths, the senselab source, the regime and the run are not part of it."""

    task: NonEmptyStr
    task_version: NonEmptyStr
    backend: NonEmptyStr
    backend_version: NonEmptyStr
    backend_params: Params
    formation: Formation
    model: ModelIdentity | None
    env: MeasureEnv
    service: Attestation | None
    hardware: HardwareClass
    observation_format: NonEmptyStr = OBSERVATION_FORMAT

    @classmethod
    def of(  # noqa: PLR0913 - the identity's parts, each from a different owner
        cls,
        placement: NodePlacement,
        model: ModelIdentity | None,
        env_fields: Mapping[str, object],
        hardware: HardwareClass,
        *,
        task: str,
        task_version: str,
        formation: str,
        service: Attestation | None = None,
    ) -> "MeasureUnit":
        """Return the unit of `placement`'s backend and parameters, `model`, the env's `env_fields` and `hardware`.

        `env_fields` holds `env_lock`, `wheel_set`, `driver` (`env.context_fields()`), `platform`, `decoder` and
        `numerics` (the worker's ready mark); a missing field raises `ValueError`.
        """
        names = set(MeasureEnv.model_fields)
        if missing := sorted(names - set(env_fields)):
            raise ValueError(f"env fields lack {missing}")
        return cls(
            task=task,
            task_version=task_version,
            backend=placement.backend,
            backend_version=placement.backend_version,
            backend_params=placement.backend_params,
            formation=formation,
            model=model,
            env=MeasureEnv.model_validate({k: env_fields[k] for k in names}),
            service=service,
            hardware=hardware,
        )

    def digest(self) -> str:
        """Return `identity_digest("measure-unit@1", fields)`."""
        return identity_digest(MEASURE_UNIT_TAG, self.model_dump(mode="json"))


class MeasuresRoot(Protocol):
    """A measures root, read: the observations pooled for a unit."""

    def observations(self, unit: MeasureUnit) -> Sequence[Observation]:
        """Return the observations of `unit` that still pool."""


class PackagedMeasures(Protocol):
    """The packaged table, read: a unit's estimate on exact identity."""

    def get(self, unit: MeasureUnit) -> Estimate | None:
        """Return the estimate of the row whose unit is `unit`, None when there is none."""


class MeasureSource:
    """Reads a unit's estimate: the measures roots in order, then the packaged table; the first `measured` wins."""

    def __init__(
        self,
        sources: Sequence[MeasuresRoot],
        packaged: PackagedMeasures,
        *,
        estimate: Callable[[MeasureUnit, Sequence[Observation]], Estimate],
    ) -> None:
        """Read `sources` in order, then `packaged`; `estimate` derives an estimate from a root's observations."""
        self._sources = tuple(sources)
        self._packaged = packaged
        self._estimate = estimate

    def get(self, unit: MeasureUnit) -> Estimate | None:
        """Return the first measured estimate of `unit`, None when the Pool must probe it."""
        for root in self._sources:
            observed = root.observations(unit)
            if observed and (found := self._estimate(unit, observed)).status == "measured":
                return found
        row = self._packaged.get(unit)
        return row if row is not None and row.status == "measured" else None
