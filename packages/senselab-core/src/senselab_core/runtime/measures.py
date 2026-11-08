"""What a measure is of, and where a unit's estimate is read from.

A `MeasureUnit` names the task, backend, parameters, model, env, service and hardware a measure holds for; its digest
is the key every measures root and packaged table is looked up by. `MeasureSource` reads a unit's `Estimate` from the
measures roots (pooled), then the packaged table.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Protocol, Self
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field, PlainSerializer, model_validator
from senselab_worker.digest import identity_digest

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, StrMap
from senselab_core.kernel.formats import MEASURE_OBSERVATIONS_FORMAT, MEASURE_UNIT_TAG
from senselab_core.kernel.identity import LogicalKey, ModelIdentity, Params, ProfileSliceIdentity
from senselab_core.runtime.calibration import Estimate, HardwareClass, Observation, Positive
from senselab_core.runtime.keys import Placement

DRIVER_KEYS = frozenset({"torch_build", "engine", "engine_version"})
"""The driver entries a measure holds."""
_UNMEASURED_DRIVER_KEYS = frozenset({"launch_env", "served_model", "model_identity", "config_digest", "observed"})
_BUILD_TAG = re.compile(r"cu\d+|cpu|rocm[\d.]+|xpu")

Formation = Annotated[str, Field(pattern=r"^(mixed|single|bucket:\S+)$")]
"""How a backend forms batches: `mixed`, `single`, or `bucket:<grid id>`."""

OBSERVATION_FORMAT = MEASURE_OBSERVATIONS_FORMAT.split(".")[0]
"""The observation format an identity names, as `<name>@<major>`: an observation of another cannot be pooled."""


class MeasureEnv(Frozen):
    """The env's fingerprinted context fields that decide cost: its lock and wheels, driver, platform and numerics.

    The driver holds only `DRIVER_KEYS`: the torch build (`cu128`, `cpu`) in place of its index URL, and for a server
    the engine and its version. `threads` are the worker launch's measured thread counts (`Launch.measured`).
    """

    env_lock: NonEmptyStr
    wheel_set: NonEmptyStr
    driver: StrMap
    platform: NonEmptyStr
    decoder: NonEmptyStr | None
    numerics: StrMap
    threads: Annotated[Mapping[str, Positive], AfterValidator(FrozenMapping), PlainSerializer(dict)]

    @model_validator(mode="after")
    def _check_driver(self) -> Self:
        if undeclared := sorted(set(self.driver) - DRIVER_KEYS):
            raise ValueError(f"driver holds {undeclared}; a measure holds only {sorted(DRIVER_KEYS)}")
        return self

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> Self:
        """Return the env of `fields` as `EnvHandle.context_fields()`, the ready mark and `Launch.measured` give them.

        The driver's `torch_index` URL is replaced by its build tag; its launch digest and attestation are dropped.

        Raises:
            ValueError: a field is missing.
        """
        if missing := sorted(set(cls.model_fields) - set(fields)):
            raise ValueError(f"env fields lack {missing}")
        driver = {k: v for k, v in fields["driver"].items() if k not in _UNMEASURED_DRIVER_KEYS}
        if (index := driver.pop("torch_index", None)) is not None:
            tag = urlsplit(index).path.rstrip("/").rsplit("/", 1)[-1]
            driver["torch_build"] = tag if _BUILD_TAG.fullmatch(tag) else "other"
        return cls.model_validate({**{k: fields[k] for k in cls.model_fields}, "driver": driver})


class MeasureUnit(Frozen):
    """What a measure is of: the logical key without its inputs, plus env, hardware and formation.

    The input lengths, the senselab source, the regime and the run are not part of it.
    """

    task: NonEmptyStr
    task_version: NonEmptyStr
    backend: NonEmptyStr
    backend_version: NonEmptyStr
    params: Params
    backend_params: Params
    profile_slices: Annotated[
        Mapping[str, ProfileSliceIdentity], AfterValidator(FrozenMapping), PlainSerializer(dict)
    ] = FrozenMapping({})
    formation: Formation
    chunk_grid: NonEmptyStr | None
    model: ModelIdentity | None
    env: MeasureEnv
    service: NonEmptyStr | None
    hardware: HardwareClass
    observation_format: NonEmptyStr = OBSERVATION_FORMAT

    @classmethod
    def of(
        cls, placement: Placement, key: LogicalKey, env_fields: Mapping[str, Any], hardware: HardwareClass
    ) -> "MeasureUnit":
        """Return the unit of the call `key` run as `placement` in the env `env_fields` on `hardware`.

        `env_fields` holds `env_lock`, `wheel_set`, `driver` (`env.context_fields()`), `platform`, `decoder` and
        `numerics` (the worker's ready mark), and `threads` (`Launch.measured`, `{}` for a host env). The service is
        the attestation's `config_digest`.

        Raises:
            ValueError: an env field is missing, or the placement's formation is not yet decided.
        """
        if placement.batch_formation is None:
            raise ValueError("the placement's batch formation is not decided; each job's length picks it")
        return cls(
            task=key.task,
            task_version=key.task_version,
            backend=key.backend,
            backend_version=key.backend_version,
            params=key.params,
            backend_params=key.backend_params,
            profile_slices=key.profile_slices,
            formation=placement.batch_formation,
            chunk_grid=placement.chunk_grid,
            model=key.model,
            env=MeasureEnv.from_fields(env_fields),
            service=None if key.service is None else key.service.config_digest,
            hardware=hardware,
        )

    def digest(self) -> str:
        """Return `identity_digest("measure-unit@1", fields)`."""
        return identity_digest(MEASURE_UNIT_TAG, self.model_dump(mode="json"))


class MeasuresRoot(Protocol):
    """A measures root, read: the observations of a unit that still pool (none of a superseded probe)."""

    def observations(self, unit: MeasureUnit) -> Sequence[Observation]:
        """Return the observations of `unit` that still pool."""


class PackagedMeasures(Protocol):
    """The packaged table, read: a unit's estimate on exact identity."""

    def get(self, unit: MeasureUnit) -> Estimate | None:
        """Return the estimate of the row whose unit is `unit`, None when there is none."""


class MeasureSource:
    """Reads a unit's estimate: the roots' pooled observations estimated once, then the packaged table."""

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
        """Return the estimate of `unit` over the observations of every root, else the packaged one, else None."""
        pooled = [o for root in self._sources for o in root.observations(unit)]
        if pooled and (found := self._estimate(unit, pooled)).status == "measured":
            return found
        row = self._packaged.get(unit)
        return row if row is not None and row.status == "measured" else None
