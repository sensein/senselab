"""Doubles for the pool's estimate and unit reads: one estimate for every unit, and one unit for every call."""

from collections.abc import Mapping, Sequence
from typing import Any

from senselab_core.kernel.declare import BackendSpec
from senselab_core.runtime.calibration import Estimate, EstimateSource, HardwareClass, RequestFit, StratumBatch
from senselab_core.runtime.keys import Call, Placement
from senselab_core.runtime.measures import MeasureUnit
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.units import EnvStates, Unresolved
from senselab_core.testing.calibration_samples import UNIT

CEILED = Estimate(
    status="measured",
    unit_kind="none",
    load_s=1e-3,
    load_n=3,
    item_s=0.0,
    item_n=3,
    single=RequestFit(batch=1, r=1e-6, u=None, n=6, spread=0.1, lo=0.0, hi=0.0),
    batch=(StratumBatch(lo=0.0, hi=0.0, batch=1),),
    source=EstimateSource(probe_ids=("p1",), observations=6),
)
"""An estimate under which the pool's quantum ceilings bind: each request far under a second, so a queue wants a
worker per item once `CEILED_SIZING` shortens the target."""

CEILED_SIZING = {"target_wall_s": 1e-12}
"""A `sizing` override under which a queue wants a worker per quantum of `CEILED` items: a target that short."""


class FixedEstimates:
    """Gives every unit `estimate` (None: every unit is unmeasured); the read of `MeasureSource.get`."""

    def __init__(self, estimate: Estimate | None = CEILED) -> None:
        """Give `estimate` for every unit."""
        self.estimate = estimate

    def get(self, unit: MeasureUnit) -> Estimate | None:  # noqa: ARG002 - one estimate for every unit
        """Return the estimate."""
        return self.estimate


class FixedUnit:
    """A `UnitResolver` whose every call is the unit `unit` (or the `Unresolved` it is given), over `profile`'s envs."""

    def __init__(self, profile: ExecutionProfile, unit: MeasureUnit | Unresolved = UNIT) -> None:
        """Resolve every call to `unit`; `envs` reads `profile`'s envs as the pool's servable reads do."""
        self.unit_given, self.envs = unit, EnvStates(profile)
        self.calls: list[tuple[Call, Placement, str | None, HardwareClass | None]] = []

    def unit(
        self,
        call: Call,
        placement: Placement,
        decoder: str | None = None,
        hardware: HardwareClass | None = None,
        env: Mapping[str, Any] | None = None,
    ) -> MeasureUnit | Unresolved:
        """Return the unit given, noting the call."""
        self.calls.append((call, placement, decoder, hardware))
        return self.unit_given

    def backends(self) -> Sequence[BackendSpec]:
        """Return the backends of the calls seen."""
        return [p.backend for _, p, _, _ in self.calls]
