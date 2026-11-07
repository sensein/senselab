"""A `MeasureSource` for tests: one measure for every unit, and a record of what calibration rounds put."""

from senselab_core.runtime.resources import Measured, ResourceMeasure, Unit

CEILED = ResourceMeasure(load_s=1e-3, resident_bytes=0, peak_memory_bytes=1, s_per_input_s=1e-6, input_s_per_item=None)
"""A measure under which the pool's quantum ceilings bind: a load many items long, each item far under a second."""

CEILED_SIZING = {"target_wall_s": 1e-12}
"""A `sizing` override under which a queue wants a worker per quantum of `CEILED` items: a target that short."""


class FixedMeasures:
    """Gives every unit `measure` (None: every unit is unmeasured, so pools calibrate); keeps each `put`."""

    def __init__(self, measure: ResourceMeasure | None = CEILED) -> None:
        """Give `measure` for every unit."""
        self.measure = measure
        self.puts: list[tuple[Unit, ResourceMeasure, Measured]] = []

    def get(self, unit: Unit) -> ResourceMeasure | None:
        """Return the measure, or a measure put for `unit` since."""
        put = [m for u, m, _ in self.puts if u == unit]
        return put[-1] if put else self.measure

    def put(self, unit: Unit, measure: ResourceMeasure, measured: Measured) -> None:
        """Keep `measure` of `unit`."""
        self.puts.append((unit, measure, measured))
