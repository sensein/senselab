"""The ``seed_band`` expectation: a stored headline judged against the spread of independent chains.

With ``floor`` the metric's chance floor, ``lift = stored - floor`` (signed) and ``sd`` the sample standard
deviation of the metric over the band's chains, the half-width is
``h = max(a_min, k * sd, rho * |lift|)``. The verdict, in order:

- ``characterise`` when the band declares ``characterise: true`` or ``lift <= a_min / max_width_frac``
  (at or near chance);
- ``fail`` when ``|stored - mean| > h``;
- ``uninformative`` when ``h > max_width_frac * lift``;
- ``pass`` otherwise.

``mean`` is the step's own metric value (the gather step's adapter emits the chain mean); the driver
recomputes the chain mean and returns ``E_BAND_MEAN`` when the two differ by more than a relative ``1e-9``.
For an inference row with ``redraws``, ``mean`` is the step's own full-set metric and
``sd = sd(redraws) * sqrt(n_subset / n_full)``; the verdict rules are the same.
With ``chain1`` set and at least two other chains, ``chain1_pass`` is
``|chain1 - stored| <= max(a_min, k * sd(other chains))``; it does not change the verdict.

Values are compared in the band's ``unit`` and written back in the unit of the step's own metric entry.
`evaluate_band` is pure: it reads only the objects it is given.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from senselab.utils.reproduce.metrics import (
    VERDICT_ORDER,
    MetricsError,
    convert_unit,
    metric_entry,
    metric_with_unit,
)
from senselab.utils.reproduce.spec import BandSpec, Expect

MEAN_RELATIVE = 1e-9
SLACK = 1e-12
CODES = frozenset({"E_BAND_MEAN"})
WARNINGS = frozenset({"W_BAND_AMIN_WIDE"})


def amin_too_wide(band: BandSpec, floor: float) -> bool:
    """Whether ``a_min`` alone exceeds ``max_width_frac * |lift|``, so no band of this row can be informative.

    Args:
        band: The band.
        floor: The floor in the band's unit.

    Returns:
        True when ``a_min > max_width_frac * |stored - floor|``. For a declared floor it is reported at lint
        time, for ``floor_from: entry`` at run time.
    """
    return band.a_min > band.max_width_frac * abs(band.stored - floor)


def worst_verdict(verdicts: list[str]) -> str | None:
    """The worst of some verdicts (``fail`` > ``uninformative`` > ``characterise`` > ``pass``), None for none."""
    return min(verdicts, key=VERDICT_ORDER.index) if verdicts else None


def evaluate_band(
    expect: Expect, own: Mapping[str, Any], upstream: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, float | bool | str], list[str]]:
    """Judge a ``seed_band`` expectation.

    Args:
        expect: A ``seed_band`` expectation.
        own: The step's validated metrics.
        upstream: For each chain, its driver-validated ``metrics.json``.

    Returns:
        ``(fields, codes)``. ``fields`` are the driver-written keys for the metric's entry, in that entry's
        unit: ``target`` (the stored value), ``delta`` (mean - stored), ``tolerance`` (h), ``pass``
        (verdict is ``pass``), ``verdict``, ``floor_used``, ``lift_used``, ``sd`` and, when it applies,
        ``chain1_pass``. ``codes`` holds ``E_BAND_MEAN`` and ``W_BAND_AMIN_WIDE`` when they apply.

    Raises:
        MetricsError: A fixed address code when a value cannot be read.
    """
    band = expect.band
    assert band is not None
    work = band.unit
    raw, unit = metric_with_unit(own, expect.metric)
    mean = convert_unit(raw, unit, work)
    values = {name: _value(upstream[name], expect.metric, work) for name in band.sources()}
    spread = [values[name] for name in band.sources()]
    codes: list[str] = []
    if band.chains is not None:
        if not math.isclose(mean, statistics.fmean(spread), rel_tol=MEAN_RELATIVE, abs_tol=0.0):
            codes.append("E_BAND_MEAN")
        sd = statistics.stdev(spread)
    else:
        assert band.n_subset is not None and band.n_full is not None
        sd = statistics.stdev(spread) * math.sqrt(band.n_subset / band.n_full)
    floor = band.floor if band.floor is not None else _entry_floor(own, expect.metric, work)
    if band.floor_from == "entry" and amin_too_wide(band, floor):
        codes.append("W_BAND_AMIN_WIDE")
    lift = band.stored - floor
    half = max(band.a_min, band.k * sd, band.rho * abs(lift))
    delta = mean - band.stored
    if band.characterise or lift <= band.a_min / band.max_width_frac:
        verdict = "characterise"
    elif abs(delta) > half + SLACK * max(1.0, abs(band.stored)):
        verdict = "fail"
    elif half > band.max_width_frac * lift:
        verdict = "uninformative"
    else:
        verdict = "pass"

    def back(value: float) -> float:
        return convert_unit(value, work, unit)

    fields: dict[str, float | bool | str] = {
        "target": back(band.stored),
        "delta": back(delta),
        "tolerance": back(half),
        "pass": verdict == "pass",
        "verdict": verdict,
        "floor_used": back(floor),
        "lift_used": back(lift),
        "sd": back(sd),
    }
    if band.chain1 is not None and band.chains is not None and len(band.chains) >= 3:
        others = [values[name] for name in band.chains if name != band.chain1]
        allowed = max(band.a_min, band.k * statistics.stdev(others))
        fields["chain1_pass"] = abs(values[band.chain1] - band.stored) <= allowed + SLACK * max(1.0, abs(band.stored))
    return fields, codes


def _value(obj: Mapping[str, Any], address: str, work: str) -> float:
    value, unit = metric_with_unit(obj, address)
    return convert_unit(value, unit, work)


def _entry_floor(obj: Mapping[str, Any], address: str, work: str) -> float:
    entry = metric_entry(obj, address)
    floor = None if entry is None else entry.get("floor")
    if not isinstance(floor, (int, float)) or isinstance(floor, bool):
        raise MetricsError("E_METRICS_ABSENT")
    return convert_unit(float(floor), str(entry.get("unit", "fraction")) if entry is not None else None, work)
