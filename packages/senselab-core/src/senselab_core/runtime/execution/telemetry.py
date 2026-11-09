"""What the `Pool` reads of production requests: drift from the estimate that sized them, and the epoch's actuals.

`Drift` keeps the last `drift_window` requests of each (unit, co-tenancy) as ratios `wall_s / predicted` and says when
their median lies outside the band the estimate's spread allows. `peak_beyond` and `disagree` are the two other
tests of production against an estimate. `Actuals` collects what a class's work took, to close its `placement` event.
"""

import math
import statistics
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from senselab_core.runtime.calibration import Estimate, HardwareClass, Observation
from senselab_core.runtime.clock import Clock
from senselab_core.runtime.execution.events import ClassActuals, EventLog, PlacementRecord, placement_record
from senselab_core.runtime.placement import HOUR_S
from senselab_core.runtime.probe import MEDIAN_SE
from senselab_core.runtime.site_placement import ClassNeeds, ClassPlacement, HookRef, PlacementRequest


def predicted(estimate: Estimate, request: Observation) -> tuple[float, float | None] | None:
    """Return the seconds `estimate` predicts for `request` at its batch, and the fit's spread; None without a fit.

    The prediction is in the probe regime: the single fit at batch 1, else the batched fit of that batch.
    """
    if request.batch == 1:
        fit, units = estimate.single, request.units
    else:
        fit, units = next((f for f in estimate.batched if f.batch == request.batch), None), request.padded_units
    if fit is None:
        return None
    return fit.r + (fit.u or 0.0) * units, fit.spread


def is_evidence(request: Observation, hardware: HardwareClass, warmup_requests: int) -> bool:
    """Return whether `request` is drift evidence: a warm, `ok` production request on `hardware`, not served."""
    return (
        request.probe is None
        and request.outcome == "ok"
        and request.load_s is None
        and request.since_load is not None
        and request.since_load >= warmup_requests
        and request.hardware == hardware
        and request.wall_s > 0
    )


def band(confidence: float, spread: float, window: int) -> float:
    """Return the half-width of the band a median of `window` ratios may lie in around 1."""
    return confidence * MEDIAN_SE * spread / math.sqrt(window)


@dataclass(frozen=True)
class Reading:
    """One production request against its prediction: the ratio, the spread at its batch, and the batch.

    `rho` is `wall_s` over the probe-regime prediction; `expected` is the factor of its co-tenancy in force (1 where
    none is), the ratio the window is judged against.
    """

    rho: float
    spread: float
    batch: int
    expected: float = 1.0


@dataclass(frozen=True)
class Drifted:
    """A full window whose median ratio lies outside its band.

    Attributes:
        k: The device co-tenancy of the window's requests.
        rho: The window's median ratio.
        n: The window's size.
        batch: The window's median batch.
        rhos: Every ratio of the window.
        low: The band's lower bound.
        high: Its upper bound.
    """

    k: int
    rho: float
    n: int
    batch: int
    rhos: tuple[float, ...]
    low: float
    high: float


class Drift:
    """The last `window` readings of each (unit, co-tenancy), and the test of their median against its band."""

    def __init__(self, window: int, confidence: float) -> None:
        """Judge windows of `window` readings by the two-sided z `confidence`."""
        self.window, self.confidence = window, confidence
        self._seen: dict[tuple[str, int], deque[Reading]] = {}

    def add(self, unit: str, k: int, reading: Reading) -> Drifted | None:
        """Take `reading` of unit `unit` under co-tenancy `k`; return the window if it is full and has drifted.

        A window that drifted starts over, so the next verdict is of `window` new requests.
        """
        seen = self._seen.setdefault((unit, k), deque(maxlen=self.window))
        seen.append(reading)
        if len(seen) < self.window:
            return None
        rho = statistics.median(r.rho for r in seen)
        expected = statistics.median(r.expected for r in seen)
        half = band(self.confidence, statistics.median(r.spread for r in seen), self.window)
        if abs(statistics.median(r.rho / r.expected for r in seen) - 1.0) <= half:
            return None
        found = Drifted(
            k,
            rho,
            len(seen),
            round(statistics.median(r.batch for r in seen)),
            tuple(r.rho for r in seen),
            expected * (1.0 - half),
            expected * (1.0 + half),
        )
        seen.clear()
        return found


def peak_beyond(estimate: Estimate, request: Observation, confidence: float, window: int) -> float | None:
    """Return the `m1` a device peak beyond its band implies, None when the request is within it.

    The prediction is `m0 + m1·padded_units` of the device fit; the band is the drift test's, at the spread of the
    request's batch. None as well where the estimate has no device fit, the request no device reading, or the batch no
    measured spread.
    """
    fit, found = estimate.device, predicted(estimate, request)
    if fit is None or found is None or found[1] is None or request.padded_units <= 0:
        return None
    if request.device_peak is None or request.device_after is None:
        return None
    actual = request.device_peak - request.device_after
    if actual <= (fit.m0 + fit.m1 * request.padded_units) * (1.0 + band(confidence, found[1], window)):
        return None
    return (actual - fit.m0) / request.padded_units


def disagree(suspect: Estimate, probed: Estimate, confidence: float) -> bool:
    """Return whether two single fits differ beyond their bands at the middle of `probed`'s measured range.

    The bands are the drift test's, each of its fit's spread and requests. A fit with no measured spread cannot
    disagree.
    """
    a, b = suspect.single, probed.single
    if a is None or b is None or a.spread is None or b.spread is None:
        return False
    x = (b.lo + b.hi) / 2
    ta, tb = a.r + (a.u or 0.0) * x, b.r + (b.u or 0.0) * x
    if ta <= 0 or tb <= 0:
        return False
    error = MEDIAN_SE * math.hypot(ta * a.spread / math.sqrt(a.n), tb * b.spread / math.sqrt(b.n))
    return abs(tb - ta) > confidence * error


class Actuals:
    """What each class's work took this epoch: from its first put to its last outcome, and its largest device peak."""

    def __init__(self, clock: Clock) -> None:
        """Time every class on `clock`."""
        self._clock = clock
        self._first: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._peak: dict[str, int] = {}

    def began(self, resource_class: str) -> None:
        """Note that work of `resource_class` was put, the first time this epoch."""
        self._first.setdefault(resource_class, self._clock.now())

    def landed(self, resource_class: str, observations: Sequence[Observation]) -> None:
        """Note an outcome of `resource_class` and the device peaks of its `observations`."""
        self._last[resource_class] = self._clock.now()
        peaks = [o.device_peak for o in observations if o.device_peak is not None]
        if peaks:
            self._peak[resource_class] = max(self._peak.get(resource_class, 0), *peaks)

    def closing(self, resource_class: str, worker_s: float | None) -> ClassActuals:
        """Return what the class's work took; `worker_s` is its workers' lifetimes, given for a `cuda` class."""
        first, last = self._first.get(resource_class), self._last.get(resource_class)
        return ClassActuals(
            wall_s=None if first is None or last is None else max(0.0, last - first),
            gpu_hours=None if worker_s is None else worker_s / HOUR_S,
            device_peak_bytes=self._peak.get(resource_class),
        )


class Placements:
    """One epoch's placement events for one log: each class's record as placed, and again, closed, at the epoch's end.

    `request` builds the run's request, from which a class placed with no hook takes its needs; `cuda` says whether
    a class is a GPU class, whose GPU hours are its workers' lifetimes.
    """

    def __init__(
        self,
        log: EventLog,
        request: Callable[[], PlacementRequest],
        actuals: Actuals,
        cuda: Callable[[str], bool],
    ) -> None:
        """Log to `log`, closing each class with what `actuals` read of it."""
        self._log, self._request, self._actuals, self._cuda = log, request, actuals, cuda
        self._placed: dict[str, PlacementRecord] = {}

    def record(  # noqa: PLR0913 - what a `Placer` hands its recorder
        self, _: int, name: str, needs: ClassNeeds | None, hook: HookRef | None, placed: ClassPlacement
    ) -> None:
        """Append class `name`'s placement to the log and keep it with the request it was placed against."""
        record = placement_record(name, needs, hook, placed)
        self._log.record("placement", None, placement=record)
        asked = needs if needs is not None else self._request().held(name)
        self._placed[name] = record.model_copy(update={"request": asked})

    def close(self, worker_seconds: Callable[[str], float] | None = None) -> None:
        """Append each placed class's record again with its actuals; `worker_seconds` gives a class's lifetimes."""
        for name, record in self._placed.items():
            worker_s = worker_seconds(name) if worker_seconds is not None and self._cuda(name) else None
            closing = self._actuals.closing(name, worker_s)
            self._log.record("placement", None, placement=record.model_copy(update={"closing": closing}))
        self._placed.clear()
