"""`size` over an estimate and the run's lengths: items cut at the upper prediction, batch per length, card sharing."""

import logging
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st
from senselab_core.kernel.declare import BackendSpec, Env, TaskSpec
from senselab_core.kernel.signals import PlainText
from senselab_core.runtime.calibration import (
    Estimate,
    EstimateSource,
    LengthProfile,
    MemoryFit,
    RegimeFactor,
    RequestFit,
    StratumBatch,
)
from senselab_core.runtime.execution.executor import JobRef
from senselab_core.runtime.execution.in_process import InProcess
from senselab_core.runtime.execution.job import Job
from senselab_core.runtime.execution_settings import ResourceClass
from senselab_core.runtime.fake_backend import EchoParams
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo
from senselab_core.runtime.keys import Call, Placement
from senselab_core.runtime.measures import MeasureSource, MeasureUnit
from senselab_core.runtime.probe import memory_batch, peak
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import ExtrapolationLog, Grant, Measure, Sizing, size, size_row
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.calibration_samples import UNIT

PROFILE = ExecutionProfile.load("laptop")
SIZING, CALIBRATION = PROFILE.sizing, PROFILE.calibration
GB = 10**9
CAP = 80 * GB


def _estimate(  # noqa: PLR0913 - the figures a test varies
    *,
    spread: float | None = 0.1,
    u: float | None = 0.05,
    hi: float = 1000.0,
    device: MemoryFit | None = MemoryFit(resident=10 * GB, m0=0.0, m1=1e7, n=6),
    host: MemoryFit | None = None,
    batch: tuple[StratumBatch, ...] = (StratumBatch(lo=2.0, hi=1000.0, batch=1),),
    batched: tuple[RequestFit, ...] = (),
    regimes: tuple[RegimeFactor, ...] = (),
    unit_kind: str = "audio_s",
    load_s: float = 2.0,
    item_s: float = 0.05,
) -> Estimate:
    return Estimate(
        status="measured",
        unit_kind=unit_kind,
        load_s=load_s,
        load_n=3,
        item_s=item_s,
        item_n=3,
        single=RequestFit(batch=1, r=0.5, u=u, n=6, spread=spread, lo=2.0, hi=hi),
        batched=batched,
        batch=batch,
        device=device,
        host=host,
        regimes=regimes,
        source=EstimateSource(probe_ids=("p1",), observations=12),
    )


def _grant(**fields: object) -> Grant:
    given = {"max_workers": 8, "memory_bytes": CAP, "quantum_items": 100, "quantum_s": 300.0, "on_device": True}
    return Grant(**{**given, **fields})


def _work(*lengths: float | None) -> LengthProfile:
    return LengthProfile(units=tuple(lengths))


def _sized(estimate: Estimate | None, grant: Grant, *lengths: float | None) -> Sizing:
    return size(estimate, grant, SIZING, _work(*lengths), calibration=CALIBRATION)


def _seconds(sized: Sizing, cut: tuple[int, int]) -> float:
    return sum(s or 0.0 for s in sized.predicted_s[cut[0] : cut[1]])


@pytest.mark.logic
def test_ten_minute_recordings_keep_items_within_quantum_s() -> None:
    """Items of ten-minute recordings are cut at their own predicted seconds, not at the row's median item."""
    grant, z = _grant(), CALIBRATION.confidence
    sized = _sized(_estimate(load_s=0.0, item_s=0.0), grant, *[600.0] * 40)
    upper = (0.5 + 0.05 * 600.0) * (1 + z * 0.1)
    assert all(_seconds(sized, c) * (1 + z * 0.1) <= grant.quantum_s for c in sized.cuts)
    assert max(b - a for a, b in sized.cuts) == int(grant.quantum_s // upper)
    assert [i for c in sized.cuts for i in range(*c)] == list(range(40))
    by_row = Measure(load_s=0.0, claim_s=0.0, resident_bytes=None, bytes_per_item=None, s_per_item=6.1)
    assert size_row(by_row, grant, SIZING).quantum_items > 5 * max(b - a for a, b in sized.cuts)


@pytest.mark.logic
def test_items_are_cut_at_the_upper_prediction() -> None:
    """The wider the spread, the smaller the item: an item closes where `t·(1 + z·spread)` passes the quantum."""
    grant, z = _grant(), CALIBRATION.confidence
    sizes = {
        s: max(b - a for a, b in _sized(_estimate(spread=s, load_s=0.0, item_s=0.0), grant, *[100.0] * 60).cuts)
        for s in (0.0, 0.5)
    }
    assert sizes[0.0] == int(grant.quantum_s // (0.5 + 0.05 * 100.0))
    assert sizes[0.5] == int(grant.quantum_s // ((0.5 + 0.05 * 100.0) * (1 + z * 0.5)))
    assert sizes[0.5] < sizes[0.0]


@pytest.mark.logic
def test_an_item_closes_once_it_earns_its_load() -> None:
    """With a load, an item closes when its seconds reach `(load_s + a)/load_share`, before the quantum."""
    sized = _sized(_estimate(spread=0.0), _grant(), *[100.0] * 30)
    floor = (2.0 + 0.05) / SIZING.load_share
    assert max(b - a for a, b in sized.cuts) == int(floor // 5.5) + 1


@pytest.mark.logic
@pytest.mark.parametrize("length", [5.0, 20.0, 60.0])
def test_workers_sharing_a_card_fit_it_together(length: float) -> None:
    """The workers of one card, each at its own batch peak, stay within `memory_headroom` of the card together."""
    star = (StratumBatch(lo=2.0, hi=1000.0, batch=8),)
    sized = _sized(_estimate(batch=star), _grant(devices=1), length, length)
    fit = sized.memory
    assert fit is not None and sized.budget_bytes is not None and sized.workers_per_device > 1
    batch = sized.batch_at(length)
    assert sized.workers_per_device * peak(fit, batch, length) <= SIZING.memory_headroom * CAP
    assert sized.budget_bytes == SIZING.memory_headroom * (CAP // sized.workers_per_device)
    assert sized.batch_peak == round(peak(fit, batch, length)) and sized.batch_one_peak == round(peak(fit, 1, length))


@pytest.mark.logic
def test_workers_per_device_bounded_by_the_batch_peak() -> None:
    """A bigger batch peak at the longest job leaves room for fewer workers; a measured regime lowers it further."""
    grant = _grant(devices=2)
    small = _sized(_estimate(), grant, 10.0, 10.0)
    large = _sized(_estimate(device=MemoryFit(resident=10 * GB, m0=0.0, m1=1e9, n=6)), grant, 10.0, 10.0)
    assert (small.workers_per_device, large.workers_per_device) == (int(0.9 * 80 // 10.1), int(0.9 * 80 // 20))
    flat = (RegimeFactor(k=2, rho=2.0, n=5),)
    assert _sized(_estimate(regimes=flat), grant, 10.0, 10.0).workers_per_device == 1
    rising = (RegimeFactor(k=2, rho=1.2, n=5), RegimeFactor(k=3, rho=2.0, n=5))
    assert _sized(_estimate(regimes=rising), grant, 10.0, 10.0).workers_per_device == 2
    assert _sized(_estimate(), _grant(devices=None), 10.0, 10.0).workers_per_device == 1


@pytest.mark.logic
def test_batch_is_one_above_the_measured_length_range() -> None:
    """The batch is the stratum's, found by the length's range; 1 above the range or for an unknown length."""
    strata = (StratumBatch(lo=2.0, hi=10.0, batch=8), StratumBatch(lo=40.0, hi=60.0, batch=2))
    sized = _sized(_estimate(batch=strata, hi=60.0), _grant(), 5.0)
    assert (sized.batch_at(5.0), sized.batch_at(50.0), sized.batch_at(20.0)) == (8, 2, 2)
    assert (sized.batch_at(60.5), sized.batch_at(None), sized.batch_at(5.0, jobs=3)) == (1, 1, 3)
    assert sized.measured.time and not _sized(_estimate(hi=60.0), _grant(), 5.0, 61.0).measured.time


@pytest.mark.logic
def test_a_device_batch_is_sized_only_against_device_memory() -> None:
    """On a device grant an estimate with host memory only gives batch 1; a host grant reads the host fit."""
    star = (StratumBatch(lo=2.0, hi=1000.0, batch=8),)
    host = MemoryFit(resident=GB, m0=0.0, m1=1e6, n=6)
    on_gpu = _sized(_estimate(device=None, host=host, batch=star), _grant(), 10.0)
    on_cpu = _sized(_estimate(device=None, host=host, batch=star), _grant(on_device=False), 10.0)
    assert (on_gpu.batch_at(10.0), on_gpu.measured.memory) == (1, False)
    assert on_cpu.batch_at(10.0) == 8 and on_cpu.measured.memory


@pytest.mark.logic
def test_an_unmeasured_unit_runs_the_safe_values() -> None:
    """Without an estimate: batch 1, items of `quantum_items` jobs, the configured workers, nothing predicted."""
    sized = _sized(None, _grant(quantum_items=3), *[1.0] * 7)
    assert sized.cuts == ((0, 3), (3, 6), (6, 7)) and sized.workers == CALIBRATION.unmeasured_workers
    assert sized.batch_at(1.0) == 1 and sized.wall_s is None and not any(sized.measured.model_dump().values())
    assert _sized(None, _grant()).workers == 0


_TEXT_TASK = TaskSpec(
    name="t", version="1", kind="transform", inputs={"text": "json"}, outputs={"value": "sha256"}, params=EchoParams
)


def _text_jobs(*texts: str) -> list[Job]:
    jobs = []
    for i, text in enumerate(texts):
        held = PlainText(content=text)
        call = Call(_TEXT_TASK, EchoParams(), {"text": held.identity})
        jobs.append(Job(JobRef(plan_hash="0" * 64, node="n", element=(i,)), call, {"text": held}))
    return jobs


class _Root:
    def observations(self, unit: MeasureUnit) -> Sequence[object]:
        return (object(),)


class _NoTable:
    def get(self, unit: MeasureUnit) -> None:
        return None


@pytest.mark.logic
def test_in_process_sizing_is_size_on_the_same_estimate_grant_and_lengths(tmp_path: object) -> None:
    """The in-process executor's sizing of a unit is `size` on the same estimate, grant and lengths."""
    estimate = _estimate(unit_kind="chars", host=MemoryFit(resident=GB, m0=0.0, m1=1e6, n=6), device=None)
    estimate = estimate.model_copy(update={"batch": (StratumBatch(lo=2.0, hi=1000.0, batch=8),)})
    source = MeasureSource([_Root()], _NoTable(), estimate=lambda unit, observed: estimate)  # type: ignore[list-item]
    profile = ExecutionProfile.load("laptop", {"resource_classes": {"cpu": {"mem_bytes": 2**33}}})
    executor = InProcess(
        store=ResultStore(tmp_path / "store"),  # type: ignore[operator]
        profile=profile,
        measures=source,
        units=lambda call, placement, decoder: UNIT,
    )
    jobs = _text_jobs("a" * 10, "b" * 20, "c" * 30)
    backend = _backend()
    placement = Placement(backend=backend, backend_params=backend.params(), device="cpu")
    grant = Grant.seen(profile.resource_classes["cpu"], profile.pool, None)
    direct = size(estimate, grant, SIZING, _work(10.0, 20.0, 30.0), calibration=CALIBRATION)
    assert executor.sizing(jobs, placement) == direct and direct.batch_at(30.0, 3) == 3
    bare = InProcess(store=ResultStore(tmp_path / "bare"), profile=profile)  # type: ignore[operator]
    assert bare.sizing(jobs, placement) == size(None, grant, SIZING, _work(0.0, 0.0, 0.0), calibration=CALIBRATION)


def _backend() -> BackendSpec:
    return BackendSpec(
        task="t",
        name="b",
        version="1",
        stack="s",
        env=Env.host(),
        devices=("cpu",),
        batch_invariant=True,
        worker="senselab_worker.fake",
        load_stub={},
    )


@pytest.mark.logic
def test_an_unmeasured_slope_or_spread_never_enters_as_zero() -> None:
    """With `u` unmeasured, 600 s jobs outside a short range are untimed and cut by count, as an unmeasured unit is."""
    narrow = _estimate(u=None, hi=7.0, batch=(StratumBatch(lo=5.0, hi=7.0, batch=1),))
    sized = _sized(narrow, _grant(), *[600.0] * 250)
    assert sized.cuts == _sized(None, _grant(), *[600.0] * 250).cuts and sized.cuts[0] == (0, 100)
    assert sized.workers == CALIBRATION.unmeasured_workers
    assert sized.predicted_s == (None,) * 250 and not sized.measured.time and sized.wall_s is None
    flat = _sized(_estimate(spread=None), _grant(), *[100.0] * 250)
    assert flat.cuts == sized.cuts and flat.measured.time


_UNIT = st.floats(min_value=0.0, max_value=2000.0, allow_nan=False)


@st.composite
def _estimates(draw: st.DrawFn) -> Estimate:
    lo = draw(st.floats(min_value=0.0, max_value=50.0))
    hi = lo + draw(st.floats(min_value=0.0, max_value=1000.0))
    fit = MemoryFit(
        resident=draw(st.integers(0, 40 * GB)),
        m0=draw(st.floats(0.0, 1e8)),
        m1=draw(st.floats(0.0, 1e9)),
        n=6,
    )
    return Estimate(
        status="measured",
        unit_kind="audio_s",
        load_s=draw(st.none() | st.floats(0.0, 60.0)),
        load_n=3,
        item_s=draw(st.none() | st.floats(0.0, 1.0)),
        item_n=3,
        single=RequestFit(
            batch=1,
            r=draw(st.floats(0.0, 5.0)),
            u=draw(st.none() | st.floats(0.0, 0.5)),
            n=6,
            spread=draw(st.none() | st.floats(0.0, 1.0)),
            lo=lo,
            hi=hi,
        ),
        batch=(StratumBatch(lo=lo, hi=hi, batch=draw(st.integers(1, 64))),),
        device=draw(st.none() | st.just(fit)),
        host=draw(st.none() | st.just(fit)),
        source=EstimateSource(probe_ids=("p",), observations=6),
    )


@pytest.mark.logic
@given(
    estimate=_estimates(),
    lengths=st.lists(st.none() | _UNIT, max_size=40),
    devices=st.none() | st.integers(0, 4),
    on_device=st.booleans(),
    memory=st.none() | st.integers(1, 80 * GB),
)
def test_size_keeps_its_invariants(
    estimate: Estimate, lengths: list[float | None], devices: int | None, on_device: bool, memory: int | None
) -> None:
    """Cuts partition the jobs; items hold the upper bound; a card's workers and batches fit; workers stay bounded."""
    grant = _grant(devices=devices, on_device=on_device, memory_bytes=memory, quantum_items=7)
    sized = _sized(estimate, grant, *lengths)
    assert [i for a, b in sized.cuts for i in range(a, b)] == list(range(len(lengths)))
    assert all(0 < b - a <= grant.quantum_items for a, b in sized.cuts)
    single = estimate.single
    assert single is not None
    if single.u is not None and single.spread is not None:
        z = CALIBRATION.confidence
        for a, b in sized.cuts:
            if b - a > 1:
                upper = sum(s or 0.0 for s in sized.predicted_s[a:b]) * (1 + z * single.spread)
                assert not sized.measured.time or upper + (estimate.item_s or 0.0) <= grant.quantum_s * (1 + 1e-9)
    if sized.budget_bytes is not None and sized.memory is not None:
        assert sized.workers_per_device * sized.budget_bytes <= SIZING.memory_headroom * (memory or 0) * (1 + 1e-9)
        for length in lengths:
            if length is not None:
                room = memory_batch(sized.memory, sized.budget_bytes, length, grant.quantum_items)
                assert sized.batch_at(length) <= max(1, room)
    bound = grant.max_workers if devices is None else devices * sized.workers_per_device
    assert sized.workers <= min(len(sized.cuts), grant.max_workers, bound) and (sized.workers == 0) == (
        not lengths or bound == 0
    )


@pytest.mark.logic
def test_a_device_grant_without_a_class_cap_takes_the_seen_devices_memory() -> None:
    """The device a worker sees stands in for a cap the class lacks; the cap, where set, wins."""
    uncapped = ResourceClass(device="cuda", max_workers=2)
    seen = HardwareInfo(
        cpu_count=1,
        memory_bytes=2**33,
        gpus=(
            GpuInfo(name="H100", memory_bytes=2**36, driver="1"),
            GpuInfo(name="H100", memory_bytes=2**35, driver="1"),
        ),
        apple_silicon=False,
        platform="linux-x86_64",
    )
    assert Grant.seen(uncapped, PROFILE.pool, seen).memory_bytes == 2**35
    assert Grant.seen(uncapped, PROFILE.pool, None).memory_bytes is None
    assert Grant.seen(uncapped.model_copy(update={"memory_cap_bytes": 2**30}), PROFILE.pool, seen).memory_bytes == 2**30


EDGES = (10.0, 60.0)


@pytest.mark.logic
def test_a_bucketed_formation_cuts_on_bucket_edges() -> None:
    """With a bucket grid no item spans two buckets, measured or not; without one the same jobs are a single item."""
    lengths = (5.0, 6.0, 12.0, 15.0, 70.0)
    for estimate in (_estimate(), None):
        flat = size(estimate, _grant(), SIZING, _work(*lengths), calibration=CALIBRATION)
        cut = size(estimate, _grant(), SIZING, _work(*lengths), calibration=CALIBRATION, bucket_edges=EDGES)
        assert flat.cuts == ((0, 5),)
        assert cut.cuts == ((0, 2), (2, 4), (4, 5))
    unknown = size(
        _estimate(), _grant(), SIZING, _work(5.0, None, None, 5.0), calibration=CALIBRATION, bucket_edges=EDGES
    )
    assert {stop for _, stop in unknown.cuts} >= {1, 3, 4}


@pytest.mark.logic
def test_extrapolation_is_logged_once_per_unit(caplog: pytest.LogCaptureFixture) -> None:
    """A length outside the measured range logs one line naming the unit's digest label and the range, once per run."""
    inside = _sized(_estimate(), _grant(), 10.0, 20.0)
    beyond = _sized(_estimate(), _grant(), 10.0, 5000.0)
    assert (inside.extrapolated, beyond.extrapolated) == (False, True)
    other = UNIT.model_copy(update={"backend": "elsewhere"})
    said = ExtrapolationLog()
    with caplog.at_level(logging.INFO, logger="senselab_core.runtime.sizing"):
        results = [
            said.note(UNIT.label(), inside),
            said.note(UNIT.label(), beyond),
            said.note(UNIT.label(), beyond),
            said.note(other.label(), beyond),
        ]
        assert ExtrapolationLog().note(UNIT.label(), beyond)
    assert results == [False, True, False, True]
    first = [r.getMessage() for r in caplog.records][0]
    assert UNIT.label() in first and "[2, 1000]" in first
    assert UNIT.backend not in first and UNIT.digest() not in first
