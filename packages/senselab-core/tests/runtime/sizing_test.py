"""`size` derives batch, quantum and workers from a measure; `calibrated` reads a measure from a round's samples."""

import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.execution_settings import PoolSettings, ResourceClass, SizingSettings
from senselab_core.runtime.resources import (
    ClassMeasure,
    Measured,
    ResourceMeasure,
    ResourceRow,
    ResourceTable,
    Unit,
    write_row,
)
from senselab_core.runtime.sizing import Grant, Measure, Measures, Sample, calibrated, measure_of, size

SETTINGS = SizingSettings(
    load_share=0.05,
    memory_headroom=0.9,
    target_wall_s=60,
    calibration_items=4,
    calibration_workers=1,
    device_query_s=30,
)
UNIT = Unit("t", "b", "org/m", "a" * 40, "cpu")
MEASURED = Measured(sha=None, run="r", jobs=("7",))
FOUND = ResourceMeasure(
    load_s=1, claim_s=0.01, resident_bytes=1, peak_memory_bytes=2, s_per_input_s=0.5, input_s_per_item=None
)

seconds = st.floats(min_value=0, max_value=1e4, allow_nan=False, allow_subnormal=False)
measures = st.builds(
    Measure,
    load_s=seconds,
    claim_s=seconds,
    resident_bytes=st.none() | st.integers(0, 2**36),
    bytes_per_item=st.integers(0, 2**32),
    s_per_item=seconds,
    on_device=st.booleans(),
)
grants = st.builds(
    Grant,
    max_workers=st.integers(1, 64),
    memory_bytes=st.none() | st.integers(1, 2**37),
    quantum_items=st.integers(1, 1000),
    quantum_s=st.floats(min_value=1e-3, max_value=1e5, allow_nan=False),
    devices=st.none() | st.integers(0, 8),
)
settings = st.builds(
    SizingSettings,
    load_share=st.floats(min_value=1e-3, max_value=0.999),
    memory_headroom=st.floats(min_value=1e-3, max_value=1.0),
    target_wall_s=st.floats(min_value=1e-3, max_value=1e5, allow_nan=False),
    calibration_items=st.integers(2, 8),
    calibration_workers=st.integers(1, 4),
    device_query_s=st.just(30.0),
)


@pytest.mark.logic
@given(measure=measures, grant=grants, config=settings, work=st.integers(0, 10**6))
def test_sizing_keeps_every_bound(measure: Measure, grant: Grant, config: SizingSettings, work: int) -> None:
    """Quantum within its ceilings and the load share, batch within memory, workers within the work and the grant."""
    sized = size(measure, grant, config, work=work)
    fixed = measure.load_s + measure.claim_s
    per, q, eps = measure.s_per_item, sized.quantum_items, 1e-9
    share = fixed / config.load_share
    assert 1 <= q <= grant.quantum_items and sized.quantum_s <= grant.quantum_s
    assert q == 1 or per == 0 or q * per + fixed <= grant.quantum_s * (1 + eps)
    assert (
        q in (1, grant.quantum_items)
        or q * per >= share * (1 - eps)
        or (q + 1) * per + fixed > grant.quantum_s * (1 - eps)
    )
    assert q == 1 or per == 0 or fixed == 0 or (q - 1) * per < share * (1 + eps)
    assert 1 <= sized.batch_size <= q
    if grant.memory_bytes is None or measure.resident_bytes is None:
        assert sized.batch_size == 1
    elif sized.batch_size > 1:
        used = measure.resident_bytes + sized.batch_size * (measure.bytes_per_item or 0)
        assert used <= config.memory_headroom * grant.memory_bytes * (1 + eps)
    bound = grant.worker_bound(measure)
    assert sized.workers == 0 if work == 0 or bound == 0 else 1 <= sized.workers <= min(bound, work)
    assert sized.workers <= math.ceil(work / q)
    if grant.devices is not None and not measure.on_device:
        assert bound == min(grant.max_workers, grant.devices)


@pytest.mark.logic
def test_the_quantum_holds_the_load_and_claim_to_their_share_and_workers_meet_the_target() -> None:
    """A 1.9 s load and 0.1 s claim at 0.1 s per item take a 400-job quantum; 4000 jobs in 60 s want 7 workers."""
    measure = Measure(load_s=1.9, claim_s=0.1, resident_bytes=1000, bytes_per_item=100, s_per_item=0.1)
    grant = Grant(max_workers=32, memory_bytes=10_000, quantum_items=1000, quantum_s=3600)
    sized = size(measure, grant, SETTINGS, work=4000)
    assert (sized.quantum_items, sized.workers) == (400, 7) and sized.quantum_s == pytest.approx(42.0)
    assert sized.memory_batch == (0.9 * 10_000 - 1000) // 100 and sized.batch_size == 1
    proven = measure.model_copy(update={"batched_s_per_item": 0.1})
    assert size(proven, grant, SETTINGS).batch_size == sized.memory_batch
    assert size(proven, grant, SETTINGS, batchable=False).batch_size == 1
    assert size(measure, grant.model_copy(update={"quantum_s": 12.0}), SETTINGS).quantum_items == 100
    free = measure.model_copy(update={"load_s": 0.0, "claim_s": 0.0})
    assert size(free, grant, SETTINGS).quantum_items == grant.quantum_items


@pytest.mark.logic
def test_a_grant_takes_device_memory_on_a_device_and_host_memory_otherwise() -> None:
    """On a device the class's device cap; else its memory, the host's last; the class's time bounds the quantum."""
    pool = PoolSettings(quantum_items=100, quantum_s=300, poll_s=30, first_poll_s=1, segment_items=512)
    capped = ResourceClass(device="cuda", max_workers=4, mem_bytes=2**36, memory_cap_bytes=2**33, time_s=60)
    assert Grant.of(capped, pool, on_device=True).model_dump() == {
        "max_workers": 4,
        "memory_bytes": 2**33,
        "quantum_items": 100,
        "quantum_s": 60,
        "devices": None,
    }
    assert Grant.of(capped, pool, devices=2).devices == 2
    assert Grant.of(capped, pool).memory_bytes == 2**36
    bare = ResourceClass(device="cpu", max_workers=1)
    assert Grant.of(bare, pool, devices=2).devices is None
    assert Grant.of(bare, pool).memory_bytes is None and Grant.of(bare, pool, 2**30).memory_bytes == 2**30
    device = FOUND.model_copy(update={"device_resident_bytes": 10, "device_peak_bytes": 30})
    assert measure_of(device, on_device=True).model_dump() == {
        "load_s": 1.0,
        "claim_s": 0.01,
        "resident_bytes": 10,
        "bytes_per_item": 20,
        "s_per_item": 0.5,
        "on_device": True,
        "batched_s_per_item": None,
    }
    assert measure_of(FOUND, on_device=True).on_device is False and measure_of(device).bytes_per_item == 1


@pytest.mark.logic
def test_workers_on_shared_devices_are_bounded_by_device_memory() -> None:
    """A 75 GB device peak under an 80 GB cap fits one worker per device; 30 GB fits two; unmeasured, one."""
    gpu = Grant(max_workers=8, memory_bytes=80 * 10**9, quantum_items=1, quantum_s=3600, devices=1)
    big = Measure(
        load_s=0, claim_s=0, resident_bytes=70 * 10**9, bytes_per_item=5 * 10**9, s_per_item=60, on_device=True
    )
    small = big.model_copy(update={"resident_bytes": 25 * 10**9})
    host = big.model_copy(update={"on_device": False})
    assert size(big, gpu, SETTINGS, work=100).workers == 1
    assert size(small, gpu, SETTINGS, work=100).workers == 2
    assert size(host, gpu, SETTINGS, work=100).workers == 1
    assert size(big, gpu.model_copy(update={"devices": 3}), SETTINGS, work=100).workers == 3
    assert size(big, gpu.model_copy(update={"devices": 0}), SETTINGS, work=100).workers == 0
    assert size(big, gpu.model_copy(update={"devices": None}), SETTINGS, work=100).workers == 8
    assert gpu.worker_bound(None) == 1


def _sample(wall: float, *, cold: bool = False, resident: int | None = 10_000, peak: int = 10_100) -> Sample:
    return Sample(
        wall_s=wall,
        claim_s=0.02,
        jobs=1,
        batch_size=1,
        input_s=4.0,
        baseline_bytes=500,
        resident_bytes=resident,
        peak_memory_bytes=peak,
        cold=cold,
        job="7",
        ended_s=0,
    )


@pytest.mark.logic
def test_calibrated_separates_the_load_and_the_model_from_the_items() -> None:
    """A cold 10 s item and warm 2, 2, 3 s ones give 2 s per item and an 8 s load; per-item bytes exclude the model.

    Warm items leave the model resident (10 000 B) and peak 100 B above it; the cold item peaks higher, while the
    model loads, so the per-item bytes are the peak over the warm residents, not over the pre-load baseline.
    """
    found = calibrated([_sample(10, cold=True, peak=10_050), _sample(2), _sample(2), _sample(3)])
    assert found == ResourceMeasure(
        load_s=8.0,
        claim_s=0.02,
        resident_bytes=10_000,
        peak_memory_bytes=10_100,
        s_per_input_s=0.5,
        input_s_per_item=4.0,
    )
    assert measure_of(found).bytes_per_item == 100
    assert calibrated([]) is None
    alone = calibrated([_sample(10, cold=True)])
    assert alone is not None and (alone.load_s, alone.s_per_input_s, alone.resident_bytes) == (0.0, 2.5, 10_000)
    unread = calibrated([_sample(10, cold=True, resident=None), _sample(2, resident=None)])
    assert unread is not None and (unread.resident_bytes, unread.peak_memory_bytes) == (None, None)
    unmeasured = measure_of(unread)
    assert (unmeasured.resident_bytes, unmeasured.bytes_per_item) == (None, None)
    grant = Grant(max_workers=1, memory_bytes=2**36, quantum_items=100, quantum_s=300)
    assert size(unmeasured, grant, SETTINGS).batch_size == 1
    read = [
        s.model_copy(update={"device_baseline_bytes": 0, "device_bytes": used})
        for s, used in ((_sample(10, cold=True), 1000), (_sample(2), 1010))
    ]
    served = calibrated([s.model_copy(update={"served": True}) for s in read])
    assert served is not None and (served.device_resident_bytes, served.device_peak_bytes) == (1000, 1010)
    model = calibrated(read)
    assert model is not None and (model.device_resident_bytes, model.device_peak_bytes) == (None, None)


@pytest.mark.logic
def test_each_class_keeps_its_own_derivation(tmp_path: Path) -> None:
    """Two classes of one model written by different jobs both stay, each with its own commit and jobs."""
    write_row(tmp_path, UNIT, FOUND, Measured(sha="a" * 40, jobs=("101",)))
    gpu = Unit("t", "b", "org/m", "a" * 40, "gpu:h100")
    write_row(tmp_path, gpu, FOUND.model_copy(update={"load_s": 9.0}), Measured(sha="b" * 40, jobs=("202",)))
    write_row(tmp_path, UNIT, FOUND.model_copy(update={"load_s": 3.0}), Measured(sha="c" * 40, jobs=("303",)))
    row = ResourceTable.model_validate(load_yaml(tmp_path / "t" / "b.yaml", "table")).rows[0]
    assert {c: (m.load_s, m.measured.sha, m.measured.jobs) for c, m in row.classes.items()} == {
        "cpu": (3.0, "c" * 40, ("303",)),
        "gpu:h100": (9.0, "b" * 40, ("202",)),
    }


@pytest.mark.logic
def test_measures_read_written_rows_first_then_packaged_ones(tmp_path: Path) -> None:
    """A row written under the root wins over the packaged row of the unit; with none, the packaged row is read."""
    row = ResourceRow(
        id="org/m",
        revision="a" * 40,
        download_bytes=10,
        classes={"cpu": ClassMeasure(**FOUND.model_copy(update={"load_s": 3.0}).model_dump(), measured=MEASURED)},
    )
    measures = Measures(tmp_path, packaged=(ResourceTable(task="t", backend="b", rows=(row,)),))
    found = measures.get(UNIT)
    assert found is not None and found.load_s == 3.0
    measures.put(UNIT, FOUND, MEASURED)
    found = measures.get(UNIT)
    assert found is not None and found.load_s == 1.0 and Measures(tmp_path, packaged=()).get(UNIT) is not None
    assert Measures(tmp_path, packaged=()).get(Unit("t", "b", "org/m", "a" * 40, "gpu")) is None


@pytest.mark.logic
def test_reported_request_peaks_give_the_device_bytes_per_item() -> None:
    """A backend's request peaks over what stays allocated give per-item device bytes, the model left out of them."""
    reported = [
        _sample(10, cold=True).model_copy(
            update={"device_bytes": 9000, "device_item_bytes": 40, "device_allocated_bytes": 5000}
        ),
        _sample(2).model_copy(update={"device_bytes": 9000, "device_item_bytes": 50, "device_allocated_bytes": 5000}),
    ]
    found = calibrated(reported)
    assert found is not None and (found.device_resident_bytes, found.device_peak_bytes) == (5000, 5050)
    assert measure_of(found, on_device=True).bytes_per_item == 50


@pytest.mark.logic
@pytest.mark.parametrize(("batched_wall", "batch"), [(1.0, 4), (3.0, 1)], ids=["faster", "slower"])
def test_items_measured_slower_in_a_batch_are_sized_one_per_request(batched_wall: float, batch: int) -> None:
    """Items run 2 s alone; in fours at 1 s each the memory-sized batch stands, at 3 s each it falls to one."""
    alone = [_sample(10, cold=True), _sample(2), _sample(2)]
    batched = _sample(4 * batched_wall).model_copy(update={"jobs": 4, "batch_size": 4})
    found = calibrated([*alone, batched])
    assert found is not None and found.batched is not None and found.batched.batch_size == 4
    measure = measure_of(found)
    assert (measure.s_per_item, measure.batched_s_per_item) == (2.0, batched_wall)
    grant = Grant(max_workers=1, memory_bytes=10_000_000, quantum_items=4, quantum_s=3600)
    assert size(measure, grant, SETTINGS).batch_size == batch
