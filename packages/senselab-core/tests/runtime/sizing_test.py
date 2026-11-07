"""`size` derives batch, quantum and workers from a measure; `calibrated` reads a measure from a round's samples."""

import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from senselab_core.runtime.execution_settings import PoolSettings, ResourceClass, SizingSettings
from senselab_core.runtime.resources import Measured, ResourceMeasure, ResourceRow, ResourceTable, Unit
from senselab_core.runtime.sizing import Grant, Measure, Measures, Sample, calibrated, measure_of, size

SETTINGS = SizingSettings(load_share=0.05, memory_headroom=0.9, target_wall_s=60, calibration_items=4)
UNIT = Unit("t", "b", "org/m", "a" * 40, "cpu")
MEASURED = Measured(sha=None, run="r", jobs=("7",))

seconds = st.floats(min_value=0, max_value=1e4, allow_nan=False, allow_subnormal=False)
measures = st.builds(
    Measure,
    load_s=seconds,
    resident_bytes=st.integers(0, 2**36),
    bytes_per_item=st.integers(0, 2**32),
    s_per_item=seconds,
)
grants = st.builds(
    Grant,
    max_workers=st.integers(1, 64),
    memory_bytes=st.none() | st.integers(1, 2**37),
    quantum_items=st.integers(1, 1000),
    quantum_s=st.floats(min_value=1e-3, max_value=1e5, allow_nan=False),
)
settings = st.builds(
    SizingSettings,
    load_share=st.floats(min_value=1e-3, max_value=0.999),
    memory_headroom=st.floats(min_value=1e-3, max_value=1.0),
    target_wall_s=st.floats(min_value=1e-3, max_value=1e5, allow_nan=False),
    calibration_items=st.integers(2, 8),
)


@pytest.mark.logic
@given(measure=measures, grant=grants, config=settings, work=st.integers(0, 10**6))
def test_sizing_keeps_every_bound(measure: Measure, grant: Grant, config: SizingSettings, work: int) -> None:
    """Quantum within its ceilings and the load share, batch within memory, workers within the work and the grant."""
    sized = size(measure, grant, config, work=work)
    per, load, q, eps = measure.s_per_item, measure.load_s, sized.quantum_items, 1e-9
    share = load / config.load_share
    assert 1 <= q <= grant.quantum_items and sized.quantum_s <= grant.quantum_s
    assert q == 1 or per == 0 or q * per + load <= grant.quantum_s * (1 + eps)
    assert (
        q in (1, grant.quantum_items)
        or q * per >= share * (1 - eps)
        or (q + 1) * per + load > grant.quantum_s * (1 - eps)
    )
    assert q == 1 or per == 0 or (q - 1) * per < share * (1 + eps)
    assert 1 <= sized.batch_size <= q
    if grant.memory_bytes is None:
        assert sized.batch_size == 1
    elif sized.batch_size > 1:
        used = measure.resident_bytes + sized.batch_size * measure.bytes_per_item
        assert used <= config.memory_headroom * grant.memory_bytes * (1 + eps)
    assert sized.workers == 0 if work == 0 else 1 <= sized.workers <= min(grant.max_workers, work)
    assert sized.workers <= math.ceil(work / q)


@pytest.mark.logic
def test_the_quantum_holds_the_load_to_its_share_and_workers_meet_the_target() -> None:
    """A 2 s load at 0.1 s per item takes a 400-job quantum; 4000 jobs in 60 s want 7 workers, at most 10 quanta."""
    measure = Measure(load_s=2.0, resident_bytes=1000, bytes_per_item=100, s_per_item=0.1)
    grant = Grant(max_workers=32, memory_bytes=10_000, quantum_items=1000, quantum_s=3600)
    sized = size(measure, grant, SETTINGS, work=4000)
    assert (sized.quantum_items, sized.workers) == (400, 7) and sized.quantum_s == pytest.approx(42.0)
    assert sized.batch_size == (0.9 * 10_000 - 1000) // 100
    assert size(measure, grant, SETTINGS, batchable=False).batch_size == 1
    assert size(measure, grant.model_copy(update={"quantum_s": 12.0}), SETTINGS).quantum_items == 100


@pytest.mark.logic
def test_a_grant_takes_its_classs_memory_and_time_as_ceilings() -> None:
    """The memory cap wins over the memory, the host's is the last resort; the class's time bounds `quantum_s`."""
    pool = PoolSettings(quantum_items=100, quantum_s=300, poll_s=30, first_poll_s=1, segment_items=512)
    capped = ResourceClass(device="cuda", max_workers=4, mem_bytes=2**36, memory_cap_bytes=2**33, time_s=60)
    assert Grant.of(capped, pool).model_dump() == {
        "max_workers": 4,
        "memory_bytes": 2**33,
        "quantum_items": 100,
        "quantum_s": 60,
    }
    bare = ResourceClass(device="cpu", max_workers=1)
    assert Grant.of(bare, pool).memory_bytes is None and Grant.of(bare, pool, 2**30).memory_bytes == 2**30


@pytest.mark.logic
def test_calibrated_separates_the_load_from_the_items() -> None:
    """A cold item of 10 s and warm ones of 2, 2 and 3 s give 2 s per item and an 8 s load; per input second at 4 s."""

    def sample(wall: float, cold: bool = False, peak: int = 2000) -> Sample:
        return Sample(wall_s=wall, jobs=1, input_s=4.0, baseline_bytes=500, peak_memory_bytes=peak, cold=cold, job="7")

    found = calibrated([sample(10, cold=True, peak=3000), sample(2), sample(2), sample(3)])
    assert found == ResourceMeasure(
        load_s=8.0, resident_bytes=500, peak_memory_bytes=3000, s_per_input_s=0.5, input_s_per_item=4.0
    )
    assert measure_of(found) == Measure(load_s=8.0, resident_bytes=500, bytes_per_item=2500, s_per_item=2.0)
    assert calibrated([]) is None
    alone = calibrated([sample(10, cold=True)])
    assert alone is not None and (alone.load_s, alone.s_per_input_s) == (0.0, 2.5)
    unrecorded = calibrated([sample(10, cold=True), sample(2).model_copy(update={"input_s": None})])
    assert unrecorded is not None and (unrecorded.input_s_per_item, unrecorded.s_per_input_s) == (None, 2.0)


@pytest.mark.logic
def test_measures_read_packaged_tables_first_and_write_calibration_rows(tmp_path: Path) -> None:
    """A put row is read back and keeps the classes measured alike; a packaged row of the unit wins over it."""
    found = ResourceMeasure(load_s=1, resident_bytes=1, peak_memory_bytes=2, s_per_input_s=0.5, input_s_per_item=None)
    measures = Measures(tmp_path, packaged=())
    assert measures.get(UNIT) is None
    measures.put(UNIT, found, MEASURED)
    measures.put(Unit("t", "b", "org/m", "a" * 40, "gpu"), found.model_copy(update={"load_s": 9.0}), MEASURED)
    assert measures.get(UNIT) == found and Measures(tmp_path, packaged=()).get(UNIT) == found
    written = (tmp_path / "t" / "b.yaml").read_text()
    assert "cpu:" in written and "gpu:" in written and "run: r" in written
    measures.put(UNIT, found, Measured(sha=None, run="s", jobs=("8",)))
    assert "gpu:" not in (tmp_path / "t" / "b.yaml").read_text()
    row = ResourceRow(
        id="org/m",
        revision="a" * 40,
        download_bytes=10,
        classes={"cpu": found.model_copy(update={"load_s": 3.0})},
        measured=Measured(sha="1" * 40, jobs=("1",)),
    )
    packaged = Measures(tmp_path, packaged=(ResourceTable(task="t", backend="b", rows=(row,)),))
    assert packaged.get(UNIT) is not None and packaged.get(UNIT).load_s == 3.0  # type: ignore[union-attr]
