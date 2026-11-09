"""`size` derives batch, quantum and workers from a measure; `Measures` reads rows."""

from pathlib import Path

import pytest
from senselab_core.kernel.formats import FieldsOfLaterMinor
from senselab_core.runtime.config import dump_yaml, load_yaml
from senselab_core.runtime.execution_settings import PoolSettings, ResourceClass
from senselab_core.runtime.resources import (
    ClassMeasure,
    Measured,
    ResourceMeasure,
    ResourceRow,
    ResourceTable,
    Unit,
    write_row,
)
from senselab_core.runtime.sizing import Grant, Measures

UNIT = Unit("t", "b", "org/m", "a" * 40, "cpu")
MEASURED = Measured(sha=None, run="r", jobs=("7",))
FOUND = ResourceMeasure(
    load_s=1, claim_s=0.01, resident_bytes=1, peak_memory_bytes=2, s_per_input_s=0.5, input_s_per_item=None
)


@pytest.mark.logic
def test_a_grant_takes_device_memory_on_a_device_and_host_memory_otherwise() -> None:
    """On a device the class's device cap; else its memory, the host's last; the class's time bounds the quantum."""
    pool = PoolSettings(
        quantum_items=100,
        quantum_s=300,
        poll_s=30,
        first_poll_s=1,
        presence_refresh_s=30,
        segment_items=512,
        admit_threads=8,
    )
    capped = ResourceClass(device="cuda", max_workers=4, mem_bytes=2**36, memory_cap_bytes=2**33, time_s=60)
    assert Grant.of(capped, pool, on_device=True).model_dump() == {
        "max_workers": 4,
        "memory_bytes": 2**33,
        "quantum_items": 100,
        "quantum_s": 60,
        "devices": None,
        "on_device": True,
    }
    assert Grant.of(capped, pool, devices=2).devices == 2
    assert Grant.of(capped, pool).memory_bytes == 2**36
    bare = ResourceClass(device="cpu", max_workers=1)
    assert Grant.of(bare, pool, devices=2).devices is None
    assert Grant.of(bare, pool).memory_bytes is None and Grant.of(bare, pool, 2**30).memory_bytes == 2**30


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
def test_a_table_of_a_later_minor_is_not_rewritten_but_one_that_lost_nothing_is(tmp_path: Path) -> None:
    """Rewriting drops no row's later-minor fields: such a table is refused, named by its minor, and left as it was."""
    write_row(tmp_path, UNIT, FOUND, MEASURED)
    path = tmp_path / "t" / "b.yaml"
    document = load_yaml(path, "table")
    document["format"] = "resource-table@1.1"
    path.write_text(dump_yaml(document))
    other = Unit("t", "b", "org/n", "b" * 40, "cpu")
    write_row(tmp_path, other, FOUND, MEASURED)
    assert load_yaml(path, "table")["format"] == "resource-table@1.0"
    document = load_yaml(path, "table")
    document["format"] = "resource-table@1.1"
    document["rows"][0]["added_later"] = 1
    path.write_text(dump_yaml(document))
    before = path.read_bytes()
    with pytest.raises(FieldsOfLaterMinor, match=r"resource-table@1\.1.*ResourceRow\.added_later"):
        write_row(tmp_path, UNIT, FOUND, MEASURED)
    assert path.read_bytes() == before


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
