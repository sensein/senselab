"""`benchmarks/scale/batch_sizes.py` over a fake worker: request sizes, the size-1 comparison and the probe's calls."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.logic

SCRIPT = Path(__file__).resolve().parents[4] / "benchmarks" / "scale" / "batch_sizes.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("batch_sizes", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Worker:
    """Echoes each item's id, with its request's size when `mates` is set; records each request's size."""

    def __init__(self, mates: bool) -> None:
        self.mates = mates
        self.sizes: list[int] = []

    def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG002
        self.sizes.append(len(items))
        return [{"outputs": {"id": i["id"], **({"n": len(items)} if self.mates else {})}} for i in items]


def _records(n: int) -> list[tuple[dict[str, Any], float]]:
    return [({"id": f"r{i}", "inputs": {}}, 1.0 + i % 2) for i in range(n)]


@pytest.mark.parametrize(("mates", "equal"), [(False, True), (True, False)], ids=["invariant", "batch-dependent"])
def test_explicit_sizes_compare_each_output_with_its_size_one_output(mates: bool, equal: bool) -> None:
    """Every size sends all records in requests of that size; an output that depends on batch-mates is unequal."""
    worker = _Worker(mates)
    found = _module().explicit(worker, _records(8), [4, 2])
    assert worker.sizes == [1] * 8 + [2] * 4 + [4] * 2
    assert found["1"]["equal_to_alone"] and found["4"]["equal_to_alone"] is equal and found["4"]["calls"] == 2


def test_the_probe_sends_every_record_once_in_power_of_two_calls_within_its_bound() -> None:
    """The probe's calls cover the records exactly, each a power of two up to the bound (the tail excepted)."""
    worker = _Worker(False)
    found = _module().probed(worker, _records(40), bound=8, samples=2, gain=0.05)
    assert sum(found["calls"]) == 40 and worker.sizes == found["calls"]
    assert all(c <= 8 and c & (c - 1) == 0 for c in found["calls"][:-1]) and found["calls"][:2] == [1, 1]
