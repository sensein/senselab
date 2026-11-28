"""`benchmarks/scale/batch_sizes.py` over a fake handle and device: warm-up, shuffled rounds, verdict, memory bound."""

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from senselab_core.testing.pools import POOL_PROFILE

pytestmark = pytest.mark.logic

SCRIPT = Path(__file__).resolve().parents[4] / "benchmarks" / "scale" / "batch_sizes.py"
RESIDENT, M0, M1, TOTAL = 1000, 50.0, 10.0, 10**6


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("batch_sizes", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Device:
    """Memory that grows as `M0 + M1 × size × longest` above `RESIDENT` with each request `_Handle` answers."""

    def __init__(self) -> None:
        self.peak_bytes = RESIDENT

    def allocated(self) -> int:
        return RESIDENT

    def reset(self) -> None:
        self.peak_bytes = RESIDENT

    def peak(self) -> int:
        return self.peak_bytes

    def total(self) -> int:
        return TOTAL


class _Handle:
    """Answers each item with its id (and its request's size when `mates`), or one error per item when `fail`."""

    def __init__(self, device: _Device, durations: dict[str, float], mates: bool = False, fail: bool = False) -> None:
        self.device, self.durations, self.mates, self.fail = device, durations, mates, fail
        self.sizes: list[int] = []

    def __call__(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.sizes.append(len(items))
        longest = max(self.durations[i["id"]] for i in items)
        self.device.peak_bytes = round(RESIDENT + M0 + M1 * len(items) * longest)
        if self.fail:
            return [{"error": {"fault": "content", "message": "same"}} for _ in items]
        return [{"outputs": {"id": i["id"], **({"n": len(items)} if self.mates else {})}} for i in items]


def _setup(mates: bool = False, fail: bool = False) -> tuple[ModuleType, list[Any], _Device, _Handle]:
    module = _module()
    recs = module.records(8, [2.0, 1.0])
    device = _Device()
    return module, recs, device, _Handle(device, {r[0]["id"]: r[1] for r in recs}, mates, fail)


def test_records_are_sorted_by_duration_and_every_size_is_warmed_once_per_duration() -> None:
    """Records come shortest first; the warm-up sends each size once for each distinct duration, before timing."""
    module, recs, device, handle = _setup()
    assert [s for _, s in recs] == [1.0] * 4 + [2.0] * 4
    assert module.Bench(handle, device, RESIDENT).warm_up(recs, [1, 4]) == 4 and handle.sizes == [1, 4, 1, 4]


def test_rounds_run_the_sizes_in_a_recorded_seeded_order() -> None:
    """Each round's order is a seeded shuffle of the sizes, recorded, and the same seed gives the same orders."""
    module, recs, device, handle = _setup()
    _, orders, _ = module.Bench(handle, device, RESIDENT).explicit(recs, module.Plan([1, 2, 4], 3, 7))
    again = module.Bench(_setup()[3], device, RESIDENT).explicit(recs, module.Plan([1, 2, 4], 3, 7))[1]
    assert orders == again and all(sorted(o) == [1, 2, 4] for o in orders)
    assert any(o != [1, 2, 4] for o in orders)


@pytest.mark.parametrize(
    ("mates", "invariant", "passed"),
    [(False, True, True), (True, True, False), (True, False, True)],
    ids=["invariant-agrees", "invariant-disagrees", "variant-reported"],
)
def test_the_verdict_judges_agreement_only_for_a_batch_invariant_backend(
    mates: bool, invariant: bool, passed: bool
) -> None:
    """Outputs that depend on batch-mates fail a batch-invariant backend; a batch-variant one only reports them."""
    module, recs, device, handle = _setup(mates)
    doc = module.run(module.Loaded(handle, invariant), device, POOL_PROFILE, recs, module.Plan([1, 4], 1, 0))
    assert doc["pass"] is passed
    assert doc["explicit"]["1"]["agree_with_alone"] == 8
    assert doc["explicit"]["4"]["agree_with_alone"] == (0 if mates else 8)


@pytest.mark.parametrize("invariant", [True, False], ids=["invariant", "variant"])
def test_items_answered_with_errors_fail_even_when_every_size_errs_alike(invariant: bool) -> None:
    """Identical errors at every size are no agreement: the run fails and counts them, whatever the backend."""
    module, recs, device, handle = _setup(fail=True)
    doc = module.run(module.Loaded(handle, invariant), device, POOL_PROFILE, recs, module.Plan([1, 4], 1, 0))
    assert doc["pass"] is False and doc["explicit"]["4"]["errors"] == 8


def test_the_probe_is_bounded_by_pydras_memory_bound_of_the_measured_fit() -> None:
    """The fit recovers the device's growth, the bound is pydra's formula over it, and the probe stays within it."""
    module, recs, device, handle = _setup()
    doc = module.run(module.Loaded(handle, True), device, POOL_PROFILE, recs, module.Plan([1, 2, 4], 1, 0))
    fit = doc["memory_fit"]
    assert fit["m1"] == pytest.approx(M1, rel=1e-3) and fit["m0"] == pytest.approx(M0, abs=1.0)
    room = POOL_PROFILE.sizing.memory_headroom * TOTAL - RESIDENT - fit["m0"]
    expected = max(1, min(POOL_PROFILE.pool.quantum_items, math.floor(room / (fit["m1"] * 2.0))))
    calls = doc["probe"]["calls"]
    assert doc["probe"]["bound"] == expected and sum(calls) == 8
    assert all(c <= expected and c & (c - 1) == 0 for c in calls[:-1])
