"""Parity on the held-out split, from the files an ORCD campaign produced under `$SENSELAB_PARITY_DIR`.

`reference/<task>/<backend>.json` is `parity-reference@3.0` (1.x, converted); `outputs/*.json` is `parity-outputs@2.0`
(2.x, one per backend, stack and device). Outputs are compared by kernel type under the packaged run config's
thresholds, whose digest every failure names. A failure is a finding.
"""

import importlib.util
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
from senselab_core.runtime.compare import Thresholds

ROOT = Path(__file__).resolve().parents[4]
THRESHOLDS = Thresholds.load()


def _load() -> Any:  # noqa: ANN401
    spec = importlib.util.spec_from_file_location("parity_reference", ROOT / "benchmarks" / "parity_reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


par = _load()
DIRECTORY = Path(os.environ["SENSELAB_PARITY_DIR"]) if "SENSELAB_PARITY_DIR" in os.environ else None


def _outputs() -> list[Any]:
    if DIRECTORY is None:
        return []
    return [par.load_outputs(p) for p in sorted((DIRECTORY / "outputs").glob("*.json"))]


OUTPUTS = _outputs()


def _name(o: Any) -> str:  # noqa: ANN401
    return f"{o.task}-{o.backend}-{o.stack}-{o.device}".replace(" ", "_")


def _need_outputs() -> None:
    if not OUTPUTS:
        pytest.fail("no parity outputs: set SENSELAB_PARITY_DIR to a directory with outputs/*.json")


def _batch(o: Any, batch: str) -> dict[str, Any]:  # noqa: ANN401
    return {i.id: i.batches[batch] for i in o.items if batch in i.batches}


@pytest.mark.limits
@pytest.mark.model
def test_outputs_present() -> None:
    """The campaign produced at least one 2.x output file."""
    _need_outputs()


@pytest.mark.limits
@pytest.mark.model
@pytest.mark.parametrize("output", OUTPUTS, ids=_name)
def test_matches_1x_reference(output: Any) -> None:  # noqa: ANN401
    """2.x alone-per-item outputs agree with the 1.x reference."""
    reference = DIRECTORY / "reference" / output.task / f"{output.backend}.json"  # type: ignore[operator]
    if not reference.exists():
        pytest.skip(f"no 1.x reference for {output.task} backend {output.backend}")
    ref = par.load_reference(reference)
    found = par.compare_items({i.id: i.fields for i in ref.items}, _batch(output, "1"), THRESHOLDS)
    assert not found, (THRESHOLDS.canonical_digest(), found)


@pytest.mark.limits
@pytest.mark.model
def test_cross_stack_agreement() -> None:
    """Outputs of one backend on different stacks, each device alike, agree."""
    _need_outputs()
    pairs = [
        (a, b)
        for a, b in combinations(OUTPUTS, 2)
        if (a.task, a.backend) == (b.task, b.backend) and a.stack != b.stack and a.device == b.device
    ]
    if not pairs:
        pytest.skip("no backend was run on two stacks")
    for a, b in pairs:
        found = par.compare_items(_batch(a, "1"), _batch(b, "1"), THRESHOLDS)
        assert not found, (a.stack, b.stack, THRESHOLDS.canonical_digest(), found)


@pytest.mark.limits
@pytest.mark.model
def test_claimed_batch_invariance() -> None:
    """A run that claims batch invariance has its one-request outputs agree with its alone outputs."""
    _need_outputs()
    claimed = [o for o in OUTPUTS if o.claims_batch_invariance]
    if not claimed:
        pytest.skip("no port claims batch invariance")
    for o in claimed:
        found = par.compare_items(_batch(o, "1"), _batch(o, "N"), THRESHOLDS)
        assert not found, (_name(o), THRESHOLDS.canonical_digest(), found)
