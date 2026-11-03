"""Parity on the held-out split, from the files an ORCD campaign produced under `$SENSELAB_PARITY_DIR`.

`reference/<task>/<backend>.json` is `parity-reference@3.0` (1.x, converted); `outputs/*.json` is `parity-outputs@3.0`
(2.x, one per backend, stack and device; a backend the node could not place is `not_placeable` and counts as not run
there, never as a pass). Outputs are compared by kernel type under the thresholds of the run config
at `$SENSELAB_PARITY_CONFIG`, merged over the packaged one as `parity_reference.py compare --config` merges it; each
decision against the 1.x reference and each batch-invariance decision is written as a `parity-comparison@1.0` record
under `comparisons/`, carrying the thresholds' digest. A failure is a finding.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from senselab_core.runtime.compare import Thresholds

ROOT = Path(__file__).resolve().parents[4]
CONFIG = Path(os.environ["SENSELAB_PARITY_CONFIG"]) if "SENSELAB_PARITY_CONFIG" in os.environ else None
THRESHOLDS = Thresholds.load(CONFIG)


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


def _ran(outputs: list[Any]) -> list[Any]:
    return [o for o in outputs if o.not_placeable is None]


def _need_outputs() -> None:
    if DIRECTORY is None:
        pytest.skip("no parity campaign: SENSELAB_PARITY_DIR is unset")
    if not OUTPUTS:
        pytest.fail("no parity outputs: set SENSELAB_PARITY_DIR to a directory with outputs/*.json")


def _batch(o: Any, batch: str) -> dict[str, Any]:  # noqa: ANN401
    return {i.id: i.batches[batch] for i in o.items if batch in i.batches}


def _decided(output: Any, reference: Any) -> Any:  # noqa: ANN401
    """Return the parity decision on `output`, written under `comparisons/` beside the outputs."""
    decision = par.decide(output, reference, THRESHOLDS)
    record = DIRECTORY / "comparisons" / f"{_name(output)}-{decision.against}.json"  # type: ignore[operator]
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(decision.model_dump_json(indent=1), "utf-8")
    return decision


@pytest.mark.limits
@pytest.mark.model
def test_outputs_present() -> None:
    """The campaign produced at least one 2.x output file."""
    _need_outputs()


@pytest.mark.limits
@pytest.mark.model
@pytest.mark.parametrize("output", OUTPUTS, ids=_name)
def test_matches_1x_reference(output: Any) -> None:  # noqa: ANN401
    """2.x alone-per-item outputs agree with the 1.x reference; a backend not placeable on the device is skipped."""
    if output.not_placeable is not None:
        pytest.skip(f"{output.backend} not run on {output.device}: {output.not_placeable}")
    reference = par.reference_path(DIRECTORY, output.task, output.backend)
    if not reference.exists():
        pytest.skip(f"no 1.x reference for {output.task} backend {output.backend}")
    decision = _decided(output, par.load_reference(reference))
    assert decision.agree, (decision.thresholds, decision.mismatches)


@pytest.mark.limits
@pytest.mark.model
def test_cross_stack_agreement() -> None:
    """Outputs of one task and model on different stacks, each device alike, agree, whatever the backends."""
    _need_outputs()
    pairs = par.same_model_pairs(_ran(OUTPUTS))
    if not pairs:
        pytest.skip("no task and model was run on two stacks")
    for a, b in pairs:
        found = par.compare_stacks(a, b, THRESHOLDS)
        assert not found, (a.stack, b.stack, THRESHOLDS.canonical_digest(), found)


@pytest.mark.limits
@pytest.mark.model
def test_claimed_batch_invariance() -> None:
    """A run that claims batch invariance has its one-request outputs agree with its alone outputs."""
    _need_outputs()
    claimed = [o for o in _ran(OUTPUTS) if o.claims_batch_invariance]
    if not claimed:
        pytest.skip("no port claims batch invariance")
    for o in claimed:
        decision = _decided(o, None)
        assert decision.agree, (_name(o), decision.thresholds, decision.mismatches)
