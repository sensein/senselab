"""The packaged equivalence table is read by the runtime, not the kernel (ADR-009)."""

import pytest
from senselab_core.runtime.equivalence import EQUIVALENCE, load_equivalence
from senselab_core.runtime.store import ResultStore


@pytest.mark.logic
def test_phase_1_has_no_measured_equivalence() -> None:
    """The packaged table, read from `runtime/data/equivalence.yaml`, is empty in Phase 1."""
    assert dict(load_equivalence()) == {}
    assert dict(EQUIVALENCE) == {}


@pytest.mark.logic
def test_a_store_defaults_to_the_packaged_table(tmp_path: object) -> None:
    """`ResultStore(equivalence=)` defaults to the table the runtime loaded."""
    assert ResultStore(tmp_path).equivalence is EQUIVALENCE  # type: ignore[arg-type]
