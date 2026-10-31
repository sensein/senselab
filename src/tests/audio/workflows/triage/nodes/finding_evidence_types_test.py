"""Every producer of a branch finding writes, through `write_findings`, evidence of every provenance value type."""

import enum
from decimal import Decimal
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from senselab_core.runtime.provenance import ProvStore, UnrecordableValue

from senselab.audio.workflows.triage.nodes.branches import (
    DEVIATION_TYPES,
    CountUnit,
    Finding,
    RequiredCount,
    TypicalCount,
    contest,
    count,
    count_against_instruction,
    count_beside_typical,
    deviation,
    instructed_fraction,
    measured,
    task_extent_absent,
    unviable,
    write_findings,
)


class _Shade(enum.Enum):
    LIGHT = "light"


EVIDENCE: dict[str, object] = {
    "none": None,
    "flag": True,
    "count": 3,
    "stamp_ns": 1_761_900_000_000_000_000,
    "level_dbfs": float("-inf"),
    "peak": float("inf"),
    "spread": float("nan"),
    "ratio": 0.25,
    "text": "é",
    "listed": [1, 2.5, None],
    "pairs": ((0.0, 1.0), (1.0, 2.0)),
    "nested": {"$f": "kept", "inner": {"x": [float("-inf")]}},
    "numpy_float": np.float32(0.5),
    "numpy_nonfinite": np.float64("-inf"),
    "numpy_int": np.int64(2**60),
    "numpy_bool": np.bool_(False),
    "vector": np.linspace(0.0, 1.0, 4, dtype="<f4"),
    "matrix": np.array([[1, 2], [3, 4]], dtype=">i2"),
    "path": Path("/data/a.wav"),
    "pure_path": PurePosixPath("b/c.wav"),
    "unit": CountUnit.EVENTS,
    "shade": _Shade.LIGHT,
    "decimal": Decimal("0.10"),
    "bytes": b"\x00\xff",
    "labels": {"b", "a"},
    "frozen": frozenset({2, 1}),
}
"""One value of every type the declared provenance value type admits, as branch evidence may carry it."""

DEVIATION = next(iter(DEVIATION_TYPES))
REQUIRED = RequiredCount(value=4, unit=CountUnit.EVENTS)
TYPICAL = TypicalCount(median=4, unit=CountUnit.EVENTS, derivation="specs/x.md")
PRODUCERS: dict[str, object] = {
    "deviation": lambda ev: deviation(DEVIATION, 0.0, 1.0, "src", **ev),
    "measured": lambda ev: measured("m", 0.0, 1.0, ev["level_dbfs"], "src", **ev),
    "count": lambda ev: count("c", ev, ev["vector"]),
    "count_against_instruction": lambda ev: count_against_instruction(REQUIRED, 3),
    "instructed_fraction": lambda ev: instructed_fraction(REQUIRED, 3),
    "count_beside_typical": lambda ev: count_beside_typical(TYPICAL, 3),
    "contest": lambda ev: contest("span-1", (0.0, 1.0), "c", "r"),
    "unviable": lambda ev: unviable("u", "why"),
    "task_extent_absent": lambda ev: task_extent_absent("reason", "src"),
}


def _write(findings: list[Finding]) -> list[str]:
    store = ProvStore(run_id="r")
    return write_findings(store, "act-1", "agent-1", findings, signal="raw")


@pytest.mark.parametrize("producer", sorted(PRODUCERS))
def test_every_producer_writes_evidence_of_every_declared_type(producer: str) -> None:
    """Each finding producer's evidence, holding every declared type at once, is written with a stable id."""
    build = PRODUCERS[producer]
    first = _write([build(dict(EVIDENCE))])  # type: ignore[operator]
    again = _write([build(dict(EVIDENCE))])  # type: ignore[operator]
    assert first == again and all(first)


@pytest.mark.parametrize("name", sorted(EVIDENCE))
def test_each_declared_type_alone_reaches_the_store(name: str) -> None:
    """Every declared type, as the only evidence of a deviation and of a measurement, gets an id."""
    one = {name: EVIDENCE[name]}
    assert all(_write([deviation(DEVIATION, None, None, **one), measured("m", None, None, EVIDENCE[name], **one)]))


def test_an_undeclared_evidence_type_is_refused_at_the_store_with_its_path() -> None:
    """Evidence holding a type outside the declaration fails with `UnrecordableValue` naming the attribute."""
    with pytest.raises(UnrecordableValue) as caught:
        _write([deviation(DEVIATION, None, None, window={"z": complex(1, 1)})])
    assert caught.value.path == "attributes.window.z" and caught.value.type_name == "builtins.complex"
