"""Tests for categorical extraction: the declared allow-lists, the value checks and keying."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from senselab.utils.reproduce.extract import validate_extracted
from senselab.utils.reproduce.spec import ExtractSpec, StepSpec
from tests.utils.reproduce.cache_test import _key, _step_data

REV = "1" * 40
SPECS = [
    {"name": "model", "kind": "model_id", "allowed": ["org/model-a", f"org/model-b@{REV}"]},
    {"name": "matched", "kind": "yesno"},
    {"name": "run_dir", "kind": "index", "allowed": ["run-00", "run-01", "run-02"]},
    {"name": "map_by_epoch", "kind": "series", "max_len": 3},
    {"name": "detections", "kind": "count"},
]
GOOD = {"model": "org/model-a", "matched": "yes", "run_dir": 2, "map_by_epoch": [0.1, 0.25, 0.4], "detections": 12}


def _specs(*specs: dict[str, Any]) -> list[ExtractSpec]:
    return [ExtractSpec.model_validate(s) for s in (specs or SPECS)]


def _check(obj: object, *specs: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    return validate_extracted(obj, _specs(*specs), min_cell=5)


def test_each_kind_accepts_its_allowed_values() -> None:
    """A model id from the list, yes/no, an index into the list, a short numeric series and a count pass."""
    clean, codes = _check(GOOD)
    assert codes == [] and clean == GOOD
    other = {**GOOD, "model": f"org/model-b@{REV}", "matched": "no", "run_dir": 0, "map_by_epoch": []}
    clean, codes = _check(other)
    assert codes == [] and clean == other
    clean, codes = _check({**GOOD, "detections": 3})
    assert codes == [] and clean is not None and clean["detections"] == "<5"
    clean, codes = _check({**GOOD, "detections": 0})
    assert codes == [] and clean is not None and clean["detections"] == "<5"


def test_free_text_and_undeclared_names_are_refused() -> None:
    """A path or sentence in a value, an extra name, a missing name or a non-object is ``E_EXTRACT``, unechoed."""
    secret = "see /x/y.csv"
    cases: list[object] = [
        {**GOOD, "model": secret},
        {**GOOD, "matched": secret},
        {**GOOD, "matched": True},
        {**GOOD, "note": "free text"},
        {k: v for k, v in GOOD.items() if k != "matched"},
        [GOOD],
        secret,
    ]
    for obj in cases:
        clean, codes = _check(obj)
        assert clean is None and codes == ["E_EXTRACT"], obj
        assert secret not in " ".join(codes)


def test_index_out_of_range_and_series_too_long_are_refused() -> None:
    """An index outside the list, a series past ``max_len`` or with a non-number, or a bad count are refused."""
    for change in (
        {"run_dir": 3},
        {"run_dir": -1},
        {"run_dir": True},
        {"run_dir": "run-01"},
        {"map_by_epoch": [0.1, 0.2, 0.3, 0.4]},
        {"map_by_epoch": [0.1, float("nan")]},
        {"map_by_epoch": [0.1, "0.2"]},
        {"map_by_epoch": 0.1},
        {"detections": -1},
        {"detections": 2.5},
        {"detections": "<5"},
    ):
        clean, codes = _check({**GOOD, **change})
        assert clean is None and codes == ["E_EXTRACT"], change
    for bad in (
        {"name": "model", "kind": "model_id"},
        {"name": "model", "kind": "model_id", "allowed": ["not a repo"]},
        {"name": "model", "kind": "model_id", "allowed": ["org/model@main"]},
        {"name": "run_dir", "kind": "index", "allowed": []},
        {"name": "run_dir", "kind": "index", "allowed": ["run-00", "run-00"]},
        {"name": "map_by_epoch", "kind": "series"},
        {"name": "map_by_epoch", "kind": "series", "max_len": 0},
        {"name": "matched", "kind": "yesno", "allowed": ["yes"]},
        {"name": "matched", "kind": "yesno", "max_len": 2},
        {"name": "Bad-Name", "kind": "yesno"},
    ):
        with pytest.raises(ValidationError, match="E_SPEC_EXTRACT|String should match|greater than"):
            ExtractSpec.model_validate(bad)
    with pytest.raises(ValidationError, match="E_SPEC_EXTRACT"):
        StepSpec.model_validate(_step_data(extract=[SPECS[1], SPECS[1]]))


def test_extract_changes_the_step_key(tmp_path: Path) -> None:
    """Declaring an extraction, or changing one, moves the key."""
    base = _key(tmp_path)
    one = _key(tmp_path, _step_data(extract=[SPECS[1]]))
    assert one != base
    assert _key(tmp_path, _step_data(extract=[{"name": "matched_too", "kind": "yesno"}])) != one
    longer = dict(SPECS[3], max_len=4)
    assert _key(tmp_path, _step_data(extract=[longer])) != _key(tmp_path, _step_data(extract=[SPECS[3]]))
