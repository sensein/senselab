"""`flatten@2` leaves and index-free tolerance paths."""

import numpy
import pytest
from senselab_core.runtime.tolerance import FLATTEN_VERSION, ToleranceProfileRetired, flatten, index_free, read_profile


@pytest.mark.logic
@pytest.mark.parametrize(
    ("path", "free"),
    [
        ("embedding.vector", "embedding.vector"),
        ("turns.label[0]", "turns.label[*]"),
        ("transcript.segments[12].words[3].text", "transcript.segments[*].words[*].text"),
        ("grid[0][1]", "grid[*][*]"),
        ("series.channels[0].name", "series.channels[*].name"),
    ],
)
def test_index_free(path: str, free: str) -> None:
    """Every `[n]` becomes `[*]`; a path without an index is itself."""
    assert index_free(path) == free == index_free(free)


@pytest.mark.logic
def test_flatten2_leaves() -> None:
    """Numbers, None and NaN as float lists (absent as None); a string array element by element, like a list."""
    assert FLATTEN_VERSION == "flatten@2"
    assert flatten({"span": None, "x": float("nan"), "v": [1, None]}) == {"span": [None], "v": [1.0, None], "x": [None]}
    labels = numpy.array(["A"] * 1000 + ["B"])
    flat = flatten({"label": labels})
    assert len(flat) == 1001 and flat["label[1000]"] == '"B"'
    assert flatten({"label": ["A", "B"]}) == flatten({"label": numpy.array(["A", "B"])})
    assert flatten({"m": numpy.zeros((2, 2))}) == {"m": [0.0, 0.0, 0.0, 0.0]}


@pytest.mark.logic
def test_retired_profile_refused(tmp_path: object) -> None:
    """A `tolerance@1.0` profile, of indexed paths, is refused with its migration."""
    from pathlib import Path  # noqa: PLC0415

    folder = Path(str(tmp_path))
    (folder / "t").mkdir()
    (folder / "t" / "b.yaml").write_text("format: tolerance@1.0\ntask: t\nbackend: b\n")
    with pytest.raises(ToleranceProfileRetired, match="generate_tolerance.py generate"):
        read_profile(folder, "t", "b")
