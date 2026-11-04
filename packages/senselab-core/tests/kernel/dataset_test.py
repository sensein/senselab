"""Records, the in-memory dataset and the manifest document."""

import random
from pathlib import Path

import pytest
from pydantic import ValidationError
from senselab_core.kernel.dataset import InputManifest, ManifestRow
from senselab_core.runtime.datasets import from_records

_ID = "content:" + "0" * 64


def _rows(order: list[int]) -> list[dict[str, object]]:
    return [
        {"id": f"r{i}", "entities": {"sub": f"s{i // 2}"}, "inputs": {"audio": Path(f"r{i}.wav"), "n": i}}
        for i in order
    ]


@pytest.mark.logic
def test_record_ids_stable_across_walk_order() -> None:
    """Records come out sorted by id whatever order they went in."""
    order = list(range(6))
    shuffled = random.Random(1).sample(order, 6)
    one = [r.id for r in from_records(_rows(order)).records()]
    two = [r.id for r in from_records(_rows(shuffled)).records()]
    assert one == two == [f"r{i}" for i in order]


@pytest.mark.logic
def test_shard_closed_under_subject() -> None:
    """A shard by `sub` never splits a subject; every record lands in one shard; a bad shard is refused."""
    data = from_records(_rows(list(range(6))))
    shards = [{r.id for r in data.shard("sub", i, 2).records()} for i in range(2)]
    assert shards[0] | shards[1] == {f"r{i}" for i in range(6)}
    assert not shards[0] & shards[1]
    for shard in shards:
        assert {int(i[1:]) // 2 for i in shard} <= {0, 1, 2}
        assert all({f"r{2 * s}", f"r{2 * s + 1}"} <= shard for s in {int(i[1:]) // 2 for i in shard})
    with pytest.raises(ValueError, match="not a shard"):
        data.shard("sub", 2, 2)
    with pytest.raises(ValueError, match="no entity"):
        data.shard("ses", 0, 2)


@pytest.mark.logic
def test_manifest_rows_sorted_and_unique() -> None:
    """A manifest refuses unsorted or repeated rows."""
    a = ManifestRow(id="a", entities={}, inputs={"audio": _ID})
    b = ManifestRow(id="b", entities={}, inputs={"audio": _ID})
    assert InputManifest(rows=(a, b)).digest() == InputManifest(rows=(a, b)).digest()
    with pytest.raises(ValidationError, match="sorted"):
        InputManifest(rows=(b, a))
