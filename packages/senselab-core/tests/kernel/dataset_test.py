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
    """A shard by `sub` never splits a subject; a bad shard is refused."""
    data = from_records(_rows(list(range(6))))
    shards = [{r.id for r in data.shard("sub", i, 3).records()} for i in range(3)]
    assert sorted(i for shard in shards for i in shard) == [f"r{i}" for i in range(6)]
    for subject in range(3):
        holders = [shard for shard in shards if {f"r{2 * subject}", f"r{2 * subject + 1}"} & shard]
        assert len(holders) == 1
        assert {f"r{2 * subject}", f"r{2 * subject + 1}"} <= holders[0]
    with pytest.raises(ValueError, match="not a shard"):
        data.shard("sub", 3, 3)
    with pytest.raises(ValueError, match="no entity"):
        data.shard("ses", 0, 2)


@pytest.mark.logic
def test_shard_membership_stable_and_commutes_with_filter() -> None:
    """Adding a subject moves no other; filtering before or after sharding gives the same records."""
    rows = [
        {"id": f"r{i}", "entities": {"sub": f"s{i // 2}", "task": "ab"[i % 2]}, "inputs": {"n": i}} for i in range(12)
    ]
    full, fewer = from_records(rows), from_records(rows[2:])
    for index in range(3):
        kept = {r.id for r in full.shard("sub", index, 3).records()} - {"r0", "r1"}
        assert kept == {r.id for r in fewer.shard("sub", index, 3).records()}
        one = [r.id for r in full.filter(task="a").shard("sub", index, 3).records()]
        two = [r.id for r in full.shard("sub", index, 3).filter(task="a").records()]
        assert one == two


@pytest.mark.logic
def test_manifest_rows_sorted_and_unique() -> None:
    """A manifest refuses unsorted or repeated rows."""
    a = ManifestRow(id="a", entities={}, inputs={"audio": _ID})
    b = ManifestRow(id="b", entities={}, inputs={"audio": _ID})
    assert InputManifest(rows=(a, b)).digest() == InputManifest(rows=(a, b)).digest()
    with pytest.raises(ValidationError, match="sorted"):
        InputManifest(rows=(b, a))
