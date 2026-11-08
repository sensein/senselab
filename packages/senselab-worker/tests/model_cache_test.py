"""The model cache: one load per key, LRU eviction under the budget, keys pinned by commit."""

from collections.abc import Callable

import pytest
from senselab_worker.faults import FaultClass, classify
from senselab_worker.model_cache import ModelCache, ModelExceedsBudget

COMMIT = "a" * 40
DIGEST = "b" * 64


def _key(identity: str = COMMIT, backend: str = "fake") -> tuple[str, str, str, str]:
    return (backend, identity, "cpu", "float32")


def _loader(loads: list[str], name: str) -> Callable[[], object]:
    def load() -> object:
        loads.append(name)
        return {"model": name}

    return load


@pytest.mark.logic
def test_model_loaded_once_across_calls() -> None:
    """A second `get_or_load` of one key returns the cached state without calling the loader."""
    cache, loads = ModelCache(10), list[str]()
    first = cache.get_or_load(_key(), _loader(loads, "m"), lambda _: 1)
    second = cache.get_or_load(_key(), _loader(loads, "again"), lambda _: 1)
    assert second is first and loads == ["m"]
    cache.get_or_load(_key(DIGEST), _loader(loads, "w"), lambda _: 1)
    assert loads == ["m", "w"] and cache.keys() == [_key(), _key(DIGEST)]


@pytest.mark.logic
def test_clear_drops_every_state_and_a_later_get_loads_again() -> None:
    """After `clear` the cache holds nothing and the next `get_or_load` of a key calls its loader again."""
    cache, loads = ModelCache(10), list[str]()
    cache.get_or_load(_key(), _loader(loads, "m"), lambda _: 4)
    cache.clear()
    assert cache.keys() == [] and cache.used_bytes == 0
    cache.get_or_load(_key(), _loader(loads, "again"), lambda _: 4)
    assert loads == ["m", "again"]


@pytest.mark.logic
def test_cache_evicts_under_budget() -> None:
    """A new entry evicts least recently used entries until the sizes fit the budget; a hit refreshes recency."""
    cache, loads = ModelCache(10), list[str]()
    a, b, c = (_key(backend=n) for n in "abc")
    cache.get_or_load(a, _loader(loads, "a"), lambda _: 4)
    cache.get_or_load(b, _loader(loads, "b"), lambda _: 4)
    cache.get_or_load(a, _loader(loads, "a"), lambda _: 4)
    cache.get_or_load(c, _loader(loads, "c"), lambda _: 4)
    assert cache.keys() == [a, c] and cache.used_bytes == 8
    cache.get_or_load(b, _loader(loads, "b"), lambda _: 10)
    assert cache.keys() == [b] and cache.used_bytes == 10 and loads == ["a", "b", "c", "b"]


@pytest.mark.logic
def test_model_cache_evicts_before_load() -> None:
    """A key whose size is known evicts what it needs before its loader runs, and the loader may read the cache."""
    cache = ModelCache(10)
    a, b = _key(backend="a"), _key(backend="b")
    seen: list[tuple[list[tuple[str, str, str, str]], int]] = []

    def load() -> object:
        seen.append((cache.keys(), cache.used_bytes))
        return {}

    cache.get_or_load(a, load, lambda _: 6)
    cache.get_or_load(b, load, lambda _: 6)
    assert cache.keys() == [b] and seen[1] == ([a], 6)
    cache.get_or_load(a, load, lambda _: 6)
    assert seen[2] == ([], 0) and cache.keys() == [a]


@pytest.mark.logic
def test_oversize_model_is_resource_fault_and_evicts_nothing() -> None:
    """A state larger than the whole budget is a `ResourceFault`; the cached entries stay."""
    cache = ModelCache(4)
    cache.get_or_load(_key(backend="a"), lambda: 1, lambda _: 4)
    with pytest.raises(ModelExceedsBudget) as raised:
        cache.get_or_load(_key(backend="b"), lambda: 2, lambda _: 5)
    assert classify(raised.value) is FaultClass.RESOURCE
    assert cache.keys() == [_key(backend="a")]


@pytest.mark.logic
def test_failed_load_caches_nothing() -> None:
    """A loader that raises leaves no entry, so the next call loads again."""
    cache, loads = ModelCache(4), list[str]()

    def broken() -> object:
        raise RuntimeError("no weights")

    with pytest.raises(RuntimeError):
        cache.get_or_load(_key(), broken, lambda _: 1)
    assert cache.keys() == []
    cache.get_or_load(_key(), _loader(loads, "m"), lambda _: 1)
    assert loads == ["m"]


@pytest.mark.logic
@pytest.mark.parametrize("identity", ["main", "v1.0", COMMIT[:-1], COMMIT.upper(), "a" * 41])
def test_key_refuses_a_ref(identity: str) -> None:
    """A model identity that is not a 40-hex commit or 64-hex digest is refused before any load."""
    loads: list[str] = []
    with pytest.raises(ValueError, match="never a ref"):
        ModelCache(4).get_or_load(_key(identity), _loader(loads, "m"), lambda _: 1)
    assert loads == []


@pytest.mark.logic
@pytest.mark.parametrize("key", [("fake", COMMIT, "cpu"), ("", COMMIT, "cpu", "f32"), ["fake", COMMIT, "cpu", "f32"]])
def test_key_must_be_four_strings(key: object) -> None:
    """A key is a tuple of four non-empty strings."""
    with pytest.raises(ValueError, match="model key"):
        ModelCache(4).get_or_load(key, lambda: 1, lambda _: 1)  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize("size", [-1, 1.0, True, None])
def test_size_of_must_be_a_non_negative_int(size: object) -> None:
    """`size_of` returning anything but a non-negative int is refused and nothing is cached."""
    cache = ModelCache(4)
    with pytest.raises(ValueError, match="size_of"):
        cache.get_or_load(_key(), lambda: 1, lambda _: size)  # type: ignore[arg-type, return-value]
    assert cache.keys() == []


@pytest.mark.logic
@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_budget_must_be_a_non_negative_int(budget: object) -> None:
    """The budget is a non-negative int."""
    with pytest.raises(ValueError, match="budget_bytes"):
        ModelCache(budget)  # type: ignore[arg-type]
