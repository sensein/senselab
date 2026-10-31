"""The fake backend and its declared fault model."""

import hashlib

import numpy
import pytest
from senselab_worker.canonical import canonical_npy_bytes
from senselab_worker.codec import MAX_LINE_BYTES
from senselab_worker.digest import identity_digest
from senselab_worker.fake import FakeFaults, OutOfMemoryError, handle, load
from senselab_worker.faults import ContentFault, FaultClass, ResourceFault, classify, error_record

ITEMS = [{"id": "a", "inputs": {"x": 1}}, {"id": "b", "inputs": {"x": 2}}, {"id": "c", "inputs": {"x": 3}}]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _values(results: list[dict]) -> list[str | None]:
    return [r["outputs"]["value"] if "outputs" in r else None for r in results]


@pytest.mark.logic
def test_value_is_hash_of_inputs_and_params_without_faults() -> None:
    """Each value is the `fake-output@1` digest of the item's inputs and the params, faults excluded."""
    state = load({})
    params = {"scale": 2, "faults": {}}
    [out] = handle(state, ITEMS[:1], params)
    assert out == {"outputs": {"value": identity_digest("fake-output@1", {"inputs": {"x": 1}, "params": {"scale": 2}})}}


@pytest.mark.logic
def test_arrays_enter_the_value_by_their_canonical_bytes() -> None:
    """An input array contributes the sha256 of its canonical `.npy` bytes, whatever its layout."""
    a = numpy.arange(6, dtype="<f4").reshape(2, 3)
    item = [{"id": "a", "inputs": {"x": a}}]
    expected = identity_digest("fake-output@1", {"inputs": {"x": _sha(canonical_npy_bytes(a))}, "params": {}})
    assert _values(handle(load({}), item, {})) == [expected]
    swapped = [{"id": "a", "inputs": {"x": numpy.asfortranarray(a.astype(">f4"))}}]
    assert _values(handle(load({}), swapped, {})) == [expected]


@pytest.mark.logic
def test_values_do_not_depend_on_batch_mates_or_failures() -> None:
    """Without coupling, an item's value is the same alone, in a batch, or beside a failure."""
    state = load({})
    alone = [_values(handle(state, [item], {}))[0] for item in ITEMS]
    batched = _values(handle(state, ITEMS, {}))
    with_failure = _values(handle(state, ITEMS, {"faults": {"fail_items": ["b"]}}))
    assert alone == batched and with_failure == [alone[0], None, alone[2]]


@pytest.mark.logic
def test_batch_coupled_values_depend_on_batch_mates() -> None:
    """With `batch_coupled`, a value changes with its batch."""
    state = load({})
    coupled = {"faults": {"batch_coupled": True}}
    assert _values(handle(state, ITEMS[:1], coupled)) != _values(handle(state, ITEMS, coupled))[:1]


@pytest.mark.logic
def test_nondeterministic_values_differ_between_calls() -> None:
    """With `nondeterministic`, repeated calls give different values."""
    state = load({})
    params = {"faults": {"nondeterministic": True}}
    assert _values(handle(state, ITEMS, params)) != _values(handle(state, ITEMS, params))


@pytest.mark.logic
def test_fail_and_oom_items_are_typed_content_and_resource_faults() -> None:
    """`fail_items` raise a `ContentFault`, `oom_items` a `ResourceFault`; neither is a classification gap."""
    results = handle(load({}), ITEMS, {"faults": {"fail_items": ["a"], "oom_items": ["c"]}})
    assert isinstance(results[0]["error"], ContentFault) and classify(results[0]["error"]) is FaultClass.CONTENT
    assert "outputs" in results[1]
    assert isinstance(results[2]["error"], (OutOfMemoryError, ResourceFault))
    assert classify(results[2]["error"]) is FaultClass.RESOURCE
    assert [error_record(results[i]["error"])["gap"] for i in (0, 2)] == [False, False]


@pytest.mark.logic
def test_oversize_and_unprintable_items() -> None:
    """`oversize_items` answer outputs over the line budget; `unprintable_items` an error that cannot be printed."""
    results = handle(load({}), ITEMS, {"faults": {"oversize_items": ["a"], "unprintable_items": ["b"]}})
    assert len(results[0]["outputs"]["value"]) > MAX_LINE_BYTES
    error = results[1]["error"]
    assert isinstance(error, ContentFault)
    for show in (str, repr):
        with pytest.raises(Exception):  # noqa: B017 - any exception: the point is that printing fails
            show(error)
    assert error_record(error)["message"] == f"<unprintable {type(error).__name__}>"
    assert "outputs" in results[2]


@pytest.mark.logic
def test_init_faults_are_the_default_and_params_replace_them() -> None:
    """Faults in `init` apply to every request unless the request names its own."""
    state = load({"faults": {"fail_items": ["a"]}})
    assert "error" in handle(state, ITEMS[:1], {})[0]
    assert "outputs" in handle(state, ITEMS[:1], {"faults": {}})[0]


@pytest.mark.logic
@pytest.mark.parametrize("init", [{"model": "x"}, {"faults": {"bogus": 1}}, {"faults": []}])
def test_unknown_init_is_refused(init: dict) -> None:
    """`load` accepts only `array_dir` and `faults`, with known fault fields."""
    with pytest.raises((ValueError, TypeError)):
        load(init)


@pytest.mark.logic
def test_fake_faults_defaults() -> None:
    """The default fault model injects nothing."""
    assert FakeFaults() == FakeFaults(frozenset(), frozenset(), None, False, False, 0.0, frozenset(), frozenset())
