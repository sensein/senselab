"""Generated value contracts over every `Frozen` type, and the regressions of the findings they close."""

import copy
import pickle
from collections.abc import Callable

import numpy
import pytest
from pydantic import ConfigDict, TypeAdapter, ValidationError
from senselab_core.kernel._base import FrozenMapping, as_label_array, as_ns_array, as_numeric_array, frozen_array
from senselab_core.kernel.results import Embedding, Intervals
from senselab_core.kernel.signals import Audio, ChannelInfo, RegularGroup, SignalArray
from senselab_core.kernel.time import Span, TimeAnchor
from senselab_core.testing.values import frozen_types, registered_types, value_cases
from senselab_worker.canonical import canonical_npy_bytes


@pytest.mark.contract
def test_every_frozen_type_has_samples() -> None:
    """Every concrete `Frozen` type under the kernel and the runtime has registered samples."""
    missing = sorted(t.__qualname__ for t in set(frozen_types()) - registered_types())
    assert not missing, f"register samples for every Frozen type: {missing}"


@pytest.mark.contract
def test_frozen_types_are_found() -> None:
    """The type walk reaches the kernel's value types and the runtime's, so the generated suite is not vacuous."""
    names = {t.__qualname__ for t in frozen_types()}
    assert {"Audio", "RegularGroup", "Intervals", "Embedding", "LogicalKey", "DataManifest"} <= names


@pytest.mark.contract
@pytest.mark.parametrize(("case_id", "case"), value_cases(), ids=lambda c: c if isinstance(c, str) else "")
def test_value_contract(case_id: str, case: Callable[[], None]) -> None:
    """Each generated value contract holds for its type."""
    case()


_SIGNAL: TypeAdapter[numpy.ndarray] = TypeAdapter(SignalArray, config=ConfigDict(arbitrary_types_allowed=True))
_VALIDATORS: dict[str, Callable[[object], numpy.ndarray]] = {
    "ns": as_ns_array,
    "numeric": lambda v: as_numeric_array(v, 1),
    "label": as_label_array,
    "signal": lambda v: _SIGNAL.validate_python(numpy.reshape(v, (1, -1))),
}
_DTYPES = ["?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16", "U3", "S3", "O"]
_DTYPES += ["M8[ns]", "m8[ns]", ">i4", ">f8", "V4"]


@pytest.mark.contract
@pytest.mark.parametrize("validator", sorted(_VALIDATORS))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_admitted_arrays_are_storable_and_owned(validator: str, dtype: str) -> None:
    """Every array validator either refuses a dtype with `ValueError` or admits a frozen, storable copy."""
    source = numpy.zeros(3, dtype=dtype)
    try:
        held = _VALIDATORS[validator](source)
    except (ValueError, ValidationError):
        return
    assert canonical_npy_bytes(held)
    for a in _base_chain(held):
        with pytest.raises(ValueError):
            a.flags.writeable = True


def _base_chain(a: numpy.ndarray) -> list[numpy.ndarray]:
    chain = []
    node: object = a
    while isinstance(node, numpy.ndarray):
        chain.append(node)
        node = node.base
    return chain


@pytest.mark.logic
def test_frozen_array_is_little_endian_c_order_and_detached() -> None:
    """`frozen_array` copies into little-endian C order, whose buffer no array on its chain can write."""
    x = numpy.asfortranarray(numpy.arange(6, dtype=">i4").reshape(2, 3))
    y = frozen_array(x)
    x[0, 0] = 99
    assert y.dtype == numpy.dtype("<i4") and y.flags.c_contiguous and y[0, 0] == 0
    assert isinstance(_base_chain(y)[-1].base, bytes)


@pytest.mark.logic
def test_frozen_array_refuses_subclasses() -> None:
    """A masked array or other subclass is refused, not silently stripped."""
    with pytest.raises(ValueError):
        frozen_array(numpy.ma.masked_array([1.0, 2.0], mask=[True, False]))


def _audio() -> Audio:
    return Audio.from_array(numpy.array([[0.25, 0.5, -0.5]], dtype=numpy.float32), 16_000)


@pytest.mark.logic
def test_a1_flipped_flag_cannot_mutate_held_samples() -> None:
    """A-1: the writeable flag cannot be turned back on, so samples and identity stay bound."""
    g = _audio().groups["audio"]
    assert isinstance(g, RegularGroup) and g.data is not None
    with pytest.raises(ValueError):
        g.data.flags.writeable = True
    with pytest.raises(ValueError):
        g.data[0, 0] = 999


@pytest.mark.logic
def test_a1_copies_hold_read_only_arrays() -> None:
    """A-1: pickle, copy, deepcopy and model_copy return values whose arrays stay read-only."""
    a = _audio()
    for c in (pickle.loads(pickle.dumps(a)), copy.copy(a), copy.deepcopy(a), a.model_copy(), a.model_copy(deep=True)):
        g = c.groups["audio"]
        assert isinstance(g, RegularGroup) and g.data is not None
        with pytest.raises(ValueError):
            g.data.flags.writeable = True


@pytest.mark.logic
def test_a1_group_does_not_alias_the_callers_array() -> None:
    """A-1: `RegularGroup(data=x)` copies `x`; writing `x` afterwards changes nothing held."""
    x = numpy.arange(6, dtype=numpy.float32).reshape(1, 6)
    g = RegularGroup(channels=(ChannelInfo(name="a", type="ecg", units="mV"),), sampling_rate=10.0, data=x)
    digest = g.canonical_digest()
    x[0, 0] = 999
    assert g.read()[0, 0] == 0 and g.canonical_digest() == digest


@pytest.mark.logic
def test_a12_multi_element_values_compare_and_hash() -> None:
    """A-12: `==` and `hash` work on multi-element values and agree with each other."""
    ns = numpy.array([0, 10, 20])
    values = [
        (_audio(), _audio()),
        (_audio().groups["audio"], _audio().groups["audio"]),
        (
            Intervals(start_ns=ns, end_ns=ns + 5, label=numpy.array(["a", "b", "c"]), clock_id="c"),
            Intervals(start_ns=ns, end_ns=ns + 5, label=numpy.array(["a", "b", "c"]), clock_id="c"),
        ),
        (Embedding(vector=numpy.array([1.0, 2.0]), space="s"), Embedding(vector=numpy.array([1.0, 2.0]), space="s")),
    ]
    for a, b in values:
        assert a == b and hash(a) == hash(b)
    assert Embedding(vector=numpy.array([1.0, 2.0]), space="s") != Embedding(vector=numpy.array([1.0, 3.0]), space="s")


@pytest.mark.logic
def test_a13_model_copy_revalidates() -> None:
    """A-13: `model_copy(update=)` runs every validator; `model_construct` is refused."""
    g = _audio().groups["audio"]
    assert isinstance(g, RegularGroup)
    with pytest.raises(ValidationError):
        g.model_copy(update={"sampling_rate": -5})
    with pytest.raises(ValidationError):
        g.model_copy(update={"channels": ()})
    with pytest.raises(ValidationError):
        Span(start_ns=0, end_ns=1, clock_id="c").model_copy(update={"end_ns": -1})
    with pytest.raises(TypeError):
        Span.model_construct(start_ns=5, end_ns=0, clock_id="c")


@pytest.mark.logic
def test_a13_frames_follow_the_data() -> None:
    """A-13: a copy with other data reports that data's frame count."""
    g = _audio().groups["audio"]
    assert isinstance(g, RegularGroup)
    swapped = g.model_copy(update={"data": numpy.zeros((1, 10), dtype=numpy.float32)})
    assert swapped.frames == 10 and swapped.read().shape == (1, 10)


@pytest.mark.logic
def test_a37_forged_identity_is_refused() -> None:
    """A-37: a recording's identity is derived; a supplied identity that differs is refused."""
    a = _audio()
    with pytest.raises(ValidationError):
        Audio.model_validate({**a.model_dump(), "identity": "content:" + "f" * 64})
    with pytest.raises(ValidationError):
        Audio(groups=a.groups, anchor=TimeAnchor(clock_id="c", absolute_start_ns=None), identity="sha-forged")


@pytest.mark.logic
def test_frozenmapping_has_no_mutable_reach() -> None:
    """A-37: neither the held items nor the attribute that holds them can be changed."""
    m = FrozenMapping({"k": 0})
    with pytest.raises(TypeError):
        m._items["k"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        m._items = {"k": 1}  # type: ignore[misc, assignment]
    assert m["k"] == 0 and pickle.loads(pickle.dumps(m)) == m and copy.deepcopy(m) == m
