"""Generated value contracts over every `Frozen` type, and the regressions of the findings they close."""

import copy
import hashlib
import os
import pickle
import time
import types
import unicodedata
from collections.abc import Callable
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import numpy
import pytest
import soundfile
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ConfigDict, TypeAdapter, ValidationError
from senselab_core.kernel._base import (
    Frozen,
    FrozenMapping,
    as_label_array,
    as_ns_array,
    as_numeric_array,
    frozen_array,
    value_digest,
)
from senselab_core.kernel.identity import NO_EQUIVALENCE, ExecutionContext, LogicalKey, ModelRef, acceptable
from senselab_core.kernel.license import License
from senselab_core.kernel.results import Embedding, Events, Intervals
from senselab_core.kernel.signals import Audio, ChannelInfo, Recording, RegularGroup, SignalArray, StaleSource
from senselab_core.kernel.time import Span, TimeAnchor, TimeMap
from senselab_core.runtime import audio_io
from senselab_core.runtime.audio_io import READER, open_audio, source_group
from senselab_core.testing.values import frozen_types, registered_types, value_cases
from senselab_worker.canonical import canonical_npy_bytes
from senselab_worker.digest import identity_digest


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
_VALIDATORS: dict[str, Callable[[Any], numpy.ndarray]] = {
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
def test_flipped_flag_cannot_mutate_held_samples() -> None:
    """The writeable flag cannot be turned back on, so samples and identity stay bound."""
    g = _audio().groups["audio"]
    assert isinstance(g, RegularGroup) and g.data is not None
    with pytest.raises(ValueError):
        g.data.flags.writeable = True
    with pytest.raises(ValueError):
        g.data[0, 0] = 999


@pytest.mark.logic
def test_copies_hold_read_only_arrays() -> None:
    """Pickle, copy, deepcopy and model_copy return values whose arrays stay read-only."""
    a = _audio()
    for c in (pickle.loads(pickle.dumps(a)), copy.copy(a), copy.deepcopy(a), a.model_copy(), a.model_copy(deep=True)):
        g = c.groups["audio"]
        assert isinstance(g, RegularGroup) and g.data is not None
        with pytest.raises(ValueError):
            g.data.flags.writeable = True


@pytest.mark.logic
def test_group_does_not_alias_the_callers_array() -> None:
    """`RegularGroup(data=x)` copies `x`; writing `x` afterwards changes nothing held."""
    x = numpy.arange(6, dtype=numpy.float32).reshape(1, 6)
    g = RegularGroup(channels=(ChannelInfo(name="a", type="ecg", units="mV"),), sampling_rate=10.0, data=x)
    digest = g.canonical_digest()
    x[0, 0] = 999
    assert g.read()[0, 0] == 0 and g.canonical_digest() == digest


@pytest.mark.logic
def test_multi_element_values_compare_and_hash() -> None:
    """`==` and `hash` work on multi-element values and agree with each other."""
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
def test_model_copy_revalidates() -> None:
    """`model_copy(update=)` runs every validator; `model_construct` is refused."""
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
def test_frames_follow_the_data() -> None:
    """A copy with other data reports that data's frame count."""
    g = _audio().groups["audio"]
    assert isinstance(g, RegularGroup)
    swapped = g.model_copy(update={"data": numpy.zeros((1, 10), dtype=numpy.float32)})
    assert swapped.frames == 10 and swapped.read().shape == (1, 10)


@pytest.mark.logic
def test_forged_identity_is_refused() -> None:
    """A recording's identity is derived; a supplied identity that differs is refused."""
    a = _audio()
    with pytest.raises(ValidationError):
        Audio.model_validate({**a.model_dump(), "identity": "content:" + "f" * 64})
    with pytest.raises(ValidationError):
        Audio(groups=a.groups, anchor=TimeAnchor(clock_id="c", absolute_start_ns=None), identity="sha-forged")


@pytest.mark.logic
def test_frozenmapping_has_no_mutable_reach() -> None:
    """Neither the held items nor the attribute that holds them can be changed."""
    m = FrozenMapping({"k": 0})
    with pytest.raises(TypeError):
        m._items["k"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        m._items = {"k": 1}  # type: ignore[assignment]
    assert m["k"] == 0 and pickle.loads(pickle.dumps(m)) == m and copy.deepcopy(m) == m


@pytest.mark.contract
@settings(max_examples=200, deadline=None)
@given(n=st.integers())
def test_value_digest_tells_ints_from_strings(n: int) -> None:
    """An int, its decimal text and a mapping spelling the out-of-range marker digest differently."""
    forms = [n, str(n), {"$int": str(n)}, {"$$int": str(n)}]
    digests = [value_digest("t", {"x": f}) for f in forms]
    assert len(set(digests)) == len(forms)


@pytest.mark.contract
def test_int_marker_cannot_be_forged_by_a_mapping() -> None:
    """A held mapping with a `$`-key never encodes as the marker an out-of-range int encodes as."""
    big = 2**60
    assert value_digest("t", {"x": {"$int": str(big)}}) != value_digest("t", {"x": big})
    assert value_digest("t", {"$x": 1}) != value_digest("t", {"x": 1})
    assert value_digest("t", {"$x": 1}) != value_digest("t", {"$$x": 1})


class Holder(Frozen):
    """A value with one field typed `Any`, where field types no longer fix the family."""

    x: Any


class Inner(Frozen):
    """A value held inside another."""

    a: int


class Colour(Enum):
    """An enum whose value is a string."""

    RED = "red"


def _dag(levels: int) -> list[object]:
    x: list[object] = []
    for _ in range(levels):
        x = [x, x]
    return x


def _nested(depth: int) -> list[object]:
    x: list[object] = []
    for _ in range(depth - 1):
        x = [x]
    return x


@pytest.mark.logic
def test_an_array_and_its_sha256_hex_string_differ() -> None:
    """`Holder(x=a)` and `Holder(x=<sha256 of a's canonical .npy>)` neither compare equal nor share a digest."""
    from senselab_worker.canonical import canonical_npy_bytes

    a = numpy.arange(4, dtype="<f8")
    sha = hashlib.sha256(canonical_npy_bytes(a)).hexdigest()
    assert Holder(x=a) != Holder(x=sha)
    assert Holder(x=a).canonical_digest() != Holder(x=sha).canonical_digest()


def _family_samples() -> dict[str, tuple[object, list[object]]]:
    """Per token, a value of that family and the values a merged encoding would equate with it."""
    from senselab_worker.canonical import canonical_npy_bytes

    a = numpy.arange(4, dtype="<f8")
    sha = hashlib.sha256(canonical_npy_bytes(a)).hexdigest()
    inner = f"{Inner.__module__}.{Inner.__qualname__}"
    big = 2**60
    return {
        "$array": (a, [sha, {"$array": sha}, a.tolist()]),
        "$type": (int, ["builtins.int", {"$type": "builtins.int"}]),
        "$value": (Inner(a=1), [{"a": 1}, {"$value": {"type": inner, "fields": {"a": 1}}}]),
        "$int": (big, [str(big), float(big), {"$int": str(big)}]),
        "$f": (float("nan"), ["NaN", {"$f": "NaN"}]),
    }


@pytest.mark.contract
def test_every_value_family_has_its_own_encoding() -> None:
    """For every token family, a value of it differs from its payload and from a mapping spelling its token."""
    from senselab_worker.canonical import TOKENS

    samples = _family_samples()
    assert set(samples) == set(TOKENS), "a token family has no sample here"
    digests = {token: Holder(x=value).canonical_digest() for token, (value, _) in samples.items()}
    assert len(set(digests.values())) == len(digests)
    for token, (value, lookalikes) in samples.items():
        for other in lookalikes:
            assert Holder(x=value) != Holder(x=other), f"{token}: {other!r}"
            assert digests[token] != Holder(x=other).canonical_digest(), f"{token}: {other!r}"


@pytest.mark.logic
@pytest.mark.parametrize(
    "value",
    [frozenset({1}), {1}, PurePosixPath("a/b"), Colour.RED, b"\x01", bytearray(b"\x01"), Decimal("1.5")],
    ids=["frozenset", "set", "path", "enum", "bytes", "bytearray", "decimal"],
)
def test_sets_enums_paths_bytes_and_decimals_have_no_canonical_form(value: object) -> None:
    """A set, an enum member, a path, bytes or a `Decimal` held in a value is refused, not tokenised."""
    with pytest.raises(ValueError, match="has no canonical form"):
        Holder(x=value)


@pytest.mark.contract
@given(
    st.text(alphabet="ae\u0301\u00e9\u0323 ", max_size=4),
    st.sampled_from(["\U0001fae0", "\U0001fae8", "\U0001e08f", "\u0c5d", "\U000e0080"]),
)
def test_text_with_post_13_code_points_is_its_own_family(head: str, mark: str) -> None:
    """Its value digest differs from its NFC and from every token family."""
    text = head + mark
    digest = Holder(x=text).canonical_digest()
    others = {Holder(x=v).canonical_digest() for v, _ in _family_samples().values()}
    others.add(Holder(x=unicodedata.normalize("NFC", head)).canonical_digest())
    if unicodedata.normalize("NFC", text) != text:
        others.add(Holder(x=unicodedata.normalize("NFC", text)).canonical_digest())
    assert digest not in others


@pytest.mark.logic
@pytest.mark.parametrize("make", ["cycle", "dict-cycle", "deep"])
def test_a_value_that_contains_itself_or_nests_deeply_raises_the_shared_error(make: str) -> None:
    """A cycle or 5000-deep nesting gives the shared guard's `ValueError` when the value is built, which digests it."""
    value: object
    if make == "cycle":
        cyc: list[object] = []
        cyc.append(cyc)
        value = cyc
    elif make == "dict-cycle":
        d: dict[str, object] = {}
        d["self"] = [d]
        value = d
    else:
        value = _nested(5000)
    with pytest.raises(ValueError, match="contains itself|deeper than"):
        Holder(x=value)


@pytest.mark.contract
@given(st.lists(st.sampled_from(["list", "tuple", "dict"]), min_size=1, max_size=120))
def test_any_nesting_digests_or_raises_value_error(kinds: list[str]) -> None:
    """Every nesting of every container kind either digests or raises `ValueError`; none overflows the stack."""
    value: object = 1
    for kind in reversed(kinds):
        if kind == "list":
            value = [value]
        elif kind == "dict":
            value = {"k": value}
        else:
            value = (value,)
    try:
        Holder(x=value)
    except ValueError as err:
        assert "deeper than" in str(err)


@pytest.mark.logic
def test_audio_identity_is_one_per_value() -> None:
    """Audio whose samples differ only by `-0.0` has one `content:` identity; NaNs of any sign have one `.npy` digest.

    Audio refuses NaN samples, so the NaN half is checked on the array digest audio identity is made of.
    """
    from senselab_core.kernel.signals import Audio
    from senselab_worker.digest import npy_digest

    zero = numpy.array([0.0, 1.0], dtype=numpy.float32)
    negative = numpy.array([-0.0, 1.0], dtype=numpy.float32)
    nan = numpy.array([0x7FC00000, 0], dtype="<u4").view("<f4")
    negative_nan = numpy.array([0xFFC00001, 0], dtype="<u4").view("<f4")
    assert Audio.from_array(zero, 16_000).identity == Audio.from_array(negative, 16_000).identity
    assert npy_digest("npy@1", nan) == npy_digest("npy@1", negative_nan)


def _refuse_dag(where: str) -> None:
    from senselab_core.kernel import prov
    from senselab_core.runtime.store import output_identity
    from senselab_core.runtime.vocabulary import core_vocabulary

    dag = _dag(60)
    with pytest.raises(ValueError, match="exceeds"):
        if where == "provenance":
            prov.activity(core_vocabulary(), attributes={"senselab:logicalKey": dag})
        elif where == "store":
            output_identity({"x": dag})
        else:
            Holder(x=dag).canonical_digest()


def _record_dag(levels: int) -> None:
    Holder(x=_dag(levels)).canonical_digest()


@pytest.mark.logic
@pytest.mark.parametrize("where", ["provenance", "store", "value"])
def test_a_shared_dag_is_refused_on_every_canonical_path(where: str) -> None:
    """Sixty levels of `x = [x, x]` are refused by provenance, the store and `value@2`."""
    _refuse_dag(where)


@pytest.mark.logic
def test_a_shared_dag_under_the_budget_is_walked_once_per_node() -> None:
    """Twenty-two levels (4M leaves expanded) digest."""
    _record_dag(22)


@pytest.mark.limits
@pytest.mark.parametrize("where", ["provenance", "store", "value"])
def test_a_shared_dag_is_refused_in_bounded_time(where: str) -> None:
    """Sixty levels of `x = [x, x]` are refused in under 5 s on every canonical path."""
    started = time.monotonic()
    _refuse_dag(where)
    assert time.monotonic() - started < 5.0


@pytest.mark.limits
def test_a_shared_dag_under_the_budget_is_walked_in_bounded_time() -> None:
    """Twenty-two levels (4M leaves expanded) digest in under 5 s."""
    started = time.monotonic()
    _record_dag(22)
    assert time.monotonic() - started < 5.0


RATE = 8_000


def _wav(path: Path, seed: int = 0, frames: int = RATE) -> Path:
    samples = numpy.random.default_rng(seed).integers(-2000, 2000, size=frames, dtype=numpy.int16)
    soundfile.write(path, samples, RATE, subtype="PCM_16")
    return path


def _key(**params: object) -> LogicalKey:
    return LogicalKey(
        task="t",
        task_version="1",
        backend="b",
        backend_version="1",
        model=None,
        params=params,
        backend_params={},
        inputs={},
    )


@pytest.mark.logic
def test_attributes_cannot_forge_an_identity_or_a_stamp(tmp_path: Path) -> None:
    """`from_attributes=True` is refused, so neither a forged identity nor a forged stamp is accepted."""
    audio = Audio.from_array(numpy.zeros(16, dtype=numpy.float32), RATE)
    forged = types.SimpleNamespace(groups=dict(audio.groups), anchor=audio.anchor, identity="content:" + "0" * 64)
    for cls in (Audio, Recording):
        with pytest.raises(ValidationError):
            cls.model_validate(forged, from_attributes=True)
    path = _wav(tmp_path / "a.wav")
    group = open_audio(path).groups["audio"]
    assert isinstance(group, RegularGroup) and group.source_stamp is not None
    stamp = group.source_stamp.model_copy(update={"frames": 10**9})
    held = types.SimpleNamespace(
        channels=group.channels, sampling_rate=RATE, data=None, source=path, source_stamp=stamp
    )
    with pytest.raises(ValidationError):
        RegularGroup.model_validate(held, from_attributes=True)


@pytest.mark.logic
def test_a_same_size_same_mtime_rewrite_is_stale(tmp_path: Path) -> None:
    """A same-size, same-mtime, same-inode rewrite is refused on read, and by a check with or without a memo entry."""
    path = _wav(tmp_path / "a.wav")
    audio = open_audio(path)
    group = audio.groups["audio"]
    assert isinstance(group, RegularGroup)
    st = os.stat(path)
    with soundfile.SoundFile(path, "r+") as f:
        f.seek(10)
        f.write(numpy.full(100, 0.25, dtype=numpy.float32))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (st.st_size, st.st_mtime_ns, st.st_ino)
    with pytest.raises(StaleSource):
        group.read(reader=READER)
    assert group.source_stamp is not None
    copy.deepcopy(audio)
    for _ in range(2):
        with pytest.raises(StaleSource):
            audio_io.check(group.source_stamp, path)
        audio_io.VERIFIED.clear()


@pytest.mark.logic
def test_a_frozen_mapping_is_initialised_once() -> None:
    """Re-calling `__init__` on a held mapping raises and leaves the value and its digest as they were."""
    key = _key(a=1)
    before = key.digest()
    for mapping in (key.params, key.inputs):
        assert isinstance(mapping, FrozenMapping)
        with pytest.raises(TypeError):
            mapping.__init__({"a": 5})  # type: ignore[misc]
    assert key.params == {"a": 1} and key.digest() == before == LogicalKey.model_validate(key.model_dump()).digest()


@pytest.mark.logic
def test_config_and_private_state_cannot_change_a_value() -> None:
    """Flipping `model_config`, writing a private attribute and `__setstate__` all raise."""
    span = Span(start_ns=0, end_ns=10, clock_id="c")
    other = Span(start_ns=0, end_ns=20, clock_id="c")
    with pytest.raises(TypeError):
        Span.model_config["frozen"] = False
    with pytest.raises(ValidationError):
        span.end_ns = -100  # type: ignore[misc]
    with pytest.raises(ValidationError):
        span._canonical = other.canonical_digest()
    with pytest.raises(TypeError):
        span.__pydantic_private__["_canonical"] = other.canonical_digest()  # type: ignore[index]
    with pytest.raises(TypeError):
        span.__setstate__({"__dict__": {"start_ns": 0, "end_ns": -1, "clock_id": "c"}})
    assert span.end_ns == 10 and span != other and hash(span) != hash(other)
    key = _key()
    with pytest.raises(TypeError):
        key.__pydantic_private__["_digest"] = "f" * 64  # type: ignore[index]
    blank = object.__new__(Span)
    with pytest.raises(ValidationError):
        blank.__setstate__({"__dict__": {"start_ns": 5, "end_ns": 1, "clock_id": "c"}})


@pytest.mark.logic
def test_deep_or_cyclic_params_raise_a_validation_error() -> None:
    """Params nested 5000 deep, or holding themselves, raise `ValidationError`, not `RecursionError`."""
    deep: object = 1
    for _ in range(5000):
        deep = [deep]
    loop: dict[str, object] = {}
    loop["self"] = loop
    for params in ({"a": deep}, loop):
        with pytest.raises(ValidationError):
            _key(**params)


@pytest.mark.logic
def test_labels_equal_by_value_are_one_value() -> None:
    """Label arrays of one set of strings are equal whatever width they were built at."""
    ns = numpy.array([0])
    by_width = [
        Intervals(start_ns=ns, end_ns=ns + 1, label=numpy.array(["a"], dtype=dtype), clock_id="c")
        for dtype in ("U1", "U3", "U9")
    ]
    assert by_width[0] == by_width[1] == by_width[2]
    assert Events(time_ns=ns, label=numpy.array(["a\x00"]), value=None, clock_id="c") == Events(
        time_ns=ns, label=numpy.array(["a"]), value=None, clock_id="c"
    )


@pytest.mark.logic
def test_small_loosenesses_are_refused() -> None:
    """Non-finite audio, lax floats, unknown fields, int64 overflow, unparseable refs, week dates, open classes."""
    for samples in ([0.0, float("nan")], [0.0, float("inf")], numpy.array([0.0, 1e300])):
        with pytest.raises(ValueError):
            Audio.from_array(numpy.asarray(samples), RATE)
    for drift in ("5", True):
        with pytest.raises(ValidationError):
            TimeMap(source="a", target="b", offset_ns=0, drift_ppm=drift)
    with pytest.raises(ValidationError):
        Span.model_validate({"start_ns": 0, "end_ns": 1, "clock_id": "c", "extra": 1})
    with pytest.raises(OverflowError):
        TimeMap(source="a", target="b", offset_ns=2**63 - 1).map(10)
    with pytest.raises(ValidationError):
        ModelRef(source="url", id="http://h/x#sha256=", ref="")
    with pytest.raises(ValidationError):
        License(
            expression="MIT",
            commercial_use="yes",
            copyleft="none",
            attribution=True,
            gated=False,
            source="s",
            verified={"date": "2026-W01-1", "how": "manual"},
        )
    with pytest.raises(TypeError):
        NO_EQUIVALENCE["b"] = {"c": frozenset()}  # type: ignore[index]
    context = ExecutionContext(
        env_lock="l",
        platform="p",
        wheel_set="w",
        numerics={},
        decoder=None,
        driver={},
        protocol="1",
        model_source=None,
        env_license=None,
    )
    assert acceptable(context, context, "equivalent", backend="b")


@pytest.mark.logic
def test_a_nested_unsealed_or_subclass_instance_is_validated_again(tmp_path: Path) -> None:
    """A subclass with other validators cannot carry a forged identity; a forged stamp is refused by the reader."""
    from pydantic import BaseModel

    path = _wav(tmp_path / "a.wav")
    real = open_audio(path)
    group = real.groups["audio"]
    assert isinstance(group, RegularGroup) and group.source_stamp is not None
    forged_stamp = BaseModel.model_construct.__func__(  # type: ignore[attr-defined]
        type(group.source_stamp), **{**dict(group.source_stamp.__dict__), "content": "0" * 64}
    )
    planted = BaseModel.model_construct.__func__(  # type: ignore[attr-defined]
        RegularGroup, **{**dict(group.__dict__), "source_stamp": forged_stamp}
    )
    forged = Audio(groups={"audio": planted}, anchor=real.anchor)
    assert forged.identity == "content:" + identity_digest(
        "audio-file@1", {"content": "0" * 64, "channels": [c.model_dump() for c in group.channels]}
    )
    forged_group = forged.groups["audio"]
    assert isinstance(forged_group, RegularGroup) and forged_group.source_stamp is not None
    with pytest.raises(StaleSource):
        forged_group.read(reader=READER)
    with pytest.raises(StaleSource):
        source_group(forged_group.channels, RATE, path, forged_group.source_stamp)

    class EvilAudio(Audio):
        @classmethod
        def identity_of(cls, groups: object) -> str:
            return "content:" + "0" * 64

    evil = EvilAudio(groups=dict(real.groups), anchor=real.anchor)  # type: ignore[call-arg]
    assert evil.identity == "content:" + "0" * 64
    with pytest.raises(ValidationError):
        Audio.model_validate(evil)
    unsealed = BaseModel.model_construct.__func__(Span, start_ns=5, end_ns=1, clock_id="c")  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        Span.model_validate(unsealed)


@pytest.mark.logic
def test_audio_refuses_nan_of_either_sign_and_events_hold_one_nan() -> None:
    """Audio refuses NaN whatever its sign or payload; an event value holding NaN has one digest across them."""
    nan = numpy.array([0x7FC00000, 0], dtype="<u4").view("<f4")
    negative_nan = numpy.array([0xFFC00001, 0], dtype="<u4").view("<f4")
    for samples in (nan, negative_nan):
        with pytest.raises(ValueError, match="NaN"):
            Audio.from_array(samples, RATE)
    ns = numpy.array([0, 1])
    a, b = (Events(time_ns=ns, label=numpy.array(["a", "b"]), value=v, clock_id="c") for v in (nan, negative_nan))
    assert a == b and a.canonical_digest() == b.canonical_digest()


@pytest.mark.logic
def test_a_float_wav_holding_nan_is_refused_on_decode(tmp_path: Path) -> None:
    """A file-backed audio's samples are checked as decoded, as held samples are when built."""
    path = tmp_path / "nan.wav"
    samples = numpy.zeros(64, dtype=numpy.float32)
    samples[10] = numpy.nan
    soundfile.write(path, samples, RATE, subtype="FLOAT")
    group = open_audio(path).groups["audio"]
    assert isinstance(group, RegularGroup)
    with pytest.raises(ValueError, match="NaN"):
        group.read(reader=READER)
    assert group.read(0, 10 * 10**9 // RATE, reader=READER).shape == (1, 10)


@pytest.mark.logic
def test_an_irregular_group_holds_nan_as_the_one_missing_marker() -> None:
    """NaN of any sign or payload in an irregular group is held as the quiet NaN without payload; inf is refused."""
    from senselab_core.kernel.signals import ChannelInfo, IrregularGroup

    channel = (ChannelInfo(name="x", type="pose", units="px"),)
    nans = numpy.array([[1.0, 0.0, 0.0]], dtype=numpy.float64)
    nans.view("<u8")[0, 1] = 0xFFF8000000000001
    nans.view("<u8")[0, 2] = 0x7FF8000000000000
    group = IrregularGroup(channels=channel, timestamps_ns=numpy.array([0, 5, 9]), data=nans)
    bits = group.data.view("<u8")[0]
    assert bits[1] == bits[2] == 0x7FF8000000000000
    with pytest.raises(ValidationError):
        IrregularGroup(channels=channel, timestamps_ns=numpy.array([0, 5]), data=numpy.array([[1.0, numpy.inf]]))
