"""Canonical JSON (RFC 8785) and canonical .npy bytes."""

import hashlib
import io
import json
import struct
import unicodedata
import warnings
from enum import Enum
from typing import Any

import numpy
import pytest
from hypothesis import given
from hypothesis import strategies as st
from senselab_worker.canonical import (
    canonical_npy_bytes,
    canonical_scalar,
    canonical_text,
    jcs_dumps,
    jcs_loads,
)
from senselab_worker.digest import IDENTITY_TAGS, identity_digest


@pytest.mark.logic
def test_jcs_rfc8785_vectors() -> None:
    """Keys sort, whitespace is dropped and numbers follow ECMAScript."""
    assert jcs_dumps({"b": 2, "a": [1, "x"]}) == b'{"a":[1,"x"],"b":2}'
    assert jcs_dumps(1e21) == b"1e+21" and jcs_dumps(0.0) == b"0" and jcs_dumps(-0.0) == b"0"


RFC8785_NUMBERS = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


@pytest.mark.logic
@pytest.mark.parametrize(("bits", "expected"), RFC8785_NUMBERS)
def test_jcs_rfc8785_number_vectors(bits: str, expected: str) -> None:
    """Every IEEE-754 vector of RFC 8785 Appendix B serialises as specified."""
    (value,) = struct.unpack(">d", bytes.fromhex(bits))
    assert jcs_dumps(value) == expected.encode()


@pytest.mark.logic
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_jcs_edge_values(bad: float) -> None:
    """Non-finite floats are refused."""
    with pytest.raises(ValueError):
        jcs_dumps({"x": bad})


@pytest.mark.logic
@pytest.mark.parametrize("bad", [numpy.float32("nan"), numpy.float64("-inf"), [1.0, float("-inf")]])
def test_jcs_refuses_non_finite_numpy_scalars_and_nested(bad: object) -> None:
    """Non-finite values are refused after numpy conversion and inside containers."""
    with pytest.raises(ValueError):
        jcs_dumps(bad)


@pytest.mark.logic
def test_jcs_normalises_strings_to_nfc() -> None:
    """Composed and decomposed forms of a string serialise identically."""
    assert jcs_dumps("é") == jcs_dumps("é") == '"é"'.encode()
    assert jcs_dumps({"é": 1}) == jcs_dumps({"é": 1})


@pytest.mark.logic
def test_jcs_refuses_keys_equal_after_nfc() -> None:
    """Two keys that collide under NFC are refused."""
    with pytest.raises(ValueError):
        jcs_dumps({"é": 1, "é": 2})


@pytest.mark.logic
def test_jcs_sorts_keys_by_utf16_code_units() -> None:
    """Keys sort by UTF-16 code units, so an astral character sorts below U+FB01 (RFC 8785 §3.2.3)."""
    obj = {"€": 1, "\r": 2, "ﬁ": 3, "1": 4, "\U0001f600": 5, "\u0080": 6, "ö": 7}
    keys = list(json.loads(jcs_dumps(obj)))
    assert keys == ["\r", "1", "\u0080", "ö", "€", "\U0001f600", "ﬁ"]


@pytest.mark.logic
def test_jcs_string_escapes() -> None:
    """Only quote, backslash and C0 controls are escaped; short forms where JSON has them."""
    assert jcs_dumps('"\\\b\f\n\r\t\x01\x1f\x7f€') == b'"\\"\\\\\\b\\f\\n\\r\\t\\u0001\\u001f\x7f\xe2\x82\xac"'


@pytest.mark.logic
def test_jcs_literals_and_containers() -> None:
    """Literals, tuples and nested containers serialise without whitespace."""
    assert jcs_dumps([None, True, False, (1, 2), {}, []]) == b"[null,true,false,[1,2],{},[]]"


@pytest.mark.logic
def test_jcs_converts_numpy_scalars() -> None:
    """Numpy scalars serialise as their Python value."""
    assert jcs_dumps({"a": numpy.int64(3), "b": numpy.bool_(True), "c": numpy.float32(0.5)}) == (
        b'{"a":3,"b":true,"c":0.5}'
    )


@pytest.mark.logic
def test_jcs_integer_range_is_i_json() -> None:
    """Integers within +/-(2**53 - 1) serialise; larger ones are refused."""
    assert jcs_dumps(2**53 - 1) == b"9007199254740991" and jcs_dumps(-(2**53) + 1) == b"-9007199254740991"
    for bad in (2**53, -(2**53), 2**53 + 1, numpy.int64(2**53), numpy.uint64(2**64 - 1)):
        with pytest.raises(ValueError):
            jcs_dumps(bad)


@pytest.mark.logic
@pytest.mark.parametrize("bad", [{1: "x"}, object(), numpy.zeros(2), b"bytes"])
def test_jcs_refuses_non_json_types(bad: object) -> None:
    """Non-string keys, arrays, bytes and arbitrary objects are refused."""
    with pytest.raises(TypeError):
        jcs_dumps(bad)


@pytest.mark.logic
def test_jcs_refuses_lone_surrogates() -> None:
    """A string that is not valid Unicode is refused."""
    with pytest.raises(ValueError):
        jcs_dumps("\ud800")


_NFC_TEXT = st.text().map(canonical_text)
_I_JSON_INTS = st.integers(min_value=-(2**53) + 1, max_value=2**53 - 1)
_FINITE_FLOATS = st.floats(allow_nan=False, allow_infinity=False).filter(
    lambda x: not (x.is_integer() and abs(x) >= 2**53)
)
_JSON = st.recursive(
    st.none() | st.booleans() | _I_JSON_INTS | _FINITE_FLOATS | _NFC_TEXT,
    lambda children: st.lists(children) | st.dictionaries(_NFC_TEXT, children),
    max_leaves=20,
)


@pytest.mark.logic
@given(_JSON)
def test_jcs_is_a_fixed_point_of_parse(x: object) -> None:
    """Parsing canonical JSON and serialising it again gives the same bytes."""
    assert jcs_dumps(json.loads(jcs_dumps(x))) == jcs_dumps(x)


@pytest.mark.logic
def test_canonical_npy_is_little_endian_c_order() -> None:
    """A big-endian, transposed array is written little-endian, C order, version 1.0."""
    a = numpy.arange(6, dtype=">f4").reshape(2, 3).T
    b = canonical_npy_bytes(a)
    assert numpy.load(io.BytesIO(b)).dtype.str == "<f4" and b[6:8] == b"\x01\x00"


@pytest.mark.logic
@pytest.mark.parametrize("dtype", ["<f4", ">f8", "|u1", "?", ">i2", "<U3"])
def test_canonical_npy_round_trips_and_ignores_layout(dtype: str) -> None:
    """Equal arrays give equal bytes whatever their byte order or memory layout."""
    base = numpy.arange(12).reshape(3, 4).astype(dtype)
    variants = [base, numpy.asfortranarray(base), base.astype(base.dtype.newbyteorder(">")), base.T.copy().T]
    blobs = {canonical_npy_bytes(v) for v in variants}
    assert len(blobs) == 1
    (blob,) = blobs
    loaded = numpy.load(io.BytesIO(blob))
    assert loaded.dtype.str[0] in "<|" and loaded.flags.c_contiguous
    numpy.testing.assert_array_equal(loaded, base)


@pytest.mark.logic
def test_canonical_npy_header_is_fixed() -> None:
    """The header is the version 1.0 dict, space padded to a 64-byte boundary."""
    blob = canonical_npy_bytes(numpy.zeros((2, 3), dtype="<f4"))
    (header_len,) = struct.unpack("<H", blob[8:10])
    header = blob[10 : 10 + header_len]
    assert blob[:6] == b"\x93NUMPY" and (10 + header_len) % 64 == 0
    assert header.rstrip(b" \n") == b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"
    assert header.endswith(b"\n") and len(blob) == 10 + header_len + 24


@pytest.mark.logic
@pytest.mark.parametrize("shape", [(), (0,), (5,), (1, 0, 3)])
def test_canonical_npy_edge_shapes(shape: tuple[int, ...]) -> None:
    """Scalar, empty and one-dimensional arrays round-trip."""
    a = numpy.ones(shape, dtype="<i4")
    loaded = numpy.load(io.BytesIO(canonical_npy_bytes(a)))
    assert loaded.shape == shape
    numpy.testing.assert_array_equal(loaded, a)


@pytest.mark.logic
@pytest.mark.parametrize("dtype", [object, [("a", "<f4"), ("b", "<i2")]])
def test_canonical_npy_refuses_object_and_structured(dtype: object) -> None:
    """Object and structured dtypes are refused."""
    with pytest.raises(ValueError):
        canonical_npy_bytes(numpy.zeros(2, dtype=dtype))


@pytest.mark.logic
@pytest.mark.parametrize("dtype", [numpy.longdouble, numpy.clongdouble])
def test_canonical_npy_refuses_platform_specific_dtypes(dtype: type) -> None:
    """Extended-precision dtypes, whose layout varies by platform, are refused."""
    with pytest.raises(ValueError):
        canonical_npy_bytes(numpy.zeros(2, dtype=dtype))


# The exact-type whitelist


def _numpy_scalar_types() -> list[type]:
    """Every subclass of `numpy.generic`, abstract ones included, found by walking the hierarchy."""
    found: list[type] = []
    stack: list[type] = [numpy.generic]
    while stack:
        t = stack.pop()
        if t not in found:
            found.append(t)
            stack.extend(t.__subclasses__())
    return found


_SAMPLE_SOURCES: list[object] = [
    0,
    1,
    -3,
    0.5,
    -2.25,
    1e20,
    True,
    "a",
    "é",
    b"ab",
    1 + 2j,
    numpy.datetime64("NaT"),
    numpy.datetime64("2020-01-01"),
    numpy.datetime64(1, "ns"),
    numpy.timedelta64("NaT"),
    numpy.timedelta64(1, "ns"),
    numpy.timedelta64(1, "ps"),
    numpy.zeros(1, dtype=[("a", "<i4")])[0],
    numpy.void(b"ab"),
    numpy.clongdouble(1 + 2j),
    numpy.longdouble(0.5),
]


def _samples_of(t: type) -> list[object]:
    """Instances of exactly `t`: each source converted by `t`, plus the sources already of type `t`."""
    out: list[object] = [v for v in _SAMPLE_SOURCES if type(v) is t]
    with warnings.catch_warnings(), numpy.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for source in _SAMPLE_SOURCES:
            try:
                v = t(source)
            except (TypeError, ValueError, OverflowError):
                continue
            if type(v) is t:
                out.append(v)
    return out


@pytest.mark.logic
def test_every_numpy_generic_type_raises_or_encodes_exactly() -> None:
    """Every numpy scalar type is refused, or encodes as exactly the Python value `.item()` returns."""
    concrete = 0
    for t in _numpy_scalar_types():
        samples = _samples_of(t)
        concrete += bool(samples)
        for v in samples:
            try:
                b = jcs_dumps(v)
            except (TypeError, ValueError):
                continue
            assert t.__module__ == "numpy" and numpy.dtype(t).kind in "biufU", t
            item = v.item()  # type: ignore[attr-defined]
            assert type(item) in (bool, int, float, str), t
            decoded: Any = jcs_loads(b)
            assert decoded == item and decoded == v, (t, v, b)
            assert type(decoded) is type(item) or (type(item) is float and float(decoded) == item), (t, v)
    assert concrete >= 20, f"only {concrete} numpy scalar types had samples; the walk is not covering numpy"


class _IntEnum(int, Enum):
    A = 1


class _StrEnum(str, Enum):
    A = "a"


class _Int(int):
    pass


class _Str(str):
    pass


class _List(list):  # type: ignore[type-arg]
    pass


class _Dict(dict):  # type: ignore[type-arg]
    pass


class _Int64(numpy.int64):
    pass


_LOSSY = [
    numpy.datetime64("NaT"),
    numpy.datetime64("2020-01-01"),
    numpy.timedelta64(1, "ns"),
    numpy.timedelta64(1, "ps"),
    numpy.ma.array([1, 2], mask=[0, 1]),
    _IntEnum.A,
    _StrEnum.A,
    _Int(1),
    _Str("a"),
    _List([1]),
    _Dict(a=1),
    _Int64(1),
    numpy.zeros(1, dtype=[("a", "<i4")])[0],
    numpy.void(b"ab"),
    numpy.complex128(1 + 2j),
    numpy.longdouble(0.5),
    numpy.bytes_(b"a"),
    1 + 2j,
    b"bytes",
]


@pytest.mark.logic
@pytest.mark.parametrize("v", _LOSSY, ids=lambda v: type(v).__qualname__)
def test_a4_lossy_values_refused(v: object) -> None:
    """A value whose JSON form would lose its type or unit is refused, at top level and nested."""
    for wrapped in (v, {"a": v}, [v], ({"b": [v]},)):
        with pytest.raises(TypeError):
            jcs_dumps(wrapped)
    with pytest.raises(TypeError):
        canonical_scalar(v)


@pytest.mark.logic
def test_a4_nat_and_none_never_share_bytes() -> None:
    """The probe: NaT encoded as null, and timedelta64 of any unit as its integer."""
    assert jcs_dumps({"a": None}) == b'{"a":null}'
    for v in (numpy.datetime64("NaT"), numpy.timedelta64(1, "ns"), numpy.timedelta64(1, "ps")):
        with pytest.raises(TypeError):
            jcs_dumps({"a": v})


@pytest.mark.logic
@pytest.mark.parametrize("key", [_StrEnum.A, _Str("a"), numpy.str_("a")], ids=lambda k: type(k).__qualname__)
def test_jcs_object_keys_are_exactly_str(key: object) -> None:
    """An object key must be exactly `str`; subclasses and numpy strings are refused."""
    with pytest.raises(TypeError):
        jcs_dumps({key: 1})


@pytest.mark.logic
def test_jcs_refuses_cycles() -> None:
    """A container that contains itself raises `ValueError` instead of recursing; shared children are fine."""
    a: list[object] = [1]
    a.append(a)
    d: dict[str, object] = {}
    d["self"] = d
    for cyclic in (a, d, [{"x": d}]):
        with pytest.raises(ValueError, match="contains itself"):
            jcs_dumps(cyclic)
    shared = [1, 2]
    assert jcs_dumps([shared, shared]) == b"[[1,2],[1,2]]"


@pytest.mark.logic
@pytest.mark.parametrize(
    "array",
    [
        numpy.ma.array([1, 2, 3], mask=[0, 0, 1]),
        numpy.arange(3).view(numpy.matrix),
        numpy.zeros(2, dtype="<c16"),
        numpy.zeros(2, dtype="<M8[ns]"),
        numpy.zeros(2, dtype="<m8[ps]"),
        numpy.zeros(2, dtype="|S2"),
        numpy.zeros(2, dtype="|V4"),
    ],
    ids=["masked", "matrix", "complex", "datetime", "timedelta", "bytes", "void"],
)
def test_a4_canonical_npy_refuses_subclasses_and_lossy_kinds(array: numpy.ndarray) -> None:
    """Only a plain `ndarray` of kind `b i u f U` has a canonical `.npy`; a masked array never encodes its fill."""
    with pytest.raises((TypeError, ValueError)):
        canonical_npy_bytes(array)


_ANY_FLOAT = st.floats(allow_nan=False, allow_infinity=False)
_ACCEPTED = st.recursive(
    st.none() | st.booleans() | _I_JSON_INTS | _ANY_FLOAT | _NFC_TEXT,
    lambda children: st.lists(children) | st.tuples(children, children) | st.dictionaries(_NFC_TEXT, children),
    max_leaves=20,
)


@pytest.mark.logic
@given(_ACCEPTED)
def test_jcs_round_trips_every_accepted_value(v: object) -> None:
    """Every value `jcs_dumps` accepts, integral floats beyond 2**53 included, survives `jcs_loads` byte for byte."""
    assert jcs_dumps(jcs_loads(jcs_dumps(v))) == jcs_dumps(v)


@pytest.mark.logic
def test_a24_large_integral_float_parses_back_as_a_float() -> None:
    """The probe: `1e20` serialises as an integer literal, which `jcs_loads` reads back as the float it was."""
    assert jcs_dumps(1e20) == b"100000000000000000000"
    back = jcs_loads(b"100000000000000000000")
    assert type(back) is float and back == 1e20
    assert type(jcs_loads(b"9007199254740991")) is int


@pytest.mark.logic
def test_digest_tags_distinct_and_prefix_free() -> None:
    """No registered tag is a prefix of another, and each is ASCII with a `@<major>`."""
    tags = sorted(IDENTITY_TAGS)
    assert all(not b.startswith(a) for a in tags for b in tags if a != b)
    assert all(t.isascii() and "\x00" not in t and "@" in t for t in tags)


@pytest.mark.logic
def test_tagged_digest_separates_domains() -> None:
    """One value under every tag gives as many digests, none of them the bare sha256 of its JCS."""
    value = {"a": [1, "x"]}
    digests = {identity_digest(t, value) for t in IDENTITY_TAGS}
    assert len(digests) == len(IDENTITY_TAGS)
    assert hashlib.sha256(jcs_dumps(value)).hexdigest() not in digests
    for tag in IDENTITY_TAGS:
        assert identity_digest(tag, value) == hashlib.sha256(tag.encode() + b"\x00" + jcs_dumps(value)).hexdigest()


@pytest.mark.logic
@pytest.mark.parametrize(
    "tag",
    [
        "",
        "output",
        "output@2",
        "audio@2",
        "value@1",
        "logical-key@2",
        "context@2",
        "weights@2",
        "provenance-id@2",
        "data-manifest@2",
        "logical-key@3\x00",
        "nope@1",
    ],
)
def test_tagged_digest_refuses_unregistered_tags(tag: str) -> None:
    """Only a registered tag may name a digest domain."""
    with pytest.raises(ValueError, match="unregistered"):
        identity_digest(tag, 1)
