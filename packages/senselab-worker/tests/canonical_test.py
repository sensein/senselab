"""Canonical JSON (RFC 8785) and canonical .npy bytes."""

import hashlib
import io
import json
import math
import struct
import time
import warnings
from enum import Enum
from typing import Any, Literal, cast

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import DTypeLike
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
def test_canonical_npy_refuses_object_and_structured(dtype: DTypeLike) -> None:
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
    numpy.ma.array([1, 2], mask=[False, True]),
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
def test_lossy_values_refused(v: object) -> None:
    """A value whose JSON form would lose its type or unit is refused, at top level and nested."""
    for wrapped in (v, {"a": v}, [v], ({"b": [v]},)):
        with pytest.raises(TypeError):
            jcs_dumps(wrapped)
    with pytest.raises(TypeError):
        canonical_scalar(v)


@pytest.mark.logic
def test_nat_and_none_never_share_bytes() -> None:
    """NaT is not encoded as null, and timedelta64 of any unit is not encoded as its integer."""
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
        numpy.ma.array([1, 2, 3], mask=[False, False, True]),
        numpy.arange(3).view(numpy.matrix),
        numpy.zeros(2, dtype="<c16"),
        numpy.zeros(2, dtype="<M8[ns]"),
        numpy.zeros(2, dtype="<m8[ps]"),
        numpy.zeros(2, dtype="|S2"),
        numpy.zeros(2, dtype="|V4"),
    ],
    ids=["masked", "matrix", "complex", "datetime", "timedelta", "bytes", "void"],
)
def test_canonical_npy_refuses_subclasses_and_lossy_kinds(array: numpy.ndarray) -> None:
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
def test_large_integral_float_parses_back_as_a_float() -> None:
    """`1e20` serialises as an integer literal, which `jcs_loads` reads back as the float it was."""
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
        "logical-key@4",
        "logical-key@5",
        "context@2",
        "context@3",
        "weights@2",
        "provenance-id@2",
        "data-manifest@2",
        "logical-key@6\x00",
        "nope@1",
    ],
)
def test_tagged_digest_refuses_unregistered_tags(tag: str) -> None:
    """Only a registered tag may name a digest domain."""
    with pytest.raises(ValueError, match="unregistered"):
        identity_digest(tag, 1)


_UNASSIGNED_IN_13 = "a\U0001e08f\u0323"
"""U+1E08F is a combining mark (ccc 230) in Unicode 15 and unassigned in 13, so NFC reorders it on 3.12 only."""


POST_13 = (
    "\U0001e08f",  # 15.0 combining mark
    "\U0001fae0",  # 14.0 melting face
    "\U0001faf6",  # 14.0 heart hands
    "\U0001fae8",  # 15.0 shaking face
    "\U0001fa75",  # 15.0 light blue heart
    "\u0c5d",  # 14.0 Telugu letter
    "\U00011f00",  # 15.0 Kawi sign
    "\U000e0080",  # unassigned in 15.0 too
)
"""Code points unassigned in Unicode 13.0, emoji and marks from 14.0 and 15.0 among them."""


PINNED = {
    _UNASSIGNED_IN_13: "91027814957edb1fb697c0c3f640e2f811ec745c71f3d114d4e0231619d3d40b",
    "\U0001fae0 melting": "66c5c6daf2b9da68459649adb38a6b67642ff18c3fa3d15a88bbb30e6fbb2e76",
    "e\u0301\U0001faf6": "4ce6da3badb79e2bd73d81bb0fb9ef61ba59dfe39d315baca9fad6635758defa",
    "\U0001fae8\u0301\u0323": "9a94f041798ca0909bcb10dff8d4c76164ba0e89daf474dc40b490a62f4e836c",
    "\u0c5d\u0c55": "35caf7985ee2423e87ef2a97e8cce07c10384f78e3df05d44bf94de4ea2c4193",
}
"""sha256 of `jcs_dumps({"k": [s], s: s})` per string, computed under Python 3.10 (Unicode 13.0)."""


def _jcs_sha(text: str) -> str:
    import hashlib

    from senselab_worker.canonical import jcs_dumps

    return hashlib.sha256(jcs_dumps({"k": [text], text: text})).hexdigest()


NFC_SENSITIVE = (_UNASSIGNED_IN_13, "e\u0301\U0001faf6", "\U0001fae8\u0301\u0323")
"""Pinned strings an interpreter's NFC would change."""


@pytest.mark.logic
@pytest.mark.parametrize("text", NFC_SENSITIVE, ids=ascii)
def test_text_with_post_13_code_points_has_one_digest_on_every_interpreter(text: str) -> None:
    """A string holding a code point unassigned in Unicode 13 digests as 3.10 digested it, as a value and a key."""
    assert _jcs_sha(text) == PINNED[text]


@pytest.mark.contract
def test_every_pinned_post_13_digest_is_reproduced() -> None:
    """Every string pinned under Python 3.10 digests the same under the running interpreter."""
    assert {text: _jcs_sha(text) for text in PINNED} == PINNED


def _is_portable(text: str) -> bool:
    from senselab_worker.canonical import is_portable

    return is_portable(text)


def _portable_text() -> "st.SearchStrategy[str]":
    return st.text(alphabet=st.characters(codec="utf-8"), max_size=8).filter(_is_portable)


@pytest.mark.contract
@given(
    _portable_text(),
    st.lists(st.sampled_from(POST_13), min_size=1, max_size=3),
    st.integers(min_value=0, max_value=8),
    _portable_text(),
)
def test_text_with_post_13_code_points_is_kept_raw_and_never_collides(
    head: str, marks: list[str], at: int, other: str
) -> None:
    """Never refused, written without normalisation, and never equal to the form of text assigned in 13.0."""
    import json

    from senselab_worker.canonical import canonical_text, is_portable, jcs_dumps

    text = head[:at] + "".join(marks) + head[at:]
    assert not is_portable(text)
    assert canonical_text(text) == text
    assert jcs_dumps(text) == json.dumps(text, ensure_ascii=False).encode("utf-8")
    assert jcs_dumps({text: [text]}) == (
        "{" + json.dumps(text, ensure_ascii=False) + ":[" + json.dumps(text, ensure_ascii=False) + "]}"
    ).encode("utf-8")
    assert is_portable(canonical_text(other))
    assert jcs_dumps(text) != jcs_dumps(other)
    assert jcs_dumps(text) != jcs_dumps(canonical_text(head))


def _inf_minus_inf() -> numpy.ndarray[Any, Any]:
    with numpy.errstate(invalid="ignore"):
        diff: numpy.ndarray[Any, Any] = numpy.array([numpy.inf]) - numpy.array([numpy.inf])
        return diff


def _bits(dtype: str, words: list[int]) -> numpy.ndarray:
    size = numpy.dtype(dtype).itemsize
    return numpy.array(words, dtype=f"<u{size}").view(f"<f{size}")


@pytest.mark.logic
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (numpy.array([numpy.nan]), _bits("f8", [0xFFF8000000000000])),
        (numpy.array([numpy.nan]), _bits("f8", [0x7FF0000000000001])),
        (_inf_minus_inf(), _bits("f8", [0xFFF8000000000000])),
        (numpy.array([0.0]), numpy.array([-0.0])),
        (numpy.array([True]), numpy.frombuffer(bytes([2]), dtype=numpy.bool_)),
        (_bits("f4", [0x7FC00000]), _bits("f4", [0xFFC00001])),
        (_bits("f2", [0x7E00]), _bits("f2", [0xFE01])),
    ],
    ids=["nan-sign", "nan-payload", "inf-minus-inf", "negative-zero", "bool-byte-2", "f4-nan", "f2-nan"],
)
def test_values_that_are_one_value_have_one_npy_form(left: numpy.ndarray, right: numpy.ndarray) -> None:
    """Every NaN, `0.0` and `-0.0`, and every nonzero bool byte, write the same canonical `.npy` bytes."""
    from senselab_worker.canonical import canonical_npy_bytes

    assert canonical_npy_bytes(left) == canonical_npy_bytes(right)


def _quadratic_dag(levels: int) -> list[object]:
    x: list[object] = []
    for _ in range(levels):
        x = [x, x]
    return x


@pytest.mark.logic
def test_a_shared_dag_is_written_once_per_node() -> None:
    """Twenty-three levels of `x = [x, x]` (8M leaves expanded) encode, each node written once."""
    from senselab_worker.canonical import jcs_dumps

    text = jcs_dumps(_quadratic_dag(23))
    assert len(text) == 5 * 2**23 - 3 and text.startswith(b"[[[[")


@pytest.mark.logic
def test_a_shared_dag_beyond_the_budget_is_refused() -> None:
    """Sixty-three levels (2**63 leaves) are refused by the work budget, not expanded."""
    from senselab_worker.canonical import jcs_dumps

    with pytest.raises(ValueError, match="exceeds"):
        jcs_dumps(_quadratic_dag(63))


@pytest.mark.limits
def test_a_shared_dag_is_written_and_refused_in_bounded_time() -> None:
    """Twenty-three levels encode, and sixty-three are refused, each in under 5 s."""
    from senselab_worker.canonical import jcs_dumps

    started = time.monotonic()
    jcs_dumps(_quadratic_dag(23))
    assert time.monotonic() - started < 5.0
    started = time.monotonic()
    with pytest.raises(ValueError, match="exceeds"):
        jcs_dumps(_quadratic_dag(63))
    assert time.monotonic() - started < 5.0


@pytest.mark.logic
@pytest.mark.parametrize(
    "text",
    ["1e400", "-1e400", '{"a":1,"a":2}', "[" * 100_000, "-0.0", "[1, 2]", '{"b":1,"a":2}', "1E3", '"e\u0301"'],
    ids=["overflow", "neg-overflow", "duplicate-key", "deep", "neg-zero", "spaces", "unsorted", "exponent", "nfd"],
)
def test_jcs_loads_refuses_what_jcs_dumps_cannot_have_written(text: str) -> None:
    """Overflow, duplicate keys, over-deep nesting and every non-canonical spelling raise one typed error."""
    from senselab_worker.canonical import jcs_loads

    with pytest.raises(ValueError) as caught:
        jcs_loads(text)
    assert type(caught.value).__name__ == "NotCanonicalJSON"


@pytest.mark.logic
@given(st.text(max_size=40) | st.binary(max_size=40))
def test_jcs_loads_accepts_exactly_its_fixed_points(data: str | bytes) -> None:
    """Any input either parses to a value `jcs_dumps` writes back to the same bytes, or raises `NotCanonicalJSON`."""
    from senselab_worker.canonical import NotCanonicalJSON, jcs_dumps, jcs_loads

    raw = data.encode("utf-8", "surrogatepass") if isinstance(data, str) else data
    try:
        value = jcs_loads(data)
    except NotCanonicalJSON:
        return
    assert jcs_dumps(value) == raw


_WIDTH = {"f2": (2, "<e"), "f4": (4, "<f"), "f8": (8, "<d")}


def _reference_bits(dtype: str, word: int) -> int:
    """The canonical bits of one float word, by the ADR-006 list, without numpy."""
    size, fmt = _WIDTH[dtype]
    value = struct.unpack(fmt, word.to_bytes(size, "little"))[0]
    if math.isnan(value):
        return {2: 0x7E00, 4: 0x7FC00000, 8: 0x7FF8000000000000}[size]
    return 0 if value == 0 else word


@pytest.mark.logic
@given(st.sampled_from(sorted(_WIDTH)), st.lists(st.integers(min_value=0), min_size=1, max_size=16))
def test_canonical_floats_are_the_reference_bits_of_every_word(dtype: str, raw: list[int]) -> None:
    """For any bit patterns of every float width, the canonical data is the reference canonicalisation, word by word."""
    from senselab_worker.canonical import canonical_npy_bytes, npy_bytes

    size = _WIDTH[dtype][0]
    words = [w % (1 << (8 * size)) for w in raw]
    a = _bits(dtype, words)
    expected = _bits(dtype, [_reference_bits(dtype, w) for w in words])
    assert canonical_npy_bytes(a) == npy_bytes(expected)
    assert canonical_npy_bytes(a.astype(a.dtype.newbyteorder(">"))) == canonical_npy_bytes(a)


@pytest.mark.logic
@given(st.binary(min_size=1, max_size=16))
def test_canonical_bools_are_zero_or_one_and_keep_truth(raw: bytes) -> None:
    """Any bool bytes canonicalise to 0/1 by truth, so two arrays share a form exactly when their truths agree."""
    from senselab_worker.canonical import canonical_npy_bytes, npy_bytes

    a = numpy.frombuffer(raw, dtype=numpy.bool_)
    assert canonical_npy_bytes(a) == npy_bytes(numpy.array([b != 0 for b in raw], dtype=numpy.bool_))


@pytest.mark.logic
def test_canonical_strings_are_in_canonical_form() -> None:
    """A `U` array's strings are in NFC, never narrower than the array; one with a post-13.0 code point stays raw."""
    import io

    from senselab_worker.canonical import canonical_npy_bytes

    composed = numpy.array(["\u00e9", "a"])
    decomposed = numpy.array(["e\u0301", "a"])
    assert canonical_npy_bytes(composed.astype("<U2")) == canonical_npy_bytes(decomposed)
    raw = numpy.array(["e\u0301" + _UNASSIGNED_IN_13, "\u00e9"])
    assert canonical_npy_bytes(raw) == canonical_npy_bytes(raw.copy()) and raw[0] == "e\u0301" + _UNASSIGNED_IN_13
    assert numpy.load(io.BytesIO(canonical_npy_bytes(raw)))[0] == "e\u0301" + _UNASSIGNED_IN_13


SHAPES = [(), (0,), (1,), (3,), (2, 3), (1, 2, 2)]


NUMERIC = ["?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8"]


def _reference_element(dtype: numpy.dtype, word: int) -> int:
    """The canonical bits of one element by the ADR-006 list, from its raw little-endian word, without numpy."""
    if dtype.kind == "b":
        return 1 if word else 0
    if dtype.kind == "f":
        return _reference_bits(f"f{dtype.itemsize}", word)
    return word


@pytest.mark.contract
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("code", NUMERIC)
@pytest.mark.parametrize("order", ["<", ">"])
@settings(max_examples=25)
@given(st.data())
def test_canonical_npy_is_the_reference_form_for_every_shape_and_dtype(
    shape: tuple[int, ...], code: str, order: str, data: st.DataObject
) -> None:
    """Every accepted numeric dtype, either byte order, every shape: the reference bytes, the shape kept, 0-d apart."""
    import io
    import math

    from senselab_worker.canonical import canonical_npy_bytes, npy_bytes

    dtype = numpy.dtype(code).newbyteorder(cast(Literal["<", ">"], order))
    size = dtype.itemsize
    count = math.prod(shape)
    words = data.draw(
        st.lists(
            st.sampled_from([0, 1, 2, 255]) if code == "?" else st.integers(0, (1 << (8 * size)) - 1),
            min_size=count,
            max_size=count,
        )
    )
    little = numpy.dtype(code).newbyteorder("<")
    raw = b"".join(w.to_bytes(size, "little") for w in words)
    a = numpy.frombuffer(raw, dtype=little).reshape(shape).astype(dtype)
    expected = b"".join(_reference_element(little, w).to_bytes(size, "little") for w in words)
    reference = numpy.frombuffer(expected, dtype=little).reshape(shape)
    written = canonical_npy_bytes(a)
    assert written == npy_bytes(reference)
    back = numpy.load(io.BytesIO(written))
    assert back.shape == shape and back.dtype == little
    if count == 1:
        other = a.reshape(()) if shape != () else a.reshape((1,))
        assert canonical_npy_bytes(other) != written


@pytest.mark.contract
@given(st.sampled_from(SHAPES), st.data())
def test_canonical_string_arrays_keep_their_shape(shape: tuple[int, ...], data: st.DataObject) -> None:
    """A `U` array of any shape holds each string's canonical form, keeps its shape, and 0-d differs from (1,)."""
    import io
    import math

    from senselab_worker.canonical import canonical_npy_bytes, canonical_text

    count = math.prod(shape)
    alphabet = st.sampled_from(["a", "e", "́", "é", "̣", "\U0001fae0", "\U0001e08f"])
    texts = data.draw(st.lists(st.text(alphabet=alphabet, min_size=1, max_size=3), min_size=count, max_size=count))
    a = numpy.array(texts, dtype="<U3").reshape(shape) if count else numpy.zeros(shape, dtype="<U3")
    written = canonical_npy_bytes(a)
    back = numpy.load(io.BytesIO(written))
    assert back.shape == shape and back.dtype.itemsize >= a.dtype.itemsize
    assert [str(x) for x in back.reshape(-1).tolist()] == [canonical_text(t) for t in texts]
    if count == 1:
        other = a.reshape(()) if shape != () else a.reshape((1,))
        assert canonical_npy_bytes(other) != written


@pytest.mark.logic
def test_a_zero_d_bool_with_byte_2_keeps_its_shape() -> None:
    """A 0-d bool of byte 2 is written as 0-d `True`, not as `[True]`."""
    import io

    from senselab_worker.canonical import canonical_npy_bytes

    b = numpy.frombuffer(b"\x02", dtype=numpy.bool_).reshape(())
    assert canonical_npy_bytes(b) == canonical_npy_bytes(numpy.array(True))
    assert canonical_npy_bytes(b) != canonical_npy_bytes(numpy.array([True]))
    assert numpy.load(io.BytesIO(canonical_npy_bytes(b))).shape == ()


@st.composite
def _shared_dag(draw: st.DrawFn) -> object:
    """A value whose containers are drawn from those already built, so children are shared at varying depths."""
    nodes: list[object] = [draw(st.integers(-3, 3)), "s", None]
    for _ in range(draw(st.integers(1, 12))):
        children = draw(st.lists(st.sampled_from(nodes), min_size=0, max_size=3))
        if draw(st.booleans()):
            nodes.append(list(children))
        else:
            nodes.append({f"k{i}": child for i, child in enumerate(children)})
    return nodes[-1]


@pytest.mark.contract
@given(_shared_dag())
def test_a_shared_dag_encodes_as_its_expanded_tree(dag: object) -> None:
    """JCS and the tokeniser write a DAG exactly as the tree it expands to, whatever is shared and at what depth."""
    import json

    from senselab_worker.canonical import Tokeniser, jcs_dumps

    tree = json.loads(json.dumps(dag))
    assert jcs_dumps(dag) == jcs_dumps(tree)
    tokens = Tokeniser()
    assert jcs_dumps(tokens(dag)) == jcs_dumps(tokens(tree))
