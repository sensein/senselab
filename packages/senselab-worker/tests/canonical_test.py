"""Canonical JSON (RFC 8785) and canonical .npy bytes."""

import io
import json
import struct
import unicodedata

import numpy
import pytest
from hypothesis import given
from hypothesis import strategies as st
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps


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
    """Integers up to 2**53 in magnitude serialise; larger ones are refused."""
    assert jcs_dumps(2**53) == b"9007199254740992" and jcs_dumps(-(2**53)) == b"-9007199254740992"
    for bad in (2**53 + 1, -(2**53) - 1, numpy.uint64(2**64 - 1)):
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


_NFC_TEXT = st.text().map(lambda s: unicodedata.normalize("NFC", s))
_I_JSON_INTS = st.integers(min_value=-(2**53), max_value=2**53)
_FINITE_FLOATS = st.floats(allow_nan=False, allow_infinity=False).filter(
    lambda x: not (x.is_integer() and abs(x) > 2**53)
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
@pytest.mark.parametrize("dtype", ["<f4", ">f8", "|u1", "?", ">i2", "<c16", "<U3", "<M8[ns]"])
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
