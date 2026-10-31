"""H8c re-attack findings on canonical encoding (XC-1, XC-4, XC-5, XC-6), rebuilt from reviews/phase1-reattack/canon.md.

Each probe imports what existed at 99a6d5f8 inside the test and fails there on the finding's behaviour.
"""

import struct
import time

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
"""Pinned strings an interpreter's NFC would change: the probes of XC-1."""


@pytest.mark.logic
@pytest.mark.parametrize("text", NFC_SENSITIVE, ids=lambda t: ascii(t))
def test_xc1_text_with_post_13_code_points_has_one_digest_on_every_interpreter(text: str) -> None:
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


@pytest.mark.logic
def test_xc1_weights_paths_with_post_13_code_points_are_named_raw(tmp_path: object) -> None:
    """A weight file whose relative path holds a code point unassigned in 13.0 has an identity over its raw name."""
    import hashlib
    import json
    from pathlib import Path

    from senselab_worker.hashing import weights_identity

    root = Path(str(tmp_path))
    name = f"e\u0301{_UNASSIGNED_IN_13}.bin"
    f = root / name
    f.write_bytes(b"x")
    pairs = json.dumps([[name, hashlib.sha256(b"x").hexdigest()]], ensure_ascii=False, separators=(",", ":"))
    expected = hashlib.sha256(b"weights@2\x00" + pairs.encode("utf-8")).hexdigest()
    assert weights_identity([f], root=root) == expected


def _inf_minus_inf() -> numpy.ndarray:
    with numpy.errstate(invalid="ignore"):
        return numpy.array([numpy.inf]) - numpy.array([numpy.inf])


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
def test_xc4_values_adr006_counts_as_one_have_one_npy_form(left: numpy.ndarray, right: numpy.ndarray) -> None:
    """Every NaN, `0.0` and `-0.0`, and every nonzero bool byte, write the same canonical `.npy` bytes."""
    from senselab_worker.canonical import canonical_npy_bytes

    assert canonical_npy_bytes(left) == canonical_npy_bytes(right)


def _quadratic_dag(levels: int) -> list[object]:
    x: list[object] = []
    for _ in range(levels):
        x = [x, x]
    return x


@pytest.mark.logic
def test_xc5_a_shared_dag_is_written_once_per_node() -> None:
    """Twenty-three levels of `x = [x, x]` (8M leaves expanded, 16 s at 99a6d5f8) encode in well under 5 s."""
    from senselab_worker.canonical import jcs_dumps

    started = time.monotonic()
    text = jcs_dumps(_quadratic_dag(23))
    assert time.monotonic() - started < 5.0
    assert len(text) == 5 * 2**23 - 3 and text.startswith(b"[[[[")


@pytest.mark.logic
def test_xc5_a_shared_dag_beyond_the_budget_is_refused_in_bounded_time() -> None:
    """Sixty-three levels (2**63 leaves) are refused by the work budget, not expanded."""
    from senselab_worker.canonical import jcs_dumps

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
def test_xc6_jcs_loads_refuses_what_jcs_dumps_cannot_have_written(text: str) -> None:
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
    if value != value:
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

    dtype = numpy.dtype(code).newbyteorder(order)
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
def test_x_i1_a_zero_d_bool_with_byte_2_keeps_its_shape() -> None:
    """The review's probe: a 0-d bool of byte 2 is written as 0-d `True`, not as `[True]`."""
    import io

    from senselab_worker.canonical import canonical_npy_bytes

    b = numpy.frombuffer(b"\x02", dtype=numpy.bool_).reshape(())
    assert canonical_npy_bytes(b) == canonical_npy_bytes(numpy.array(True))
    assert canonical_npy_bytes(b) != canonical_npy_bytes(numpy.array([True]))
    assert numpy.load(io.BytesIO(canonical_npy_bytes(b))).shape == ()
