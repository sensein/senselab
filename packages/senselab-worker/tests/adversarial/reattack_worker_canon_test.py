"""H8c re-attack findings on canonical encoding (XC-1, XC-4, XC-5, XC-6), rebuilt from reviews/phase1-reattack/canon.md.

Each probe imports what existed at 99a6d5f8 inside the test and fails there on the finding's behaviour.
"""

import struct
import time

import numpy
import pytest
from hypothesis import given
from hypothesis import strategies as st

_UNASSIGNED_IN_13 = "a\U0001e08f\u0323"
"""U+1E08F is a combining mark (ccc 230) in Unicode 15 and unassigned in 13, so NFC reorders it on 3.12 only."""


@pytest.mark.logic
@pytest.mark.parametrize("where", ["value", "key", "nested"])
def test_xc1_text_whose_nfc_differs_by_interpreter_is_refused(where: str) -> None:
    """A string holding a code point unassigned in Unicode 13 has no canonical form, as a value or a key."""
    from senselab_worker.canonical import jcs_dumps

    value = {"value": _UNASSIGNED_IN_13, "key": {_UNASSIGNED_IN_13: 1}, "nested": [{"k": [_UNASSIGNED_IN_13]}]}[where]
    with pytest.raises(ValueError, match="unassigned in Unicode 13.0.0"):
        jcs_dumps(value)


@pytest.mark.logic
def test_xc1_weights_paths_hold_only_portable_text(tmp_path: object) -> None:
    """A weight file whose relative path holds an unassigned code point has no `weights@2` identity."""
    from pathlib import Path

    from senselab_worker.hashing import weights_identity

    root = Path(str(tmp_path))
    f = root / f"w{_UNASSIGNED_IN_13}.bin"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="unassigned"):
        weights_identity([f], root=root)


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
def test_canonical_strings_are_nfc_and_portable() -> None:
    """A `U` array's strings are in NFC, never narrower than the array, and an unassigned code point is refused."""
    from senselab_worker.canonical import canonical_npy_bytes

    composed = numpy.array(["\u00e9", "a"])
    decomposed = numpy.array(["e\u0301", "a"])
    assert canonical_npy_bytes(composed.astype("<U2")) == canonical_npy_bytes(decomposed)
    with pytest.raises(ValueError, match="unassigned"):
        canonical_npy_bytes(numpy.array([_UNASSIGNED_IN_13]))
