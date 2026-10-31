"""H8c re-attack finding XW-5 on the wire reader, rebuilt from reviews/phase1-reattack/wire.md.

Each `test_xw5_*` rebuilds the probe and fails at 99a6d5f8; each is generated or parametrised over the class.
"""

import io
import json
from pathlib import Path
from typing import Any

import numpy
import pytest
from hypothesis import given
from hypothesis import strategies as st
from senselab_worker import _raw_digest
from senselab_worker.canonical import npy_bytes
from senselab_worker.codec import from_wire, parse_line
from senselab_worker.faults import ArrayIntegrityError, ProtocolError


def _hand_written(header: str, align: int, body: bytes) -> bytes:
    padded = (header + " " * (-(10 + len(header) + 1) % align) + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + len(padded).to_bytes(2, "little") + padded + body


def _npy_variants() -> dict[str, bytes]:
    a = numpy.arange(6, dtype="<f4").reshape(2, 3)
    out: dict[str, bytes] = {}
    for name, array, version in [
        ("big_endian", a.astype(">f4"), (1, 0)),
        ("fortran_order", numpy.asfortranarray(a), (1, 0)),
        ("version_2", a, (2, 0)),
    ]:
        buf = io.BytesIO()
        numpy.lib.format.write_array(buf, array, version=version)
        out[name] = buf.getvalue()
    out["trailing_byte"] = npy_bytes(a) + b"\0"
    out["key_order"] = _hand_written("{'fortran_order': False, 'descr': '<f4', 'shape': (2, 3), }", 64, a.tobytes())
    out["short_padding"] = _hand_written("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", 16, a.tobytes())
    return out


NPY_VARIANTS = _npy_variants()


def _ref(directory: Path, data: bytes) -> dict[str, Any]:
    sha = _raw_digest.bytes_digest("npy@1", data)
    path = directory / f"{sha}.npy"
    path.write_bytes(data)
    loaded = numpy.load(io.BytesIO(data), allow_pickle=False)
    return {"__array__": {"path": str(path), "sha256": sha, "dtype": loaded.dtype.str, "shape": list(loaded.shape)}}


@pytest.mark.logic
@pytest.mark.parametrize("variant", sorted(NPY_VARIANTS))
def test_xw5_reader_refuses_a_sha_consistent_non_canonical_npy(tmp_path: Path, variant: str) -> None:
    """The probe: a big-endian or Fortran-order file whose sha and reference agree was returned as it was.

    Every file that is not `npy_bytes` of the array it holds is refused; the canonical file of the same array passes.
    """
    data = NPY_VARIANTS[variant]
    assert data != npy_bytes(numpy.load(io.BytesIO(data), allow_pickle=False))
    with pytest.raises(ArrayIntegrityError, match="not npy_bytes"):
        from_wire({"x": _ref(tmp_path, data)}, array_dir=tmp_path)
    good = from_wire({"x": _ref(tmp_path, npy_bytes(numpy.arange(6, dtype="<f4").reshape(2, 3)))}, array_dir=tmp_path)
    assert good["x"].dtype.str == "<f4" and good["x"].flags.c_contiguous


MARK = "@@mark@@"
_clean = st.text(st.characters(blacklist_categories=("Cs",), blacklist_characters="@x"), max_size=6)
_trees = st.recursive(
    st.one_of(st.integers(-5, 5), _clean, st.just(MARK)),
    lambda inner: st.one_of(
        st.lists(inner, max_size=3), st.dictionaries(st.one_of(_clean, st.just(MARK)), inner, max_size=3)
    ),
    max_leaves=8,
)


@pytest.mark.logic
@given(tree=_trees, surrogate=st.integers(0xD800, 0xDFFF))
def test_xw5_parse_line_refuses_a_lone_surrogate_anywhere(tree: object, surrogate: int) -> None:
    """The probe: an escaped lone surrogate in an output key or value parsed, a string with no UTF-8 form.

    A lone surrogate escaped in any key or string of a line is refused; a surrogate pair, one astral
    character, is not.
    """
    line = json.dumps({"id": "r", "item": "a", "outputs": {"v": tree}}, ensure_ascii=True)
    if MARK not in line:
        return
    assert parse_line(line.replace(MARK, "x\\ud83d\\ude00y").encode())
    with pytest.raises(ProtocolError, match="lone surrogate"):
        parse_line(line.replace(MARK, f"x\\u{surrogate:04x}y").encode())
