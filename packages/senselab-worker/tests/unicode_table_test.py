"""The packaged table of code points canonical text may hold, against the interpreter running the tests.

Run under Python 3.10 (Unicode 13.0.0) the table is regenerated and compared; run under any later Python
the same NFC fingerprint must come out of that interpreter's database, so both agree on NFC for every
allowed code point.
"""

import importlib.util
import sys
import unicodedata
from pathlib import Path
from types import ModuleType

import pytest
from hypothesis import given
from hypothesis import strategies as st
from senselab_worker.canonical import UNICODE_TABLE, UNICODE_VERSION, InvalidText, canonical_text, is_portable

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "unicode_table.py"
TABLE = Path(__file__).resolve().parents[1] / "src" / "senselab_worker" / UNICODE_TABLE


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("unicode_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table() -> tuple[dict[str, str], list[tuple[int, int]]]:
    header: dict[str, str] = {}
    ranges: list[tuple[int, int]] = []
    for line in TABLE.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        first, second = line.split(" ")
        if first in ("unicode_version", "nfc_fingerprint"):
            header[first] = second
        else:
            ranges.append((int(first, 16), int(second, 16)))
    return header, ranges


def _allowed(cp: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= cp <= b for a, b in ranges)


@pytest.mark.contract
def test_table_names_the_floor_unicode_version() -> None:
    """The table is for the Unicode version the encoders name, which is Python 3.10's."""
    header, _ = _table()
    assert header["unicode_version"] == UNICODE_VERSION == _generator().FLOOR_UNICODE == "13.0.0"


@pytest.mark.contract
def test_this_interpreter_normalises_every_allowed_code_point_as_the_floor_does() -> None:
    """The NFC fingerprint of every allowed code point under this interpreter equals the one the floor wrote."""
    header, ranges = _table()
    generator = _generator()
    if unicodedata.unidata_version == UNICODE_VERSION:
        assert generator.render(generator.allowed(), header["nfc_fingerprint"], UNICODE_VERSION) == TABLE.read_text(
            encoding="ascii"
        ), "the table is not what the generator writes from this database"
    else:
        assert all(unicodedata.category(chr(a)) != "Cn" and unicodedata.category(chr(b)) != "Cn" for a, b in ranges)
    assert generator.nfc_fingerprint(ranges) == header["nfc_fingerprint"], (
        f"Python {sys.version_info[:2]} (Unicode {unicodedata.unidata_version}) normalises an allowed code point "
        f"differently from Unicode {UNICODE_VERSION}"
    )


@pytest.mark.contract
@given(st.integers(min_value=0x80, max_value=0x10FFFF))
def test_canonical_text_normalises_exactly_the_code_points_in_the_table(cp: int) -> None:
    """One code point is put in NFC if the table allows it, kept as it is otherwise; a surrogate raises."""
    _, ranges = _table()
    ch = chr(cp)
    if 0xD800 <= cp <= 0xDFFF:
        with pytest.raises(InvalidText):
            canonical_text(ch)
        return
    assert is_portable(ch) == _allowed(cp, ranges)
    assert canonical_text(ch) == (unicodedata.normalize("NFC", ch) if _allowed(cp, ranges) else ch)


@pytest.mark.contract
def test_normalising_text_in_the_table_never_leaves_it() -> None:
    """NFC and NFD of every allowed code point hold only allowed code points, so normalised text never reads as raw."""
    _, ranges = _table()
    for start, end in ranges:
        for cp in range(start, end + 1):
            ch = chr(cp)
            assert is_portable(unicodedata.normalize("NFC", ch)) and is_portable(unicodedata.normalize("NFD", ch)), hex(
                cp
            )
