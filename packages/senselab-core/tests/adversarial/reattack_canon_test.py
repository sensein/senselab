# ruff: noqa: ANN401
"""H8c re-attack findings on canonical identity (XC-2..XC-5, XC-7, XC-8), rebuilt from reviews/phase1-reattack/canon.md.

Each `test_xc<n>_*` probe imports what existed at 99a6d5f8 inside the test and fails there on the finding's
behaviour; the other tests are contracts over the class the finding belongs to.
"""

import hashlib
import re
import time
import unicodedata
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import numpy
import pytest
from hypothesis import given
from hypothesis import strategies as st
from senselab_core.kernel._base import Frozen

REPO = Path(__file__).resolve().parents[4]
ADR_006 = REPO / "specs" / "20261030-140022-senselab-2x-api" / "adr" / "ADR-006-formats.md"


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
def test_xc2_an_array_and_its_sha256_hex_string_differ() -> None:
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
    colour = f"{Colour.__module__}.{Colour.__qualname__}"
    big = 2**60
    return {
        "$array": (a, [sha, {"$array": sha}, a.tolist()]),
        "$set": (frozenset({1, 2}), [[1, 2], (1, 2), {"$set": [1, 2]}]),
        "$path": (PurePosixPath("a/b"), ["a/b", {"$path": "a/b"}]),
        "$enum": (Colour.RED, ["red", {"$enum": [colour, "red"]}]),
        "$type": (int, ["builtins.int", {"$type": "builtins.int"}]),
        "$value": (Inner(a=1), [{"a": 1}, {"$value": {"type": inner, "fields": {"a": 1}}}]),
        "$int": (big, [str(big), float(big), {"$int": str(big)}]),
        "$f": (float("nan"), ["NaN", {"$f": "NaN"}]),
        "$bytes": (b"\x01", ["01", {"$bytes": "01"}]),
        "$dec": (Decimal("1.5"), ["1.5", 1.5, {"$dec": "1.5"}]),
    }


@pytest.mark.contract
def test_every_value_family_has_its_own_encoding() -> None:
    """For every token family, a value of it differs from its payload and from a mapping spelling its token."""
    from senselab_worker.canonical import TOKENS

    samples = _family_samples()
    assert set(samples) | {"$nd"} == set(TOKENS), "a token family has no sample here"
    digests = {token: Holder(x=value).canonical_digest() for token, (value, _) in samples.items()}
    assert len(set(digests.values())) == len(digests)
    for token, (value, lookalikes) in samples.items():
        for other in lookalikes:
            assert Holder(x=value) != Holder(x=other), f"{token}: {other!r}"
            assert digests[token] != Holder(x=other).canonical_digest(), f"{token}: {other!r}"


@pytest.mark.contract
@given(
    st.text(alphabet="ae\u0301\u00e9\u0323 ", max_size=4),
    st.sampled_from(["\U0001fae0", "\U0001fae8", "\U0001e08f", "\u0c5d", "\U000e0080"]),
)
def test_text_with_post_13_code_points_is_recorded_and_is_its_own_family(head: str, mark: str) -> None:
    """Provenance records it (H-R29, H-R30); its value digest differs from its NFC and from every token family."""
    from senselab_core.runtime.provenance import ProvStore

    text = head + mark
    ProvStore(run_id="t").entity(prov_type="span", extent=None, attributes={text: [text]})
    digest = Holder(x=text).canonical_digest()
    others = {Holder(x=v).canonical_digest() for v, _ in _family_samples().values()}
    others.add(Holder(x=unicodedata.normalize("NFC", head)).canonical_digest())
    if unicodedata.normalize("NFC", text) != text:
        others.add(Holder(x=unicodedata.normalize("NFC", text)).canonical_digest())
    assert digest not in others


@pytest.mark.contract
def test_provenance_tokens_are_the_shared_families() -> None:
    """Provenance writes each family with the shared tokeniser, arrays by value as `$nd`."""
    from senselab_core.runtime.provenance import normalised

    assert normalised(numpy.array([1, 2], dtype="<i2")) == {"$nd": {"dtype": "int16", "shape": [2], "values": [1, 2]}}
    assert normalised({"$nd": 1}) == {"$$nd": 1}
    for token, (value, _) in _family_samples().items():
        if token in ("$array", "$type", "$value"):
            continue
        assert list(normalised(value)) == [token]  # type: ignore[call-overload]


@pytest.mark.logic
@pytest.mark.parametrize("make", ["cycle", "dict-cycle", "deep"])
def test_xc3_a_value_that_contains_itself_or_nests_deeply_raises_the_shared_error(make: str) -> None:
    """A cycle or 5000-deep nesting gives the shared guard's `ValueError`, from the digest, `==` and `hash`."""
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
    held = Holder(x=value)
    for read in (held.canonical_digest, lambda: held == Holder(x=1), lambda: hash(held)):
        with pytest.raises(ValueError, match="contains itself|deeper than"):
            read()


@pytest.mark.contract
@given(st.lists(st.sampled_from(["list", "tuple", "dict", "frozenset"]), min_size=1, max_size=120))
def test_any_nesting_digests_or_raises_value_error(kinds: list[str]) -> None:
    """Every nesting of every container kind either digests or raises `ValueError`; none overflows the stack."""
    value: object = 1
    for kind in reversed(kinds):
        if kind == "list":
            value = [value]
        elif kind == "dict":
            value = {"k": value}
        elif kind == "tuple":
            value = (value,)
        else:
            try:
                value = frozenset({value})
            except TypeError:
                value = (value,)
    try:
        Holder(x=value).canonical_digest()
    except ValueError as err:
        assert "deeper than" in str(err) or "has no canonical form" in str(err)


@pytest.mark.logic
def test_xc4_audio_identity_is_one_per_value() -> None:
    """Audio whose samples differ only by `-0.0` or a NaN's sign has one `content:` identity."""
    from senselab_core.kernel.signals import Audio

    zero = numpy.array([0.0, 1.0], dtype=numpy.float32)
    negative = numpy.array([-0.0, 1.0], dtype=numpy.float32)
    nan = numpy.array([0x7FC00000, 0], dtype="<u4").view("<f4")
    negative_nan = numpy.array([0xFFC00001, 0], dtype="<u4").view("<f4")
    assert Audio.from_array(zero, 16_000).identity == Audio.from_array(negative, 16_000).identity
    assert Audio.from_array(nan, 16_000).identity == Audio.from_array(negative_nan, 16_000).identity


@pytest.mark.logic
@pytest.mark.parametrize("where", ["provenance", "store", "value"])
def test_xc5_a_shared_dag_is_bounded_on_every_canonical_path(where: str) -> None:
    """Sixty levels of `x = [x, x]` are refused in bounded time by provenance, the store and `value@2`."""
    from senselab_core.runtime.provenance import ProvStore
    from senselab_core.runtime.store import output_identity

    dag = _dag(60)
    started = time.monotonic()
    with pytest.raises(ValueError, match="exceeds"):
        if where == "provenance":
            ProvStore(run_id="t").entity(prov_type="span", extent=None, attributes={"x": dag})
        elif where == "store":
            output_identity({"x": dag})
        else:
            Holder(x=dag).canonical_digest()
    assert time.monotonic() - started < 5.0


@pytest.mark.logic
def test_xc5_a_shared_dag_under_the_budget_is_walked_once_per_node() -> None:
    """Twenty-two levels (4M leaves expanded) normalise and digest in well under 5 s."""
    from senselab_core.runtime.provenance import ProvStore

    started = time.monotonic()
    ProvStore(run_id="t").entity(prov_type="span", extent=None, attributes={"x": _dag(22)})
    assert time.monotonic() - started < 5.0


@pytest.mark.logic
def test_xc7_set_elements_of_one_canonical_form_are_one_element() -> None:
    """NFC-equal strings, and distinct NaN objects, are one element of a set in provenance and `value@2`."""
    from senselab_core.runtime.provenance import normalised
    from senselab_worker.canonical import jcs_dumps

    composed, decomposed = "\u00e9", "e\u0301"
    assert jcs_dumps(normalised({composed, decomposed})) == jcs_dumps(normalised({composed}))
    nan_a, nan_b = float("nan"), float("nan")
    assert jcs_dumps(normalised({nan_a, nan_b})) == jcs_dumps(normalised({nan_a}))
    assert Holder(x={composed, decomposed}) == Holder(x={composed})


@pytest.mark.contract
@given(st.frozensets(st.text(alphabet="ae\u0301\u00e9\u0323\u1ea1", max_size=3), max_size=6))
def test_a_set_is_its_elements_canonical_forms(strings: frozenset[str]) -> None:
    """A set of strings encodes as the set of their NFC forms, in provenance and `value@2`."""
    from senselab_core.runtime.provenance import normalised
    from senselab_worker.canonical import jcs_dumps

    forms = frozenset(unicodedata.normalize("NFC", s) for s in strings)
    assert jcs_dumps(normalised(strings)) == jcs_dumps(normalised(forms))
    assert Holder(x=strings) == Holder(x=forms)


def _adr_tags(heading: str) -> set[str]:
    _, found, rest = ADR_006.read_text(encoding="utf-8").partition(f"**{heading}**")
    return set(re.findall(r"`([a-z][a-z0-9-]*@\d+)`", rest.split("\n\n", 1)[0])) if found else set()


@pytest.mark.contract
def test_xc8_adr006_lists_exactly_the_registered_tags() -> None:
    """ADR-006's lists of identity and blob tags are `IDENTITY_TAGS` and `BLOB_TAGS`, no more and no fewer."""
    from senselab_worker.digest import BLOB_TAGS, IDENTITY_TAGS

    assert _adr_tags("Registered identity tags") == set(IDENTITY_TAGS)
    assert _adr_tags("Registered blob tags") == set(BLOB_TAGS)
