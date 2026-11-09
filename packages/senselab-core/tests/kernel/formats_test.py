"""Format tags on persisted artifacts."""

import json
import logging

import pytest
from pydantic import ValidationError
from senselab_core.kernel._base import Frozen
from senselab_core.kernel.formats import FormatTag, Formatted, FormatVerdict, check, reading


class _Thing(Formatted):
    format: FormatTag = "thing@2.1"
    x: int


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["annotation@1.0", "a.b_c-d@10.23", "x9@0.0"])
def test_formatted_accepts_name_at_major_minor(tag: str) -> None:
    """A tag is a lowercase name, `@`, and a major.minor version."""
    assert Formatted(format=tag).format == tag


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["Annotation@1.0", "annotation@1", "annotation", "@1.0", "a@1.0.0", "a b@1.0", ""])
def test_formatted_refuses_malformed_tags(tag: str) -> None:
    """Anything else is refused."""
    with pytest.raises(ValidationError):
        Formatted(format=tag)


@pytest.mark.logic
def test_subclass_writes_its_format_and_reads_its_minor_versions() -> None:
    """A subclass fills its own tag, reads any minor of its major, and ignores unknown fields."""
    assert _Thing(x=1).format == "thing@2.1"
    assert _Thing.model_validate({"format": "thing@2.7", "x": 1, "added_later": True}).format == "thing@2.7"


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["other@2.1", "thing@1.9", "thing@3.0"])
def test_subclass_refuses_another_name_or_major(tag: str) -> None:
    """A different artifact, or a major this reader does not know, is refused."""
    with pytest.raises(ValidationError):
        _Thing(format=tag, x=1)


@pytest.mark.logic
def test_subclass_with_malformed_default_is_refused_at_definition() -> None:
    """A subclass cannot declare a tag outside the pattern."""
    with pytest.raises(TypeError, match="malformed"):

        class _Bad(Formatted):
            format: FormatTag = "Bad"


@pytest.mark.logic
@pytest.mark.parametrize(
    ("found", "verdict"),
    [
        ("result@2.0", FormatVerdict.ACCEPTED),
        ("result@2.7", FormatVerdict.ACCEPTED),
        ("result@1.0", FormatVerdict.RETIRED),
        ("result@1.4", FormatVerdict.RETIRED),
        ("result@3.0", FormatVerdict.UNKNOWN),
        ("other@2.0", FormatVerdict.UNKNOWN),
        ("result@2", FormatVerdict.MALFORMED),
        (None, FormatVerdict.MALFORMED),
        (2, FormatVerdict.MALFORMED),
    ],
)
def test_one_format_gate(found: object, verdict: FormatVerdict) -> None:
    """A reader of `result@2.0` with `result@1` retired: its minors accepted, other tags skew, non-tags corrupt."""
    assert check("result@2.0", found, {"result@1"}) is verdict


class _Inner(Frozen):
    n: int


class _Outer(Formatted):
    format: FormatTag = "outer@1.0"
    inner: _Inner
    inners: tuple[_Inner, ...] = ()
    by_name: dict[str, _Inner] = {}


def _record(format: str = "outer@1.1", **extra: object) -> dict[str, object]:
    return {
        "format": format,
        "inner": {"n": 1, "later": 1},
        "inners": [{"n": 2, "later": 2}],
        "by_name": {"a": {"n": 3, "later": 3}},
        **extra,
    }


@pytest.mark.logic
def test_an_older_reader_reads_a_newer_minor_with_nested_fields() -> None:
    """Unknown fields of a later minor are ignored at every nesting level, and the known fields come back."""
    got = _Outer.model_validate(_record(top_later=1))
    assert got.inner.n == 1 and got.inners[0].n == 2 and got.by_name["a"].n == 3 and got.format == "outer@1.1"
    assert _Outer.model_validate_json(json.dumps(_record())).inner.n == 1


class _OuterAt11(Formatted):
    format: FormatTag = "outer@1.1"
    inner: _Inner


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["outer@1.0", "outer@1.1"])
def test_an_unknown_field_in_the_same_minor_is_refused(tag: str) -> None:
    """Under the reader's own minor an unknown field, nested or at the top, is corruption."""
    reader = _Outer if tag == "outer@1.0" else _OuterAt11
    for record in (_record(tag), {"format": tag, "inner": {"n": 1}, "top_later": 1}):
        with pytest.raises(ValidationError):
            reader.model_validate(record)


@pytest.mark.logic
def test_a_newer_major_is_refused() -> None:
    """A later major is refused, not read as a later minor."""
    with pytest.raises(ValidationError):
        _Outer.model_validate(_record("outer@2.0"))
    assert check("outer@1.0", "outer@2.0") is FormatVerdict.UNKNOWN


@pytest.mark.logic
def test_the_reading_scope_covers_a_bare_frozen_and_ends_with_the_read() -> None:
    """`reading` makes a non-`Formatted` model read a later minor; leaving it restores strictness."""
    with reading("outer@1.0", "outer@1.3"):
        assert _Inner.model_validate({"n": 1, "later": 1}).n == 1
    with reading("outer@1.0", "outer@1.0"), pytest.raises(ValidationError):
        _Inner.model_validate({"n": 1, "later": 1})
    with pytest.raises(ValidationError):
        _Inner.model_validate({"n": 1, "later": 1})


@pytest.mark.logic
def test_a_later_minor_is_reported_once(caplog: pytest.LogCaptureFixture) -> None:
    """The newer minor is logged the first time a reader meets it, not on every record."""
    caplog.set_level(logging.INFO, logger="senselab_core.kernel.formats")
    for _ in range(3):
        _Outer.model_validate(_record("outer@1.7"))
    assert [r.getMessage() for r in caplog.records].count(
        "reading outer@1.7 as outer@1.0: fields it does not know are ignored"
    ) == 1


@pytest.mark.logic
def test_writing_stays_strict() -> None:
    """Constructing a value with a field the type does not declare is refused, whatever the format."""
    with pytest.raises(ValidationError):
        _Outer(inner=_Inner(n=1), **{"later": 1})
    with pytest.raises(ValidationError):
        _Outer(inner={"n": 1, "later": 1})
