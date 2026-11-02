"""Format tags on persisted artifacts."""

import pytest
from pydantic import ValidationError
from senselab_core.kernel.formats import FormatTag, Formatted, FormatVerdict, check


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
