"""Format tags on persisted artifacts."""

import pytest
from pydantic import ValidationError
from senselab_core.kernel.formats import FormatTag, Formatted


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
