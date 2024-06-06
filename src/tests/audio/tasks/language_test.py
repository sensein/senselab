"""Tests the Language data structure."""

import pytest
from iso639 import Language as IsoLanguage
from pydantic import ValidationError

from senselab.utils.data_structures.language import Language


def test_validate_language_valid_code() -> None:
    """Valid language code."""
    lang = Language(language_code="en")
    assert lang.iso_639_1 == "en"


def test_validate_language_invalid_code() -> None:
    """Invalid language code."""
    with pytest.raises(ValidationError):
        Language(language_code="invalid")


def test_iso_639_1_property() -> None:
    """ISO 639-1 property."""
    lang = Language(language_code="en")
    assert lang.iso_639_1 == "en"


def test_iso_639_2b_property() -> None:
    """ISO 639-2b property."""
    lang = Language(language_code="en")
    assert lang.iso_639_2b == IsoLanguage.from_part1("en").part2b


def test_iso_639_2t_property() -> None:
    """ISO 639-2t property."""
    lang = Language(language_code="en")
    assert lang.iso_639_2t == IsoLanguage.from_part1("en").part2t


def test_iso_639_3_property() -> None:
    """ISO 639-3 property."""
    lang = Language(language_code="en")
    assert lang.iso_639_3 == IsoLanguage.from_part1("en").part3


def test_name_property() -> None:
    """Name property."""
    lang = Language(language_code="en")
    assert lang.name == IsoLanguage.from_part1("en").name


def test_family_property() -> None:
    """Family property."""
    lang = Language(language_code="en")
    assert lang.family == IsoLanguage.from_part1("en").macrolanguage
