"""Tests the Language data structure."""

import pytest
from pydantic import ValidationError

from senselab.utils.data_structures import Language


def test_validate_language_valid_code_alpha_2() -> None:
    """Test valid ISO 639-1 language code."""
    lang = Language(language_code="en")
    assert lang.language_code == "eng"  # ISO 639-3 code


def test_validate_language_valid_code_alpha_3() -> None:
    """Test valid ISO 639-3 language code."""
    lang = Language(language_code="eng")
    assert lang.language_code == "eng"


def test_validate_language_valid_name() -> None:
    """Test valid language name."""
    lang = Language(language_code="English")
    assert lang.language_code == "eng"


def test_validate_language_invalid_code() -> None:
    """Test invalid language code."""
    with pytest.raises(ValidationError):
        Language(language_code="invalid")


def test_alpha_2_property() -> None:
    """Test alpha_2 property."""
    lang = Language(language_code="en")
    assert lang.alpha_2 == "en"


def test_alpha_3_property() -> None:
    """Test alpha_3 property."""
    lang = Language(language_code="en")
    assert lang.alpha_3 == "eng"


def test_name_property() -> None:
    """Test name property."""
    lang = Language(language_code="en")
    assert lang.name == "English"
