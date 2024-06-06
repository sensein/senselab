"""This module provides the implementation of language data structures."""

from iso639 import Language as IsoLanguage
from iso639 import LanguageNotFoundError
from pydantic import BaseModel, field_validator


class Language(BaseModel):
    """Data structure for a language."""

    language_code: str

    @field_validator("language_code", mode="before")
    def validate_language(cls, v: str) -> str:
        """Validate that the language code is valid."""
        try:
            lang = IsoLanguage.match(v)
        except LanguageNotFoundError:
            raise ValueError(f"{v} is not a valid ISO language code or name")
        return lang.part1  # Default to ISO 639-1 format

    @property
    def iso_639_1(self) -> str:
        """Get the ISO 639-1 code for the language."""
        return self.language_code

    @property
    def iso_639_2b(self) -> str:
        """Get the ISO 639-2b code for the language."""
        lang = IsoLanguage.from_part1(self.language_code)
        return lang.part2b

    @property
    def iso_639_2t(self) -> str:
        """Get the ISO 639-2t code for the language."""
        lang = IsoLanguage.from_part1(self.language_code)
        return lang.part2t

    @property
    def iso_639_3(self) -> str:
        """Get the ISO 639-3 code for the language."""
        lang = IsoLanguage.from_part1(self.language_code)
        return lang.part3

    @property
    def name(self) -> str:
        """Get the name of the language."""
        lang = IsoLanguage.from_part1(self.language_code)
        return lang.name

    @property
    def family(self) -> str:
        """Get the family of the language."""
        lang = IsoLanguage.from_part1(self.language_code)
        return lang.macrolanguage
