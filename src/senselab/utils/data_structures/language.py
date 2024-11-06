"""This module provides the implementation of language data structures."""

import pycountry
import pycountry.db
from pydantic import BaseModel, field_validator


class Language(BaseModel):
    """Data structure for a language."""

    language_code: str

    @field_validator("language_code", mode="before")
    def validate_language(cls, v: str) -> str:
        """Validate that the language code is valid."""
        lang = (
            pycountry.languages.get(alpha_2=v) or pycountry.languages.get(alpha_3=v) or pycountry.languages.get(name=v)
        )

        if not lang:
            raise ValueError(f"{v} is not a valid ISO language code or name")

        return lang.alpha_3  # ISO 639-3

    @property
    def alpha_2(self) -> str:
        """Get the ISO 639-2 code for the language."""
        return pycountry.languages.get(alpha_3=self.language_code).alpha_2

    @property
    def alpha_3(self) -> str:
        """Get the ISO 639-3 code for the language."""
        return self.language_code

    @property
    def name(self) -> str:
        """Get the name of the language."""
        return pycountry.languages.get(alpha_3=self.language_code).name
