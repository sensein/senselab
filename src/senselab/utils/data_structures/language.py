"""Language data structures and normalization.

This module defines a `Language` pydantic model that **normalizes** an input
language identifier (ISO 639-1, ISO 639-3, or English name) to a canonical
**ISO 639-3** code at initialization.

Supported inputs for `language_code`:
  - ISO 639-1 two-letter codes (e.g., `"en"`, `"it"`),
  - ISO 639-3 three-letter codes (e.g., `"eng"`, `"ita"`),
  - English language names (e.g., `"English"`, `"Italian"`).

Internally it uses `pycountry.languages` to validate and resolve the code.

Notes:
  - Not every language has an ISO 639-1 (two-letter) code; in those cases,
    accessing `.alpha_2` will raise an `AttributeError` from `pycountry`.
  - The model stores the **alpha-3** code as its canonical representation.
"""

import pycountry
import pycountry.db
from pydantic import BaseModel, field_validator


class Language(BaseModel):
    """Normalized language identifier (canonical ISO 639-3).

    Initializing `Language(language_code=...)` accepts multiple forms (ISO 639-1,
    ISO 639-3, or English name) and **normalizes** the stored value to the
    ISO 639-3 `alpha_3` code.

    Attributes:
        language_code (str): Canonical ISO 639-3 code after validation.

    Example:
        >>> from senselab.utils.data_structures.language import Language
        >>> Language(language_code="en").alpha_3
        'eng'
        >>> Language(language_code="English").alpha_2
        'en'
        >>> Language(language_code="ita").name
        'Italian'
    """

    language_code: str

    @field_validator("language_code", mode="before")
    def validate_language(cls, v: str) -> str:
        """Validate and normalize input to ISO 639-3.

        Accepts ISO 639-1 (`alpha_2`), ISO 639-3 (`alpha_3`), or English name,
        and returns the canonical `alpha_3` code.

        Args:
            v (str): Language identifier (e.g., `"en"`, `"eng"`, `"English"`).

        Returns:
            str: ISO 639-3 code.

        Raises:
            ValueError: If the value does not correspond to a known language in
                `pycountry.languages`.
        """
        lang = (
            pycountry.languages.get(alpha_2=v) or pycountry.languages.get(alpha_3=v) or pycountry.languages.get(name=v)
        )

        if not lang:
            raise ValueError(f"{v} is not a valid ISO language code or name")

        return lang.alpha_3  # ISO 639-3

    @property
    def alpha_2(self) -> str:
        """Return the ISO 639-1 two-letter code.

        Returns:
            str: ISO 639-1 (`alpha_2`) code.

        Raises:
            AttributeError: If the language has no ISO 639-1 code in the
                `pycountry` database.

        Example:
            >>> Language(language_code="eng").alpha_2
            'en'
        """
        return pycountry.languages.get(alpha_3=self.language_code).alpha_2

    @property
    def alpha_3(self) -> str:
        """Return the ISO 639-3 three-letter code.

        Returns:
            str: ISO 639-3 (`alpha_3`) code (the model’s canonical form).

        Example:
            >>> Language(language_code="en").alpha_3
            'eng'
        """
        return self.language_code

    @property
    def name(self) -> str:
        """Return the English name of the language.

        Returns:
            str: English language name from `pycountry`.

        Example:
            >>> Language(language_code="eng").name
            'English'
        """
        return pycountry.languages.get(alpha_3=self.language_code).name
