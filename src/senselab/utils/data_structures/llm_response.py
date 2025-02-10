"""This module contains the definition of the LLMResponse object."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Represents a response from a language model."""

    speaker: str
    text: str
    latency: Optional[float]
    in_tokens: Optional[int]
    out_tokens: Optional[int]

    def to_dict(self: "LLMResponse") -> dict:
        """Return a dictionary representation of the response.

        Returns:
            dict: A dictionary representation of the response.
        """
        return self.__dict__
