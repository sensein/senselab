"""This module contains the definition of the ScriptLine class."""

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator


class ScriptLine(BaseModel):
    """A class to represent a line in a script.

    Attributes:
        text (Optional[str]): The full text of the line (if provided).
        speaker (Optional[str]): The speaker of the line (if provided).
        chunks (Optional[List['ScriptLine']]): A list of script lines (if provided).
    """

    text: Optional[str] = None
    speaker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    chunks: Optional[List["ScriptLine"]] = None

    @model_validator(mode="before")
    def validate_text_and_speaker(cls, values: Dict[str, Any], _: ValidationInfo) -> Dict[str, Any]:
        """Validate that at least one of text or speaker is provided."""
        if not values.get("text") and not values.get("speaker"):
            raise ValueError("At least text or speaker must be provided.")
        return values

    @field_validator("text", "speaker")
    def strings_must_be_stripped(cls, v: str, _: ValidationInfo) -> str:
        """Strip the string of leading and trailing whitespace.

        Args:
            v (str): The text or speaker of the script line.

        Returns:
            str: The validated text or speaker of the script line.
        """
        if v is not None:
            v = v.strip()
        return v

    @field_validator("start", "end")
    def timestamps_must_be_positive(cls, v: float, _: ValidationInfo) -> float:
        """Validate that the start and end timestamps are positive.

        Args:
            v (float): The timestamp of the script line.

        Returns:
            float: The validated timestamp.
        """
        if v is not None:
            if v < 0:
                raise ValueError("Timestamps must be non-negative")
        return v

    def get_text(self) -> Union[str, None]:
        """Get the full text of the script line.

        Returns:
            Optional[str, None]: The full text of the script line, or None if not provided.
        """
        return self.text

    def get_speaker(self) -> Optional[str]:
        """Get the speaker of the script line.

        Returns:
            Optional[str]: The speaker of the script line.
        """
        return self.speaker

    def get_timestamps(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the start and end timestamps of the script line.

        Returns:
            Tuple[Optional[float], Optional[float]]: The start and end timestamps of the script line.
        """
        return self.start, self.end

    def get_chunks(self) -> Optional[List["ScriptLine"]]:
        """Get the list of chunks in the script line.

        Returns:
            Optional[List['ScriptLine']]: The list of chunks.
        """
        return self.chunks

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScriptLine":
        """Create a ScriptLine instance from a dictionary.

        Args:
            d (Dict[str, Any]): The dictionary containing the script line data.

        Returns:
            ScriptLine: An instance of ScriptLine.
        """
        if "timestamps" in d:
            start = d["timestamps"][0]
            end = d["timestamps"][1]
        elif "chunks" in d and "timestamps" in d["chunks"][0] and "timestamps" in d["chunks"][-1]:
            start = d["chunks"][0]["timestamps"][0]
            end = d["chunks"][-1]["timestamps"][1]
        else:
            start = None
            end = None
        return cls(
            text=d["text"] if "text" in d else None,
            speaker=d["speaker"] if "speaker" in d else None,
            start=start,
            end=end,
            chunks=[cls.from_dict(c) for c in d["chunks"]] if "chunks" in d else None,
        )
