"""This module contains the definition of the Transcript class."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationInfo, field_validator


class Transcript(BaseModel):
    """A class to represent a transcript.

    Attributes:
        text (str): The full text of the transcript.
        chunks (Optional[List['Transcript.Chunk']]): A list of chunks of the transcript.
    """

    text: str
    chunks: Optional[List["Transcript.Chunk"]] = None

    def get_text(self) -> str:
        """Get the full text of the transcript.

        Returns:
            str: The full text of the transcript.
        """
        return self.text

    def get_chunks(self) -> Optional[List["Transcript.Chunk"]]:
        """Get the list of chunks in the transcript.

        Returns:
            Optional[List['Transcript.Chunk']]: The list of chunks.
        """
        return self.chunks

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transcript":
        """Create a Transcript instance from a dictionary.

        Args:
            d (Dict[str, Any]): The dictionary containing the transcript data.

        Returns:
            Transcript: An instance of Transcript.
        """
        return cls(
            text=d["text"].strip(), chunks=[cls.Chunk.from_dict(c) for c in d["chunks"]] if "chunks" in d else None
        )

    class Chunk(BaseModel):
        """A class to represent a chunk of the transcript.

        Attributes:
            text (str): The text of the chunk.
            start (float): The start timestamp of the chunk.
            end (float): The end timestamp of the chunk.
        """

        text: str
        start: float
        end: float
        speaker: Optional[str] = None

        @field_validator("text")
        def text_must_be_non_empty(cls, v: str, _: ValidationInfo) -> str:
            """Validate that the text is non-empty.

            Args:
                v (str): The text of the chunk.

            Returns:
                str: The validated text of the chunk.
            """
            if not v.strip():
                raise ValueError("Chunk text must be non-empty")
            return v

        @field_validator("start", "end")
        def timestamps_must_be_positive(cls, v: float, _: ValidationInfo) -> float:
            """Validate that the start and end timestamps are positive.

            Args:
                v (float): The timestamp of the chunk.

            Returns:
                float: The validated timestamp.
            """
            if v < 0:
                raise ValueError("Timestamps must be non-negative")
            return v
        
        @field_validator("speaker")
        def speaker_must_be_non_empty(cls, v: str, _: ValidationInfo) -> str:
            """Validate that the speaker is non-empty, if provided.

            Args:
                v (str): The speaker of the chunk.

            Returns:
                str: The validated speaker.
            """
            if v is not None:
                if not isinstance(v, str) and not v.strip():
                    raise ValueError("Speaker, if provided, must be a non-empty string")
            return v

        @classmethod
        def from_dict(cls, d: Dict[str, Any]) -> "Transcript.Chunk":
            """Create a Chunk instance from a dictionary.

            Args:
                d (Dict[str, Any]): The dictionary containing the chunk data.

            Returns:
                Transcript.Chunk: An instance of Chunk.
            """
            return cls(text=d["text"].strip(), 
                       start=d["timestamp"][0], 
                       end=d["timestamp"][1], 
                       speaker=d["speaker"] if "speaker" in d else None)