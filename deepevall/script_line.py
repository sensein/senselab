from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any, Tuple, Union

class ScriptLine(BaseModel):
    """A class to represent a line in a script.

    Attributes:
        text (Optional[str]): The full text of the line (if provided).
        speaker (Optional[str]): The speaker of the line (if provided).
        start (Optional[float]): The start timestamp of the line.
        end (Optional[float]): The end timestamp of the line.
        chunks (Optional[List['ScriptLine']]): A list of script lines (if provided).
    """

    text: Optional[str] = None
    speaker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    chunks: Optional[List["ScriptLine"]] = None

    @validator("text", "speaker", pre=True, always=True)
    def strings_must_be_stripped(cls, v: Optional[str]) -> Optional[str]:
        """Strip the string of leading and trailing whitespace."""
        return v.strip() if isinstance(v, str) else v

    @validator("start", "end")
    def timestamps_must_be_positive(cls, v: Optional[float]) -> Optional[float]:
        """Validate that the start and end timestamps are positive."""
        if v is not None and v < 0:
            raise ValueError("Timestamps must be non-negative")
        return v

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScriptLine":
        """Create a ScriptLine instance from a dictionary."""
        start, end = None, None
        if "timestamps" in d:
            start, end = d["timestamps"]
        elif "chunks" in d:
            start = d['chunks'][0].get('timestamps', [None])[0]
            end = d['chunks'][-1].get('timestamps', [None])[0]
        
        return cls(
            text=d.get("text"),
            speaker=d.get("speaker"),
            start=start,
            end=end,
            chunks=[cls.from_dict(c) for c in d.get("chunks", [])]
        )
