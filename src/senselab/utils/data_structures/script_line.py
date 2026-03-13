"""Transcript data structures: `ScriptLine`.

This module defines `ScriptLine`, a lightweight, validated container for
transcripts and diarization segments. A `ScriptLine` may represent:

- a top-level utterance (with optional `speaker`, `start`, `end`),
- a container with nested `chunks` (each a `ScriptLine`),
- or both (text + timing + nested chunks).

Validation rules:
    - At least one of `text` or `speaker` must be provided.
    - `start`/`end` timestamps, when present, must be non-negative.
    - `text` and `speaker` strings are automatically stripped.

Utilities:
    - `.from_dict(...)` builds a `ScriptLine` from dicts produced by ASR/diarization
      outputs. If `timestamps` are not on the top-level but present on the first and
      last chunk, the line’s `(start, end)` are inferred from those.

Examples:
    Basic line:
        >>> from senselab.utils.data_structures.script_line import ScriptLine
        >>> ScriptLine(text="hello world", start=0.0, end=1.2)
        hello world [0.00 - 1.20]

    With speaker:
        >>> ScriptLine(text="hi", speaker="spk1", start=0.5, end=1.0)
        spk1: hi [0.50 - 1.00]

    Nested chunks:
        >>> parent = ScriptLine(
        ...     text="hello world",
        ...     speaker="spk1",
        ...     chunks=[ScriptLine(text="hello", start=0.0, end=0.5),
        ...             ScriptLine(text="world", start=0.6, end=1.2)]
        ... )
        >>> print(parent)
        spk1: hello world
        >>> print(parent._str_with_indent())
        spk1: hello world
            hello [0.00 - 0.50]
            world [0.60 - 1.20]
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator


class ScriptLine(BaseModel):
    """A single transcript or diarization line (optionally with nested chunks).

    Attributes:
        text (str | None): The text content, if available.
        speaker (str | None): Speaker/label identifier, if available (e.g., `"spk1"`).
        start (float | None): Start time in seconds (non-negative).
        end (float | None): End time in seconds (non-negative).
        chunks (list[ScriptLine] | None): Optional nested segments, each a `ScriptLine`.

    Notes:
        - Either `text` or `speaker` must be provided (or both).
        - When printing, `__str__` shows `"speaker: text [start - end]"` if timestamps
          are present; otherwise omits the time bracket.

    Example:
        >>> from senselab.utils.data_structures.script_line import ScriptLine
        >>> ScriptLine(text="okay", speaker="spk2", start=1.0, end=1.4)
        spk2: okay [1.00 - 1.40]
    """

    text: Optional[str] = None
    speaker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    chunks: Optional[List["ScriptLine"]] = None

    @model_validator(mode="before")
    def validate_text_and_speaker(cls, values: Dict[str, Any], _: ValidationInfo) -> Dict[str, Any]:
        """Ensure that at least one of `text` or `speaker` is provided.

        Args:
            values: Incoming field values prior to model construction.

        Returns:
            The (possibly modified) values dict.

        Raises:
            ValueError: If both `text` and `speaker` are missing/empty.
        """
        if not values.get("text") and not values.get("speaker"):
            raise ValueError("At least text or speaker must be provided.")
        return values

    @field_validator("text", "speaker")
    def strings_must_be_stripped(cls, v: str, _: ValidationInfo) -> str:
        """Strip leading/trailing whitespace from `text` and `speaker`.

        Args:
            v: The input string.

        Returns:
            The stripped string (or `None` if input was `None`).
        """
        if v is not None:
            v = v.strip()
        return v

    @field_validator("start", "end")
    def timestamps_must_be_positive(cls, v: float, _: ValidationInfo) -> float:
        """Validate that `start` and `end` are non-negative (when present).

        Args:
            v: The timestamp value.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If a provided timestamp is negative.
        """
        if v is not None:
            if v < 0:
                raise ValueError("Timestamps must be non-negative")
        return v

    def get_text(self) -> Union[str, None]:
        """Return the text content (if any).

        Returns:
            str | None: The line's text, or `None` if not provided.

        Example:
            >>> ScriptLine(text="hi").get_text()
            'hi'
        """
        return self.text

    def get_speaker(self) -> Optional[str]:
        """Return the speaker/label (if any).

        Returns:
            str | None: The speaker identifier, or `None`.

        Example:
            >>> ScriptLine(speaker="spk1").get_speaker()
            'spk1'
        """
        return self.speaker

    def get_timestamps(self) -> Tuple[Optional[float], Optional[float]]:
        """Return `(start, end)` timestamps.

        Returns:
            tuple[float | None, float | None]: `(start, end)` in seconds (either may be `None`).

        Example:
            >>> ScriptLine(start=0.0, end=1.2).get_timestamps()
            (0.0, 1.2)
        """
        return self.start, self.end

    def get_chunks(self) -> Optional[List["ScriptLine"]]:
        """Return nested chunks, if present.

        Returns:
            list[ScriptLine] | None: The `chunks` list, or `None`.

        Example:
            >>> ScriptLine(chunks=[ScriptLine(text="a")]).get_chunks() is not None
            True
        """
        return self.chunks

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScriptLine":
        """Construct a `ScriptLine` from a nested dictionary.

        Accepted structure:
            - Top-level may include: `"text"`, `"speaker"`, `"timestamps": [start, end]`, `"chunks": [...]`.
            - If `"timestamps"` are missing at the top level *but* present on the first
              and last chunk, `(start, end)` are inferred from those chunk endpoints.

        Args:
            d: Dictionary representation (e.g., from ASR/diarization results).

        Returns:
            ScriptLine: The constructed instance.

        Example:
            >>> data = {
            ...     "text": "hello world",
            ...     "speaker": "spk1",
            ...     "chunks": [
            ...         {"text": "hello", "timestamps": [0.0, 0.5]},
            ...         {"text": "world", "timestamps": [0.6, 1.2]},
            ...     ],
            ... }
            >>> sl = ScriptLine.from_dict(data)
            >>> sl.start, sl.end
            (0.0, 1.2)
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

    def __str__(self) -> str:
        """Return a concise, human-readable representation.

        Format:
            ``"{speaker}: {text} [start - end]"`` with time bracket only if both
            `start` and `end` are present.

        Returns:
            str: A single-line string.

        Example:
            >>> str(ScriptLine(text="ok", speaker="spk1", start=0.0, end=0.5))
            'spk1: ok [0.00 - 0.50]'
        """

        def format_timestamp(start: Optional[float], end: Optional[float]) -> str:
            if start is not None and end is not None:
                return f" [{start:.2f} - {end:.2f}]"
            return ""

        speaker_part = f"{self.speaker}: " if self.speaker else ""
        timestamp_part = format_timestamp(self.start, self.end)
        text_part = self.text if self.text is not None else ""

        return f"{speaker_part}{text_part}{timestamp_part}"

    def _str_with_indent(self, indent: int = 0) -> str:
        """Pretty-print recursively with indentation (for nested chunks).

        Args:
            indent: Current indentation level (0 for top level).

        Returns:
            str: Multi-line indented string.

        Example:
            >>> parent = ScriptLine(
            ...     text="parent",
            ...     chunks=[ScriptLine(text="child", start=0, end=1)]
            ... )
            >>> print(parent._str_with_indent())  # doctest: +NORMALIZE_WHITESPACE
            parent
                child [0.00 - 1.00]
        """
        indent_space = "    " * indent
        lines = [f"{indent_space}{self.__str__()}"]
        if self.chunks:
            for chunk in self.chunks:
                lines.append(chunk._str_with_indent(indent + 1))
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Representation mirrors the indented pretty-print.

        Returns:
            str: Multi-line string suitable for debugging and logs.

        Example:
            >>> repr(ScriptLine(text="x"))  # doctest: +ELLIPSIS
            'x'
        """
        return self._str_with_indent(0)
