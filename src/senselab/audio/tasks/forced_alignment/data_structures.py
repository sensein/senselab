"""Provides data structures for align functionality."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict


class SingleWordSegment(TypedDict):
    """A single word of a speech."""

    word: str
    start: Optional[float]
    end: Optional[float]
    score: float


class SingleCharSegment(TypedDict):
    """A single char of a speech."""

    char: str
    start: Optional[float]
    end: Optional[float]
    score: float


class SingleSegment(TypedDict, total=False):
    """A single segment (up to multiple sentences) of a speech."""

    start: float
    end: float
    text: str
    clean_char: Optional[List[str]]
    clean_cdx: Optional[List[int]]
    clean_wdx: Optional[List[int]]
    sentence_spans: Optional[List[Tuple[int, int]]]


class SingleAlignedSegment(TypedDict):
    """A single segment (up to multiple sentences) of a speech with word alignment."""

    start: float
    end: float
    text: str
    words: List[SingleWordSegment]
    chars: Optional[List[SingleCharSegment]]


class TranscriptionResult(TypedDict):
    """A list of segments and word segments of a speech."""

    segments: List[SingleSegment]
    language: str


class AlignedTranscriptionResult(TypedDict):
    """A list of segments and word segments of a speech."""

    segments: List[SingleAlignedSegment]
    word_segments: List[SingleWordSegment]


@dataclass
class Point:
    """Represents a point in the alignment path.

    Attributes:
        token_index (int): The index of the token in the transcript.
        time_index (int): The index of the time frame in the audio.
        score (float): The alignment score for this point.
    """

    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    """Represents a segment of aligned text.

    Attributes:
        label (str): The text label of the segment.
        start (int): The start time index of the segment.
        end (int): The end time index of the segment.
        score (float): The alignment score for the segment.
    """

    label: str
    start: int
    end: int
    score: float

    def __repr__(self) -> str:
        """Provides a string representation of the segment.

        Returns:
            str: A string representation of the segment.
        """
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self) -> int:
        """Calculates the length of the segment."""
        return self.end - self.start
