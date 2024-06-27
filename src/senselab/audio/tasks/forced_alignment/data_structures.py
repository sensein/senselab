"""Provides data structures for align functionality."""

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
