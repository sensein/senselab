"""The six shapes an L1 measurement can have (D-18).

``SignalRow(measurement: Mapping[str, float])`` fits only the scalar-per-bucket case, and L1's
outputs are six different kinds of object — four of which have no per-bucket scalar form at all.
Forcing them through one tabular row is what produced every reduction the real-run audit found: a
per-speaker probability matrix stored as its mean, 527 label scores stored as a hand-picked sum, a
span set stored as a covered fraction, a transcript stored as a word-overlap duration — each on a
0.1 s grid none of them was measured at, beside provenance describing the measurement that was
discarded.

Each reduction is a **decision**, and every one of them is now an L2 derivative that names its
choice. What L1 stores is the native shape, which is what this module is:

===========  ====================================================================
``Series``   ``(n_frames,)`` at a fixed hop, one named quantity
``Matrix``   ``(n_frames × n_channels)``, channels **named** or **arbitrary**
``Categorical``  ``(n_windows × k)`` over a fixed vocabulary, top-*k* truncated
``Embedding``    ``(n_windows × n_dims)``
``Spans``    variable-length ``[(start, end, label)]`` — on no grid at all
``Tree``     a ``ScriptLine``: text, nested chunks, per-node scores
===========  ====================================================================

**A bucket grid means something different to each of them**, which is the distinction one row type
could not express and :class:`GridRelation` now carries: it is a *resample* for ``Series`` and
``Matrix``, a *projection* for ``Categorical`` and ``Embedding`` (a 0.96 s window is not a 0.1 s
bucket), and a *reduction* for ``Spans`` and ``Tree`` (a transcript has no natural per-bucket
value). Conflating the three is what made one row type look sufficient.

Nothing here folds, thresholds, selects or rescales. A value the tool did not report is ``None``,
never ``0.0`` — zero is a confident claim, and imputing it manufactures confidence nobody expressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Literal, Optional, Union

__all__ = [
    "Capacity",
    "Categorical",
    "ChannelSemantics",
    "Embedding",
    "GridRelation",
    "LabelScore",
    "Matrix",
    "Measurement",
    "Series",
    "Span",
    "Spans",
    "TIMESTAMP_SOURCES",
    "TimestampSource",
    "Tree",
    "Window",
]


class GridRelation(Enum):
    """What imposing a bucket grid on a shape actually does to it.

    Named because the three are not interchangeable and the code that projects has to know which
    it is performing. A ``RESAMPLE`` is arithmetic over values that exist at a finer hop; a
    ``PROJECT`` assigns one window's value to buckets it spans, asserting nothing new; a ``REDUCE``
    invents a per-bucket quantity that the object does not have, which is a decision and belongs to
    L2 with its choice named.
    """

    RESAMPLE = "resample"
    PROJECT = "project"
    REDUCE = "reduce"


ChannelSemantics = Literal["named", "arbitrary"]
"""Do a matrix's channels mean the same thing in every frame?

``named`` — a fixed meaning per column (frequency bands, model heads). ``arbitrary`` — the column
order is an artifact of the inference, as a diarizer's speaker columns are within each chunk. The
distinction decides whether *any* per-channel operation across frames is meaningful, and storing a
mean is what erased it (D-5): the mean is a choice for named channels and is meaningless for
arbitrary ones.
"""

TimestampSource = Literal["native", "bundled_aligner", "external_aligner"]
"""Where a transcript's word boundaries came from.

An ASR output *is* a time-aligned output, so the aligner is provenance on the transcript rather than
a signal of its own (D-20). Recorded because two transcripts timed by one aligner have correlated
word boundaries, and the ASR axis compares word boundaries — this field is what lets the correlation
be measured rather than assumed absent.
"""

TIMESTAMP_SOURCES: Final[frozenset[str]] = frozenset(("native", "bundled_aligner", "external_aligner"))
"""The closed set, checked at construction. A fourth kind of timing provenance is a design change."""

Capacity = Union[int, Literal["unbounded"], None]
"""A speaker-aware tool's maximum representable speaker count (D-19).

Three states, all meaningful and none substitutable:

- ``int`` — a fixed ceiling (sortformer 4, the 8-speaker variant 8). At the ceiling the tool
  contributes a **lower bound**, not a point: it cannot report one more and does not say so.
- ``"unbounded"`` — a clustering pipeline with no fixed-width head (``community-1``).
- ``None`` — capacity does not apply (a word span set has no speaker ceiling).

Without the distinction a reader cannot tell "3 speakers active" from "3 active and the model had no
fourth column", and a count posterior fused across tools of differing capacity is biased toward the
smallest.
"""


@dataclass(frozen=True, slots=True)
class Series:
    """One named quantity at a fixed hop — brouhaha's heads, HNR, LUFS, a level above floor.

    Attributes:
        values: One reading per frame. ``None`` where the tool reported nothing.
        hop_s: Frame advance, in seconds — the resolution this actually has.
        window_s: Analysis window, in seconds. Longer than ``hop_s`` when frames overlap.
        units: The tool's own units (``"dB"``, ``"probability"``, ``"LUFS"``). Never ``"mixed"``:
            one quantity per series is the point.
        start_s: Offset of the first frame.
    """

    values: tuple[Optional[float], ...]
    hop_s: float
    window_s: float
    units: str
    start_s: float = 0.0

    grid_relation: ClassVar[GridRelation] = GridRelation.RESAMPLE

    def __post_init__(self) -> None:
        """Reject a hop that cannot describe a frame advance."""
        if self.hop_s <= 0:
            raise ValueError(f"hop_s must be positive, got {self.hop_s!r}")

    @property
    def duration_s(self) -> float:
        """Span covered by the frames, at the hop they were measured at."""
        return len(self.values) * self.hop_s

    @property
    def measured_count(self) -> int:
        """Frames the tool actually reported — not the same as ``len(values)``."""
        return sum(1 for v in self.values if v is not None)


@dataclass(frozen=True, slots=True)
class Matrix:
    """``n_frames × n_channels`` at a fixed hop — a per-band noise floor, a multi-head output.

    The channels survive L1 because pooling them is a choice among ``mean`` / ``max`` / ``noisy-or``
    that changes the answer. Storing the pooled value made that choice invisibly, and it is what
    returned ``1.0000`` in 100% of frames on a clip that was half digital silence.

    Attributes:
        rows: One tuple per frame, each as wide as ``channels``.
        channels: Column names, in order.
        channel_semantics: Whether those names mean the same thing in every frame.
        hop_s: Frame advance, in seconds.
        window_s: Analysis window, in seconds.
        units: The tool's own units, shared by every channel. Channels in different units are
            different signals.
        start_s: Offset of the first frame.
    """

    rows: tuple[tuple[Optional[float], ...], ...]
    channels: tuple[str, ...]
    hop_s: float
    window_s: float
    units: str
    channel_semantics: ChannelSemantics = "named"
    start_s: float = 0.0

    grid_relation: ClassVar[GridRelation] = GridRelation.RESAMPLE

    def __post_init__(self) -> None:
        """Reject a ragged matrix, which would misalign every channel after the short row."""
        if self.hop_s <= 0:
            raise ValueError(f"hop_s must be positive, got {self.hop_s!r}")
        width = len(self.channels)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"frame {index} has {len(row)} values for {width} channel names")

    @property
    def n_channels(self) -> int:
        """How many channels — the count is permutation-invariant even when the columns are not."""
        return len(self.channels)

    @property
    def channels_are_comparable_across_frames(self) -> bool:
        """May a consumer track one column through time, or align it with another matrix's?

        ``False`` for arbitrary channels, where column *k* of frame *i* and of frame *j* are
        unrelated. A count of active channels stays well-defined either way, which is why a count is
        answerable from an arbitrary-channel matrix and an average is not.
        """
        return self.channel_semantics == "named"

    def channel(self, name: str) -> tuple[Optional[float], ...]:
        """One channel through time.

        Raises:
            KeyError: For a channel nothing declared — a typo used to yield an empty column and
                read as "this channel had nothing to say".
            ValueError: When the channels are arbitrary, so a column through time is not a thing
                that exists.
        """
        if not self.channels_are_comparable_across_frames:
            raise ValueError("channels are permutation-arbitrary; a column through time is not defined")
        try:
            index = self.channels.index(name)
        except ValueError:
            raise KeyError(f"no channel named {name!r}; have {self.channels}") from None
        return tuple(row[index] for row in self.rows)


@dataclass(frozen=True, slots=True)
class LabelScore:
    """One label and its score, in the classifier's own units."""

    label: str
    score: float


@dataclass(frozen=True, slots=True)
class Window:
    """One analysis window's label scores, descending."""

    start: float
    end: float
    scores: tuple[LabelScore, ...]

    def __post_init__(self) -> None:
        """Reject a window that ends before it starts."""
        if self.end < self.start:
            raise ValueError(f"window end {self.end!r} precedes start {self.start!r}")


@dataclass(frozen=True, slots=True)
class Categorical:
    """Per-window scores over a fixed vocabulary — AudioSet (527), YAMNet (521).

    The target is the **label distribution itself**. Which labels count as speech, music or
    environment is an L2 mapping over it, which keeps the category map changeable without re-running
    the model and means a new category needs no new signal.

    Attributes:
        windows: One entry per native window, each carrying its own boundaries.
        vocabulary_id: Which label set, so two classifiers' scores are only compared under a
            recorded mapping.
        vocabulary_size: Its full size, against which ``top_k`` is a truncation.
        top_k: How many labels were kept. **On the row**, because label mass over a set whose
            members fell outside it is not recoverable — without ``k`` a consumer cannot tell "this
            label scored below the *k*-th" from "this label scored nothing".
        units: ``"probability"`` for a multi-label sigmoid. Not softmaxed: a softmax over 527
            classes structurally suppresses secondary background categories.
    """

    windows: tuple[Window, ...]
    vocabulary_id: str
    vocabulary_size: int
    top_k: int
    units: str = "probability"

    grid_relation: ClassVar[GridRelation] = GridRelation.PROJECT

    def __post_init__(self) -> None:
        """Reject a truncation wider than the vocabulary it truncates."""
        if self.top_k > self.vocabulary_size:
            raise ValueError(f"top_k {self.top_k} exceeds vocabulary size {self.vocabulary_size}")

    @property
    def mass_is_truncated(self) -> bool:
        """Could a label be missing because it fell below the cutoff rather than scoring nothing?

        A category map whose labels routinely fall outside ``top_k`` is a reason to raise it, and
        this makes that a visible decision rather than a silent truncation.
        """
        return self.top_k < self.vocabulary_size

    @property
    def windows_overlap(self) -> bool:
        """Do consecutive windows share signal?

        Overlapping windows are not independent observations, so a consumer that averages them is
        double-counting the shared span. Measured from the boundaries rather than declared, because
        the hop is a run parameter and the windows are what was actually scored.
        """
        return any(later.start < earlier.end for earlier, later in zip(self.windows, self.windows[1:]))


@dataclass(frozen=True, slots=True)
class Embedding:
    """``n_windows × n_dims`` — 192 for ECAPA, 256 for ResNet.

    Stored as the intermediate a diarizer is built from, keyed on ``(model, window, hop,
    audio_signature)``. A vector votes on nothing, so it is a cache rather than a signal, and it is
    recomputed only when the framing changes.
    """

    vectors: tuple[tuple[float, ...], ...]
    window_s: float
    hop_s: float

    grid_relation: ClassVar[GridRelation] = GridRelation.PROJECT

    def __post_init__(self) -> None:
        """Reject ragged vectors, whose raggedness would otherwise surface as a distance."""
        if not self.vectors:
            return
        width = len(self.vectors[0])
        for index, vector in enumerate(self.vectors):
            if len(vector) != width:
                raise ValueError(f"vector {index} has width {len(vector)}, expected {width}")

    @property
    def dims(self) -> int:
        """Vector width, or 0 when there are none."""
        return len(self.vectors[0]) if self.vectors else 0


@dataclass(frozen=True, slots=True)
class Span:
    """One ``(start, end, label)`` at the tool's own boundaries.

    Attributes:
        start: Seconds.
        end: Seconds.
        label: The tool's own id, in the tool's own namespace (``SPEAKER_00``, ``spk0``) — never
            harmonised here, because harmonising across tools is an L2 derivative.
        confidence: The tool's own confidence, when it reported one. ``None`` means it did not,
            which is different from ``0.0``: defaulting to zero asserts maximal doubt on the tool's
            behalf.
    """

    start: float
    end: float
    label: str
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        """Reject a span that ends before it starts."""
        if self.end < self.start:
            raise ValueError(f"span end {self.end!r} precedes start {self.start!r}")


@dataclass(frozen=True, slots=True)
class Spans:
    """A variable-length span set — diarization output, word boundaries.

    **On no grid at all**, so there is no resolution to record and no projection to get wrong. L2
    derives occupancy or a count by projecting them, which is why no frame-level per-speaker
    posterior is needed: the object L1 owes is a span set.

    Attributes:
        spans: In no required order; L2 sorts for what it needs.
        capacity: The tool's speaker ceiling — see :data:`Capacity`.
    """

    spans: tuple[Span, ...]
    capacity: Capacity = None

    grid_relation: ClassVar[GridRelation] = GridRelation.REDUCE

    def is_censored_at(self, count: int) -> bool:
        """Is a claim of ``count`` speakers at this tool's ceiling, and therefore a lower bound?

        A capacity-bounded tool asked about a recording that exceeds its bound does not fail — it
        produces a confident wrong answer. So at the ceiling its contribution is *censored*, which
        is neither absence nor a bound merely being met, and a fold that treats it as evidence
        against one more speaker is biased toward the smallest-capacity tool.
        """
        return isinstance(self.capacity, int) and count >= self.capacity


@dataclass(frozen=True, slots=True)
class Tree:
    """A ``ScriptLine`` verbatim: text, nested word chunks with their own boundaries, node scores.

    The tool's own scores stay **inside** — ``avg_logprob``, ``no_speech_prob``, ``token_entropy``
    are quantities the tool reported, which is what L1 records. Nothing is folded, thresholded or
    rescaled; the ``avg_logprobs: []`` list the old row carried existed only because a 0.1 s bucket
    spanned several segments, and it disappears with the bucket.

    Attributes:
        script_line: The tree as the recognizer produced it.
        timestamp_source: Where its word boundaries came from — see :data:`TimestampSource`.
    """

    script_line: Any
    timestamp_source: TimestampSource

    grid_relation: ClassVar[GridRelation] = GridRelation.REDUCE

    def __post_init__(self) -> None:
        """Reject an unrecorded kind of timing provenance rather than storing an unknown string."""
        if self.timestamp_source not in TIMESTAMP_SOURCES:
            raise ValueError(f"timestamp_source {self.timestamp_source!r} is not one of {sorted(TIMESTAMP_SOURCES)}")


Measurement = Union[Series, Matrix, Categorical, Embedding, Spans, Tree]
"""What a signal or a derivative holds. A union, not a base class.

Six shapes with genuinely different structure — a common base would only have the fields all six
share, which is none of the ones that matter. Consumers dispatch on
:attr:`~GridRelation`-carrying membership instead, which is the property that actually differs.
"""
