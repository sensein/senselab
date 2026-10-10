"""The consensus text stream over the ASR hypotheses.

Sequences are aligned as sequences (``harmonize_transcripts``); one word is emitted per aligned
column, in column order. Each word records every source's own reading and timing verbatim and
carries a derived onset and offset from a monotone fit over the whole stream. The design and its
measurements are in ``specs/20260817-triage-workflow-dag/consensus-asr-redesign.md``; the owner's
rulings R-1..R-5 in ``consensus-asr-rulings.md`` beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import yaml

from senselab.audio.workflows.audio_analysis.harmonize import harmonize_transcripts, normalise_token

__all__ = [
    "ALGORITHM",
    "CONSENSUS_VERSION",
    "NORMALISATION",
    "ROUTINE",
    "SOURCE_ORDER",
    "TIME_FIT",
    "Consensus",
    "ConsensusWord",
    "Rebracketed",
    "SourceHypothesis",
    "Variant",
    "align_sources",
    "bracketed_form",
    "degenerate_parameters",
    "degenerate_token",
    "is_bracketed",
    "isotonic_median_fit",
    "located_span",
    "missing_span",
    "rebracket",
    "render_transcript",
    "vocabulary_key",
    "word_attributes",
    "word_from_attributes",
    "word_timing_parameters",
]

ALGORITHM = "star_sequence_alignment"
SOURCE_ORDER = "lexicographic_by_source_name"
TIME_FIT = "weighted_isotonic_median_over_located_spans"
CONSENSUS_VERSION = 2
NORMALISATION = "casefold; keep alphanumerics and apostrophe"
ROUTINE = "senselab.audio.workflows.triage.consensus.align_sources"

Outcome = Literal["agreement", "variant", "insertion"]

_VOCABULARY_EDGE_PUNCTUATION = ".,;:!?\"'()"
_AGREEMENT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SourceHypothesis:
    """One recognizer's word stream, as it reached the store.

    Attributes:
        name: The source's name; the key every consensus record uses.
        words: ``(text, start_s, end_s)`` per word, in the recognizer's own order; a time is None where the
            recognizer gave none.
        timestamp_source: How the words were timed (``native``, ``bundled_aligner``, ...).
        timestamp_model: The aligner that timed them, or None when the recognizer did.
        degenerate: The positions in ``words`` the recognizer produced as a degenerate run
            (:func:`degenerate_token`).
    """

    name: str
    words: tuple[tuple[str, float | None, float | None], ...]
    timestamp_source: str
    timestamp_model: str | None
    degenerate: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Variant:
    """One reading of a contested column.

    Attributes:
        text: The reading's label surface.
        sources: The sources that produced it, in source order.
        share: ``len(sources) / n_sources``.
    """

    text: str
    sources: tuple[str, ...]
    share: float


@dataclass(frozen=True)
class ConsensusWord:
    """One position of the consensus text stream.

    Attributes:
        index: The position in the stream; the only order a reader may use.
        text: The label surface; for a variant, ``variants[0].text``.
        bracketed: Whether ``text`` is a bracketed token.
        outcome: ``agreement``, ``variant`` or ``insertion``.
        sources: The sources with a member in this column, in source order.
        readings: ``source → surface as the recognizer produced it``.
        timings: ``source → (start_s, end_s)``, that source's own span, verbatim, for every source whose span
            is not missing (:func:`missing_span`).
        extent: The derived ``(onset_s, offset_s)`` from the monotone fit over the located spans
            (:func:`located_span`); for a word no source located, a zero-length extent where the stream
            before it ends.
        onset_spread_s: ``max − min`` of the members' located starts; 0.0 where none is located.
        offset_spread_s: ``max − min`` of the members' located ends; 0.0 where none is located.
        temporal_uncertainty_s: The wider of the widths of ``starts ∪ {onset}`` and
            ``ends ∪ {offset}`` over the located spans; None where none is located.
        variants: Every distinct reading, empty unless ``outcome == "variant"``.
        agreement: ``|largest same-key group| / n_sources``, or None where only one hypothesis
            reached the consensus, so no agreement could be read.
        degenerate_sources: The sources whose reading in the column is a degenerate run.
        untimed_sources: The sources whose span for the word is missing.
        unconfirmed: Whether only one recognizer read the word and none located it.
    """

    index: int
    text: str
    bracketed: bool
    outcome: Outcome
    sources: tuple[str, ...]
    readings: dict[str, str]
    timings: dict[str, tuple[float, float]]
    extent: tuple[float, float]
    onset_spread_s: float
    offset_spread_s: float
    temporal_uncertainty_s: float | None
    variants: tuple[Variant, ...]
    agreement: float | None
    degenerate_sources: tuple[str, ...] = ()
    untimed_sources: tuple[str, ...] = ()
    unconfirmed: bool = False

    @property
    def degenerate(self) -> bool:
        """Whether every reading in the column is a degenerate run."""
        return bool(self.degenerate_sources) and set(self.degenerate_sources) == set(self.sources)


@dataclass(frozen=True)
class Consensus:
    """The stream and the record of how it was produced.

    Attributes:
        words: The positions, in stream order.
        provenance: The fields of the ``consensus_transcript`` measurement this module owns.
    """

    words: tuple[ConsensusWord, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Rebracketed:
    """One aligned column read again under a vocabulary.

    Attributes:
        word: The position, with its surfaces and its ``bracketed`` flag re-read. Every other field
            is the alignment's and is unchanged.
        bracket_overrides: How many of the column's reading groups are a bracketed token outvoting a
            plain twin sharing its key.
    """

    word: ConsensusWord
    bracket_overrides: int


@dataclass(frozen=True)
class _Member:
    source: str
    raw: str
    display: str
    key: str
    start: float | None
    end: float | None
    degenerate: bool = False
    span: tuple[float, float] | None = None


DEGENERATE_PATH = Path(__file__).parent / "data" / "asr_degenerate.yaml"
WORD_TIMING_PATH = Path(__file__).parent / "data" / "word_timing.yaml"


@cache
def word_timing_parameters() -> dict[str, Any]:
    """The parameters of ``data/word_timing.yaml``.

    Returns:
        ``point_width_s``, the width a point span is read over.
    """
    document = yaml.safe_load(WORD_TIMING_PATH.read_text()) or {}
    return {"point_width_s": float(document["point_width_s"])}


def missing_span(start: float | None, end: float | None) -> bool:
    """Whether a recognizer's span for a word carries no timing.

    Args:
        start: The span's start in seconds, or None.
        end: The span's end in seconds, or None.

    Returns:
        True where either time is None or not finite, where both are zero, or where the end precedes the start.
    """
    if start is None or end is None:
        return True
    start, end = float(start), float(end)
    if not (math.isfinite(start) and math.isfinite(end)):
        return True
    return (start == 0.0 and end == 0.0) or end < start


def located_span(
    start: float | None,
    end: float | None,
    point_width_s: float | None = None,
    duration_s: float | None = None,
) -> tuple[float, float] | None:
    """Where a recognizer's span places a word.

    Args:
        start: The span's start in seconds, or None.
        end: The span's end in seconds, or None.
        point_width_s: The width a point span is read over; ``data/word_timing.yaml`` when None.
        duration_s: The duration of the audio the span was timed against, or None where unknown.

    Returns:
        None for a missing span (:func:`missing_span`); for a point span (``start == end``), the
        ``point_width_s`` window centred on it, inside ``[0, duration_s]``: shifted off an edge it would
        cross, and narrowed to ``duration_s`` only where the recording is shorter than the window. Without
        ``duration_s`` the window is floored at zero and its end is not bounded. Otherwise the span itself.
    """
    if missing_span(start, end):
        return None
    start_s, end_s = float(cast(float, start)), float(cast(float, end))
    if end_s > start_s:
        return start_s, end_s
    width = word_timing_parameters()["point_width_s"] if point_width_s is None else float(point_width_s)
    if duration_s is None:
        return max(0.0, start_s - width / 2.0), start_s + width / 2.0
    bound = max(0.0, float(duration_s))
    width = min(width, bound)
    low = min(max(0.0, start_s - width / 2.0), bound - width)
    return low, min(low + width, bound)


@cache
def degenerate_parameters() -> dict[str, Any]:
    """The bounds of ``data/asr_degenerate.yaml``, keyed by :func:`degenerate_token`'s parameter names.

    Returns:
        ``chars_min``, ``unit_chars_max`` and ``repeats_min``.
    """
    document = yaml.safe_load(DEGENERATE_PATH.read_text()) or {}
    return {
        "chars_min": int(document["chars_min"]),
        "unit_chars_max": int(document["unit_chars_max"]),
        "repeats_min": float(document["repeats_min"]),
    }


def degenerate_token(text: str, *, chars_min: int, unit_chars_max: int, repeats_min: float) -> bool:
    """Whether one recognizer token is a degenerate run: a decoder loop rather than a word.

    The token is casefolded and reduced to its letters and digits. It is degenerate when that is at
    least ``chars_min`` long, or when some unit of at most ``unit_chars_max`` characters repeats back to
    back at least ``repeats_min`` times inside it.

    Args:
        text: The token as the recognizer produced it.
        chars_min: The reduced length at or over which a single token is a run.
        unit_chars_max: The longest repeating unit looked for.
        repeats_min: The back-to-back repeats of one unit at or over which a token is a run.

    Returns:
        True for a degenerate run.
    """
    reduced = "".join(character for character in text.casefold() if character.isalnum())
    if len(reduced) >= chars_min:
        return True
    for period in range(1, unit_chars_max + 1):
        run = 0
        for position in range(len(reduced) - period):
            run = run + 1 if reduced[position] == reduced[position + period] else 0
            if (run + period) / period >= repeats_min:
                return True
    return False


def is_bracketed(text: str) -> bool:
    """Whether a token is a bracketed non-lexical marker such as ``[COUGH]``.

    Args:
        text: The token.

    Returns:
        True when the stripped token starts with ``[`` and ends with ``]``.
    """
    stripped = text.strip()
    return len(stripped) >= 2 and stripped.startswith("[") and stripped.endswith("]")


def vocabulary_key(token: str) -> str:
    """A token normalised for vocabulary matching: casefolded, edge punctuation stripped.

    Args:
        token: The raw token.

    Returns:
        The key.
    """
    return token.casefold().strip(_VOCABULARY_EDGE_PUNCTUATION)


def bracketed_form(text: str, onomatopoeic: set[str]) -> str | None:
    """The bracketed display of a non-lexical token, or None when the token is a plain word.

    Args:
        text: The token as the recognizer produced it.
        onomatopoeic: The ``words.onomatopoeic_tokens`` vocabulary, each entry a
            :func:`vocabulary_key`; empty while the key is null.

    Returns:
        The stripped token when it is already bracketed, ``[KEY]`` when its vocabulary key is in
        ``onomatopoeic``, else None.
    """
    stripped = text.strip()
    if is_bracketed(stripped):
        return stripped
    key = vocabulary_key(stripped)
    if key and key in onomatopoeic:
        return f"[{key.upper()}]"
    return None


def _midpoint_median(sorted_values: Sequence[float]) -> float:
    count = len(sorted_values)
    if count % 2:
        return sorted_values[count // 2]
    return (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2.0


def isotonic_median_fit(readings: Sequence[Sequence[float]]) -> tuple[list[float], list[bool]]:
    """The non-decreasing fit of one reading set per position, by pool-adjacent-violators on medians.

    A position's value is the median of its readings, an even count taking the midpoint of the two
    middle ones; adjacent positions whose values decrease are pooled onto the median of every
    reading in the block.

    Args:
        readings: One non-empty sequence of readings per position, in stream order.

    Returns:
        The fitted value per position, and per position whether it shares a block with another.

    Raises:
        ValueError: If a position carries no reading.
    """
    blocks: list[tuple[list[float], float, int]] = []
    for position, values in enumerate(readings):
        members = sorted(float(value) for value in values)
        if not members:
            raise ValueError(f"position {position} carries no reading")
        blocks.append((members, _midpoint_median(members), 1))
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
            right = blocks.pop()
            left = blocks.pop()
            merged = sorted(left[0] + right[0])
            blocks.append((merged, _midpoint_median(merged), left[2] + right[2]))
    fitted: list[float] = []
    pooled: list[bool] = []
    for _, value, size in blocks:
        fitted.extend([value] * size)
        pooled.extend([size > 1] * size)
    return fitted, pooled


def _surface(group: Sequence[_Member]) -> str:
    for member in group:
        if is_bracketed(member.display):
            return member.display
    return group[0].display


def _is_bracket_override(group: Sequence[_Member]) -> bool:
    """Whether a group is a bracketed token outvoting a plain twin sharing its key."""
    return any(is_bracketed(m.display) for m in group) and not all(is_bracketed(m.display) for m in group)


def _column_word(
    members: Sequence[_Member], n_sources: int
) -> tuple[str, Outcome, tuple[Variant, ...], float | None, int]:
    groups: dict[str, list[_Member]] = {}
    for member in members:
        groups.setdefault(member.key, []).append(member)
    ordered = sorted(groups.values(), key=lambda group: (all(m.degenerate for m in group), -len(group)))
    largest = ordered[0]
    if n_sources == 1:
        return _surface(largest), "insertion", (), None, sum(1 for group in ordered if _is_bracket_override(group))
    agreement = len(largest) / n_sources
    overrides = sum(1 for group in ordered if _is_bracket_override(group))
    if len(ordered) > 1:
        variants = tuple(
            Variant(text=_surface(group), sources=tuple(m.source for m in group), share=len(group) / n_sources)
            for group in ordered
        )
        return variants[0].text, "variant", variants, agreement, overrides
    text = _surface(largest)
    outcome: Outcome = "agreement" if len(largest) == n_sources else "insertion"
    return text, outcome, (), agreement, overrides


def align_sources(
    sources: Sequence[SourceHypothesis], *, onomatopoeic: set[str], duration_s: float | None = None
) -> Consensus:
    """Align the hypotheses as sequences and emit one word per aligned column, in column order.

    A single hypothesis is its own stream, read as one recognizer's: every word an ``insertion`` with
    no agreement, and the provenance's ``single_hypothesis`` True. A member whose span is missing
    (:func:`missing_span`) takes part in the alignment and stays out of the time fit; a column no member
    locates is ``unconfirmed`` where only one recognizer read it. A recognizer every one of whose words is
    missing its span is named in the provenance's ``aligner_failed``.

    Args:
        sources: The hypotheses, in any order; they are ordered by name.
        onomatopoeic: The ``words.onomatopoeic_tokens`` vocabulary, each entry a :func:`vocabulary_key`.
        duration_s: The duration of the audio the hypotheses were timed against; every located span is
            kept inside it (:func:`located_span`). Unbounded where None.

    Returns:
        The consensus stream and its provenance.

    Raises:
        LookupError: If no hypothesis was given.
    """
    if not sources:
        raise LookupError("consensus needs at least one asr_hypothesis measurement; found none")
    width = word_timing_parameters()["point_width_s"]
    ordered = sorted(sources, key=lambda source: source.name)
    names = [source.name for source in ordered]
    n_sources = len(ordered)

    members_by_source: dict[str, list[_Member]] = {}
    empty_dropped: dict[str, int] = {}
    untimed_members: dict[str, int] = {}
    for source in ordered:
        kept: list[_Member] = []
        dropped = 0
        for position, (raw, start, end) in enumerate(source.words):
            display = bracketed_form(raw, onomatopoeic)
            if display is None:
                display = raw
            key = normalise_token(display)
            if not key:
                dropped += 1
                continue
            missing = missing_span(start, end)
            kept.append(
                _Member(
                    source.name,
                    raw,
                    display,
                    key,
                    None if missing else float(cast(float, start)),
                    None if missing else float(cast(float, end)),
                    position in source.degenerate,
                    located_span(start, end, width, duration_s),
                )
            )
        members_by_source[source.name] = kept
        empty_dropped[source.name] = dropped
        untimed_members[source.name] = sum(1 for member in kept if member.span is None)
    aligner_failed = [
        source.name for source in ordered if source.words and all(missing_span(s, e) for _, s, e in source.words)
    ]

    lattice = harmonize_transcripts(
        {
            name: [(m.start or 0.0, m.end or 0.0, m.display) for m in members]
            for name, members in members_by_source.items()
            if members
        }
    )

    columns: list[list[_Member]] = []
    for slot in lattice.slots:
        column = [members_by_source[name][index] for name in names if (index := slot.indices.get(name)) is not None]
        columns.append(column)

    spans_of = {
        index: [m.span for m in column if m.span is not None]
        for index, column in enumerate(columns)
        if any(m.span is not None for m in column)
    }
    located = sorted(spans_of)
    fitted_onsets, onset_pooled = isotonic_median_fit([[span[0] for span in spans_of[i]] for i in located])
    fitted_offsets, offset_pooled = isotonic_median_fit([[span[1] for span in spans_of[i]] for i in located])
    # Onsets and offsets are fitted independently, so the pair is clamped rather than assumed.
    fitted_offsets = [max(offset, onset) for onset, offset in zip(fitted_onsets, fitted_offsets)]
    fit = {index: position for position, index in enumerate(located)}

    words: list[ConsensusWord] = []
    outcomes = {"agreement": 0, "variant": 0, "insertion": 0}
    overrides = 0
    max_shift = 0.0
    last_offset: float | None = None
    for index, column in enumerate(columns):
        text, outcome, variants, agreement, column_overrides = _column_word(column, n_sources)
        outcomes[outcome] += 1
        overrides += column_overrides
        spans = spans_of.get(index, [])
        uncertainty: float | None = None
        if spans:
            onset, offset = fitted_onsets[fit[index]], fitted_offsets[fit[index]]
            starts = [span[0] for span in spans]
            ends = [span[1] for span in spans]
            max_shift = max(
                max_shift,
                abs(onset - _midpoint_median(sorted(starts))),
                abs(offset - _midpoint_median(sorted(ends))),
            )
            onset_spread, offset_spread = max(starts) - min(starts), max(ends) - min(ends)
            uncertainty = max(
                max(*starts, onset) - min(*starts, onset),
                max(*ends, offset) - min(*ends, offset),
            )
            last_offset = offset
        else:
            following = next((fitted_onsets[fit[later]] for later in located if later > index), 0.0)
            onset = offset = last_offset if last_offset is not None else following
            onset_spread = offset_spread = 0.0
        words.append(
            ConsensusWord(
                index=index,
                text=text,
                bracketed=is_bracketed(text),
                outcome=outcome,
                sources=tuple(m.source for m in column),
                readings={m.source: m.raw for m in column},
                timings={m.source: (m.start, m.end) for m in column if m.start is not None and m.end is not None},
                extent=(onset, offset),
                onset_spread_s=onset_spread,
                offset_spread_s=offset_spread,
                temporal_uncertainty_s=uncertainty,
                variants=variants,
                agreement=agreement,
                degenerate_sources=tuple(m.source for m in column if m.degenerate),
                untimed_sources=tuple(m.source for m in column if m.span is None),
                unconfirmed=not spans and len(column) < 2,
            )
        )

    provenance: dict[str, Any] = {
        "version": CONSENSUS_VERSION,
        "algorithm": ALGORITHM,
        "routine": ROUTINE,
        "normalisation": NORMALISATION,
        "source_order": SOURCE_ORDER,
        "sources": [
            {
                "name": source.name,
                "n_words": len(source.words),
                "timestamp_source": source.timestamp_source,
                "timestamp_model": source.timestamp_model,
            }
            for source in ordered
        ],
        "n_sources": n_sources,
        "single_hypothesis": n_sources == 1,
        "degenerate_n": sum(1 for word in words if word.degenerate),
        "reference_source": lattice.reference,
        "n_words": len(words),
        "outcomes": outcomes,
        "bracket_overrides_n": overrides,
        "empty_tokens_dropped": empty_dropped,
        "time_fit": TIME_FIT,
        "point_width_s": width,
        "untimed_members": untimed_members,
        "unlocated_n": len(columns) - len(located),
        "unconfirmed_n": sum(1 for word in words if word.unconfirmed),
        "aligner_failed": aligner_failed,
        "n_words_time_shifted": sum(1 for a, b in zip(onset_pooled, offset_pooled) if a or b),
        "max_time_shift_s": max_shift,
    }
    return Consensus(words=tuple(words), provenance=provenance)


def word_attributes(word: ConsensusWord) -> dict[str, Any]:
    """The attributes one position of the stream is stored under.

    Args:
        word: The position.

    Returns:
        The mapping a ``word`` entity carries. The extent is the entity's own and is not in it.
        ``degenerate`` and ``degenerate_sources`` are present only where some reading is degenerate,
        ``untimed_sources`` only where some span is missing, and ``unconfirmed`` only where it is True.
    """
    attributes = {
        "text": word.text,
        "bracketed": word.bracketed,
        "outcome": word.outcome,
        "sources": list(word.sources),
        "readings": dict(word.readings),
        "timings": {source: list(span) for source, span in word.timings.items()},
        "onset_spread_s": word.onset_spread_s,
        "offset_spread_s": word.offset_spread_s,
        "temporal_uncertainty_s": word.temporal_uncertainty_s,
        "variants": [
            {"text": variant.text, "sources": list(variant.sources), "share": variant.share}
            for variant in word.variants
        ],
        "agreement": word.agreement,
        "index": word.index,
    }
    if word.degenerate_sources:
        attributes["degenerate"] = word.degenerate
        attributes["degenerate_sources"] = list(word.degenerate_sources)
    if word.untimed_sources:
        attributes["untimed_sources"] = list(word.untimed_sources)
    if word.unconfirmed:
        attributes["unconfirmed"] = True
    return attributes


def word_from_attributes(attributes: Mapping[str, Any], extent: tuple[float, float]) -> ConsensusWord:
    """One stored position, back as the record :func:`align_sources` emitted.

    The inverse of :func:`word_attributes`.

    Args:
        attributes: The ``word`` entity's attributes.
        extent: The entity's own extent, which the attributes do not carry.

    Returns:
        The position.

    Raises:
        KeyError: If the mapping is missing a field every ``word`` entity carries.
    """
    return ConsensusWord(
        index=int(attributes["index"]),
        text=str(attributes["text"]),
        bracketed=bool(attributes["bracketed"]),
        outcome=cast(Outcome, str(attributes["outcome"])),
        sources=tuple(str(source) for source in attributes["sources"]),
        readings={str(source): str(text) for source, text in attributes["readings"].items()},
        timings={str(source): (float(span[0]), float(span[1])) for source, span in attributes["timings"].items()},
        extent=(float(extent[0]), float(extent[1])),
        onset_spread_s=float(attributes["onset_spread_s"]),
        offset_spread_s=float(attributes["offset_spread_s"]),
        temporal_uncertainty_s=None
        if attributes["temporal_uncertainty_s"] is None
        else float(attributes["temporal_uncertainty_s"]),
        variants=tuple(
            Variant(
                text=str(variant["text"]),
                sources=tuple(str(source) for source in variant["sources"]),
                share=float(variant["share"]),
            )
            for variant in attributes["variants"]
        ),
        agreement=None if attributes["agreement"] is None else float(attributes["agreement"]),
        degenerate_sources=tuple(str(source) for source in attributes.get("degenerate_sources") or ()),
        untimed_sources=tuple(str(source) for source in attributes.get("untimed_sources") or ()),
        unconfirmed=bool(attributes.get("unconfirmed")),
    )


def rebracket(word: ConsensusWord, *, onomatopoeic: set[str], n_sources: int) -> Rebracketed:
    """Read one already-aligned column again under a vocabulary, aligning nothing.

    The column's membership, ``outcome`` and ``agreement`` are the same under every vocabulary; what
    the vocabulary decides is each member's display, and through it the column's surface, its
    ``bracketed`` flag and its variants' surfaces.

    Args:
        word: The stored position, from :func:`word_from_attributes`.
        onomatopoeic: The ``words.onomatopoeic_tokens`` vocabulary, each entry a
            :func:`vocabulary_key`.
        n_sources: How many recognizers the stream was aligned over, from the consensus provenance.

    Returns:
        The re-read position and its bracket-override count.

    Raises:
        ValueError: If the re-read column's outcome or agreement differs from the stored one.
    """
    members = [
        _Member(
            source=source,
            raw=word.readings[source],
            display=bracketed_form(word.readings[source], onomatopoeic) or word.readings[source],
            key=normalise_token(bracketed_form(word.readings[source], onomatopoeic) or word.readings[source]),
            start=word.timings[source][0] if source in word.timings else None,
            end=word.timings[source][1] if source in word.timings else None,
            degenerate=source in word.degenerate_sources,
        )
        for source in word.sources
    ]
    text, outcome, variants, agreement, overrides = _column_word(members, n_sources)
    if outcome != word.outcome or (agreement is None) != (word.agreement is None):
        drifted = True
    else:
        drifted = (
            agreement is not None
            and word.agreement is not None
            and abs(agreement - word.agreement) > _AGREEMENT_TOLERANCE
        )
    if drifted:
        raise ValueError(
            f"word {word.index} reads back as {outcome} at {agreement} where the store holds "
            f"{word.outcome} at {word.agreement}"
        )
    return Rebracketed(
        word=replace(word, text=text, bracketed=is_bracketed(text), variants=variants),
        bracket_overrides=overrides,
    )


def render_transcript(words: Sequence[ConsensusWord], *, strong: tuple[str, str] = ("**", "**")) -> str:
    """The stream as text, in ``index`` order.

    Args:
        words: The consensus words.
        strong: The marks wrapped around an agreement word. Both empty renders the plain
            transcript, a variant contributing its ``text`` alone rather than every reading
            joined by ``/``.

    Returns:
        The rendered text, words separated by single spaces.
    """
    plain = strong == ("", "")
    parts: list[str] = []
    for word in sorted(words, key=lambda w: w.index):
        token = word.text
        if word.outcome == "variant" and not plain:
            token = "/".join(variant.text for variant in word.variants)
        if word.outcome == "agreement":
            token = f"{strong[0]}{token}{strong[1]}"
        parts.append(token)
    return " ".join(parts)
