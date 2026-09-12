"""The consensus text stream over the ASR hypotheses.

Sequences are aligned as sequences (``harmonize_transcripts``); one word is emitted per aligned
column, in column order. Each word records every source's own reading and timing verbatim and
carries a derived onset and offset from a monotone fit over the whole stream. The design and its
measurements are in ``specs/20260817-triage-workflow-dag/consensus-asr-redesign.md``; the owner's
rulings R-1..R-5 in ``consensus-asr-rulings.md`` beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence, cast

from senselab.audio.workflows.audio_analysis.harmonize import harmonize_transcripts, normalise_token

__all__ = [
    "ALGORITHM",
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
    "is_bracketed",
    "isotonic_median_fit",
    "rebracket",
    "render_transcript",
    "vocabulary_key",
    "word_attributes",
    "word_from_attributes",
]

ALGORITHM = "star_sequence_alignment"
SOURCE_ORDER = "lexicographic_by_source_name"
TIME_FIT = "weighted_isotonic_median"
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
        words: ``(text, start_s, end_s)`` per word, in the recognizer's own order.
        timestamp_source: How the words were timed (``native``, ``bundled_aligner``, ...).
        timestamp_model: The aligner that timed them, or None when the recognizer did.
    """

    name: str
    words: tuple[tuple[str, float, float], ...]
    timestamp_source: str
    timestamp_model: str | None


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
        timings: ``source → (start_s, end_s)``, that source's own span, verbatim.
        extent: The derived ``(onset_s, offset_s)`` from the monotone fit over the stream.
        onset_spread_s: ``max − min`` of the members' own starts.
        offset_spread_s: ``max − min`` of the members' own ends.
        temporal_uncertainty_s: The wider of the widths of ``starts ∪ {onset}`` and
            ``ends ∪ {offset}``.
        variants: Every distinct reading, empty unless ``outcome == "variant"``.
        agreement: ``|largest same-key group| / n_sources``.
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
    temporal_uncertainty_s: float
    variants: tuple[Variant, ...]
    agreement: float


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
    start: float
    end: float


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

    Each position's own value is the median of its readings; an even count takes the midpoint of
    the two middle readings. Adjacent positions whose values decrease are pooled, and a pooled
    block's value is the median of every reading in it. The output is non-decreasing for any input.

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


def _column_word(members: Sequence[_Member], n_sources: int) -> tuple[str, Outcome, tuple[Variant, ...], float, int]:
    groups: dict[str, list[_Member]] = {}
    for member in members:
        groups.setdefault(member.key, []).append(member)
    ordered = sorted(groups.values(), key=lambda group: -len(group))
    largest = ordered[0]
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


def align_sources(sources: Sequence[SourceHypothesis], *, onomatopoeic: set[str]) -> Consensus:
    """Align the hypotheses as sequences and emit one word per aligned column, in column order.

    Args:
        sources: The hypotheses, in any order; they are ordered by name.
        onomatopoeic: The ``words.onomatopoeic_tokens`` vocabulary, each entry a :func:`vocabulary_key`.

    Returns:
        The consensus stream and its provenance.

    Raises:
        LookupError: If fewer than two hypotheses were given.
    """
    if len(sources) < 2:
        names = [source.name for source in sources]
        raise LookupError(f"consensus needs at least two asr_hypothesis measurements; found {len(sources)}: {names}")
    ordered = sorted(sources, key=lambda source: source.name)
    names = [source.name for source in ordered]
    n_sources = len(ordered)

    members_by_source: dict[str, list[_Member]] = {}
    empty_dropped: dict[str, int] = {}
    for source in ordered:
        kept: list[_Member] = []
        dropped = 0
        for raw, start, end in source.words:
            display = bracketed_form(raw, onomatopoeic)
            if display is None:
                display = raw
            key = normalise_token(display)
            if not key:
                dropped += 1
                continue
            kept.append(_Member(source.name, raw, display, key, float(start), float(end)))
        members_by_source[source.name] = kept
        empty_dropped[source.name] = dropped

    lattice = harmonize_transcripts(
        {name: [(m.start, m.end, m.display) for m in members] for name, members in members_by_source.items() if members}
    )

    columns: list[list[_Member]] = []
    for slot in lattice.slots:
        column = [members_by_source[name][index] for name in names if (index := slot.indices.get(name)) is not None]
        columns.append(column)

    onsets, onset_pooled = isotonic_median_fit([[m.start for m in column] for column in columns])
    offsets, offset_pooled = isotonic_median_fit([[m.end for m in column] for column in columns])
    # Onsets and offsets are fitted independently, so the pair is clamped rather than assumed.
    offsets = [max(offset, onset) for onset, offset in zip(onsets, offsets)]

    words: list[ConsensusWord] = []
    outcomes = {"agreement": 0, "variant": 0, "insertion": 0}
    overrides = 0
    max_shift = 0.0
    for index, column in enumerate(columns):
        text, outcome, variants, agreement, column_overrides = _column_word(column, n_sources)
        outcomes[outcome] += 1
        overrides += column_overrides
        starts = [m.start for m in column]
        ends = [m.end for m in column]
        onset, offset = onsets[index], offsets[index]
        max_shift = max(
            max_shift,
            abs(onset - _midpoint_median(sorted(starts))),
            abs(offset - _midpoint_median(sorted(ends))),
        )
        words.append(
            ConsensusWord(
                index=index,
                text=text,
                bracketed=is_bracketed(text),
                outcome=outcome,
                sources=tuple(m.source for m in column),
                readings={m.source: m.raw for m in column},
                timings={m.source: (m.start, m.end) for m in column},
                extent=(onset, offset),
                onset_spread_s=max(starts) - min(starts),
                offset_spread_s=max(ends) - min(ends),
                temporal_uncertainty_s=max(
                    max(*starts, onset) - min(*starts, onset),
                    max(*ends, offset) - min(*ends, offset),
                ),
                variants=variants,
                agreement=agreement,
            )
        )

    provenance: dict[str, Any] = {
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
        "reference_source": lattice.reference,
        "n_words": len(words),
        "outcomes": outcomes,
        "bracket_overrides_n": overrides,
        "empty_tokens_dropped": empty_dropped,
        "time_fit": TIME_FIT,
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
    """
    return {
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


def word_from_attributes(attributes: Mapping[str, Any], extent: tuple[float, float]) -> ConsensusWord:
    """One stored position, back as the record :func:`align_sources` emitted.

    The inverse of :func:`word_attributes`, so a reader that has only the store can work in the
    stream's own vocabulary rather than in raw mappings.

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
        temporal_uncertainty_s=float(attributes["temporal_uncertainty_s"]),
        variants=tuple(
            Variant(
                text=str(variant["text"]),
                sources=tuple(str(source) for source in variant["sources"]),
                share=float(variant["share"]),
            )
            for variant in attributes["variants"]
        ),
        agreement=float(attributes["agreement"]),
    )


def rebracket(word: ConsensusWord, *, onomatopoeic: set[str], n_sources: int) -> Rebracketed:
    """Read one already-aligned column again under a vocabulary, aligning nothing.

    A member's group key is ``normalise_token`` of its display, and a display differs from its raw
    token only by brackets and edge punctuation, both of which ``normalise_token`` drops. The key is
    therefore the same under every vocabulary, and so are the column's membership, its ``outcome``
    and its ``agreement``; what the vocabulary decides is each member's display, and through it the
    column's surface, its ``bracketed`` flag and its variants' surfaces.

    Args:
        word: The stored position, from :func:`word_from_attributes`.
        onomatopoeic: The ``words.onomatopoeic_tokens`` vocabulary, each entry a
            :func:`vocabulary_key`.
        n_sources: How many recognizers the stream was aligned over, from the consensus provenance.

    Returns:
        The re-read position and its bracket-override count.

    Raises:
        ValueError: If the re-read column's outcome or agreement differs from the stored one. The
            column was then not rebuilt from the readings it was built from, and nothing is written.
    """
    members = [
        _Member(
            source=source,
            raw=word.readings[source],
            display=bracketed_form(word.readings[source], onomatopoeic) or word.readings[source],
            key=normalise_token(bracketed_form(word.readings[source], onomatopoeic) or word.readings[source]),
            start=word.timings[source][0],
            end=word.timings[source][1],
        )
        for source in word.sources
    ]
    text, outcome, variants, agreement, overrides = _column_word(members, n_sources)
    if outcome != word.outcome or abs(agreement - word.agreement) > _AGREEMENT_TOLERANCE:
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
        strong: The marks wrapped around an agreement word. When both are empty the result is the
            plain transcript: no mark, and a variant contributes its ``text`` alone rather than
            every reading joined by ``/``.

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
