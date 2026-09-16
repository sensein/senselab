"""The consensus word stream aligned against what the recording was declared to expect.

One alignment, three projections: per expected token its realisation and extent, the boundaries of
the expected structure, and the lexical complement -- words that matched no expectation. The design
and its measurements are in
``specs/20260817-triage-workflow-dag/stimulus-alignment.md``; the derivative is D1 of
``preprocess-derivatives-for-expected-patterns.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from senselab.audio.data_structures.audio_hints import ExpectedSpeech
from senselab.audio.workflows.audio_analysis.harmonize import align_pair, normalise_token

__all__ = [
    "ALGORITHM",
    "NORMALISATION",
    "ROUTINE",
    "ExpectedToken",
    "LexicalWord",
    "Realisation",
    "StimulusAlignment",
    "StructureUnit",
    "TokenRun",
    "UnexpectedWord",
    "align_stimulus",
    "split_prompts",
]

ALGORITHM = "weighted_levenshtein_alignment"
ROUTINE = "senselab.audio.workflows.triage.stimulus.align_stimulus"
NORMALISATION = "casefold; keep alphanumerics and apostrophe"

Realisation = Literal["realised", "substituted", "absent"]


@dataclass(frozen=True)
class LexicalWord:
    """One consensus word as the alignment reads it.

    Attributes:
        index: The word's position in the consensus stream; the only order a reader may use.
        text: The label surface.
        extent: The word's derived ``(onset_s, offset_s)``, or None when it carries none.
        agreement: The consensus word's agreement share.
    """

    index: int
    text: str
    extent: tuple[float, float] | None
    agreement: float


@dataclass(frozen=True)
class ExpectedToken:
    """One token of the declared utterance, and what realised it.

    Attributes:
        index: The token's position in the whole expected stream.
        unit: Which :class:`StructureUnit` it belongs to.
        text: The token verbatim, as the prompt spelled it.
        key: ``text`` under :data:`NORMALISATION`.
        realisation: ``realised`` when a consensus word carried the same key, ``substituted`` when
            one was aligned to it under a different key, ``absent`` when none was aligned at all.
        word_index: The consensus word's index, or None when ``realisation`` is ``absent``.
        read: The consensus word's surface, or None when ``realisation`` is ``absent``.
        extent: The consensus word's extent, or None when nothing realised the token or the word
            carried no extent.
        agreement: The consensus word's agreement share, or None when nothing realised the token.
    """

    index: int
    unit: int
    text: str
    key: str
    realisation: Realisation
    word_index: int | None
    read: str | None
    extent: tuple[float, float] | None
    agreement: float | None


@dataclass(frozen=True)
class UnexpectedWord:
    """One consensus word the alignment paired with no expected token.

    Attributes:
        index: The word's position in the consensus stream.
        text: The label surface.
        extent: The word's derived extent, or None.
        agreement: The consensus word's agreement share.
        after: The expected-token index this word follows, ``-1`` before the first.
    """

    index: int
    text: str
    extent: tuple[float, float] | None
    agreement: float
    after: int


@dataclass(frozen=True)
class StructureUnit:
    """One declared unit of the expectation -- a prompt, or a sentence inside one.

    Attributes:
        index: The unit's position in the expected stream.
        prompt: Which ``expected_speech`` entry the unit came from.
        text: The unit verbatim.
        token_indices: The expected-token indices the unit covers.
        n_realised: Tokens of the unit whose realisation is ``realised``.
        n_substituted: Tokens of the unit whose realisation is ``substituted``.
        extent: The hull of every extent-carrying token aligned to the unit, or None when none was.
    """

    index: int
    prompt: int
    text: str
    token_indices: tuple[int, ...]
    n_realised: int
    n_substituted: int
    extent: tuple[float, float] | None


@dataclass(frozen=True)
class TokenRun:
    """A contiguous run of expected tokens, and where it was realised.

    Attributes:
        token_indices: The expected-token indices the run covers, in order.
        extent: The hull of the run's extent-carrying tokens, or None when none was realised.
        substitutions: The run's ``substituted`` tokens, in order.
        omissions: The run's ``absent`` tokens, in order.
    """

    token_indices: tuple[int, ...]
    extent: tuple[float, float] | None
    substitutions: tuple[ExpectedToken, ...]
    omissions: tuple[ExpectedToken, ...]


def _hull(extents: Sequence[tuple[float, float] | None]) -> tuple[float, float] | None:
    """The hull of the extents that are not None, or None when none is."""
    present = [extent for extent in extents if extent is not None]
    if not present:
        return None
    return min(start for start, _ in present), max(end for _, end in present)


@dataclass(frozen=True)
class StimulusAlignment:
    """One alignment of the consensus word stream against the declared utterance.

    Attributes:
        expected: One record per expected token, in order.
        units: The declared structure, in order.
        unexpected: The lexical complement -- consensus words matching no expected token, in
            stream order.
        provenance: The algorithm, the normalisation, and the counts the alignment produced.
    """

    expected: tuple[ExpectedToken, ...]
    units: tuple[StructureUnit, ...]
    unexpected: tuple[UnexpectedWord, ...]
    provenance: dict[str, Any]

    @property
    def substitutions(self) -> tuple[ExpectedToken, ...]:
        """Every expected token a consensus word was aligned to under a different key."""
        return tuple(token for token in self.expected if token.realisation == "substituted")

    @property
    def omissions(self) -> tuple[ExpectedToken, ...]:
        """Every expected token no consensus word was aligned to."""
        return tuple(token for token in self.expected if token.realisation == "absent")

    def structure_spans(self) -> tuple[StructureUnit, ...]:
        """The declared units that were realised somewhere, i.e. those carrying an extent."""
        return tuple(unit for unit in self.units if unit.extent is not None)

    def run_for(self, keys: Sequence[str]) -> TokenRun | None:
        """The first contiguous run of expected tokens whose keys are ``keys``, in order.

        Args:
            keys: The token keys to find, each already under :data:`NORMALISATION`.

        Returns:
            The run, or None when the expected stream does not contain it.
        """
        wanted = [str(key) for key in keys]
        if not wanted:
            return None
        stream = [token.key for token in self.expected]
        for start in range(len(stream) - len(wanted) + 1):
            if stream[start : start + len(wanted)] != wanted:
                continue
            run = self.expected[start : start + len(wanted)]
            return TokenRun(
                token_indices=tuple(token.index for token in run),
                extent=_hull([token.extent for token in run]),
                substitutions=tuple(token for token in run if token.realisation == "substituted"),
                omissions=tuple(token for token in run if token.realisation == "absent"),
            )
        return None


def split_prompts(prompts: Sequence[ExpectedSpeech], *, terminators: str) -> list[tuple[int, str, list[str]]]:
    """Split the declared prompts into structure units and their verbatim tokens.

    A prompt is one unit unless it carries a sentence terminator, in which case each sentence is a
    unit; a caller who declares six sentences as six entries and a caller who declares them as one
    string therefore get the same six units.

    Args:
        prompts: The ``expected_speech`` entries, in declared order.
        terminators: The characters that close a unit inside one prompt.

    Returns:
        ``(prompt index, unit text, tokens)`` per unit, in order. A unit whose tokens all normalise
        to the empty key is dropped; a prompt of only such tokens contributes no unit.
    """
    units: list[tuple[int, str, list[str]]] = []
    for prompt_index, prompt in enumerate(prompts):
        text = (prompt.text or "").strip()
        if not text:
            continue
        current: list[str] = []
        for token in text.split():
            current.append(token)
            if token and token[-1] in terminators:
                units.append((prompt_index, " ".join(current), list(current)))
                current = []
        if current:
            units.append((prompt_index, " ".join(current), list(current)))
    return [
        (prompt_index, unit_text, tokens)
        for prompt_index, unit_text, tokens in units
        if any(normalise_token(token) for token in tokens)
    ]


def align_stimulus(
    prompts: Sequence[ExpectedSpeech], words: Sequence[LexicalWord], *, terminators: str
) -> StimulusAlignment:
    """Align the consensus word stream against the declared utterance.

    Args:
        prompts: The ``expected_speech`` entries, in declared order. May be empty, which is a
            declared expectation of no particular words: every lexical word is then unexpected.
        words: The lexical consensus words, in stream order.
        terminators: The characters that close a structure unit inside one prompt.

    Returns:
        The alignment and its provenance.
    """
    units_source = split_prompts(prompts, terminators=terminators)
    expected_keys: list[str] = []
    expected_text: list[str] = []
    expected_unit: list[int] = []
    unit_tokens: list[list[int]] = []
    for unit_index, (_, _, tokens) in enumerate(units_source):
        covered: list[int] = []
        for token in tokens:
            key = normalise_token(token)
            if not key:
                continue
            covered.append(len(expected_keys))
            expected_keys.append(key)
            expected_text.append(token)
            expected_unit.append(unit_index)
        unit_tokens.append(covered)

    word_keys = [normalise_token(word.text) for word in words]
    path = align_pair(expected_keys, word_keys)

    paired: dict[int, int] = {}
    unexpected: list[UnexpectedWord] = []
    last_expected = -1
    for expected_index, word_index in path:
        if expected_index is not None and word_index is not None:
            paired[expected_index] = word_index
            last_expected = expected_index
        elif expected_index is not None:
            last_expected = expected_index
        elif word_index is not None:
            word = words[word_index]
            unexpected.append(
                UnexpectedWord(
                    index=word.index,
                    text=word.text,
                    extent=word.extent,
                    agreement=word.agreement,
                    after=last_expected,
                )
            )

    expected: list[ExpectedToken] = []
    for index, key in enumerate(expected_keys):
        word_index = paired.get(index)
        if word_index is None:
            expected.append(
                ExpectedToken(
                    index=index,
                    unit=expected_unit[index],
                    text=expected_text[index],
                    key=key,
                    realisation="absent",
                    word_index=None,
                    read=None,
                    extent=None,
                    agreement=None,
                )
            )
            continue
        word = words[word_index]
        expected.append(
            ExpectedToken(
                index=index,
                unit=expected_unit[index],
                text=expected_text[index],
                key=key,
                realisation="realised" if word_keys[word_index] == key else "substituted",
                word_index=word.index,
                read=word.text,
                extent=word.extent,
                agreement=word.agreement,
            )
        )

    units: list[StructureUnit] = []
    for unit_index, (prompt_index, unit_text, _) in enumerate(units_source):
        covered = unit_tokens[unit_index]
        units.append(
            StructureUnit(
                index=unit_index,
                prompt=prompt_index,
                text=unit_text,
                token_indices=tuple(covered),
                n_realised=sum(1 for i in covered if expected[i].realisation == "realised"),
                n_substituted=sum(1 for i in covered if expected[i].realisation == "substituted"),
                extent=_hull([expected[i].extent for i in covered]),
            )
        )

    realised = sum(1 for token in expected if token.realisation == "realised")
    substituted = sum(1 for token in expected if token.realisation == "substituted")
    provenance = {
        "algorithm": ALGORITHM,
        "routine": ROUTINE,
        "normalisation": NORMALISATION,
        "sentence_terminators": terminators,
        "n_prompts": len(prompts),
        "n_units": len(units),
        "n_expected": len(expected),
        "n_lexical_words": len(words),
        "n_realised": realised,
        "n_substituted": substituted,
        "n_absent": len(expected) - realised - substituted,
        "n_unexpected": len(unexpected),
        "realised_fraction": realised / len(expected) if expected else None,
    }
    return StimulusAlignment(
        expected=tuple(expected), units=tuple(units), unexpected=tuple(unexpected), provenance=provenance
    )
