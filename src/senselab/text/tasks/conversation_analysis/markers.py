"""Discourse markers, hedges, politeness strategies, pronouns and lexical alignment per turn."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from senselab.text.tasks.conversation_analysis.config import ConversationAnalysisConfig, Lexicons
from senselab.text.tasks.conversation_analysis.data_structures import DialogueActSegment, Turn, TurnMarkers
from senselab.text.tasks.conversation_analysis.text_utils import count_phrases, starts_with, tokens_of

POLITENESS_STRATEGIES = (
    "gratitude",
    "apologizing",
    "please",
    "please_start",
    "indirect_btw",
    "direct_question",
    "direct_start",
    "counterfactual_modal",
    "indicative_modal",
    "first_person_start",
    "first_person",
    "first_person_plural",
    "second_person",
    "second_person_start",
    "hedges",
    "factuality",
    "deference",
    "greeting",
    "positive_lexicon",
    "negative_lexicon",
)
"""Strategies counted per turn, after Danescu-Niculescu-Mizil et al. (2013)."""


def politeness_counts(segment_texts: Sequence[str], lexicons: Lexicons) -> Dict[str, int]:
    """Count politeness strategies over a turn's sentence-like segments.

    Start-anchored strategies (``please_start``, ``direct_question``, ``direct_start``,
    ``first_person_start``, ``second_person_start``, ``deference``, ``greeting``) count segments that
    begin with the cue; ``first_person`` and ``second_person`` count non-initial occurrences, as in the
    original feature set; the rest count occurrences anywhere.

    Args:
        segment_texts: Segment texts of one turn.
        lexicons: Lexicons.

    Returns:
        ``{strategy: count}`` for every strategy in :data:`POLITENESS_STRATEGIES`.
    """
    pol = lexicons.politeness
    counts = dict.fromkeys(POLITENESS_STRATEGIES, 0)
    for text in segment_texts:
        tokens = tokens_of(text)
        if not tokens:
            continue
        rest = tokens[1:]
        please = count_phrases(tokens, pol["please"])
        please_start = int(tokens[0] == "please")
        counts["gratitude"] += count_phrases(tokens, pol["gratitude"])
        counts["apologizing"] += count_phrases(tokens, pol["apologizing"])
        counts["please"] += please - please_start
        counts["please_start"] += please_start
        counts["indirect_btw"] += count_phrases(tokens, pol["indirect_btw"])
        counts["direct_question"] += int(starts_with(tokens, pol["direct_question_start"]))
        counts["direct_start"] += int(starts_with(tokens, pol["direct_start"]))
        counts["counterfactual_modal"] += count_phrases(tokens, pol["counterfactual_modal"])
        counts["indicative_modal"] += count_phrases(tokens, pol["indicative_modal"])
        counts["first_person_start"] += int(starts_with(tokens, pol["first_person_start"]))
        counts["first_person"] += count_phrases(rest, pol["first_person"])
        counts["first_person_plural"] += count_phrases(tokens, pol["first_person_plural"])
        counts["second_person"] += count_phrases(rest, pol["second_person"])
        counts["second_person_start"] += int(starts_with(tokens, pol["second_person_start"]))
        counts["hedges"] += count_phrases(tokens, lexicons.hedges)
        counts["factuality"] += count_phrases(tokens, pol["factuality"])
        counts["deference"] += int(starts_with(tokens, pol["deference_start"]))
        counts["greeting"] += int(starts_with(tokens, pol["greeting"]))
        counts["positive_lexicon"] += count_phrases(tokens, pol["positive_lexicon"])
        counts["negative_lexicon"] += count_phrases(tokens, pol["negative_lexicon"])
    return counts


def content_tokens(text: str, lexicons: Lexicons) -> set[str]:
    """Distinct lower-cased content tokens: not stopwords, not fillers, containing a letter.

    Args:
        text: Free text.
        lexicons: Lexicons.

    Returns:
        The token set.
    """
    return {
        t.rstrip("-")
        for t in tokens_of(text)
        if t not in lexicons.stopwords and t not in lexicons.fillers and any(c.isalpha() for c in t)
    }


def turn_markers(
    turns: Sequence[Turn],
    segments: Sequence[DialogueActSegment],
    config: ConversationAnalysisConfig,
) -> List[TurnMarkers]:
    """Markers for every turn.

    Lexical alignment compares a turn with the latest earlier-starting turn by another speaker.

    Args:
        turns: Turns.
        segments: Dialogue-act segments (their texts are the politeness units).
        config: Analysis configuration.

    Returns:
        One :class:`TurnMarkers` per turn, in turn order.
    """
    lex = config.lexicons
    seg_texts: Dict[int, List[str]] = {}
    for seg in segments:
        if seg.turn_id is not None:
            seg_texts.setdefault(seg.turn_id, []).append(seg.text)
    ordered = sorted(turns, key=lambda t: (t.start, t.id))
    out: List[TurnMarkers] = []
    for index, turn in enumerate(ordered):
        tokens = tokens_of(turn.text)
        content = content_tokens(turn.text, lex)
        previous: Optional[Turn] = next((t for t in reversed(ordered[:index]) if t.speaker != turn.speaker), None)
        alignment: Optional[float] = None
        if previous is not None and content:
            alignment = len(content & content_tokens(previous.text, lex)) / len(content)
        out.append(
            TurnMarkers(
                turn_id=turn.id,
                speaker=turn.speaker,
                n_words=turn.n_words,
                discourse_markers=count_phrases(tokens, lex.discourse_markers),
                hedges=count_phrases(tokens, lex.hedges),
                politeness=politeness_counts(seg_texts.get(turn.id, [turn.text]), lex),
                pronoun_i=sum(t in lex.pronouns["i"] for t in tokens),
                pronoun_we=sum(t in lex.pronouns["we"] for t in tokens),
                pronoun_you=sum(t in lex.pronouns["you"] for t in tokens),
                n_content_tokens=len(content),
                lexical_alignment=alignment,
            )
        )
    return sorted(out, key=lambda m: m.turn_id)
