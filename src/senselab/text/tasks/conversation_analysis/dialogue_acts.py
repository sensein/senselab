"""Dialogue-act segments and the rule-based labeller.

``label_dialogue_acts`` takes any :class:`DialogueActLabeller`; :class:`RuleBasedLabeller` is the default.
A model-based labeller (e.g. a local LLM) only has to map a sequence of segments to one
:class:`DialogueActLabel` per segment.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple

from senselab.text.tasks.conversation_analysis.config import ConversationAnalysisConfig, Lexicons
from senselab.text.tasks.conversation_analysis.data_structures import (
    IPU,
    QUESTION_ACTS,
    DialogueAct,
    DialogueActLabel,
    DialogueActSegment,
    Turn,
    Word,
)
from senselab.text.tasks.conversation_analysis.text_utils import (
    composed_of,
    contains_phrase,
    starts_with,
    strip_lead,
    terminal_punctuation,
    tokens_of,
)

ACT_PRIORITY: Tuple[str, ...] = (
    DialogueAct.WH_QUESTION.value,
    DialogueAct.YES_NO_QUESTION.value,
    DialogueAct.DECLARATIVE_QUESTION.value,
    DialogueAct.DIRECTIVE.value,
    DialogueAct.DISAGREEMENT.value,
    DialogueAct.AGREEMENT.value,
    DialogueAct.APOLOGY.value,
    DialogueAct.THANKING.value,
    DialogueAct.GREETING_CLOSING.value,
    DialogueAct.BACKCHANNEL.value,
    DialogueAct.STATEMENT.value,
)
"""Order in which a turn's segment acts compete for the turn's primary act (first wins)."""

SUBSTANTIVE_ACTS = frozenset(QUESTION_ACTS | {DialogueAct.DIRECTIVE.value, DialogueAct.STATEMENT.value})
"""Acts whose segment, with enough content words, makes the turn substantive."""

MINIMAL_RESPONSE_ACTS = frozenset({DialogueAct.BACKCHANNEL.value, DialogueAct.AGREEMENT.value})
"""Acts that cannot be the primary act of a substantive turn."""


class DialogueActLabeller(Protocol):
    """Anything that labels a sequence of segments, one label per segment, in order."""

    def __call__(self, segments: Sequence[DialogueActSegment]) -> Sequence[DialogueActLabel]:
        """Label segments.

        Args:
            segments: Segments in time order (context is available to the labeller).

        Returns:
            One label per segment.
        """
        ...


def segment_dialogue_acts(
    turns: Sequence[Turn], ipus: Sequence[IPU], words: Sequence[Word]
) -> List[DialogueActSegment]:
    """Split turns into sentence-like segments at words ending in ``.``, ``?`` or ``!``.

    A turn without terminal punctuation is one segment. Each backchannel IPU is one segment with
    ``turn_id=None``.

    Args:
        turns: Turns.
        ipus: IPUs with backchannels marked.
        words: Words.

    Returns:
        Unlabelled segments in time order.
    """
    drafts: List[DialogueActSegment] = []
    for turn in turns:
        current: List[int] = []
        index = 0
        for wid in turn.word_ids:
            current.append(wid)
            if terminal_punctuation(words[wid].text):
                drafts.append(_segment(current, words, turn_id=turn.id, segment_index=index))
                index += 1
                current = []
        if current:
            drafts.append(_segment(current, words, turn_id=turn.id, segment_index=index))
    for ipu in ipus:
        if ipu.is_backchannel:
            drafts.append(_segment(list(ipu.word_ids), words, ipu_id=ipu.id))
    drafts.sort(key=lambda s: (s.start, s.end, s.id))
    return [s.model_copy(update={"id": i}) for i, s in enumerate(drafts)]


def _segment(
    word_ids: List[int],
    words: Sequence[Word],
    turn_id: Optional[int] = None,
    ipu_id: Optional[int] = None,
    segment_index: int = 0,
) -> DialogueActSegment:
    return DialogueActSegment(
        id=0,
        speaker=words[word_ids[0]].speaker,
        start=min(words[w].start for w in word_ids),
        end=max(words[w].end for w in word_ids),
        text=" ".join(words[w].text for w in word_ids),
        word_ids=list(word_ids),
        turn_id=turn_id,
        ipu_id=ipu_id,
        segment_index=segment_index,
    )


def rule_act(text: str, lexicons: Lexicons, previous_other_act: Optional[str] = None) -> Tuple[str, float]:
    """Label one segment by lexical and punctuation rules.

    Args:
        text: Segment text.
        lexicons: Lexicons.
        previous_other_act: Act of the latest earlier segment by another speaker, used to read a bare
            "yes"/"yeah" after a question as agreement rather than a backchannel.

    Returns:
        ``(act, confidence)``; ``confidence`` is the rule's strength, not a calibrated probability.
    """
    every = tokens_of(text)
    tokens = [t for t in every if t not in lexicons.fillers]
    if not tokens:
        if every and composed_of(every, lexicons.backchannel):
            return DialogueAct.BACKCHANNEL.value, 0.5
        return DialogueAct.STATEMENT.value, 0.1
    punct = terminal_punctuation(text)
    head = strip_lead(tokens, lexicons.lead_skip)
    first = head[0] if head else ""
    asked = punct == "?"
    unpunctuated = punct is None

    if first in lexicons.wh_words and (asked or unpunctuated):
        return DialogueAct.WH_QUESTION.value, 0.9 if asked else 0.6
    if asked and any(t in lexicons.wh_words for t in tokens[1:]) and first not in lexicons.auxiliaries:
        return DialogueAct.WH_QUESTION.value, 0.7
    if starts_with(head, lexicons.directive_phrases) or (head[:1] == ["please"] and len(head) > 1):
        return DialogueAct.DIRECTIVE.value, 0.8
    if first in lexicons.auxiliaries and (asked or unpunctuated) and len(head) > 1:
        return DialogueAct.YES_NO_QUESTION.value, 0.9 if asked else 0.6
    if first in lexicons.imperative_verbs and not asked and len(head) > 1:
        return DialogueAct.DIRECTIVE.value, 0.7
    if "please" in tokens and len(tokens) > 2:
        return DialogueAct.DIRECTIVE.value, 0.7
    if asked:
        return DialogueAct.DECLARATIVE_QUESTION.value, 0.8
    if contains_phrase(tokens, lexicons.thanking):
        return DialogueAct.THANKING.value, 0.9
    if contains_phrase(tokens, lexicons.apology):
        return DialogueAct.APOLOGY.value, 0.9
    if starts_with(tokens, lexicons.greeting_start) or contains_phrase(tokens, lexicons.closing):
        return DialogueAct.GREETING_CLOSING.value, 0.85
    if starts_with(head or tokens, lexicons.disagreement_phrases):
        return DialogueAct.DISAGREEMENT.value, 0.7 if len(tokens) <= 3 else 0.6
    if len(tokens) <= 3 and composed_of(tokens, lexicons.backchannel):
        if previous_other_act in QUESTION_ACTS and composed_of(tokens, lexicons.agreement_tokens):
            return DialogueAct.AGREEMENT.value, 0.7
        if composed_of(tokens, lexicons.agreement_phrases):
            return DialogueAct.AGREEMENT.value, 0.6
        return DialogueAct.BACKCHANNEL.value, 0.7
    if starts_with(tokens, lexicons.agreement_phrases) or contains_phrase(
        tokens, {p for p in lexicons.agreement_phrases if len(p) > 1}
    ):
        return DialogueAct.AGREEMENT.value, 0.6
    return DialogueAct.STATEMENT.value, 0.3


class RuleBasedLabeller:
    """Default labeller: :func:`rule_act` per segment; backchannel IPUs are ``backchannel`` (0.9)."""

    name = "rules"

    def __init__(self, config: Optional[ConversationAnalysisConfig] = None) -> None:
        """Create the labeller.

        Args:
            config: Configuration whose lexicons are used; the packaged defaults when omitted.
        """
        self.lexicons = (config or ConversationAnalysisConfig()).lexicons

    def __call__(self, segments: Sequence[DialogueActSegment]) -> List[DialogueActLabel]:
        """Label segments in order, carrying the latest other-speaker act as context.

        Args:
            segments: Segments in time order.

        Returns:
            One label per segment.
        """
        labels: List[DialogueActLabel] = []
        last_act: dict[str, str] = {}
        for seg in segments:
            if seg.ipu_id is not None:
                act, confidence = DialogueAct.BACKCHANNEL.value, 0.9
            else:
                previous = None
                others = [(s, a) for s, a in last_act.items() if s != seg.speaker]
                if others:
                    previous = others[-1][1]
                act, confidence = rule_act(seg.text, self.lexicons, previous)
            last_act.pop(seg.speaker, None)
            last_act[seg.speaker] = act
            labels.append(DialogueActLabel(act=act, confidence=confidence, source=self.name))
        return labels


def label_dialogue_acts(
    segments: Sequence[DialogueActSegment],
    labeller: Optional[DialogueActLabeller] = None,
) -> List[DialogueActSegment]:
    """Attach a dialogue act, confidence and source to each segment.

    Args:
        segments: Segments from :func:`segment_dialogue_acts`.
        labeller: Any :class:`DialogueActLabeller`; :class:`RuleBasedLabeller` when omitted.

    Returns:
        Labelled copies of the segments.

    Raises:
        ValueError: The labeller returned a different number of labels than segments.
    """
    labeller = labeller or RuleBasedLabeller()
    labels = list(labeller(segments))
    if len(labels) != len(segments):
        raise ValueError(f"labeller returned {len(labels)} labels for {len(segments)} segments")
    return [
        seg.model_copy(update={"act": lab.act, "confidence": lab.confidence, "source": lab.source})
        for seg, lab in zip(segments, labels)
    ]


def content_words(text: str, lexicons: Lexicons) -> int:
    """Number of tokens in ``text`` that are neither stopwords nor fillers.

    Args:
        text: Free text.
        lexicons: Lexicons supplying ``stopwords`` and ``fillers``.

    Returns:
        The content-word count.
    """
    excluded = lexicons.stopwords | lexicons.fillers
    return sum(1 for t in tokens_of(text) if t not in excluded)


def primary_act(
    segments: Sequence[DialogueActSegment], config: Optional[ConversationAnalysisConfig] = None
) -> Optional[str]:
    """A turn's primary act from its labelled segments.

    Precedence:

    1. A turn is *substantive* when any segment's act is in :data:`SUBSTANTIVE_ACTS` (a question,
       directive or statement) and its text holds at least ``config.substantive_min_content_words``
       content words (:func:`content_words`).
    2. In a substantive turn the acts in :data:`MINIMAL_RESPONSE_ACTS` (backchannel, agreement) are
       not candidates, so "Sure. I work as a nurse." is a statement.
    3. The remaining candidates are ranked by :data:`ACT_PRIORITY`; acts outside it rank after it,
       first seen wins.

    Args:
        segments: The turn's labelled segments, in order; unlabelled segments are ignored.
        config: Thresholds and lexicons; packaged defaults when omitted.

    Returns:
        The primary act; ``None`` when no segment carries an act.
    """
    labelled = [s for s in segments if s.act is not None]
    if not labelled:
        return None
    config = config or ConversationAnalysisConfig()
    substantive = any(
        s.act in SUBSTANTIVE_ACTS and content_words(s.text, config.lexicons) >= config.substantive_min_content_words
        for s in labelled
    )
    acts = [str(s.act) for s in labelled]
    candidates = [a for a in acts if not (substantive and a in MINIMAL_RESPONSE_ACTS)] or acts
    rank = {act: i for i, act in enumerate(ACT_PRIORITY)}
    return min(candidates, key=lambda a: rank.get(a, len(rank)))
