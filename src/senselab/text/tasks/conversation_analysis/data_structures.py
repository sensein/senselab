"""Typed results of conversation analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type

import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field

from senselab.text.tasks.conversation_analysis.config import ConversationAnalysisConfig


class Word(BaseModel):
    """One timed, speaker-attributed word.

    Attributes:
        id: Position in time order (0-based).
        speaker: Opaque speaker label.
        start: Onset in seconds.
        end: Offset in seconds (``end >= start``; zero-duration words are allowed).
        text: The word exactly as given, fillers and brackets included.
        token: Normalised form (lower case, edge punctuation stripped; brackets and cut-off hyphen kept).
        confidence: Recogniser confidence, when given.
    """

    id: int
    speaker: str
    start: float
    end: float
    text: str
    token: str
    confidence: Optional[float] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """``end - start`` in seconds."""
        return self.end - self.start


class IPU(BaseModel):
    """An inter-pausal unit: one speaker's words with no internal silence of ``ipu_gap_s`` or more.

    Attributes:
        id: Position in (start, speaker) order.
        speaker: Speaker label.
        start: Onset of the first word.
        end: Latest offset of its words.
        word_ids: Ids of its words, in time order.
        text: Its words joined by spaces.
        n_words: Number of words.
        is_backchannel: Whether the IPU is a backchannel.
        backchannel_type: ``"overlapping"`` (overlaps another speaker's IPU) or ``"in_pause"``.
    """

    id: int
    speaker: str
    start: float
    end: float
    word_ids: List[int]
    text: str
    n_words: int
    is_backchannel: bool = False
    backchannel_type: Optional[Literal["overlapping", "in_pause"]] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """``end - start`` in seconds."""
        return self.end - self.start


class Turn(BaseModel):
    """A run of one speaker's non-backchannel IPUs not separated by another speaker taking the floor.

    Attributes:
        id: Position in start order.
        speaker: Speaker label.
        start: Onset of its first IPU.
        end: Latest offset of its IPUs.
        ipu_ids: Ids of its IPUs.
        word_ids: Ids of its words, in time order.
        words: Its words as given, in time order.
        text: Its words joined by spaces.
        n_words: Number of words.
        embedded: True when the turn lies wholly inside another speaker's turn and so never held the
            floor (an unsuccessful interruption or other non-backchannel overlap).
        dialogue_act: Primary dialogue act of the turn, once labelled.
    """

    id: int
    speaker: str
    start: float
    end: float
    ipu_ids: List[int]
    word_ids: List[int]
    words: List[str]
    text: str
    n_words: int
    embedded: bool = False
    dialogue_act: Optional[str] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """``end - start`` in seconds."""
        return self.end - self.start


class FloorTransfer(BaseModel):
    """A change of floor between two consecutive floor-holding turns.

    Attributes:
        id: Position in time order.
        from_turn_id: Turn that held the floor.
        to_turn_id: Turn that took it.
        from_speaker: Speaker yielding the floor.
        to_speaker: Speaker taking it.
        fto: Floor-transfer offset: ``to.start - from.end`` (negative = overlap).
        transfer_type: ``"gap"``, ``"overlap"`` or ``"no_gap_no_overlap"``.
        after_question: Whether the yielding turn ended with a question act.
    """

    id: int
    from_turn_id: int
    to_turn_id: int
    from_speaker: str
    to_speaker: str
    fto: float
    transfer_type: Literal["gap", "overlap", "no_gap_no_overlap"]
    after_question: bool = False


class Overlap(BaseModel):
    """A stretch in which two speakers' IPUs are both running.

    Attributes:
        id: Position in time order.
        start: Overlap onset.
        end: Overlap offset.
        holder_speaker: Speaker whose IPU started first.
        overlapper_speaker: Speaker who started second.
        holder_ipu_id: Holder's IPU.
        overlapper_ipu_id: Overlapper's IPU.
        overlap_type: ``"backchannel"`` (either IPU is a backchannel), ``"non_competitive"`` (onset in
            the terminal window of the holder's IPU) or ``"competitive"`` (an interruption).
        interruption_outcome: For competitive overlaps, ``"successful"`` or ``"unsuccessful"``.
    """

    id: int
    start: float
    end: float
    holder_speaker: str
    overlapper_speaker: str
    holder_ipu_id: int
    overlapper_ipu_id: int
    overlap_type: Literal["backchannel", "non_competitive", "competitive"]
    interruption_outcome: Optional[Literal["successful", "unsuccessful"]] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """``end - start`` in seconds."""
        return self.end - self.start


DisfluencyType = Literal["filled_pause", "repetition", "cutoff", "prolongation"]


class Disfluency(BaseModel):
    """One disfluency event.

    Attributes:
        id: Position in time order.
        speaker: Speaker label.
        disfluency_type: ``"filled_pause"``, ``"repetition"``, ``"cutoff"`` or ``"prolongation"``.
        word_ids: Words making up the event (both copies for a repetition).
        text: Those words as given.
        start: Onset of the first word.
        end: Offset of the last word.
        turn_id: Turn containing the event.
        filler_family: ``"uh"``, ``"um"`` or ``"other"`` for filled pauses.
        position: Filled pauses only: ``"turn_initial"``, ``"turn_medial"``, ``"turn_final"`` or
            ``"turn_isolated"`` (the turn's only word).
        pause_before: Filled pauses only: silence since the speaker's previous word in the turn.
        pause_after: Filled pauses only: silence until the speaker's next word in the turn.
        pre_pause: Filled pauses only: ``pause_after >= silent_pause_min_s``.
        post_pause: Filled pauses only: ``pause_before >= silent_pause_min_s``.
        next_event: Filled pauses only: ``"pause"``, ``"word"``, ``"speaker_change"`` or ``"end"``.
        zscore: Prolongations only: per-character duration z-score.
    """

    id: int
    speaker: str
    disfluency_type: DisfluencyType
    word_ids: List[int]
    text: str
    start: float
    end: float
    turn_id: int
    filler_family: Optional[Literal["uh", "um", "other"]] = None
    position: Optional[Literal["turn_initial", "turn_medial", "turn_final", "turn_isolated"]] = None
    pause_before: Optional[float] = None
    pause_after: Optional[float] = None
    pre_pause: Optional[bool] = None
    post_pause: Optional[bool] = None
    next_event: Optional[Literal["pause", "word", "speaker_change", "end"]] = None
    zscore: Optional[float] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """``end - start`` in seconds."""
        return self.end - self.start


class DialogueAct(str, Enum):
    """Coarse dialogue-act classes of the rule layer."""

    WH_QUESTION = "wh_question"
    YES_NO_QUESTION = "yes_no_question"
    DECLARATIVE_QUESTION = "declarative_question"
    DIRECTIVE = "directive"
    BACKCHANNEL = "backchannel"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    THANKING = "thanking"
    APOLOGY = "apology"
    GREETING_CLOSING = "greeting_closing"
    STATEMENT = "statement"


QUESTION_ACTS = frozenset(
    {DialogueAct.WH_QUESTION.value, DialogueAct.YES_NO_QUESTION.value, DialogueAct.DECLARATIVE_QUESTION.value}
)
"""Acts that count as questions."""


class DialogueActSegment(BaseModel):
    """A sentence-like segment of a turn, or a whole backchannel IPU, with its dialogue act.

    Attributes:
        id: Position in time order.
        speaker: Speaker label.
        start: Onset of its first word.
        end: Offset of its last word.
        text: Its words joined by spaces.
        word_ids: Its words.
        turn_id: Containing turn; ``None`` for a backchannel IPU.
        ipu_id: The backchannel IPU, when the segment is one.
        segment_index: Position within the turn.
        act: Dialogue act, once labelled.
        confidence: Labeller's confidence in ``[0, 1]`` (rule strength for the rule layer).
        source: Which labeller produced ``act``.
    """

    id: int
    speaker: str
    start: float
    end: float
    text: str
    word_ids: List[int]
    turn_id: Optional[int] = None
    ipu_id: Optional[int] = None
    segment_index: int = 0
    act: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class DialogueActLabel(BaseModel):
    """One labeller output.

    Attributes:
        act: The act (a :class:`DialogueAct` value for the rule layer; any string for other labellers).
        confidence: Confidence in ``[0, 1]``.
        source: Labeller name.
    """

    act: str
    confidence: float
    source: str


class TurnMarkers(BaseModel):
    """Engagement, politeness and alignment markers of one turn.

    Attributes:
        turn_id: The turn.
        speaker: Speaker label.
        n_words: Words in the turn.
        discourse_markers: Count of discourse-marker occurrences.
        hedges: Count of hedge occurrences.
        politeness: Per-strategy counts (Danescu-Niculescu-Mizil et al. 2013 lexical subset).
        pronoun_i: First-person-singular pronoun count.
        pronoun_we: First-person-plural pronoun count.
        pronoun_you: Second-person pronoun count.
        n_content_tokens: Distinct content tokens in the turn.
        lexical_alignment: Fraction of the turn's distinct content tokens that occur in the previous
            turn by another speaker; ``None`` without a previous turn or content tokens.
    """

    turn_id: int
    speaker: str
    n_words: int
    discourse_markers: int
    hedges: int
    politeness: Dict[str, int]
    pronoun_i: int
    pronoun_we: int
    pronoun_you: int
    n_content_tokens: int
    lexical_alignment: Optional[float] = None


class EntrainmentResult(BaseModel):
    """Entrainment of one feature, pooled (``target_speaker="all"``) or for one responding speaker.

    Attributes:
        feature: Feature name.
        target_speaker: The responding speaker, or ``"all"``.
        n_pairs: Adjacent speaker-change pairs used.
        adjacent_diff_mean: Mean ``|x_t - y_{t-1}|`` over adjacent pairs.
        nonadjacent_diff_mean: Mean over pairs of the mean ``|x_t - y_j|`` across the partner's
            non-adjacent turns.
        proximity: ``nonadjacent_diff_mean - adjacent_diff_mean`` over pairs with a non-adjacent
            baseline (positive = closer to the partner's adjacent turn than to their others).
        proximity_surrogate_mean: Mean proximity over shuffled pairings.
        proximity_p: One-sided surrogate p-value (surrogate >= observed).
        convergence_r: Pearson r of adjacent ``|diff|`` with time (negative = converging).
        convergence_surrogate_mean: Mean r over shuffled time orders.
        convergence_p: Two-sided surrogate p-value on ``|r|``.
        synchrony_r: Pearson r of ``x_t`` with ``y_{t-1}`` over adjacent pairs.
        synchrony_surrogate_mean: Mean r over shuffled pairings.
        synchrony_p: Two-sided surrogate p-value on ``|r|``.
    """

    feature: str
    target_speaker: str
    n_pairs: int
    adjacent_diff_mean: Optional[float] = None
    nonadjacent_diff_mean: Optional[float] = None
    proximity: Optional[float] = None
    proximity_surrogate_mean: Optional[float] = None
    proximity_p: Optional[float] = None
    convergence_r: Optional[float] = None
    convergence_surrogate_mean: Optional[float] = None
    convergence_p: Optional[float] = None
    synchrony_r: Optional[float] = None
    synchrony_surrogate_mean: Optional[float] = None
    synchrony_p: Optional[float] = None


def models_to_frame(items: List[Any], model: Type[BaseModel]) -> pd.DataFrame:  # noqa: ANN401
    """One row per model, columns in field order, empty frame with the same columns when no items.

    Args:
        items: Model instances.
        model: Their class (for the empty case's columns).

    Returns:
        The frame.
    """
    columns = list(model.model_fields) + list(model.model_computed_fields)
    if not items:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([item.model_dump(mode="json") for item in items], columns=columns)


class ConversationAnalysis(BaseModel):
    """Everything :func:`analyze_conversation` measures over one conversation.

    Attributes:
        speakers: Speaker labels in order of first appearance.
        start: Onset of the first word.
        end: Offset of the last word.
        words: Normalised words in time order.
        ipus: Inter-pausal units.
        turns: Turns.
        floor_transfers: Floor transfers between floor-holding turns.
        overlaps: Overlap episodes.
        disfluencies: Disfluency events.
        dialogue_acts: Dialogue-act segments.
        markers: Per-turn engagement/politeness/alignment markers.
        speaker_summaries: Per-speaker summary, keyed by speaker.
        entrainment: Entrainment results, when per-turn features were given.
        config: The configuration used.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    speakers: List[str]
    start: float
    end: float
    words: List[Word]
    ipus: List[IPU]
    turns: List[Turn]
    floor_transfers: List[FloorTransfer]
    overlaps: List[Overlap]
    disfluencies: List[Disfluency]
    dialogue_acts: List[DialogueActSegment]
    markers: List[TurnMarkers]
    speaker_summaries: Dict[str, Dict[str, Any]]
    entrainment: Optional[List[EntrainmentResult]] = None
    config: ConversationAnalysisConfig

    @property
    def duration(self) -> float:
        """Conversation span in seconds (``end - start``)."""
        return self.end - self.start

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Per-speaker summary.

        Returns:
            ``{speaker: {metric: value}}``; undefined metrics are ``None``.
        """
        return {speaker: dict(values) for speaker, values in self.speaker_summaries.items()}

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Tables of every result list.

        Returns:
            Frames keyed ``words``, ``ipus``, ``turns``, ``overlaps``, ``floor_transfers``,
            ``disfluencies``, ``dialogue_acts``, ``markers``, ``summary`` and, when computed,
            ``entrainment``. ``markers`` flattens politeness counts into ``politeness_<strategy>``.
        """
        markers = models_to_frame(self.markers, TurnMarkers)
        if not markers.empty:
            politeness = pd.DataFrame(list(markers.pop("politeness"))).add_prefix("politeness_")
            markers = pd.concat([markers.reset_index(drop=True), politeness], axis=1)
        frames = {
            "words": models_to_frame(self.words, Word),
            "ipus": models_to_frame(self.ipus, IPU),
            "turns": models_to_frame(self.turns, Turn),
            "overlaps": models_to_frame(self.overlaps, Overlap),
            "floor_transfers": models_to_frame(self.floor_transfers, FloorTransfer),
            "disfluencies": models_to_frame(self.disfluencies, Disfluency),
            "dialogue_acts": models_to_frame(self.dialogue_acts, DialogueActSegment),
            "markers": markers,
            "summary": pd.DataFrame.from_dict(self.speaker_summaries, orient="index").rename_axis("speaker"),
        }
        if self.entrainment is not None:
            frames["entrainment"] = models_to_frame(self.entrainment, EntrainmentResult)
        return frames
