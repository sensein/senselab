"""Public entry points: :func:`analyze_conversation` and :func:`turn_entrainment`."""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from senselab.text.tasks.conversation_analysis.config import ConversationAnalysisConfig
from senselab.text.tasks.conversation_analysis.data_structures import (
    QUESTION_ACTS,
    ConversationAnalysis,
    DialogueActSegment,
    EntrainmentResult,
    Turn,
)
from senselab.text.tasks.conversation_analysis.dialogue_acts import (
    DialogueActLabeller,
    RuleBasedLabeller,
    label_dialogue_acts,
    primary_act,
    segment_dialogue_acts,
)
from senselab.text.tasks.conversation_analysis.disfluency import detect_disfluencies
from senselab.text.tasks.conversation_analysis.entrainment import compute_entrainment
from senselab.text.tasks.conversation_analysis.markers import turn_markers
from senselab.text.tasks.conversation_analysis.segmentation import (
    WordInput,
    build_ipus,
    build_turns,
    mark_backchannels,
    normalize_words,
)
from senselab.text.tasks.conversation_analysis.summary import speaker_summaries
from senselab.text.tasks.conversation_analysis.turn_taking import floor_transfers, overlaps
from senselab.utils.data_structures import ScriptLine


def analyze_conversation(
    words: Sequence[WordInput],
    diarization: Optional[Sequence[ScriptLine]] = None,
    config: Optional[ConversationAnalysisConfig] = None,
    *,
    labeller: Optional[DialogueActLabeller] = None,
    turn_features: Optional[pd.DataFrame] = None,
    entrainment_surrogates: int = 1000,
    seed: int = 0,
) -> ConversationAnalysis:
    """Turn-taking, disfluency, dialogue-act, marker and entrainment analysis of one conversation.

    Args:
        words: Word-level transcript: ``ScriptLine`` words (or lines whose ``chunks`` are words) or
            mappings with ``speaker``, ``start``, ``end``, ``text`` and optional ``confidence``.
            Any order; speakers are opaque labels; any number of speakers.
        diarization: Speaker segments; used to attribute words without a speaker and reported as
            ``diarized_speaking_time_s``.
        config: Thresholds and lexicons; packaged defaults when omitted.
        labeller: Dialogue-act labeller; the rule layer when omitted.
        turn_features: Per-turn numeric features (e.g. mean f0) with a ``turn_id`` column matching
            :attr:`Turn.id`; entrainment is computed over floor-holding turns when given.
        entrainment_surrogates: Surrogate shuffles for the entrainment baseline.
        seed: Seed for the entrainment surrogates.

    Returns:
        The analysis; see :class:`ConversationAnalysis`.

    Raises:
        ValueError: Invalid words (see :func:`normalize_words`) or ``turn_features`` without ``turn_id``.
    """
    config = config or ConversationAnalysisConfig()
    norm = normalize_words(words, diarization)
    speakers: List[str] = list(dict.fromkeys(w.speaker for w in norm))
    start = min((w.start for w in norm), default=0.0)
    end = max((w.end for w in norm), default=0.0)

    ipus = mark_backchannels(build_ipus(norm, config), norm, config)
    turns = build_turns(ipus, norm)
    transfers = floor_transfers(turns, config)
    episodes = overlaps(ipus, config)

    segments = label_dialogue_acts(segment_dialogue_acts(turns, ipus, norm), labeller or RuleBasedLabeller(config))
    segments_by_turn: dict[int, List[DialogueActSegment]] = {}
    last_act: dict[int, str] = {}
    for seg in sorted(segments, key=lambda g: (g.turn_id or 0, g.segment_index)):
        if seg.turn_id is not None and seg.act is not None:
            segments_by_turn.setdefault(seg.turn_id, []).append(seg)
            last_act[seg.turn_id] = seg.act
    turns = [t.model_copy(update={"dialogue_act": primary_act(segments_by_turn.get(t.id, []), config)}) for t in turns]
    transfers = [
        f.model_copy(update={"after_question": last_act.get(f.from_turn_id) in QUESTION_ACTS}) for f in transfers
    ]

    disfluencies = detect_disfluencies(turns, norm, config)
    markers = turn_markers(turns, segments, config)
    summaries = speaker_summaries(
        speakers, end - start, norm, ipus, turns, transfers, episodes, disfluencies, segments, markers, diarization
    )

    entrainment: Optional[List[EntrainmentResult]] = None
    if turn_features is not None:
        entrainment = turn_entrainment(turns, turn_features, n_surrogates=entrainment_surrogates, seed=seed)

    return ConversationAnalysis(
        speakers=speakers,
        start=start,
        end=end,
        words=norm,
        ipus=ipus,
        turns=turns,
        floor_transfers=transfers,
        overlaps=episodes,
        disfluencies=disfluencies,
        dialogue_acts=segments,
        markers=markers,
        speaker_summaries=summaries,
        entrainment=entrainment,
        config=config,
    )


def turn_entrainment(
    turns: Sequence[Turn],
    features: pd.DataFrame,
    *,
    n_surrogates: int = 1000,
    seed: int = 0,
) -> List[EntrainmentResult]:
    """Entrainment of per-turn features over the floor-holding turns of a conversation.

    Embedded turns are dropped; each remaining turn's speaker and start come from ``turns`` and its
    numeric features from the ``features`` row with the same ``turn_id``. The series then go through
    :func:`compute_entrainment`.

    Args:
        turns: Turns, e.g. :attr:`ConversationAnalysis.turns`.
        features: Per-turn numeric features with a ``turn_id`` column matching :attr:`Turn.id`; any
            ``speaker``/``start`` columns are ignored in favour of the turns'. Turns without a row get
            missing values.
        n_surrogates: Surrogate shuffles for the baseline.
        seed: Seed for the surrogates.

    Returns:
        One result per numeric feature column.

    Raises:
        ValueError: ``features`` has no ``turn_id`` column.
    """
    if "turn_id" not in features.columns:
        raise ValueError("features needs a 'turn_id' column")
    floor = pd.DataFrame(
        [{"turn_id": t.id, "speaker": t.speaker, "start": t.start} for t in turns if not t.embedded],
        columns=["turn_id", "speaker", "start"],
    )
    columns = [c for c in features.select_dtypes("number").columns if c not in ("turn_id", "speaker", "start")]
    merged = floor.merge(features[["turn_id", *columns]], on="turn_id", how="left")
    return compute_entrainment(merged, columns, n_surrogates=n_surrogates, seed=seed)
