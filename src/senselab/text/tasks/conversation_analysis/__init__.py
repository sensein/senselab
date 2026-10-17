""".. include:: ./doc.md"""  # noqa: D415

from senselab.text.tasks.conversation_analysis.api import analyze_conversation
from senselab.text.tasks.conversation_analysis.config import ConversationAnalysisConfig, Lexicons, load_default_lexicons
from senselab.text.tasks.conversation_analysis.data_structures import (
    IPU,
    QUESTION_ACTS,
    ConversationAnalysis,
    DialogueAct,
    DialogueActLabel,
    DialogueActSegment,
    Disfluency,
    EntrainmentResult,
    FloorTransfer,
    Overlap,
    Turn,
    TurnMarkers,
    Word,
)
from senselab.text.tasks.conversation_analysis.dialogue_acts import (
    DialogueActLabeller,
    RuleBasedLabeller,
    label_dialogue_acts,
    rule_act,
    segment_dialogue_acts,
)
from senselab.text.tasks.conversation_analysis.disfluency import detect_disfluencies
from senselab.text.tasks.conversation_analysis.entrainment import compute_entrainment
from senselab.text.tasks.conversation_analysis.markers import politeness_counts, turn_markers
from senselab.text.tasks.conversation_analysis.segmentation import (
    build_ipus,
    build_turns,
    mark_backchannels,
    normalize_words,
)
from senselab.text.tasks.conversation_analysis.turn_taking import floor_transfers, overlaps

__all__ = [
    "IPU",
    "QUESTION_ACTS",
    "ConversationAnalysis",
    "ConversationAnalysisConfig",
    "DialogueAct",
    "DialogueActLabel",
    "DialogueActLabeller",
    "DialogueActSegment",
    "Disfluency",
    "EntrainmentResult",
    "FloorTransfer",
    "Lexicons",
    "Overlap",
    "RuleBasedLabeller",
    "Turn",
    "TurnMarkers",
    "Word",
    "analyze_conversation",
    "build_ipus",
    "build_turns",
    "compute_entrainment",
    "detect_disfluencies",
    "floor_transfers",
    "label_dialogue_acts",
    "load_default_lexicons",
    "mark_backchannels",
    "normalize_words",
    "overlaps",
    "politeness_counts",
    "rule_act",
    "segment_dialogue_acts",
    "turn_markers",
]
