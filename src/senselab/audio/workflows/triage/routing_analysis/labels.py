"""The classifier labels each candidate routing detector reads, per classifier and per kind.

AudioSet labels are shared by YAMNet and AST; HeAR carries its own eight-label vocabulary. The
airway sets are the ones ``taxonomy.audioset_airway_labels`` and ``taxonomy.hear_airway_labels``
already name in ``data/config/default.yaml``.
"""

from __future__ import annotations

from typing import Mapping

CLASSIFIERS: tuple[str, ...] = ("yamnet", "ast", "hear")
"""Every classifier PREPROCESS summarises whole-file, kept apart because their grids differ."""

STREAMS: tuple[str, ...] = ("plain", "enhanced", "residual")
"""Every signal a classifier was run over, kept apart because their content differs."""

AUDIOSET_SPEECH: tuple[str, ...] = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
)

AUDIOSET_AIRWAY: tuple[str, ...] = (
    "Cough",
    "Throat clearing",
    "Sneeze",
    "Sniff",
    "Breathing",
    "Wheeze",
    "Snoring",
    "Gasp",
    "Sigh",
)

AUDIOSET_VOICE: tuple[str, ...] = (
    "Chant",
    "Mantra",
    "Singing",
    "Humming",
    "Choir",
    "Vocal music",
    "Yodeling",
)

HEAR_SPEECH: tuple[str, ...] = ("Speech",)

HEAR_AIRWAY: tuple[str, ...] = ("Cough", "Snore", "Baby Cough", "Breathe", "Sneeze", "Throat Clear")

HEAR_VOICE: tuple[str, ...] = ()

FAMILIES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "speech": {"yamnet": AUDIOSET_SPEECH, "ast": AUDIOSET_SPEECH, "hear": HEAR_SPEECH},
    "airway": {"yamnet": AUDIOSET_AIRWAY, "ast": AUDIOSET_AIRWAY, "hear": HEAR_AIRWAY},
    "voice": {"yamnet": AUDIOSET_VOICE, "ast": AUDIOSET_VOICE, "hear": HEAR_VOICE},
}
"""Each kind's label family per classifier."""

TRACKED_LABELS: Mapping[str, frozenset[str]] = {
    classifier: frozenset(label for kind in FAMILIES.values() for label in kind[classifier])
    for classifier in CLASSIFIERS
}
"""Every label whose peak the extractor keeps, per classifier. Everything else is dropped."""


def peak_key(stream: str, classifier: str, label: str) -> str:
    """The flat key one label's peak is stored under.

    Args:
        stream: ``plain``, ``enhanced``, ``residual`` or ``consensus``.
        classifier: ``yamnet``, ``ast`` or ``hear``.
        label: The classifier's own label string.

    Returns:
        ``"<stream>|<classifier>|<label>"``.
    """
    return f"{stream}|{classifier}|{label}"
