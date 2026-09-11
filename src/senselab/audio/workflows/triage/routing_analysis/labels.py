"""The classifier labels each candidate routing detector reads, per classifier and per kind.

AudioSet labels are shared by YAMNet and AST; HeAR carries its own eight-label vocabulary. The
airway family is the one ``taxonomy.audioset_airway_labels`` already names in
``data/config/default.yaml``.

The cough and breath sets are **not** listed here. They are read from the classifier-ontology
profile in ``data/classifier_ontology/``, so an AudioSet class enters a set by being the mapped node
of a HeAR label or a descendant of it, not by having been typed into two files that can drift apart.
See ``specs/20260910-classifier-ontology-mapping/design.md``.
"""

from __future__ import annotations

from typing import Mapping

from senselab.audio.workflows.triage.classifier_ontology import audioset_labels_for_group, hear_labels_in_group

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

HEAR_AIRWAY: tuple[str, ...] = hear_labels_in_group("cough") + hear_labels_in_group("breath")

HEAR_VOICE: tuple[str, ...] = ()

FAMILIES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "speech": {"yamnet": AUDIOSET_SPEECH, "ast": AUDIOSET_SPEECH, "hear": HEAR_SPEECH},
    "airway": {"yamnet": AUDIOSET_AIRWAY, "ast": AUDIOSET_AIRWAY, "hear": HEAR_AIRWAY},
    "voice": {"yamnet": AUDIOSET_VOICE, "ast": AUDIOSET_VOICE, "hear": HEAR_VOICE},
}
"""Each kind's label family per classifier."""

AUDIOSET_SINGING: tuple[str, ...] = (
    "A capella",
    "Chant",
    "Child singing",
    "Choir",
    "Female singing",
    "Humming",
    "Male singing",
    "Mantra",
    "Singing",
    "Synthetic singing",
    "Vocal music",
    "Yodeling",
)
"""The AudioSet singing subtree. YAMNet's 521-label grid carries all but the two gendered variants."""

AUDIOSET_COUGH: tuple[str, ...] = audioset_labels_for_group("cough")
"""The AudioSet closure of every HeAR cough-group label. Derived; see the module docstring."""

AUDIOSET_BREATH: tuple[str, ...] = audioset_labels_for_group("breath")
"""The AudioSet closure of every HeAR breath-group label. Derived; see the module docstring."""

AUDIOSET_WHISTLE: tuple[str, ...] = ("Whistling", "Whistle")

HEAR_COUGH: tuple[str, ...] = hear_labels_in_group("cough")
"""The HeAR labels the profile puts in the cough group. Derived; see the module docstring."""

HEAR_BREATH: tuple[str, ...] = hear_labels_in_group("breath")
"""The HeAR labels the profile puts in the breath group. Derived; see the module docstring."""

LABEL_SETS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "singing": {"yamnet": AUDIOSET_SINGING, "ast": AUDIOSET_SINGING, "hear": ()},
    "cough_labels": {"yamnet": AUDIOSET_COUGH, "ast": AUDIOSET_COUGH, "hear": HEAR_COUGH},
    "breath_labels": {"yamnet": AUDIOSET_BREATH, "ast": AUDIOSET_BREATH, "hear": HEAR_BREATH},
    "whistle": {"yamnet": AUDIOSET_WHISTLE, "ast": AUDIOSET_WHISTLE, "hear": ()},
}
"""Named label unions beyond the three routing kinds, read by a ``peak_set`` detector."""

TRACKED_LABELS: Mapping[str, frozenset[str]] = {
    classifier: frozenset(
        label
        for group in (*FAMILIES.values(), *LABEL_SETS.values())
        for label in group[classifier]  # type: ignore[index]
    )
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
