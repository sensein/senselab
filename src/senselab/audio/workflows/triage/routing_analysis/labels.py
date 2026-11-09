"""The classifier labels each candidate routing detector reads, per classifier and per kind.

AudioSet labels are shared by YAMNet and AST; HeAR carries its own eight-label vocabulary.

The airway, cough and breath sets are not listed here: they are read from the classifier-ontology
profile in ``data/classifier_ontology/``. The airway family is the closure of the roots the
configuration names in ``taxonomy.airway_ontology_roots``, minus the nodes no classifier
can emit. They are read when first asked for, under the active dataset profile (its ontology and its
configuration's roots), never at import: :func:`families`, :func:`label_sets`, :func:`tracked_labels`.
The module's derived constants (``AUDIOSET_AIRWAY``, ``FAMILIES``, ...) are the same values, read on each
access. See ``specs/20260910-classifier-ontology-mapping/design.md``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from senselab.audio.workflows.triage.classifier_ontology import (
    AIRWAY_ROOTS_PATH,
    PROFILE_PATH_KEY,
    audioset_labels_for_group,
    audioset_labels_under_roots,
    hear_labels_in_group,
    hear_labels_under_roots,
)
from senselab.audio.workflows.triage.dataset_profile import active_profile, lookup_value, profile_cache

CLASSIFIERS: tuple[str, ...] = ("yamnet", "ast", "hear")
"""Every classifier PREPROCESS summarises whole-file, one grid each."""

STREAMS: tuple[str, ...] = ("plain", "enhanced", "residual")
"""Every signal a classifier was run over."""

AUDIOSET_SPEECH: tuple[str, ...] = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
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

HEAR_VOICE: tuple[str, ...] = ()

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

AUDIOSET_WHISTLE: tuple[str, ...] = ("Whistling", "Whistle")

FAMILY_CLASSIFIERS: Mapping[str, tuple[str, ...]] = {
    "speech": CLASSIFIERS,
    "airway": CLASSIFIERS,
    "voice": ("yamnet", "ast"),
}
"""The classifiers that carry each kind's family (:func:`families`), known without reading the ontology: HeAR has
no voice label. The detector candidates are declared from it at import."""


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


@profile_cache
def airway_labels() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The airway family's AudioSet and HeAR labels, from the active profile's configured roots and ontology.

    The roots and the ontology file are the packaged configuration's with the active profile's overrides merged
    over it (:func:`~senselab.audio.workflows.triage.config.profile_config_values`).

    Returns:
        ``(AudioSet labels, HeAR labels)``: the emittable AudioSet closure of the roots, sorted, and the HeAR
        labels whose mapped node falls inside it, in graph order.

    Raises:
        ValueError: If the roots are unset, or if one is not an AudioSet class in the ontology.
    """
    from senselab.audio.workflows.triage.config import profile_config_values

    values = profile_config_values(active_profile())
    roots = lookup_value(values, AIRWAY_ROOTS_PATH.split("."))
    if roots is None:
        raise ValueError(f"the configuration has no {AIRWAY_ROOTS_PATH}")
    path = lookup_value(values, PROFILE_PATH_KEY.split("."))
    return audioset_labels_under_roots(roots, path), hear_labels_under_roots(roots, path)


@profile_cache
def families() -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    """Each kind's label family per classifier, under the active profile.

    Returns:
        ``{kind: {classifier: labels}}`` for ``speech``, ``airway`` and ``voice``.
    """
    audioset_airway, hear_airway = airway_labels()
    return {
        "speech": {"yamnet": AUDIOSET_SPEECH, "ast": AUDIOSET_SPEECH, "hear": HEAR_SPEECH},
        "airway": {"yamnet": audioset_airway, "ast": audioset_airway, "hear": hear_airway},
        "voice": {"yamnet": AUDIOSET_VOICE, "ast": AUDIOSET_VOICE, "hear": HEAR_VOICE},
    }


@profile_cache
def label_sets() -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    """Named label unions beyond the three routing kinds, read by a ``peak_set`` detector, under the active profile.

    Returns:
        ``{set name: {classifier: labels}}``; the cough and breath sets are the ontology's groups.
    """
    audioset_cough, audioset_breath = audioset_labels_for_group("cough"), audioset_labels_for_group("breath")
    return {
        "singing": {"yamnet": AUDIOSET_SINGING, "ast": AUDIOSET_SINGING, "hear": ()},
        "cough_labels": {"yamnet": audioset_cough, "ast": audioset_cough, "hear": hear_labels_in_group("cough")},
        "breath_labels": {"yamnet": audioset_breath, "ast": audioset_breath, "hear": hear_labels_in_group("breath")},
        "whistle": {"yamnet": AUDIOSET_WHISTLE, "ast": AUDIOSET_WHISTLE, "hear": ()},
    }


@profile_cache
def tracked_labels() -> Mapping[str, frozenset[str]]:
    """Every label whose peak the extractor keeps, per classifier, under the active profile.

    Returns:
        ``{classifier: labels}``: the union of :func:`families` and :func:`label_sets`.
    """
    groups = (*families().values(), *label_sets().values())
    return {
        classifier: frozenset(label for group in groups for label in group[classifier]) for classifier in CLASSIFIERS
    }


_DERIVED: Mapping[str, Callable[[], Any]] = {
    "AUDIOSET_AIRWAY": lambda: airway_labels()[0],
    "HEAR_AIRWAY": lambda: airway_labels()[1],
    "AUDIOSET_COUGH": lambda: audioset_labels_for_group("cough"),
    "AUDIOSET_BREATH": lambda: audioset_labels_for_group("breath"),
    "HEAR_COUGH": lambda: hear_labels_in_group("cough"),
    "HEAR_BREATH": lambda: hear_labels_in_group("breath"),
    "FAMILIES": families,
    "LABEL_SETS": label_sets,
    "TRACKED_LABELS": tracked_labels,
}
"""The derived constants this module once computed at import, each now read under the profile active when asked."""


def __getattr__(name: str) -> Any:  # noqa: ANN401 -- one of the derived constants
    if name in _DERIVED:
        return _DERIVED[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
