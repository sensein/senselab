"""COHORT: readings that need more than one recording, written into each recording's store before VERDICT.

Two families of reading, over what the per-recording stages already stored:

- **Speaker enrollment.** One speaker vector per participant and session, pooled from the
  cleanest single-speaker spans of that session's lexical speech tasks, and per session a participant
  vector pooled from the participant's other sessions. Each recording then gives at most two pooled
  embeddings, each compared with the session's vector, or with the other sessions' where the session has
  none: its target speech (:func:`target_rule`) and its speech outside the task. A pool shorter than
  ``comparison.pooled_min_s`` is not measured; one under its ``comparison.cosine_min`` is a non-match. A session whose
  vector is far from its other sessions' is ``session_speaker_inconsistent``.
- **Distribution checks.** Corpus quantiles per declared family, stored once per run as a cohort
  artefact with its own provenance, and per-recording checks against them
  (:data:`CHECK_FUNCTIONS`; ``recording_duration_outlier`` is the first).

Each recording gains one ``cohort_reading`` measurement (:mod:`~senselab.audio.workflows.triage.cohort_fold`
is its reader). A run that sees no cohort writes the reading ``unavailable``
(:func:`write_cohort_unavailable`). The parameters are ``data/cohort_stage.yaml``; the design is
``specs/20261007-task-events-in-background/design.md``, section "Cohort stage".
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from senselab.audio.workflows.triage.cohort_fold import (
    CHECKS,
    COHORT_NODE,
    COHORT_READING,
    KEY_SESSION_SPEAKER_INCONSISTENT,
    MEASURED,
    NOT_APPLICABLE,
    NOT_MEASURED,
    OTHER_SPEAKER,
    POOL_NON_TASK,
    POOL_TARGET,
    UNAVAILABLE,
)
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.decision import PASS, REVIEW
from senselab.audio.workflows.triage.live_evidence import declared_task, recording_stem
from senselab.audio.workflows.triage.nodes.background import session_key
from senselab.audio.workflows.triage.nodes.branches import derivative_arrays, merge
from senselab.audio.workflows.triage.nodes.common import (
    consensus_words,
    find_measurement,
    find_measurements,
    lexical_words,
    live_entities,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.residue import is_non_lexical
from senselab.audio.workflows.triage.routing_analysis import families as family_sets
from senselab.audio.workflows.triage.routing_analysis.families import DECLARED_KIND, LEXICAL_SPEECH
from senselab.audio.workflows.triage.task_content import task_content_ids, task_events
from senselab.audio.workflows.triage.task_lexicon import declared_names_lexicon
from senselab.audio.workflows.triage.task_speech import (
    TASK_SPEECH_READING,
    lexical_voice_family,
    task_speech_parameters,
    task_text_ids,
)
from senselab.audio.workflows.triage.vocabulary import standing_task_extents
from senselab.utils.prov_store import Entity, ProvStore

PARAMETERS_PATH = Path(__file__).parent / "data" / "cohort_stage.yaml"

RECORDING_STREAM = "recording"
PLAIN_STREAM = "plain"
QUANTILES_KIND = "cohort_quantiles"
QUANTILES_FILE = "cohort_quantiles.json"
FAMILY_LEVEL = "family"

SCOPE_SESSION = "session"
SCOPE_PARTICIPANT = "participant"
SOURCE_SESSION = "session"
SOURCE_PARTICIPANT_OTHER_SESSIONS = "participant_other_sessions"
SOURCE_NONE = "none"
"""``enrollment_source``: what a recording's runs were compared against."""

CONSISTENCY_CHECK = "session_speaker_consistency"
"""The check comparing a session's enrollment with the participant's other sessions."""

NO_COHORT = "no cohort was read: the run saw this recording alone"
NO_ENROLLMENT_SPANS = "no lexical speech span long enough to enroll"
FEWER_THAN_TWO_ENROLLED_SESSIONS = "the participant has fewer than two sessions with an enrollment of their own"
SESSION_NOT_ENROLLED = "the session has no enrollment of its own"
OTHER_SESSIONS_NOT_ENROLLED = "the participant's other sessions hold no enrollment"
CONSISTENCY_UNSET = "participant.consistency_cosine_min is unset"
ENROLLMENT_UNMEASURED = "speech.enrollment_model or speech.target_match_cosine is unset"
NO_QUANTILES = "no cohort quantiles were supplied"
FAMILY_NOT_IN_QUANTILES = "the recording's family has no quantiles in the cohort artefact"
NO_DURATION = "the store records no recording duration"
PLAIN_UNREADABLE = "the plain stream could not be read"
TOO_SHORT_FOR_SPEAKER_ID = "too_short_for_speaker_id"
NO_TARGET_COMPARISON = "no_target_comparison"
NO_SPEECH_OUTSIDE_TASK = "no_speech_outside_task"
NOT_EMBEDDED = "the pooled audio embedded to nothing"

TARGET_SPEECH = "speech"
TARGET_TASK_TEXT = "task_text"
TARGET_NONE = "none"
TARGET_RULES = (TARGET_SPEECH, TARGET_TASK_TEXT, TARGET_NONE)
"""``comparison.target`` rules: lexical words in the task extent, the declared task text, or no target pool."""

Embedder = Callable[[np.ndarray, int], "np.ndarray | None"]
"""Embeds one span's mono samples at a sampling rate into a unit vector, or None where it cannot."""

AudioLoader = Callable[[], "tuple[np.ndarray, int] | None"]
"""Loads one recording's plain stream as mono float32 samples and their rate, or None."""


@cache
def cohort_parameters() -> dict[str, Any]:
    """The packaged ``data/cohort_stage.yaml``.

    Returns:
        The parsed file.
    """
    return dict(yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8")))


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _length(spans: Sequence[tuple[float, float]]) -> float:
    return float(sum(max(0.0, end - start) for start, end in spans))


def _intersect(spans: Sequence[tuple[float, float]], window: tuple[float, float]) -> list[tuple[float, float]]:
    lo, hi = window
    return [(max(start, lo), min(end, hi)) for start, end in spans if min(end, hi) > max(start, lo)]


def _subtract(spans: Sequence[tuple[float, float]], cut: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    removed = merge(list(cut))
    for start, end in merge(list(spans)):
        cursor = start
        for other_start, other_end in removed:
            if other_end <= cursor or other_start >= end:
                continue
            if other_start > cursor:
                kept.append((cursor, other_start))
            cursor = max(cursor, other_end)
        if end > cursor:
            kept.append((cursor, end))
    return kept


# ------------------------------------------------------------------------------ one recording


@dataclass(frozen=True)
class RecordingFacts:
    """What COHORT reads off one recording's store.

    Attributes:
        stem: The recording's file stem.
        session: ``sub-<label>_ses-<label>``, or None where the stem names no session.
        family: The declared task family.
        lexical_task: Whether that family is a lexical speech task.
        duration_s: The recording's duration, or None where the store records none.
        task_extent: The hull of the standing task extents, or None where no branch wrote one.
        diarization: ``(start, end, speaker)`` per diarized segment on the configured stream.
        lexical: Every lexical consensus word's extent, in index order.
        target: The extents of the words the target pool holds (:func:`target_words`), in index order.
        non_task: The extents of the words of speech outside the task (:func:`non_task_words`), in index order.
        removed_speech: Speech-classified on the plain stream and not on the enhanced one: a voice the
            enhancer removed as background.
        task_events: The task events the branches read (breaths, coughs, phonation holds, syllables).
    """

    stem: str
    session: str | None
    family: str
    lexical_task: bool
    duration_s: float | None
    task_extent: tuple[float, float] | None
    diarization: tuple[tuple[float, float, str], ...] = ()
    lexical: tuple[tuple[float, float], ...] = ()
    target: tuple[tuple[float, float], ...] = ()
    non_task: tuple[tuple[float, float], ...] = ()
    removed_speech: tuple[tuple[float, float], ...] = ()
    task_events: tuple[tuple[float, float], ...] = ()

    @property
    def extent_fraction(self) -> float | None:
        """The task extent's share of the recording; 0.0 where no branch wrote one.

        Returns:
            The fraction, or None where the duration is unknown.
        """
        if not self.duration_s:
            return None
        if self.task_extent is None:
            return 0.0
        return max(0.0, self.task_extent[1] - self.task_extent[0]) / float(self.duration_s)

    def fact_row(self) -> dict[str, Any]:
        """The row the corpus quantiles are taken over.

        Returns:
            ``stem``, ``family``, ``duration_s``, ``extent_s`` and ``extent_fraction``.
        """
        extent = None if self.task_extent is None else round(self.task_extent[1] - self.task_extent[0], 3)
        fraction = self.extent_fraction
        return {
            "stem": self.stem,
            "family": self.family,
            "duration_s": self.duration_s,
            "extent_s": extent,
            "extent_fraction": None if fraction is None else round(fraction, 4),
        }


def _recording_duration(store: ProvStore) -> float | None:
    found = [
        entity
        for entity in store.entities("stream")
        if entity.attributes.get("name") == RECORDING_STREAM and not store.is_invalidated(entity.id)
    ]
    if not found or found[-1].extent is None:
        return None
    return float(found[-1].extent[1])


def diarization_measurement(stream: str) -> str:
    """The measurement name PREPROCESS writes one stream's whole-file diarization under.

    Args:
        stream: The stream's name.

    Returns:
        ``<stream>_diarization``.
    """
    return f"{stream}_diarization"


def _yamnet_windows(store: ProvStore, run_dir: Path, name: str) -> list[dict[str, Any]]:
    measurement = find_measurement(store, name)
    if measurement is None or not measurement.attributes.get("path"):
        return []
    path = run_dir / str(measurement.attributes["path"])
    if not path.is_file():
        return []
    return list(json.loads(path.read_text()))


def speech_regions(
    windows: Sequence[Mapping[str, Any]], labels: Sequence[str], score_min: float
) -> list[tuple[float, float]]:
    """The classifier windows whose highest speech-label score reaches ``score_min``, merged.

    Args:
        windows: Classifier windows, each with ``start``, ``end`` and ``label_scores``.
        labels: The labels that are speech.
        score_min: The score at or above which a window is speech.

    Returns:
        The merged regions, earliest first.
    """
    out: list[tuple[float, float]] = []
    for window in windows:
        scores: dict[str, float] = {}
        raw = window.get("label_scores") or {}
        for pair in raw if isinstance(raw, list) else [raw]:
            scores.update({str(k): float(v) for k, v in dict(pair).items()})
        if max((scores.get(label, 0.0) for label in labels), default=0.0) >= score_min:
            out.append((float(window["start"]), float(window["end"])))
    return merge(out)


def _task_events(store: ProvStore) -> list[tuple[float, float]]:
    from senselab.audio.workflows.triage.nodes.quality import task_events_of  # noqa: PLC0415

    return list(task_events_of(store)[1])


def target_rule(family: str) -> str:
    """What a family's target pool holds (``comparison.target``).

    Args:
        family: The declared task family.

    Returns:
        One of :data:`TARGET_RULES`: the family's own entry, else its declared kind's, else the default.

    Raises:
        ValueError: Where the configured rule is not one of :data:`TARGET_RULES`.
    """
    table = cohort_parameters()["comparison"]["target"]
    rule = (table.get("families") or {}).get(family)
    if rule is None:
        rule = next(
            (value for kind, value in (table.get("kinds") or {}).items() if family in DECLARED_KIND[str(kind)]),
            table["default"],
        )
    if rule not in TARGET_RULES:
        raise ValueError(f"cohort_stage.yaml comparison.target: {rule!r} for {family!r} is not one of {TARGET_RULES}")
    return str(rule)


def lexical_task(family: str) -> bool:
    """Whether a family is a lexical speech task for COHORT: in ``LEXICAL_SPEECH`` and not read as a voice task.

    Args:
        family: The declared task family.

    Returns:
        True for a lexical speech family other than one under ``task_speech.yaml`` ``voice_task_text``.
    """
    return family in LEXICAL_SPEECH and not lexical_voice_family(family)


def speech_outside_task(store: ProvStore, family: str) -> list[Entity]:
    """The consensus words of a non-lexical task that are speech outside it.

    The owning branch's ``task_speech_reading`` names them where it stored one; otherwise they are read here
    as it reads them: lexical consensus words that are not a vocalisation, not the family's expected tokens
    and not a misreading of the task's own events. A word spelled only from the family's task syllables
    (:func:`spells_task_syllables`) is never one.

    Args:
        store: The recording's store.
        family: The declared task family.

    Returns:
        The words, in stream order.
    """
    words = consensus_words(store)
    reading = find_measurement(store, TASK_SPEECH_READING)
    if reading is not None:
        named = {str(i) for i in reading.attributes.get("word_ids") or ()}
        said = [word for word in words if word.id in named]
    else:
        texts = [str(word.attributes.get("text") or "") for word in words]
        lexicon = {words[i].id for i in declared_names_lexicon(family).positions(texts)}
        exempt = task_content_ids(words, task_events(store, family), family=family, lexicon_ids=lexicon)
        exempt |= task_text_ids(words, family, ())
        said = [
            word
            for word in lexical_words(store)
            if word.id not in exempt and not is_non_lexical(str(word.attributes.get("text") or ""), vocal_task=True)
        ]
    return [word for word in said if not spells_task_syllables(str(word.attributes.get("text") or ""), family)]


def task_text_words(store: ProvStore, family: str) -> list[Entity]:
    """The lexical consensus words that are the family's declared task text (``task_speech.task_text_ids``).

    Args:
        store: The recording's store.
        family: The declared task family.

    Returns:
        The words, in stream order; empty for a family declaring no task text.
    """
    ids = task_text_ids(consensus_words(store), family, ())
    return [word for word in lexical_words(store) if word.id in ids]


def _midpoint_in(word: Entity, span: tuple[float, float] | None) -> bool:
    if word.extent is None or span is None:
        return False
    return span[0] <= (float(word.extent[0]) + float(word.extent[1])) / 2.0 <= span[1]


def target_words(store: ProvStore, family: str, extent: tuple[float, float] | None) -> list[Entity]:
    """The words the target pool holds, by :func:`target_rule`.

    Args:
        store: The recording's store.
        family: The declared task family.
        extent: The task extent's hull, or None where no branch wrote one (the whole recording).

    Returns:
        ``speech``: the lexical words whose midpoint is inside the extent; ``task_text``: the declared task-text
        words; ``none``: nothing. In stream order.
    """
    rule = target_rule(family)
    if rule == TARGET_TASK_TEXT:
        return task_text_words(store, family)
    if rule == TARGET_SPEECH:
        words = lexical_words(store)
        return words if extent is None else [word for word in words if _midpoint_in(word, extent)]
    return []


def non_task_words(store: ProvStore, family: str, extent: tuple[float, float] | None) -> list[Entity]:
    """The words of speech outside the task.

    Args:
        store: The recording's store.
        family: The declared task family.
        extent: The task extent's hull, or None where no branch wrote one.

    Returns:
        In a lexical speech task (:func:`lexical_task`) the lexical words whose midpoint is outside the extent
        (none without one); elsewhere :func:`speech_outside_task`. In stream order.
    """
    if lexical_task(family):
        if extent is None:
            return []
        return [word for word in lexical_words(store) if word.extent is not None and not _midpoint_in(word, extent)]
    return speech_outside_task(store, family)


def spells_task_syllables(text: str, family: str) -> bool:
    """Whether a word is spelled only from its family's task syllables (``non_task.task_syllables``).

    Args:
        text: The word as the recogniser wrote it.
        family: The declared task family.

    Returns:
        True where the family's set names syllables and the word's letters are one or more of them run
        together (``pa``, ``Pa-ta-ka``, ``buttercup``); False otherwise.
    """
    syllables = _task_syllables(family)
    letters = "".join(ch for ch in text.lower() if ch.isalpha())
    return bool(syllables and letters) and _spelled(letters, syllables)


def _task_syllables(family: str) -> tuple[str, ...]:
    for set_name, syllables in (cohort_parameters()["non_task"].get("task_syllables") or {}).items():
        if family in getattr(family_sets, str(set_name), frozenset()):
            return tuple(sorted({str(s).lower() for s in syllables}, key=len, reverse=True))
    return ()


def _spelled(letters: str, syllables: Sequence[str]) -> bool:
    reach = [True] + [False] * len(letters)
    for end in range(1, len(letters) + 1):
        reach[end] = any(end >= len(s) and reach[end - len(s)] and letters[end - len(s) : end] == s for s in syllables)
    return reach[-1]


def facts_of(
    store: ProvStore, run_dir: Path | None, *, diarization_stream: str | None = None, stem: str | None = None
) -> RecordingFacts:
    """Read COHORT's inputs off one recording's store.

    Args:
        store: The recording's store.
        run_dir: Its run directory, for the diarization sidecar; None reads no diarization.
        diarization_stream: The stream whose diarization is read; None reads none.
        stem: The stem to use where the store's ``recording`` stream names none.

    Returns:
        The facts.
    """
    stem = recording_stem(store) or (stem or "")
    family = declared_task(stem)[1]
    extents = [span.extent for span in standing_task_extents(live_entities(store, "span")) if span.extent]
    hull = (min(float(e[0]) for e in extents), max(float(e[1]) for e in extents)) if extents else None
    segments: tuple[tuple[float, float, str], ...] = ()
    if run_dir is not None and diarization_stream is not None:
        arrays = derivative_arrays(store, run_dir, diarization_measurement(diarization_stream))
        if arrays is not None:
            segments = tuple(
                sorted(
                    (float(start), float(end), str(speaker))
                    for start, end, speaker in zip(arrays["starts"], arrays["ends"], arrays["speakers"])
                )
            )
    by_name: dict[str, list[tuple[float, float]]] = {}
    classes = cohort_parameters()["speech"]
    removed = classes["removed"]
    if run_dir is not None:
        for name in sorted(set(removed.values())):
            by_name[str(name)] = speech_regions(
                _yamnet_windows(store, run_dir, str(name)), classes["labels"], float(classes["score_min"])
            )
    removed_speech = _subtract(by_name.get(removed["from"], []), by_name.get(removed["kept"], []))
    return RecordingFacts(
        stem=stem,
        session=session_key(stem),
        family=family,
        lexical_task=lexical_task(family),
        duration_s=_recording_duration(store),
        task_extent=hull,
        diarization=segments,
        lexical=_extents(lexical_words(store)),
        target=_extents(target_words(store, family, hull)),
        non_task=_extents(non_task_words(store, family, hull)),
        removed_speech=tuple(removed_speech),
        task_events=tuple(merge(_task_events(store))),
    )


def _extents(words: Sequence[Entity]) -> tuple[tuple[float, float], ...]:
    return tuple((float(w.extent[0]), float(w.extent[1])) for w in words if w.extent is not None)


def exclusive_pieces(segments: Sequence[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Each speaker's diarized time with every stretch another speaker also holds removed.

    Args:
        segments: ``(start, end, speaker)`` per segment.

    Returns:
        ``(start, end, speaker)`` per contiguous exclusive piece, earliest first.
    """
    speakers = sorted({speaker for _, _, speaker in segments})
    pieces: list[tuple[float, float, str]] = []
    for label in speakers:
        mine = [(s, e) for s, e, speaker in segments if speaker == label]
        theirs = [(s, e) for s, e, speaker in segments if speaker != label]
        pieces.extend((start, end, label) for start, end in _subtract(mine, theirs))
    return sorted(pieces)


def target_spans(facts: RecordingFacts, *, lexical_gap_s: float) -> list[tuple[float, float]]:
    """The spans the target pool embeds: the target words merged across gaps up to ``lexical_gap_s``.

    Args:
        facts: The recording's facts.
        lexical_gap_s: Largest gap two words may straddle and stay one span.

    Returns:
        The spans, earliest first.
    """
    return lexical_runs_of(facts.target, lexical_gap_s)


def non_task_spans(facts: RecordingFacts, *, lexical_gap_s: float) -> list[tuple[float, float]]:
    """The spans the non-task pool embeds: its words, and the removed speech holding one, less task events.

    Args:
        facts: The recording's facts.
        lexical_gap_s: Largest gap two words may straddle and stay one span.

    Returns:
        The merged spans, earliest first; empty where the recording has no speech outside the task.
    """
    if not facts.non_task:
        return []
    removed = [
        (s, e)
        for s, e in _subtract(list(facts.removed_speech), facts.task_events)
        if any(s <= (a + b) / 2.0 <= e for a, b in facts.non_task)
    ]
    return _subtract(merge([*lexical_runs_of(facts.non_task, lexical_gap_s), *removed]), facts.task_events)


@dataclass(frozen=True)
class EnrollmentSpans:
    """The spans one recording contributes to its session's enrollment.

    Attributes:
        stem: The recording.
        spans: ``(start, end)`` per span, each at least the floor.
        single_speaker: Whether the diarizer held one speaker inside the task extent.
    """

    stem: str
    spans: tuple[tuple[float, float], ...]
    single_speaker: bool


def enrollment_spans(facts: RecordingFacts, *, min_s: float, lexical_gap_s: float) -> EnrollmentSpans:
    """The cleanest single-speaker spans of one lexical speech recording.

    The dominant diarized speaker's exclusive pieces inside the task extent (the whole recording
    where none stands); with no diarization, the lexical word runs inside it.

    Args:
        facts: The recording's facts.
        min_s: The shortest span kept.
        lexical_gap_s: Largest gap two lexical words may straddle and stay one run.

    Returns:
        The spans, possibly empty.
    """
    window = facts.task_extent or (0.0, float(facts.duration_s or math.inf))
    inside = [
        (max(s, window[0]), min(e, window[1]), k)
        for s, e, k in facts.diarization
        if min(e, window[1]) > max(s, window[0])
    ]
    if inside:
        held: dict[str, float] = {}
        for start, end, speaker in inside:
            held[speaker] = held.get(speaker, 0.0) + (end - start)
        dominant = max(sorted(held), key=lambda label: held[label])
        pieces = [(s, e) for s, e, k in exclusive_pieces(inside) if k == dominant and e - s >= min_s]
        return EnrollmentSpans(facts.stem, tuple(pieces), single_speaker=len(held) == 1)
    runs = [(s, e) for s, e in _intersect(lexical_runs_of(facts.lexical, lexical_gap_s), window) if e - s >= min_s]
    return EnrollmentSpans(facts.stem, tuple(runs), single_speaker=False)


def lexical_runs_of(words: Sequence[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    """Word extents merged across gaps of at most ``gap_s``.

    Args:
        words: Word extents, in index order.
        gap_s: The largest gap bridged.

    Returns:
        The runs, in order.
    """
    runs: list[tuple[float, float]] = []
    for start, end in words:
        if runs and start - runs[-1][1] <= gap_s:
            runs[-1] = (runs[-1][0], max(runs[-1][1], end))
        else:
            runs.append((start, end))
    return runs


# ------------------------------------------------------------------------------ embedding


def cut_samples(samples: np.ndarray, rate: int, span: tuple[float, float]) -> np.ndarray:
    """One span of a mono signal.

    Args:
        samples: The signal.
        rate: Its sampling rate.
        span: ``(start, end)`` in seconds.

    Returns:
        The samples inside the span.
    """
    first = max(0, int(round(span[0] * rate)))
    last = min(samples.shape[-1], int(round(span[1] * rate)))
    return samples[first:last]


def ecapa_embedder(model_id: str, commit: str, *, window_s: float, hop_s: float, device: Any = None) -> Embedder:  # noqa: ANN401
    """An embedder over one span at a time: windowed, each window embedded, the windows' spherical mean.

    Every call embeds one span, and every window of one span is the same length, so no batch mixes
    durations.

    Args:
        model_id: The speaker-embedding model.
        commit: Its resolved 40-hex commit.
        window_s: The window length.
        hop_s: The window hop.
        device: A senselab ``DeviceType``, or None for the default.

    Returns:
        The embedder.
    """

    def embed(samples: np.ndarray, rate: int) -> np.ndarray | None:
        import torch  # noqa: PLC0415

        from senselab.audio.data_structures import Audio  # noqa: PLC0415
        from senselab.audio.tasks.speaker_embeddings.windowing import extract_per_window_embeddings  # noqa: PLC0415

        if samples.size == 0:
            return None
        waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        failures: dict[str, str] = {}
        windows = extract_per_window_embeddings(
            audio=Audio(waveform=waveform, sampling_rate=rate),
            models=[model_id],
            window_s=window_s,
            hop_s=hop_s,
            device=device,
            revision=commit,
            failures=failures,
        ).get(model_id, [])
        vectors = [_unit(np.asarray(w.vector, dtype=np.float64)) for w in windows]
        vectors = [v for v in vectors if np.linalg.norm(v) > 0]
        if not vectors:
            return None
        return _unit(np.vstack(vectors).mean(axis=0))

    return embed


def plain_loader(store: ProvStore, run_dir: Path) -> AudioLoader:
    """A loader for one recording's plain stream, read on first call.

    Args:
        store: The recording's store.
        run_dir: Its run directory.

    Returns:
        The loader; it returns None where the stream is absent or unreadable.
    """

    def load() -> tuple[np.ndarray, int] | None:
        from senselab.utils.portable_audio_io import read_audio  # noqa: PLC0415

        found = [
            e
            for e in store.entities("stream")
            if e.attributes.get("name") == PLAIN_STREAM and not store.is_invalidated(e.id)
        ]
        if not found:
            return None
        path = Path(str(found[-1].attributes.get("path")))
        if not path.is_absolute():
            path = run_dir / path
        try:
            samples, rate = read_audio(path, always_2d=True, channels_first=False)
        except (OSError, RuntimeError):
            return None
        return np.ascontiguousarray(samples[:, 0]), int(rate)

    return load


# ------------------------------------------------------------------------------ the enrollment


def participant_key(session: str) -> str:
    """The participant a session key (or a stem standing for one) belongs to.

    Args:
        session: ``sub-<label>_ses-<label>``, or a stem naming no session.

    Returns:
        ``sub-<label>``: everything before the first underscore.
    """
    return session.split("_", 1)[0]


@dataclass
class Enrollment:
    """One participant's speaker vector, pooled over one session or over several.

    Attributes:
        key: The session key for a session enrollment; the participant label for a participant one.
        scope: :data:`SCOPE_SESSION` or :data:`SCOPE_PARTICIPANT`.
        vector: The unit vector, or None where nothing could enroll.
        reason: Why there is no vector, else None.
        model_id: The embedding model.
        model_commit_sha: Its resolved commit.
        sessions: The sessions whose recordings were candidates.
        excluded_session: The session a leave-one-session-out enrollment was built without, else None.
        excluded_recording: The recording a leave-the-recording-out enrollment was built without, else None.
        recordings: The stems whose spans were pooled.
        left_out: Stems left out by the leave-one-out step.
        spans_n: How many spans were embedded.
        seconds: Their total length.
        single_speaker_only: Whether only single-speaker recordings enrolled.
    """

    key: str
    scope: str
    vector: np.ndarray | None
    reason: str | None
    model_id: str | None
    model_commit_sha: str | None
    sessions: list[str] = field(default_factory=list)
    excluded_session: str | None = None
    excluded_recording: str | None = None
    recordings: list[str] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    spans_n: int = 0
    seconds: float = 0.0
    single_speaker_only: bool = False

    @property
    def vector_sha256(self) -> str | None:
        """The digest of the vector's float32 bytes, or None.

        Returns:
            The hex digest.
        """
        if self.vector is None:
            return None
        return hashlib.sha256(np.asarray(self.vector, dtype=np.float32).tobytes()).hexdigest()

    def record(self) -> dict[str, Any]:
        """The enrollment's description, without the vector.

        Returns:
            Its fields, JSON-ready.
        """
        return {
            "key": self.key,
            "scope": self.scope,
            "status": MEASURED if self.vector is not None else UNAVAILABLE,
            "reason": self.reason,
            "model_id": self.model_id,
            "model_commit_sha": self.model_commit_sha,
            "method": "recording_equal_spherical_mean",
            "sessions": list(self.sessions),
            "excluded_session": self.excluded_session,
            "excluded_recording": self.excluded_recording,
            "recordings": list(self.recordings),
            "recordings_n": len(self.recordings),
            "left_out": list(self.left_out),
            "spans_n": self.spans_n,
            "seconds": round(self.seconds, 3),
            "single_speaker_only": self.single_speaker_only,
            "vector_sha256": self.vector_sha256,
        }


@dataclass(frozen=True)
class EnrollmentCandidate:
    """One lexical speech recording that offers enrollment spans.

    Attributes:
        facts: The recording's facts.
        loader: Its plain-stream loader.
        spans: The spans it offers.
    """

    facts: RecordingFacts
    loader: AudioLoader
    spans: EnrollmentSpans


def enrollment_candidates(
    members: Sequence[tuple[RecordingFacts, AudioLoader]],
    *,
    min_s: float,
    lexical_gap_s: float,
    parameters: Mapping[str, Any] | None = None,
) -> list[EnrollmentCandidate]:
    """The members of an enrolling family that offer at least one span.

    Args:
        members: Recordings with their plain-stream loaders.
        min_s: The shortest span kept.
        lexical_gap_s: Largest gap two lexical words may straddle and stay one run.
        parameters: The ``enrollment`` block of ``data/cohort_stage.yaml``; None reads the packaged one.

    Returns:
        The candidates, in member order; a lexical family read as a voice task (``loudness``) never enrolls.
    """
    params = dict(parameters if parameters is not None else cohort_parameters()["enrollment"])
    out = []
    for facts, loader in members:
        if facts.family not in DECLARED_KIND[str(params["families"])] or lexical_voice_family(facts.family):
            continue
        spans = enrollment_spans(facts, min_s=min_s, lexical_gap_s=lexical_gap_s)
        if spans.spans:
            out.append(EnrollmentCandidate(facts, loader, spans))
    return out


Centroid = tuple[np.ndarray, int, float]
"""One recording's spherical mean over its spans, the spans embedded and their seconds."""


class CentroidCache:
    """Each candidate's recording centroid, embedded once however many pools read it."""

    def __init__(self, embed: Embedder) -> None:
        """Hold the embedder.

        Args:
            embed: The span embedder.
        """
        self.embed = embed
        self._held: dict[str, Centroid | None] = {}

    def __call__(self, candidate: EnrollmentCandidate) -> Centroid | None:
        """The candidate's centroid, or None where it loads or embeds nothing.

        Args:
            candidate: The recording.

        Returns:
            ``(vector, spans embedded, seconds)``.
        """
        stem = candidate.facts.stem
        if stem not in self._held:
            self._held[stem] = self._compute(candidate)
        return self._held[stem]

    def _compute(self, candidate: EnrollmentCandidate) -> Centroid | None:
        loaded = candidate.loader()
        if loaded is None:
            return None
        samples, rate = loaded
        spans = candidate.spans.spans
        vectors = [v for v in (self.embed(cut_samples(samples, rate, span), rate) for span in spans) if v is not None]
        if not vectors:
            return None
        return _unit(np.vstack(vectors).mean(axis=0)), len(vectors), _length(spans)


def pool_enrollment(
    key: str,
    scope: str,
    candidates: Sequence[EnrollmentCandidate],
    centroid: Callable[[EnrollmentCandidate], Centroid | None],
    *,
    model_id: str,
    commit: str,
    cut: float,
    parameters: Mapping[str, Any] | None = None,
    excluded_session: str | None = None,
) -> Enrollment:
    """Pool candidates into one speaker vector, recording-equal, under the enrollment's quality rules.

    Where at least ``single_speaker_recordings_min`` candidates are single-speaker, only they enroll;
    with ``leave_one_out_min`` or more centroids, one under the cut against the rest is left out.

    Args:
        key: The enrollment's key.
        scope: :data:`SCOPE_SESSION` or :data:`SCOPE_PARTICIPANT`.
        candidates: The recordings offering spans.
        centroid: Each candidate's centroid (:class:`CentroidCache`).
        model_id: The embedding model, recorded.
        commit: Its resolved commit, recorded.
        cut: The match cut, used by the leave-one-out step.
        parameters: The ``enrollment`` block of ``data/cohort_stage.yaml``; None reads the packaged one.
        excluded_session: The session left out of a participant enrollment, recorded.

    Returns:
        The enrollment; one with no vector names why.
    """
    params = dict(parameters if parameters is not None else cohort_parameters()["enrollment"])
    sessions = sorted({c.facts.session or c.facts.stem for c in candidates})
    single = [c for c in candidates if c.spans.single_speaker]
    single_only = len(single) >= int(params["single_speaker_recordings_min"])
    chosen = single if single_only else list(candidates)
    centroids: list[tuple[str, np.ndarray, int, float]] = []
    for candidate in chosen:
        found = centroid(candidate)
        if found is not None:
            centroids.append((candidate.facts.stem, *found))
    if not centroids:
        return Enrollment(
            key,
            scope,
            None,
            NO_ENROLLMENT_SPANS,
            model_id,
            commit,
            sessions=sessions,
            excluded_session=excluded_session,
        )
    left_out: list[str] = []
    if len(centroids) >= int(params["leave_one_out_min"]):
        kept = []
        for index, (stem, vector, n, seconds) in enumerate(centroids):
            others = _unit(np.vstack([c[1] for j, c in enumerate(centroids) if j != index]).mean(axis=0))
            if float(vector @ others) >= cut:
                kept.append((stem, vector, n, seconds))
            else:
                left_out.append(stem)
        centroids = kept or centroids
        left_out = left_out if kept else []
    return Enrollment(
        key,
        scope,
        _unit(np.vstack([c[1] for c in centroids]).mean(axis=0)),
        None,
        model_id,
        commit,
        sessions=sessions,
        excluded_session=excluded_session,
        recordings=[c[0] for c in centroids],
        left_out=left_out,
        spans_n=sum(c[2] for c in centroids),
        seconds=sum(c[3] for c in centroids),
        single_speaker_only=single_only,
    )


def enroll_session(
    session: str,
    members: Sequence[tuple[RecordingFacts, AudioLoader]],
    embed: Embedder,
    *,
    model_id: str,
    commit: str,
    cut: float,
    min_s: float,
    lexical_gap_s: float,
    parameters: Mapping[str, Any] | None = None,
) -> Enrollment:
    """Pool one session's lexical speech spans into one speaker vector (:func:`pool_enrollment`).

    Args:
        session: The session key.
        members: Every recording of the session, with a loader for its plain stream.
        embed: The span embedder.
        model_id: The embedding model, recorded.
        commit: Its resolved commit, recorded.
        cut: The match cut, used by the leave-one-out step.
        min_s: The shortest span embedded.
        lexical_gap_s: Largest gap two lexical words may straddle and stay one run.
        parameters: The ``enrollment`` block of ``data/cohort_stage.yaml``; None reads the packaged one.

    Returns:
        The session enrollment.
    """
    candidates = enrollment_candidates(members, min_s=min_s, lexical_gap_s=lexical_gap_s, parameters=parameters)
    return pool_enrollment(
        session,
        SCOPE_SESSION,
        candidates,
        CentroidCache(embed),
        model_id=model_id,
        commit=commit,
        cut=cut,
        parameters=parameters,
    )


@dataclass
class ParticipantEnrollments:
    """Every enrollment one participant's sessions yield.

    Attributes:
        participant: The participant label.
        sessions: Each session's own enrollment, by session key.
        other_sessions: Per session, the participant enrollment pooled from every other session; empty
            for a participant with one session.
        pooled: The participant enrollment over every session.
        without: Re-pools one session's enrollment without one recording: ``(session, stem) -> Enrollment``.
    """

    participant: str
    sessions: dict[str, Enrollment]
    other_sessions: dict[str, Enrollment]
    pooled: Enrollment
    without: Callable[[str, str], Enrollment] | None = None

    @property
    def enrolled_sessions(self) -> list[str]:
        """The sessions whose own enrollment holds a vector, in order."""
        return [key for key, enrollment in self.sessions.items() if enrollment.vector is not None]

    def source_for(self, session: str) -> tuple[str, Enrollment]:
        """The enrollment a session's recordings are compared against, and where it came from.

        Args:
            session: The session key.

        Returns:
            ``(enrollment_source, enrollment)``: the session's own where it enrolled, else the other
            sessions' participant enrollment where that holds a vector, else :data:`SOURCE_NONE` and
            the session's own (vectorless) enrollment.
        """
        own = self.sessions[session]
        if own.vector is not None:
            return SOURCE_SESSION, own
        other = self.other_sessions.get(session)
        if other is not None and other.vector is not None:
            return SOURCE_PARTICIPANT_OTHER_SESSIONS, other
        return SOURCE_NONE, own

    def source_for_recording(self, session: str, stem: str) -> tuple[str, Enrollment]:
        """The enrollment one recording is compared against: :meth:`source_for`, with the recording left out.

        Args:
            session: The session key.
            stem: The recording.

        Returns:
            ``(enrollment_source, enrollment)``. Where the recording enrolled its own session, the session's
            enrollment re-pooled without it; where that holds no vector, the other sessions' enrollment, else
            :data:`SOURCE_NONE`.
        """
        source, enrollment = self.source_for(session)
        if source != SOURCE_SESSION or stem not in enrollment.recordings or self.without is None:
            return source, enrollment
        left = self.without(session, stem)
        if left.vector is not None:
            return SOURCE_SESSION, left
        other = self.other_sessions.get(session)
        if other is not None and other.vector is not None:
            return SOURCE_PARTICIPANT_OTHER_SESSIONS, other
        return SOURCE_NONE, left


def enroll_participant(
    participant: str,
    sessions: Sequence[tuple[str, Sequence[tuple[RecordingFacts, AudioLoader]]]],
    embed: Embedder,
    *,
    model_id: str,
    commit: str,
    cut: float,
    min_s: float,
    lexical_gap_s: float,
    parameters: Mapping[str, Any] | None = None,
) -> ParticipantEnrollments:
    """Each session's enrollment, each session's leave-one-session-out participant enrollment, and the pool.

    Every pool applies :func:`pool_enrollment`'s rules to its own candidates. Each recording's
    centroid is embedded once and shared by every pool that reads it, including the session pools re-built
    without one recording (:meth:`ParticipantEnrollments.source_for_recording`).

    Args:
        participant: The participant label.
        sessions: ``(session key, members)`` per session of the participant.
        embed: The span embedder.
        model_id: The embedding model, recorded.
        commit: Its resolved commit, recorded.
        cut: The match cut, used by the leave-one-out step.
        min_s: The shortest span embedded.
        lexical_gap_s: Largest gap two lexical words may straddle and stay one run.
        parameters: The ``enrollment`` block of ``data/cohort_stage.yaml``; None reads the packaged one.

    Returns:
        The enrollments.
    """
    centroid = CentroidCache(embed)
    by_session = {
        session: enrollment_candidates(members, min_s=min_s, lexical_gap_s=lexical_gap_s, parameters=parameters)
        for session, members in sessions
    }

    def pool(key: str, scope: str, candidates: list[EnrollmentCandidate], excluded: str | None = None) -> Enrollment:
        return pool_enrollment(
            key,
            scope,
            candidates,
            centroid,
            model_id=model_id,
            commit=commit,
            cut=cut,
            parameters=parameters,
            excluded_session=excluded,
        )

    def without(session: str, stem: str) -> Enrollment:
        left = pool(session, SCOPE_SESSION, [c for c in by_session.get(session, []) if c.facts.stem != stem])
        return dataclasses.replace(left, excluded_recording=stem)

    own = {session: pool(session, SCOPE_SESSION, candidates) for session, candidates in by_session.items()}
    others = {
        session: pool(
            participant,
            SCOPE_PARTICIPANT,
            [c for other, candidates in by_session.items() if other != session for c in candidates],
            session,
        )
        for session in by_session
        if len(by_session) > 1
    }
    pooled = pool(participant, SCOPE_PARTICIPANT, [c for candidates in by_session.values() for c in candidates])
    return ParticipantEnrollments(participant, own, others, pooled, without)


def consistency_check(enrollments: ParticipantEnrollments, session: str, *, bound: float | None) -> dict[str, Any]:
    """``session_speaker_consistency``: the session's enrollment against the participant's other sessions.

    Args:
        enrollments: The participant's enrollments.
        session: The session key.
        bound: The cosine under which the session is inconsistent, or None where it is unset.

    Returns:
        The check's block: not applicable under two enrolled sessions, unavailable where the session
        did not enroll, else the cosine and its outcome.
    """
    enrolled = enrollments.enrolled_sessions
    count = {"enrolled_sessions_n": len(enrolled)}
    if len(enrolled) < 2:
        return {"status": NOT_APPLICABLE, "reason": FEWER_THAN_TWO_ENROLLED_SESSIONS, **count}
    own, other = enrollments.sessions[session], enrollments.other_sessions.get(session)
    if own.vector is None:
        return {"status": UNAVAILABLE, "reason": SESSION_NOT_ENROLLED, **count}
    if other is None or other.vector is None:
        return {"status": UNAVAILABLE, "reason": OTHER_SESSIONS_NOT_ENROLLED, **count}
    if bound is None:
        return {"status": UNAVAILABLE, "reason": CONSISTENCY_UNSET, **count}
    cosine = float(own.vector @ other.vector)
    return {
        "status": MEASURED,
        "ground": KEY_SESSION_SPEAKER_INCONSISTENT,
        "outcome": REVIEW if cosine < bound else PASS,
        "why": (
            f"the session's enrollment reads cosine {cosine:.3f} against the participant's "
            f"{len(other.sessions)} other session(s)"
        ),
        **count,
        "other_sessions": list(other.sessions),
        "comparisons": [
            {
                "name": "session_vs_other_sessions_cosine",
                "value": round(cosine, 4),
                "unit": None,
                "comparison": ">=",
                "threshold": bound,
            }
        ],
    }


def pooled_samples(samples: np.ndarray, rate: int, spans: Sequence[tuple[float, float]]) -> np.ndarray:
    """The spans of a mono signal, concatenated in order.

    Args:
        samples: The signal.
        rate: Its sampling rate.
        spans: ``(start, end)`` per span, in seconds.

    Returns:
        The concatenated samples; empty where no span holds any.
    """
    pieces = [cut_samples(samples, rate, span) for span in spans]
    pieces = [piece for piece in pieces if piece.size]
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=samples.dtype)


def pool_record(spans: Sequence[tuple[float, float]], *, min_s: float, reason_empty: str) -> dict[str, Any]:
    """A pool's record before embedding: its spans and seconds, and whether it is long enough to compare.

    Args:
        spans: The pool's spans.
        min_s: ``comparison.pooled_min_s``.
        reason_empty: The ``not_applicable`` reason where the pool holds nothing.

    Returns:
        ``status`` :data:`NOT_APPLICABLE` (no span), :data:`NOT_MEASURED` with ``reason``
        :data:`TOO_SHORT_FOR_SPEAKER_ID` (under ``min_s``), or :data:`MEASURED` pending a cosine; with
        ``spans``, ``spans_n`` and ``pooled_s``.
    """
    seconds = _length(spans)
    head = {
        "spans": [[round(a, 3), round(b, 3)] for a, b in spans],
        "spans_n": len(spans),
        "pooled_s": round(seconds, 3),
    }
    if not spans:
        return {"status": NOT_APPLICABLE, "reason": reason_empty, **head}
    if seconds < min_s:
        return {"status": NOT_MEASURED, "reason": TOO_SHORT_FOR_SPEAKER_ID, **head}
    return {"status": MEASURED, **head}


def pool_cosine_min(family: str, pool: str) -> float | None:
    """The cosine under which a pool does not match the enrollment (``comparison.cosine_min``).

    Args:
        family: The declared task family.
        pool: :data:`POOL_TARGET` or :data:`POOL_NON_TASK`.

    Returns:
        ``task_text`` for the target pool of a family compared on its task text (loudness), else ``default``.
        None where the bound is null: the cosine is recorded and decides nothing.
    """
    bounds = cohort_parameters()["comparison"]["cosine_min"]
    key = TARGET_TASK_TEXT if pool == POOL_TARGET and target_rule(family) == TARGET_TASK_TEXT else "default"
    value = bounds.get(key)
    return None if value is None else float(value)


def compare_speaker(
    facts: RecordingFacts,
    loader: AudioLoader,
    enrollment: Enrollment,
    embed: Embedder,
    *,
    min_s: float,
    lexical_gap_s: float,
    source: str = SOURCE_SESSION,
    participant: Enrollment | None = None,
) -> dict[str, Any]:
    """Compare one recording's pooled target speech and pooled speech outside the task with an enrollment.

    Each pool's spans are concatenated on the plain stream and embedded once (the enrollment's windows); a pool
    under ``min_s`` is not embedded. A pool matches at or above its :func:`pool_cosine_min`.

    Args:
        facts: The recording's facts.
        loader: Its plain-stream loader.
        enrollment: The enrollment compared against, with a vector.
        embed: The span embedder.
        min_s: ``comparison.pooled_min_s``.
        lexical_gap_s: Largest gap two words may straddle and stay one span.
        source: Where ``enrollment`` came from (:data:`SOURCE_SESSION` or
            :data:`SOURCE_PARTICIPANT_OTHER_SESSIONS`), recorded.
        participant: A second enrollment each pool is also compared with, reported as ``participant_cosine``;
            it decides nothing.

    Returns:
        The ``other_speaker`` block: one record per pool (:data:`POOL_TARGET`, :data:`POOL_NON_TASK`), the
        non-matching pools and their merged spans.
    """
    if target_rule(facts.family) == TARGET_NONE:
        target: dict[str, Any] = {"status": NOT_APPLICABLE, "reason": NO_TARGET_COMPARISON}
    else:
        target = pool_record(
            target_spans(facts, lexical_gap_s=lexical_gap_s), min_s=min_s, reason_empty=TOO_SHORT_FOR_SPEAKER_ID
        )
        if target["status"] == NOT_APPLICABLE:
            target["status"] = NOT_MEASURED
    pools = {
        POOL_TARGET: target,
        POOL_NON_TASK: pool_record(
            non_task_spans(facts, lexical_gap_s=lexical_gap_s), min_s=min_s, reason_empty=NO_SPEECH_OUTSIDE_TASK
        ),
    }
    head = {
        "enrollment_source": source,
        "enrollment": enrollment.record(),
        "participant_enrollment": None if participant is None else participant.record(),
    }
    wanted = [name for name, pool in pools.items() if pool["status"] == MEASURED]
    loaded = loader() if wanted else None
    if wanted and loaded is None:
        return {"status": UNAVAILABLE, "reason": PLAIN_UNREADABLE, **head, **pools}
    for name in wanted:
        pool = pools[name]
        samples, rate = loaded  # type: ignore[misc]
        vector = embed(pooled_samples(samples, rate, [(float(a), float(b)) for a, b in pool["spans"]]), rate)
        if vector is None or enrollment.vector is None:
            pools[name] = {**pool, "status": UNAVAILABLE, "reason": NOT_EMBEDDED}
            continue
        cosine = float(vector @ enrollment.vector)
        bound = pool_cosine_min(facts.family, name)
        pool.update(cosine=round(cosine, 4), cosine_min=bound, match=None if bound is None else cosine >= bound)
        if participant is not None and participant.vector is not None:
            pool["participant_cosine"] = round(float(vector @ participant.vector), 4)
    nonmatch = [name for name, pool in pools.items() if pool["status"] == MEASURED and pool["match"] is False]
    spans = merge([(float(a), float(b)) for name in nonmatch for a, b in pools[name]["spans"]])
    return {
        "status": MEASURED,
        **head,
        "model_id": enrollment.model_id,
        "model_commit_sha": enrollment.model_commit_sha,
        "pooled_min_s": min_s,
        **pools,
        "nonmatch": nonmatch,
        "nonmatch_s": round(_length(spans), 3),
        "nonmatch_spans": [[round(a, 3), round(b, 3)] for a, b in spans],
    }


# ------------------------------------------------------------------------------ distribution checks


def _quantile_block(values: Sequence[float], quantiles: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {f"q{q:g}": round(float(np.quantile(array, q)), 4) for q in quantiles}


def family_quantiles(rows: Sequence[Mapping[str, Any]], quantiles: Sequence[float]) -> dict[str, dict[str, Any]]:
    """Per-family quantiles of recording duration and task-extent fraction.

    Args:
        rows: :meth:`RecordingFacts.fact_row` per recording.
        quantiles: The quantiles taken, each in [0, 1].

    Returns:
        Per family: ``n``, ``median_duration_s``, and a quantile block for each quantity.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("duration_s"):
            grouped.setdefault(str(row["family"]), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for family, members in sorted(grouped.items()):
        durations = [float(r["duration_s"]) for r in members]
        fractions = [float(r["extent_fraction"]) for r in members if r.get("extent_fraction") is not None]
        out[family] = {
            "n": len(members),
            "median_duration_s": round(float(np.median(durations)), 4),
            "recording_duration_s": _quantile_block(durations, quantiles),
            "task_extent_fraction": _quantile_block(fractions, quantiles) if fractions else {},
        }
    return out


def build_quantile_artefact(
    rows: Sequence[Mapping[str, Any]], *, source: Mapping[str, Any], commit: str | None, created_at: str | None
) -> dict[str, Any]:
    """The cohort quantile artefact for one run.

    Args:
        rows: :meth:`RecordingFacts.fact_row` per recording.
        source: What the rows were read from, recorded as given.
        commit: The code revision.
        created_at: ISO-8601 time, or None.

    Returns:
        The artefact: kind, version, provenance, the quantiles taken and the per-family blocks.
    """
    params = cohort_parameters()
    quantiles = [float(q) for q in params["quantiles"]]
    return {
        "kind": QUANTILES_KIND,
        "version": int(params["version"]),
        "commit": commit,
        "created_at": created_at,
        "source": dict(source),
        "rows_n": len(rows),
        "rows_sha256": _digest(sorted((str(r.get("stem")), r.get("duration_s"), r.get("extent_s")) for r in rows)),
        "quantiles": quantiles,
        "checks": params["checks"],
        "families": family_quantiles(rows, quantiles),
    }


def write_quantile_artefact(artefact: Mapping[str, Any], path: Path) -> str:
    """Write the artefact as JSON and return the digest a reading references it by.

    Args:
        artefact: :func:`build_quantile_artefact`'s result.
        path: Where it goes.

    Returns:
        The file's SHA-256.
    """
    data = (json.dumps(artefact, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_quantile_artefact(path: Path) -> tuple[dict[str, Any], str]:
    """Read an artefact back with its digest.

    Args:
        path: The JSON file.

    Returns:
        The artefact and its SHA-256.

    Raises:
        ValueError: If the file is not a cohort quantile artefact.
    """
    data = path.read_bytes()
    artefact = json.loads(data)
    if artefact.get("kind") != QUANTILES_KIND:
        raise ValueError(f"{path} is not a {QUANTILES_KIND} artefact")
    return artefact, hashlib.sha256(data).hexdigest()


def duration_outlier(facts: RecordingFacts, level: Mapping[str, Any], bounds: Mapping[str, Any]) -> dict[str, Any]:
    """``recording_duration_outlier``: far longer than the family's median, with the task covering little.

    Args:
        facts: The recording's facts.
        level: The family's block of the quantile artefact.
        bounds: The check's block of ``data/cohort_stage.yaml``.

    Returns:
        The check's block: its comparisons and outcome.
    """
    if not facts.duration_s:
        return {"status": UNAVAILABLE, "reason": NO_DURATION}
    median = float(level["median_duration_s"])
    ratio = float(facts.duration_s) / median if median > 0 else math.inf
    fraction = float(facts.extent_fraction or 0.0)
    ratio_max, fraction_min = float(bounds["duration_ratio_max"]), float(bounds["extent_fraction_min"])
    outlier = ratio > ratio_max and fraction < fraction_min
    return {
        "status": MEASURED,
        "ground": str(bounds["ground"]),
        "outcome": REVIEW if outlier else PASS,
        "why": (
            f"{facts.duration_s:.1f} s is {ratio:.2f} x the family median {median:.1f} s and the task covers "
            f"{fraction:.2f} of it"
        ),
        "family_median_s": median,
        "comparisons": [
            {
                "name": "duration_over_family_median",
                "value": round(ratio, 3),
                "unit": None,
                "comparison": "<=",
                "threshold": ratio_max,
            },
            {
                "name": "task_extent_fraction",
                "value": round(fraction, 4),
                "unit": None,
                "comparison": ">=",
                "threshold": fraction_min,
            },
        ],
    }


CHECK_FUNCTIONS: dict[str, Callable[[RecordingFacts, Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]] = {
    "recording_duration_outlier": duration_outlier,
}
"""Each distribution check by name: ``(facts, level block, bounds) -> check block``."""


def _level_key(facts: RecordingFacts, level: str) -> str:
    if level == FAMILY_LEVEL:
        return facts.family
    raise ValueError(f"unknown cohort check level {level!r}")


def run_checks(facts: RecordingFacts, artefact: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Every configured distribution check over one recording.

    Args:
        facts: The recording's facts.
        artefact: The quantile artefact, or None where none was built.

    Returns:
        Each check's block, by name.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, bounds in cohort_parameters()["checks"].items():
        kinds = bounds.get("kinds")
        if kinds and not any(facts.family in DECLARED_KIND[str(kind)] for kind in kinds):
            out[name] = {"status": NOT_APPLICABLE}
            continue
        if artefact is None:
            out[name] = {"status": UNAVAILABLE, "reason": NO_QUANTILES}
            continue
        level = (artefact.get("families") or {}).get(_level_key(facts, str(bounds["level"])))
        if not level:
            out[name] = {"status": UNAVAILABLE, "reason": FAMILY_NOT_IN_QUANTILES}
            continue
        out[name] = CHECK_FUNCTIONS[name](facts, level, bounds)
    return out


# ------------------------------------------------------------------------------ the reading


def cohort_key(
    *,
    group: str | None,
    members: Sequence[RecordingFacts],
    quantiles_sha256: str | None,
    enrollment_config: Mapping[str, Any],
) -> str:
    """What a participant's readings were computed from, so a repeated pass can tell it has nothing to do.

    Args:
        group: The participant label (every session of it is one group).
        members: Every member's facts: stem, duration, task extent, diarization and lexical runs.
        quantiles_sha256: The artefact's digest, or None.
        enrollment_config: The model, its commit, the cut and the floor.

    Returns:
        A digest of those, the parameters file and the parameters version.
    """
    return _digest(
        {
            "version": cohort_parameters()["version"],
            "parameters": cohort_parameters(),
            "voice_task_text": task_speech_parameters().get("voice_task_text"),
            "group": group,
            "members": sorted(dataclasses.astuple(facts) for facts in members),
            "quantiles": quantiles_sha256,
            "enrollment": dict(enrollment_config),
        }
    )[:32]


def cohort_attributes(
    facts: RecordingFacts,
    *,
    other_speaker: Mapping[str, Any],
    checks: Mapping[str, Any],
    quantiles: Mapping[str, Any] | None,
    key: str | None,
) -> dict[str, Any]:
    """The ``cohort_reading`` attributes for one recording.

    Args:
        facts: The recording's facts.
        other_speaker: The enrollment-match block.
        checks: The distribution checks' blocks.
        quantiles: ``{path, sha256, version}`` of the artefact read, or None.
        key: :func:`cohort_key`, or None for a run that saw no cohort.

    Returns:
        The attributes.
    """
    measured = other_speaker.get("status") == MEASURED or any(c.get("status") == MEASURED for c in checks.values())
    return {
        "version": int(cohort_parameters()["version"]),
        "status": MEASURED if measured else UNAVAILABLE,
        "session": facts.session,
        "family": facts.family,
        "lexical_task": facts.lexical_task,
        OTHER_SPEAKER: dict(other_speaker),
        CHECKS: {name: dict(block) for name, block in checks.items()},
        "quantiles": dict(quantiles) if quantiles else None,
        "cohort_key": key,
    }


def write_cohort_reading(store: ProvStore, attributes: Mapping[str, Any]) -> str:
    """Write a ``cohort_reading``, retiring a live one it replaces.

    Writing the reading that already stands mints nothing new.

    Args:
        store: The recording's store.
        attributes: :func:`cohort_attributes`'s result.

    Returns:
        The reading's id.
    """
    software = software_agent(store)
    activity = store.activity(
        node=COHORT_NODE,
        step=COHORT_READING,
        parameters={"version": attributes.get("version"), "cohort_key": attributes.get("cohort_key")},
    )
    store.was_associated_with(activity, software)
    held = find_measurements(store, COHORT_READING)
    new_id = write_measurement(
        store, activity, software, name=COHORT_READING, signal=RECORDING_STREAM, attributes=dict(attributes)
    )
    for entity in held:
        if entity.id == new_id:
            continue
        retire = store.activity(
            node=COHORT_NODE,
            step="cohort_reading_superseded",
            parameters={"superseded": entity.id, "by": new_id},
        )
        store.was_associated_with(retire, software)
        store.used(retire, entity.id)
        store.was_invalidated_by(entity.id, retire)
    return new_id


def write_enrollment_entity(store: ProvStore, enrollment: Enrollment, reading_id: str) -> str:
    """Record the enrollment a reading compared against, without its vector.

    Args:
        store: The recording's store.
        enrollment: The session's own enrollment, or the participant's from the other sessions.
        reading_id: The ``cohort_reading`` derived from it.

    Returns:
        The enrollment entity's id.
    """
    software = software_agent(store)
    record = enrollment.record()
    activity = store.activity(
        node=COHORT_NODE,
        step=f"{enrollment.scope}_enrollment",
        parameters={"key": enrollment.key, "excluded_session": enrollment.excluded_session},
    )
    store.was_associated_with(activity, software)
    entity_id = store.entity(
        prov_type="enrollment",
        extent=None,
        attributes={
            "subject_id": participant_key(enrollment.key),
            "scope": enrollment.scope,
            "sessions": record["sessions"],
            "excluded_session": enrollment.excluded_session,
            "model_id": record["model_id"],
            "model_commit_sha": record["model_commit_sha"],
            "method": record["method"],
            "sources": record["recordings"],
            "vector_sha256": record["vector_sha256"],
        },
    )
    if store.generated_by(entity_id) is None:
        store.was_generated_by(entity_id, activity)
        store.was_attributed_to(entity_id, software)
    store.was_derived_from(reading_id, entity_id)
    return entity_id


def unavailable_attributes(facts: RecordingFacts) -> dict[str, Any]:
    """The reading of a run that saw no cohort: every check unavailable.

    Args:
        facts: The recording's facts.

    Returns:
        The attributes.
    """
    checks = {**run_checks(facts, None), CONSISTENCY_CHECK: {"status": UNAVAILABLE}}
    return cohort_attributes(
        facts,
        other_speaker={"status": UNAVAILABLE, "reason": NO_COHORT, "enrollment_source": SOURCE_NONE},
        checks={
            name: {**block, "reason": NO_COHORT} if block["status"] == UNAVAILABLE else block
            for name, block in checks.items()
        },
        quantiles=None,
        key=None,
    )


def write_cohort_unavailable(store: ProvStore) -> str:
    """COHORT on a single-file run: write ``unavailable``, unless a corpus pass's reading already stands.

    Args:
        store: The recording's store.

    Returns:
        The id of the reading that stands.
    """
    held = find_measurement(store, COHORT_READING)
    if held is not None:
        return held.id
    return write_cohort_reading(store, unavailable_attributes(facts_of(store, None)))


@dataclass(frozen=True)
class EnrollmentConfig:
    """The model, commit and cut the enrollment and the probes share.

    Attributes:
        model_id: ``speech.enrollment_model.model_id``, or None.
        commit: ``speech.enrollment_model.revision``, or None.
        cut: ``speech.target_match_cosine``, or None.
        min_s: ``comparison.pooled_min_s``: the shortest pool compared and the shortest enrollment span.
        lexical_gap_s: ``branch.run_gap_max_s``.
    """

    model_id: str | None
    commit: str | None
    cut: float | None
    min_s: float
    lexical_gap_s: float

    @property
    def usable(self) -> bool:
        """Whether every value an enrollment needs is set."""
        return self.model_id is not None and self.commit is not None and self.cut is not None

    def record(self) -> dict[str, Any]:
        """The configuration, JSON-ready.

        Returns:
            Its fields.
        """
        return {
            "model_id": self.model_id,
            "commit": self.commit,
            "cut": self.cut,
            "min_s": self.min_s,
            "lexical_gap_s": self.lexical_gap_s,
        }


def enrollment_config(config: TriageConfig) -> EnrollmentConfig:
    """Read the enrollment's settings off the triage configuration.

    Args:
        config: The triage configuration.

    Returns:
        The settings; any unset one leaves the configuration unusable rather than raising.
    """
    model = config.get("speech.enrollment_model")
    cut = config.get("speech.target_match_cosine")
    comparison = cohort_parameters()["comparison"]
    gap_key = str(comparison["lexical_gap_key"])
    return EnrollmentConfig(
        model_id=None if not isinstance(model, Mapping) else str(model.get("model_id")),
        commit=None if not isinstance(model, Mapping) else str(model.get("revision")),
        cut=None if cut is None else float(cut),
        min_s=float(comparison["pooled_min_s"]),
        lexical_gap_s=float(config.require(gap_key)),
    )


def consistency_bound() -> float | None:
    """``participant.consistency_cosine_min`` from ``data/cohort_stage.yaml``.

    Returns:
        The bound, or None where it is unset.
    """
    value = (cohort_parameters().get("participant") or {}).get("consistency_cosine_min")
    return None if value is None else float(value)


def read_participant(
    participant: str,
    sessions: Sequence[tuple[str, Sequence[tuple[RecordingFacts, AudioLoader]]]],
    *,
    settings: EnrollmentConfig,
    embed: Embedder | None,
    artefact: Mapping[str, Any] | None,
    quantiles_ref: Mapping[str, Any] | None,
) -> tuple[ParticipantEnrollments | None, dict[str, list[dict[str, Any]]]]:
    """COHORT over one participant: the enrollments, then every member's reading, session by session.

    A session that enrolled compares its members against its own enrollment and reports each run's
    cosine against the other sessions' too; one that did not falls back to the other sessions'.

    Args:
        participant: The participant label.
        sessions: ``(session key, members)`` per session, each member with its plain-stream loader.
        settings: The enrollment settings.
        embed: The span embedder, or None where the settings are unusable.
        artefact: The quantile artefact, or None.
        quantiles_ref: ``{path, sha256, version}`` of that artefact, or None.

    Returns:
        The enrollments (None where the settings are unusable) and, per session, one attribute mapping
        per member, in order.
    """
    key = cohort_key(
        group=participant,
        members=[facts for _, members in sessions for facts, _ in members],
        quantiles_sha256=None if quantiles_ref is None else str(quantiles_ref.get("sha256")),
        enrollment_config=settings.record(),
    )
    enrollments: ParticipantEnrollments | None = None
    if settings.usable and embed is not None:
        enrollments = enroll_participant(
            participant,
            sessions,
            embed,
            model_id=str(settings.model_id),
            commit=str(settings.commit),
            cut=float(settings.cut or 0.0),
            min_s=settings.min_s,
            lexical_gap_s=settings.lexical_gap_s,
        )
    bound = consistency_bound()
    out: dict[str, list[dict[str, Any]]] = {}
    for session, members in sessions:
        readings: list[dict[str, Any]] = []
        if enrollments is None:
            source, enrollment, second = SOURCE_NONE, None, None
            consistency: dict[str, Any] = {"status": UNAVAILABLE, "reason": ENROLLMENT_UNMEASURED}
        else:
            source, enrollment = enrollments.source_for(session)
            second = enrollments.other_sessions.get(session) if source == SOURCE_SESSION else None
            consistency = consistency_check(enrollments, session, bound=bound)
        for facts, loader in members:
            if enrollments is not None:
                source, enrollment = enrollments.source_for_recording(session, facts.stem)
            if enrollment is None:
                block: dict[str, Any] = {
                    "status": UNAVAILABLE,
                    "reason": ENROLLMENT_UNMEASURED,
                    "enrollment_source": SOURCE_NONE,
                }
            elif source == SOURCE_NONE:
                block = {
                    "status": UNAVAILABLE,
                    "reason": enrollment.reason,
                    "enrollment_source": SOURCE_NONE,
                    "enrollment": enrollment.record(),
                }
            else:
                block = compare_speaker(
                    facts,
                    loader,
                    enrollment,
                    embed if embed is not None else (lambda samples, rate: None),
                    min_s=settings.min_s,
                    lexical_gap_s=settings.lexical_gap_s,
                    source=source,
                    participant=second if source == SOURCE_SESSION else None,
                )
            checks = {**run_checks(facts, artefact), CONSISTENCY_CHECK: consistency}
            readings.append(
                cohort_attributes(facts, other_speaker=block, checks=checks, quantiles=quantiles_ref, key=key)
            )
        out[session] = readings
    return enrollments, out
