"""COHORT: readings that need more than one recording, written into each recording's store before VERDICT.

Two families of reading, over what the per-recording stages already stored:

- **Session speaker enrollment.** One speaker vector per participant and session, pooled from the
  cleanest single-speaker spans of that session's lexical speech tasks. Every recording's speech --
  its diarized segments on the enhanced stream and its runs of lexical consensus words -- is then
  embedded run by run and compared with it. A run under the match cut is a non-match.
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
    MEASURED,
    NOT_APPLICABLE,
    OTHER_SPEAKER,
    UNAVAILABLE,
)
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.decision import PASS, REVIEW
from senselab.audio.workflows.triage.live_evidence import declared_task, recording_stem
from senselab.audio.workflows.triage.nodes.background import session_key
from senselab.audio.workflows.triage.nodes.branches import derivative_arrays, merge
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    find_measurements,
    lexical_words,
    live_entities,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.routing_analysis.families import DECLARED_KIND, LEXICAL_SPEECH
from senselab.audio.workflows.triage.speaker_vectors import load_min_extent_s
from senselab.audio.workflows.triage.vocabulary import standing_task_extents
from senselab.utils.prov_store import ProvStore

PARAMETERS_PATH = Path(__file__).parent / "data" / "cohort_stage.yaml"

RECORDING_STREAM = "recording"
PLAIN_STREAM = "plain"
SOURCE_DIARIZATION = "diarization"
SOURCE_LEXICAL = "lexical"
SOURCE_WINDOW = "window"
QUANTILES_KIND = "cohort_quantiles"
QUANTILES_FILE = "cohort_quantiles.json"
FAMILY_LEVEL = "family"

NO_COHORT = "no cohort was read: the run saw this recording alone"
NO_ENROLLMENT_SPANS = "the session holds no lexical speech span long enough to enroll"
ENROLLMENT_UNMEASURED = "speech.enrollment_model or speech.target_match_cosine is unset"
NO_QUANTILES = "no cohort quantiles were supplied"
FAMILY_NOT_IN_QUANTILES = "the recording's family has no quantiles in the cohort artefact"
NO_DURATION = "the store records no recording duration"
PLAIN_UNREADABLE = "the plain stream could not be read"

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
        lexical: The lexical consensus words' extents, in index order.
        speech: The speech-classified regions (:func:`speech_regions`), merged.
        task_events: The task events the branches read (breaths, coughs, phonation holds, syllables).
        active: BACKGROUND's activity regions on the plain stream.
    """

    stem: str
    session: str | None
    family: str
    lexical_task: bool
    duration_s: float | None
    task_extent: tuple[float, float] | None
    diarization: tuple[tuple[float, float, str], ...] = ()
    lexical: tuple[tuple[float, float], ...] = ()
    speech: tuple[tuple[float, float], ...] = ()
    task_events: tuple[tuple[float, float], ...] = ()
    active: tuple[tuple[float, float], ...] = ()

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


def _activity(store: ProvStore) -> list[tuple[float, float]]:
    reading = find_measurement(store, "background_model")
    regions = [] if reading is None else reading.attributes.get("regions") or []
    return merge([(float(r["start_s"]), float(r["end_s"])) for r in regions])


def active_s(facts: RecordingFacts, span: tuple[float, float]) -> float:
    """How much of a span BACKGROUND read as active.

    Args:
        facts: The recording's facts.
        span: ``(start, end)``.

    Returns:
        The active seconds inside it.
    """
    return _length(_intersect(list(facts.active), span))


def _task_events(store: ProvStore) -> list[tuple[float, float]]:
    from senselab.audio.workflows.triage.nodes.quality import task_events_of  # noqa: PLC0415

    return list(task_events_of(store)[1])


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
    words = tuple((float(w.extent[0]), float(w.extent[1])) for w in lexical_words(store) if w.extent is not None)
    speech: list[tuple[float, float]] = []
    if run_dir is not None:
        classes = cohort_parameters()["speech"]
        for name in classes["measurements"]:
            speech.extend(
                speech_regions(
                    _yamnet_windows(store, run_dir, str(name)), classes["labels"], float(classes["score_min"])
                )
            )
    return RecordingFacts(
        stem=stem,
        session=session_key(stem),
        family=family,
        lexical_task=family in LEXICAL_SPEECH,
        duration_s=_recording_duration(store),
        task_extent=hull,
        diarization=segments,
        lexical=words,
        speech=tuple(merge(speech)),
        task_events=tuple(merge(_task_events(store))),
        active=tuple(_activity(store)),
    )


@dataclass(frozen=True)
class Run:
    """One stretch of speech compared against the enrollment.

    Attributes:
        start: Its start, in seconds.
        end: Its end, in seconds.
        source: :data:`SOURCE_DIARIZATION` or :data:`SOURCE_LEXICAL`.
        speaker: The diarizer's label for a diarized run, else None.
    """

    start: float
    end: float
    source: str
    speaker: str | None = None

    @property
    def duration_s(self) -> float:
        """The run's length in seconds."""
        return self.end - self.start


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


def candidate_runs(facts: RecordingFacts, *, min_s: float, lexical_gap_s: float) -> list[Run]:
    """Every run of at least ``min_s`` the recording's speech offers for comparison.

    A diarized piece is compared only where it carries speech: its intersection with the lexical words
    and the speech-classified regions, less the task events.

    Args:
        facts: The recording's facts.
        min_s: The shortest run embedded.
        lexical_gap_s: Largest gap two lexical words may straddle and stay one run.

    Returns:
        The diarized speech pieces, then the lexical runs, each kept where it reaches ``min_s``.
    """
    speech = _subtract(merge([*facts.speech, *facts.lexical]), facts.task_events)
    runs = [
        Run(start, end, SOURCE_DIARIZATION, speaker)
        for piece_start, piece_end, speaker in exclusive_pieces(facts.diarization)
        for start, end in _intersect(speech, (piece_start, piece_end))
        if end - start >= min_s
    ]
    runs.extend(
        Run(start, end, SOURCE_LEXICAL)
        for start, end in lexical_runs_of(facts.lexical, lexical_gap_s)
        if end - start >= min_s
    )
    return runs


def window_regions(facts: RecordingFacts, *, min_s: float, lexical_gap_s: float) -> list[tuple[float, float]]:
    """The speech-classified time no compared run (:func:`candidate_runs`) or task event covers.

    Args:
        facts: The recording's facts.
        min_s: The shortest region kept.
        lexical_gap_s: Largest gap two lexical words may straddle and stay one run.

    Returns:
        The regions the sliding windows read, each at least ``min_s``.
    """
    covered = [
        *((run.start, run.end) for run in candidate_runs(facts, min_s=min_s, lexical_gap_s=lexical_gap_s)),
        *facts.task_events,
    ]
    return [(s, e) for s, e in _subtract(list(facts.speech), covered) if e - s >= min_s]


def window_runs(
    regions: Sequence[tuple[float, float]],
    cosine_at: Callable[[tuple[float, float]], float | None],
    *,
    window_s: float,
    hop_s: float,
    cut: float,
    active_fraction: Callable[[tuple[float, float]], float] = lambda span: 1.0,
    active_fraction_min: float = 0.0,
) -> list[dict[str, Any]]:
    """Sliding windows over each region, grouped into runs of contiguous windows on one side of the cut.

    Args:
        regions: The regions to read.
        cosine_at: The cosine of one window against the enrollment, or None where it cannot be embedded.
        window_s: The window length.
        hop_s: The hop.
        cut: The match cut.
        active_fraction: The share of a window BACKGROUND read as active.
        active_fraction_min: A window under this share is not embedded and ends the group it would join.

    Returns:
        One run record per group, its span the union of its windows and its cosine their mean.
    """
    out: list[dict[str, Any]] = []
    for start, end in regions:
        group: list[tuple[float, float, float, float]] = []
        t = start
        while t + window_s <= end + 1e-9:
            share = active_fraction((t, t + window_s))
            cosine = cosine_at((t, t + window_s)) if share >= active_fraction_min else None
            if cosine is None:
                if group:
                    out.append(_window_group(group, cut))
                    group = []
            else:
                if group and (cosine >= cut) != (group[-1][2] >= cut):
                    out.append(_window_group(group, cut))
                    group = []
                group.append((t, t + window_s, cosine, share))
            t += hop_s
        if group:
            out.append(_window_group(group, cut))
    return out


def _window_group(group: Sequence[tuple[float, float, float, float]], cut: float) -> dict[str, Any]:
    cosine = float(np.mean([c for _, _, c, _ in group]))
    start, end = group[0][0], group[-1][1]
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration_s": round(end - start, 3),
        "source": SOURCE_WINDOW,
        "speaker": None,
        "cosine": round(cosine, 4),
        "windows_n": len(group),
        "windows": [[round(a, 3), round(c, 4), round(f, 3)] for a, _, c, f in group],
        "match": group[0][2] >= cut,
    }


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
        import soundfile as sf  # noqa: PLC0415

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
            samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
        except (OSError, RuntimeError):
            return None
        return np.ascontiguousarray(samples[:, 0]), int(rate)

    return load


# ------------------------------------------------------------------------------ the enrollment


@dataclass
class SessionEnrollment:
    """One participant's speaker vector for one session.

    Attributes:
        session: The session key.
        vector: The unit vector, or None where the session could not enroll.
        reason: Why there is no vector, else None.
        model_id: The embedding model.
        model_commit_sha: Its resolved commit.
        recordings: The stems whose spans were pooled.
        left_out: Stems left out by the leave-one-out step.
        spans_n: How many spans were embedded.
        seconds: Their total length.
        single_speaker_only: Whether only single-speaker recordings enrolled.
    """

    session: str
    vector: np.ndarray | None
    reason: str | None
    model_id: str | None
    model_commit_sha: str | None
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
            "session": self.session,
            "status": MEASURED if self.vector is not None else UNAVAILABLE,
            "reason": self.reason,
            "model_id": self.model_id,
            "model_commit_sha": self.model_commit_sha,
            "method": "recording_equal_spherical_mean",
            "recordings": list(self.recordings),
            "recordings_n": len(self.recordings),
            "left_out": list(self.left_out),
            "spans_n": self.spans_n,
            "seconds": round(self.seconds, 3),
            "single_speaker_only": self.single_speaker_only,
            "vector_sha256": self.vector_sha256,
        }


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
) -> SessionEnrollment:
    """Pool one session's lexical speech spans into one speaker vector, recording-equal.

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
        The enrollment; one with no vector names why.
    """
    params = dict(parameters if parameters is not None else cohort_parameters()["enrollment"])
    candidates = [
        (facts, loader, enrollment_spans(facts, min_s=min_s, lexical_gap_s=lexical_gap_s))
        for facts, loader in members
        if facts.family in DECLARED_KIND[str(params["families"])]
    ]
    candidates = [(facts, loader, spans) for facts, loader, spans in candidates if spans.spans]
    single = [entry for entry in candidates if entry[2].single_speaker]
    single_only = len(single) >= int(params["single_speaker_recordings_min"])
    chosen = single if single_only else candidates
    centroids: list[tuple[str, np.ndarray, int, float]] = []
    for facts, loader, spans in chosen:
        loaded = loader()
        if loaded is None:
            continue
        samples, rate = loaded
        vectors = [v for v in (embed(cut_samples(samples, rate, span), rate) for span in spans.spans) if v is not None]
        if vectors:
            centroids.append((facts.stem, _unit(np.vstack(vectors).mean(axis=0)), len(vectors), _length(spans.spans)))
    if not centroids:
        return SessionEnrollment(session, None, NO_ENROLLMENT_SPANS, model_id, commit)
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
    vector = _unit(np.vstack([c[1] for c in centroids]).mean(axis=0))
    return SessionEnrollment(
        session,
        vector,
        None,
        model_id,
        commit,
        recordings=[c[0] for c in centroids],
        left_out=left_out,
        spans_n=sum(c[2] for c in centroids),
        seconds=sum(c[3] for c in centroids),
        single_speaker_only=single_only,
    )


def match_runs(
    facts: RecordingFacts,
    loader: AudioLoader,
    enrollment: SessionEnrollment,
    embed: Embedder,
    *,
    cut: float,
    min_s: float,
    lexical_gap_s: float,
) -> dict[str, Any]:
    """Compare every run of one recording's speech with the session enrollment.

    Args:
        facts: The recording's facts.
        loader: Its plain-stream loader.
        enrollment: The session's enrollment, with a vector.
        embed: The span embedder.
        cut: The match cut: a run at or above it matches.
        min_s: The shortest run embedded.
        lexical_gap_s: Largest gap two lexical words may straddle and stay one run.

    Returns:
        The ``other_speaker`` block: each run with its cosine and match, the counts, and the merged
        non-matching spans.
    """
    runs = candidate_runs(facts, min_s=min_s, lexical_gap_s=lexical_gap_s)
    windows = cohort_parameters()["windows"]
    gates = cohort_parameters()["runs"]
    regions = window_regions(facts, min_s=float(windows["window_s"]), lexical_gap_s=lexical_gap_s)
    loaded = loader() if runs or regions else None
    if (runs or regions) and loaded is None:
        return {"status": UNAVAILABLE, "reason": PLAIN_UNREADABLE, "enrollment": enrollment.record()}
    records: list[dict[str, Any]] = []
    if loaded is not None and enrollment.vector is not None:
        samples, rate = loaded
        for run in runs:
            active = active_s(facts, (run.start, run.end))
            if facts.active and active < float(gates["active_min_s"]):
                continue
            vector = embed(cut_samples(samples, rate, (run.start, run.end)), rate)
            if vector is None:
                continue
            cosine = float(vector @ enrollment.vector)
            records.append(
                {
                    "start": round(run.start, 3),
                    "end": round(run.end, 3),
                    "duration_s": round(run.duration_s, 3),
                    "source": run.source,
                    "speaker": run.speaker,
                    "active_s": round(active, 3),
                    "cosine": round(cosine, 4),
                    "match": cosine >= cut,
                }
            )

        def cosine_at(span: tuple[float, float]) -> float | None:
            vector = embed(cut_samples(samples, rate, span), rate)
            return None if vector is None or enrollment.vector is None else float(vector @ enrollment.vector)

        records.extend(
            window_runs(
                regions,
                cosine_at,
                window_s=float(windows["window_s"]),
                hop_s=float(windows["hop_s"]),
                cut=cut,
                active_fraction=lambda span: active_s(facts, span) / (span[1] - span[0]) if facts.active else 1.0,
                active_fraction_min=float(windows["active_fraction_min"]),
            )
        )
    nonmatch = [r for r in records if not r["match"]]
    spans = merge([(r["start"], r["end"]) for r in nonmatch])
    return {
        "status": MEASURED,
        "enrollment": enrollment.record(),
        "model_id": enrollment.model_id,
        "model_commit_sha": enrollment.model_commit_sha,
        "cut": cut,
        "min_run_s": min_s,
        "runs": records,
        "runs_n": len(records),
        "candidate_runs_n": len(runs),
        "window_regions_s": round(_length(regions), 3),
        "nonmatch_n": len(nonmatch),
        "nonmatch_s": round(_length(spans), 3),
        "nonmatch_spans": [[round(s, 3), round(e, 3)] for s, e in spans],
        "lowest_cosine": min((r["cosine"] for r in records), default=None),
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
    session: str | None,
    members: Sequence[RecordingFacts],
    quantiles_sha256: str | None,
    enrollment_config: Mapping[str, Any],
) -> str:
    """What a session's readings were computed from, so a repeated pass can tell it has nothing to do.

    Args:
        session: The session key.
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
            "session": session,
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


def write_enrollment_entity(store: ProvStore, enrollment: SessionEnrollment, reading_id: str) -> str:
    """Record the session enrollment a reading compared against, without its vector.

    Args:
        store: The recording's store.
        enrollment: The session enrollment.
        reading_id: The ``cohort_reading`` derived from it.

    Returns:
        The enrollment entity's id.
    """
    software = software_agent(store)
    record = enrollment.record()
    activity = store.activity(node=COHORT_NODE, step="session_enrollment", parameters={"session": enrollment.session})
    store.was_associated_with(activity, software)
    entity_id = store.entity(
        prov_type="enrollment",
        extent=None,
        attributes={
            "subject_id": enrollment.session.split("_", 1)[0],
            "session": enrollment.session,
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
    return cohort_attributes(
        facts,
        other_speaker={"status": UNAVAILABLE, "reason": NO_COHORT},
        checks={
            name: {**block, "reason": NO_COHORT} if block["status"] == UNAVAILABLE else block
            for name, block in run_checks(facts, None).items()
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
        min_s: The speaker-vector refusal floor.
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
    gap_key = str(cohort_parameters()["runs"]["lexical_gap_key"])
    return EnrollmentConfig(
        model_id=None if not isinstance(model, Mapping) else str(model.get("model_id")),
        commit=None if not isinstance(model, Mapping) else str(model.get("revision")),
        cut=None if cut is None else float(cut),
        min_s=load_min_extent_s(),
        lexical_gap_s=float(config.require(gap_key)),
    )


def read_session(
    session: str,
    members: Sequence[tuple[RecordingFacts, AudioLoader]],
    *,
    settings: EnrollmentConfig,
    embed: Embedder | None,
    artefact: Mapping[str, Any] | None,
    quantiles_ref: Mapping[str, Any] | None,
) -> tuple[SessionEnrollment | None, list[dict[str, Any]]]:
    """COHORT over one session: its enrollment, then every member's reading.

    Args:
        session: The session key.
        members: Every recording of the session, with its plain-stream loader.
        settings: The enrollment settings.
        embed: The span embedder, or None where the settings are unusable.
        artefact: The quantile artefact, or None.
        quantiles_ref: ``{path, sha256, version}`` of that artefact, or None.

    Returns:
        The enrollment (None where the settings are unusable) and one attribute mapping per member,
        in order.
    """
    key = cohort_key(
        session=session,
        members=[facts for facts, _ in members],
        quantiles_sha256=None if quantiles_ref is None else str(quantiles_ref.get("sha256")),
        enrollment_config=settings.record(),
    )
    enrollment: SessionEnrollment | None = None
    if settings.usable and embed is not None:
        enrollment = enroll_session(
            session,
            members,
            embed,
            model_id=str(settings.model_id),
            commit=str(settings.commit),
            cut=float(settings.cut or 0.0),
            min_s=settings.min_s,
            lexical_gap_s=settings.lexical_gap_s,
        )
    readings: list[dict[str, Any]] = []
    for facts, loader in members:
        if enrollment is None:
            block: dict[str, Any] = {"status": UNAVAILABLE, "reason": ENROLLMENT_UNMEASURED}
        elif enrollment.vector is None:
            block = {"status": UNAVAILABLE, "reason": enrollment.reason, "enrollment": enrollment.record()}
        else:
            block = match_runs(
                facts,
                loader,
                enrollment,
                embed if embed is not None else (lambda samples, rate: None),
                cut=float(settings.cut or 0.0),
                min_s=settings.min_s,
                lexical_gap_s=settings.lexical_gap_s,
            )
        readings.append(
            cohort_attributes(
                facts, other_speaker=block, checks=run_checks(facts, artefact), quantiles=quantiles_ref, key=key
            )
        )
    return enrollment, readings
