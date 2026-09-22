"""One speaker embedding per subject, pooled over the task extents that subject's recordings mint.

A row is a speaker, not a recording: :mod:`senselab.audio.workflows.triage.recording_vectors` is
the per-recording sibling and the two share their scan and shard conventions but nothing else.

The unit of input is the ``task_extent`` span -- the single interval a branch mints in align mode
to delimit the portion of a recording that serves its declared task. A recording that minted none
contributes nothing; there is no fallback to the whole file, because absence of the span is itself
the branch's record that it found no task.

The column list, the pooling rule and the refusal floor are specified in
``specs/20260922-speaker-vectors/schema.md``, which is the interface a reader decodes against. No
column here may change without that file changing with it.

A speaker embedding is a biometric derived from human-subject audio: the parquet is written mode
600, never committed, and never copied to a shared location.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa

SCHEMA_VERSION = 1
"""Bumped whenever a column is added, removed or retyped, or the pooling rule changes."""

STORE_NAME = "store.jsonl"
RUN_SUBDIR = "run"
STREAMS_SUBDIR = "streams"
EMBED_STREAM = "plain"
"""The stream every extent is cut from.

All PREPROCESS streams share the source recording's time axis, so an extent's seconds index any
of them identically. Why ``plain`` and not ``enhanced`` or ``redacted``:
``specs/20260922-speaker-vectors/design.md``.
"""

TASK_EXTENT_ROLE = "task_extent"
SPAN_TYPE = "span"
BRANCH_FAMILIES = ("airway", "speech", "voice")

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
WINDOW_S = 2.0
HOP_S = 1.0
AGGREGATOR = "spherical_mean"
EMBEDDING_DIM = 192

_TIMESTAMP = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")
_PARTICIPANT = re.compile(r"(sub-[^_]+)")
_SESSION = re.compile(r"(ses-[^_]+)")
_TASK = re.compile(r"(task-[^_]+)")

_PROFILE_PATH = Path(__file__).parent / "data" / "speaker_vector_profile" / "2026-09-22.yaml"
_PROFILE_VERSION = "1"


# ------------------------------------------------------------------------------------ the floor


def load_min_extent_s(path: Optional[Path] = None) -> float:
    """Load the shortest extent this module will embed, in seconds.

    Args:
        path: Profile path, or ``None`` for the bundled default.

    Returns:
        The floor in seconds.

    Raises:
        ValueError: If the profile is missing, its ``schema_version`` does not match, or
            ``min_extent_s`` is absent or not positive. There is no code-literal fallback: a
            refusal floor nobody fitted, reported as if it were measured, is worse than a crash.
    """
    import yaml  # type: ignore[import-untyped]

    target = Path(path) if path is not None else _PROFILE_PATH
    if not target.exists():
        raise ValueError(
            f"speaker-vector profile {target} is missing from this install. Refusing to "
            "substitute an unfitted floor and report it as measured; reinstall senselab or pass "
            "an explicit path."
        )
    doc = yaml.safe_load(target.read_text()) or {}
    if str(doc.get("schema_version")) != _PROFILE_VERSION:
        raise ValueError(f"{target}: schema_version {doc.get('schema_version')!r}, expected {_PROFILE_VERSION!r}")
    value = doc.get("min_extent_s")
    if not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"{target}: min_extent_s must be a positive number, got {value!r}")
    return float(value)


# ------------------------------------------------------------------------------------ store view


@dataclass(frozen=True)
class Extent:
    """One task extent, with everything needed to cut it and to account for it."""

    subject: str
    session: Optional[str]
    task: Optional[str]
    stem: str
    run_dir: str
    audio_path: Path
    start_s: float
    end_s: float
    family: Optional[str]
    production: Optional[str]

    @property
    def duration_s(self) -> float:
        """Length of the extent in seconds."""
        return self.end_s - self.start_s

    @property
    def extent_id(self) -> str:
        """A stable id for this extent, used as the file id in the pooled distribution."""
        return f"{self.stem}@{self.start_s:.3f}-{self.end_s:.3f}"


def stem_of(run_root: Path) -> str:
    """Return the BIDS stem of a run directory, with its timestamp removed.

    Args:
        run_root: A finished run directory.

    Returns:
        The directory name minus a trailing ``_YYYYmmdd-HHMMSS`` and any collision suffix.
    """
    return _TIMESTAMP.sub("", run_root.name)


def identity(stem: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Split a stem into participant, session and task labels.

    Args:
        stem: A BIDS stem.

    Returns:
        ``(participant, session, task)``, each ``None`` when the stem does not carry it.
    """
    p = _PARTICIPANT.search(stem)
    s = _SESSION.search(stem)
    t = _TASK.search(stem)
    return (p.group(1) if p else None, s.group(1) if s else None, t.group(1) if t else None)


def read_task_extents(run_root: Path) -> list[Extent]:
    """Read every live task extent a finished run minted.

    Args:
        run_root: A run directory holding ``run/store.jsonl``.

    Returns:
        The extents, deduplicated on their interval. Empty when the branch minted none, which is
        the branch's record that it found no task -- not an error.

    Raises:
        OSError: If the store cannot be read.
    """
    store = run_root / RUN_SUBDIR / STORE_NAME
    audio = run_root / RUN_SUBDIR / STREAMS_SUBDIR / f"{EMBED_STREAM}.flac"
    stem = stem_of(run_root)
    subject, session, task = identity(stem)
    if subject is None:
        return []

    invalidated: set[str] = set()
    candidates: dict[tuple[float, float], dict[str, Any]] = {}
    with store.open() as handle:
        for line in handle:
            if TASK_EXTENT_ROLE not in line and "wasInvalidatedBy" not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("record") == "relation" and record.get("relation") == "wasInvalidatedBy":
                invalidated.add(str(record.get("source")))
                continue
            if record.get("record") != "entity" or record.get("prov_type") != SPAN_TYPE:
                continue
            attributes = record.get("attributes") or {}
            if attributes.get("role") != TASK_EXTENT_ROLE:
                continue
            span = record.get("extent")
            if not span or len(span) != 2:
                continue
            key = (round(float(span[0]), 6), round(float(span[1]), 6))
            candidates[key] = {"id": str(record.get("id")), "attributes": attributes}

    out: list[Extent] = []
    for (start_s, end_s), found in sorted(candidates.items()):
        if found["id"] in invalidated:
            continue
        attributes = found["attributes"]
        out.append(
            Extent(
                subject=subject,
                session=session,
                task=task,
                stem=stem,
                run_dir=run_root.name,
                audio_path=audio,
                start_s=start_s,
                end_s=end_s,
                family=attributes.get("family"),
                production=attributes.get("production"),
            )
        )
    return out


# ------------------------------------------------------------------------------------- the scan


def _listdir(path: Path) -> list[Path]:
    """List a directory, treating an unreadable one as empty.

    Args:
        path: The directory.

    Returns:
        Its entries, or ``[]`` when it cannot be read -- a tree being written mid-scan is
        tolerated rather than raised.
    """
    try:
        return sorted(path.iterdir())
    except OSError:
        return []


def subject_dirs(root: Path) -> Iterator[Path]:
    """Yield the ``sub-*`` directories of a corpus tree.

    Args:
        root: The corpus root.

    Yields:
        Each subject directory.
    """
    for entry in _listdir(root):
        if entry.is_dir() and entry.name.startswith("sub-"):
            yield entry


def run_dirs_of(subject_dir: Path) -> Iterator[Path]:
    """Yield the run directories under one subject, newest run per stem only.

    Args:
        subject_dir: A ``sub-*`` directory.

    Yields:
        One run directory per stem: when a stem was run more than once, the latest timestamp.
    """
    latest: dict[str, Path] = {}
    for session in _listdir(subject_dir):
        if not session.is_dir():
            continue
        for run_root in _listdir(session):
            if not run_root.is_dir() or not (run_root / RUN_SUBDIR / STORE_NAME).exists():
                continue
            stem = stem_of(run_root)
            current = latest.get(stem)
            if current is None or run_root.name > current.name:
                latest[stem] = run_root
    yield from (latest[stem] for stem in sorted(latest))


def shard_of(subject: str, slices: int) -> int:
    """Assign a subject to a shard by content, so a growing tree never reshuffles.

    Args:
        subject: The ``sub-*`` label.
        slices: How many workers share the tree.

    Returns:
        The shard index.
    """
    if slices <= 1:
        return 0
    digest = hashlib.sha1(subject.encode(), usedforsecurity=False).hexdigest()[:8]
    return int(digest, 16) % slices


@dataclass
class ScanReport:
    """What a shard saw, so a thin result is never mistaken for a clean one."""

    subjects_considered: int = 0
    subjects_written: int = 0
    subjects_without_extent: list[str] = field(default_factory=list)
    subjects_all_refused: list[str] = field(default_factory=list)
    subjects_failed: dict[str, str] = field(default_factory=dict)
    recordings_seen: int = 0
    recordings_with_extent: int = 0
    recordings_unreadable: int = 0
    extents_admitted: int = 0
    extents_refused_short: int = 0
    extents_missing_audio: int = 0


def gather(root: Path, slice_index: int = 0, slices: int = 1) -> tuple[dict[str, list[Extent]], ScanReport]:
    """Collect this shard's task extents, grouped by subject.

    Args:
        root: The corpus root, a tree of ``sub-*/ses-*/<stem>_<stamp>/`` run directories.
        slice_index: This worker's index.
        slices: How many workers share the tree.

    Returns:
        ``({subject: [Extent, ...]}, report)``. Extents below the profile floor, and those whose
        stream file is missing, are excluded and counted.
    """
    floor = load_min_extent_s()
    report = ScanReport()
    grouped: dict[str, list[Extent]] = {}
    for subject_dir in subject_dirs(root):
        subject = subject_dir.name
        if shard_of(subject, slices) != slice_index:
            continue
        report.subjects_considered += 1
        admitted: list[Extent] = []
        saw_any = False
        for run_root in run_dirs_of(subject_dir):
            report.recordings_seen += 1
            try:
                extents = read_task_extents(run_root)
            except OSError:
                report.recordings_unreadable += 1
                continue
            if extents:
                report.recordings_with_extent += 1
            for extent in extents:
                saw_any = True
                if extent.duration_s < floor:
                    report.extents_refused_short += 1
                    continue
                if not extent.audio_path.exists():
                    report.extents_missing_audio += 1
                    continue
                admitted.append(extent)
        if admitted:
            grouped[subject] = admitted
            report.extents_admitted += len(admitted)
        elif saw_any:
            report.subjects_all_refused.append(subject)
        else:
            report.subjects_without_extent.append(subject)
    return grouped, report


# --------------------------------------------------------------------------------------- the row


def row_for(subject: str, extents: Sequence[Extent], embedding: Any, root: Path) -> dict[str, Any]:  # noqa: ANN401
    """Build the parquet row for one speaker.

    Args:
        subject: The ``sub-*`` label.
        extents: The extents that were handed to the estimator, in the order they were handed.
        embedding: The ``TargetSpeakerEmbedding`` the estimator returned.
        root: The corpus root, so ``run_dir`` can be recorded relative to it.

    Returns:
        A mapping keyed by column name. Every count is a real count: a statistic that could not be
        computed is ``None``, never a substituted zero.
    """
    provenance = embedding.provenance
    distribution = embedding.distribution
    by_id = {e.extent_id: e for e in extents}
    contributed = [eid for eid in provenance.source_files if eid not in provenance.extraction_failures]

    per_file = distribution.counts.vectors_per_file
    loo = distribution.centroid_robustness.leave_one_file_out_cos
    to_pooled = distribution.cross_file.cos_file_centroid_to_pooled
    pairwise = distribution.cross_file.file_centroid_pairwise_cos

    vector = np.asarray(embedding.vector, dtype=np.float64).reshape(-1)
    seconds = sum(by_id[eid].duration_s for eid in contributed if eid in by_id)

    row: dict[str, Any] = {
        "speaker_id": subject,
        "vector": [float(v) for v in vector],
        "dim": int(vector.size),
        "schema_version": SCHEMA_VERSION,
        "model_id": provenance.model_id,
        "model_commit_sha": provenance.model_commit_sha,
        "unresolved_reason": provenance.unresolved_reason,
        "method": provenance.method,
        "window_s": provenance.window_s,
        "hop_s": provenance.hop_s,
        "n_extents": len(contributed),
        "n_recordings": len({by_id[eid].stem for eid in contributed if eid in by_id}),
        "n_sessions": len({by_id[eid].session for eid in contributed if eid in by_id}),
        "extent_seconds": float(seconds),
        "n_windows_used": int(provenance.n_windows_used),
        "n_windows_dropped": int(provenance.n_windows_dropped),
        "n_effective_windows": distribution.counts.n_effective,
        "n_extents_failed": len(provenance.extraction_failures),
        "rbar": float(distribution.rbar),
        "rbar_null": float(distribution.nulls.rbar_null),
        "cos_to_centroid_loo_q05": float(distribution.cos_to_centroid_loo.q05),
        "cos_to_centroid_loo_q50": float(distribution.cos_to_centroid_loo.q50),
        "cos_extent_centroid_to_pooled_min": min(to_pooled.values()) if to_pooled else None,
        "cos_extent_pairwise_q50": float(pairwise.q50) if pairwise is not None else None,
        "auc_same_extent_vs_diff_extent": distribution.file_effect.auc_same_file_vs_diff_file,
        "cos_mean_vs_trimmed10": float(distribution.centroid_robustness.cos_mean_vs_trimmed10),
        "cos_mean_vs_medoid": float(distribution.centroid_robustness.cos_mean_vs_medoid),
        "leave_one_extent_out_cos_min": min(loo.values()) if loo else None,
        "participation_ratio": float(distribution.spectrum.participation_ratio),
        "participation_ratio_null": float(distribution.nulls.participation_ratio_null),
        "pc1_share_centred": float(distribution.spectrum.pc1_share_centred),
    }

    stems: list[str] = []
    run_dirs: list[str] = []
    sessions: list[Optional[str]] = []
    tasks: list[Optional[str]] = []
    families: list[Optional[str]] = []
    starts: list[float] = []
    ends: list[float] = []
    windows: list[Optional[int]] = []
    loo_cos: list[Optional[float]] = []
    pooled_cos: list[Optional[float]] = []
    for eid in contributed:
        extent = by_id.get(eid)
        if extent is None:
            continue
        stems.append(extent.stem)
        run_dirs.append(str(Path(extent.run_dir)))
        sessions.append(extent.session)
        tasks.append(extent.task)
        families.append(extent.family)
        starts.append(extent.start_s)
        ends.append(extent.end_s)
        windows.append(int(per_file[eid]) if eid in per_file else None)
        loo_cos.append(float(loo[eid]) if eid in loo else None)
        pooled_cos.append(float(to_pooled[eid]) if eid in to_pooled else None)
    row.update(
        {
            "extent_stem": stems,
            "extent_run_dir": run_dirs,
            "extent_session": sessions,
            "extent_task": tasks,
            "extent_family": families,
            "extent_start_s": starts,
            "extent_end_s": ends,
            "extent_windows_n": windows,
            "extent_leave_one_out_cos": loo_cos,
            "extent_centroid_to_pooled_cos": pooled_cos,
            "corpus_root": str(root),
        }
    )
    return row


# ------------------------------------------------------------------------------------- the schema

_STRING_COLUMNS = (
    "speaker_id",
    "model_id",
    "model_commit_sha",
    "unresolved_reason",
    "method",
    "corpus_root",
)
_INT_COLUMNS = (
    "dim",
    "schema_version",
    "n_extents",
    "n_recordings",
    "n_sessions",
    "n_windows_used",
    "n_windows_dropped",
    "n_extents_failed",
)
_FLOAT_COLUMNS = (
    "window_s",
    "hop_s",
    "extent_seconds",
    "n_effective_windows",
    "rbar",
    "rbar_null",
    "cos_to_centroid_loo_q05",
    "cos_to_centroid_loo_q50",
    "cos_extent_centroid_to_pooled_min",
    "cos_extent_pairwise_q50",
    "auc_same_extent_vs_diff_extent",
    "cos_mean_vs_trimmed10",
    "cos_mean_vs_medoid",
    "leave_one_extent_out_cos_min",
    "participation_ratio",
    "participation_ratio_null",
    "pc1_share_centred",
)
_STRING_LIST_COLUMNS = (
    "extent_stem",
    "extent_run_dir",
    "extent_session",
    "extent_task",
    "extent_family",
)
_FLOAT_LIST_COLUMNS = (
    "vector",
    "extent_start_s",
    "extent_end_s",
    "extent_leave_one_out_cos",
    "extent_centroid_to_pooled_cos",
)
_INT_LIST_COLUMNS = ("extent_windows_n",)


def schema() -> pa.Schema:
    """Return the parquet schema, with the schema version in its metadata.

    Returns:
        The arrow schema. Identity first, then the vector, then provenance, then the counts a
        reader needs to weight a row, then the per-extent parallel lists.
    """
    fields = [pa.field(name, pa.string()) for name in _STRING_COLUMNS]
    fields += [pa.field(name, pa.int32()) for name in _INT_COLUMNS]
    fields += [pa.field(name, pa.float64()) for name in _FLOAT_COLUMNS]
    fields += [pa.field(name, pa.list_(pa.float64())) for name in _FLOAT_LIST_COLUMNS]
    fields += [pa.field(name, pa.list_(pa.string())) for name in _STRING_LIST_COLUMNS]
    fields += [pa.field(name, pa.list_(pa.int32())) for name in _INT_LIST_COLUMNS]
    ordered = ["speaker_id", "vector", "dim", "n_extents", "n_recordings", "extent_seconds"]
    by_name = {f.name: f for f in fields}
    rest = [f for f in fields if f.name not in ordered]
    return pa.schema(
        [by_name[name] for name in ordered] + rest,
        metadata={b"senselab.speaker_vectors.schema_version": str(SCHEMA_VERSION).encode()},
    )


def to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    """Build an arrow table, substituting nothing.

    Args:
        rows: Rows from :func:`row_for`.

    Returns:
        A table with exactly :func:`schema`. A key a row does not carry becomes null, never a
        default.
    """
    target = schema()
    columns = [pa.array([row.get(f.name) for row in rows], type=f.type) for f in target]
    return pa.Table.from_arrays(columns, schema=target)


# ------------------------------------------------------------------------------------ the estimate


def embed_subject(
    subject: str,
    extents: Sequence[Extent],
    root: Path,
    *,
    device: Any = None,  # noqa: ANN401 -- a senselab DeviceType, imported lazily with the backend
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Pool one subject's task extents into a single speaker vector.

    Each extent is cut from its recording's ``plain`` stream, windowed at :data:`WINDOW_S` on
    :data:`HOP_S`, and every window embedded; windows from all of the subject's extents are
    pooled by spherical mean. Extents are embedded separately, never concatenated.

    Batching is duration-homogeneous by construction: within one extent every window is exactly
    :data:`WINDOW_S` long, and a shorter extent yields a single window in a batch of one. Why
    that matters, and what a mixed batch does instead:
    ``specs/20260922-speaker-vectors/design.md``.

    Args:
        subject: The ``sub-*`` label.
        extents: The subject's admitted extents.
        root: The corpus root, recorded on the row.
        device: Optional device override.
        created_at: ISO-8601 timestamp for provenance, or ``None`` to leave it unstamped.

    Returns:
        The parquet row.

    Raises:
        ValueError: If ``extents`` is empty, or no window survived extraction for any of them.
    """
    import soundfile as sf  # noqa: PLC0415 -- heavy; only needed when an estimate actually runs
    import torch  # noqa: PLC0415

    from senselab.audio.data_structures import Audio  # noqa: PLC0415
    from senselab.audio.tasks.speaker_embeddings.api import (  # noqa: PLC0415
        estimate_speaker_embedding_from_audios,
    )
    from senselab.utils.data_structures import SpeechBrainModel  # noqa: PLC0415
    from senselab.utils.model_revision import resolve_revision  # noqa: PLC0415

    if not extents:
        raise ValueError(f"{subject}: no extents to embed")

    audios = []
    kept: list[Extent] = []
    for extent in extents:
        info = sf.info(str(extent.audio_path))
        first = max(0, int(round(extent.start_s * info.samplerate)))
        last = min(info.frames, int(round(extent.end_s * info.samplerate)))
        if last <= first:
            continue
        samples, rate = sf.read(str(extent.audio_path), start=first, stop=last, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.ascontiguousarray(samples[:, 0])).unsqueeze(0)
        audios.append(Audio(waveform=waveform, sampling_rate=rate))
        kept.append(extent)
    if not audios:
        raise ValueError(f"{subject}: every extent decoded to zero samples")

    model: SpeechBrainModel = SpeechBrainModel(path_or_uri=MODEL_ID, revision=resolve_revision(MODEL_ID, "main"))
    embedding = estimate_speaker_embedding_from_audios(
        audios=audios,
        model=model,
        device=device,
        window_s=WINDOW_S,
        hop_s=HOP_S,
        aggregator=AGGREGATOR,
        created_at=created_at,
        file_ids=[e.extent_id for e in kept],
    )
    return row_for(subject, kept, embedding, root)


def scan(
    root: Path,
    slice_index: int = 0,
    slices: int = 1,
    *,
    device: Any = None,  # noqa: ANN401 -- a senselab DeviceType, passed straight through
    created_at: Optional[str] = None,
    subjects: Optional[Iterable[str]] = None,
) -> tuple[list[dict[str, Any]], ScanReport]:
    """Embed every subject in this shard.

    Args:
        root: The corpus root.
        slice_index: This worker's index.
        slices: How many workers share the tree.
        device: Optional device override.
        created_at: ISO-8601 timestamp for provenance.
        subjects: Optional explicit subject allowlist, for a targeted re-run.

    Returns:
        ``(rows, report)``. A subject whose estimate raised contributes no row and appears in
        ``report.subjects_failed`` keyed by subject with the exception that ended it -- an opaque
        count would hide the one failure mode the floor does not cover, a subject whose whole
        supply is a single extent shorter than :data:`WINDOW_S`, which yields one window where the
        distribution describer needs two.
    """
    grouped, report = gather(root, slice_index, slices)
    wanted = set(subjects) if subjects is not None else None
    rows: list[dict[str, Any]] = []
    for subject in sorted(grouped):
        if wanted is not None and subject not in wanted:
            continue
        try:
            rows.append(embed_subject(subject, grouped[subject], root, device=device, created_at=created_at))
        except Exception as exc:  # noqa: BLE001 -- one subject's failure must not end the shard
            report.subjects_failed[subject] = f"{type(exc).__name__}: {exc}"
            continue
    report.subjects_written = len(rows)
    return rows, report


__all__ = [
    "AGGREGATOR",
    "EMBEDDING_DIM",
    "EMBED_STREAM",
    "HOP_S",
    "MODEL_ID",
    "SCHEMA_VERSION",
    "WINDOW_S",
    "Extent",
    "ScanReport",
    "embed_subject",
    "gather",
    "identity",
    "load_min_extent_s",
    "read_task_extents",
    "row_for",
    "run_dirs_of",
    "scan",
    "schema",
    "shard_of",
    "stem_of",
    "subject_dirs",
    "to_table",
]
