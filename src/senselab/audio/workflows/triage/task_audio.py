"""Task-extent audio: the plain, enhanced and released-redacted streams cut to where the task was performed.

One extent per recording, :func:`task_extent`, is cut from every stream that has one, by
:func:`cut_task_audio`. Each cut is a store stream of its own (``task_plain``, ``task_enhanced``,
``task_redacted``) derived from its source stream and from the ``task_audio`` measurement that
records the extent, and is written to ``run/streams/task_<source>.flac`` beside a sidecar,
``run/derivatives/task_audio.json``.

See ``specs/20261002-task-extent-audio/design.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.redaction.api import RedactionExtent
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    STREAM_SUFFIX,
    find_measurement,
    find_verdict,
    live_entities,
    path_attributes,
    software_agent,
    write_measurement,
    write_stream,
)
from senselab.audio.workflows.triage.nodes.redact import released_audio
from senselab.audio.workflows.triage.nodes.verdict import NODE as VERDICT_NODE
from senselab.audio.workflows.triage.vocabulary import Release
from senselab.utils.prov_store import Entity, ProvStore, file_digest

NODE = "TASK_AUDIO"
"""The node name the cut's own activities carry."""

PROCESS_VERSION = 1
"""The cut's behaviour version. Bump when what it writes for the same inputs changes."""

TASK_EXTENT_ROLE = "task_extent"
"""The span role the branches mark the performed task with."""

DEFINITION = "hull_of_live_task_extent_spans"
"""The one extent definition: first live task-extent span's start to the last one's end, padded, clamped."""

MEASUREMENT = "task_audio"
"""The measurement recording the extent and which cuts were written."""

SIDECAR = "derivatives/task_audio.json"
"""The sidecar path, relative to the run directory."""

RECORDING_STREAM = "recording"
"""The stream whose extent is the recording's duration."""

PLAIN = "plain"
ENHANCED = "enhanced"
REDACTED = "redacted"
SOURCES = (PLAIN, ENHANCED, REDACTED)
"""What is cut, in order. ``redacted`` is the copy the fold released, not REDACT's stream as such."""

PREFIX = "task_"
"""A cut's stream name and file stem: ``task_`` and the source's name."""

PRESENT = "present"
"""The cut already in the store was made from these same inputs, so nothing was written."""

WRITTEN = "written"
"""The cut was written."""

ABSENT = "absent"
"""The source does not exist for this recording, so nothing was cut from it."""


@dataclass(frozen=True)
class TaskExtent:
    """Where the task was performed, as cut from every stream.

    Attributes:
        start_s: The cut's start, in seconds on the recording's time base, padded and clamped.
        end_s: The cut's end, likewise.
        hull_start_s: The earliest live task-extent span's start, unpadded.
        hull_end_s: The latest live task-extent span's end, unpadded.
        padding_s: The outward pad applied to each side.
        recording_s: The recording's duration the extent is clamped to.
        span_ids: The task-extent spans the hull is over.
        definition: Which definition produced it, :data:`DEFINITION`.
    """

    start_s: float
    end_s: float
    hull_start_s: float
    hull_end_s: float
    padding_s: float
    recording_s: float
    span_ids: tuple[str, ...]
    definition: str = DEFINITION

    @property
    def duration_s(self) -> float:
        """The cut's duration in seconds."""
        return self.end_s - self.start_s

    @property
    def clamped_start(self) -> bool:
        """Whether the pad ran past the recording's start."""
        return self.hull_start_s - self.padding_s < 0.0

    @property
    def clamped_end(self) -> bool:
        """Whether the pad ran past the recording's end."""
        return self.hull_end_s + self.padding_s > self.recording_s

    def as_dict(self) -> dict[str, Any]:
        """The extent as the measurement and the sidecar record it."""
        return {
            "definition": self.definition,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "hull_start_s": self.hull_start_s,
            "hull_end_s": self.hull_end_s,
            "padding_s": self.padding_s,
            "recording_s": self.recording_s,
            "clamped_start": self.clamped_start,
            "clamped_end": self.clamped_end,
            "span_ids": list(self.span_ids),
        }


def task_extent(store: ProvStore, *, padding_s: float) -> TaskExtent | None:
    """The recording's one task extent: the hull of its live task-extent spans, padded and clamped.

    Args:
        store: The provenance store.
        padding_s: ``task_audio.padding_s``, applied outward on each side.

    Returns:
        The extent, or None where no live span carries the task-extent role, where the store holds
        no recording stream with an extent, or where the clamped extent is empty.

    Raises:
        ValueError: If ``padding_s`` is negative.
    """
    if padding_s < 0.0:
        raise ValueError(f"task_audio.padding_s must be at least 0, got {padding_s}")
    spans = [
        span
        for span in live_entities(store, "span")
        if span.attributes.get("role") == TASK_EXTENT_ROLE and span.extent is not None
    ]
    recording = _recording_extent(store)
    if not spans or recording is None:
        return None
    hull_start = min(float(span.extent[0]) for span in spans if span.extent is not None)
    hull_end = max(float(span.extent[1]) for span in spans if span.extent is not None)
    recording_s = float(recording[1])
    start = min(max(0.0, hull_start - padding_s), recording_s)
    end = max(0.0, min(recording_s, hull_end + padding_s))
    if end <= start:
        return None
    return TaskExtent(
        start_s=start,
        end_s=end,
        hull_start_s=hull_start,
        hull_end_s=hull_end,
        padding_s=padding_s,
        recording_s=recording_s,
        span_ids=tuple(span.id for span in spans),
    )


def _recording_extent(store: ProvStore) -> tuple[float, float] | None:
    found = [e for e in live_entities(store, "stream") if e.attributes.get("name") == RECORDING_STREAM and e.extent]
    return None if not found or found[-1].extent is None else found[-1].extent


def sample_bounds(start_s: float, end_s: float, sampling_rate: int, n_samples: int) -> tuple[int, int]:
    """The half-open sample range ``[lo, hi)`` an extent in seconds selects at one rate.

    The start rounds down and the end rounds up, both clamped to the stream, the rule
    :func:`~senselab.audio.tasks.redaction.api.apply_redactions` masks by, so a mask inside the
    extent lands on the same samples of the cut as of the stream it was cut from.

    Args:
        start_s: The start, in seconds.
        end_s: The end, in seconds.
        sampling_rate: The stream's rate.
        n_samples: The stream's length in samples.

    Returns:
        ``(lo, hi)`` with ``0 <= lo <= hi <= n_samples``.
    """
    lo = min(n_samples, max(0, int(start_s * sampling_rate)))
    hi = max(lo, min(n_samples, math.ceil(end_s * sampling_rate)))
    return lo, hi


def cut(audio: Audio, extent: TaskExtent) -> tuple[Audio, int, int]:
    """One stream cut to the extent, at its own rate and channels, without resampling.

    Args:
        audio: The stream.
        extent: The task extent.

    Returns:
        The cut and the ``[lo, hi)`` sample range of the stream it holds.
    """
    waveform = audio.waveform
    lo, hi = sample_bounds(extent.start_s, extent.end_s, audio.sampling_rate, int(waveform.shape[-1]))
    return Audio(waveform=waveform[..., lo:hi].clone(), sampling_rate=audio.sampling_rate), lo, hi


def masks_in_cut(masks: Sequence[RedactionExtent], lo: int, hi: int, sampling_rate: int) -> list[dict[str, Any]]:
    """The masks that fall within a cut, as sample ranges of the cut.

    Args:
        masks: The final masks, in seconds on the recording's time base.
        lo: The cut's first sample in the masked stream.
        hi: One past its last.
        sampling_rate: The masked stream's rate.

    Returns:
        One record per mask overlapping ``[lo, hi)``: its category, its sample range in the cut
        and in the masked stream, and its bounds in seconds on the cut's own time base.
    """
    out: list[dict[str, Any]] = []
    for mask in masks:
        m_lo, m_hi = sample_bounds(float(mask.start), float(mask.end), sampling_rate, hi)
        m_lo = max(m_lo, lo)
        if m_hi <= m_lo:
            continue
        out.append(
            {
                "category": mask.category,
                "source_samples": [m_lo, m_hi],
                "cut_samples": [m_lo - lo, m_hi - lo],
                "cut_start_s": (m_lo - lo) / sampling_rate,
                "cut_end_s": (m_hi - lo) / sampling_rate,
            }
        )
    return out


def verify_masks(cut_audio: Audio, masked: Audio, lo: int, masks: Sequence[dict[str, Any]], fill: str) -> None:
    """Check the cut carries each mask on the same samples as the masked stream does.

    Args:
        cut_audio: The redacted cut, as read back from its file.
        masked: The masked stream it was cut from.
        lo: The cut's first sample in the masked stream.
        masks: :func:`masks_in_cut`'s records.
        fill: The fill the masks were written with.

    Raises:
        ValueError: If a silence mask holds a non-zero sample in the cut, or the cut is not the
            masked stream's samples ``[lo, lo + len)``.
    """
    got = cut_audio.waveform
    want = masked.waveform[..., lo : lo + int(got.shape[-1])]
    if got.shape != want.shape:
        raise ValueError(f"the redacted cut is {tuple(got.shape)}, the masked stream's range {tuple(want.shape)}")
    for mask in masks:
        a, b = mask["cut_samples"]
        if fill == "silence" and bool(torch.any(got[..., a:b] != 0)):
            raise ValueError(f"the redacted cut holds signal inside the mask at samples [{a}, {b})")


def cut_key(source_digest: str | None, extent: TaskExtent, lo: int, hi: int, extra: Any = None) -> str:  # noqa: ANN401
    """The key one cut is cached under: its input's bytes, its sample range and the cut's version.

    Args:
        source_digest: The source stream's recorded SHA-256.
        extent: The task extent.
        lo: The first sample cut.
        hi: One past the last.
        extra: Anything else the cut's bytes depend on, JSON-serialisable.

    Returns:
        A 64-character hex sha256 digest.
    """
    payload = {
        "process": NODE,
        "process_version": PROCESS_VERSION,
        "input": source_digest,
        "definition": extent.definition,
        "samples": [lo, hi],
        "extra": extra,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class Source:
    """One stream a cut is taken from.

    Attributes:
        entity_id: The stream entity the cut is derived from.
        audio: Its audio.
        path: Its sidecar path, as the store records it.
        sha256: Its recorded digest.
        also_from: Further ids the cut is derived from: the PII ledger, for a re-masked copy.
        masks: The final masks, for the redacted copy; empty otherwise.
        fill: The masks' fill, for the redacted copy.
        remasked: Whether the redacted copy was masked again with the final masks.
    """

    entity_id: str
    audio: Audio
    path: str
    sha256: str | None
    also_from: tuple[str, ...] = ()
    masks: tuple[RedactionExtent, ...] = ()
    fill: str | None = None
    remasked: bool | None = None


@dataclass(frozen=True)
class TaskAudioOutcome:
    """What one cut pass did to one run.

    Attributes:
        extent: The extent cut, or None where the recording has none.
        cuts: Each source's outcome: :data:`WRITTEN`, :data:`PRESENT`, or ``absent: <reason>``.
        changed: Whether the store was written to.
    """

    extent: TaskExtent | None
    cuts: dict[str, str]
    changed: bool


def cut_task_audio(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> TaskAudioOutcome:
    """Cut the plain, enhanced and released-redacted streams to the recording's task extent.

    The redacted cut is taken only where the live fold releases the redacted copy; under every other
    release there is no masked copy anyone may hand on, so none is cut and any earlier one is retired.
    A cut whose inputs and sample range match the one the store already holds is not written again,
    and where every cut matches, nothing is written at all.

    Args:
        store: The run's store, after VERDICT.
        config: The triage configuration, for ``task_audio.padding_s`` and ``redaction.bleep_hz``.
        run_dir: The run directory streams and sidecars live under.

    Returns:
        What was cut.

    Raises:
        ValueError: If a redacted cut does not carry its masks on the masked copy's samples.
    """
    extent = task_extent(store, padding_s=float(config.require("task_audio.padding_s")))
    held = {name: _live_stream(store, f"{PREFIX}{name}") for name in SOURCES}
    cuts: dict[str, str] = {}
    sources: dict[str, Source] = {}
    if extent is None:
        cuts = {name: f"{ABSENT}: no task extent" for name in SOURCES}
    else:
        for name in SOURCES:
            try:
                sources[name] = _source(store, run_dir, name, config)
            except LookupError as error:
                cuts[name] = f"{ABSENT}: {error}"

    keys: dict[str, tuple[str, int, int]] = {}
    for name, source in sources.items():
        assert extent is not None
        lo, hi = sample_bounds(extent.start_s, extent.end_s, source.audio.sampling_rate, _length(source.audio))
        masks = [(m.start, m.end, m.category) for m in source.masks]
        keys[name] = (cut_key(source.sha256, extent, lo, hi, {"masks": masks, "fill": source.fill}), lo, hi)

    kept = {
        name
        for name, entity in held.items()
        if entity is not None and name in keys and _holds(entity, keys[name][0], run_dir)
    }
    stale = [entity for name, entity in held.items() if entity is not None and name not in kept]
    standing = find_measurement(store, MEASUREMENT)
    if extent is not None and not stale and kept == set(keys) and _same_extent(standing, extent, sorted(keys)):
        return TaskAudioOutcome(extent, {**cuts, **{name: PRESENT for name in keys}}, changed=False)
    if extent is None and not stale and standing is None:
        return TaskAudioOutcome(None, cuts, changed=False)

    software = software_agent(store)
    _retire(store, stale, software=software, reason="task extent or source changed")
    if standing is not None:
        _retire(store, [standing], software=software, reason="task extent recomputed")
    for name in SOURCES:
        if name not in kept:
            (run_dir / f"streams/{PREFIX}{name}{STREAM_SUFFIX}").unlink(missing_ok=True)
    if extent is None:
        (run_dir / SIDECAR).unlink(missing_ok=True)
        return TaskAudioOutcome(None, cuts, changed=True)

    activity = store.activity(
        node=NODE,
        step="cut",
        parameters={"process_version": PROCESS_VERSION, "padding_s": extent.padding_s, "definition": DEFINITION},
    )
    store.was_associated_with(activity, software)
    for span_id in extent.span_ids:
        store.used(activity, span_id)
    measurement = write_measurement(
        store,
        activity,
        software,
        name=MEASUREMENT,
        signal=RECORDING_STREAM,
        attributes={**extent.as_dict(), "cuts": sorted(keys), "process_version": PROCESS_VERSION},
        derived_from=extent.span_ids,
        extent=(extent.start_s, extent.end_s),
    )

    records: dict[str, Any] = {}
    for name in SOURCES:
        if name not in sources:
            records[name] = {ABSENT: cuts[name].removeprefix(f"{ABSENT}: ")}
            continue
        if name in kept:
            entity = held[name]
            assert entity is not None
            store.was_derived_from(entity.id, measurement)
            cuts[name] = PRESENT
            records[name] = _cut_record(entity)
            continue
        source = sources[name]
        key, lo, hi = keys[name]
        piece, _, _ = cut(source.audio, extent)
        relative, report = write_stream(piece, run_dir, f"{PREFIX}{name}")
        sr = source.audio.sampling_rate
        attributes: dict[str, Any] = {
            "name": f"{PREFIX}{name}",
            **path_attributes(relative, run_dir),
            "sampling_rate": sr,
            "channels": int(piece.waveform.shape[0]),
            "write_gain": report.gain,
            "source": name,
            "source_path": source.path,
            "source_sha256": source.sha256,
            "source_samples": [lo, hi],
            "start_s": extent.start_s,
            "end_s": extent.end_s,
            "cut_key": key,
            "process_version": PROCESS_VERSION,
        }
        if name == REDACTED:
            inside = masks_in_cut(source.masks, lo, hi, sr)
            verify_masks(Audio(filepath=str(run_dir / relative)), source.audio, lo, inside, str(source.fill))
            attributes.update(
                {"fill": source.fill, "remasked": source.remasked, "masks": inside, "masks_verified": True}
            )
        stream_id = store.entity(prov_type="stream", extent=(0.0, (hi - lo) / sr), attributes=attributes)
        store.was_generated_by(stream_id, activity)
        store.was_attributed_to(stream_id, software)
        store.used(activity, source.entity_id)
        store.was_derived_from(stream_id, source.entity_id)
        store.was_derived_from(stream_id, measurement)
        for other in source.also_from:
            store.was_derived_from(stream_id, other)
        cuts[name] = WRITTEN
        records[name] = _cut_record(store.get_entity(stream_id))

    sidecar = {"process": NODE, "process_version": PROCESS_VERSION, "extent": extent.as_dict(), "cuts": records}
    path = run_dir / SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TaskAudioOutcome(extent, cuts, changed=True)


def _length(audio: Audio) -> int:
    return int(audio.waveform.shape[-1])


def _same_extent(measurement: Entity | None, extent: TaskExtent, cuts: list[str]) -> bool:
    if measurement is None:
        return False
    held = measurement.attributes
    return all(held.get(key) == value for key, value in extent.as_dict().items()) and held.get("cuts") == cuts


_RECORD_KEYS = (
    "path",
    "checksum_sha256",
    "sampling_rate",
    "channels",
    "write_gain",
    "source",
    "source_path",
    "source_sha256",
    "source_samples",
    "cut_key",
    "fill",
    "remasked",
    "masks",
    "masks_verified",
)


def _cut_record(entity: Entity) -> dict[str, Any]:
    return {"stream_id": entity.id, **{k: entity.attributes[k] for k in _RECORD_KEYS if k in entity.attributes}}


def _holds(entity: Entity, key: str, run_dir: Path) -> bool:
    if entity.attributes.get("cut_key") != key:
        return False
    digest, _ = file_digest(run_dir / str(entity.attributes.get("path")))
    return digest is not None and digest == entity.attributes.get("checksum_sha256")


def _retire(store: ProvStore, entities: Sequence[Entity], *, software: str, reason: str) -> None:
    for entity in entities:
        activity = store.activity(
            node=NODE, step="cut_superseded", parameters={"superseded": entity.id, "reason": reason}
        )
        store.was_associated_with(activity, software)
        store.used(activity, entity.id)
        store.was_invalidated_by(entity.id, activity)


def _source(store: ProvStore, run_dir: Path, name: str, config: TriageConfig) -> Source:
    """One stream to cut.

    Args:
        store: The run's store.
        run_dir: The run directory.
        name: One of :data:`SOURCES`.
        config: The triage configuration, for ``redaction.bleep_hz``.

    Returns:
        The source.

    Raises:
        LookupError: Where this recording has no such stream, or releases no redacted copy.
    """
    if name != REDACTED:
        entity = _live_stream(store, name)
        if entity is None:
            raise LookupError(f"no {name} stream")
        path = str(entity.attributes["path"])
        audio = Audio(filepath=str(path if Path(path).is_absolute() else run_dir / path))
        return Source(entity.id, audio, path, entity.attributes.get("checksum_sha256"))
    folded = find_verdict(store, VERDICT_NODE)
    release = None if folded is None else str(folded.attributes.get("release") or "")
    if release != Release.WITH_REDACTION.value:
        raise LookupError(f"release is {release or 'unfolded'}, so no redacted copy is released")
    released = released_audio(store, run_dir, bleep_hz=config.get("redaction.bleep_hz"))
    first = store.get_entity(released.derived_from[0])
    return Source(
        entity_id=first.id,
        audio=released.audio,
        path=str(first.attributes.get("path")),
        sha256=first.attributes.get("checksum_sha256"),
        also_from=released.derived_from[1:],
        masks=released.masks,
        fill=released.fill,
        remasked=released.remasked,
    )


def _live_stream(store: ProvStore, name: str) -> Entity | None:
    found = [e for e in live_entities(store, "stream") if e.attributes.get("name") == name]
    return found[-1] if found else None
