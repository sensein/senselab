"""Read and write behaviour events in common interchange formats.

Each format is a `read_*` / `write_*` pair. `FORMAT_FIELDS[fmt]` names the fields a format carries;
those fields round-trip, and a reader takes the rest (schema id, source, confidence kind) as
arguments. Files are read and written as UTF-8 whatever the locale.

| Format | Functions | Carries |
|---|---|---|
| per-frame label table | `read_frame_table`, `write_frame_table` | times (frame-aligned), label, duration |
| segment CSV | `read_segments_csv`, `write_segments_csv` | every `BehaviorEvent` field |
| RTTM | `read_rttm`, `write_rttm` | times, label, confidence; one track per recording id |
| BIDS `events.tsv` | `read_bids_events`, `write_bids_events` | times, label, confidence |
| ActivityNet / OpenTAD JSON | `read_activitynet_json`, `write_activitynet_json` | times, label, duration; per video |
| Label Studio export | `read_label_studio`, `write_label_studio` | times, label |

**Per-frame label table.** A CSV with a `Frame` column (0-based frame numbers, one row per frame)
and one column per category. Each cell holds that category's label at that frame, or a background
token (default `N/A`). An event `[onset_s, offset_s)` covers frames `[onset_s * fps, offset_s * fps)`.
It is lossless only up to `k / fps`: a time within `_FRAME_ALIGN_TOLERANCE` frames of frame `k` is
written as frame `k` and reads back as exactly `k / fps`.

**RTTM** has no record for an empty track, so a recording id whose track has no events is not
written and does not come back.

**BIDS `events.tsv`** stores a duration per row; a zero duration (an impulse event) is refused on
read, because a `BehaviorEvent` needs `offset_s > onset_s`.

**Label Studio.** Only ``labels`` results with `start`/`end` in seconds (audio-style regions) are
covered; video ``timelinelabels`` results with frame ranges are not. The clip duration is written
as each result's ``original_length``, so an empty track cannot carry it; `duration_s` is therefore
not in this format's `FORMAT_FIELDS`, although the reader restores it whenever a result holds it.

**Float exactness.** RTTM and BIDS store a duration, so the reader rebuilds the offset as
`onset + duration`. Times are written as the shortest decimal that round-trips to the same float,
so for times that are multiples of a power of two (such as 1/64 s) the offset is restored exactly;
for other times it is restored within one unit in the last place of the offset.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Union

from senselab.utils.data_structures.events import AnnotationSchema, BehaviorEvent, ConfidenceKind, EventTrack

PathLike = Union[str, Path]

_TIMES_AND_LABEL = frozenset({"onset_s", "offset_s", "label"})

FORMAT_FIELDS: Dict[str, FrozenSet[str]] = {
    "frame_table": _TIMES_AND_LABEL | {"duration_s"},
    "segments_csv": _TIMES_AND_LABEL | {"confidence", "confidence_kind", "source", "subject_ref", "track_id"},
    "rttm": _TIMES_AND_LABEL | {"confidence"},
    "bids_events": _TIMES_AND_LABEL | {"confidence"},
    "activitynet_json": _TIMES_AND_LABEL | {"duration_s"},
    "label_studio": _TIMES_AND_LABEL,
}

SEGMENT_COLUMNS = ("onset_s", "offset_s", "label", "confidence", "confidence_kind", "source", "subject_ref", "track_id")
FRAME_COLUMN = "Frame"
RTTM_NA = "<NA>"
BIDS_NA = "n/a"
BIDS_REQUIRED = ("onset", "duration", "trial_type")
_FRAME_ALIGN_TOLERANCE = 1e-6
_ENCODING = "utf-8"


def _num(value: float) -> str:
    """The shortest decimal string that reads back as the same float."""
    return repr(float(value))


def _event(
    onset: float,
    offset: float,
    label: str,
    source: str,
    confidence: Optional[float] = None,
    confidence_kind: Optional[ConfidenceKind] = None,
) -> BehaviorEvent:
    if confidence is None:
        return BehaviorEvent(onset_s=onset, offset_s=offset, label=label, source=source)
    if confidence_kind is None or confidence_kind == "none":
        raise ValueError("the file holds confidences; pass confidence_kind to say what they are")
    return BehaviorEvent(
        onset_s=onset,
        offset_s=offset,
        label=label,
        source=source,
        confidence=confidence,
        confidence_kind=confidence_kind,
    )


def _refuse_chars(values: Iterable[str], what: str, forbidden: Optional[str]) -> None:
    """Refuse empty values, and values holding any of `forbidden` (any whitespace when `None`)."""
    for value in values:
        if not value:
            raise ValueError(f"{what} must be non-empty")
        bad = any(c.isspace() for c in value) if forbidden is None else any(c in forbidden for c in value)
        if bad:
            kind = "whitespace" if forbidden is None else "tabs or line breaks"
            raise ValueError(f"{what} must not contain {kind}")


def _check_fps(fps: float) -> None:
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not (math.isfinite(fps) and fps > 0):
        raise ValueError("fps must be a finite number > 0 (not a bool)")


# ── per-frame label table ────────────────────────────────────────────────


def _frame_index(t: float, fps: float, what: str) -> int:
    position = t * fps
    index = round(position)
    if abs(position - index) > _FRAME_ALIGN_TOLERANCE:
        raise ValueError(f"{what} {t!r} s is not frame-aligned at {fps!r} fps")
    return index


def write_frame_table(
    tracks: Mapping[str, EventTrack], path: PathLike, fps: float, n_frames: int, background: str = "N/A"
) -> None:
    """Write tracks as a per-frame label table, one column per category.

    Args:
        tracks: Category column name to that category's track.
        path: The CSV file to write.
        fps: Frames per second; finite and > 0.
        n_frames: Number of frames (rows); at least 1, since a table with no rows has no duration.
        background: The token written where a category has no event.

    Raises:
        ValueError: If `fps` or `n_frames` is out of range; a track's `duration_s` is set and differs
            from `n_frames / fps`; an onset or offset is not frame-aligned; an event covers no frame
            or runs past `n_frames`; two events of one category overlap; two adjacent events of one
            category share a label (they would read back as one); or a label is the background token.
    """
    _check_fps(fps)
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    columns = list(tracks)
    if FRAME_COLUMN in columns:
        raise ValueError(f"{FRAME_COLUMN!r} is reserved for the frame number column")
    table_duration = n_frames / fps
    cells = {column: [background] * n_frames for column in columns}
    for column, track in tracks.items():
        if track.duration_s is not None and not math.isclose(track.duration_s, table_duration, rel_tol=1e-12):
            raise ValueError(
                f"column {column!r}: duration_s {track.duration_s!r} != n_frames / fps = {table_duration!r}"
            )
        previous: Optional[tuple[int, str]] = None
        for event in track.events:
            if event.label == background:
                raise ValueError(f"column {column!r}: a label equals the background token {background!r}")
            start = _frame_index(event.onset_s, fps, "onset")
            end = _frame_index(event.offset_s, fps, "offset")
            if end <= start:
                raise ValueError(f"column {column!r}: an event covers no frame at {fps!r} fps")
            if end > n_frames:
                raise ValueError(f"column {column!r}: an event ends after frame {n_frames}")
            if previous is not None:
                if start < previous[0]:
                    raise ValueError(f"column {column!r}: events overlap")
                if start == previous[0] and event.label == previous[1]:
                    raise ValueError(f"column {column!r}: adjacent events share a label and would merge")
            cells[column][start:end] = [event.label] * (end - start)
            previous = (end, event.label)
    with Path(path).open("w", encoding=_ENCODING, newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([FRAME_COLUMN, *columns])
        for frame in range(n_frames):
            writer.writerow([frame, *(cells[column][frame] for column in columns)])


def read_frame_table(
    path: PathLike,
    fps: float,
    schemas: Mapping[str, AnnotationSchema],
    *,
    source: str,
    background: str = "N/A",
) -> Dict[str, EventTrack]:
    """Read a per-frame label table into one track per category.

    A run of consecutive frames with the same label becomes one event.

    Args:
        path: The CSV file.
        fps: Frames per second; finite and > 0.
        schemas: Category column name to the schema its labels come from; only these columns are read.
        source: The source recorded on every event.
        background: The token that means "no event".

    Returns:
        Category column name to its track, with `duration_s` = frames / fps.

    Raises:
        ValueError: If `fps` is out of range, the table has no `Frame` column or no rows, the frame
            numbers are not 0, 1, 2, ..., a schema's column is missing, or a label is not in its schema.
    """
    _check_fps(fps)
    with Path(path).open(encoding=_ENCODING, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = set(reader.fieldnames or [])
    if FRAME_COLUMN not in header:
        raise ValueError(f"the frame table has no {FRAME_COLUMN!r} column")
    if not rows:
        raise ValueError("the frame table has no rows")
    missing = sorted(set(schemas) - header)
    if missing:
        raise ValueError(f"columns missing from the frame table: {missing}")
    if [row[FRAME_COLUMN] for row in rows] != [str(i) for i in range(len(rows))]:
        raise ValueError(f"{FRAME_COLUMN} must count 0, 1, 2, ... with one row per frame")
    duration = len(rows) / fps
    tracks: Dict[str, EventTrack] = {}
    for column, schema in schemas.items():
        labels = [row[column] for row in rows] + [background]
        events: List[BehaviorEvent] = []
        start = 0
        for frame in range(1, len(labels)):
            if labels[frame] != labels[frame - 1]:
                if labels[start] != background:
                    events.append(_event(start / fps, frame / fps, labels[start], source))
                start = frame
        track = EventTrack(schema_id=schema.schema_id, events=events, duration_s=duration)
        track.check(schema)
        tracks[column] = track
    return tracks


# ── segment CSV ─────────────────────────────────────────────────────────


def write_segments_csv(track: EventTrack, path: PathLike) -> None:
    """Write a track as a segment CSV with every event field; a missing value is an empty cell.

    Empty strings cannot occur in the optional fields (`BehaviorEvent` refuses them), so an empty
    cell always means `None`.

    Args:
        track: The track.
        path: The CSV file to write.
    """
    with Path(path).open("w", encoding=_ENCODING, newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SEGMENT_COLUMNS)
        for e in track.events:
            confidence = "" if e.confidence is None else _num(e.confidence)
            writer.writerow(
                [
                    _num(e.onset_s),
                    _num(e.offset_s),
                    e.label,
                    confidence,
                    e.confidence_kind,
                    e.source,
                    e.subject_ref or "",
                    e.track_id or "",
                ]
            )


def read_segments_csv(path: PathLike, *, schema_id: str, duration_s: Optional[float] = None) -> EventTrack:
    """Read a segment CSV written by `write_segments_csv`.

    Args:
        path: The CSV file.
        schema_id: The schema id of the returned track.
        duration_s: Optional clip duration for the returned track.

    Returns:
        The track.
    """
    with Path(path).open(encoding=_ENCODING, newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = [
        BehaviorEvent(
            onset_s=float(row["onset_s"]),
            offset_s=float(row["offset_s"]),
            label=row["label"],
            confidence=float(row["confidence"]) if row["confidence"] else None,
            confidence_kind=row["confidence_kind"],  # type: ignore[arg-type]
            source=row["source"],
            subject_ref=row["subject_ref"] or None,
            track_id=row["track_id"] or None,
        )
        for row in rows
    ]
    return EventTrack(schema_id=schema_id, events=events, duration_s=duration_s)


# ── RTTM ────────────────────────────────────────────────────────────────


def write_rttm(tracks: Mapping[str, EventTrack], path: PathLike) -> None:
    """Write tracks as RTTM ``SPEAKER`` lines, with the label in the speaker-name field.

    A track with no events writes nothing, so its recording id does not come back on read.

    Args:
        tracks: Recording id (the RTTM file id) to its track.
        path: The RTTM file to write.

    Raises:
        ValueError: If a recording id or label is empty or contains whitespace.
    """
    _refuse_chars(tracks, "RTTM recording ids", None)
    lines = []
    for uri, track in tracks.items():
        _refuse_chars((e.label for e in track.events), "RTTM labels", None)
        for e in track.events:
            confidence = RTTM_NA if e.confidence is None else _num(e.confidence)
            duration = _num(e.offset_s - e.onset_s)
            fields = ["SPEAKER", uri, "1", _num(e.onset_s), duration, RTTM_NA, RTTM_NA, e.label, confidence, RTTM_NA]
            lines.append(" ".join(fields))
    Path(path).write_text("".join(line + "\n" for line in lines), encoding=_ENCODING)


def read_rttm(
    path: PathLike, *, schema_id: str, source: str, confidence_kind: Optional[ConfidenceKind] = None
) -> Dict[str, EventTrack]:
    """Read the ``SPEAKER`` lines of an RTTM file; other record types and blank lines are skipped.

    Args:
        path: The RTTM file.
        schema_id: The schema id of the returned tracks.
        source: The source recorded on every event.
        confidence_kind: What the confidence column holds; required when any line has a confidence.

    Returns:
        Recording id to its track.

    Raises:
        ValueError: If a ``SPEAKER`` line has fewer than 9 fields, a non-numeric time or confidence,
            or a duration that is not positive; the message starts with the line number.
    """
    by_uri: Dict[str, List[BehaviorEvent]] = {}
    for number, line in enumerate(Path(path).read_text(encoding=_ENCODING).splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        try:
            event = _rttm_event(fields, source, confidence_kind)
        except ValueError as error:
            raise ValueError(f"line {number}: {error}") from error
        by_uri.setdefault(fields[1], []).append(event)
    return {uri: EventTrack(schema_id=schema_id, events=events) for uri, events in by_uri.items()}


def _rttm_event(fields: List[str], source: str, confidence_kind: Optional[ConfidenceKind]) -> BehaviorEvent:
    if len(fields) < 9:
        raise ValueError(f"a SPEAKER line needs at least 9 fields, found {len(fields)}")
    onset, duration = float(fields[3]), float(fields[4])
    if duration <= 0:
        raise ValueError(f"duration is {fields[4]}; an event needs offset > onset")
    confidence = None if fields[8] == RTTM_NA else float(fields[8])
    return _event(onset, onset + duration, fields[7], source, confidence, confidence_kind)


# ── BIDS events.tsv ─────────────────────────────────────────────────────


def write_bids_events(track: EventTrack, path: PathLike) -> None:
    """Write a track as a BIDS ``events.tsv``; ``confidence`` is a column only when an event has one.

    Args:
        track: The track.
        path: The TSV file to write.

    Raises:
        ValueError: If a label is empty or contains a tab or a line break (spaces are allowed).
    """
    _refuse_chars((e.label for e in track.events), "BIDS trial_type labels", "\t\r\n")
    with_confidence = any(e.confidence is not None for e in track.events)
    header = ["onset", "duration", "trial_type"] + (["confidence"] if with_confidence else [])
    lines = ["\t".join(header)]
    for e in track.events:
        row = [_num(e.onset_s), _num(e.offset_s - e.onset_s), e.label]
        if with_confidence:
            row.append(BIDS_NA if e.confidence is None else _num(e.confidence))
        lines.append("\t".join(row))
    with Path(path).open("w", encoding=_ENCODING, newline="") as handle:
        handle.write("".join(line + "\n" for line in lines))


def _bids_number(row: Dict[str, Optional[str]], column: str) -> float:
    raw = row.get(column)
    if raw is None:
        raise ValueError(f"the row has no {column} field")
    if raw in (BIDS_NA, ""):
        raise ValueError(f"{column} is {raw or 'empty'!r}; an event needs a number here")
    return float(raw)


def _bids_event(row: Dict[str, Optional[str]], source: str, confidence_kind: Optional[ConfidenceKind]) -> BehaviorEvent:
    onset = _bids_number(row, "onset")
    duration = _bids_number(row, "duration")
    if duration == 0:
        raise ValueError("duration is 0; an event needs offset > onset, so impulse events are refused")
    if duration < 0:
        raise ValueError(f"duration is {duration!r}; an event needs offset > onset")
    label = row.get("trial_type")
    if label is None:
        raise ValueError("the row has no trial_type field")
    raw = row.get("confidence")
    confidence = None if raw is None or raw in (BIDS_NA, "") else float(raw)
    return _event(onset, onset + duration, label, source, confidence, confidence_kind)


def read_bids_events(
    path: PathLike,
    *,
    schema_id: str,
    source: str,
    confidence_kind: Optional[ConfidenceKind] = None,
    duration_s: Optional[float] = None,
) -> EventTrack:
    """Read a BIDS ``events.tsv``; ``n/a`` in ``confidence`` reads as no confidence.

    Fields are split on tabs only, with no quote handling, so a label may start with a quote character.
    A leading byte-order mark is ignored.

    Args:
        path: The TSV file.
        schema_id: The schema id of the returned track.
        source: The source recorded on every event.
        confidence_kind: What the ``confidence`` column holds; required when any row has a confidence.
        duration_s: Optional clip duration for the returned track.

    Returns:
        The track.

    Raises:
        ValueError: If ``onset``, ``duration`` or ``trial_type`` is not a column; or a row lacks a field,
            holds a non-numeric or ``n/a`` onset or duration, or has a duration of 0 (an impulse event:
            `BehaviorEvent` needs `offset_s > onset_s`) or below. Row errors start with the file's line
            number.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = list(reader)
        header = reader.fieldnames or []
    missing = [column for column in BIDS_REQUIRED if column not in header]
    if missing:
        raise ValueError(f"events.tsv lacks the columns {missing}")
    events = []
    for line, row in enumerate(rows, start=2):
        try:
            events.append(_bids_event(row, source, confidence_kind))
        except ValueError as error:
            raise ValueError(f"line {line}: {error}") from error
    return EventTrack(schema_id=schema_id, events=events, duration_s=duration_s)


# ── ActivityNet / OpenTAD JSON ──────────────────────────────────────────


def write_activitynet_json(tracks: Mapping[str, EventTrack], path: PathLike) -> None:
    """Write tracks as an ActivityNet-style ground-truth database, as OpenTAD reads it.

    Args:
        tracks: Video id to its track; a missing `duration_s` is written as ``null``.
        path: The JSON file to write.
    """
    database = {
        video: {
            "duration": track.duration_s,
            "annotations": [{"segment": [e.onset_s, e.offset_s], "label": e.label} for e in track.events],
        }
        for video, track in tracks.items()
    }
    text = json.dumps({"database": database}, indent=2, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding=_ENCODING)


def read_activitynet_json(path: PathLike, *, schema_id: str, source: str) -> Dict[str, EventTrack]:
    """Read an ActivityNet-style ground-truth database.

    Args:
        path: The JSON file.
        schema_id: The schema id of the returned tracks.
        source: The source recorded on every event.

    Returns:
        Video id to its track.
    """
    database: Dict[str, Any] = json.loads(Path(path).read_text(encoding=_ENCODING))["database"]
    tracks = {}
    for video, entry in database.items():
        events = [
            _event(float(a["segment"][0]), float(a["segment"][1]), a["label"], source) for a in entry["annotations"]
        ]
        duration = entry.get("duration")
        tracks[video] = EventTrack(
            schema_id=schema_id, events=events, duration_s=None if duration is None else float(duration)
        )
    return tracks


# ── Label Studio ────────────────────────────────────────────────────────


def write_label_studio(track: EventTrack, path: PathLike, *, from_name: str, to_name: str) -> None:
    """Write a track as a one-task Label Studio export of ``labels`` results.

    Args:
        track: The track.
        path: The JSON file to write.
        from_name: The labelling control's name in the Label Studio config.
        to_name: The media object's name in the Label Studio config.
    """
    results = []
    for i, e in enumerate(track.events):
        result: Dict[str, Any] = {
            "id": f"r{i}",
            "type": "labels",
            "from_name": from_name,
            "to_name": to_name,
            "value": {"start": e.onset_s, "end": e.offset_s, "labels": [e.label]},
        }
        if track.duration_s is not None:
            result["original_length"] = track.duration_s
        results.append(result)
    text = json.dumps([{"annotations": [{"result": results}]}], indent=2, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding=_ENCODING)


def read_label_studio(path: PathLike, *, schema_id: str, source: str) -> EventTrack:
    """Read the ``labels`` results of the first annotation of the first task in a Label Studio export.

    Args:
        path: The JSON file.
        schema_id: The schema id of the returned track.
        source: The source recorded on every event.

    Returns:
        The track; `duration_s` comes from ``original_length`` when any result holds it.

    Raises:
        ValueError: If the file is not a non-empty list of tasks.
    """
    tasks = json.loads(Path(path).read_text(encoding=_ENCODING))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("a Label Studio export must be a non-empty list of tasks")
    annotation = (tasks[0].get("annotations") or [{}])[0]
    events = []
    lengths = []
    for item in annotation.get("result") or []:
        if item.get("original_length") is not None:
            lengths.append(float(item["original_length"]))
        if item.get("type") != "labels":
            continue
        value = item["value"]
        for label in value["labels"]:
            events.append(_event(float(value["start"]), float(value["end"]), label, source))
    duration = max(lengths) if lengths else None
    return EventTrack(schema_id=schema_id, events=events, duration_s=duration)
