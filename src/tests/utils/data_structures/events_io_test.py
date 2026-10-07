"""Tests for the behaviour-event interchange formats."""

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from senselab.utils.data_structures.events import AnnotationSchema, BehaviorEvent, EventTrack
from senselab.utils.data_structures.events_io import (
    FORMAT_FIELDS,
    read_activitynet_json,
    read_bids_events,
    read_frame_table,
    read_label_studio,
    read_rttm,
    read_segments_csv,
    write_activitynet_json,
    write_bids_events,
    write_frame_table,
    write_label_studio,
    write_rttm,
    write_segments_csv,
)

# Every time is a multiple of 1/64 s, so onset + duration reconstructs the offset exactly.
FPS = 32.0
N_FRAMES = 320
DURATION = N_FRAMES / FPS


def _event(onset: float, offset: float, label: str, **kwargs: object) -> BehaviorEvent:
    fields: dict = {"onset_s": onset, "offset_s": offset, "label": label, "source": "model_x"}
    fields.update(kwargs)
    return BehaviorEvent(**fields)


def _track(schema_id: str = "motion") -> EventTrack:
    events = [
        _event(0.015625, 1.5, "reach", confidence=0.75, confidence_kind="prob", subject_ref="s01", track_id="p1"),
        _event(2.0, 3.25, "wave", confidence=0.5, confidence_kind="prob", subject_ref="s01", track_id="p1"),
        _event(4.5, 9.984375, "point", confidence=0.125, confidence_kind="prob", subject_ref="s01", track_id="p2"),
    ]
    return EventTrack(schema_id=schema_id, events=events, duration_s=DURATION)


def _carried(track: EventTrack, fmt: str) -> list[tuple]:
    event_fields = [f for f in FORMAT_FIELDS[fmt] if f != "duration_s"]
    return [tuple(getattr(e, f) for f in event_fields) for e in track.events]


def test_format_fields_cover_every_format() -> None:
    """Each format declares the fields it carries, and every format carries times and the label."""
    assert set(FORMAT_FIELDS) == {
        "frame_table",
        "segments_csv",
        "rttm",
        "bids_events",
        "activitynet_json",
        "label_studio",
    }
    for fields in FORMAT_FIELDS.values():
        assert {"onset_s", "offset_s", "label"} <= fields


def test_segments_csv_round_trip(tmp_path: Path) -> None:
    """The segment CSV carries every event field."""
    track = _track()
    path = tmp_path / "segments.csv"
    write_segments_csv(track, path)
    back = read_segments_csv(path, schema_id="motion")
    assert back.events == track.events


def test_rttm_round_trip(tmp_path: Path) -> None:
    """RTTM carries times, label and confidence, keyed by recording id."""
    track = _track()
    path = tmp_path / "events.rttm"
    write_rttm({"rec1": track, "rec2": EventTrack(schema_id="motion", events=[_event(0.5, 1.0, "wave")])}, path)
    back = read_rttm(path, schema_id="motion", source="model_x", confidence_kind="prob")
    assert sorted(back) == ["rec1", "rec2"]
    assert _carried(back["rec1"], "rttm") == _carried(track, "rttm")
    assert back["rec2"].events[0].confidence is None


def test_rttm_reader_skips_other_lines(tmp_path: Path) -> None:
    """Only ``SPEAKER`` lines are events; other RTTM record types and blank lines are skipped."""
    path = tmp_path / "events.rttm"
    path.write_text(
        "SPKR-INFO rec1 1 <NA> <NA> <NA> unknown reach <NA> <NA>\n"
        "\n"
        "SPEAKER rec1 1 0.5 0.25 <NA> <NA> reach <NA> <NA>\n"
        "LEXEME rec1 1 0.5 0.25 hello lex reach <NA> <NA>\n"
    )
    back = read_rttm(path, schema_id="motion", source="model_x")
    assert [(e.onset_s, e.offset_s, e.label) for e in back["rec1"].events] == [(0.5, 0.75, "reach")]


def test_bids_events_round_trip(tmp_path: Path) -> None:
    """BIDS ``events.tsv`` carries times, label and confidence."""
    track = _track()
    path = tmp_path / "events.tsv"
    write_bids_events(track, path)
    assert path.read_text().splitlines()[0].split("\t") == ["onset", "duration", "trial_type", "confidence"]
    back = read_bids_events(path, schema_id="motion", source="model_x", confidence_kind="prob")
    assert _carried(back, "bids_events") == _carried(track, "bids_events")


def test_bids_na_reads_as_none(tmp_path: Path) -> None:
    """``n/a`` in the confidence column reads as no confidence."""
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\ttrial_type\tconfidence\n0.5\t0.25\treach\tn/a\n1.0\t0.5\twave\t0.5\n")
    back = read_bids_events(path, schema_id="motion", source="model_x", confidence_kind="vote")
    assert [(e.confidence, e.confidence_kind) for e in back.events] == [(None, "none"), (0.5, "vote")]


def test_reader_needs_a_kind_for_confidences(tmp_path: Path) -> None:
    """A format that does not carry the confidence kind needs the caller to name it."""
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\ttrial_type\tconfidence\n0.5\t0.25\treach\t0.5\n")
    with pytest.raises(ValueError, match="confidence_kind"):
        read_bids_events(path, schema_id="motion", source="model_x")


def test_activitynet_json_round_trip(tmp_path: Path) -> None:
    """The ActivityNet/OpenTAD ground-truth JSON carries times, label and duration per video."""
    track = _track()
    path = tmp_path / "gt.json"
    write_activitynet_json({"vid1": track}, path)
    data = json.loads(path.read_text())
    assert data["database"]["vid1"]["annotations"][0] == {"segment": [0.015625, 1.5], "label": "reach"}
    back = read_activitynet_json(path, schema_id="motion", source="model_x")
    assert _carried(back["vid1"], "activitynet_json") == _carried(track, "activitynet_json")
    assert back["vid1"].duration_s == DURATION


def test_label_studio_round_trip(tmp_path: Path) -> None:
    """The Label Studio export carries times, label and duration in ``labels`` results."""
    track = _track()
    path = tmp_path / "ls.json"
    write_label_studio(track, path, from_name="label", to_name="video")
    result = json.loads(path.read_text())[0]["annotations"][0]["result"][0]
    assert result["type"] == "labels" and (result["from_name"], result["to_name"]) == ("label", "video")
    assert result["value"] == {"start": 0.015625, "end": 1.5, "labels": ["reach"]}
    back = read_label_studio(path, schema_id="motion", source="model_x")
    assert _carried(back, "label_studio") == _carried(track, "label_studio")
    assert back.duration_s == DURATION


def _frame_schemas() -> dict[str, AnnotationSchema]:
    return {
        "Gesture": AnnotationSchema(schema_id="gesture", task_type="segmentation", labels=["reach", "point"]),
        "Locomotion": AnnotationSchema(schema_id="locomotion", task_type="segmentation", labels=["walk", "crawl"]),
    }


def _frame_tracks() -> dict[str, EventTrack]:
    gesture = [_event(0.0, 0.5, "reach"), _event(1.0, 1.5, "point"), _event(1.5, 2.0, "reach")]
    locomotion = [_event(0.25, 3.0, "walk"), _event(5.0, DURATION, "crawl")]
    return {
        "Gesture": EventTrack(schema_id="gesture", events=gesture, duration_s=DURATION),
        "Locomotion": EventTrack(schema_id="locomotion", events=locomotion, duration_s=DURATION),
    }


def test_frame_table_round_trip(tmp_path: Path) -> None:
    """Two categories with background frames between their events round-trip through the table."""
    tracks = _frame_tracks()
    path = tmp_path / "frames.csv"
    write_frame_table(tracks, path, fps=FPS, n_frames=N_FRAMES)
    lines = path.read_text().splitlines()
    assert lines[0] == "Frame,Gesture,Locomotion"
    assert lines[1] == "0,reach,N/A"
    assert len(lines) == N_FRAMES + 1
    back = read_frame_table(path, fps=FPS, schemas=_frame_schemas(), source="model_x")
    for column, track in tracks.items():
        assert _carried(back[column], "frame_table") == _carried(track, "frame_table")
        assert back[column].duration_s == DURATION and back[column].schema_id == track.schema_id


def test_frame_table_background_token_is_a_parameter(tmp_path: Path) -> None:
    """The background token can be changed, and the reader uses the same one."""
    tracks = {"Gesture": _frame_tracks()["Gesture"]}
    path = tmp_path / "frames.csv"
    write_frame_table(tracks, path, fps=FPS, n_frames=N_FRAMES, background="-")
    assert path.read_text().splitlines()[-1] == f"{N_FRAMES - 1},-"
    back = read_frame_table(path, fps=FPS, schemas={"Gesture": _frame_schemas()["Gesture"]}, source="x", background="-")
    assert len(back["Gesture"].events) == 3


def test_frame_table_writer_rejects_overlap(tmp_path: Path) -> None:
    """Two events of one category cannot share a frame."""
    overlapping = EventTrack(schema_id="gesture", events=[_event(0.0, 1.0, "reach"), _event(0.5, 1.5, "point")])
    with pytest.raises(ValueError, match="overlap"):
        write_frame_table({"Gesture": overlapping}, tmp_path / "f.csv", fps=FPS, n_frames=N_FRAMES)


def test_frame_table_writer_rejects_unaligned_onset(tmp_path: Path) -> None:
    """An onset between two frames cannot be written without moving it."""
    unaligned = EventTrack(schema_id="gesture", events=[_event(0.01, 1.0, "reach")])
    with pytest.raises(ValueError, match="frame-aligned"):
        write_frame_table({"Gesture": unaligned}, tmp_path / "f.csv", fps=FPS, n_frames=N_FRAMES)


@pytest.mark.parametrize(
    ("write", "read"),
    [
        (
            lambda t, p: write_rttm({"rec": t}, p),
            lambda p: read_rttm(p, schema_id="motion", source="model_x")["rec"],
        ),
        (write_bids_events, lambda p: read_bids_events(p, schema_id="motion", source="model_x")),
    ],
    ids=["rttm", "bids_events"],
)
def test_duration_formats_reconstruct_offsets_within_one_ulp(
    tmp_path: Path, write: Callable[[EventTrack, Path], None], read: Callable[[Path], EventTrack]
) -> None:
    """Formats that store a duration reconstruct a non-dyadic offset within one ulp.

    The last three pairs are ones where ``onset + (offset - onset)`` is not exactly ``offset``.
    """
    times = [(0.1, 0.3), (8.045, 24.048), (24.304, 60.879), (14.636, 49.52)]
    track = EventTrack(schema_id="motion", events=[_event(on, off, "reach") for on, off in times])
    path = tmp_path / "events.txt"
    write(track, path)
    back = read(path)
    for original, restored in zip(track.events, back.events):
        assert restored.onset_s == original.onset_s
        off = original.offset_s
        assert restored.offset_s in {off, float(np.nextafter(off, np.inf)), float(np.nextafter(off, -np.inf))}
    assert any(r.offset_s != o.offset_s for o, r in zip(track.events, back.events))
