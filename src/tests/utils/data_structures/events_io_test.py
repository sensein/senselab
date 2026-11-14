"""Tests for the behaviour-event interchange formats."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

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
    """Formats that store a duration reconstruct a non-dyadic offset within one ulp of the offset.

    ``(8.045, 24.048)``, ``(24.304, 60.879)`` and ``(14.636, 49.52)`` are pairs where
    ``onset + (offset - onset)`` is not exactly ``offset``; ``(0.3, 4.0)`` has a power-of-two offset,
    where the float spacing below the offset is half the spacing above it.
    """
    times = [(0.1, 0.3), (8.045, 24.048), (24.304, 60.879), (14.636, 49.52), (0.3, 4.0)]
    track = EventTrack(schema_id="motion", events=[_event(on, off, "reach") for on, off in times])
    path = tmp_path / "events.txt"
    write(track, path)
    back = read(path)
    for original, restored in zip(track.events, back.events):
        assert restored.onset_s == original.onset_s
        off = original.offset_s
        assert abs(restored.offset_s - off) <= math.ulp(off)
    assert any(r.offset_s != o.offset_s for o, r in zip(track.events, back.events))


# ── review fixes: lossless round trips and input checks ─────────────────


def test_bids_label_starting_with_a_quote_round_trips(tmp_path: Path) -> None:
    """BIDS fields split on tabs only, so a label may start with a quote and may contain spaces."""
    track = EventTrack(schema_id="motion", events=[_event(0.5, 1.0, '"quoted'), _event(1.0, 2.0, "two words")])
    path = tmp_path / "events.tsv"
    write_bids_events(track, path)
    back = read_bids_events(path, schema_id="motion", source="model_x")
    assert [e.label for e in back.events] == ['"quoted', "two words"]


@pytest.mark.parametrize("label", ["a\tb", "a\nb", "a\rb"])
def test_bids_writer_refuses_tabs_and_line_breaks(tmp_path: Path, label: str) -> None:
    """A tab or line break inside a label would split a row; the writer refuses it."""
    track = EventTrack(schema_id="motion", events=[_event(0.5, 1.0, label)])
    with pytest.raises(ValueError, match="tabs or line breaks"):
        write_bids_events(track, tmp_path / "events.tsv")


def test_bids_duration_na_raises_with_line(tmp_path: Path) -> None:
    """An ``n/a`` duration cannot make an event; the error names the file line."""
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\ttrial_type\n0.5\t0.25\treach\n1.0\tn/a\twave\n")
    with pytest.raises(ValueError, match="line 3: duration"):
        read_bids_events(path, schema_id="motion", source="model_x")


def test_bids_zero_duration_raises_with_line(tmp_path: Path) -> None:
    """An impulse event (duration 0) is refused, since an event needs offset > onset."""
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\ttrial_type\n0.5\t0\treach\n")
    with pytest.raises(ValueError, match="line 2: duration is 0; an event needs offset > onset"):
        read_bids_events(path, schema_id="motion", source="model_x")


def test_bids_missing_trial_type_raises(tmp_path: Path) -> None:
    """A file without the label column is refused with a ValueError naming it."""
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\n0.5\t0.25\n")
    with pytest.raises(ValueError, match="trial_type"):
        read_bids_events(path, schema_id="motion", source="model_x")


def test_frame_table_writer_refuses_an_event_covering_no_frame(tmp_path: Path) -> None:
    """An event shorter than the alignment tolerance rounds to zero frames and is refused."""
    tiny = EventTrack(schema_id="gesture", events=[_event(0.0, 1e-9, "reach")])
    with pytest.raises(ValueError, match="covers no frame"):
        write_frame_table({"Gesture": tiny}, tmp_path / "f.csv", fps=FPS, n_frames=N_FRAMES)


def test_frame_table_writer_refuses_a_duration_mismatch(tmp_path: Path) -> None:
    """A track whose duration is not ``n_frames / fps`` would not read back with its duration."""
    track = EventTrack(schema_id="gesture", events=[_event(0.0, 1.0, "reach")], duration_s=DURATION + 1)
    with pytest.raises(ValueError, match="n_frames / fps"):
        write_frame_table({"Gesture": track}, tmp_path / "f.csv", fps=FPS, n_frames=N_FRAMES)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan")])
def test_frame_table_refuses_bad_fps(tmp_path: Path, fps: float) -> None:
    """Both directions need a positive, finite frame rate."""
    path = tmp_path / "f.csv"
    with pytest.raises(ValueError, match="fps"):
        write_frame_table(_frame_tracks(), path, fps=fps, n_frames=N_FRAMES)
    write_frame_table(_frame_tracks(), path, fps=FPS, n_frames=N_FRAMES)
    with pytest.raises(ValueError, match="fps"):
        read_frame_table(path, fps=fps, schemas=_frame_schemas(), source="model_x")


def test_frame_table_writer_refuses_zero_frames(tmp_path: Path) -> None:
    """A table needs at least one frame."""
    with pytest.raises(ValueError, match="n_frames"):
        write_frame_table({}, tmp_path / "f.csv", fps=FPS, n_frames=0)


def test_frame_table_reader_needs_a_frame_column(tmp_path: Path) -> None:
    """A table without the ``Frame`` column is a ValueError, not a KeyError."""
    path = tmp_path / "f.csv"
    path.write_text("Index,Gesture\n0,reach\n")
    with pytest.raises(ValueError, match="Frame"):
        read_frame_table(path, fps=FPS, schemas={"Gesture": _frame_schemas()["Gesture"]}, source="model_x")


def test_label_studio_does_not_claim_duration(tmp_path: Path) -> None:
    """An empty track has no result to hold the duration, so the format does not declare it."""
    assert "duration_s" not in FORMAT_FIELDS["label_studio"]
    path = tmp_path / "ls.json"
    write_label_studio(EventTrack(schema_id="motion", events=[], duration_s=DURATION), path, from_name="l", to_name="v")
    back = read_label_studio(path, schema_id="motion", source="model_x")
    assert back.events == () and back.duration_s is None


def test_rttm_drops_empty_tracks(tmp_path: Path) -> None:
    """RTTM has no record for an empty track, so its recording id does not come back."""
    path = tmp_path / "events.rttm"
    write_rttm({"empty": EventTrack(schema_id="motion", events=[]), "full": _track()}, path)
    assert sorted(read_rttm(path, schema_id="motion", source="model_x", confidence_kind="prob")) == ["full"]


def test_rttm_short_speaker_line_raises_with_line(tmp_path: Path) -> None:
    """A truncated ``SPEAKER`` line is an error naming its line, not a silently skipped event."""
    path = tmp_path / "events.rttm"
    path.write_text("SPEAKER rec1 1 0.5 0.25 <NA> <NA> reach <NA> <NA>\nSPEAKER rec1 1 0.5 0.25 <NA>\n")
    with pytest.raises(ValueError, match="line 2"):
        read_rttm(path, schema_id="motion", source="model_x")


_UNICODE_ROUND_TRIP = r"""
import locale, shutil, sys, tempfile
from pathlib import Path
from senselab.utils.data_structures.events import AnnotationSchema, BehaviorEvent, EventTrack
from senselab.utils.data_structures import events_io as io

assert "utf" not in locale.getpreferredencoding(False).lower(), locale.getpreferredencoding(False)
label = "\u00f1and\u00fa"
event = BehaviorEvent(onset_s=0.0, offset_s=1.0, label=label, source="m\u00f6del", subject_ref="s\u00e9")
track = EventTrack(schema_id="motion", events=[event], duration_s=10.0)
schema = AnnotationSchema(schema_id="motion", task_type="segmentation", labels=[label])
tmp = Path(tempfile.mkdtemp())
io.write_segments_csv(track, tmp / "s.csv")
assert io.read_segments_csv(tmp / "s.csv", schema_id="motion", duration_s=10.0) == track
io.write_rttm({"r": track}, tmp / "e.rttm")
assert io.read_rttm(tmp / "e.rttm", schema_id="motion", source="x")["r"].events[0].label == label
io.write_bids_events(track, tmp / "e.tsv")
assert io.read_bids_events(tmp / "e.tsv", schema_id="motion", source="x").events[0].label == label
io.write_activitynet_json({"v": track}, tmp / "a.json")
assert io.read_activitynet_json(tmp / "a.json", schema_id="motion", source="x")["v"].events[0].label == label
io.write_label_studio(track, tmp / "l.json", from_name="l", to_name="v")
assert io.read_label_studio(tmp / "l.json", schema_id="motion", source="x").events[0].label == label
io.write_frame_table({"C": track}, tmp / "f.csv", fps=10.0, n_frames=100)
assert io.read_frame_table(tmp / "f.csv", 10.0, {"C": schema}, source="x")["C"].events[0].label == label
(tmp / "schema.yaml").write_bytes(("schema_id: motion\ntask_type: segmentation\nlabels: [" + label + "]\n").encode())
assert AnnotationSchema.from_yaml(tmp / "schema.yaml").labels == [label]
shutil.rmtree(tmp)
print(tmp)
print("unicode-ok")
"""


def test_unicode_round_trips_under_a_non_utf8_locale() -> None:
    """Every reader and writer uses UTF-8 even when the locale's preferred encoding is ASCII."""
    src = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONUTF8": "0", "LC_ALL": "C", "LANG": "C"}
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONPATH"] = os.pathsep.join([str(src), env.get("PYTHONPATH", "")])
    done = subprocess.run(
        [sys.executable, "-c", _UNICODE_ROUND_TRIP], env=env, capture_output=True, text=True, encoding="utf-8"
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = done.stdout.strip().splitlines()
    assert lines[-1] == "unicode-ok"
    assert not Path(lines[-2]).exists()


# ── Review fixes ──────────────────────────────────────────────────────


def test_bids_reader_accepts_a_byte_order_mark(tmp_path: Path) -> None:
    """A spreadsheet-exported TSV often starts with a BOM; the header still reads as ``onset``."""
    path = tmp_path / "events.tsv"
    path.write_bytes("\ufeffonset\tduration\ttrial_type\n0.5\t0.25\treach\n".encode("utf-8"))
    back = read_bids_events(path, schema_id="motion", source="model_x")
    assert [(e.onset_s, e.offset_s, e.label) for e in back.events] == [(0.5, 0.75, "reach")]


@pytest.mark.parametrize(
    "row",
    ["abc\t0.25\treach", "0.5\t-0.25\treach", "0.5\t0.25", "0.5\tx\treach"],
    ids=["non-numeric-onset", "negative-duration", "missing-field", "non-numeric-duration"],
)
def test_bids_row_errors_name_the_line(tmp_path: Path, row: str) -> None:
    """Every row-level error names the file line it came from."""
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\ttrial_type\n0.0\t0.25\treach\n" + row + "\n")
    with pytest.raises(ValueError, match="line 3"):
        read_bids_events(path, schema_id="motion", source="model_x")


@pytest.mark.parametrize(
    "line",
    [
        "SPEAKER rec1 1 abc 0.25 <NA> <NA> reach <NA> <NA>",
        "SPEAKER rec1 1 0.5 0 <NA> <NA> reach <NA> <NA>",
        "SPEAKER rec1 1 0.5 -1 <NA> <NA> reach <NA> <NA>",
    ],
    ids=["non-numeric-onset", "zero-duration", "negative-duration"],
)
def test_rttm_line_errors_name_the_line(tmp_path: Path, line: str) -> None:
    """RTTM value errors name the file line, including a zero duration."""
    path = tmp_path / "events.rttm"
    path.write_text("SPEAKER rec1 1 0.0 0.25 <NA> <NA> reach <NA> <NA>\n" + line + "\n")
    with pytest.raises(ValueError, match="line 2"):
        read_rttm(path, schema_id="motion", source="model_x")


def test_frame_table_refuses_a_boolean_fps(tmp_path: Path) -> None:
    """``True`` is not a frame rate, even though it compares equal to 1."""
    path = tmp_path / "f.csv"
    with pytest.raises(ValueError, match="fps"):
        write_frame_table(_frame_tracks(), path, fps=True, n_frames=N_FRAMES)
    write_frame_table(_frame_tracks(), path, fps=FPS, n_frames=N_FRAMES)
    with pytest.raises(ValueError, match="fps"):
        read_frame_table(path, fps=True, schemas=_frame_schemas(), source="model_x")
