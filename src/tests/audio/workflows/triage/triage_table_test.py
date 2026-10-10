"""Tests for triage.tsv: one flat row per recording over a synthetic decisions and evidence pair."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from senselab.audio.workflows.triage import decision_tables, triage_table

BIDS = "bids"
HEADER = [
    "participant_id",
    "session_id",
    "task",
    "task_family",
    "branch",
    "source_path",
    "verdict",
    "release",
    "release_reason",
    "reason",
    "reasons",
    "run_status",
    "missing",
    "recording_duration_s",
    "task_start_s",
    "task_end_s",
    "task_duration_s",
    "task_events_found",
    "task_events_instructed",
    "annotations",
    "decisive_evidence",
    "level_rel_session_db",
    "active_s",
    "interference_in_task",
    "fault_in_task",
    "task_audio_plain",
    "task_audio_enhanced",
    "task_audio_redacted",
    "figure",
    "pipeline_commit",
    "config_hash",
    "schema_version",
]


def _item(name: str, value: Any, effect: str, comparison: str | None = None, threshold: Any = None) -> dict:  # noqa: ANN401
    """An evidence item as ``triage_decisions.evidence`` stores it."""
    return {
        "name": name,
        "value": json.dumps(value),
        "unit": None,
        "comparison": comparison,
        "threshold": None if threshold is None else json.dumps(threshold),
        "effect": effect,
    }


def _recording(
    root: Path, task: str, family: str, verdict: str, release: str | None, evidence: list[dict], cuts: list[str]
) -> dict[str, Any]:
    """One run directory in the tree, and its ``triage_decisions`` row."""
    participant, session = "sub-0123abcd", "ses-01"
    stem = f"{participant}_{session}_task-{task}"
    run_dir = Path(participant) / session / f"{stem}_20261008-000000"
    streams = root / "out" / run_dir / "run" / "streams"
    streams.mkdir(parents=True)
    source = root / BIDS / participant / session / "audio" / f"{stem}.wav"
    lines = [
        {"record": "activity", "node": "ADMIT", "parameters": {}},
        {
            "record": "entity",
            "id": "rec",
            "prov_type": "stream",
            "extent": [0.0, 12.5],
            "attributes": {"name": "recording", "path": str(source)},
        },
    ]
    (streams.parent / "store.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    for cut in cuts:
        (streams / f"{cut}.flac").write_bytes(b"fLaC")
    audio = [str(run_dir / "run" / "streams" / f"{name}.flac") for name in ("task_plain", "task_enhanced")]
    return {
        "participant": participant,
        "session": session,
        "task": task,
        "declared_family": family,
        "stem": stem,
        "run_dir": str(run_dir),
        "verdict": verdict,
        "release": release,
        "release_reason": None if release is None else "scan_found_nothing",
        "reason": None if verdict == "pass" else "weak_events",
        "reasons": [] if verdict == "pass" else ["weak_events", "second\treason"],
        "run_status": "complete",
        "missing": ["phonation_tracks", "residual"] if verdict == "discard" else [],
        "extent_start_s": 1.13,
        "extent_end_s": 9.5800000000000001,
        "extent_duration_s": 8.45,
        "annotations": ["fault_outside_task:clip", "line\nbreak"],
        "evidence": evidence,
        "figure_path": str(run_dir / "summary" / "summary.pdf"),
        "audio_paths": audio,
        "commit": "abc123",
        "config_hash": "c0ffee",
        "schema_version": 1,
    }


def _evidence(stem: str, name: str, value: Any, threshold: Any = None) -> dict[str, Any]:  # noqa: ANN401
    """One ``triage_evidence`` row."""
    return {
        "stem": stem,
        "declared_family": None,
        "verdict": None,
        "group": "acquisition",
        "name": name,
        "value": json.dumps(value),
        "unit": None,
        "comparison": ">=" if threshold is not None else None,
        "threshold": None if threshold is None else json.dumps(threshold),
        "effect": "annotation",
        "decisive": False,
        "schema_version": 1,
    }


def _tables(root: Path) -> tuple[pa.Table, pa.Table]:
    """Three recordings, one per branch and verdict, with their evidence."""
    breath = _recording(
        root,
        "respiration-and-cough-breath-1",
        "respiration-and-cough-breath",
        "review",
        "as_is",
        [
            _item("breath_event_db_over_local", 9.4, "review", ">=", 16.0),
            _item("breath_review_low_confidence", True, "review", "==", False),
        ],
        ["task_plain", "task_enhanced"],
    )
    speech = _recording(
        root,
        "free-speech-1",
        "free-speech",
        "pass",
        "redacted",
        [_item("gate:dominant_speaker_share_min", 0.9898385139140226, "pass", ">=", 0.9)],
        ["task_plain"],
    )
    vowel = _recording(
        root, "prolonged-vowel", "prolonged-vowel", "discard", "as_is", [_item("phonation_found", False, "discard")], []
    )
    evidence = [
        _evidence(breath["stem"], "capture.level_rel_db", -3.217),
        _evidence(breath["stem"], "capture.plain_active_s", 7.5),
        _evidence(breath["stem"], "breaths_found", 3, 5),
        _evidence(breath["stem"], "interference_in_task:other_voice", 2, 0),
        _evidence(breath["stem"], "fault_in_task:dropout", 5.526, 0.5),
        _evidence(breath["stem"], "fault_in_task:clip", 0.047, 0.02),
        _evidence(speech["stem"], "capture.level_rel_db", 0.5),
        _evidence(speech["stem"], "lexical_words", 40),
    ]
    decisions = pa.Table.from_pylist([breath, speech, vowel], schema=decision_tables.decisions_schema())
    return decisions, pa.Table.from_pylist(evidence, schema=decision_tables.evidence_schema())


def _written(tmp_path: Path) -> tuple[list[list[str]], str]:
    """Build and write the table; return its parsed rows and raw text."""
    decisions, evidence = _tables(tmp_path)
    rows = triage_table.rows(decisions, evidence, tmp_path / "out", tmp_path / BIDS, threads=2)
    path = tmp_path / "pub" / triage_table.TABLE_NAME
    assert triage_table.write_tsv(rows, path) == 3
    text = path.read_text(encoding="utf-8")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)), text


def test_the_header_is_the_column_dictionary_in_order(tmp_path: Path) -> None:
    """The dictionary lists the approved columns in order, and the TSV's header is that list."""
    assert triage_table.columns() == HEADER
    parsed, _ = _written(tmp_path)
    assert parsed[0] == HEADER
    copied = yaml.safe_load((tmp_path / "pub" / triage_table.COLUMNS_NAME).read_text())
    assert [column["name"] for column in copied["columns"]] == HEADER


def test_no_field_carries_a_tab_or_a_line_break(tmp_path: Path) -> None:
    """Every line splits into exactly the header's fields, so no tab or newline leaked into a field."""
    parsed, text = _written(tmp_path)
    lines = text.split("\n")
    assert lines[-1] == "" and len(lines) == 5
    assert all(line.count("\t") == len(HEADER) - 1 for line in lines[:-1])
    assert "\r" not in text
    assert all(len(row) == len(HEADER) for row in parsed)


def test_lists_are_joined_and_acquisition_readings_fill_their_columns(tmp_path: Path) -> None:
    """Reasons, annotations, interference and faults join with ;; the counts come from the evidence."""
    parsed, _ = _written(tmp_path)
    breath = dict(zip(HEADER, parsed[1]))
    assert breath["branch"] == "AIRWAY"
    assert breath["reasons"] == "weak_events;second reason"
    assert breath["annotations"] == "fault_outside_task:clip;line break"
    assert breath["interference_in_task"] == "other_voice"
    assert breath["fault_in_task"] == "clip:0.047;dropout:5.526"
    assert (breath["task_events_found"], breath["task_events_instructed"]) == ("3", "5")
    assert (breath["level_rel_session_db"], breath["active_s"]) == ("-3.217", "7.5")
    assert (breath["recording_duration_s"], breath["task_start_s"], breath["task_end_s"]) == ("12.5", "1.13", "9.58")
    assert breath["source_path"] == (
        "sub-0123abcd/ses-01/audio/sub-0123abcd_ses-01_task-respiration-and-cough-breath-1.wav"
    )
    assert breath["task_audio_plain"].endswith("run/streams/task_plain.flac")
    assert breath["task_audio_enhanced"].endswith("run/streams/task_enhanced.flac")
    assert breath["task_audio_redacted"] == ""
    assert (breath["pipeline_commit"], breath["config_hash"], breath["schema_version"]) == ("abc123", "c0ffee", "2")
    speech = dict(zip(HEADER, parsed[2]))
    assert speech["branch"] == "SPEECH" and speech["task_audio_enhanced"] == ""
    assert speech["task_events_found"] == "" and speech["interference_in_task"] == ""


def test_a_discard_has_no_release(tmp_path: Path) -> None:
    """A discard row's release is empty even where the decision row carries one."""
    parsed, _ = _written(tmp_path)
    vowel = dict(zip(HEADER, parsed[3]))
    assert (vowel["verdict"], vowel["release"], vowel["branch"]) == ("discard", "", "VOICE")
    assert vowel["missing"] == "phonation_tracks;residual"
    assert dict(zip(HEADER, parsed[2]))["release"] == "redacted"


def test_decisive_evidence_is_name_value_op_threshold_arrow_effect(tmp_path: Path) -> None:
    """Each decisive item is name=value<op>threshold→effect, joined with "; "."""
    parsed, _ = _written(tmp_path)
    assert dict(zip(HEADER, parsed[1]))["decisive_evidence"] == (
        "breath_event_db_over_local=9.4>=16→review; breath_review_low_confidence=true==false→review"
    )
    assert dict(zip(HEADER, parsed[2]))["decisive_evidence"] == "gate:dominant_speaker_share_min=0.9898>=0.9→pass"
    assert dict(zip(HEADER, parsed[3]))["decisive_evidence"] == "phonation_found=false→discard"


def test_evidence_text_reads_strings_and_structures_compactly() -> None:
    """A JSON string reads bare and a list as compact JSON."""
    assert triage_table.evidence_text(_item("release_ground", "findings_are_task_content", "pass")) == (
        "release_ground=findings_are_task_content→pass"
    )
    assert triage_table.evidence_text(_item("kinds", ["a", "b"], "review", "not in", ["c"])) == (
        'kinds=["a","b"]not in["c"]→review'
    )
