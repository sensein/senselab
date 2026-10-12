"""Tests for triage.tsv: one flat row per recording over a synthetic decisions and evidence pair."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
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
    "issues_n",
    "run_status",
    "missing",
    "recording_duration_s",
    "task_start_s",
    "task_end_s",
    "task_duration_s",
    "task_fraction",
    "task_events_found",
    "task_events_instructed",
    "annotations",
    "decisive_evidence",
    "level_rel_session_db",
    "active_s",
    "interference_in_task",
    "fault_in_task",
    "snr_db",
    "noise_floor_dbfs",
    "session_background_dbfs",
    "speech_over_residual_db",
    "squim_pesq",
    "squim_stoi",
    "squim_si_sdr",
    "clipped_s",
    "dropout_s",
    "lexical_words_n",
    "off_task_words_n",
    "asr_agreement",
    "asr_single_hypothesis",
    "asr_degenerate_n",
    "masked_s",
    "masked_fraction",
    "speaker_target_cosine",
    "speaker_non_task_cosine",
    "session_consistency_cosine",
    "enrollment_source",
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


def _measurement(name: str, **attributes: Any) -> dict[str, Any]:  # noqa: ANN401
    """A live measurement entity line."""
    return {
        "record": "entity",
        "id": f"m-{name}",
        "prov_type": "measurement",
        "attributes": {"name": name, **attributes},
    }


SPEECH_STORE = [
    _measurement("session_floor", session_background={"level_db": -71.75}),
    _measurement("speech_residual", foreground_minus_residual_db={"whole": 24.8, "speech": 25.0}),
    _measurement("disruptions_file", clipped_s=0.0, dropout_s=0.0),
    {"record": "entity", "id": "clip-1", "prov_type": "span", "extent": [1.0, 1.1], "attributes": {"family": "clip"}},
    {"record": "entity", "id": "clip-2", "prov_type": "span", "extent": [2.0, 2.15], "attributes": {"family": "clip"}},
    _measurement(
        "consensus_transcript",
        outcomes={"agreement": 30, "variant": 6, "insertion": 4},
        single_hypothesis=False,
        degenerate_n=2,
    ),
    _measurement(
        "pii_ledger",
        non_task_speech_ids=["w1", "w2", "w3"],
        final_masks=[{"start_s": 0.5, "end_s": 2.13}, {"start_s": 2.0, "end_s": 3.0}, {"start_s": 9.0, "end_s": 11.0}],
    ),
    _measurement(
        "cohort_reading",
        other_speaker={
            "status": "measured",
            "enrollment_source": "session",
            "target": {"status": "measured", "cosine": 0.71},
            "non_task": {"status": "not_measured"},
        },
        checks={
            "session_speaker_consistency": {
                "status": "measured",
                "comparisons": [{"name": "session_vs_other_sessions_cosine", "value": 0.82}],
            }
        },
    ),
]


def _recording(
    root: Path,
    task: str,
    family: str,
    verdict: str,
    release: str | None,
    evidence: list[dict],
    cuts: list[str],
    store: list[dict] | None = None,
) -> dict[str, Any]:
    """One run directory in the tree, and its ``triage_decisions`` row."""
    participant, session = "sub-SUBJ13", "ses-01"
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
        *(store or ()),
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
        SPEECH_STORE,
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
        _evidence(breath["stem"], "speech_in_task", 7, 3),
    ]
    decisions = pa.Table.from_pylist([breath, speech, vowel], schema=decision_tables.decisions_schema())
    return decisions, pa.Table.from_pylist(evidence, schema=decision_tables.evidence_schema())


def _vectors(decisions: pa.Table) -> pa.Table:
    """The recording vectors' whole-file readings for every recording."""
    stems = decisions.column("stem").to_pylist()
    return pa.table(
        {
            "stem": stems,
            "q_plain_snr_db": [31.5, 28.25, None],
            "q_plain_floor_dbfs": [-70.0, -66.5, None],
            "q_plain_squim_pesq": [2.5, 3.25, None],
            "q_plain_squim_stoi": [0.9, 0.95, None],
            "q_plain_squim_si_sdr": [12.0, 18.5, None],
        }
    )


def _written(tmp_path: Path) -> tuple[list[list[str]], str]:
    """Build and write the table; return its parsed rows and raw text."""
    decisions, evidence = _tables(tmp_path)
    rows = triage_table.rows(
        decisions, evidence, tmp_path / "out", tmp_path / BIDS, threads=2, vectors=_vectors(decisions)
    )
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
        "sub-SUBJ13/ses-01/audio/sub-SUBJ13_ses-01_task-respiration-and-cough-breath-1.wav"
    )
    assert breath["task_audio_plain"].endswith("run/streams/task_plain.flac")
    assert breath["task_audio_enhanced"].endswith("run/streams/task_enhanced.flac")
    assert breath["task_audio_redacted"] == ""
    assert (breath["pipeline_commit"], breath["config_hash"], breath["schema_version"]) == ("abc123", "c0ffee", "3")
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


def test_the_whole_file_qc_columns_fill_from_the_store_the_vectors_and_the_evidence(tmp_path: Path) -> None:
    """Issues, task fraction, quality, speech-content and speaker columns, each from its stated source."""
    parsed, _ = _written(tmp_path)
    breath, speech, vowel = (dict(zip(HEADER, row)) for row in parsed[1:4])
    assert breath["issues_n"] == "4" and speech["issues_n"] == "2"
    assert float(breath["task_fraction"]) == pytest.approx(8.45 / 12.5, abs=1e-3)
    assert (breath["snr_db"], breath["noise_floor_dbfs"]) == ("31.5", "-70")
    assert (breath["squim_pesq"], breath["squim_stoi"], breath["squim_si_sdr"]) == ("", "", ""), "airway: no SQUIM"
    assert (speech["squim_pesq"], speech["squim_stoi"], speech["squim_si_sdr"]) == ("3.25", "0.95", "18.5")
    assert (speech["session_background_dbfs"], speech["speech_over_residual_db"]) == ("-71.75", "24.8")
    assert (speech["clipped_s"], speech["dropout_s"]) == ("0.25", "0")
    assert (speech["lexical_words_n"], speech["off_task_words_n"]) == ("40", "3")
    assert breath["off_task_words_n"] == "7", "a non-lexical task's speech words, with no ledger"
    assert (speech["asr_agreement"], speech["asr_single_hypothesis"], speech["asr_degenerate_n"]) == (
        "0.75",
        "false",
        "2",
    )
    assert speech["masked_s"] == "2.45", "the masks inside the 1.13-9.58 s extent, merged"
    assert float(speech["masked_fraction"]) == pytest.approx(2.45 / 8.45, abs=1e-3)
    assert (speech["speaker_target_cosine"], speech["speaker_non_task_cosine"]) == ("0.71", "")
    assert (speech["session_consistency_cosine"], speech["enrollment_source"]) == ("0.82", "session")
    assert vowel["masked_s"] == "" and vowel["snr_db"] == ""
    assert breath["enrollment_source"] == "" and breath["asr_agreement"] == ""


def test_a_withheld_release_reports_no_masked_seconds(tmp_path: Path) -> None:
    """Withheld audio releases nothing, so nothing of it is masked."""
    decisions, evidence = _tables(tmp_path)
    rows = list(triage_table.rows(decisions, evidence, tmp_path / "out", tmp_path / BIDS, threads=1))
    withheld = dict(decisions.to_pylist()[1], release="withheld")
    tree = triage_table.run_tree_facts(withheld["run_dir"], (), tmp_path / "out", tmp_path / BIDS, extent=(1.13, 9.58))
    assert rows[1]["masked_s"] == 2.45 and rows[1]["snr_db"] is None
    assert triage_table.row(withheld, {}, tree, 3)["masked_s"] is None


def test_an_older_consensus_reads_single_hypothesis_off_its_source_count() -> None:
    """A consensus written before single_hypothesis existed is single where it had one source."""
    from senselab.audio.workflows.triage.recording_vectors import Entity, StoreView

    def view(attributes: dict[str, Any]) -> StoreView:
        entity = Entity("c", "measurement", None, {"name": "consensus_transcript", **attributes})
        return StoreView(entities=[entity])

    assert triage_table.store_readings(view({"n_sources": 1, "outcomes": {}}), None)["asr_single_hypothesis"] is True
    assert triage_table.store_readings(view({"n_sources": 2, "outcomes": {}}), None)["asr_single_hypothesis"] is False
    assert triage_table.store_readings(view({"outcomes": {}}), None)["asr_single_hypothesis"] is None
