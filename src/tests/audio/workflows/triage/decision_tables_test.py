"""Tests for the decision tables: a row per recording from the fold's own record, and every evidence item."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from senselab.audio.workflows.triage import decision_tables
from senselab.audio.workflows.triage.vocabulary import (
    ROUTED,
    BranchDecision,
    NodeVerdict,
    Outcome,
    TaskEvidence,
    Triage,
    fold_file_verdict,
)

STEM = "sub-0123abcd_ses-01_task-respiration-and-cough-breath"


def _fold(**task: Any) -> dict[str, Any]:  # noqa: ANN401 -- TaskEvidence fields
    """A fold over a breath task, its record as VERDICT stores it."""
    folded = fold_file_verdict(
        [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
        branch_decisions={
            "AIRWAY": BranchDecision(branch="AIRWAY", will_run=False, route_state=ROUTED, forced_by_declaration=False),
            "SPEECH": BranchDecision(
                branch="SPEECH", will_run=False, route_state="declined", forced_by_declaration=False
            ),
        },
        ran={},
        hint_claims={},
        route_state=ROUTED,
        declared_family="respiration-and-cough-breath",
        task=TaskEvidence(**task),
    )
    return {"node": "VERDICT", **folded.record()}


def _run_root(root: Path, decision: dict[str, Any]) -> Path:
    """A finished run directory holding one fold, a task extent, a cut and its activities."""
    run_root = root / "sub-0123abcd" / "ses-01" / f"{STEM}_20261007-120000"
    (run_root / "run").mkdir(parents=True)
    (run_root / "summary").mkdir()
    (run_root / "summary" / "summary.pdf").write_bytes(b"%PDF")
    lines = [
        {
            "record": "entity",
            "id": "rec",
            "prov_type": "stream",
            "extent": [0.0, 9.0],
            "attributes": {"name": "recording", "path": "streams/recording.flac"},
        },
        {
            "record": "entity",
            "id": "cut",
            "prov_type": "stream",
            "extent": [1.0, 7.0],
            "attributes": {"name": "task_plain", "path": "streams/task_plain.flac"},
        },
        {
            "record": "entity",
            "id": "ext",
            "prov_type": "span",
            "extent": [1.0, 7.0],
            "attributes": {"role": "task_extent"},
        },
        {"record": "entity", "id": "fold", "prov_type": "verdict", "extent": None, "attributes": decision},
        {"record": "activity", "node": "VERDICT", "step": None, "parameters": {"config_hash": "c0ffee"}},
        {"record": "activity", "node": "REFOLD", "step": "marker", "parameters": {"commit": "abc123"}},
    ]
    (run_root / "run" / "store.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return run_root


def _breath(floor_db: float, local_db: float | None) -> dict[str, Any]:  # noqa: ANN401
    """A breath task's evidence with the task layer's decision inputs."""
    inputs = {"floor_db": floor_db, "local_db": local_db, "entangled": False, "snr_low_db": 10.0, "snr_high_db": 16.0}
    decision = "absent" if floor_db < 10.0 else "present" if (local_db or 0) >= 16.0 else "review"
    return {
        "owning_branches": ("AIRWAY",),
        "duration_s": 9.0,
        "minimum_duration_s": 1.0,
        "required_event": "breath",
        "breath_mode": "sustained",
        "breath_decision": decision,
        "breath_review": decision == "review",
        "breath_train_breaths": 4,
        "breath_reading": {"evidence": {"inputs": inputs, "rhythm": {"hz": 0.25, "prominence_db": 9.0}}},
    }


def test_a_pass_row_carries_its_decisive_task_evidence_and_its_paths(tmp_path: Path) -> None:
    """The decision row reads verdict, release and reason off the fold; the evidence is the decisive items."""
    run_root = _run_root(tmp_path, _fold(**_breath(24.0, 20.0)))
    row, evidence = decision_tables.decision_rows(run_root, tmp_path)  # type: ignore[misc]
    assert row["verdict"] == Triage.PASS.value
    assert row["release"] == "as_is" and row["reason"] is None
    assert (row["extent_start_s"], row["extent_end_s"], row["extent_duration_s"]) == (1.0, 7.0, 6.0)
    assert {item["name"] for item in row["evidence"]} == {
        "breath_event_db_over_floor",
        "breath_event_db_over_local",
        "breath_events_entangled",
    }
    assert row["figure_path"].endswith("summary/summary.pdf")
    assert row["audio_paths"] == [str((run_root / "run" / "streams" / "task_plain.flac").relative_to(tmp_path))]
    assert (row["commit"], row["config_hash"]) == ("abc123", "c0ffee")
    names = {item["name"]: item for item in evidence}
    assert names["breath_rhythm_hz"]["effect"] == "annotation" and not names["breath_rhythm_hz"]["decisive"]
    assert names["breath_event_db_over_local"]["threshold"] == "16.0"
    assert names["breath_event_db_over_local"]["group"] == "breath"


def test_a_discard_row_names_the_one_reading_that_discarded_it(tmp_path: Path) -> None:
    """No breath over the floor: the floor reading is the only decisive item, and there is no release."""
    row, _ = decision_tables.decision_rows(_run_root(tmp_path, _fold(**_breath(6.0, None))), tmp_path)  # type: ignore[misc]
    assert row["verdict"] == Triage.DISCARD.value
    assert row["release"] is None and row["reason"] == "no_task_captured"
    assert [(item["name"], item["effect"]) for item in row["evidence"]] == [("breath_event_db_over_floor", "discard")]


def test_a_review_row_carries_every_review_item_and_its_reasons(tmp_path: Path) -> None:
    """A weak breath: the local-background reading and the review ground decide it."""
    row, _ = decision_tables.decision_rows(_run_root(tmp_path, _fold(**_breath(14.0, 12.0))), tmp_path)  # type: ignore[misc]
    assert row["verdict"] == Triage.REVIEW.value
    assert row["reason"] == "weak_events" and row["reasons"] == ["weak_events"]
    decisive = {item["name"] for item in row["evidence"]}
    assert {"breath_event_db_over_local", "breath_review_low_confidence"} <= decisive


def test_the_tables_write_parquet_and_a_tsv_whose_lists_are_json(tmp_path: Path) -> None:
    """The TSV carries the same rows, its list and struct columns as compact JSON."""
    runs = tmp_path / "runs"
    decisions, evidence = decision_tables.tables([_run_root(runs, _fold(**_breath(24.0, 20.0)))], runs)
    paths = decision_tables.write_tables(decisions, evidence, tmp_path / "out")
    assert pq.read_table(paths["triage_decisions"]).num_rows == 1
    assert pq.read_table(paths["triage_evidence"]).num_rows == evidence.num_rows > 0
    header, line = paths["triage_decisions_tsv"].read_text().splitlines()
    cells = dict(zip(header.split("\t"), line.split("\t")))
    assert json.loads(cells["evidence"])[0]["name"] == "breath_event_db_over_floor"


def test_every_emitted_item_name_is_in_the_evidence_vocabulary(tmp_path: Path) -> None:
    """An item the vocabulary does not group falls to the flag group only if it is a ground key."""
    _, evidence = decision_tables.decision_rows(_run_root(tmp_path, _fold(**_breath(14.0, 12.0))), tmp_path)  # type: ignore[misc]
    vocabulary = decision_tables.evidence_vocabulary()
    grouped = {name for items in vocabulary["groups"].values() for name in items}
    for item in evidence:
        assert item["name"] in grouped or item["group"] == vocabulary["flags"]["group"]
