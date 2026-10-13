"""Tests for ``scripts/evaluate_task_events.py``: owner labels against triage decisions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "evaluate_task_events.py"
_spec = importlib.util.spec_from_file_location("evaluate_task_events", SCRIPT)
assert _spec is not None and _spec.loader is not None
ev = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_task_events"] = ev
_spec.loader.exec_module(ev)

LABEL_MAP = yaml.safe_load((ev.DATA / "owner_label_map.yaml").read_text())


def _stem(hexid: str, task: str = "respiration-and-cough-breath") -> str:
    return f"sub-{hexid}-0000-0000-0000-000000000000_ses-AAAA_task-{task}"


def _labels() -> pd.DataFrame:
    rows = [
        ("set_a", _stem(f"{i:08x}"), "respiration-and-cough-breath", label, "")
        for i, label in enumerate(["breath_present"] * 6 + ["no_breath"] * 4)
    ]
    rows.append(("set_a", _stem("ffffffff"), "respiration-and-cough-breath", "", "ok if flagged or discarded"))
    rows.append(("set_a", _stem("eeeeeeee"), "respiration-and-cough-breath", "not_judged", ""))
    return pd.DataFrame(rows, columns=["listen_set", "stem", "family", "owner_label", "owner_note"])


def test_every_owner_label_in_the_map_names_a_known_target() -> None:
    """A label mapped to a target the map does not define could never agree."""
    targets = set(LABEL_MAP["targets"])
    assert {entry["target"] for entry in LABEL_MAP["labels"].values()} <= targets
    assert {rule["target"] for rule in LABEL_MAP["notes"]} <= targets


def test_a_free_text_note_maps_when_the_label_is_empty_and_unmapped_rows_are_reported() -> None:
    """The note rule applies to an empty label; a label no rule covers is listed, not guessed."""
    labels = pd.concat(
        [_labels(), pd.DataFrame([("set_a", _stem("dddddddd"), "x", "brand_new_label", "")], columns=_labels().columns)]
    )
    mapped, unmapped = ev.map_labels(labels, LABEL_MAP)
    assert mapped.set_index("owner_label").loc["", "target"] == "review_acceptable"
    assert len(unmapped) == 1 and "dddddddd" in unmapped[0]


def test_the_split_is_deterministic_grouped_by_subject_and_holds_out_about_a_third() -> None:
    """Same labels, same split; each stratum holds out round(30%) of its subjects."""
    mapped, _ = ev.map_labels(_labels(), LABEL_MAP)
    mapped = mapped[mapped["target"] != "excluded"]
    first = ev.make_split(mapped)
    assert first == ev.make_split(mapped)
    held = first["set_a"]["heldout"]
    assert len(held) == round(0.3 * 6) + round(0.3 * 4) + round(0.3 * 1)
    assert not set(held) & set(first["set_a"]["fit"])


def test_a_subject_absent_from_the_split_file_is_held_out(tmp_path: Path) -> None:
    """Labels added after the split was written go to held-out."""
    mapped, _ = ev.map_labels(_labels(), LABEL_MAP)
    split = ev.load_or_write_split(tmp_path / "split.yaml", mapped[mapped["target"] != "excluded"])
    assert (tmp_path / "split.yaml").exists()
    new_row = pd.Series({"stem": _stem("12345678"), "listen_set": "set_a"})
    assert ev.side(new_row, split) == "heldout"
    reread = ev.load_or_write_split(tmp_path / "split.yaml", mapped.iloc[:0])
    assert reread == split


def test_scoring_reads_a_rows_directory_and_counts_review_acceptable_both_ways(tmp_path: Path) -> None:
    """A refold row's verdict is read off ``REFOLD``; review-acceptable agrees with review and discard."""
    rows = tmp_path / "rows"
    rows.mkdir()
    lines = [
        {"stem": _stem("00000000") + "_20260920-030735", "REFOLD": "pass/as_is"},
        {"stem": _stem("00000006") + "_20260920-030735", "REFOLD": "pass/as_is"},
        {"stem": _stem("ffffffff"), "verdict": "discard", "ground_keys": ["no_breath_captured"]},
    ]
    (rows / "slice-0.jsonl").write_text("\n".join(json.dumps(line) for line in lines))
    decisions = ev.load_decisions(rows)
    mapped, _ = ev.map_labels(_labels(), LABEL_MAP)
    mapped["side"] = "fit"
    scored = ev.score(mapped, decisions, LABEL_MAP).set_index("subject")
    assert bool(scored.loc["00000000", "agree"]) is True
    assert bool(scored.loc["00000006", "agree"]) is False
    assert bool(scored.loc["ffffffff", "agree"]) is True
    assert scored.loc["00000001", "outcome"] == "missing"
    assert "eeeeeeee" not in scored.index


def test_the_review_band_rate_is_reported_per_group_over_kept_recordings() -> None:
    """Only kept recordings count; a review-band key marks one in the band."""
    decisions = pd.DataFrame(
        {
            "stem": ["a", "b", "c", "d"],
            "verdict": ["pass", "review", "discard", "review"],
            "ground_keys": [[], ["breath_review_low_confidence"], [], ["voice_review_low_confidence"]],
            "declared_family": ["respiration-and-cough-breath"] * 3 + ["prolonged-vowel"],
        }
    )
    rate = ev.review_band_rate(decisions)
    assert rate["breath"] == {"kept": 2, "in_review_band": 1, "rate": 0.5}
    assert rate["voice"]["rate"] == 1.0


def test_a_review_page_export_reads_as_labels_and_scores(tmp_path: Path) -> None:
    """The review page's JSON export carries the label table's keys; its verdicts map to targets."""
    export = {
        "schema": "senselab.triage.review",
        "version": 1,
        "build": "abc123",
        "entries": [
            {
                "listen_set": "triage_review_abc123",
                "stem": _stem("00000000") + "_20260920-030735",
                "family": "respiration-and-cough-breath",
                "instructed": "",
                "duration_s": 9.0,
                "owner_label": "reviewer_discard",
                "owner_note": "only the room",
                "pipeline_at_listen": "review (weak_events)",
                "reviewer_verdict": "discard",
            }
        ],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(export))
    labels = ev.load_labels(path)
    assert list(labels["owner_label"]) == ["reviewer_discard"]
    mapped, unmapped = ev.map_labels(labels, LABEL_MAP)
    assert unmapped == [] and list(mapped["target"]) == ["absent"]
    mapped["side"] = "fit"
    decisions = pd.DataFrame({"stem": [_stem("00000000")], "verdict": ["discard"], "ground_keys": [[]]})
    decisions["key"] = decisions["stem"].map(ev.recording_key)
    scored = ev.score(mapped, decisions.set_index("key"), LABEL_MAP)
    assert bool(scored.iloc[0]["agree"]) is True


def test_a_json_that_is_not_a_review_export_is_refused(tmp_path: Path) -> None:
    """A JSON from another tool is not read as labels."""
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "senselab.fsreview.decisions", "entries": []}))
    try:
        ev.load_labels(path)
    except ValueError as error:
        assert "not a triage review export" in str(error)
    else:
        raise AssertionError("expected a ValueError")
