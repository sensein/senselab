"""Tests for ``scripts/evaluate_task_events.py``: owner labels against triage decisions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.conftest import FIXTURE_PROFILE

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "evaluate_task_events.py"
_spec = importlib.util.spec_from_file_location("evaluate_task_events", SCRIPT)
assert _spec is not None and _spec.loader is not None
ev = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_task_events"] = ev
_spec.loader.exec_module(ev)

EVALUATION = FIXTURE_PROFILE / "evaluation"
LABEL_MAP = yaml.safe_load((EVALUATION / "owner_label_map.yaml").read_text())


def _stem(hexid: str, task: str = "respiration-and-cough-breath") -> str:
    return f"rec-{hexid}-{task}"


def _decisions(labels: pd.DataFrame) -> pd.DataFrame:
    """A decision table naming each labelled recording's participant, ``p<hexid>``, as the corpus table does."""
    table = pd.DataFrame(
        {
            "stem": list(labels["stem"]),
            "verdict": ["pass"] * len(labels),
            "ground_keys": [[] for _ in range(len(labels))],
            "participant": [f"p{stem.split('-')[1]}" for stem in labels["stem"]],
        }
    )
    table["key"] = table["stem"].map(ev.recording_key)
    return table.drop_duplicates("key", keep="last").set_index("key")


def _subjects(labels: pd.DataFrame) -> pd.DataFrame:
    return ev.with_subjects(labels, _decisions(labels))


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
    mapped, unmapped = ev.map_labels(_subjects(labels), LABEL_MAP)
    assert mapped.set_index("owner_label").loc["", "target"] == "review_acceptable"
    assert len(unmapped) == 1 and "dddddddd" in unmapped[0]


def test_the_split_is_deterministic_grouped_by_subject_and_holds_out_about_a_third() -> None:
    """Same labels, same split; each stratum holds out round(30%) of its subjects."""
    mapped, _ = ev.map_labels(_subjects(_labels()), LABEL_MAP)
    mapped = mapped[mapped["target"] != "excluded"]
    first = ev.make_split(mapped)
    assert first == ev.make_split(mapped)
    held = first["set_a"]["heldout"]
    assert len(held) == round(0.3 * 6) + round(0.3 * 4) + round(0.3 * 1)
    assert not set(held) & set(first["set_a"]["fit"])


def test_a_subject_absent_from_the_split_file_is_held_out(tmp_path: Path) -> None:
    """Labels added after the split was written go to held-out."""
    mapped, _ = ev.map_labels(_subjects(_labels()), LABEL_MAP)
    split = ev.load_or_write_split(tmp_path / "split.yaml", mapped[mapped["target"] != "excluded"])
    assert (tmp_path / "split.yaml").exists()
    new_row = pd.Series({"stem": _stem("12345678"), "listen_set": "set_a", "subject": "p12345678"})
    assert ev.side(new_row, split) == "heldout"
    reread = ev.load_or_write_split(tmp_path / "split.yaml", mapped.iloc[:0])
    assert reread == split


def test_scoring_reads_a_rows_directory_and_counts_review_acceptable_both_ways(
    tmp_path: Path, triage_profile: object
) -> None:
    """A refold row's verdict is read off ``REFOLD``; review-acceptable agrees with review and discard."""
    rows = tmp_path / "rows"
    rows.mkdir()
    lines = [
        {"stem": _stem("00000000") + "_20260920-030735", "REFOLD": "pass/as_is", "participant": "p00000000"},
        {"stem": _stem("00000006") + "_20260920-030735", "REFOLD": "pass/as_is", "participant": "p00000006"},
        {"stem": _stem("ffffffff"), "verdict": "discard", "ground_keys": ["no_breath_captured"], "participant": "pf"},
    ]
    (rows / "slice-0.jsonl").write_text("\n".join(json.dumps(line) for line in lines))
    decisions = ev.load_decisions(rows)
    mapped, _ = ev.map_labels(ev.with_subjects(_labels(), decisions), LABEL_MAP)
    mapped["side"] = "fit"
    scored = ev.score(mapped, decisions, LABEL_MAP).set_index("subject")
    assert bool(scored.loc["p00000000", "agree"]) is True
    assert bool(scored.loc["p00000006", "agree"]) is False
    assert bool(scored.loc["pf", "agree"]) is True
    assert scored.loc[_stem("00000001"), "outcome"] == "missing", "no decision: the recording is its own subject"
    assert not any("eeeeeeee" in str(name) for name in scored.index)
    assert set(scored["group"]) == {"breath"}


def test_the_subject_is_the_participant_column_and_a_split_may_name_it_by_a_prefix() -> None:
    """Review L5: no stem is parsed for a subject; a split entry names the subjects whose label starts with it."""
    labels = _labels().iloc[:1]
    decisions = _decisions(labels).assign(participant=["sub-Subj01-x"])
    subjects = ev.with_subjects(labels, decisions)
    assert list(subjects["subject"]) == ["sub-Subj01-x"]
    row = subjects.iloc[0].copy()
    row["listen_set"] = "set_a"
    assert ev.side(row, {"set_a": {"fit": ["subj01"], "heldout": []}}) == "fit"
    assert ev.side(row, {"set_a": {"fit": ["sub-Subj01-x"], "heldout": []}}) == "fit"
    assert ev.side(row, {"set_a": {"fit": ["subj02"], "heldout": []}}) == "heldout"


def test_a_family_s_group_is_its_kind_in_the_task_list(triage_profile: object) -> None:
    """Review L5: voice, cough and breath are the task list's kinds, not names matched by substring."""
    assert ev.family_group("prolonged-vowel", "set_a") == "voice"
    assert ev.family_group("voluntary-cough", "set_a") == "cough"
    assert ev.family_group("respiration-and-cough-fivebreaths", "set_a") == "breath"
    assert ev.family_group("rainbow-passage", "set_a") == "other"
    assert ev.family_group("not-a-task", "set_a") == "other"


def test_the_split_and_the_label_map_are_required_and_a_profile_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review L5: no fixture default; without a profile the driver exits 2 naming it."""
    labels, decisions = tmp_path / "labels.csv", tmp_path / "rows"
    with pytest.raises(SystemExit) as stopped:
        ev.main([str(labels), str(decisions)])
    assert stopped.value.code == 2
    split, label_map = EVALUATION / "task_events_split.yaml", EVALUATION / "owner_label_map.yaml"
    assert ev.main([str(labels), str(decisions), "--split", str(split), "--label-map", str(label_map)]) == 2
    assert "dataset profile" in capsys.readouterr().err


def test_every_entry_of_the_fixture_split_is_a_string() -> None:
    """Two entries were written unquoted as ``.inf``, which YAML reads as a float."""
    split = yaml.safe_load((EVALUATION / "task_events_split.yaml").read_text())
    assert all(isinstance(entry, str) for sides in split.values() for part in sides.values() for entry in part or [])


def test_the_review_band_rate_is_reported_per_group_over_kept_recordings(triage_profile: object) -> None:
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


def test_a_review_page_export_reads_as_labels_and_scores(tmp_path: Path, triage_profile: object) -> None:
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
    decisions = pd.DataFrame({"stem": [_stem("00000000")], "verdict": ["discard"], "ground_keys": [[]]})
    decisions["key"] = decisions["stem"].map(ev.recording_key)
    decisions = decisions.set_index("key")
    mapped, unmapped = ev.map_labels(ev.with_subjects(labels, decisions), LABEL_MAP)
    assert unmapped == [] and list(mapped["target"]) == ["absent"]
    mapped["side"] = "fit"
    scored = ev.score(mapped, decisions, LABEL_MAP)
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
