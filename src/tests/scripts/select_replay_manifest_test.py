"""Tests for ``scripts/select_replay_manifest.py``: which recordings the airway replay takes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "select_replay_manifest.py"
_spec = importlib.util.spec_from_file_location("select_replay_manifest", SCRIPT)
assert _spec is not None and _spec.loader is not None
selector = importlib.util.module_from_spec(_spec)
sys.modules["select_replay_manifest"] = selector
_spec.loader.exec_module(selector)


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stem": ["a", "b", "c", "d"],
            "run_dir": ["ra", "rb", "rc", "rd"],
            "declared_family": ["voluntary-cough", "harvard-sentences-list", "respiration-and-cough-breath", None],
            "verdict": ["pass", "review", "review", "review"],
            "run_status": ["complete", "incomplete", "incomplete", "complete"],
            "participant": ["sub-1", "sub-2", None, "sub-4"],
            "session": ["ses-1", "ses-2", None, "ses-4"],
            "task": ["voluntary-cough", "harvard-sentences-list-1", None, "x"],
            "participant_from": ["hint", "stem", None, "stem"],
            "session_from": ["hint", "stem", None, "stem"],
            "task_from": ["hint", "stem", None, "stem"],
        }
    )


def test_airway_families_and_incomplete_runs_are_selected_and_say_why() -> None:
    """Both airway rows, the incomplete speech row, and not the reviewed row of no family."""
    chosen = selector.select(_table(), branch="AIRWAY", include_incomplete=True)
    assert dict(zip(chosen["stem"], chosen["selected_by"])) == {
        "a": "family",
        "b": "incomplete",
        "c": "family+incomplete",
    }


def test_no_incomplete_takes_the_families_alone() -> None:
    """``--no-incomplete`` leaves an incomplete run of another branch's family out."""
    chosen = selector.select(_table(), branch="AIRWAY", include_incomplete=False)
    assert list(chosen["stem"]) == ["a", "c"]


def test_the_manifest_carries_the_enhanced_path_the_drivers_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each line names ``<out-root>/<run_dir>/run/streams/enhanced.flac``."""
    table = tmp_path / "rv.parquet"
    _table().to_parquet(table)
    out = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", str(table), str(out), "--out-root", "/corpus"])
    assert selector.main() == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["enhanced"] for row in rows] == [
        "/corpus/ra/run/streams/enhanced.flac",
        "/corpus/rb/run/streams/enhanced.flac",
        "/corpus/rc/run/streams/enhanced.flac",
    ]


def test_every_manifest_row_carries_the_recorded_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All six identity columns are copied; a run that records none passes through with them null."""
    table = tmp_path / "rv.parquet"
    _table().to_parquet(table)
    out = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", str(table), str(out), "--out-root", "/corpus"])
    assert selector.main() == 0
    rows = {row["stem"]: row for row in map(json.loads, out.read_text().splitlines())}
    assert {name: rows["a"][name] for name in selector.IDENTITY_COLUMNS} == {
        "participant": "sub-1",
        "session": "ses-1",
        "task": "voluntary-cough",
        "participant_from": "hint",
        "session_from": "hint",
        "task_from": "hint",
    }
    assert all(rows["c"][name] is None for name in selector.IDENTITY_COLUMNS)


def test_a_table_without_identity_columns_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corpus table built before it carried identity is refused, and no manifest is written."""
    table = tmp_path / "rv.parquet"
    _table().drop(columns=["participant_from", "session_from", "task_from"]).to_parquet(table)
    out = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", str(table), str(out), "--out-root", "/corpus"])
    assert selector.main() == 2
    assert "rebuild the corpus table" in capsys.readouterr().err
    assert not out.exists()
