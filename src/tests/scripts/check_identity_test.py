"""The identity-check driver, end to end: a recording list with its hints before a run, a corpus table after."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from senselab.audio.workflows.triage.identity import FROM_HINT, FROM_STEM
from senselab.audio.workflows.triage.identity_check import (
    DUPLICATE_IDENTITY,
    DUPLICATE_STEM,
    NO_RECORDED_IDENTITY,
    SESSION_UNDER_TWO_PARTICIPANTS,
    STEM_DISAGREES,
    UNKNOWN_TASK_FAMILY,
    UNREADABLE_IDENTITY,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "check_identity.py"
_spec = importlib.util.spec_from_file_location("check_identity_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["check_identity_under_test"] = cli
_spec.loader.exec_module(cli)


def _hints(tmp_path: Path, body: str) -> Path:
    """A ``hints.py`` whose ``build_hint`` runs ``body`` over the recording's ``path``."""
    directory = tmp_path / "scope"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hints.py").write_text(
        "from senselab.audio.data_structures import AudioHints\n\n\ndef build_hint(path):\n" + body, encoding="utf-8"
    )
    return directory


def _recordings(tmp_path: Path, lines: list[str]) -> Path:
    """A recording list; the files need not exist, since no recording is opened."""
    path = tmp_path / "recordings.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _table(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """A corpus table carrying the identity columns, plus one the check must not need."""
    columns = ("stem", "run_dir", "participant", "session", "task", "participant_from", "session_from", "task_from")
    data = {name: [row.get(name) for row in rows] for name in columns}
    data["verdict"] = ["release"] * len(rows)
    path = tmp_path / "recording_vectors.parquet"
    pq.write_table(pa.table({name: pa.array(values, type=pa.string()) for name, values in data.items()}), path)
    return path


def _row(stem: str, run_dir: str, participant: str | None, session: str | None, task: str | None) -> dict[str, Any]:
    """A corpus-table row whose identity was read off the stem."""
    named = participant is not None or session is not None or task is not None
    source = FROM_STEM if named else None
    return {
        "stem": stem,
        "run_dir": run_dir,
        "participant": participant,
        "session": session,
        "task": task,
        "participant_from": source,
        "session_from": source,
        "task_from": source,
    }


def _report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class TestBeforeARun:
    """``--recordings``: the identity ADMIT would record, from the stem and the hint."""

    def test_a_clean_list_passes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Distinct BIDS stems of known tasks, no hints: exit 0, every count zero."""
        listing = _recordings(
            tmp_path,
            [
                "# a comment",
                "",
                "/c/sub-1_ses-1_task-rainbow-passage.wav",
                "/c/sub-1_ses-1_task-harvard-sentences-list-1.wav",
            ],
        )
        out = tmp_path / "report.json"
        assert cli.main(["--recordings", str(listing), "--out", str(out)]) == 0
        report = _report(out)
        assert report["mode"] == "recordings"
        assert report["recordings"] == 2
        assert report["session_scope"] == "participant"
        assert not any(report["counts"].values())
        assert os.stat(out).st_mode & 0o777 == 0o600
        assert "Passed" in capsys.readouterr().out

    def test_a_hint_declaring_one_identity_for_two_recordings_is_refused(self, tmp_path: Path) -> None:
        """Every recording is declared ``sub-x``/``ses-x``/``rainbow-passage``: one key, two recordings."""
        hints = _hints(
            tmp_path,
            "    return (AudioHints(metadata={'participant_id': 'sub-x', 'session_id': 'ses-x',"
            " 'task_token': 'rainbow-passage'}),)\n",
        )
        listing = _recordings(
            tmp_path, ["/c/sub-1_ses-1_task-rainbow-passage.wav", "/c/sub-2_ses-1_task-rainbow-passage.wav"]
        )
        out = tmp_path / "report.json"
        assert cli.main(["--recordings", str(listing), "--hints", str(hints), "--out", str(out)]) == 1
        report = _report(out)
        assert report["refused"] is True
        assert report["counts"][DUPLICATE_IDENTITY] == 1
        assert report["counts"][STEM_DISAGREES] == 2, "the declared identity disagrees with each stem"
        (duplicate,) = [each for each in report["findings"] if each["check"] == DUPLICATE_IDENTITY]
        assert duplicate["key"] == "sub-x/ses-x/rainbow-passage"
        assert report["hints"] == str(hints)

    def test_session_scope_decides_whether_a_shared_session_label_is_reported(self, tmp_path: Path) -> None:
        """``ses-1`` under two participants is two sessions by default; under ``global`` it is reported, not refused."""
        listing = _recordings(
            tmp_path, ["/c/sub-1_ses-1_task-rainbow-passage.wav", "/c/sub-2_ses-1_task-rainbow-passage.wav"]
        )
        out = tmp_path / "report.json"
        assert cli.main(["--recordings", str(listing)]) == 0
        assert cli.main(["--recordings", str(listing), "--session-scope", "global", "--out", str(out)]) == 0
        assert _report(out)["counts"][SESSION_UNDER_TWO_PARTICIPANTS] == 1

    def test_json_rows_unnamed_files_and_a_failing_hint(self, tmp_path: Path) -> None:
        """A JSONL row's ``source`` is read; unnamed stems do not collide; a hint that raises is refused."""
        hints = _hints(
            tmp_path,
            "    if path.stem == 'broken':\n        raise OSError('no sidecar')\n    return (None,)\n",
        )
        listing = _recordings(
            tmp_path,
            [json.dumps({"source": "/a/clip_one.wav"}), "/a/clip_two.wav", "/b/clip_one.wav", "/a/broken.wav"],
        )
        out = tmp_path / "report.json"
        assert cli.main(["--recordings", str(listing), "--hints", str(hints), "--out", str(out)]) == 1
        counts = _report(out)["counts"]
        assert counts[UNREADABLE_IDENTITY] == 1
        assert counts[DUPLICATE_STEM] == 1, "clip_one is in /a and /b"
        assert counts[DUPLICATE_IDENTITY] == 1, "unnamed recordings are keyed by stem, and clip_one is twice"
        assert counts[UNKNOWN_TASK_FAMILY] == 1, "no stem names a task"

    def test_hints_with_a_table_are_refused(self, tmp_path: Path) -> None:
        """A finished run's identity is the one it recorded, so no hint applies."""
        table = _table(tmp_path, [_row("a", "a", None, None, None)])
        with pytest.raises(SystemExit) as raised:
            cli.main(["--table", str(table), "--hints", str(tmp_path)])
        assert raised.value.code == 2

    def test_a_missing_list_is_exit_2(self, tmp_path: Path) -> None:
        """Nothing is checked when the input is not there."""
        assert cli.main(["--recordings", str(tmp_path / "absent.txt")]) == 2


class TestAfterARun:
    """``--table``: each run's recorded identity, from the corpus table."""

    def test_a_clean_table_passes_and_a_run_without_identity_is_reported(self, tmp_path: Path) -> None:
        """A row whose ``*_from`` columns are empty records no identity; that is listed, not refused."""
        table = _table(
            tmp_path,
            [
                _row("sub-1_ses-1_task-rainbow-passage", "sub-1/ses-1/r1", "sub-1", "ses-1", "rainbow-passage"),
                _row("sub-1_ses-1_task-humming", "sub-1/ses-1/r2", None, None, None),
            ],
        )
        out = tmp_path / "report.json"
        assert cli.main(["--table", str(table), "--out", str(out)]) == 0
        report = _report(out)
        assert report["mode"] == "table"
        assert report["counts"][NO_RECORDED_IDENTITY] == 1
        (finding,) = report["findings"]
        assert finding["locations"] == ["sub-1/ses-1/r2"]

    def test_a_duplicated_identity_is_exit_1_and_a_repeated_stem_only_reported(self, tmp_path: Path) -> None:
        """Two runs of one identity are refused; one stem under two run dirs is listed."""
        stem = "sub-1_ses-1_task-rainbow-passage"
        table = _table(
            tmp_path,
            [
                _row(stem, "sub-1/ses-1/a", "sub-1", "ses-1", "rainbow-passage"),
                _row(stem, "sub-1/ses-1/b", "sub-1", "ses-1", "rainbow-passage"),
            ],
        )
        out = tmp_path / "report.json"
        assert cli.main(["--table", str(table), "--out", str(out)]) == 1
        counts = _report(out)["counts"]
        assert counts[DUPLICATE_IDENTITY] == 1
        assert counts[DUPLICATE_STEM] == 1

    def test_a_declared_identity_that_differs_from_the_stem_is_reported(self, tmp_path: Path) -> None:
        """A hint-declared participant over a BIDS stem naming another one."""
        row = _row("sub-1_ses-1_task-rainbow-passage", "r", "sub-7", "ses-1", "rainbow-passage")
        row["participant_from"] = FROM_HINT
        out = tmp_path / "report.json"
        assert cli.main(["--table", str(_table(tmp_path, [row])), "--out", str(out)]) == 0
        (finding,) = _report(out)["findings"]
        assert finding["check"] == STEM_DISAGREES
        assert finding["detail"]["fields"] == {"participant": {"declared": "sub-7", "stem": "sub-1"}}

    def test_a_table_without_identity_columns_is_exit_2(self, tmp_path: Path) -> None:
        """A corpus table from before identity columns cannot be checked; rebuild it."""
        path = tmp_path / "old.parquet"
        pq.write_table(pa.table({"stem": ["a"], "run_dir": ["a"]}), path)
        assert cli.main(["--table", str(path)]) == 2
