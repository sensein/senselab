"""The review rows driver: the replayed recordings whose review must be read again, split by whether they have words."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

rows_driver = importlib.import_module("senselab.audio.workflows.triage.drivers.triage_review_rows")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _row(stem: str, root: Path) -> dict[str, str]:
    return {"stem": stem, "enhanced": str(root / stem / "run" / "streams" / "enhanced.flac"), "source": f"/{stem}.wav"}


def test_rereads_are_the_replayed_rows_marked_needs_reread(tmp_path: Path) -> None:
    """Only ``needs_reread`` rows are picked; every state is counted, a later log line for a stem winning."""
    manifest = _write_jsonl(tmp_path / "replay.jsonl", [_row(stem, tmp_path) for stem in ("a", "b", "c", "d")])
    logs = tmp_path / "logs"
    _write_jsonl(logs / "slices" / "replay-0.jsonl", [{"stem": "a", "review": "none"}, {"stem": "b", "review": "none"}])
    _write_jsonl(logs / "slices" / "replay-1.jsonl", [{"stem": "b", "review": "needs_reread"}, {"stem": "c"}])
    out = tmp_path / "rows_needs_reread.jsonl"
    summary = rows_driver.pick_rereads(manifest, logs, out)
    assert [json.loads(line)["stem"] for line in out.read_text().splitlines()] == ["b"]
    assert summary["states"] == {"needs_reread": 1, "none": 1, "not_in_replay_log": 1, "unrecorded": 1}
    assert json.loads((tmp_path / "rows_needs_reread.counts.json").read_text()) == summary


def test_split_follows_the_reviewers_words_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transcript with words is a words row, a blank one a nowords row, a store that will not open neither."""
    texts = {"talk": "hello there", "quiet": "   ", "broken": None}

    def read_store(run_root: Path) -> str:
        if texts[run_root.name] is None:
            raise OSError("no store")
        return run_root.name

    redact = importlib.import_module("senselab.audio.workflows.triage.nodes.redact")
    monkeypatch.setattr(rows_driver, "read_store", read_store)
    monkeypatch.setattr(redact, "transcript_texts", lambda store: (texts[store], None))
    first = _write_jsonl(tmp_path / "one.jsonl", [_row("talk", tmp_path), _row("quiet", tmp_path)])
    second = _write_jsonl(tmp_path / "two.jsonl", [_row("talk", tmp_path), _row("broken", tmp_path)])
    out = tmp_path / "split"
    counts = rows_driver.split_rows([first, second], out, workers=1)
    assert counts["rows"] == 3 and (counts["words"], counts["nowords"], counts["error"]) == (1, 1, 1)
    assert [json.loads(line)["stem"] for line in (out / "rows_words.jsonl").read_text().splitlines()] == ["talk"]
    assert [json.loads(line)["stem"] for line in (out / "rows_nowords.jsonl").read_text().splitlines()] == ["quiet"]
    assert (out / "errors.txt").read_text() == "broken\terror:OSError\n"
