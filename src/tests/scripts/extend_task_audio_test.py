"""The task-audio driver: a finished corpus cut to each recording's task extent, resumably."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from senselab.audio.workflows.triage.extend import read_store
from tests.audio.workflows.triage.task_audio_test import _seed

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_task_audio.py"
_spec = importlib.util.spec_from_file_location("extend_task_audio_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_task_audio_under_test"] = cli
_spec.loader.exec_module(cli)


def _corpus(tmp_path: Path) -> Path:
    """Two finished runs, one with a task extent and one without, and their manifest."""
    rows = []
    for stem, tasks in (("sub-a_task-reading", ((2.0, 7.5),)), ("sub-b_task-free", ())):
        run_root = tmp_path / stem
        store, run_dir = _seed(run_root, tasks=tasks)
        store.write_jsonl(run_dir / "store.jsonl")
        rows.append({"stem": stem, "enhanced": str(run_dir / "streams" / "enhanced.flac")})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def _log(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "slices" / "task-audio-slice-0-of-1.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_slice_cuts_then_a_resubmission_finds_every_cut_present(tmp_path: Path) -> None:
    """The first pass writes the cuts and the store; the second reads them back unchanged."""
    manifest = _corpus(tmp_path)
    assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
    first = {row["stem"]: row for row in _log(tmp_path)}
    assert first["sub-a_task-reading"]["status"] == "ok"
    assert first["sub-a_task-reading"]["task_plain"] == "written"
    assert first["sub-b_task-free"]["status"] == "absent"
    run_root = tmp_path / "sub-a_task-reading"
    stored = read_store(run_root)
    assert any(e.attributes.get("name") == "task_plain" for e in stored.entities("stream"))
    assert (run_root / "prov").is_dir()
    before = (run_root / "run" / "store.jsonl").read_bytes()

    assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
    second = {row["stem"]: row for row in _log(tmp_path)}
    assert second["sub-a_task-reading"]["status"] == "present"
    assert {second["sub-a_task-reading"][f"task_{n}"] for n in ("plain", "enhanced", "redacted")} == {"present"}
    assert (run_root / "run" / "store.jsonl").read_bytes() == before


def test_a_row_outside_a_run_root_is_an_error(tmp_path: Path) -> None:
    """A manifest path that is not <run_root>/run/streams/<stream> fails the row, and the task exits 1."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"stem": "x", "enhanced": str(tmp_path / "x.flac")}) + "\n")
    assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 1
    assert _log(tmp_path)[0]["status"] == "error"
