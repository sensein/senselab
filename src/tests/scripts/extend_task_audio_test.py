"""The task-audio driver: a finished corpus cut to each recording's task extent, resumably."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from senselab.audio.workflows.triage.extend import read_store
from tests.audio.workflows.triage.task_audio_test import _seed

pytestmark = pytest.mark.usefixtures("triage_profile")
"""The drivers' internals these tests call read the fixture profile's data; a driver run is given ``--profile``."""

cli = importlib.import_module("senselab.audio.workflows.triage.drivers.extend_task_audio")


def _corpus(tmp_path: Path) -> Path:
    """Two finished runs, one with a task extent and one without, and their manifest."""
    rows = []
    for stem, tasks in (("rec-0001", ((2.0, 7.5),)), ("rec-0002", ())):
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
    assert first["rec-0001"]["status"] == "ok"
    assert first["rec-0001"]["task_plain"] == "written"
    assert first["rec-0002"]["status"] == "absent"
    run_root = tmp_path / "rec-0001"
    stored = read_store(run_root)
    assert any(e.attributes.get("name") == "task_plain" for e in stored.entities("stream"))
    assert (run_root / "prov").is_dir()
    before = (run_root / "run" / "store.jsonl").read_bytes()

    assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
    second = {row["stem"]: row for row in _log(tmp_path)}
    assert second["rec-0001"]["status"] == "present"
    assert {second["rec-0001"][f"task_{n}"] for n in ("plain", "enhanced", "redacted")} == {"present"}
    assert (run_root / "run" / "store.jsonl").read_bytes() == before


def test_a_row_outside_a_run_root_is_an_error(tmp_path: Path) -> None:
    """A manifest path that is not <run_root>/run/streams/<stream> fails the row, and the task exits 1."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"stem": "x", "enhanced": str(tmp_path / "x.flac")}) + "\n")
    assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 1
    assert _log(tmp_path)[0]["status"] == "error"
