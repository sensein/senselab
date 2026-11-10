"""The corpus driver: one slice of a manifest triaged resumably, and the finished runs listed as the passes' manifest.

``run_triage`` is replaced by a stand-in that writes a run root as the real one lays it out, so these tests cover
the driver's slicing, resume, clean-up and listing, not the graph.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from senselab.audio.workflows.triage.identity import identity_of, row_identity
from senselab.audio.workflows.triage.run import entity_subdir
from tests.audio.workflows.triage.conftest import hint_for

driver = importlib.import_module("senselab.audio.workflows.triage.drivers.triage_corpus")

CONFIG = SimpleNamespace(config_hash="cfg0", profile=None)


@dataclass
class _Outcome:
    state: Any
    error: str | None = None


@dataclass
class _Result:
    run_dir: Path
    store_path: Path
    summary: dict[str, Path]
    released: dict[str, Path] = field(default_factory=dict)
    nodes: dict[str, _Outcome] = field(default_factory=dict)


class _FakeRun:
    """Stands in for ``run_triage``: lays out a run root under the identity's folders and records each call."""

    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail = fail or set()

    def __call__(self, source: Path, out_dir: Path, config: Any, **kwargs: Any) -> _Result:  # noqa: ANN401
        stem = Path(source).stem
        self.calls.append(stem)
        if stem in self.fail:
            raise RuntimeError("the graph raised")
        root = Path(out_dir) / entity_subdir(identity_of(kwargs.get("hint"), stem=stem)) / f"{stem}_20261109-120000"
        run_dir = root / "run"
        (run_dir / "streams").mkdir(parents=True)
        (run_dir / "streams" / "enhanced.flac").write_bytes(b"audio")
        store = run_dir / "store.jsonl"
        store.write_text("{}\n")
        (root / "summary").mkdir()
        product = root / "summary" / "summary.json"
        product.write_text("{}")
        state = SimpleNamespace(value="completed")
        return _Result(run_dir, store, {"json": product}, nodes={"ADMIT": _Outcome(state)})


def _manifest(tmp_path: Path, stems: list[str]) -> Path:
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    lines = []
    for stem in stems:
        wav = audio / f"{stem}.wav"
        wav.write_bytes(b"RIFF")
        lines.append(json.dumps({"source": str(wav)}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def test_a_slice_runs_its_stride_and_a_rerun_skips_what_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows ``[i::n]`` run once; a second attempt at the slice runs nothing."""
    fake = _FakeRun()
    monkeypatch.setattr(driver, "run_triage", fake)
    manifest = _manifest(tmp_path, ["a", "b", "c", "d"])
    out = tmp_path / "out"
    first = driver.run_slice(manifest, out_root=out, slice_index=1, slice_count=2, config=CONFIG)
    assert fake.calls == ["b", "d"]
    assert first["counts"] == {"ok": 2, "failed": 0, "skipped": 0}
    second = driver.run_slice(manifest, out_root=out, slice_index=1, slice_count=2, config=CONFIG)
    assert fake.calls == ["b", "d"]
    assert second["counts"] == {"ok": 0, "failed": 0, "skipped": 2}


def test_runs_land_under_the_hints_participant_and_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a hint builder, a recording's row and run root go under its declared participant and session."""
    monkeypatch.setattr(driver, "run_triage", _FakeRun())
    manifest = _manifest(tmp_path, ["rec1"])
    out = tmp_path / "out"

    def build_hint(path: Path) -> tuple[Any, None]:  # noqa: ANN401
        return hint_for("p07", "s2"), None

    driver.run_slice(manifest, out_root=out, slice_index=0, slice_count=1, config=CONFIG, build_hint=build_hint)
    row = json.loads((out / "rows" / "p07" / "s2" / "rec1.row.json").read_text())
    assert row["ok"] and row["config_hash"] == "cfg0"
    assert Path(row["run_root"]).parent == out / "out" / "p07" / "s2"
    listing = tmp_path / "runs.jsonl"
    driver.write_manifest(out, listing)
    (manifest_row,) = [json.loads(line) for line in listing.read_text().splitlines()]
    assert row_identity(manifest_row) == identity_of(hint_for("p07", "s2"), stem="rec1")
    assert manifest_row["recording_id"] == "rec1"


def test_an_unfinished_attempt_is_removed_before_the_recording_runs_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run root left with no finished row is swept; another recording whose stem extends this one is not."""
    monkeypatch.setattr(driver, "run_triage", _FakeRun())
    manifest = _manifest(tmp_path, ["rec"])
    out = tmp_path / "out"
    stale = out / "out" / "rec_20261108-010203-1"
    stale.mkdir(parents=True)
    neighbour = out / "out" / "rec_more_20261108-010203"
    neighbour.mkdir(parents=True)
    driver.run_slice(manifest, out_root=out, slice_index=0, slice_count=1, config=CONFIG)
    row = json.loads((out / "rows" / "rec.row.json").read_text())
    assert row["removed_attempts"] == ["rec_20261108-010203-1"]
    assert not stale.exists() and neighbour.exists()


def test_a_failed_recording_is_recorded_and_the_slice_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A raise is the recording's row, ``ok`` false; the next recording still runs, and the failed one runs again."""
    fake = _FakeRun(fail={"a"})
    monkeypatch.setattr(driver, "run_triage", fake)
    manifest = _manifest(tmp_path, ["a", "b"])
    out = tmp_path / "out"
    summary = driver.run_slice(manifest, out_root=out, slice_index=0, slice_count=1, config=CONFIG)
    assert summary["counts"] == {"ok": 1, "failed": 1, "skipped": 0} and summary["failed"] == ["a"]
    row = json.loads((out / "rows" / "a.row.json").read_text())
    assert row["ok"] is False and row["error"].startswith("RuntimeError")
    fake.fail.clear()
    driver.run_slice(manifest, out_root=out, slice_index=0, slice_count=1, config=CONFIG)
    assert fake.calls == ["a", "b", "a"]


def test_a_missing_recording_is_a_failed_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest row whose audio is gone fails without running the graph."""
    fake = _FakeRun()
    monkeypatch.setattr(driver, "run_triage", fake)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"source": str(tmp_path / "gone.wav")}) + "\n")
    summary = driver.run_slice(manifest, out_root=tmp_path / "out", slice_index=0, slice_count=1, config=CONFIG)
    assert summary["failed"] == ["gone"] and fake.calls == []


def test_the_manifest_lists_finished_runs_and_reports_the_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``manifest`` writes stem, source and enhanced per finished run, and ``--expect`` names what did not finish."""
    monkeypatch.setattr(driver, "run_triage", _FakeRun(fail={"b"}))
    manifest = _manifest(tmp_path, ["a", "b"])
    out = tmp_path / "out"
    driver.run_slice(manifest, out_root=out, slice_index=0, slice_count=1, config=CONFIG)
    listing = tmp_path / "runs.jsonl"
    assert driver.main(["manifest", str(out), "--out", str(listing), "--expect", str(manifest)]) == 1
    rows = [json.loads(line) for line in listing.read_text().splitlines()]
    assert [row["stem"] for row in rows] == ["a"]
    assert rows[0]["enhanced"].endswith("run/streams/enhanced.flac")
    assert Path(rows[0]["enhanced"]).is_file()
    summary = driver.write_manifest(out, listing, expect=manifest)
    assert summary["missing"] == ["b"]


def test_the_duration_is_read_from_the_header(tmp_path: Path) -> None:
    """The isolation rule reads a recording's length from its header, through senselab's audio reader."""
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "one.wav"
    sf.write(wav, np.zeros(16000 * 3, dtype=np.float32), 16000)
    assert driver._audio_seconds(wav) == pytest.approx(3.0)
    assert driver._audio_seconds(tmp_path / "missing.wav") is None
