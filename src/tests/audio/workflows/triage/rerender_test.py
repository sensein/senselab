"""Behavioural tests for re-rendering a completed run's summary from its store.

REPORT reads the store and writes nothing back, so a renderer fix does not need the graph re-run.
What the store does not carry is its own run id, and the summary's title is built from it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[5] / "scripts" / "triage_rerender.py"
_SPEC = importlib.util.spec_from_file_location("triage_rerender", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
rerender_module = importlib.util.module_from_spec(_SPEC)
sys.modules["triage_rerender"] = rerender_module
_SPEC.loader.exec_module(rerender_module)


class TestFindingTheRuns:
    """A run directory is found by the store it holds, at whatever depth."""

    def test_every_store_under_the_tree_is_one_run(self, tmp_path: Path) -> None:
        """A corpus tree nests by subject and session; the depth is not fixed."""
        for stem in ("sub-a/ses-1/rec_2026/run", "sub-b/ses-2/rec_2027/run"):
            path = tmp_path / stem
            path.mkdir(parents=True)
            (path / "store.jsonl").write_text("")
        found = rerender_module.run_dirs(tmp_path)
        assert [path.name for path in found] == ["run", "run"]
        assert {path.parent.name for path in found} == {"rec_2026", "rec_2027"}

    def test_a_tree_with_no_store_yields_nothing_rather_than_raising(self, tmp_path: Path) -> None:
        """Pointing it at the wrong directory must say so by counting zero, not by a traceback."""
        (tmp_path / "empty").mkdir()
        assert rerender_module.run_dirs(tmp_path) == []


class TestTheRunIdComesFromTheDirectory:
    """The store file does not carry its run id, and the title is built from it."""

    def test_the_run_id_is_the_run_directory_s_parent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without it every summary in a corpus is titled 'read', which is read_jsonl's default."""
        run_dir = tmp_path / "sub-a_ses-1_task-loudness_20260920-030735" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "store.jsonl").write_text("")
        seen: dict[str, Any] = {}

        def _read(path: Path, run_id: str = "read") -> object:
            seen["run_id"] = run_id
            return object()

        def _report(store: object, summary_dir: Path, config: object, *, run_dir: Path) -> dict[str, Path]:
            seen["summary_dir"] = summary_dir
            return {}

        monkeypatch.setattr(rerender_module.ProvStore, "read_jsonl", staticmethod(_read))
        monkeypatch.setattr(rerender_module, "report", _report)
        rerender_module.rerender(run_dir, config=object())
        assert seen["run_id"] == "sub-a_ses-1_task-loudness_20260920-030735"
        assert seen["run_id"] != "read"

    def test_the_summary_goes_beside_the_store_not_under_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where the run itself put it, so a re-render replaces rather than accumulates."""
        run_dir = tmp_path / "rec_2026" / "run"
        run_dir.mkdir(parents=True)
        seen: dict[str, Any] = {}
        monkeypatch.setattr(rerender_module.ProvStore, "read_jsonl", staticmethod(lambda path, run_id="read": object()))
        monkeypatch.setattr(
            rerender_module,
            "report",
            lambda store, summary_dir, config, *, run_dir: seen.update(summary_dir=summary_dir) or {},
        )
        rerender_module.rerender(run_dir, config=object())
        assert seen["summary_dir"] == tmp_path / "rec_2026" / "summary"
