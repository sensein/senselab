"""The two guards on the extract path: a manifest that resolves nothing, and an extract that reads it.

Both come from one incident. Pointed at a stale output directory, the run reused a manifest carrying
paths from before the corpus was relocated, every store resolved to ``missing``, the extract wrote 0
features out of 62,550 rows and the job exited 0. ``--expect`` checks the manifest's row count,
which was right; what was wrong was what the rows pointed at.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "analyze_routing_evidence.py"
_spec = importlib.util.spec_from_file_location("analyze_routing_evidence_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["analyze_routing_evidence_under_test"] = cli
_spec.loader.exec_module(cli)


def _manifest(out_dir: Path, stores: list[Path]) -> Path:
    """Write a manifest naming one store per recording.

    Args:
        out_dir: The analysis out dir the manifest belongs in.
        stores: Where each recording's store is said to be.

    Returns:
        The manifest's path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cli.MANIFEST
    path.write_text(
        "".join(
            json.dumps(
                {
                    "stem": f"sub-{index:02d}",
                    "run_root": str(store.parent.parent),
                    "store": str(store),
                    "task_id": "1",
                    "family": "harvard",
                }
            )
            + "\n"
            for index, store in enumerate(stores)
        ),
        encoding="utf-8",
    )
    return path


def _store(root: Path) -> Path:
    """Create an empty store file under a run root, so its path resolves.

    Args:
        root: The run root.

    Returns:
        The store's path.
    """
    store = root / "run" / "store.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("", encoding="utf-8")
    return store


class TestAReusedManifest:
    """A manifest is a resolution of one tree; reusing it is only sound if it still resolves."""

    def test_a_manifest_whose_stores_are_all_gone_is_refused(self, tmp_path: Path) -> None:
        """This is the relocated corpus: every row intact, every path pointing at nothing."""
        out_dir = tmp_path / "analysis"
        manifest = _manifest(
            out_dir, [tmp_path / "moved" / f"sub-{index:02d}" / "run" / "store.jsonl" for index in range(200)]
        )

        with pytest.raises(FileNotFoundError, match="none of the"):
            cli.build_manifest(tmp_path / "runs", out_dir)
        assert manifest.exists()

    def test_a_manifest_that_still_resolves_is_reused(self, tmp_path: Path) -> None:
        """The guard must not cost a legitimate resume its manifest."""
        out_dir = tmp_path / "analysis"
        stores = [_store(tmp_path / "runs" / f"sub-{index:02d}") for index in range(4)]
        manifest = _manifest(out_dir, stores)

        assert cli.build_manifest(tmp_path / "runs", out_dir) == manifest

    def test_a_few_missing_recordings_do_not_refuse_the_manifest(self, tmp_path: Path) -> None:
        """A recording deleted since the manifest was built is the extract's ``missing``, not a refusal."""
        out_dir = tmp_path / "analysis"
        stores = [_store(tmp_path / "runs" / f"sub-{index:02d}") for index in range(4)]
        stores[0].unlink()
        _manifest(out_dir, stores)

        assert cli.build_manifest(tmp_path / "runs", out_dir).exists()

    def test_an_empty_manifest_is_not_refused_by_the_probe(self, tmp_path: Path) -> None:
        """There is nothing to probe; the row count is what ``--expect`` is for."""
        out_dir = tmp_path / "analysis"
        _manifest(out_dir, [])

        assert cli.build_manifest(tmp_path / "runs", out_dir).exists()


class TestAnExtractThatReadsNothing:
    """Zero features from a manifest with rows in it is a failed run, whatever the exit code was."""

    def test_a_zero_feature_extract_fails_rather_than_exiting_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The only signal was a downstream scorer reporting zero recordings, one job later.

        The extract itself is stood in for: every store it would read is missing, which is the
        incident, and reading them is what the shard directory being empty already says.
        """
        run_dir = tmp_path / "runs"
        summary_dir = run_dir / "sub-00"
        summary_dir.mkdir(parents=True)
        (summary_dir / "sub-00_task-1.summary.json").write_text(
            json.dumps({"stem": "sub-00_task-1", "run_root": str(summary_dir / "sub-00_task-1_20260912-000000")}),
            encoding="utf-8",
        )
        shard_dir = tmp_path / "analysis" / cli.SHARD_DIR

        def _extracted_nothing(*_: object, **__: object) -> Path:
            shard_dir.mkdir(parents=True, exist_ok=True)
            return shard_dir

        monkeypatch.setattr(cli, "extract_all", _extracted_nothing)

        with pytest.raises(SystemExit, match="extracted 0 features"):
            cli.main([str(run_dir), str(tmp_path / "analysis"), "--workers", "1"])
