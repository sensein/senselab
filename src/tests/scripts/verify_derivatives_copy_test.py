"""The derivatives-copy verifier: symlinks followed in the source, top-level files ignored in the copy."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "verify_derivatives_copy.py"
_spec = importlib.util.spec_from_file_location("verify_derivatives_copy_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["verify_derivatives_copy_under_test"] = cli
_spec.loader.exec_module(cli)

NEEDLE = "/orcd/scratch"


def _trees(root: Path) -> tuple[Path, Path]:
    """A source whose store is a symlink into an earlier tree, and a faithful copy of it.

    The source's run root holds a real file and a symlinked store whose target mentions the needle.
    The copy holds both as regular files plus a top-level file that the comparison must ignore.
    """
    earlier = root / "earlier"
    earlier.mkdir()
    (earlier / "store.jsonl").write_text(json.dumps({"path": f"{NEEDLE}/x"}) + "\n")
    run = root / "src" / "sub-a" / "ses-1" / "run"
    run.mkdir(parents=True)
    (run / "store.jsonl").symlink_to(earlier / "store.jsonl")
    (run / "notes.txt").write_text("hello\n")
    (root / "src" / "sub-b").mkdir()
    (root / "src" / "sub-b" / "store.jsonl").write_text("{}\n")
    dst = root / "dst"
    shutil.copytree(root / "src", dst, symlinks=False)
    (dst / "dataset_description.json").write_text("{}")
    return root / "src", dst


def test_a_faithful_copy_matches_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Same files and bytes, no symlink in the copy, the needle counted once, exit 0."""
    src, dst = _trees(tmp_path)
    out = tmp_path / "verify.json"
    assert cli.main(["--src", str(src), "--dst", str(dst), "--threads", "3", "--json", str(out)]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.endswith("| MATCH")
    assert "symlinks in dst=0" in line
    assert f"stores mentioning {NEEDLE}: 1 of 2" in line
    report = json.loads(out.read_text())
    assert report["match"] is True
    assert report["src"]["files"] == report["dst"]["files"] == 3
    assert report["src"]["symlinks"] == 1
    assert report["src"]["bytes"] == report["dst"]["bytes"]


def test_the_top_level_file_beside_the_copy_is_not_counted(tmp_path: Path) -> None:
    """Only sub-* entries under the copy are counted, so a top-level file changes nothing."""
    src, dst = _trees(tmp_path)
    without = cli.census(dst, threads=2, follow=False, needle=NEEDLE, only_prefix="sub-")
    (dst / "extra.json").write_text("{" * 100)
    again = cli.census(dst, threads=2, follow=False, needle=NEEDLE, only_prefix="sub-")
    assert (without.files, without.bytes) == (again.files, again.bytes)


def test_a_symlink_in_the_copy_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A symlink left in the copy fails the check even when the counts still match."""
    src, dst = _trees(tmp_path)
    (dst / "sub-b" / "dangling").symlink_to(tmp_path / "nowhere")
    assert cli.main(["--src", str(src), "--dst", str(dst), "--threads", "2"]) == 1
    assert "symlinks in dst=1" in capsys.readouterr().out


def test_a_directory_symlink_in_the_copy_is_counted(tmp_path: Path) -> None:
    """An unfollowed symlink to a directory is counted, not silently skipped."""
    src, dst = _trees(tmp_path)
    (dst / "sub-b" / "linkdir").symlink_to(tmp_path / "earlier")
    tally = cli.census(dst, threads=2, follow=False, needle=NEEDLE, only_prefix="sub-")
    assert tally.symlinks == 1


def test_a_missing_file_is_a_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A file absent from the copy changes the counts and the exit status."""
    src, dst = _trees(tmp_path)
    (dst / "sub-a" / "ses-1" / "run" / "notes.txt").unlink()
    assert cli.main(["--src", str(src), "--dst", str(dst), "--threads", "2"]) == 1
    assert capsys.readouterr().out.strip().endswith("| MISMATCH")
