"""The threaded walk: same tree as os.walk, symlink and top-level filters, errors, early close."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from senselab.utils import fastio
from senselab.utils.fastio import WalkError, find, map_files, ordered_map, walk


def _tree(root: Path) -> Path:
    """A small tree with a file symlink, a directory symlink and an out-of-tree target."""
    (root / "tree" / "sub-a" / "ses-1" / "run").mkdir(parents=True)
    (root / "tree" / "sub-b" / "ses-1").mkdir(parents=True)
    (root / "tree" / "other").mkdir()
    (root / "outside" / "deep").mkdir(parents=True)
    (root / "outside" / "deep" / "store.jsonl").write_text("{}\n")
    (root / "tree" / "sub-a" / "ses-1" / "run" / "store.jsonl").write_text("{}\n")
    (root / "tree" / "sub-b" / "ses-1" / "notes.txt").write_text("x")
    (root / "tree" / "other" / "store.jsonl").write_text("{}\n")
    (root / "tree" / "top.json").write_text("{}")
    (root / "tree" / "sub-b" / "linked").symlink_to(root / "outside")
    (root / "tree" / "sub-b" / "ses-1" / "alias.txt").symlink_to(root / "tree" / "top.json")
    return root / "tree"


def _as_set(rows: list[tuple[str, list[str], list[str]]]) -> set[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    return {(path, tuple(dirs), tuple(files)) for path, dirs, files in rows}


@pytest.mark.parametrize("follow", [False, True])
def test_walk_matches_os_walk(tmp_path: Path, follow: bool) -> None:
    """Every (path, dirs, files) os.walk yields, and nothing else, for either symlink mode."""
    top = _tree(tmp_path)
    expected = _as_set([(p, sorted(d), sorted(f)) for p, d, f in os.walk(top, followlinks=follow)])
    assert _as_set(list(walk(top, threads=4, follow_symlinks=follow))) == expected


def test_follow_reaches_through_a_directory_symlink(tmp_path: Path) -> None:
    """The store under the linked directory is found only when links are followed."""
    top = _tree(tmp_path)
    linked = top / "sub-b" / "linked" / "deep" / "store.jsonl"
    assert linked not in find(top, "store.jsonl", threads=3)
    assert linked in find(top, "store.jsonl", threads=3, follow_symlinks=True)


def test_keep_filters_only_the_top_level(tmp_path: Path) -> None:
    """The keep predicate drops names directly under top; deeper names are untouched."""
    top = _tree(tmp_path)
    paths = {p for p, _, _ in walk(top, threads=2, keep=lambda name: name.startswith("sub-"))}
    assert str(top / "other") not in paths
    assert str(top / "sub-a" / "ses-1" / "run") in paths
    top_files = next(
        files for path, _, files in walk(top, keep=lambda name: name.startswith("sub-")) if path == str(top)
    )
    assert top_files == []


def test_map_files_runs_fn_on_matches(tmp_path: Path) -> None:
    """The function sees each matching file once; the pattern is on the name."""
    top = _tree(tmp_path)
    sizes = sorted(map_files(top, os.path.getsize, threads=4, pattern="store.jsonl"))
    assert sizes == [3, 3]
    assert find(top, "store.jsonl") == sorted(
        [top / "other" / "store.jsonl", top / "sub-a" / "ses-1" / "run" / "store.jsonl"]
    )


def test_fn_exception_reaches_the_consumer(tmp_path: Path) -> None:
    """An exception from fn stops the walk and is raised to the caller, with no thread left behind."""
    top = _tree(tmp_path)
    before = threading.active_count()

    def boom(path: str) -> None:
        raise RuntimeError(f"cannot read {path}")

    with pytest.raises(RuntimeError, match="cannot read"):
        list(map_files(top, boom, threads=4, pattern="store.jsonl"))
    assert threading.active_count() == before


def test_listing_errors_raise_after_the_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory that cannot be listed is reported, the rest is still walked."""
    top = _tree(tmp_path)
    bad = str(top / "other")
    real = os.scandir

    def scandir(path: str) -> object:
        if os.fspath(path) == bad:
            raise PermissionError(13, "denied", path)
        return real(path)

    monkeypatch.setattr(fastio.os, "scandir", scandir)
    seen: list[str] = []
    with pytest.raises(WalkError) as caught:
        for path, _, _ in walk(top, threads=3):
            seen.append(path)
    assert [path for path, _ in caught.value.errors] == [bad]
    assert str(top / "sub-a" / "ses-1" / "run") in seen


def test_on_error_receives_errors_instead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With on_error the walk completes without raising and the callback sees the failure."""
    top = _tree(tmp_path)
    bad = str(top / "other")
    real = os.scandir

    def scandir(path: str) -> object:
        if os.fspath(path) == bad:
            raise PermissionError(13, "denied", path)
        return real(path)

    monkeypatch.setattr(fastio.os, "scandir", scandir)
    errors: list[str] = []
    list(walk(top, threads=3, on_error=lambda path, error: errors.append(path)))
    assert errors == [bad]


def test_early_close_stops_every_worker(tmp_path: Path) -> None:
    """Closing the generator after one item joins the whole pool."""
    for i in range(30):
        (tmp_path / f"d{i}" / "x").mkdir(parents=True)
    before = threading.active_count()
    walker = walk(tmp_path, threads=8)
    next(walker)
    walker.close()
    assert threading.active_count() == before


def test_missing_top_yields_nothing(tmp_path: Path) -> None:
    """A top that does not exist is an empty walk, as with os.walk."""
    assert list(walk(tmp_path / "absent")) == []


def test_threads_must_be_positive(tmp_path: Path) -> None:
    """A pool of zero threads is refused."""
    with pytest.raises(ValueError, match="at least 1"):
        list(walk(tmp_path, threads=0))


def test_ordered_map_keeps_input_order() -> None:
    """Results come back in the order of the inputs whatever order the threads finish in."""
    import time

    def slow_for_small(n: int) -> int:
        time.sleep(0.001 * (20 - n))
        return n * n

    assert list(ordered_map(slow_for_small, range(20), threads=8)) == [n * n for n in range(20)]


def test_ordered_map_reraises_and_leaves_no_thread() -> None:
    """An exception from fn reaches the caller and the pool is shut down."""
    before = threading.active_count()

    def fail_on_three(n: int) -> int:
        if n == 3:
            raise KeyError(n)
        return n

    with pytest.raises(KeyError):
        list(ordered_map(fail_on_three, range(10), threads=4))
    assert threading.active_count() == before


def test_links_as_files_hands_an_unfollowed_directory_symlink_to_fn(tmp_path: Path) -> None:
    """With links_as_files, the directory symlink reaches fn and is not descended into."""
    top = _tree(tmp_path)
    linked = str(top / "sub-b" / "linked")
    plain = set(map_files(top, str, threads=3))
    assert linked not in plain
    seen = set(map_files(top, str, threads=3, links_as_files=True))
    assert linked in seen
    assert str(top / "sub-b" / "linked" / "deep" / "store.jsonl") not in seen
    followed = set(map_files(top, str, threads=3, follow_symlinks=True, links_as_files=True))
    assert linked not in followed
