"""`remove_owned`: the one deletion helper removes only a matching direct child, never following a link."""

import itertools
import os
import re
from pathlib import Path

import pytest
from senselab_worker.ownership import ARRAY_NAME, ARRAY_TEMP_NAME, UUID_HEX, remove_owned

KINDS = ["file", "dir", "link_to_file", "link_to_dir", "fifo"]
PLACES = ["child", "grandchild", "sibling"]
PATTERN = re.compile(r"^owned-[0-9]$")


def _make(where: Path, name: str, kind: str, victim: Path) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    path = where / name
    if kind == "file":
        path.write_bytes(b"x")
    elif kind == "dir":
        path.mkdir()
        (path / "inner.bin").write_bytes(b"i")
        os.symlink(victim, path / "to_victim")
    elif kind == "link_to_file":
        os.symlink(victim / "keep.bin", path)
    elif kind == "link_to_dir":
        os.symlink(victim, path)
    else:
        os.mkfifo(path)
    return path


@pytest.mark.logic
@pytest.mark.parametrize(("kind", "place", "matches"), list(itertools.product(KINDS, PLACES, [True, False])))
def test_removes_exactly_a_matching_direct_child(tmp_path: Path, kind: str, place: str, matches: bool) -> None:
    """Removed iff a direct child of `root` with a matching name; a link is removed, its target never touched."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.bin").write_bytes(b"k")
    root = tmp_path / "root"
    root.mkdir()
    where = {"child": root, "grandchild": root / "sub", "sibling": tmp_path / "other"}[place]
    path = _make(where, "owned-1" if matches else "other-1", kind, victim)
    expected = matches and place == "child"
    assert remove_owned(root, path, pattern=PATTERN) is expected
    assert os.path.lexists(path) is not expected
    assert sorted(p.name for p in victim.iterdir()) == ["keep.bin"] and (victim / "keep.bin").read_bytes() == b"k"


@pytest.mark.logic
def test_absence_is_not_an_error(tmp_path: Path) -> None:
    """A missing path, or a missing root, returns False."""
    assert remove_owned(tmp_path, tmp_path / "owned-1", pattern=PATTERN) is False
    assert remove_owned(tmp_path / "gone", tmp_path / "gone" / "owned-1", pattern=PATTERN) is False


@pytest.mark.logic
def test_dot_names_and_the_root_itself_are_refused(tmp_path: Path) -> None:
    """`.`, `..` and the root never count as a child, whatever the pattern."""
    anything = re.compile(r".*")
    for path in (tmp_path, tmp_path / ".", tmp_path / "x" / ".."):
        assert remove_owned(tmp_path, path, pattern=anything) is False
    assert tmp_path.is_dir()


@pytest.mark.logic
@pytest.mark.parametrize(
    ("pattern", "good", "bad"),
    [
        (ARRAY_NAME, "a" * 64 + ".npy", ["a" * 64 + ".npy\n", "A" * 64 + ".npy", "a" * 63 + ".npy", "x.npy"]),
        (ARRAY_TEMP_NAME, "." + "0" * 32 + ".tmp", ["0" * 32 + ".tmp", "." + "0" * 31 + ".tmp"]),
        (UUID_HEX, "0" * 32, ["0" * 31, "0" * 32 + "\n", "g" * 32]),
    ],
)
def test_owned_name_patterns_are_exact(pattern: re.Pattern[str], good: str, bad: list[str]) -> None:
    """Each pattern matches its own names and nothing near them."""
    assert pattern.fullmatch(good)
    assert not any(pattern.fullmatch(b) for b in bad)
