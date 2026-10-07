"""Tests for the input-set digest."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from senselab.utils.reproduce.digest import (
    DigestError,
    digest_unchanged,
    input_set_digest,
    pattern_matches,
)


def _tree(root: Path) -> Path:
    (root / "a" / "deep").mkdir(parents=True)
    (root / "a" / "one.csv").write_text("x,y\n1,2\n")
    (root / "a" / "deep" / "two.csv").write_text("x,y\n3,4\n")
    (root / "a" / "notes.txt").write_text("not matched\n")
    (root / "top.json").write_text("{}")
    return root


PATTERNS = ["a/**/*.csv", "top.json"]


def test_identical_trees_digest_identically(tmp_path: Path) -> None:
    """Identical trees digest identically."""
    first = input_set_digest(_tree(tmp_path / "one"), PATTERNS)
    second = input_set_digest(_tree(tmp_path / "two"), PATTERNS)
    assert first == second
    assert first[1] == 3
    assert len(first[0]) == 64


def test_digest_is_stable_across_calls(tmp_path: Path) -> None:
    """Digest is stable across calls."""
    root = _tree(tmp_path / "t")
    assert input_set_digest(root, PATTERNS) == input_set_digest(root, PATTERNS)


def test_one_byte_changes_the_digest(tmp_path: Path) -> None:
    """One byte changes the digest."""
    root = _tree(tmp_path / "t")
    before = input_set_digest(root, PATTERNS)
    (root / "a" / "deep" / "two.csv").write_text("x,y\n3,5\n")
    after = input_set_digest(root, PATTERNS)
    assert after[1] == before[1]
    assert not digest_unchanged(before, after)


def test_rename_changes_the_digest(tmp_path: Path) -> None:
    """Rename changes the digest."""
    root = _tree(tmp_path / "t")
    before = input_set_digest(root, PATTERNS)
    os.rename(root / "a" / "one.csv", root / "a" / "uno.csv")
    assert not digest_unchanged(before, input_set_digest(root, PATTERNS))


def test_added_file_changes_the_digest_and_count(tmp_path: Path) -> None:
    """Added file changes the digest and count."""
    root = _tree(tmp_path / "t")
    before = input_set_digest(root, PATTERNS)
    (root / "a" / "three.csv").write_text("z\n")
    after = input_set_digest(root, PATTERNS)
    assert after[1] == before[1] + 1
    assert not digest_unchanged(before, after)


def test_unmatched_file_does_not_change_the_digest(tmp_path: Path) -> None:
    """Unmatched file does not change the digest."""
    root = _tree(tmp_path / "t")
    before = input_set_digest(root, PATTERNS)
    (root / "a" / "notes.txt").write_text("changed\n")
    (root / "other.bin").write_bytes(b"\0")
    assert digest_unchanged(before, input_set_digest(root, PATTERNS))


def test_exclusion_skips_a_subtree(tmp_path: Path) -> None:
    """Exclusion skips a subtree."""
    root = tmp_path / "out"
    (root / "cuda-cache" / "k").mkdir(parents=True)
    (root / "result.json").write_text("{}")
    (root / "cuda-cache" / "k" / "blob").write_bytes(b"1")
    before = input_set_digest(root, ["**"], exclude=["cuda-cache"])
    assert before[1] == 1
    (root / "cuda-cache" / "k" / "blob").write_bytes(b"2")
    (root / "cuda-cache" / "new").write_bytes(b"3")
    assert digest_unchanged(before, input_set_digest(root, ["**"], exclude=["cuda-cache"]))
    assert input_set_digest(root, ["**"])[1] == 3


def test_exclusion_is_by_path_component(tmp_path: Path) -> None:
    """Exclusion is by path component."""
    root = tmp_path / "out"
    root.mkdir()
    (root / "cuda-cache-not").write_text("kept")
    assert input_set_digest(root, ["**"], exclude=["cuda-cache"])[1] == 1


def test_overlapping_patterns_count_each_file_once(tmp_path: Path) -> None:
    """Overlapping patterns count each file once."""
    root = _tree(tmp_path / "t")
    assert input_set_digest(root, ["a/**", "a/*.csv", "**/*.csv"])[1] == 3


def test_digest_reveals_no_names_and_does_not_depend_on_root(tmp_path: Path) -> None:
    """Digest reveals no names and does not depend on root."""
    one = _tree(tmp_path / "secret-name-one")
    two = tmp_path / "elsewhere"
    shutil.copytree(one, two)
    digest, _ = input_set_digest(one, PATTERNS)
    assert digest == input_set_digest(two, PATTERNS)[0]
    assert "one.csv" not in digest


@pytest.mark.parametrize(
    ("rel", "pattern", "expected"),
    [
        ("a/one.csv", "a/*.csv", True),
        ("a/deep/two.csv", "a/*.csv", False),
        ("a/deep/two.csv", "a/**/*.csv", True),
        ("a/one.csv", "a/**/*.csv", True),
        ("top.json", "**", True),
        ("x/y/z", "x/**", True),
        ("x", "x/**", False),
        ("ab.csv", "a?.csv", True),
        ("a1.csv", "a[0-9].csv", True),
    ],
)
def test_pattern_matches(rel: str, pattern: str, expected: bool) -> None:
    """``*`` stays within a component, ``**`` spans components, and classes and ``?`` work."""
    assert pattern_matches(rel, pattern) is expected


def test_missing_root_raises_a_code_without_the_path(tmp_path: Path) -> None:
    """Missing root raises a code without the path."""
    missing = tmp_path / "private-dir-name"
    with pytest.raises(DigestError) as info:
        input_set_digest(missing, ["**"])
    assert info.value.code == "E_DIGEST_NO_ROOT"
    assert str(info.value) == "E_DIGEST_NO_ROOT"
    assert "private-dir-name" not in str(info.value)
    assert info.value.__cause__ is None


@pytest.mark.parametrize("pattern", ["/abs/*.csv", "../up/*.csv", "a/../../b"])
def test_bad_pattern_is_refused(tmp_path: Path, pattern: str) -> None:
    """Bad pattern is refused."""
    with pytest.raises(DigestError) as info:
        input_set_digest(_tree(tmp_path / "t"), [pattern])
    assert info.value.code == "E_DIGEST_BAD_PATTERN"
    assert pattern not in str(info.value)


def test_unreadable_file_raises_a_code_without_the_path(tmp_path: Path) -> None:
    """Unreadable file raises a code without the path."""
    root = _tree(tmp_path / "t")
    (root / "a" / "dangling-private.csv").symlink_to(root / "nowhere-private")
    with pytest.raises(DigestError) as info:
        input_set_digest(root, PATTERNS)
    assert info.value.code == "E_DIGEST_UNREADABLE"
    assert "private" not in str(info.value)
    assert info.value.__cause__ is None


def test_non_regular_file_is_refused(tmp_path: Path) -> None:
    """Non regular file is refused."""
    root = _tree(tmp_path / "t")
    os.mkfifo(root / "a" / "pipe.csv")
    with pytest.raises(DigestError) as info:
        input_set_digest(root, PATTERNS)
    assert info.value.code == "E_DIGEST_UNREADABLE"


def test_empty_match_is_a_zero_count(tmp_path: Path) -> None:
    """Empty match is a zero count."""
    root = _tree(tmp_path / "t")
    digest, count = input_set_digest(root, ["nothing/*.csv"])
    assert count == 0
    assert len(digest) == 64
