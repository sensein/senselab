"""Each runtime call site that deletes refuses a name outside its own grammar."""

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from planted_objects_fixture import (
    NAME_BOUND_S,
    NAME_OPERATIONS,
    NAME_TYPED,
    OPERATIONS,
    PLANTS,
    PRIMITIVE,
    TYPED,
    bounded,
)
from senselab_core.runtime import fetch as fetch_module
from senselab_core.runtime import storage as storage_module


def _foreign_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="notes"))


def _failing_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr(os, "replace", refuse)


@pytest.mark.logic
def test_link_staging_refuses_a_name_outside_its_grammar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged symlink whose suffix is not a uuid4 hex is never created: the write grammar refuses it."""
    blob = tmp_path / "blob"
    blob.write_bytes(b"b")
    link = tmp_path / "voice.wav"
    _foreign_uuid(monkeypatch)
    with (
        pytest.raises(ValueError, match="not a name this writer owns"),
        storage_module.StorageDir.open_root(tmp_path) as files,
    ):
        fetch_module._link(files, link.name, blob, 0o2770)
    assert not os.path.lexists(tmp_path / ".voice.wav.notes") and not os.path.lexists(link)


@pytest.mark.logic
def test_atomic_write_cleanup_refuses_a_name_outside_its_grammar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A temporary file whose suffix is not a uuid4 hex is left when the write fails."""
    target = tmp_path / "prov.jsonl"
    _foreign_uuid(monkeypatch)
    _failing_replace(monkeypatch)
    with pytest.raises(OSError, match="replace refused"):
        storage_module.write_atomically(target, b"x")
    assert [p.name.endswith(".notes.tmp") for p in tmp_path.iterdir()] == [True]


@pytest.mark.logic
@pytest.mark.parametrize(
    ("grammar", "own", "foreign"),
    [
        (fetch_module.LINK_STAGING_NAME, ".voice.wav." + "0" * 32, ["voice.wav", ".voice.wav.notes", "notes.md"]),
        (
            storage_module.TEMPORARY_NAME,
            ".prov.jsonl.12.34." + "0" * 32 + ".tmp",
            ["prov.jsonl", ".prov.jsonl.12.34.notes.tmp", "notes.md"],
        ),
    ],
)
def test_staging_grammars_accept_their_own_names_only(grammar: object, own: str, foreign: list[str]) -> None:
    """The fixed grammars match what the runtime writes and nothing a caller would name."""
    assert grammar.fullmatch(own)  # type: ignore[attr-defined]
    assert not any(grammar.fullmatch(name) for name in foreign)  # type: ignore[attr-defined]


@pytest.mark.logic
def test_a_removed_tree_never_follows_a_symlink_inside_it(tmp_path: Path) -> None:
    """Removing a uuid-named tree unlinks a symlink in it, never its target."""
    from senselab_worker import ownership

    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep").write_bytes(b"k")
    root = tmp_path / "root"
    tree = root / ("a" * 32)
    (tree / "inner").mkdir(parents=True)
    (tree / "inner" / "f").write_bytes(b"x")
    os.symlink(victim, tree / "inner" / "out")
    with storage_module.StorageDir.open_root(root) as directory:
        assert directory.remove_owned(tree.name, pattern=ownership.UUID_HEX)
    assert list(root.iterdir()) == [] and (victim / "keep").read_bytes() == b"k"


@pytest.mark.contract
@pytest.mark.parametrize("plant", ["symlink", "file"])
@pytest.mark.parametrize("op", OPERATIONS)
def test_no_operation_writes_through_or_wedges_on_a_planted_object_at_any_directory(
    tmp_path: Path, op: str, plant: str
) -> None:
    """At every directory an operation reaches, a planted symlink or file is never written or deleted through.

    The operation either completes correctly or raises a typed, non-retryable error.
    """
    reports = bounded("probe_planted", str(tmp_path), op, plant)
    assert reports, "the operation reached no directory"
    for report in reports:
        assert report["victim_unchanged"], report
        assert report["error"] in TYPED, report
        assert report["error"] or report["right"], report
    if op != "fetch" and plant == "symlink":
        assert all(r["error"] == "" for r in reports), reports


@pytest.mark.contract
@pytest.mark.parametrize("operation", sorted(PRIMITIVE))
def test_every_storage_operation_on_a_planted_symlink_leaves_its_target_alone(tmp_path: Path, operation: str) -> None:
    """Every `StorageDir` operation given a symlink refuses it or acts on the symlink itself.

    The symlink is a directory on the way, the name operated on, or the target; what it points at is never
    written, moved or removed.
    """
    report = bounded("probe_primitive", str(tmp_path), operation)
    assert report["victim_unchanged"], report
    assert report["error"] == PRIMITIVE[operation], report
    if operation in ("write_atomically", "rename_onto", "write_atomically_hard_link", "fs_now_symlink"):
        assert report["value"] == "file"
    if operation == "link_source":
        assert report["value"] == "symlink"


@pytest.mark.contract
@pytest.mark.parametrize("plant", PLANTS)
@pytest.mark.parametrize("op", NAME_OPERATIONS)
def test_no_written_name_is_written_through_or_wedged_by_a_planted_object(tmp_path: Path, op: str, plant: str) -> None:
    """At every name the store, fetch, a lease or a run manifest writes, a planted object changes no victim.

    No victim's bytes, mode or mtime change, nothing is made outside the root, and each outcome is a completed
    operation or a typed error within the bound.
    """
    result = bounded("probe_planted_names", str(tmp_path), op, plant)
    assert result["reports"], "the operation wrote no name"
    for report in result["reports"]:
        assert report["victim_unchanged"], report
        assert report["outside"] == ["root", "victim"], report
        assert report["error"] in NAME_TYPED, report
        assert report["error"] or report["right"], report
        assert report["seconds"] < NAME_BOUND_S, report
