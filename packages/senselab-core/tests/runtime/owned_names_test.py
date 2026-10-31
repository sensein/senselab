"""Each runtime call site that deletes refuses a name outside its own grammar."""

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from senselab_core.runtime import fetch as fetch_module
from senselab_core.runtime import storage as storage_module
from senselab_core.runtime.locks import SharedFileLock


def _foreign_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="notes"))


def _failing_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr(os, "replace", refuse)


@pytest.mark.logic
def test_lock_release_refuses_a_name_that_is_not_a_sidecar(tmp_path: Path) -> None:
    """A holder path that is not `<name>.heartbeat` or `<name>.holder` survives the lock's release."""
    foreign = tmp_path / "notes.md"
    lock = SharedFileLock(tmp_path / "resource")
    with lock:
        foreign.write_text("mine")
        lock._holder_path = foreign
    assert foreign.read_text() == "mine"
    assert not (tmp_path / "resource.heartbeat").exists()


@pytest.mark.logic
def test_link_staging_cleanup_refuses_a_name_outside_its_grammar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged symlink whose suffix is not a uuid4 hex is left when the link cannot be placed."""
    blob = tmp_path / "blob"
    blob.write_bytes(b"b")
    link = tmp_path / "voice.wav"
    _foreign_uuid(monkeypatch)
    _failing_replace(monkeypatch)
    with pytest.raises(OSError, match="replace refused"):
        fetch_module._link(link, blob, 0o2770)
    assert (tmp_path / ".voice.wav.notes").is_symlink()


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
