"""The store's file checksums: what they record, and what they say when they cannot."""

import hashlib
import os
from pathlib import Path
from typing import get_args

import pytest

from senselab.utils.prov_store import (
    CHECKSUM_KEY,
    CHECKSUM_UNRESOLVED_KEY,
    MTIME_KEY,
    PROV_TYPE,
    SIZE_KEY,
    ProvStore,
    file_attributes,
    file_digest,
)


def test_digest_matches_hashlib(tmp_path: Path) -> None:
    """The recorded digest is the file's SHA-256."""
    target = tmp_path / "a.bin"
    payload = os.urandom(1 << 21)
    target.write_bytes(payload)
    digest, reason = file_digest(target)
    assert reason is None
    assert digest == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_digest_says_why_it_is_unknown(tmp_path: Path, kind: str) -> None:
    """An undigestable path records a reason, not None and not an exception."""
    target = tmp_path / "gone" if kind == "missing" else tmp_path
    digest, reason = file_digest(target)
    assert digest is None
    assert reason and reason.strip()


def test_attributes_carry_size_and_mtime(tmp_path: Path) -> None:
    """A digest is accompanied by the size and mtime, so a swap is detectable without it."""
    target = tmp_path / "a.bin"
    target.write_bytes(b"x" * 17)
    attributes = file_attributes(target)
    assert attributes[SIZE_KEY] == 17
    assert attributes[MTIME_KEY] == target.stat().st_mtime_ns
    assert CHECKSUM_UNRESOLVED_KEY not in attributes


def test_attributes_of_a_missing_file_carry_the_reason_alone(tmp_path: Path) -> None:
    """Nothing is invented for a file that is not there."""
    attributes = file_attributes(tmp_path / "gone")
    assert attributes == {CHECKSUM_UNRESOLVED_KEY: "file not found"}


def test_attributes_accept_a_digest_taken_earlier(tmp_path: Path) -> None:
    """A caller that already read the file does not read it again."""
    target = tmp_path / "a.bin"
    target.write_bytes(b"y")
    digest = hashlib.sha256(b"y").hexdigest()
    assert file_attributes(target, digest=digest)[CHECKSUM_KEY] == digest


def test_attributes_refuse_a_digest_and_a_reason_together(tmp_path: Path) -> None:
    """The two contradict each other."""
    with pytest.raises(ValueError, match="contradict"):
        file_attributes(tmp_path / "a.bin", digest="0" * 64, reason="file not found")


@pytest.mark.parametrize(
    ("attributes", "match"),
    [
        ({"path": "a", CHECKSUM_KEY: "0" * 64, CHECKSUM_UNRESOLVED_KEY: "gone"}, "contradict"),
        ({"path": "a", CHECKSUM_KEY: "0" * 63}, "64 lowercase hex"),
        ({"path": "a", CHECKSUM_KEY: "A" * 64}, "64 lowercase hex"),
        ({"path": "a", CHECKSUM_UNRESOLVED_KEY: "  "}, "must not be empty"),
        ({CHECKSUM_KEY: "0" * 64}, "naming the file"),
    ],
)
def test_entity_refuses_a_dishonest_checksum(attributes: dict[str, object], match: str) -> None:
    """The store refuses at write time what it would refuse at read-back."""
    store = ProvStore(run_id="r")
    with pytest.raises(ValueError, match=match):
        store.entity(prov_type="stream", extent=None, attributes=attributes)


@pytest.mark.parametrize(
    "attributes", [{CHECKSUM_KEY: "0" * 64}, {CHECKSUM_UNRESOLVED_KEY: "input set is empty"}], ids=["digest", "reason"]
)
def test_an_artifact_checksum_needs_no_path(tmp_path: Path, attributes: dict[str, object]) -> None:
    """An artifact is named by its content alone, and reads back unchanged."""
    store = ProvStore(run_id="r")
    eid = store.entity(prov_type="artifact", extent=None, attributes={"role": "input", **attributes})
    path = tmp_path / "store.jsonl"
    store.write_jsonl(path)
    assert ProvStore.read_jsonl(path).get_entity(eid).attributes == {"role": "input", **attributes}


@pytest.mark.parametrize("prov_type", sorted(set(get_args(PROV_TYPE)) - {"artifact"}))
def test_every_other_type_still_needs_a_path_for_its_checksum(prov_type: PROV_TYPE) -> None:
    """Only an artifact is exempt; any other entity's checksum must still say which file it describes."""
    store = ProvStore(run_id="r")
    with pytest.raises(ValueError, match="naming the file"):
        store.entity(prov_type=prov_type, extent=None, attributes={CHECKSUM_KEY: "0" * 64})


def test_read_back_holds_a_written_store_to_the_same_checksum_invariants(tmp_path: Path) -> None:
    """A hand-edited store.jsonl carrying a malformed digest is refused, naming the entity."""
    store = ProvStore(run_id="r")
    store.entity(prov_type="stream", extent=None, attributes={"path": "a", CHECKSUM_KEY: "0" * 64})
    path = tmp_path / "store.jsonl"
    store.write_jsonl(path)
    path.write_text(path.read_text().replace("0" * 64, "0" * 63))
    with pytest.raises(ValueError, match="64 lowercase hex"):
        ProvStore.read_jsonl(path)


def test_a_store_written_without_checksums_still_reads_back(tmp_path: Path) -> None:
    """Runs recorded before checksums existed convert; they record no digest rather than a wrong one."""
    store = ProvStore(run_id="r")
    store.entity(prov_type="stream", extent=None, attributes={"path": "streams/plain.flac"})
    path = tmp_path / "store.jsonl"
    store.write_jsonl(path)
    [entity] = ProvStore.read_jsonl(path).entities()
    assert CHECKSUM_KEY not in entity.attributes
    assert CHECKSUM_UNRESOLVED_KEY not in entity.attributes


def test_entity_identity_follows_content(tmp_path: Path) -> None:
    """Two files at the same path with different bytes are different entities."""
    store = ProvStore(run_id="r")
    target = tmp_path / "a.bin"
    target.write_bytes(b"one")
    first = store.entity(prov_type="stream", extent=None, attributes={"path": "a.bin", **file_attributes(target)})
    target.write_bytes(b"two")
    second = store.entity(prov_type="stream", extent=None, attributes={"path": "a.bin", **file_attributes(target)})
    assert first != second
