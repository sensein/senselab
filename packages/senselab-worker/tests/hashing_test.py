"""Content identity of files and weight sets."""

import hashlib
import os
from pathlib import Path

import pytest
from senselab_worker.canonical import jcs_dumps
from senselab_worker.hashing import content_identity, sha256_bytes, weights_identity


@pytest.mark.logic
def test_sha256_bytes_is_hex_digest() -> None:
    """The digest is lowercase hex sha256."""
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


@pytest.mark.logic
def test_content_identity_chunks(tmp_path: Path) -> None:
    """A 10-byte file in 4-byte chunks has three chunk hashes and a root over their concatenation."""
    p = tmp_path / "x.bin"
    p.write_bytes(b"a" * 10)
    cid = content_identity(p, chunk_size=4)
    assert len(cid["chunks"]) == 3 and cid["root"] == sha256_bytes("".join(cid["chunks"]).encode())
    assert cid["chunks"] == [sha256_bytes(b"aaaa"), sha256_bytes(b"aaaa"), sha256_bytes(b"aa")]
    assert cid["algo"] == "sha256" and cid["chunk_size"] == 4


@pytest.mark.logic
def test_content_identity_default_chunk_size_and_empty_file(tmp_path: Path) -> None:
    """The default chunk is 64 MiB, and an empty file has no chunks."""
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    cid = content_identity(p)
    assert cid == {"algo": "sha256", "chunk_size": 64 * 2**20, "chunks": [], "root": sha256_bytes(b"")}


@pytest.mark.logic
def test_content_identity_exact_multiple(tmp_path: Path) -> None:
    """A file that is a whole number of chunks has no trailing empty chunk."""
    p = tmp_path / "x.bin"
    p.write_bytes(b"abcdefgh")
    assert content_identity(p, chunk_size=4)["chunks"] == [sha256_bytes(b"abcd"), sha256_bytes(b"efgh")]


@pytest.mark.logic
@pytest.mark.parametrize("bad", [0, -1])
def test_content_identity_refuses_non_positive_chunk(tmp_path: Path, bad: int) -> None:
    """A chunk size below one byte is refused."""
    p = tmp_path / "x.bin"
    p.write_bytes(b"a")
    with pytest.raises(ValueError):
        content_identity(p, chunk_size=bad)


def _weights_dir(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    (root / "sub").mkdir(parents=True)
    (root / "config.json").write_bytes(b"{}")
    (root / "sub" / "w.bin").write_bytes(b"\x00\x01\x02")
    return root


@pytest.mark.logic
def test_weights_identity_ignores_listing_order(tmp_path: Path) -> None:
    """The identity does not depend on the order the files are listed in."""
    root = _weights_dir(tmp_path)
    a = weights_identity([root / "config.json", root / "sub" / "w.bin"], root=root)
    b = weights_identity([str(root / "sub" / "w.bin"), str(root / "config.json")], root=str(root))
    assert a == b


@pytest.mark.logic
def test_weights_identity_is_jcs_of_sorted_pairs(tmp_path: Path) -> None:
    """The identity hashes the JCS of sorted [POSIX relpath, file sha256] pairs."""
    root = _weights_dir(tmp_path)
    expected = sha256_bytes(
        jcs_dumps([["config.json", sha256_bytes(b"{}")], ["sub/w.bin", sha256_bytes(b"\x00\x01\x02")]])
    )
    assert weights_identity([root / "sub" / "w.bin", root / "config.json"], root=root) == expected


@pytest.mark.logic
def test_weights_identity_depends_on_names_and_bytes(tmp_path: Path) -> None:
    """Renaming a file or changing a byte changes the identity."""
    root = _weights_dir(tmp_path)
    before = weights_identity([root / "config.json", root / "sub" / "w.bin"], root=root)
    (root / "sub" / "w.bin").rename(root / "sub" / "v.bin")
    renamed = weights_identity([root / "config.json", root / "sub" / "v.bin"], root=root)
    (root / "sub" / "v.bin").write_bytes(b"\x00\x01\x03")
    changed = weights_identity([root / "config.json", root / "sub" / "v.bin"], root=root)
    assert len({before, renamed, changed}) == 3


@pytest.mark.logic
def test_weights_identity_hashes_symlink_targets_under_root_names(tmp_path: Path) -> None:
    """A symlink inside the root is named by its link path and hashed by its target's bytes."""
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    root = tmp_path / "snapshot"
    root.mkdir()
    os.symlink(blob, root / "model.bin")
    expected = sha256_bytes(jcs_dumps([["model.bin", sha256_bytes(b"weights")]]))
    assert weights_identity([root / "model.bin"], root=root) == expected


@pytest.mark.logic
def test_weights_identity_refuses_paths_outside_root_and_duplicates(tmp_path: Path) -> None:
    """A path outside the root, or the same path twice, is refused."""
    root = _weights_dir(tmp_path)
    outside = tmp_path / "other.bin"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        weights_identity([outside], root=root)
    with pytest.raises(ValueError):
        weights_identity([root / "config.json", root / "sub" / ".." / "config.json"], root=root)


@pytest.mark.logic
def test_weights_identity_normalises_names_to_nfc(tmp_path: Path) -> None:
    """A file named in NFD hashes as its NFC name, so macOS and Linux checkouts agree."""
    root = tmp_path / "model"
    root.mkdir()
    nfd = "café.bin"
    (root / nfd).write_bytes(b"w")
    expected = sha256_bytes(jcs_dumps([["café.bin", sha256_bytes(b"w")]]))
    assert weights_identity([root / nfd], root=root) == expected
    with pytest.raises(ValueError):
        weights_identity([root / nfd, root / "café.bin"], root=root)


@pytest.mark.logic
def test_weights_identity_sorts_by_utf16_code_units(tmp_path: Path) -> None:
    """Pairs sort by UTF-16 code units, so an astral name sorts below U+FB01."""
    root = tmp_path / "model"
    root.mkdir()
    for name in ("ﬁ.bin", "\U0001f600.bin"):
        (root / name).write_bytes(name.encode())
    pairs = [[n, sha256_bytes(n.encode())] for n in ("\U0001f600.bin", "ﬁ.bin")]
    assert weights_identity([root / "ﬁ.bin", root / "\U0001f600.bin"], root=root) == sha256_bytes(jcs_dumps(pairs))
