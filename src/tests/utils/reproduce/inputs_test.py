"""Tests for registered inputs: the registry entries, the per-file manifest format and the manifest writer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import xxhash
import yaml
from pydantic import ValidationError

from senselab.utils.reproduce import inputs
from senselab.utils.reproduce.inputs import InputError, RegisteredInput, load_registry, read_manifest, write_manifest
from senselab.utils.reproduce.spec import StepSpec
from tests.utils.reproduce.cache_test import _step_data

SHA = "ab" * 32


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    data: dict[str, Any] = {
        "id": "tables",
        "root": "main",
        "relpath": "derivatives/tables",
        "manifest": {"root": "main", "relpath": "_manifests/tables.xxh3.tsv", "sha256": SHA},
        "rung": "a",
        "rows": ["C1-C5", "N3"],
        "restricted": False,
        "licence": "CC-BY-NC-SA-3.0",
        "category": "derivative",
    }
    data.update(over)
    return data


def test_write_then_read_manifest_round_trip_with_links_and_excludes(tmp_path: Path) -> None:
    """A file, a relative symlink and an excluded dotfile: the link is recorded by target, the dotfile left out."""
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.csv").write_text("x,y\n1,2\n")
    os.symlink("sub/a.csv", root / "link.csv")
    (root / "sub" / ".clip.mp4.part").write_text("partial")
    out = tmp_path / "out" / "tree.xxh3.tsv"
    counts = write_manifest(root, out, exclude=["**/.*.mp4.*"])
    assert counts == {"entries": 2, "links": 1, "unreadable": 0, "walk_errors": 0, "bytes": 8}
    rows = read_manifest(out, _sha(out))
    assert [r.relpath for r in rows] == ["link.csv", "sub/a.csv"]
    link, data = rows
    assert link.link_target == "sub/a.csv" and link.size is None and link.xxh3_128 is None
    st = os.lstat(root / "sub" / "a.csv")
    assert (data.size, data.mtime_ns, data.link_target) == (8, st.st_mtime_ns, None)
    assert data.xxh3_128 == xxhash.xxh3_128(b"x,y\n1,2\n").hexdigest() and data.sha256 is None
    with_sha = tmp_path / "out" / "tree-sha.xxh3.tsv"
    write_manifest(root, with_sha, exclude=["**/.*.mp4.*"], sha256=True)
    (_, data_sha) = read_manifest(with_sha, _sha(with_sha))
    assert data_sha.sha256 == hashlib.sha256(b"x,y\n1,2\n").hexdigest()
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".")]
    old = os.umask(0o027)
    try:
        write_manifest(root, out, exclude=["**/.*.mp4.*"])
    finally:
        os.umask(old)
    assert out.stat().st_mode & 0o777 == 0o640


def test_existing_hash_tree_format_parses(tmp_path: Path) -> None:
    """The format the existing tree hasher writes: one ``# root=`` line, the header, then rows."""
    text = (
        "# root=/data/tree date=2026-01-01T00:00:00 host=node-1 xxhash=3.6.0\n"
        "relpath\tsize\tmtime_ns\txxh3_128\tlink_target\n"
        "a.csv\t8\t1700000000000000000\t" + "0" * 32 + "\t\n"
        "b\t\t\t\tsub/b\n"
    )
    path = tmp_path / "m.xxh3.tsv"
    path.write_text(text)
    rows = read_manifest(path, _sha(path))
    assert rows[0] == inputs.Row("a.csv", 8, 1700000000000000000, "0" * 32, None, None)
    assert rows[1] == inputs.Row("b", None, None, None, "sub/b", None)
    for bad in (
        text.replace("# root=", "# note="),
        text.replace("relpath\tsize", "path\tsize"),
        text.replace("0" * 32, "ERR:PermissionError"),
        text + "short\t1\n",
        text.replace("a.csv\t", "/abs.csv\t"),
        text.replace("a.csv\t", "../a.csv\t"),
        text.replace("a.csv\t", "x/./a.csv\t"),
        text.replace("b\t\t", "a.csv\t\t"),
        text.replace("a.csv\t", "c.csv\t"),
    ):
        path.write_text(bad)
        with pytest.raises(InputError) as info:
            read_manifest(path, _sha(path))
        assert info.value.code == "E_INPUT_REGISTRY" and str(info.value) == "E_INPUT_REGISTRY"


def test_manifest_sha_mismatch_is_e_input_registry(tmp_path: Path) -> None:
    """A manifest whose bytes differ from the registered sha256, or that is missing, is ``E_INPUT_REGISTRY``."""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.csv").write_text("1\n")
    out = tmp_path / "a.xxh3.tsv"
    write_manifest(root, out)
    for path, sha in ((out, "cd" * 32), (tmp_path / "missing.tsv", _sha(out))):
        with pytest.raises(InputError) as info:
            read_manifest(path, sha)
        assert info.value.code == "E_INPUT_REGISTRY" and str(tmp_path) not in str(info.value)


def test_registry_id_must_match_file_stem(tmp_path: Path) -> None:
    """``inputs/<id>.yaml`` must exist, validate and carry its own stem as ``id``."""
    registry = tmp_path / inputs.REGISTRY_DIR
    registry.mkdir()
    (registry / "tables.yaml").write_text(yaml.safe_dump(_entry()))
    (registry / "other.yaml").write_text(yaml.safe_dump(_entry()))
    (registry / "broken.yaml").write_text(yaml.safe_dump(_entry(id="broken", relpath="../up")))
    loaded = load_registry(tmp_path, ["tables"])
    assert loaded["tables"] == RegisteredInput.model_validate(_entry())
    for ids in (["other"], ["missing"], ["broken"]):
        with pytest.raises(InputError) as info:
            load_registry(tmp_path, ids)
        assert info.value.code == "E_INPUT_REGISTRY"
    original = _entry(category="derivative", original={"root": "main", "relpath": "legacy/tables"})
    assert RegisteredInput.model_validate(original).original is not None
    assert RegisteredInput.model_validate(_entry(relpath=".")).relpath == "."
    assert RegisteredInput.model_validate(_entry(include=["**/*.csv"], exclude=["**/.*"])).include == ["**/*.csv"]
    for patterns in ({"include": ["/abs"]}, {"include": ["a/../b"]}, {"exclude": ["a//b"]}, {"exclude": [""]}):
        with pytest.raises(ValidationError):
            RegisteredInput.model_validate(_entry(**patterns))


def test_input_placeholder_must_be_declared() -> None:
    """``{input:<id>}`` is a keyed placeholder; it must name a ``lock_inputs`` entry."""
    argv = ["{venv}/bin/python", "run.py", "--table", "{input:tables}/a.csv", "--out", "{out}", "{up:prep}/y"]
    step = StepSpec.model_validate(_step_data(argv=argv, lock_inputs=["tables"]))
    assert step.lock_inputs == ["tables"]
    with pytest.raises(ValidationError, match="E_SPEC_INPUT_UNDECLARED"):
        StepSpec.model_validate(_step_data(argv=argv))
    with pytest.raises(ValidationError, match="E_SPEC_INPUT_UNDECLARED"):
        StepSpec.model_validate(_step_data(pythonpath=["{input:other}"], lock_inputs=["tables"]))
    with pytest.raises(ValidationError, match="duplicates"):
        StepSpec.model_validate(_step_data(lock_inputs=["tables", "tables"]))


def test_a_literal_include_walks_nothing_else(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``include: [one.csv]`` lstats that file: 1 entry, no walk error, and the root is never listed."""
    root = tmp_path / "big"
    root.mkdir()
    for i in range(1000):
        (root / f"f{i:04d}.txt").write_text("x")
    (root / "one.csv").write_text("1\n")
    locked = root / "locked"
    locked.mkdir()
    (locked / "secret.txt").write_text("s")
    locked.chmod(0)
    listed: list[str] = []
    real = os.scandir

    def spy(path: Any = ".") -> Any:  # noqa: ANN401
        listed.append(os.fspath(path))
        return real(path)

    monkeypatch.setattr(os, "scandir", spy)
    try:
        counts = write_manifest(root, tmp_path / "one.xxh3.tsv", include=["one.csv"])
    finally:
        locked.chmod(0o700)
    assert counts["entries"] == 1 and counts["walk_errors"] == 0 and counts["unreadable"] == 0
    assert os.fspath(root) not in listed
    rows = read_manifest(tmp_path / "one.xxh3.tsv", _sha(tmp_path / "one.xxh3.tsv"))
    assert [r.relpath for r in rows] == ["one.csv"]
    monkeypatch.setattr(os, "scandir", real)
    for missing in (["absent.csv"], ["nodir/x.csv"]):
        with pytest.raises(InputError) as info:
            write_manifest(root, tmp_path / "missing.xxh3.tsv", include=missing)
        assert info.value.code == "E_INPUT_REGISTRY"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "a.csv").write_text("a\n")
    os.symlink(elsewhere, root / "linked")
    for include in (["linked/a.csv"], ["linked/*.csv"], ["linked/sub/**"]):
        out = tmp_path / "linked.xxh3.tsv"
        assert write_manifest(root, out, include=include)["links"] == 1, include
        (link,) = read_manifest(out, _sha(out))
        assert (link.relpath, link.link_target) == ("linked", os.fspath(elsewhere)), include


def test_licence_and_rows_grammar() -> None:
    """SPDX-like licences and row references with ascending same-letter ranges; ``none`` is no rows."""
    for licence in ("Apache-2.0", "CC-BY-NC-SA-3.0", "LicenseRef-internal-use"):
        assert RegisteredInput.model_validate(_entry(licence=licence)).licence == licence
    assert RegisteredInput.model_validate(_entry(rows=["C1-C5", "N3"])).rows == ["C1-C5", "N3"]
    assert RegisteredInput.model_validate(_entry(rows="none")).rows == []
    assert RegisteredInput.model_validate(_entry(rows=[])).rows == []
    for rows in (["C5-C1"], ["C1-N3"], ["C1–C5"], ["all rows"], "all rows", ["C3-C3"]):
        with pytest.raises(ValidationError):
            RegisteredInput.model_validate(_entry(rows=rows))
    for licence in ("", "-x", "has space", "a" * 65):
        with pytest.raises(ValidationError):
            RegisteredInput.model_validate(_entry(licence=licence))
