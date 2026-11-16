"""Weights stamps across hosts: a stamp names the filesystem by its mount, not by ``st_dev``.

An NFS mount gets a different device number on each client, so a stamp prefetch wrote on one host must still
match on another; a blob on a different filesystem (or one that changed) must not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from senselab.utils.reproduce import weights
from senselab.utils.reproduce.spec import StepSpec
from tests.utils.reproduce.weights_test import (
    REPO,
    REVISION,
    WEIGHTS,
    _bump_mtime,
    _hub,
    _url_step,
    _url_world,
    _weights,
    make_snapshot,
)

MOUNTINFO = """\
22 1 8:1 / / rw,relatime shared:1 - xfs /dev/sda1 rw
40 22 0:52 / /net/proj rw,relatime shared:30 - nfs4 nfs1:/group/proj rw,vers=4.2
41 22 0:53 /sub /mnt/with\\040space rw shared:31 master:7 - nfs4 srv:/export\\040x\\134y\\011z rw
42 40 0:54 / /net/proj/hub rw shared:32 - nfs4 nfs2:/hub rw
43 40 0:55 / /net/proj/hub rw shared:33 - nfs4 nfs3:/hub rw
"""


@pytest.mark.parametrize(
    ("path", "identity"),
    [
        ("/net/proj/reruns/x.bin", "nfs4 nfs1:/group/proj"),
        ("/net/proj", "nfs4 nfs1:/group/proj"),
        ("/net/proj2/x", "xfs /dev/sda1"),
        ("/mnt/with space/a", "nfs4 srv:/export x\\y\tz"),
        ("/net/proj/hub/blobs/b", "nfs4 nfs3:/hub"),
        ("/tmp/x", "xfs /dev/sda1"),
    ],
)
def test_the_mount_identity_is_the_longest_mount_point_holding_the_path(path: str, identity: str) -> None:
    """The longest mount-point prefix (on path components) wins; of two at one point, the later; escapes decoded."""
    assert weights._mount_identity(path, MOUNTINFO) == identity


def test_no_mount_identity_without_a_parsable_line() -> None:
    """Unparsable or empty mountinfo gives None, so `filesystem_identity` falls back to the device number."""
    assert weights._mount_identity("/x", "") is None
    assert weights._mount_identity("/x", "garbage line\n") is None


def _stamped(tmp_path: Path) -> tuple[StepSpec, weights.Profile, Path, Path]:
    world = _url_world(tmp_path)
    top = make_snapshot(_hub(world), REPO, REVISION, {"weights.bin": WEIGHTS})
    step = StepSpec.model_validate(_url_step(world, hf_weights=_weights({"weights.bin": WEIGHTS})))
    profile, stamps = world.profile(), world.cache / weights.STAMPS_DIR
    assert weights.verify_hf([step], profile, {}, stamps=stamps) == []
    assert weights.verify_url([step], profile, {}, stamps=stamps) == []
    assert weights.check_stamps(step, profile, stamps) == []
    return step, profile, stamps, top


def _other_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``os.lstat`` (process-wide) report another device number for every file, as on another NFS client."""
    real = os.lstat

    def lstat(path: str | os.PathLike[str]) -> SimpleNamespace:
        st = real(path)
        fields = {f: getattr(st, f) for f in dir(st) if f.startswith("st_")}
        return SimpleNamespace(**{**fields, "st_dev": st.st_dev + 1000})

    monkeypatch.setattr(weights.os, "lstat", lstat)


def test_a_stamp_written_on_one_host_matches_on_another(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate on a second host (same files, same mount, another ``st_dev``) accepts the prefetch host's stamps."""
    step, profile, stamps, _ = _stamped(tmp_path)
    _other_device(monkeypatch)
    assert weights.check_stamps(step, profile, stamps) == []


def test_a_change_is_still_caught_on_another_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With another ``st_dev``, a blob whose mtime changed is still ``E_WEIGHTS_CHANGED``."""
    step, profile, stamps, top = _stamped(tmp_path)
    _other_device(monkeypatch)
    _bump_mtime(top / "blobs" / weights.hashlib.sha256(WEIGHTS).hexdigest())
    assert weights.check_stamps(step, profile, stamps) == ["E_WEIGHTS_CHANGED"]


def _mountinfo(tmp_path: Path, name: str, source: str) -> Path:
    """A mountinfo file whose only mount holds everything under ``/``, from ``source``."""
    path = tmp_path / name
    path.write_text(f"22 1 0:52 / / rw shared:1 - nfs4 {source} rw\n")
    return path


def test_a_blob_on_another_filesystem_is_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same path, size, mtime and inode under another mount source is ``E_WEIGHTS_CHANGED``."""
    monkeypatch.setattr(weights, "_MOUNTINFO", _mountinfo(tmp_path, "mountinfo-a", "nfs1:/group/proj"))
    step, profile, stamps, _ = _stamped(tmp_path)
    monkeypatch.setattr(weights, "_MOUNTINFO", _mountinfo(tmp_path, "mountinfo-b", "nfs1:/group/proj"))
    assert weights.check_stamps(step, profile, stamps) == []
    monkeypatch.setattr(weights, "_MOUNTINFO", _mountinfo(tmp_path, "mountinfo-c", "nfs9:/elsewhere"))
    assert weights.check_stamps(step, profile, stamps) == ["E_WEIGHTS_CHANGED"]


def test_without_mountinfo_the_identity_is_the_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where mountinfo cannot be read the identity falls back to ``dev:<st_dev>`` (one host, as before)."""
    monkeypatch.setattr(weights, "_MOUNTINFO", tmp_path / "absent")
    st = os.lstat(tmp_path)
    assert weights.filesystem_identity(str(tmp_path), st) == f"dev:{st.st_dev}"
    step, profile, stamps, _ = _stamped(tmp_path)
    assert weights.check_stamps(step, profile, stamps) == []


def test_a_stamp_without_a_filesystem_is_unverified(tmp_path: Path) -> None:
    """A stamp from before the filesystem was recorded (it holds ``st_dev``) is ``E_WEIGHTS_UNVERIFIED``."""
    step, profile, stamps, _ = _stamped(tmp_path)
    for stamp in stamps.glob("*.json"):
        data = json.loads(stamp.read_text())
        data.pop("fs")
        data["st_dev"] = 1
        stamp.write_text(json.dumps(data))
    assert weights.check_stamps(step, profile, stamps) == ["E_WEIGHTS_UNVERIFIED"]
    assert weights.verify_url([step], profile, {}, stamps=stamps) == []
    assert weights.verify_hf([step], profile, {}, stamps=stamps) == []
    assert weights.check_stamps(step, profile, stamps) == []


def test_the_stamp_records_the_mount_identity(tmp_path: Path) -> None:
    """Each stamp records ``fs`` as `filesystem_identity` of its blob and no ``st_dev``."""
    _, _, stamps, _ = _stamped(tmp_path)
    for stamp in stamps.glob("*.json"):
        data = json.loads(stamp.read_text())
        assert "st_dev" not in data
        assert data["fs"] == weights.filesystem_identity(data["blob"], os.lstat(data["blob"]))
