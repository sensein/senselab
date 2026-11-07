"""`probe_capabilities`: what a root does, measured on the root itself."""

import errno
from pathlib import Path

import pytest
from senselab_core.runtime.storage import StorageCapabilities, StorageUnsupported, probe_capabilities
from syscall_faults_fixture import refuse_call

pytestmark = pytest.mark.logic


def test_a_local_directory_has_every_primitive_and_leaves_nothing(tmp_path: Path) -> None:
    """A local disk passes every probe with a small delay, and the probe directory is removed."""
    found = probe_capabilities(tmp_path)
    assert found.missing() == [] and 0 <= found.attribute_delay_s < 1
    found.require(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_root_not_made_yet_is_judged_by_its_ancestor_and_not_made(tmp_path: Path) -> None:
    """A missing root is probed through its nearest existing ancestor; nothing is created."""
    assert probe_capabilities(tmp_path / "a" / "b").missing() == []
    assert list(tmp_path.iterdir()) == []


def test_a_store_without_directories_fails_atomic_mkdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A root whose `mkdir` is not supported (an object store) is `StorageUnsupported`, naming `mkdir`."""
    refuse_call(monkeypatch, "mkdir", ".senselab-probe", OSError(errno.ENOTSUP, "Operation not supported"))
    with pytest.raises(StorageUnsupported, match="atomic mkdir") as raised:
        probe_capabilities(tmp_path)
    assert raised.value.errno == errno.ENOTSUP and list(tmp_path.iterdir()) == []


def test_a_mount_that_refuses_link_lacks_atomic_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A root that refuses `link(2)` is judged to lack `atomic_rename`, and `require` names it."""
    refuse_call(monkeypatch, "link", "b", OSError(errno.EPERM, "Operation not permitted"))
    found = probe_capabilities(tmp_path)
    assert found.atomic_rename is False and found.missing() == ["atomic_rename"]
    with pytest.raises(StorageUnsupported, match="atomic_rename"):
        found.require(tmp_path)


def test_require_names_each_missing_primitive() -> None:
    """Every missing primitive is named, in a fixed order."""
    found = StorageCapabilities(False, True, False, True, 0.0)
    with pytest.raises(StorageUnsupported, match="atomic_mkdir, dir_mtime_server_clock"):
        found.require(Path("/store"))
