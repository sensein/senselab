"""Markers for the reproduce tests.

``slow`` tests build real venvs and need network; ``network`` tests need a compute node with
network outside the sandbox. Neither runs unless selected with ``-m``. Tests that use ``nfs_dir``
run only when ``SENSELAB_REPRO_NFS_DIR`` names a directory to create on an NFS mount.
"""

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the two markers so ``--strict-markers`` accepts them."""
    config.addinivalue_line("markers", "slow: builds real environments and needs network")
    config.addinivalue_line("markers", "network: needs a compute node with network, outside the sandbox")


NFS_DIR_VARIABLE = "SENSELAB_REPRO_NFS_DIR"


def _filesystem_type(path: Path) -> str | None:
    """The type of the mount holding ``path``, from the longest matching ``/proc/mounts`` entry."""
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return None
    best, kind = "", None
    target = os.path.realpath(path)
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = fields[1].replace("\\040", " ")
        if (target == mount or target.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
            best, kind = mount, fields[2]
    return kind


@pytest.fixture
def nfs_dir() -> Iterator[Path]:
    """A scratch directory on NFS, named by ``SENSELAB_REPRO_NFS_DIR``; created here and removed afterwards.

    Skipped when the variable is unset, the directory already exists, or its parent is not on NFS.
    """
    value = os.environ.get(NFS_DIR_VARIABLE, "")
    if not value:
        pytest.skip(f"{NFS_DIR_VARIABLE} is not set")
    path = Path(value)
    if path.exists():
        pytest.skip(f"{path} already exists; it is created and removed by the test")
    kind = _filesystem_type(path.parent)
    if kind is None or not kind.startswith("nfs"):
        pytest.skip(f"{path.parent} is on {kind}, not NFS")
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
