"""Markers for the reproduce tests.

``slow`` tests build real environments; ``network`` tests reach PyPI, a package index or Slurm from a node
with network outside the sandbox (a test can carry both). Neither kind runs by default: the collection hook
here deselects them unless the ``-m`` expression names one of the test's markers, or
``SENSELAB_REPRO_SLOW=1`` is set. Tests that use ``nfs_dir`` run only when ``SENSELAB_REPRO_NFS_DIR`` names
a directory to create on an NFS mount.
"""

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from senselab.utils.reproduce import sandbox

OPT_IN = ("slow", "network")
SLOW_VARIABLE = "SENSELAB_REPRO_SLOW"
HERE = Path(__file__).resolve().parent


def pytest_configure(config: pytest.Config) -> None:
    """Register the two markers so ``--strict-markers`` accepts them."""
    config.addinivalue_line("markers", "slow: builds real environments (opt-in)")
    config.addinivalue_line("markers", "network: needs network outside the sandbox: PyPI, an index or Slurm (opt-in)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect this directory's ``slow`` and ``network`` tests unless ``-m`` names one of their markers.

    ``SENSELAB_REPRO_SLOW=1`` keeps them all.
    """
    if os.environ.get(SLOW_VARIABLE) == "1":
        return
    expression = str(config.getoption("markexpr", "") or "")
    named = {m for m in OPT_IN if m in expression}
    keep: list[pytest.Item] = []
    drop: list[pytest.Item] = []
    for item in items:
        marks = {m.name for m in item.iter_markers()} & set(OPT_IN)
        here = Path(str(item.path)).resolve().is_relative_to(HERE)
        (drop if here and marks and not (marks & named) else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


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


@pytest.fixture(autouse=True)
def _tool_masks() -> Iterator[None]:
    """Give every test an empty mask set for build and env sandboxes, as an entry point with a profile would."""
    with sandbox.tool_masks(()):
        yield
