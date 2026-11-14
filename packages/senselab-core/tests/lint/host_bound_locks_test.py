"""No 2.x source holds a lock whose stale break depends on the holder's host: `filelock`, `flock`, `lockf`.

`filelock`'s `SoftFileLock` breaks a stale lock only when its holder's pid is dead on the host that wrote it, so a
coordinator killed on one node leaves a lock no process of another node can break. The runtime's exclusion is the lease
(`runtime/locks.py`) and the epoch, both judged on the storage clock. A lock primitive of the forms below, anywhere in
a package's `src/`, fails here unless `ALLOWED` names it and says why it may stay. A lock that a dependency takes
(`pydra`'s `save`) is not source of ours; the resume tests run in the `another host` shape to meet it.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

ROOT = Path(__file__).resolve().parents[4]
PACKAGES = ROOT / "packages"
MODULES = frozenset({"filelock", "portalocker", "fasteners", "lockfile", "pid", "oslo_concurrency"})
"""Importable names whose locks break by a holder's pid and host (or by nothing)."""
CLASSES = frozenset({"SoftFileLock", "FileLock", "UnixFileLock", "WindowsFileLock", "InterProcessLock", "lockutils"})
"""Names of lock classes of those modules, found where they are used."""
CALLS = frozenset({("fcntl", "flock"), ("fcntl", "lockf"), ("msvcrt", "locking")})
"""Calls that take an advisory lock the kernel drops with the process but a network file system does not."""
ALLOWED: dict[tuple[str, str], str] = {}
"""(path under `packages/`, primitive) -> why that use may stay. Empty: no use stays."""


def _use(node: ast.AST) -> str | None:
    """Return the host-bound lock primitive `node` is a use of, or None."""
    if isinstance(node, ast.Import):
        return next((f"import {a.name}" for a in node.names if a.name.split(".")[0] in MODULES), None)
    if isinstance(node, ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        return f"from {node.module} import" if root in MODULES | {"fcntl", "msvcrt"} else None
    if isinstance(node, ast.Name):
        return node.id if node.id in CLASSES else None
    if isinstance(node, ast.Attribute):
        if node.attr in CLASSES:
            return node.attr
        if isinstance(node.value, ast.Name) and (node.value.id, node.attr) in CALLS:
            return f"{node.value.id}.{node.attr}"
    return None


def _roots(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Yield the line and primitive of each host-bound lock use in `tree`."""
    for node in ast.walk(tree):
        if (primitive := _use(node)) is not None:
            yield getattr(node, "lineno", 0), primitive


def findings(source: str) -> list[tuple[int, str]]:
    """Return the line and primitive of each host-bound lock use in `source`."""
    return sorted(set(_roots(ast.parse(source))))


def tree_findings() -> dict[tuple[str, str], list[int]]:
    """Return the lines of each (path under `packages/`, primitive) use in the packages' `src/` trees."""
    found: dict[tuple[str, str], list[int]] = {}
    for path in sorted(PACKAGES.glob("*/src/**/*.py")):
        for line, primitive in findings(path.read_text()):
            found.setdefault((str(path.relative_to(PACKAGES)), primitive), []).append(line)
    return found


@pytest.mark.parametrize(
    ("source", "primitive"),
    [
        ("import filelock\n", "import filelock"),
        ("from filelock import SoftFileLock\n", "from filelock import"),
        ("import filelock as fl\nlock = fl.SoftFileLock('x')\n", "SoftFileLock"),
        ("lock = FileLock('x')\n", "FileLock"),
        ("import fcntl\nfcntl.flock(fd, fcntl.LOCK_EX)\n", "fcntl.flock"),
        ("import fcntl\nfcntl.lockf(fd, 1)\n", "fcntl.lockf"),
        ("from fcntl import flock\n", "from fcntl import"),
        ("import portalocker\n", "import portalocker"),
    ],
)
def test_every_host_bound_lock_form_is_found(source: str, primitive: str) -> None:
    """Each way to take a lock that depends on its holder's host is reported with its line."""
    assert primitive in [p for _, p in findings(source)]


def test_a_lease_and_an_atomic_create_are_not_host_bound_locks() -> None:
    """The runtime's own exclusions, and a file created exclusively, pass."""
    source = (
        "import os\nfd = os.open(p, os.O_CREAT | os.O_EXCL)\nfrom senselab_core.runtime.locks import Lease\nLease(p)\n"
    )
    assert findings(source) == []


def test_the_runtime_holds_no_host_bound_lock_outside_the_allowlist() -> None:
    """No package's `src/` takes a lock whose stale break reads the holder's host, bar `ALLOWED`."""
    unlisted = {key: lines for key, lines in tree_findings().items() if key not in ALLOWED}
    assert unlisted == {}, "a lock primitive whose stale break is host-bound; use a lease, or name it in ALLOWED"


def test_every_allowed_use_still_exists_and_has_a_reason() -> None:
    """An entry of `ALLOWED` that matches no use, or gives no reason, is removed or written."""
    found = tree_findings()
    assert [key for key in ALLOWED if key not in found] == []
    assert [key for key, why in ALLOWED.items() if not why.strip()] == []
