"""The worker modules of isolated backends import only what their env holds, never `senselab` or `senselab_core`."""

import ast
import importlib
import re
import sys
from pathlib import Path

import pytest
from senselab._api import generated  # noqa: F401 - importing it declares every backend
from senselab._api.declarations import DECLARATION_MODULES
from senselab_core.kernel.declare import REGISTRY

for _module in DECLARATION_MODULES:
    importlib.import_module(_module)

LOCKS = Path(__file__).parents[1] / "src" / "senselab" / "data" / "env_locks"
ALWAYS = {"numpy", "senselab_worker"}
IMPORT_NAMES = {"pyannote_audio": "pyannote"}
"""Import names that differ from their distribution's name."""


def _lock_libraries(env: str) -> set[str]:
    """Return the import names of the distributions in env `env`'s lock, torch pins included (`-` is `_`)."""
    text = (LOCKS / f"{env}.txt").read_text("utf-8")
    pins = re.search(r"^# torch: (.*)$", text, re.MULTILINE)
    names = re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==", text, re.MULTILINE)
    names += re.findall(r"([A-Za-z0-9._-]+)==", pins.group(1)) if pins else []
    libraries = {n.lower().replace("-", "_").replace(".", "_") for n in names}
    return libraries | {IMPORT_NAMES[n] for n in libraries & IMPORT_NAMES.keys()}


def outside_imports(source: str, module: str, siblings: set[str], allowed: set[str]) -> set[str]:
    """Return the absolute names `source` (module `module`) imports that are neither allowed nor placed siblings.

    A relative import resolves against `module`'s package; `senselab_core` is never allowed.
    """
    package = module.rpartition(".")[0].split(".")
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
                continue
            base = ".".join(package[: len(package) - (node.level - 1)] or ["<above-root>"])
            targets = [f"{base}.{node.module}"] if node.module else [f"{base}.{a.name}" for a in node.names]
            found |= set(targets)
    stdlib = sys.stdlib_module_names
    return {
        m
        for m in found
        if m not in siblings and (m.split(".")[0] == "senselab_core" or m.split(".")[0] not in stdlib | allowed)
    }


_PLACED = [(b, b.worker_sources) for t in REGISTRY.tasks() for b in REGISTRY.backends(t.name) if b.worker_sources]


@pytest.mark.contract
def test_some_backend_places_worker_sources() -> None:
    """The sweep below is not empty."""
    assert _PLACED


@pytest.mark.contract
@pytest.mark.parametrize(("backend", "sources"), _PLACED, ids=[b.name for b, _ in _PLACED])
def test_worker_modules_import_only_stdlib_numpy_the_runtime_their_env_and_each_other(
    backend: object, sources: dict[str, str]
) -> None:
    """Each import of a placed module is stdlib, numpy, `senselab_worker`, a library of its env's lock or a sibling."""
    env = backend.env.name  # type: ignore[attr-defined]
    allowed = ALWAYS | _lock_libraries(env)
    for module, source in sources.items():
        assert Path(source).is_file(), (module, source)
        outside = outside_imports(Path(source).read_text("utf-8"), module, set(sources), allowed)
        assert not outside, (module, sorted(outside))


MODULE = "pkg.sub.worker"
SIBLINGS = {"pkg.sub.worker", "pkg.sub.helper"}


@pytest.mark.contract
@pytest.mark.parametrize(
    "line",
    [
        "from .. import x",
        "from ... import x",
        "from ...kernel import y",
        "from ..other import z",
        "from . import missing",
        "import senselab_core",
        "from senselab_core.kernel import y",
        "def f():\n    import senselab_core.runtime",
        "def f():\n    from senselab_core import y",
        "import senselab.other",
        "import scipy",
    ],
)
def test_import_rule_refuses(line: str) -> None:
    """A relative import that leaves the placed siblings, any `senselab_core` import and an undeclared library fail."""
    assert outside_imports(line, MODULE, SIBLINGS, ALWAYS)


@pytest.mark.contract
@pytest.mark.parametrize(
    "line",
    [
        "from . import helper",
        "from .helper import f",
        "import os",
        "import numpy",
        "from senselab_worker.faults import X",
    ],
)
def test_import_rule_accepts(line: str) -> None:
    """A sibling (absolute or relative), stdlib, numpy and the runtime pass."""
    assert not outside_imports(line, MODULE, SIBLINGS, ALWAYS)
