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


def _lock_libraries(env: str) -> set[str]:
    """Return the import names of the distributions in env `env`'s lock, torch pins included (`-` is `_`)."""
    text = (LOCKS / f"{env}.txt").read_text("utf-8")
    pins = re.search(r"^# torch: (.*)$", text, re.MULTILINE)
    names = re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==", text, re.MULTILINE)
    names += re.findall(r"([A-Za-z0-9._-]+)==", pins.group(1)) if pins else []
    return {n.lower().replace("-", "_").replace(".", "_") for n in names}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add(node.module or "")
    return found


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
        outside = {
            m
            for m in _imports(Path(source))
            if m not in sources and m.split(".")[0] not in sys.stdlib_module_names | allowed
        }
        assert not outside, (module, sorted(outside))
        assert not {m for m in _imports(Path(source)) if m.split(".")[0] == "senselab_core"}
