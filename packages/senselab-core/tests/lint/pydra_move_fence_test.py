"""The modules planned to move to pydra import no senselab module or name beyond their recorded baseline.

`BASELINE` maps each module (path under `senselab_core/runtime/`) to the senselab names it imports today, each with
a one-line reason pointing at a section of `specs/20261030-140022-senselab-2x-api/route-b.md`. A name is
`module.attribute` for `from module import attribute` and `module` for `import module`. Every `senselab*` root counts,
at any nesting depth, and a relative import is resolved against the module's package first. A new import fails here
until it is added with its reason.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

RUNTIME = Path(__file__).resolve().parents[2] / "src" / "senselab_core" / "runtime"
BASELINE: dict[str, dict[str, str]] = {
    "locks.py": {
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b.md Summary)"
        ),
        "senselab_core.runtime.clock.SystemClock": ("default of the Clock seam (route-b.md Summary)"),
        "senselab_core.runtime.execution_settings.JobLease": (
            "lease timing; becomes pydra LeaseSettings (route-b.md Summary)"
        ),
        "senselab_core.runtime.profiles.ExecutionProfile": (
            "policy carrier; passed as hooks when the module moves (route-b.md Summary)"
        ),
        "senselab_core.runtime.storage.StorageDir": (
            "same move group: storage.py moves with this module (route-b.md section 4)"
        ),
        "senselab_core.runtime.storage.WrongKindError": (
            "same move group: storage.py moves with this module (route-b.md section 4)"
        ),
        "senselab_core.runtime.storage.owner_remedy": (
            "same move group: storage.py moves with this module (route-b.md section 4)"
        ),
        "senselab_core.runtime.storage.path_component": (
            "same move group: storage.py moves with this module (route-b.md section 4)"
        ),
        "senselab_worker.ownership.UUID_HEX": ("stays in senselab-worker (route-b.md Summary)"),
    },
    "storage.py": {
        "senselab_core.kernel._base.Frozen": ("pydantic base of the module's value types"),
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b.md Summary)"
        ),
        "senselab_core.runtime.clock.SystemClock": ("default of the Clock seam (route-b.md Summary)"),
        "senselab_core.runtime.config.load_yaml": ("the one YAML loader"),
        "senselab_core.runtime.execution_settings.StoreSettings": (
            "store settings; policy passed in at the move (route-b.md Summary)"
        ),
        "senselab_core.runtime.faults.is_transient": ("fault classification of storage errors (route-b.md Summary)"),
        "senselab_core.runtime.fetch": ("deferred import; both import this module (route-b.md Summary)"),
        "senselab_core.runtime.profiles.ExecutionProfile": (
            "policy carrier; passed as hooks when the module moves (route-b.md Summary)"
        ),
        "senselab_core.runtime.store": ("deferred import; both import this module (route-b.md Summary)"),
        "senselab_worker.ownership.UUID_HEX": ("stays in senselab-worker (route-b.md Summary)"),
        "senselab_worker.ownership.remove_owned": ("stays in senselab-worker (route-b.md Summary)"),
    },
    "execution/supervisor.py": {
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b.md Summary)"
        ),
        "senselab_core.runtime.clock.SystemClock": ("default of the Clock seam (route-b.md Summary)"),
        "senselab_core.runtime.execution_settings.SupervisorSettings": (
            "shutdown graces; hook of the G2 Venv runner (route-b.md Summary)"
        ),
        "senselab_core.runtime.hardware.proc_stats": (
            "process statistics helper for the supervisor (route-b.md Summary)"
        ),
        "senselab_core.runtime.profiles.ExecutionProfile": (
            "policy carrier; passed as hooks when the module moves (route-b.md Summary)"
        ),
    },
    "worker_pipes.py": {
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b.md Summary)"
        ),
        "senselab_worker.codec.Deadline": ("stays in senselab-worker (route-b.md Summary)"),
    },
}
"""module -> imported senselab name -> why it is there."""


def senselab_imports(source: str, module: str) -> set[str]:
    """Return every senselab name imported anywhere in `source`, a relative import resolved against `module`."""
    package = ["senselab_core", "runtime", *module.removesuffix(".py").split("/")[:-1]]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            target = ".".join([*base, *([node.module] if node.module else [])])
            if target.split(".")[0].startswith("senselab"):
                names |= {f"{target}.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names if alias.name.split(".")[0].startswith("senselab")}
    return names


def new_imports(source: str, module: str) -> set[str]:
    """Return the senselab names `source` imports that `BASELINE[module]` does not list."""
    return senselab_imports(source, module) - BASELINE[module].keys()


@pytest.mark.parametrize("module", sorted(BASELINE))
def test_module_imports_nothing_new_from_senselab(module: str) -> None:
    """The module's senselab imports are all in its baseline, and the baseline lists no import that is gone."""
    path = RUNTIME / module
    assert path.is_file(), f"{module} moved or was renamed: update BASELINE"
    imported = senselab_imports(path.read_text(encoding="utf-8"), module)
    assert not imported - BASELINE[module].keys(), (
        f"{module} gained senselab imports: {sorted(imported - BASELINE[module].keys())}"
    )
    assert not BASELINE[module].keys() - imported, (
        f"{module} dropped imports, remove from BASELINE: {sorted(BASELINE[module].keys() - imported)}"
    )


@pytest.mark.parametrize("module", sorted(BASELINE))
def test_every_baseline_entry_has_a_reason(module: str) -> None:
    """Each baseline entry carries a non-empty one-line reason."""
    assert all(reason.strip() and "\n" not in reason for reason in BASELINE[module].values())


@pytest.mark.parametrize(
    ("planted", "expected"),
    [
        ("from senselab_core.runtime.store import ResultStore\n", {"senselab_core.runtime.store.ResultStore"}),
        ("import senselab_core.kernel.values\n", {"senselab_core.kernel.values"}),
        ("from senselab.audio import Audio\n", {"senselab.audio.Audio"}),
        ("def f():\n    from senselab_worker.wire import Frame\n", {"senselab_worker.wire.Frame"}),
        ("from os import path\nimport numpy\n", set()),
    ],
)
def test_planted_import_is_found(planted: str, expected: set[str]) -> None:
    """A senselab import outside the baseline, in any form or depth, is reported; other imports are not."""
    for module in BASELINE:
        assert new_imports(planted, module) == expected


@pytest.mark.parametrize("module", sorted(BASELINE))
def test_planted_import_in_real_source_is_new(module: str) -> None:
    """A senselab import appended to a module's real source is the only new import reported."""
    source = (RUNTIME / module).read_text(encoding="utf-8") + "\nfrom senselab_core.runtime.planted import Planted\n"
    assert new_imports(source, module) == {"senselab_core.runtime.planted.Planted"}


@pytest.mark.parametrize(
    ("planted", "module", "expected"),
    [
        ("from . import x\n", "locks.py", {"senselab_core.runtime.x"}),
        ("from ..store import X\n", "execution/supervisor.py", {"senselab_core.runtime.store.X"}),
        ("from .sub import Y\n", "execution/supervisor.py", {"senselab_core.runtime.execution.sub.Y"}),
        ("from ..store import X\n", "storage.py", {"senselab_core.store.X"}),
        ("from .clock import Clock\n", "worker_pipes.py", set()),
    ],
)
def test_relative_imports_are_resolved(planted: str, module: str, expected: set[str]) -> None:
    """A relative import is fenced by its absolute name."""
    assert new_imports(planted, module) == expected
