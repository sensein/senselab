"""The modules planned to move to pydra import no senselab module or name beyond their recorded baseline.

`BASELINE` maps each module (path under `senselab_core/runtime/`) to the senselab names it imports today, each with
a one-line reason. A name is `module.attribute` for `from module import attribute` and `module` for `import module`;
every `senselab*` root counts, at any nesting depth. A new import fails here until it is added with its reason.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

RUNTIME = Path(__file__).resolve().parents[2] / "src" / "senselab_core" / "runtime"
BASELINE: dict[str, dict[str, str]] = {
    "locks.py": {
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b summary, lease row)"
        ),
        "senselab_core.runtime.clock.SystemClock": ("default of the Clock seam (route-b summary, lease row)"),
        "senselab_core.runtime.execution_settings.JobLease": (
            "lease timing; becomes pydra LeaseSettings (route-b summary, lease row)"
        ),
        "senselab_core.runtime.profiles.ExecutionProfile": (
            "policy carrier; passed as hooks when the module moves (route-b summary, hooks column)"
        ),
        "senselab_core.runtime.storage.StorageDir": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.WrongKindError": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.owner_remedy": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.path_component": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_worker.ownership.UUID_HEX": ("stays in senselab-worker (route-b summary, supervision row)"),
    },
    "storage.py": {
        "senselab_core.kernel._base.Frozen": ("pydantic base of the module's value types"),
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b summary, lease row)"
        ),
        "senselab_core.runtime.clock.SystemClock": ("default of the Clock seam (route-b summary, lease row)"),
        "senselab_core.runtime.config.load_yaml": ("the one YAML loader (import-linter contract one-yaml-loader)"),
        "senselab_core.runtime.execution_settings.StoreSettings": (
            "store settings; policy passed in at the move (route-b summary, result store row)"
        ),
        "senselab_core.runtime.faults.is_transient": (
            "fault classification of storage errors (route-b summary, hooks column)"
        ),
        "senselab_core.runtime.fetch": ("deferred import; both import this module (route-b summary, result store row)"),
        "senselab_core.runtime.profiles.ExecutionProfile": (
            "policy carrier; passed as hooks when the module moves (route-b summary, hooks column)"
        ),
        "senselab_core.runtime.store": ("deferred import; both import this module (route-b summary, result store row)"),
        "senselab_worker.ownership.UUID_HEX": ("stays in senselab-worker (route-b summary, supervision row)"),
        "senselab_worker.ownership.remove_owned": ("stays in senselab-worker (route-b summary, supervision row)"),
    },
    "execution/queue.py": {
        "senselab_core.runtime.calibration.HardwareClass": (
            "item weight and placement hook of the pool (route-b summary, queue row)"
        ),
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b summary, lease row)"
        ),
        "senselab_core.runtime.clock.SystemClock": ("default of the Clock seam (route-b summary, lease row)"),
        "senselab_core.runtime.execution.epoch.Epoch": (
            "same move group: epoch.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.execution.epoch.run_lease": (
            "same move group: epoch.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.execution.executor.CorruptItem": (
            "queue item and fault types; opaque payload at the move (route-b summary, queue row)"
        ),
        "senselab_core.runtime.execution.executor.FaultRecord": (
            "queue item and fault types; opaque payload at the move (route-b summary, queue row)"
        ),
        "senselab_core.runtime.execution.executor.MissingItem": (
            "queue item and fault types; opaque payload at the move (route-b summary, queue row)"
        ),
        "senselab_core.runtime.execution.executor.PlanFragment": (
            "queue item and fault types; opaque payload at the move (route-b summary, queue row)"
        ),
        "senselab_core.runtime.execution_settings.JobLease": (
            "lease timing; becomes pydra LeaseSettings (route-b summary, lease row)"
        ),
        "senselab_core.runtime.locks.BrokenLease": (
            "same move group: locks.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.locks.Lease": (
            "same move group: locks.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.locks.LeaseState": (
            "same move group: locks.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.locks.LeaseTimeout": (
            "same move group: locks.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.profiles.ExecutionProfile": (
            "policy carrier; passed as hooks when the module moves (route-b summary, hooks column)"
        ),
        "senselab_core.runtime.storage.STAGING_DIR_MODE": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.STAGING_FILE_MODE": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.StorageDir": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.TEMPORARY_NAME": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.WrongKindError": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.temporary_name": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_worker.canonical.jcs_dumps": ("stays in senselab-worker (route-b summary, supervision row)"),
        "senselab_worker.canonical.jcs_loads": ("stays in senselab-worker (route-b summary, supervision row)"),
        "senselab_worker.digest.identity_digest": ("stays in senselab-worker (route-b summary, supervision row)"),
    },
    "execution/epoch.py": {
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b summary, lease row)"
        ),
        "senselab_core.runtime.clock.SystemClock": ("default of the Clock seam (route-b summary, lease row)"),
        "senselab_core.runtime.execution_settings.JobLease": (
            "lease timing; becomes pydra LeaseSettings (route-b summary, lease row)"
        ),
        "senselab_core.runtime.locks.Lease": (
            "same move group: locks.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.locks.LeaseState": (
            "same move group: locks.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.profiles.ExecutionProfile": (
            "policy carrier; passed as hooks when the module moves (route-b summary, hooks column)"
        ),
        "senselab_core.runtime.storage.StorageDir": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
        "senselab_core.runtime.storage.publish": (
            "same move group: storage.py moves with this module (route-b section 4, M1 and M4)"
        ),
    },
    "execution/supervisor.py": {
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b summary, lease row)"
        ),
        "senselab_core.runtime.clock.SystemClock": ("default of the Clock seam (route-b summary, lease row)"),
        "senselab_core.runtime.execution_settings.SupervisorSettings": (
            "shutdown graces; hook of the G2 Venv runner (route-b summary, supervision row)"
        ),
        "senselab_core.runtime.hardware.proc_stats": (
            "process statistics helper for the supervisor (route-b summary, supervision row)"
        ),
        "senselab_core.runtime.profiles.ExecutionProfile": (
            "policy carrier; passed as hooks when the module moves (route-b summary, hooks column)"
        ),
    },
    "worker_pipes.py": {
        "senselab_core.runtime.clock.Clock": (
            "time seam; pydra takes a clock through its settings (route-b summary, lease row)"
        ),
        "senselab_worker.codec.Deadline": ("stays in senselab-worker (route-b summary, supervision row)"),
    },
}
"""module -> imported senselab name -> why it is there."""


def senselab_imports(source: str) -> set[str]:
    """Return every senselab name imported anywhere in `source`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0].startswith("senselab"):
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names if alias.name.split(".")[0].startswith("senselab")}
    return names


def new_imports(source: str, module: str) -> set[str]:
    """Return the senselab names `source` imports that `BASELINE[module]` does not list."""
    return senselab_imports(source) - BASELINE[module].keys()


@pytest.mark.parametrize("module", sorted(BASELINE))
def test_module_imports_nothing_new_from_senselab(module: str) -> None:
    """The module's senselab imports are all in its baseline, and the baseline lists no import that is gone."""
    path = RUNTIME / module
    assert path.is_file(), f"{module} moved or was renamed: update BASELINE"
    imported = senselab_imports(path.read_text(encoding="utf-8"))
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
