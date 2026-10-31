"""Phase 1 adversarial findings with no regression test elsewhere, rebuilt from the report's break column.

Rows A-26, A-27, A-28 and A-33 are guards over the repository itself. Each probe plants the offending input
and asserts the guard refuses it. The last test fails if any report row has no ``test_a<n>_*`` under ``packages/``.
"""

import base64
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[4]
PACKAGES = REPO / "packages"
REPORT = REPO / "specs" / "20261030-140022-senselab-2x-api" / "reviews" / "phase1-adversarial.md"
BUDGET = REPO / "scripts" / "check_fixture_budget.py"
GUARD = REPO / "src" / "tests" / "utils" / "moved_modules_test.py"
DEFERRED = frozenset({15, 30})
OWNER_ACTION = frozenset({39})


def _load(path: Path, name: str) -> ModuleType:
    """Import the module at ``path`` under ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _porcelain() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def worktree_untouched() -> Iterator[None]:
    """Fail the test if it changes anything ``git status`` can see."""
    before = _porcelain()
    yield
    assert _porcelain() == before, "the test wrote under the worktree"


def _pytest(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _guard_tree(base: Path, plant: dict[str, str]) -> Path:
    """A miniature repository holding the moved-module guard, the budget script it loads, and ``plant``."""
    for rel, source in {
        "src/tests/utils/moved_modules_test.py": GUARD,
        "scripts/check_fixture_budget.py": BUDGET,
    }.items():
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, base / rel)
    for pkg in ("senselab", "senselab/utils", "senselab/audio"):
        (base / "src" / pkg).mkdir(parents=True, exist_ok=True)
        (base / "src" / pkg / "__init__.py").write_text("")
    for rel, text in plant.items():
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text(text)
    return base / "src" / "tests" / "utils" / "moved_modules_test.py"


GUARD_SELECTION = "test_no_moved_module_paths_remain or test_no_file_imports_a_moved_module or test_tombstones_hold"

BYPASSES = {
    "relative-import": {"src/senselab/audio/bypass.py": "from ..utils import file_lock\n"},
    "importlib-concat": {
        "src/senselab/audio/bypass.py": (
            "import importlib\nimportlib.import_module('senselab.utils.' + 'file_' + 'lock')\n"
        )
    },
    "shim-at-old-path": {"src/senselab/utils/file_lock.py": "from senselab_core.runtime.locks import *\n"},
}


@pytest.mark.contract
def test_moved_module_guard_passes_a_clean_tree(tmp_path: Path) -> None:
    """Control: the guard, run on a tree with no reference to a moved module, passes."""
    guard = _guard_tree(tmp_path, {})
    result = _pytest(["-q", "--rootdir", str(tmp_path), "-k", GUARD_SELECTION, str(guard)], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.contract
@pytest.mark.parametrize("bypass", sorted(BYPASSES))
def test_a26_moved_module_guard_catches_the_bypass(tmp_path: Path, bypass: str, worktree_untouched: None) -> None:
    """A relative import, a concatenated ``importlib`` argument and a shim at the old path each fail the guard."""
    guard = _guard_tree(tmp_path, BYPASSES[bypass])
    result = _pytest(["-q", "--rootdir", str(tmp_path), "-k", GUARD_SELECTION, str(guard)], tmp_path)
    assert result.returncode == 1, f"{bypass} passed the guard:\n{result.stdout}{result.stderr}"


@pytest.mark.logic
def test_a27_fixture_budget_refuses_a_symlink_and_a_base64_code_file(tmp_path: Path, worktree_untouched: None) -> None:
    """A tracked symlink is a violation, and a 5 MB base64 ``.py`` under a test tree exceeds the code-file cap."""
    budget = _load(BUDGET, "a27_check_fixture_budget")
    tests = tmp_path / "src" / "tests"
    tests.mkdir(parents=True)
    target = tests / "real.txt"
    target.write_text("x")
    link = tests / "link.txt"
    link.symlink_to(target)
    assert any("symlink" in v for v in budget.check([link], allow={}, root=tmp_path))

    blob = tests / "payload.py"
    blob.write_text("DATA = '" + base64.b64encode(os.urandom(3_750_000)).decode() + "'\n")
    assert blob.stat().st_size > 5_000_000
    policy = budget.load_policy()
    violations = budget.check(
        [blob], allow={}, root=tmp_path, code_max=policy.code_file_max_bytes, code_total=policy.code_total_max_bytes
    )
    assert any("payload.py" in v for v in violations)


@pytest.mark.contract
def test_a28_untiered_test_under_packages_fails_collection(pytester: pytest.Pytester, worktree_untouched: None) -> None:
    """The collection hook of ``packages/`` refuses a test with no tier marker, which CI's ``-m`` would skip."""
    hook = PACKAGES / "conftest.py"
    pytester.makeini("[pytest]\nmarkers =\n    logic\n    contract\n    limits\n    usecase\n    env\n")
    pytester.mkdir("packages")
    (pytester.path / "packages" / "conftest.py").write_text(hook.read_text() if hook.is_file() else "")
    (pytester.path / "packages" / "untiered_test.py").write_text("def test_never_runs_in_ci():\n    pass\n")
    result = pytester.runpytest("packages/untiered_test.py")
    assert result.ret == pytest.ExitCode.USAGE_ERROR, result.stdout.str()
    assert "test_never_runs_in_ci" in result.stderr.str()


@pytest.mark.contract
def test_a33_undeclared_external_import_in_a_package_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worktree_untouched: None
) -> None:
    """A package module importing a distribution it does not declare is reported by the dependency contract."""
    guard = _load(PACKAGES / "senselab-core" / "tests" / "lint" / "dependencies_test.py", "a33_dependencies_test")
    pkg = tmp_path / "a33pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("import os\nimport librosa\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    graph = guard.grimp.build_graph("a33pkg", include_external_packages=True)
    bad = guard.undeclared_by_module(guard.external_imports(graph, {"a33pkg"}), set(), {})
    assert bad == {"a33pkg.mod": {"librosa"}}


def _report_rows() -> set[int]:
    text = REPORT.read_text()
    table = {int(m) for m in re.findall(r"^\| A-(\d+) \|", text, flags=re.M)}
    bullets = {int(m) for m in re.findall(r"^- \*\*A-(\d+)\.\*\*", text, flags=re.M)}
    return table | bullets


def _tested_rows() -> set[int]:
    found: set[int] = set()
    for path in PACKAGES.rglob("*_test.py"):
        found |= {int(n) for n in re.findall(r"^\s*def test_a(\d+)_", path.read_text(), flags=re.M)}
    return found


@pytest.mark.logic
def test_every_adversarial_row_has_a_regression_test() -> None:
    """Each report row A-1..A-39, bar the deferred and owner-action ones, has a ``test_a<n>_*`` under packages."""
    rows = _report_rows()
    assert rows >= set(range(1, 40)), f"report rows missing: {sorted(set(range(1, 40)) - rows)}"
    missing = sorted(rows - DEFERRED - OWNER_ACTION - _tested_rows())
    assert missing == [], f"rows with no test_a<n>_* regression: {missing}"
