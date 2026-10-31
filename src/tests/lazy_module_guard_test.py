"""Collection survives an unimportable lazy module in ``sys.modules`` when a hypothesis test is collected."""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from tests.lazy_module_guard_fixture import prune_unimportable_lazy_modules

SRC = str(Path(__file__).resolve().parents[1])

PLANT = """
import sys
import types


class Unimportable(types.ModuleType):
    def __getattr__(self, name):
        raise ImportError("planted: optional backend not installed")


sys.modules["planted_lazy_backend"] = Unimportable("planted_lazy_backend")
"""

HOOK = """
import pytest
from tests.lazy_module_guard_fixture import prune_unimportable_lazy_modules


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems():
    prune_unimportable_lazy_modules()
"""

HYPOTHESIS_TEST = """
from hypothesis import given, strategies as st


@given(st.integers())
def test_uses_hypothesis(n):
    assert n == n
"""


def _collect(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *, with_hook: bool
) -> subprocess.CompletedProcess:
    monkeypatch.setenv("PYTHONPATH", SRC)
    pytester.makeconftest(PLANT + (HOOK if with_hook else ""))
    pytester.makepyfile(test_planted=HYPOTHESIS_TEST)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(pytester.path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=pytester.path,
    )


@pytest.mark.contract
def test_x9collect_hook_makes_collection_succeed(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the pruning hook a planted unimportable lazy module no longer aborts collection."""
    result = _collect(pytester, monkeypatch, with_hook=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_uses_hypothesis" in result.stdout


@pytest.mark.contract
def test_x9collect_without_hook_collection_crashes(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the hook the same tree aborts with INTERNALERROR (exit code 3)."""
    result = _collect(pytester, monkeypatch, with_hook=False)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "INTERNALERROR" in result.stdout + result.stderr
    assert "planted: optional backend not installed" in result.stdout + result.stderr


@pytest.mark.logic
def test_x9collect_prune_leaves_real_and_loaded_modules() -> None:
    """Plain modules, importable lazy modules and None entries survive; only the unimportable one goes."""

    class Unimportable(types.ModuleType):
        def __getattr__(self, name: str) -> object:
            raise ImportError(name)

    class Importable(types.ModuleType):
        def __getattr__(self, name: str) -> object:
            raise AttributeError(name)

    plain = types.ModuleType("x9_plain")
    keep = Importable("x9_importable")
    drop = Unimportable("x9_unimportable")
    saved = dict(sys.modules)
    try:
        sys.modules.update({"x9_plain": plain, "x9_importable": keep, "x9_unimportable": drop, "x9_none": None})  # type: ignore[dict-item]
        removed = prune_unimportable_lazy_modules()
        assert removed == ["x9_unimportable"]
        assert sys.modules["x9_plain"] is plain
        assert sys.modules["x9_importable"] is keep
        assert "x9_none" in sys.modules
        assert "x9_unimportable" not in sys.modules
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
