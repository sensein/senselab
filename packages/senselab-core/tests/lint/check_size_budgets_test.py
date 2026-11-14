"""The size budgets: file and function length limits from a data file."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_size_budgets.py"
_spec = importlib.util.spec_from_file_location("check_size_budgets", _SCRIPT)
assert _spec and _spec.loader
sizes = importlib.util.module_from_spec(_spec)
sys.modules["check_size_budgets"] = sizes
_spec.loader.exec_module(sizes)

BUDGET = sizes.load_budget()
SRC = "packages/a/src/a/m.py"
TEST = "packages/a/tests/m_test.py"


def _violations(source: str, path: str, budget: object = BUDGET) -> list[str]:
    file, functions = sizes.measure(path, source)
    found: list[str] = sizes.violations([file], functions, budget)
    return found


def _body(lines: int) -> str:
    return "def f() -> None:\n" + "    pass\n" * (lines - 1)


def test_the_tree_passes_its_budget() -> None:
    """The script exits 0 over the committed index."""
    done = subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize(("path", "key"), [(SRC, "src_function_max_lines"), (TEST, "test_function_max_lines")])
def test_a_function_one_line_over_its_limit_is_named(path: str, key: str) -> None:
    """A function at the limit passes; one line more is reported with its name and length."""
    limit = BUDGET.limits[key]
    assert _violations(_body(limit), path) == []
    (message,) = _violations(_body(limit + 1), path)
    assert f"{path}::f" in message and str(limit + 1) in message


@pytest.mark.parametrize(("path", "key"), [(SRC, "src_file_max_lines"), (TEST, "test_file_max_lines")])
def test_a_file_one_line_over_its_limit_is_named(path: str, key: str) -> None:
    """A file at the limit passes; one line more is reported."""
    limit = BUDGET.limits[key]
    assert _violations("x = 1\n" * limit, path) == []
    (message,) = _violations("x = 1\n" * (limit + 1), path)
    assert path in message


def test_nested_and_async_functions_are_measured() -> None:
    """A long function nested in another, and an async one, count."""
    limit = BUDGET.limits["src_function_max_lines"]
    nested = "def outer() -> None:\n" + "".join("    " + line + "\n" for line in _body(limit + 1).splitlines())
    assert any("::f" in v for v in _violations(nested, SRC))
    assert _violations("async " + _body(limit + 1), SRC)


def test_an_exempt_function_is_allowed_over_the_limit() -> None:
    """A function named in `exempt` is not reported; the same name in another file is."""
    limit = BUDGET.limits["src_function_max_lines"]
    exempt = sizes.Budget(BUDGET.limits, frozenset({(SRC, "f")}))
    assert _violations(_body(limit + 1), SRC, exempt) == []
    assert _violations(_body(limit + 1), "packages/a/src/a/other.py", exempt)


def test_a_bad_limit_or_an_unreasoned_exemption_is_refused(tmp_path: Path) -> None:
    """A limit that is not a positive integer, or an exemption without a reason, is refused."""
    text = _SCRIPT.parent.joinpath("data", "size_budget.yaml").read_text()
    bare = tmp_path / "bare.yaml"
    bare.write_text(text.replace("src_file_max_lines: 1200", "src_file_max_lines: {value: 1200}", 1))
    with pytest.raises(ValueError, match="src_file_max_lines"):
        sizes.load_budget(bare)
    unreasoned = tmp_path / "unreasoned.yaml"
    unreasoned.write_text(text.replace("reason:", "why:"))
    with pytest.raises(ValueError, match="reason"):
        sizes.load_budget(unreasoned)


def test_a_test_directory_or_conftest_is_a_test_file() -> None:
    """Files under `tests/` and `conftest.py` use the test limits; the rest, the source limits."""
    assert sizes.kind("packages/a/tests/x/y.py") == "test"
    assert sizes.kind("packages/a/src/a/conftest.py") == "test"
    assert sizes.kind("packages/a/src/a/x.py") == "src"
