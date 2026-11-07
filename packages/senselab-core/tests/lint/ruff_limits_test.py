"""The ruff limits on package source, scripts and benchmarks: statements per function, literals in comparisons."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

REPO = Path(__file__).resolve().parents[4]
WHERE = ["packages/senselab-core/src/senselab_core/probe.py", "benchmarks/probe.py"]


def _codes(source: str, where: str) -> set[str]:
    ruff = shutil.which("ruff")
    assert ruff, "ruff is not on PATH"
    args = [ruff, "check", "--no-cache", "--output-format", "concise", "--stdin-filename", where, "-"]
    done = subprocess.run(args, input=source, capture_output=True, text=True, cwd=REPO, check=False)
    return {word for line in done.stdout.splitlines() for word in line.split() if word[:2] in ("PL", "C9")}


def _function_of(statements: int) -> str:
    """A function of `statements` statements, its docstring included."""
    body = "".join(f"    x{i} = {i}\n" for i in range(statements - 1))
    return f'"""M."""\n\n\ndef f() -> None:\n    """F."""\n{body}'


@pytest.mark.parametrize("where", WHERE)
def test_a_function_over_fifty_statements_fails(where: str) -> None:
    """Fifty statements pass and fifty-one fail `PLR0915`."""
    assert "PLR0915" not in _codes(_function_of(50), where)
    assert "PLR0915" in _codes(_function_of(51), where)


@pytest.mark.parametrize("where", WHERE)
def test_a_literal_in_a_comparison_fails(where: str) -> None:
    """`return x > 3` fails `PLR2004`."""
    assert "PLR2004" in _codes('"""M."""\n\n\ndef f(x: int) -> bool:\n    """F."""\n    return x > 3\n', where)
