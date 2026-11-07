"""The ruff limits on package source: statements per function and literals in comparisons."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

PACKAGES = Path(__file__).resolve().parents[3]
SOURCE = PACKAGES / "senselab-core" / "src" / "senselab_core" / "probe.py"


def _codes(source: str) -> set[str]:
    ruff = shutil.which("ruff")
    assert ruff, "ruff is not on PATH"
    args = [ruff, "check", "--no-cache", "--output-format", "concise", "--stdin-filename", str(SOURCE), "-"]
    done = subprocess.run(args, input=source, capture_output=True, text=True, cwd=PACKAGES, check=False)
    return {word for line in done.stdout.splitlines() for word in line.split() if word[:2] in ("PL", "C9")}


def _function(statements: int) -> str:
    body = "".join(f"    x{i} = {i}\n" for i in range(statements))
    return f'"""M."""\n\n\ndef f() -> None:\n    """F."""\n{body}'


def test_a_function_over_fifty_statements_fails() -> None:
    """Fifty statements pass and fifty-one fail `PLR0915`."""
    assert "PLR0915" not in _codes(_function(49))
    assert "PLR0915" in _codes(_function(50))


def test_a_literal_in_a_comparison_fails() -> None:
    """`if x > 3:` in package source fails `PLR2004`."""
    assert "PLR2004" in _codes('"""M."""\n\n\ndef f(x: int) -> bool:\n    """F."""\n    return x > 3\n')
