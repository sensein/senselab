"""Package sources never build a model without validation."""

import ast
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[3]
SOURCES = sorted(PACKAGES.glob("*/src/**/*.py"))
ALLOWED = {"senselab_core/testing/values.py": "construct_refused calls it to assert that it raises"}
"""Source files that may call `model_construct`, each with the reason."""


def construct_calls(source: str, module: str) -> list[str]:
    """Return `module:line` for every call of an attribute named `model_construct` outside `ALLOWED`."""
    if any(module.endswith(where) for where in ALLOWED):
        return []
    return [
        f"{module}:{node.lineno}"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "model_construct"
    ]


@pytest.mark.logic
def test_construct_calls_are_found() -> None:
    """The sweep sees a call through any receiver."""
    assert construct_calls("Span.model_construct(a=1)\nx.y.model_construct()\n", "m") == ["m:1", "m:2"]


@pytest.mark.logic
def test_no_model_construct_in_package_sources() -> None:
    """No package source calls `model_construct`, which skips every validator."""
    assert SOURCES, "no package sources found"
    found = [hit for path in SOURCES for hit in construct_calls(path.read_text(encoding="utf-8"), str(path))]
    assert not found, f"model_construct skips validation: {found}"
