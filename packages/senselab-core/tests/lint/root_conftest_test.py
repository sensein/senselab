"""``packages/conftest.py`` imports at module level only the standard library and pytest.

It loads in every package's test run, the worker floor's included.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT_CONFTEST = Path(__file__).resolve().parents[3] / "conftest.py"
ALLOWED = frozenset({*sys.stdlib_module_names, "pytest", "_pytest"})


def top_level_imports(source: str) -> set[str]:
    """The top-level package of every import statement in the module body, outside any function or class."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.logic
def test_the_root_conftest_imports_only_the_standard_library_and_pytest_at_module_level() -> None:
    """A third-party import in the root conftest fails collection of every test where that package is absent."""
    assert top_level_imports(ROOT_CONFTEST.read_text()) - ALLOWED == set()


@pytest.mark.logic
def test_a_third_party_import_at_module_level_is_found_and_one_in_a_function_is_not() -> None:
    """The check reads the module body; an import inside a fixture is that fixture's dependency."""
    planted = "import json\nimport pytest\nimport yaml\nfrom numpy import array\n\ndef f():\n    import torch\n"
    assert top_level_imports(planted) - ALLOWED == {"yaml", "numpy"}
