"""Nothing under audio/tasks/ may import from audio/workflows/.

Workflows compose tasks; a task importing a workflow inverts that. The rule matters here because
this change promotes two primitives *down* from the workflow into the task layer, and without a
guard they drift back the first time someone needs a workflow helper in a task.

An AST sweep rather than a text search: an import inside a function body or guarded by
TYPE_CHECKING is still an import, and a commented-out one is not.
"""

import ast
from pathlib import Path

_TASKS = Path("src/senselab/audio/tasks")
_FORBIDDEN_PREFIX = "senselab.audio.workflows"


def _imported_modules(path: Path) -> list[str]:
    """Every module name imported anywhere in a file, including inside function bodies."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def test_no_task_imports_a_workflow() -> None:
    """The dependency direction is one-way, and this is what keeps it that way."""
    offenders: list[str] = []
    for path in sorted(_TASKS.rglob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(_FORBIDDEN_PREFIX):
                offenders.append(f"{path}: imports {module}")
    assert not offenders, "audio/tasks must not import audio/workflows:\n" + "\n".join(offenders)


def test_the_guard_can_actually_see_a_violation() -> None:
    """A guard that cannot fail is not a guard.

    Proves the AST sweep detects the pattern it is meant to catch, including an import nested
    inside a function, which a naive top-of-file scan would miss.
    """
    import tempfile

    source = "def f():\n    from senselab.audio.workflows.audio_analysis import embeddings\n    return embeddings\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "offender.py"
        p.write_text(source, encoding="utf-8")
        assert any(m.startswith(_FORBIDDEN_PREFIX) for m in _imported_modules(p))
