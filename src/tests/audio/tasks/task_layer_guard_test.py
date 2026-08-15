"""Nothing under audio/tasks/ may import from audio/workflows/, with one documented exception.

Workflows compose tasks; a task importing a workflow inverts that. The rule matters here because
this change promotes two primitives *down* from the workflow into the task layer, and without a
guard they drift back the first time someone needs a workflow helper in a task.

An AST sweep rather than a text search: an import inside a function body or guarded by
TYPE_CHECKING is still an import, and a commented-out one is not.

One file already violates the rule and predates this branch (confirmed against
``origin/alpha``), so a bare assertion here would make this test's own addition the thing that
breaks CI for debt this branch did not create. ``KNOWN_LAYERING_VIOLATIONS`` documents that
exception the same way ``hf_load_coverage_test.py`` and ``revision_pinning_guard_test.py``
document theirs: an explicit, reviewed set rather than a silent pass, with a companion test
that fails the moment an entry stops describing reality.
"""

import ast
from pathlib import Path

_SRC = Path("src/senselab")
_TASKS = _SRC / "audio/tasks"
_FORBIDDEN_PREFIX = "senselab.audio.workflows"

# Per-file exceptions to the audio/tasks -> audio/workflows layering rule, tolerated for now
# rather than fixed here. A new violation must not be added to this set to make the guard
# pass -- it must be reviewed and justified the way the entry below was.
#
# audio/tasks/speech_to_text_ensemble/api.py imports one constant, MIN_EVIDENCE_WEIGHT, from
# the leaf module audio/workflows/audio_analysis/floors.py -- a model-independent ROVER-fusion
# corroboration floor with no home outside that workflow module yet. Predates this branch
# (present on origin/alpha before the windowing-primitives move this test file was added for).
# Moving `floors.py`, or just the constant, down a layer is a separate change with its own
# justification -- restructuring speech_to_text_ensemble is not what this task is for.
KNOWN_LAYERING_VIOLATIONS = {
    "audio/tasks/speech_to_text_ensemble/api.py",
}


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
    """The dependency direction is one-way, and this is what keeps it that way.

    Skips files in ``KNOWN_LAYERING_VIOLATIONS`` -- the *only* files exempt, and only because
    they are individually reviewed and recorded above, not because the rule bends for them.
    """
    offenders: list[str] = []
    for path in sorted(_TASKS.rglob("*.py")):
        if str(path.relative_to(_SRC)) in KNOWN_LAYERING_VIOLATIONS:
            continue
        for module in _imported_modules(path):
            if module.startswith(_FORBIDDEN_PREFIX):
                offenders.append(f"{path}: imports {module}")
    assert not offenders, "audio/tasks must not import audio/workflows:\n" + "\n".join(offenders)


def test_known_layering_violations_are_not_stale() -> None:
    """Every allowlist entry must still exist and must still actually violate the rule.

    Mirrors the stale-entry checks in ``hf_load_coverage_test.py`` and
    ``revision_pinning_guard_test.py``: an allowlist that is never pruned silently stops
    meaning "reviewed, tolerated debt" and starts meaning "nobody looked since." Two ways an
    entry goes stale -- the file it names was moved or deleted, or the file was fixed and no
    longer imports anything under ``audio/workflows`` -- and either would let the allowlist
    keep exempting a file the *general* test above would otherwise cover again, which is
    exactly the gap that lets a genuinely new violation hide behind an old, no-longer-true one.
    """
    missing = [f for f in sorted(KNOWN_LAYERING_VIOLATIONS) if not (_SRC / f).is_file()]
    assert not missing, "allowlist entr(ies) name files that no longer exist:\n" + "\n".join(f"  {f}" for f in missing)

    fixed = [
        relpath
        for relpath in sorted(KNOWN_LAYERING_VIOLATIONS)
        if not any(m.startswith(_FORBIDDEN_PREFIX) for m in _imported_modules(_SRC / relpath))
    ]
    assert not fixed, (
        "allowlist entr(ies) no longer import audio/workflows -- remove them from "
        "KNOWN_LAYERING_VIOLATIONS so the general guard covers the file again:\n" + "\n".join(f"  {f}" for f in fixed)
    )


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
