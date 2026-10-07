"""The hand-maintained compatibility document must be tracked, and outside the docs build output.

Two documents once shared one path. ``scripts/generate-compat-matrix.py`` wrote its per-function
table over ``docs/compatibility-matrix.md``, a hand-maintained Python/dependency-version document.
The reason nobody noticed is the second half of the defect: ``docs/`` is pdoc's output directory and
is gitignored, so the hand-maintained file was invisible to ``git status`` and lived where a build
writes.

Reasoning: ``specs/20260819-131500-compat-matrix-generator/design.md``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from senselab.utils.compatibility import GENERATED_DOC, generate_matrix_markdown

# The hand-maintained document: version bounds verified by pinning, which no generator can derive.
HAND_MAINTAINED_DOC = "COMPATIBILITY.md"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_ignored(path: str) -> bool:
    """Reports whether git ignores ``path``.

    Args:
        path: Repo-relative path.

    Returns:
        True when a gitignore rule covers it.
    """
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=_repo_root(),
            capture_output=True,
        ).returncode
        == 0
    )


def test_the_hand_maintained_document_exists_and_is_tracked() -> None:
    """A document only a human can write must be under version control, not build output."""
    doc = _repo_root() / HAND_MAINTAINED_DOC
    assert doc.is_file(), f"{HAND_MAINTAINED_DOC} is missing"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", HAND_MAINTAINED_DOC],
        cwd=_repo_root(),
        capture_output=True,
    )
    assert tracked.returncode == 0, f"{HAND_MAINTAINED_DOC} is not tracked by git"
    assert not _is_ignored(HAND_MAINTAINED_DOC), (
        f"{HAND_MAINTAINED_DOC} sits under a gitignore rule. It is hand-maintained, so it must not "
        "live in a directory a build writes to -- that is how the generator overwrote it unnoticed."
    )


def test_the_generated_table_goes_to_build_output_not_over_the_hand_maintained_document() -> None:
    """The generated table is derived, so it is rebuilt rather than committed."""
    assert GENERATED_DOC != HAND_MAINTAINED_DOC
    assert GENERATED_DOC != "docs/compatibility-matrix.md"
    assert _is_ignored(GENERATED_DOC), (
        f"{GENERATED_DOC} is not ignored. It is regenerated from COMPATIBILITY_MATRIX on every "
        "docs build, so committing it would let a stale copy outlive the matrix it describes."
    )
    assert generate_matrix_markdown() != (_repo_root() / HAND_MAINTAINED_DOC).read_text()


def test_the_docs_build_runs_the_generator() -> None:
    """Since the table is not committed, the build is the only thing that can produce it."""
    for workflow in ("docs.yaml", "docs-preview.yaml"):
        text = (_repo_root() / ".github" / "workflows" / workflow).read_text()
        assert "generate-compat-matrix.py" in text, (
            f"{workflow} publishes docs/ but never generates {GENERATED_DOC}, so the table would "
            "be absent from the published site."
        )


def test_the_script_writes_where_the_module_says() -> None:
    """One source of truth for the path, so the script and the test cannot disagree."""
    script = (_repo_root() / "scripts" / "generate-compat-matrix.py").read_text()
    assert "GENERATED_DOC" in script, "the script should import the path, not repeat the literal"
    assert '"docs" / "compatibility-matrix.md"' not in script
