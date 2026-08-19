"""The generated dependency table and the hand-maintained compatibility matrix are two documents.

They were one path. ``scripts/generate-compat-matrix.py`` wrote its per-function table over
``docs/compatibility-matrix.md``, which is a hand-maintained Python/dependency-version document,
so running the generator replaced one document with an unrelated one.

Reasoning: ``specs/20260819-131500-compat-matrix-generator/design.md``.
"""

from __future__ import annotations

from pathlib import Path

from senselab.utils.compatibility import GENERATED_DOC, generate_matrix_markdown


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_the_generated_document_in_the_tree_is_what_the_generator_produces() -> None:
    """Regenerating must be a no-op, so the table cannot drift from the matrix it describes."""
    committed = _repo_root() / GENERATED_DOC
    assert committed.is_file(), f"{GENERATED_DOC} is missing; run scripts/generate-compat-matrix.py"
    assert committed.read_text() == generate_matrix_markdown(), (
        f"{GENERATED_DOC} is stale. Run `uv run python scripts/generate-compat-matrix.py` "
        "and commit the result -- do not hand-edit it."
    )


def test_the_generator_does_not_write_over_the_hand_maintained_matrix() -> None:
    """The version matrix records bounds verified by pinning, which no generator can reproduce."""
    assert GENERATED_DOC != "docs/compatibility-matrix.md"

    hand_maintained = _repo_root() / "docs" / "compatibility-matrix.md"
    if hand_maintained.is_file():
        assert hand_maintained.read_text() != generate_matrix_markdown()


def test_the_script_writes_where_the_module_says() -> None:
    """One source of truth for the path, so the script and the test cannot disagree."""
    script = (_repo_root() / "scripts" / "generate-compat-matrix.py").read_text()
    assert "GENERATED_DOC" in script, "the script should import the path, not repeat the literal"
    assert '"docs" / "compatibility-matrix.md"' not in script
