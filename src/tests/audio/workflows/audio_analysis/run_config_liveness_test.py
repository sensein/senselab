"""Every ``RunConfig`` field must have a reader.

``remediation-config.md`` found four policy fields that ``_build()`` assigned and nothing
consumed. An operator setting such a key gets a changed config hash and no behaviour change,
which is worse than a bare literal because it looks like control. This test fails when a new
one appears.

Detection is AST-based: an attribute access ``.<field>`` or a ``getattr(obj, "<field>")`` call,
found anywhere under ``src/senselab/`` or ``scripts/`` except ``run_config.py`` itself. Excluding
that whole file is sufficient to skip both the field's own declaration and ``_build()``'s
keyword-argument assignment -- neither is an ``ast.Attribute`` node nor a ``getattr`` call, so no
extra special-casing is needed. Comments and docstring prose are invisible to ``ast``, unlike a
plain-text grep, so a field merely *mentioned* in a comment does not count as read.

One blind spot, stated rather than silently accepted: the checker matches the field *name*, not
"``RunConfig``'s ``foo`` specifically" -- it has no type information and cannot tell whose
``.foo`` a given access belongs to. A field whose name coincides with another class's attribute
could therefore read as "read" without any ``RunConfig`` instance ever being touched. None of the
fields below collide with anything else in the tree (checked by hand during Phase 1 Task 2, by
grepping each name across the whole repository); a future field with a very generic name should
get the same check before its "read" verdict is trusted.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from senselab.audio.workflows.audio_analysis.run_config import RunConfig

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRC_SENSELAB = _REPO_ROOT / "src" / "senselab"
_SCRIPTS = _REPO_ROOT / "scripts"
_RUN_CONFIG_FILE = (_SRC_SENSELAB / "audio" / "workflows" / "audio_analysis" / "run_config.py").resolve()

# Fields ``_build()`` assigns that nothing under ``src/senselab/`` or ``scripts/`` reads. Verified
# by hand for Phase 1 Task 2 (2026-08-16), not inherited from the audit unchecked -- see the plan's
# own instruction to resolve the register's speaker_policy discrepancy with evidence rather than
# transcribe it.
#
# The five ``*_policy`` fields (``rounds_policy``, ``speaker_policy``, ``quality_policy``,
# ``labelstudio_policy``, ``support_policy``) are each read exactly once outside this module: by
# ``run_config_test.py``'s ``getattr(cfg, section)`` / ``getattr(cfg, attr)`` (its
# ``MOVED_CONSTANTS`` and ``probes`` dicts name all five), which is real, currently-passing
# regression coverage of the D2 migration -- not a decorative mention. Task 2's own instruction is
# "a field read only by a test still counts as read -- say so and leave it," so none of the five
# are deleted here; deleting any would break that test file, which is not in this task's file list.
#
# This also resolves the register's apparent contradiction. Its summary counts "four dead fields
# plus two orphaned keys inside speaker_policy," while its D3/D4 rows say ``speaker_policy`` itself
# "is built and never read anywhere else in the tree." Both readings under-describe the same fact:
# ``speaker_policy`` has *exactly* the same status as the other four -- no reader outside
# ``run_config.py`` other than this one test file. It is not a fifth, distinctly-dead field, and it
# is not distinctly alive either. The audit's own D14 grep ("confirmed... zero hits") missed this
# because it matched a literal ``.rounds_policy``-shaped pattern, which does not match
# ``getattr(cfg, attr)`` where ``attr`` is a runtime string -- the exact trap this file's docstring
# names for AST-based detection.
KNOWN_UNREAD = {
    "rounds_policy",
    "speaker_policy",
    "quality_policy",
    "labelstudio_policy",
    "support_policy",
    # Discovered during this same verification sweep, not among remediation-config.md's D5-D14/D3-D4
    # candidates Task 2 was scoped to resolve. These six have *zero* readers anywhere in the tree --
    # not even a test -- a stricter kind of dead than the five policy fields above.
    # ``RunConfig.skipped_stages`` (built in the same ``_build()`` from the same ``stages:`` YAML
    # block) is what production actually branches on; these booleans are its unused predecessor.
    # Left here rather than deleted because deleting them was not this task's brief; flagged for its
    # own register finding rather than folded silently into this one.
    "run_diarization",
    "run_ast",
    "run_yamnet",
    "run_features",
    "run_asr",
    "run_alignment",
    # Same dead-predecessor-of-skipped_stages family as the six above, but with one test reader:
    # ``src/tests/scripts/analyze_audio_test.py`` asserts ``cfg.run_comparisons is False``. Same
    # "read only by a test still counts as read" rule as the policy fields; same out-of-scope
    # reasoning as the six above for why it is not deleted by this task.
    "run_comparisons",
}


def _iter_python_files() -> list[Path]:
    """Every ``.py`` file under ``src/senselab/`` or ``scripts/``, excluding ``run_config.py`` itself."""
    files = list(_SRC_SENSELAB.rglob("*.py")) + list(_SCRIPTS.glob("*.py"))
    return [p for p in files if p.resolve() != _RUN_CONFIG_FILE]


def _read_field_names(paths: list[Path]) -> set[str]:
    """Field names obtained via ``.field`` attribute access or ``getattr(obj, "field")`` in ``paths``."""
    found: set[str] = set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                found.add(node.attr)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                found.add(node.args[1].value)
    return found


def test_every_runconfig_field_has_a_reader_or_is_known_unread() -> None:
    """A ``RunConfig`` field with no reader outside ``run_config.py`` is a decision with no effect.

    ``KNOWN_UNREAD`` is the escape hatch for a field Task 2's Step 1 found genuinely unread in
    production but chose not to delete -- see its comment for why each entry is there. A field
    landing here that is in neither the read set nor ``KNOWN_UNREAD`` is new and must be either
    wired to a real call site or deleted, not silently left uncovered.
    """
    read = _read_field_names(_iter_python_files())
    declared = {f.name for f in dataclasses.fields(RunConfig)}
    unread = sorted(declared - read - KNOWN_UNREAD)
    assert not unread, (
        f"RunConfig field(s) with no reader outside run_config.py and not in KNOWN_UNREAD: {unread}. "
        "Either thread it to a real call site or add it to KNOWN_UNREAD with the reason."
    )


def test_known_unread_has_no_stale_entries() -> None:
    """A ``KNOWN_UNREAD`` entry that gained a reader (or lost its field) should be noticed, not kept quietly."""
    read = _read_field_names(_iter_python_files())
    declared = {f.name for f in dataclasses.fields(RunConfig)}
    stale = sorted((KNOWN_UNREAD - declared) | (KNOWN_UNREAD & read))
    assert not stale, f"KNOWN_UNREAD entr(ies) no longer belong here (field gone, or now read): {stale}"
