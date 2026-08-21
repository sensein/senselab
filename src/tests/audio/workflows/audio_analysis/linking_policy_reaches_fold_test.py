"""``fuse_consensus_words`` must be called with the run's policy, not left on its defaults.

The fold takes ``policy=None`` and falls back to its own ``0.3`` / ``0.15``. Those happen to equal
the packaged ``linking.asr_slot_overlap`` / ``asr_slot_mid_tol_s``, so a dropped ``policy=`` changes
no output today and no test caught it (F-162). It shows up the first time someone edits the config
and nothing happens -- and worse, the run's provenance then records the value that actually ran
while the config records a different one, so the artifact and the config disagree about the same
decision.

A behavioural test cannot see this while the two values coincide, and making them differ in a
fixture would test the fixture rather than the wiring. So this is a call-site guard, the same shape
as ``revision_pinning_guard_test.py``: a new call that drops ``policy=`` fails here until it is
either fixed or explicitly allowlisted below with its reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[5] / "src" / "senselab" / "audio" / "workflows" / "audio_analysis"

# Call sites that legitimately cannot pass ``policy=`` yet, each with why.
#
# Both sit inside ``asr.py`` itself, on the ``fused is None`` branch of a harvester whose signature
# has no ``policy`` parameter to forward. ``compute.harvest_pass`` always supplies ``fused=``, so the
# branch is dead on the path the triage graph lifts -- but it is *not* dead in the refiner:
# ``adaptive/interventions.py:595`` calls ``harvest_asr_votes`` with no ``fused=``, so an escalation
# round re-folds the words with the policy dropped. That is a second instance of F-162 that the
# register does not name, in refiner-only code the triage graph never imports. Closing it means
# threading a policy through the adaptive loop's intervention context, which is its own change with
# its own review -- recorded here rather than fixed silently or forgotten.
KNOWN_UNPOLICIED = {
    ("asr.py", "_consensus_word_doubt"),
    ("asr.py", "harvest_asr_votes"),
}


def _enclosing_function(tree: ast.Module, target: ast.AST) -> str:
    """Name of the innermost function containing ``target``, or ``"<module>"``."""
    best = "<module>"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is target:
                    best = node.name
    return best


def _unpolicied_call_sites() -> set[tuple[str, str]]:
    """``(filename, enclosing function)`` for every ``fuse_consensus_words`` call lacking ``policy=``."""
    found: set[tuple[str, str]] = set()
    for path in _PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "fuse_consensus_words":
                continue
            if any(kw.arg == "policy" for kw in node.keywords):
                continue
            found.add((path.name, _enclosing_function(tree, node)))
    return found


def test_every_fold_call_supplies_the_run_policy() -> None:
    """A dropped ``policy=`` makes ``linking:`` decorative on that path -- config that cannot act."""
    offenders = sorted(_unpolicied_call_sites() - KNOWN_UNPOLICIED)
    assert not offenders, (
        f"fuse_consensus_words called without policy= at {offenders}. The linking: config cannot "
        "reach the word-slot join from there. Pass the run's policy, or allowlist it with a reason."
    )


def test_the_allowlist_has_no_stale_entries() -> None:
    """An allowlisted site that gained its ``policy=`` should leave the list, not sit there unread."""
    stale = sorted(KNOWN_UNPOLICIED - _unpolicied_call_sites())
    assert not stale, f"KNOWN_UNPOLICIED entr(ies) no longer drop policy=, remove them: {stale}"
