"""The 2.x pull-request template carries the per-commit checklist of the engineering standards, word for word."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

REPO = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("check_pr_template", REPO / "scripts" / "check_pr_template.py")
assert _spec and _spec.loader
tpl = importlib.util.module_from_spec(_spec)
sys.modules["check_pr_template"] = tpl
_spec.loader.exec_module(tpl)

STANDARDS = "# E\n\n## Per-commit checklist\n\nIntro.\n\n- [ ] One (P-1).\n- [ ] Two,\n  wrapped (P-2).\n\n## Next\n"
TEMPLATE = (
    "## Models\n- a\n\n## Per-commit checklist\n- [ ] One (P-1).\n- [ ] Two, wrapped (P-2).\n\n## Gates\n- [ ] g\n"
)


def test_the_tree_template_matches_engineering_md() -> None:
    """The tracked template's checklist equals engineering.md's, and it has at least the nine original items."""
    assert tpl.main([]) == 0
    assert len(tpl.checklist()) >= 9


@pytest.mark.parametrize(
    ("template", "standards"),
    [
        (TEMPLATE.replace("One (P-1)", "One (P-9)"), STANDARDS),
        (TEMPLATE, STANDARDS.replace("wrapped (P-2)", "wrapped (P-3)")),
        (TEMPLATE.replace("- [ ] One (P-1).\n", ""), STANDARDS),
        (TEMPLATE, STANDARDS.replace("- [ ] One (P-1).\n", "- [ ] One (P-1).\n- [ ] Three (P-3).\n")),
    ],
)
def test_editing_one_line_of_either_file_fails(template: str, standards: str) -> None:
    """A changed, dropped or added line in the template or in engineering.md is reported."""
    assert not tpl.problems(TEMPLATE, STANDARDS)
    assert tpl.problems(template, standards)


def test_write_regenerates_the_checklist_and_nothing_else() -> None:
    """Regenerating replaces the checklist items verbatim and keeps the other sections."""
    stale = TEMPLATE.replace("- [ ] Two, wrapped (P-2).\n", "- [ ] Old (P-0).\n")
    fresh = tpl.regenerate(stale, STANDARDS)
    assert not tpl.problems(fresh, STANDARDS)
    assert fresh.startswith("## Models\n- a\n") and fresh.endswith("## Gates\n- [ ] g\n")


def test_the_default_template_stays_the_one_for_1x() -> None:
    """The default template is not the 2.x one: a 1.x PR keeps its own checklist."""
    default = (REPO / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
    assert "Per-commit checklist" not in default
