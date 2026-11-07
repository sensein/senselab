"""The brief check: three headings, engineering.md's checklist verbatim, and templates rendered without gaps."""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.logic

REPO = Path(__file__).resolve().parents[4]
TEMPLATES = sorted((REPO / "scripts" / "data" / "briefs").glob("*.md"))


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("check_pr_template")
cb = _load("check_brief")
REQUIRED = ["One (P-1)."]
BRIEF = "# B\n\n## Principles\n\nP.\n\n## Checklist\n\n- [ ] One (P-1).\n\n## Files\n\n- f\n"


def test_a_brief_with_the_three_headings_and_the_checklist_passes() -> None:
    """The minimal brief passes."""
    assert cb.problems(BRIEF, REQUIRED) == []


@pytest.mark.parametrize("heading", cb.HEADINGS)
def test_a_missing_heading_is_refused(heading: str) -> None:
    """A brief without one of the three headings fails, naming it."""
    assert f"missing heading {heading!r}" in cb.problems(BRIEF.replace(heading + "\n", "\n"), REQUIRED)


def test_a_checklist_that_differs_from_engineering_md_is_refused() -> None:
    """A reworded or extra checklist item fails."""
    assert cb.problems(BRIEF.replace("One (P-1)", "One (P-2)"), REQUIRED)
    assert cb.problems(BRIEF.replace("- [ ] One (P-1).\n", "- [ ] One (P-1).\n- [ ] Two.\n"), REQUIRED)


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.stem)
def test_each_template_carries_engineering_mds_checklist(template: Path) -> None:
    """The tracked templates pass the check against the tree's engineering.md."""
    assert cb.problems(template.read_text(), cb.checklist()) == []


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.stem)
def test_a_template_renders_only_with_every_placeholder(template: Path) -> None:
    """Rendering fills every placeholder and refuses a missing one."""
    text = template.read_text()
    names = set(re.findall(r"\$\{(\w+)\}", text))
    assert names
    assert "${" not in cb.render(text, {n: "x" for n in names})
    with pytest.raises(KeyError):
        cb.render(text, {n: "x" for n in sorted(names)[1:]})
