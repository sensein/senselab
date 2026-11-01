"""The 2.x pull-request template carries the per-commit checklist of the engineering standards, word for word."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

REPO = Path(__file__).resolve().parents[4]
TEMPLATE = REPO / ".github" / "PULL_REQUEST_TEMPLATE" / "2x.md"
STANDARDS = REPO / "specs" / "20261030-140022-senselab-2x-api" / "engineering.md"


def _items(text: str) -> list[str]:
    """The checklist items of `text`, each continuation line joined, in order."""
    return [" ".join(item.split()) for item in re.findall(r"^- \[ \] (.+(?:\n  .+)*)", text, re.MULTILINE)]


def _per_commit(text: str) -> str:
    return text.split("## Per-commit checklist", 1)[1].split("\n## ", 1)[0]


def test_the_template_carries_every_per_commit_item() -> None:
    """Each item of the standards' per-commit checklist appears in the template, in the same order."""
    expected = _items(_per_commit(STANDARDS.read_text()))
    assert len(expected) >= 9
    assert _items(_per_commit(TEMPLATE.read_text())) == expected


def test_the_default_template_stays_the_one_for_1x() -> None:
    """The default template is not the 2.x one: a 1.x PR keeps its own checklist."""
    default = (REPO / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
    assert "Per-commit checklist" not in default
