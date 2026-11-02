"""The PR models check: an implementer and a different reviewer model are named."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_pr_models.py"
_spec = importlib.util.spec_from_file_location("check_pr_models", _SCRIPT)
assert _spec and _spec.loader
pr = importlib.util.module_from_spec(_spec)
sys.modules["check_pr_models"] = pr
_spec.loader.exec_module(pr)

TEMPLATE = Path(__file__).resolve().parents[4] / ".github" / "PULL_REQUEST_TEMPLATE" / "2x.md"


def _body(implementer: str, reviewer: str) -> str:
    return f"- Implementer (model, effort): {implementer}\n- Reviewer (model, effort; other): {reviewer}\n"


def test_equal_models_refused() -> None:
    """The same model for both roles fails, whatever the case or effort."""
    assert pr.problems(_body("Sonnet 5.5, high", "sonnet 5.5, low"))
    assert not pr.problems(_body("Sonnet 5.5, high", "Opus 5.5, high"))


@pytest.mark.parametrize("body", ["", _body("", "Opus"), _body("Sonnet", " "), "- Implementer: a\n"])
def test_a_missing_model_is_refused(body: str) -> None:
    """A body without both models, or with a blank one, fails."""
    assert pr.problems(body)


def test_the_template_is_refused_until_filled_in() -> None:
    """The unfilled 2.x template fails, and its comments are not read as values."""
    assert pr.problems(TEMPLATE.read_text())
