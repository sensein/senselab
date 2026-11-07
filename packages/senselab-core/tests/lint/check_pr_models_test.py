"""The PR body check: both models, a different reviewer or a declared top-tier kernel pairing, every checklist line."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.logic

REPO = Path(__file__).resolve().parents[4]
TEMPLATE = REPO / ".github" / "PULL_REQUEST_TEMPLATE" / "2x.md"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("check_pr_template")
pr = _load("check_pr_models")
REQUIRED = ["One item (P-1).", "Two items, the second wrapped (P-2)."]
CHECKLIST = "- [x] One item (P-1).\n- [ ] Two items, the second\n  wrapped (P-2).\n"


def _body(implementer: str, reviewer: str, kernel: str = "no", checklist: str = CHECKLIST) -> str:
    return (
        f"- Implementer (model, effort): {implementer}\n- Reviewer (model, effort; other): {reviewer}\n"
        f"- Kernel item (yes or no): {kernel}\n{checklist}"
    )


def test_different_models_with_every_checklist_line_pass() -> None:
    """Two different models and every checklist line, ticked or not, pass."""
    assert not pr.problems(_body("Sonnet 5.5, high", "Opus 5.5, high"), REQUIRED)


def test_equal_models_refused() -> None:
    """The same model for both roles fails, whatever the case or effort."""
    assert pr.problems(_body("Sonnet 5.5, high", "sonnet 5.5, low"), REQUIRED)


@pytest.mark.parametrize("model", ["Opus 5.5", "Fable 1"])
def test_the_top_tier_may_review_itself_only_on_a_declared_kernel_item(model: str) -> None:
    """Opus with opus, or fable with fable, passes on a kernel item the body declares, and fails otherwise."""
    assert not pr.problems(_body(f"{model}, high", f"{model}, high", kernel="yes"), REQUIRED)
    assert pr.problems(_body(f"{model}, high", f"{model}, high", kernel="no"), REQUIRED)
    assert pr.problems(_body("Sonnet 5.5", "Sonnet 5.5", kernel="yes"), REQUIRED)


@pytest.mark.parametrize("body", ["", _body("", "Opus"), _body("Sonnet", " "), "- Implementer: a\n"])
def test_a_missing_model_is_refused(body: str) -> None:
    """A body without both models, or with a blank one, fails."""
    assert pr.problems(body, REQUIRED)


def test_a_missing_checklist_line_is_refused() -> None:
    """A body that drops or rewords one checklist line fails, naming it."""
    found = pr.problems(_body("Sonnet", "Opus", checklist="- [x] One item (P-1).\n- [ ] Two items (P-2).\n"), REQUIRED)
    assert found == ["missing checklist line: Two items, the second wrapped (P-2)."]


def test_the_template_is_refused_until_filled_in() -> None:
    """The unfilled 2.x template fails on its models, and carries every line of engineering.md's checklist."""
    text = TEMPLATE.read_text()
    found = pr.problems(text, pr.checklist())
    assert found and not [line for line in found if line.startswith("missing checklist line")]
