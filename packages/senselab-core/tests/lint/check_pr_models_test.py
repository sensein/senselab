"""The PR body check: P-40's pairing (sonnet implements, opus reviews, fable only for a declared class), every line."""

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


def _body(implementer: str, reviewer: str, stakes: str = "none", checklist: str = CHECKLIST) -> str:
    return (
        f"- Implementer (model, effort; other): {implementer}\n- Reviewer (model, effort; other): {reviewer}\n"
        f"- High-stakes class (none or a class): {stakes}\n{checklist}"
    )


def _fails(body: str, fragment: str) -> bool:
    return any(fragment in line for line in pr.problems(body, REQUIRED))


def test_sonnet_implemented_and_opus_reviewed_passes() -> None:
    """The default pairing, with every checklist line ticked or not, passes."""
    assert not pr.problems(_body("Sonnet 5.5, high", "Opus 5.5, high"), REQUIRED)


@pytest.mark.parametrize("reviewer", ["Sonnet 5.5, low", "Haiku 4.5, low", "GPT-9", "mystery"])
def test_only_opus_or_fable_reviews(reviewer: str) -> None:
    """A reviewer that is not opus or fable fails, including the implementer's own sonnet."""
    assert _fails(_body("Sonnet 5.5", reviewer, stakes="protocol-adr"), "reviews run on opus")


@pytest.mark.parametrize("implementer", ["Haiku 4.5, low", "GPT-9"])
def test_implementation_runs_on_sonnet_or_its_escalation(implementer: str) -> None:
    """A haiku or unknown implementer fails."""
    assert _fails(_body(implementer, "Opus 5.5"), "implementation runs on sonnet")


@pytest.mark.parametrize("line", ["Opus 5.5, high, escalated after two failures", "Opus 5.5, high, design"])
def test_opus_implements_only_when_escalated_or_design(line: str) -> None:
    """Opus implementing says why on its line; an opus reviewer then says it is a fresh context."""
    assert not pr.problems(_body(line, "Opus 5.5, high, fresh context"), REQUIRED)
    assert _fails(_body("Opus 5.5, high", "Opus 5.5, high, fresh"), "`escalated`")
    assert _fails(_body(line, "Opus 5.5, high"), "`fresh`")


def test_an_opus_reviewer_of_a_sonnet_implementer_needs_no_fresh_note() -> None:
    """The fresh-context note is required only where the reviewer's model equals the implementer's."""
    assert not pr.problems(_body("Sonnet 5.5", "Opus 5.5"), REQUIRED)


@pytest.mark.parametrize("stakes", ["identity-privacy-security", "protocol-adr", "second-round-design"])
def test_fable_reviews_only_a_declared_high_stakes_class(stakes: str) -> None:
    """Fable as reviewer passes on each high-stakes class and fails on none, a blank or a missing line."""
    assert not pr.problems(_body("Sonnet 5.5", "Fable 1", stakes=stakes), REQUIRED)
    for undeclared in ("none", "", "None, kernel"):
        assert _fails(_body("Sonnet 5.5", "Fable 1", stakes=undeclared), "high-stakes classes only")
    no_line = "\n".join(_body("Sonnet 5.5", "Fable 1").splitlines()[:2]) + "\n" + CHECKLIST
    assert _fails(no_line, "high-stakes classes only")


def test_fable_implementing_needs_a_class_too() -> None:
    """Fable on either side needs the declared class; the same model on both sides is not itself the rule."""
    assert _fails(_body("Fable 1", "Opus 5.5"), "high-stakes classes only")
    assert not pr.problems(_body("Fable 1", "Fable 1", stakes="protocol-adr"), REQUIRED)


def test_an_unknown_high_stakes_class_is_refused() -> None:
    """A class outside the list is refused even when no fable is named."""
    assert _fails(_body("Sonnet 5.5", "Opus 5.5", stakes="kernel"), "high-stakes class is 'kernel'")


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
