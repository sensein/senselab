"""The CI jobs that run ``packages/`` together select every test tier."""

import shlex
from pathlib import Path

import pytest
import yaml
from _pytest.mark.expression import Expression

REPO = Path(__file__).resolve().parents[4]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yaml"
TIERS = ("logic", "contract", "limits", "usecase", "env")


def _commands(run: str) -> list[list[str]]:
    """The shell commands of a ``run`` block, line continuations joined, as argument lists."""
    joined = run.replace("\\\n", " ")
    out = []
    for line in joined.splitlines():
        try:
            out.append(shlex.split(line))
        except ValueError:
            continue
    return out


def package_selections(workflow: str) -> list[tuple[str, str | None]]:
    """``(job, -m expression)`` for each ``pytest`` command in the workflow that runs a path under ``packages``."""
    selections = []
    jobs = yaml.safe_load(workflow).get("jobs", {})
    for name, job in jobs.items():
        for step in job.get("steps", []):
            for argv in _commands(step.get("run", "")):
                if not any(Path(a).name == "pytest" for a in argv):
                    continue
                args = argv[1:]
                marker = next((args[i + 1] for i, a in enumerate(args) if a == "-m" and i + 1 < len(args)), None)
                paths = [a for a in args if not a.startswith("-") and a != marker]
                if any(p == "packages" or p.startswith("packages/") for p in paths):
                    selections.append((name, marker))
    return selections


def selected_tiers(workflow: str) -> set[str]:
    """The tiers that some ``packages`` job selects: a job with no ``-m`` selects them all."""
    selected: set[str] = set()
    for _job, marker in package_selections(workflow):
        if marker is None:
            return set(TIERS)
        expression = Expression.compile(marker)
        selected |= {t for t in TIERS if expression.evaluate(lambda name, tier=t, **_: name == tier)}
    return selected


def _workflow(*selections: str) -> str:
    jobs = {f"job{i}": {"steps": [{"run": f"pytest packages -m '{m}'"}]} for i, m in enumerate(selections)}
    return yaml.safe_dump({"jobs": jobs})


@pytest.mark.logic
def test_ci_selects_every_tier() -> None:
    """The union of the ``-m`` expressions of the jobs that run ``packages`` covers all five tiers."""
    assert package_selections(WORKFLOW.read_text())
    assert selected_tiers(WORKFLOW.read_text()) == set(TIERS)


@pytest.mark.logic
def test_a_tier_no_job_selects_is_found() -> None:
    """A workflow whose jobs leave one tier out is reported as missing exactly that tier."""
    for left_out in TIERS:
        kept = [t for t in TIERS if t != left_out]
        assert set(TIERS) - selected_tiers(_workflow(*kept)) == {left_out}


@pytest.mark.logic
def test_marker_expressions_are_evaluated_not_matched_by_text() -> None:
    """``not network``, ``and`` and parentheses decide the selection."""
    assert selected_tiers(_workflow("(logic or env) and not env")) == {"logic"}
    assert selected_tiers(_workflow("logic and contract")) == set()
    assert selected_tiers(_workflow("not logic")) == {"contract", "limits", "usecase", "env"}


@pytest.mark.logic
def test_jobs_outside_packages_and_unmarked_runs_are_handled() -> None:
    """A job on ``src/tests`` selects nothing here; a ``packages`` run with no ``-m`` selects everything."""
    other = yaml.safe_dump({"jobs": {"a": {"steps": [{"run": "uv run pytest src/tests -m logic"}]}}})
    assert selected_tiers(other) == set()
    bare = yaml.safe_dump({"jobs": {"a": {"steps": [{"run": "uv run pytest \\\n  packages/senselab-core"}]}}})
    assert selected_tiers(bare) == set(TIERS)
