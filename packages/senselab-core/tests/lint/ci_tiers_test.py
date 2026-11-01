"""The CI cadence for the test tiers: what a PR runs, what waits for the nightly run or the `limits` label."""

import shlex
import tomllib
from pathlib import Path

import pytest
import yaml
from _pytest.mark.expression import Expression

REPO = Path(__file__).resolve().parents[4]
WORKFLOW = REPO / ".github" / "workflows" / "2x-tests.yaml"
PYPROJECT = REPO / "pyproject.toml"
TIERS = ("logic", "contract", "limits", "usecase", "env")
EVERY_PR, ON_LABEL, NIGHTLY = "every_pr", "on_label", "nightly"
LABEL = "limits"


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


def _marker(args: list[str]) -> str | None:
    return next((args[i + 1] for i, a in enumerate(args) if a == "-m" and i + 1 < len(args)), None)


def package_selections(workflow: str) -> list[tuple[str, str | None, str]]:
    """``(job, -m expression, job condition)`` for each ``pytest`` command that runs a path under ``packages``."""
    selections = []
    for name, job in yaml.safe_load(workflow).get("jobs", {}).items():
        for step in job.get("steps", []):
            for argv in _commands(step.get("run", "")):
                if not any(Path(a).name == "pytest" for a in argv):
                    continue
                args = argv[1:]
                marker = _marker(args)
                paths = [a for a in args if not a.startswith("-") and a != marker]
                if any(p == "packages" or p.startswith("packages/") for p in paths):
                    selections.append((name, marker, str(job.get("if", ""))))
    return selections


def pytest_paths_outside_packages(workflow: str) -> list[tuple[str, str]]:
    """``(job, path)`` for each path a ``pytest`` command in the workflow runs that is not under ``packages``."""
    found = []
    for name, job in yaml.safe_load(workflow).get("jobs", {}).items():
        for step in job.get("steps", []):
            for argv in _commands(step.get("run", "")):
                at = next((i for i, a in enumerate(argv) if Path(a).name == "pytest"), None)
                if at is None or "install" in argv[:at]:
                    continue
                args = argv[at + 1 :]
                values = {args[i + 1] for i, a in enumerate(args) if a in ("-m", "-p", "-k") and i + 1 < len(args)}
                paths = [a for a in args if not a.startswith("-") and a not in values]
                found += [(name, p) for p in paths if p != "packages" and not p.startswith("packages/")]
    return found


def tiers_of(marker: str | None) -> set[str]:
    """The tiers a ``-m`` expression selects; no expression selects them all."""
    if marker is None:
        return set(TIERS)
    expression = Expression.compile(marker)
    return {t for t in TIERS if expression.evaluate(lambda name, tier=t, **_: name == tier)}


def cadence(workflow: str) -> dict[str, set[str]]:
    """The tiers selected on every PR, on a PR carrying the `limits` label, and nightly.

    A job with no condition, or one that only checks the PR is not a draft, runs on every PR and every night. A job
    whose condition names the schedule runs nightly; one whose condition names the PR's labels runs on that label.
    """
    when: dict[str, set[str]] = {EVERY_PR: set(), ON_LABEL: set(), NIGHTLY: set()}
    for _job, marker, condition in package_selections(workflow):
        tiers = tiers_of(marker)
        gated = False
        if "'schedule'" in condition:
            when[NIGHTLY] |= tiers
            gated = True
        if "labels" in condition and f"'{LABEL}'" in condition:
            when[ON_LABEL] |= tiers
            gated = True
        if not gated:
            for key in when:
                when[key] |= tiers
    return when


def default_tiers(pyproject: str) -> set[str]:
    """The tiers a bare ``pytest`` selects under the ``addopts`` of the pytest configuration."""
    addopts = shlex.split(tomllib.loads(pyproject)["tool"]["pytest"]["ini_options"].get("addopts", ""))
    return tiers_of(_marker(addopts))


def _workflow(*jobs: tuple[str, str]) -> str:
    """A workflow with one job per ``(-m expression, condition)``; an empty condition means no ``if``."""
    out = {}
    for i, (marker, condition) in enumerate(jobs):
        step = {"run": f"pytest packages -m '{marker}'"}
        out[f"job{i}"] = {"steps": [step], **({"if": condition} if condition else {})}
    return yaml.safe_dump({"jobs": out})


NIGHTLY_IF = "github.event_name == 'schedule'"
LABEL_IF = "contains(github.event.pull_request.labels.*.name, 'limits')"


@pytest.mark.logic
def test_a_pull_request_runs_logic_and_contract_and_nothing_slower() -> None:
    """The jobs with no schedule or label condition select logic and contract, and neither limits nor use cases."""
    every_pr = cadence(WORKFLOW.read_text())[EVERY_PR]
    assert {"logic", "contract"} <= every_pr
    assert not {"limits", "usecase"} & every_pr


@pytest.mark.logic
def test_limits_run_nightly_and_on_the_limits_label_and_use_cases_only_nightly() -> None:
    """A job gated on the schedule selects limits and use cases; one gated on the `limits` label, limits only."""
    when = cadence(WORKFLOW.read_text())
    assert {"limits", "usecase"} <= when[NIGHTLY]
    assert "limits" in when[ON_LABEL]
    assert "usecase" not in when[ON_LABEL]


@pytest.mark.logic
def test_the_env_tier_runs_on_every_pr() -> None:
    """The release-gate and packaging tests of the `env` tier run on every PR."""
    assert "env" in cadence(WORKFLOW.read_text())[EVERY_PR]


@pytest.mark.logic
def test_the_workflow_runs_nightly_and_reruns_when_a_label_is_added() -> None:
    """The workflow has a cron schedule, and a PR that gains a label triggers it."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["schedule"] and all("cron" in entry for entry in triggers["schedule"])
    assert "labeled" in triggers["pull_request"]["types"]


@pytest.mark.logic
def test_a_bare_pytest_selects_the_logic_tier_only() -> None:
    """The pytest configuration deselects every tier but logic unless a run names a marker expression."""
    assert default_tiers(PYPROJECT.read_text()) == {"logic"}


@pytest.mark.logic
def test_every_ci_test_run_is_under_packages() -> None:
    """No 2.x CI job runs a test path outside ``packages/``: the 1.x source pool under ``src/`` is not tested here."""
    assert package_selections(WORKFLOW.read_text())
    assert pytest_paths_outside_packages(WORKFLOW.read_text()) == []


@pytest.mark.logic
def test_a_slow_tier_on_every_pr_is_found() -> None:
    """A job with no condition that selects limits is reported as running them on every PR and every night."""
    planted = _workflow(("logic or contract", ""), ("limits", ""))
    assert cadence(planted)[EVERY_PR] == {"logic", "contract", "limits"}
    assert cadence(planted)[NIGHTLY] == {"logic", "contract", "limits"}


@pytest.mark.logic
def test_a_job_gated_on_the_schedule_or_the_label_runs_only_there() -> None:
    """A conditioned job adds its tiers to the nightly or labelled run and not to every PR."""
    when = cadence(_workflow(("logic", ""), ("limits", NIGHTLY_IF), ("usecase", LABEL_IF)))
    assert when[EVERY_PR] == {"logic"}
    assert when[NIGHTLY] == {"logic", "limits"}
    assert when[ON_LABEL] == {"logic", "usecase"}
    both = cadence(_workflow(("limits", f"{NIGHTLY_IF} || {LABEL_IF}")))
    assert both[NIGHTLY] == both[ON_LABEL] == {"limits"} and both[EVERY_PR] == set()


@pytest.mark.logic
def test_a_draft_check_alone_does_not_gate_a_job() -> None:
    """A condition that only checks the PR is not a draft leaves the job on every PR."""
    planted = _workflow(("logic", "github.event.pull_request.draft == false"))
    assert cadence(planted)[EVERY_PR] == {"logic"}


@pytest.mark.logic
def test_marker_expressions_are_evaluated_not_matched_by_text() -> None:
    """``not network``, ``and`` and parentheses decide the selection."""
    assert tiers_of("(logic or env) and not env") == {"logic"}
    assert tiers_of("logic and contract") == set()
    assert tiers_of("not logic") == {"contract", "limits", "usecase", "env"}


@pytest.mark.logic
@pytest.mark.parametrize(
    ("addopts", "expected"),
    [
        ('"--strict-markers -m logic"', {"logic"}),
        ("\"-m 'logic or contract'\"", {"logic", "contract"}),
        ('"--strict-markers"', set(TIERS)),
    ],
    ids=["logic", "logic-and-contract", "no-marker"],
)
def test_the_default_selection_is_read_from_addopts(addopts: str, expected: set[str]) -> None:
    """The default tiers are those of the ``-m`` in ``addopts``; with none, every tier."""
    assert default_tiers(f"[tool.pytest.ini_options]\naddopts = {addopts}\n") == expected


@pytest.mark.logic
def test_jobs_outside_packages_and_unmarked_runs_are_handled() -> None:
    """A job on ``src/tests`` selects nothing here; a ``packages`` run with no ``-m`` selects everything."""
    other = yaml.safe_dump({"jobs": {"a": {"steps": [{"run": "uv run pytest src/tests -m logic"}]}}})
    assert package_selections(other) == []
    bare = yaml.safe_dump({"jobs": {"a": {"steps": [{"run": "uv run pytest \\\n  packages/senselab-core"}]}}})
    assert cadence(bare)[EVERY_PR] == set(TIERS)


@pytest.mark.logic
def test_a_job_on_src_tests_is_found() -> None:
    """A planted ``pytest src/tests`` job, with its markers and plugin flags, is reported by its path alone."""
    planted = yaml.safe_dump(
        {"jobs": {"a": {"steps": [{"run": "uv run pytest src/tests -m logic -p no:cacheprovider packages"}]}}}
    )
    assert pytest_paths_outside_packages(planted) == [("a", "src/tests")]
