"""Shared fixtures of the job engine tests."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from host_fixture import host_shape  # noqa: F401 - a fixture of every test here
from lease_clock_fixture import ManualClock
from senselab_core.runtime.execution.provenance_check import check_provenance

CLOCK_START = 1000.0


@pytest.fixture(autouse=True)
def fixture_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let host workers import the fixture backends as `backends.<name>`."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))


@pytest.fixture
def clock(lease_clock: Callable[..., ManualClock]) -> ManualClock:
    """A clock that starts at `CLOCK_START` and is moved by hand."""
    return lease_clock(start=CLOCK_START)


@pytest.fixture
def fake_poll_s(slack_tests: dict[str, Any]) -> float:
    """The real seconds a fake worker thread waits between looks at its queue."""
    return float(slack_tests["fake_worker"]["poll_s"])


UNCHECKED = {
    "pool_probe_test::test_a_worker_builds_an_unbuilt_env_before_its_mark_and_the_unit_probes": (
        "a hand commits the probe items"
    ),
    "run_test::test_a_failed_provenance_write_fails_the_run": "its provenance write is planted to fail",
    "provenance_check_test::test_a_committed_key_whose_activity_was_never_written_is_named": "it removes a document",
    "provenance_check_test::test_a_writer_records_at_most_one_reuse_per_item_per_attempt": "it adds a second Reuse",
    "provenance_check_test::test_a_failed_job_and_a_dead_item_are_not_committed_keys": "it rewrites outcomes",
    "provenance_check_test::test_status_check_provenance_exits_zero_for_a_complete_run_and_one_otherwise": (
        "it removes a document"
    ),
}
"""The tests whose runs `provenance_is_complete` leaves unchecked (`<module>::<test>`), each with why."""


@pytest.fixture(autouse=True)
def provenance_is_complete(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[None]:
    """After each test, every run it left under `tmp_path` has an execution or a Reuse for each key it committed.

    A run's keys are checked by `check_provenance`; a test that resumes and knows the Reuse it expects calls it again
    with `reuses`. A test whose runs are planted, or whose jobs the pool does not run, is named in `UNCHECKED`.
    """
    yield
    if f"{request.node.module.__name__}::{request.node.originalname}" in UNCHECKED:
        return
    for run_dir in sorted({p.parent for p in tmp_path.rglob("plan.json") if (p.parent / "epoch").is_dir()}):
        check_provenance(run_dir).require()
