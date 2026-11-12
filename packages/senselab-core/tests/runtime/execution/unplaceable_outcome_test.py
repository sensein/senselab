"""An unplaceable plan is a plan outcome: not the env's fault, with its own headline in status and explain."""

from pathlib import Path

import pytest
from senselab_core.cli import main as cli_main
from senselab_core.runtime.execution.events import EventLog, read_events, status_from
from senselab_core.runtime.execution.executor import FaultRecord, JobRef
from senselab_core.runtime.fit import ClassTried, Unplaceable

pytestmark = pytest.mark.logic

TRIED = (
    ClassTried("cpu", 10**9, "too small", 950_000_000, 1000.0, 9e8),
    ClassTried("cpu:large", 8 * 10**9, "unfit", unfit_grant=8 * 10**9),
)


def test_unplaceable_is_a_resource_outcome_and_never_counts_against_the_env() -> None:
    """Its record is not infrastructure and not a gap: no env budget is spent on it."""
    record = FaultRecord.from_exception(Unplaceable("unit abc123def456", TRIED))
    assert (record.fault, record.gap, record.charged, record.load) == ("resource", False, False, False)
    assert not record.counts_against_env and record.type == "Unplaceable"


def _log(tmp_path: Path, count: int) -> None:
    log = EventLog(tmp_path, "a1", 1)
    failure = FaultRecord.from_exception(Unplaceable("unit abc123def456", TRIED))
    for i in range(count):
        job = JobRef(plan_hash="a" * 64, node="a", element=(i,))
        log.record("queued", "a", job=job, env="host")
        log.record("failed", "a", job=job, env="host", failure=failure)


def test_status_gives_it_its_own_headline_and_no_broken_env(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Many failed jobs of one plan outcome break no env; the status names the unit, each class and the figures once."""
    _log(tmp_path, 8)
    status = status_from(read_events(tmp_path), broken_after=2)
    assert status.broken == () and len(status.unplaceable) == 1
    assert cli_main(["status", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "broken env" not in out
    [headline] = [line for line in out.splitlines() if line.startswith("unplaceable:")]
    for part in ("unit abc123def456", "class cpu (1000000000 bytes) is too small", "resident 950000000", "unfit"):
        assert part in headline
