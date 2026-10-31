"""Shutdown under interruption: every injection at every boundary ends typed, bounded and with no orphan."""

from pathlib import Path

import pytest
from shutdown_schedule_fixture import KINDS, RUN_BOUNDARIES, SHUTDOWN_BOUNDARIES, cases, run_schedule, violations


@pytest.mark.contract
@pytest.mark.parametrize("kind", KINDS)
def test_every_injection_at_every_boundary_ends_typed_bounded_and_clean(tmp_path: Path, kind: str) -> None:
    """SIGINT, SIGTERM, a signal handler's close() and close() from either thread, at each boundary of both paths."""
    todo = [case for case in cases() if case["kind"] == kind]
    reports = run_schedule(tmp_path, todo)
    failed = {f"{r['path']} {r['boundary']}": v for r in reports if (v := violations(r))}
    assert failed == {}
    assert {r["boundary"] for r in reports} == set(RUN_BOUNDARIES) | set(SHUTDOWN_BOUNDARIES)
