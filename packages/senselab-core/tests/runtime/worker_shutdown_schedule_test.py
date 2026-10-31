"""Shutdown under interruption: every injection at every boundary call ends typed, bounded and with no orphan."""

from pathlib import Path
from typing import Any

import pytest
from shutdown_schedule_fixture import KINDS, counted, run_schedule, schedule, violations


def _check(tmp_path: Path, kind: str, occurrences: range) -> list[dict[str, Any]]:
    todo = [case for case in schedule(counted(tmp_path), occurrences) if case["kind"] == kind]
    reports = run_schedule(tmp_path, todo)
    failed = {f"{r['path']} {r['boundary']}#{r['occurrence']}": v for r in reports if (v := violations(r))}
    assert failed == {}
    return reports


@pytest.mark.contract
@pytest.mark.parametrize("kind", KINDS)
def test_every_injection_at_every_boundary_ends_typed_bounded_and_clean(tmp_path: Path, kind: str) -> None:
    """Each injection at the first call of each boundary, on construction, a timed-out run, an idle close.

    An injection that raises inside the shutdown runs the SIGKILL fallback and a second pass of the steps.
    """
    reports = _check(tmp_path, kind, range(1, 2))
    if kind in ("sigint", "handler_raises_base", "handler_raises_exception"):
        assert any(r["calls"]["client:_kill_group_now"] for r in reports)
        assert any(r["calls"]["client:_steps"] >= 2 and r["path"] == "close" for r in reports)


@pytest.mark.limits
@pytest.mark.parametrize("kind", KINDS)
def test_every_injection_at_every_later_boundary_call_ends_typed_bounded_and_clean(tmp_path: Path, kind: str) -> None:
    """The same at every later call of each boundary: the SIGKILL `_signal_group`, the second `_members_gone`."""
    _check(tmp_path, kind, range(2, 100))
