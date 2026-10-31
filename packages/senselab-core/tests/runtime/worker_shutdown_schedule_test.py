"""Shutdown under interruption: every injection at every boundary call ends typed, bounded and with no orphan."""

from pathlib import Path
from typing import Any

import pytest
from shutdown_schedule_fixture import (
    CASE_BOUND_S,
    KINDS,
    PATHS,
    REAL_GRACES,
    case_bound,
    counted,
    ctrl_c_bound,
    run_schedule,
    schedule,
    violations,
)


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


@pytest.mark.logic
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("real_graces", [False, True])
def test_l3_every_case_bound_carries_load_slack_yet_fails_a_real_hang(path: str, real_graces: bool) -> None:
    """A shrunk-grace bound stays below the real STOP grace and a real-grace bound below the faulthandler timeout.

    A case that ignores the shrunk graces waits out at least the real STOP grace, and a deadlock ends at
    `CASE_BOUND_S`; both must exceed the bound, so the load allowance never hides either.
    """
    bound = case_bound({"path": path, "real_graces": real_graces})
    assert bound < (CASE_BOUND_S if real_graces else REAL_GRACES["STOP_GRACE_S"])
    assert ctrl_c_bound() < REAL_GRACES["STOP_GRACE_S"]
