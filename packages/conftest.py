"""Tier hook: every test under ``packages/`` carries exactly one of the five tier markers.

The `model` sub-marker (loads weights or needs a backend env, ORCD only) goes with `limits`, `usecase` or `env`.
A `gpu` test is skipped on a host with no GPU. A test with no `timeout` mark gets its tier's budget from
`senselab/tests/data/tier_budgets.json`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

TIERS = frozenset({"logic", "contract", "limits", "usecase", "env"})
MODEL_TIERS = frozenset({"limits", "usecase", "env"})
PACKAGES = Path(__file__).resolve().parent
BUDGETS = PACKAGES / "senselab" / "tests" / "data" / "tier_budgets.json"
LOCK = PACKAGES.parent / "uv.lock"
LOCK_DIGEST = "senselab-lock.sha"
"""Beside the venv's interpreter prefix: the digest of the uv.lock it was synced from (scripts/sync_local_venv.sh)."""


def venv_status(prefix: Path, lock: Path) -> str:
    """`STALE VENV ...` when the venv at `prefix` was synced from another `lock`; else whether its digest matches."""
    recorded = prefix / LOCK_DIGEST
    if not recorded.is_file():
        return f"venv lock digest: unrecorded ({recorded} is missing)"
    want = hashlib.sha256(lock.read_bytes()).hexdigest()
    have = recorded.read_text().strip()
    if have != want:
        return f"STALE VENV: synced from uv.lock {have[:12]}, this tree's is {want[:12]}; results are not evidence"
    return f"venv lock digest: {want[:12]}"


def pytest_report_header() -> str:
    """The venv's lock status, first line of every run."""
    return venv_status(Path(sys.prefix), LOCK)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Repeat a stale venv at the end of the run, beside the result."""
    status = venv_status(Path(sys.prefix), LOCK)
    if status.startswith("STALE VENV"):
        terminalreporter.write_line(status, red=True)


def tier_budgets() -> dict[str, float]:
    """Return the time budget in seconds of one test of each tier, from `tier_budgets.json`."""
    document = json.loads(BUDGETS.read_text())
    return {tier: float(row["budget_s"]) for tier, row in document["tiers"].items()}


@pytest.fixture(scope="session")
def tier_names() -> frozenset[str]:
    """The five tier marker names."""
    return TIERS


@pytest.fixture(scope="session")
def budgets_file() -> Path:
    """The file holding each tier's time budget."""
    return BUDGETS


@pytest.fixture(scope="session")
def tier_budget_s() -> dict[str, float]:
    """The time budget in seconds of one test of each tier."""
    return tier_budgets()


@pytest.fixture
def slack(request: pytest.FixtureRequest) -> Path:
    """The test slack file of the requesting test's package: `<package>/tests/data/test_slack.yaml`.

    Test slack is a profile override (P-34); its derivations are in config-derivations.md.
    """
    tests = next(p for p in Path(str(request.path)).resolve().parents if p.name == "tests")
    path = tests / "data" / "test_slack.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: this package has no test slack file")
    return path


@pytest.fixture
def tier_bound(request: pytest.FixtureRequest) -> float:
    """The seconds the requesting test's tier allows it, which bounds every wait on a sentinel in it."""
    marker = request.node.get_closest_marker("timeout")
    assert marker is not None
    return float(marker.args[0])


def _apply_budgets(items: list[pytest.Item]) -> None:
    if not items:
        return
    budgets = tier_budgets()
    for item in items:
        tier = next((m.name for m in item.iter_markers() if m.name in TIERS), None)
        if tier is not None and not item.get_closest_marker("timeout"):
            item.add_marker(pytest.mark.timeout(budgets[tier]))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail the session, listing the node ids, when a test under ``packages/`` has zero or several tier marks."""
    offenders = []
    owned = [item for item in items if PACKAGES in Path(str(item.path)).resolve().parents]
    for item in owned:
        tiers = sorted(m.name for m in item.iter_markers() if m.name in TIERS)
        if len(set(tiers)) != 1:
            offenders.append(f"{item.nodeid}: {sorted(set(tiers)) or 'no tier'}")
        elif item.get_closest_marker("model") and tiers[0] not in MODEL_TIERS:
            offenders.append(f"{item.nodeid}: `model` goes with one of {sorted(MODEL_TIERS)}, not {tiers[0]!r}")
    if offenders:
        raise pytest.UsageError(
            f"each test under packages/ needs exactly one tier marker of {sorted(TIERS)}, "
            f"and `model` only beside {sorted(MODEL_TIERS)}:\n" + "\n".join(offenders)
        )
    _apply_budgets(owned)
    gpu = [item for item in items if item.get_closest_marker("gpu")]
    if gpu:
        from senselab_core.runtime.hardware import (
            probe_hardware,  # noqa: PLC0415 - probed only when a gpu test is collected
        )

        if not probe_hardware().gpus:
            for item in gpu:
                item.add_marker(pytest.mark.skip(reason="needs a GPU; this host has none"))
