"""Tier hook: every test under ``packages/`` carries exactly one of the five tier markers.

The `model` sub-marker (loads weights or needs a backend env, ORCD only) goes with `limits`, `usecase` or `env`.
A `gpu` test is skipped on a host with no GPU. A test with no `timeout` mark gets its tier's budget from
`senselab/tests/data/tier_budgets.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

TIERS = frozenset({"logic", "contract", "limits", "usecase", "env"})
MODEL_TIERS = frozenset({"limits", "usecase", "env"})
PACKAGES = Path(__file__).resolve().parent
BUDGETS = PACKAGES / "senselab" / "tests" / "data" / "tier_budgets.yaml"


def tier_budgets() -> dict[str, float]:
    """Return the time budget in seconds of one test of each tier, from `tier_budgets.yaml`."""
    document = yaml.safe_load(BUDGETS.read_text())
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
