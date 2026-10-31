"""Tier hook: every test under ``packages/`` carries exactly one of the five tier markers."""

from __future__ import annotations

from pathlib import Path

import pytest

TIERS = frozenset({"logic", "contract", "limits", "usecase", "env"})
PACKAGES = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail the session, listing the node ids, when a test under ``packages/`` has zero or several tier marks."""
    offenders = []
    for item in items:
        if PACKAGES not in Path(str(item.path)).resolve().parents:
            continue
        tiers = sorted(m.name for m in item.iter_markers() if m.name in TIERS)
        if len(set(tiers)) != 1:
            offenders.append(f"{item.nodeid}: {sorted(set(tiers)) or 'no tier'}")
    if offenders:
        raise pytest.UsageError(
            f"each test under packages/ needs exactly one tier marker of {sorted(TIERS)}:\n" + "\n".join(offenders)
        )
