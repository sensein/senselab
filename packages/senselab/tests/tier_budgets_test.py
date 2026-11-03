"""The per-tier time budgets: held as data with a derivation each, and applied to every test by its tier."""

from pathlib import Path

import pytest
import yaml


@pytest.mark.logic
def test_every_tier_has_a_positive_budget_and_a_derivation(budgets_file: Path, tier_names: frozenset[str]) -> None:
    """The file names exactly the five tiers, each with a positive `budget_s` and a written `derivation`."""
    tiers = yaml.safe_load(budgets_file.read_text())["tiers"]
    assert set(tiers) == tier_names
    for tier, row in tiers.items():
        assert row["budget_s"] > 0 and str(row["derivation"]).strip(), tier
        measured = row["measured_s"]
        assert measured is None or 0 < measured <= row["budget_s"], tier


@pytest.mark.logic
def test_a_test_carries_its_tiers_budget_as_its_timeout(
    request: pytest.FixtureRequest, tier_budget_s: dict[str, float]
) -> None:
    """The collection hook marks a test with the budget of its tier, which this test is of `logic`."""
    marker = request.node.get_closest_marker("timeout")
    assert marker is not None and marker.args == (tier_budget_s["logic"],)
