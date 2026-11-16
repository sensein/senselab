"""The script tests run triage on Bridge2AI-Voice fixtures, so they read the B2AI adult triage profile."""

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _triage_reads_the_b2ai_profile(b2ai_triage_profile: Any) -> None:  # noqa: ANN401 -- the fixture's profile
    """Every script test reads triage's dataset data from the Bridge2AI-Voice adult profile."""
