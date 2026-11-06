"""The script tests run triage on the fixture profile's recordings, so they read its data."""

from typing import Any

import pytest

from tests.conftest import activate_fixture_profile_for_collection  # noqa: E402

activate_fixture_profile_for_collection()


@pytest.fixture(autouse=True)
def _triage_reads_the_fixture_profile(triage_profile: Any) -> None:  # noqa: ANN401 -- the fixture's profile
    """Every script test reads triage's dataset data from the fixture profile."""
