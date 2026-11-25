"""The script tests: a driver activates the profile it is given, so no test starts with one active.

The fixture profile is active while the test modules are imported (some read triage values at import); each test
then starts with none, as a driver does. A test that calls a driver's internals, which read the profile's data,
takes the ``triage_profile`` fixture (src/tests/conftest.py); one that runs a driver passes ``--profile``.
"""

from collections.abc import Iterator

import pytest

from tests.conftest import activate_fixture_profile_for_collection  # noqa: E402

activate_fixture_profile_for_collection()


@pytest.fixture(autouse=True)
def _no_profile_unless_the_test_takes_one() -> Iterator[None]:
    """No dataset profile is active when a script test starts, nor after it."""
    from senselab.audio.workflows.triage import dataset_profile

    dataset_profile.deactivate_profile()
    dataset_profile.set_run_data(None)
    yield
    dataset_profile.deactivate_profile()
    dataset_profile.set_run_data(None)
