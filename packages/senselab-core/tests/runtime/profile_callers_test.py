"""A caller's profile reaches the components that read it: its device query bound is obeyed."""

import subprocess
from typing import Any

import pytest
from senselab_core.runtime.call import host_hardware
from senselab_core.runtime.execution.workers import process_cache
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.units import UnitResolver

PACKAGED = ExecutionProfile.packaged()


@pytest.mark.logic
def test_every_reader_of_the_host_hardware_probes_within_the_profiles_device_query_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model-cache budget and the unit resolver use `sizing.device_query_s`."""
    seen: list[float] = []

    def run(cmd: list[str], **kwargs: Any) -> Any:  # noqa: ANN401
        seen.append(kwargs["timeout"])
        raise FileNotFoundError(cmd[0])

    profile = PACKAGED.model_copy(update={"sizing": PACKAGED.sizing.model_copy(update={"device_query_s": 17.25})})
    monkeypatch.setattr(subprocess, "run", run)
    process_cache(None, profile.sizing.device_query_s)
    assert UnitResolver(profile).hardware is not None
    assert host_hardware(profile.sizing.device_query_s) is host_hardware(17.25)
    assert seen and set(seen) == {17.25}
