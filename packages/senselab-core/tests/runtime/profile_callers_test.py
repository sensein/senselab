"""A caller's profile reaches the components that read it: its `store` section and its fetch timeout are obeyed."""

import contextlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from lease_clock_fixture import ManualClock
from senselab_core.runtime.call import host_hardware
from senselab_core.runtime.execution.in_process import InProcess, default_store
from senselab_core.runtime.execution.observing import hardware_of
from senselab_core.runtime.execution.pool import run_store
from senselab_core.runtime.execution.pool_worker import check_reachable
from senselab_core.runtime.execution.run import MANIFEST_FILE, Run, results_store
from senselab_core.runtime.execution.substrates import StoreUnreachable
from senselab_core.runtime.execution.workers import process_cache
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import UnitResolver

PACKAGED = ExecutionProfile.packaged()
STORE = PACKAGED.store.model_copy(update={"commit_attempts": 7, "clock_tolerance_s": 1e13})
PROFILE = PACKAGED.model_copy(update={"store": STORE})


@pytest.mark.logic
def test_the_default_store_of_an_executor_is_made_under_its_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`default_store(profile)` and an `InProcess` given no store both read the profile's `store` section."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path))
    assert default_store(PROFILE).settings.commit_attempts == 7
    assert default_store().settings.commit_attempts == PACKAGED.store.commit_attempts
    assert InProcess(profile=PROFILE).store.settings.commit_attempts == 7


@pytest.mark.logic
def test_the_store_a_run_reads_is_made_under_the_profile_it_was_given(tmp_path: Path) -> None:
    """`Run` and `results_store` read `store` from the profile; a store the manifest names is opened the same way."""
    named = tmp_path / "named"
    (tmp_path / "run" / "private").mkdir(parents=True)
    (tmp_path / "run" / MANIFEST_FILE).write_text(json.dumps({"store": str(named)}))
    run = Run(tmp_path / "run", profile=PROFILE)
    assert run._results_store().settings.commit_attempts == 7 and run._results_store().root == named
    assert results_store(PROFILE).settings.commit_attempts == 7
    assert results_store(PROFILE, str(named)).root == named


@pytest.mark.logic
def test_the_store_a_pool_worker_names_is_made_under_the_profile(tmp_path: Path) -> None:
    """`run_store` opens the run's store under the profile it is given."""
    assert run_store(tmp_path, PROFILE).settings.commit_attempts == 7


@pytest.mark.logic
def test_reachability_is_probed_under_the_profiles_store_section(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A root whose clock is an hour off is unreachable under the default tolerance and reachable under a wide one."""
    skewed = lease_clock()
    skewed.local_now = lambda: 1e12  # type: ignore[method-assign]
    store = ResultStore(tmp_path / "store")
    (tmp_path / "run").mkdir()
    (tmp_path / "store").mkdir()
    with pytest.raises(StoreUnreachable):
        check_reachable(tmp_path / "run", store, PACKAGED.store, clock=skewed)
    check_reachable(tmp_path / "run", store, STORE, clock=skewed)


@pytest.mark.logic
def test_every_reader_of_the_host_hardware_probes_within_the_profiles_device_query_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model-cache budget, the worker's hardware class and the unit resolver use `sizing.device_query_s`."""
    seen: list[float] = []

    def run(cmd: list[str], **kwargs: Any) -> Any:  # noqa: ANN401
        seen.append(kwargs["timeout"])
        raise FileNotFoundError(cmd[0])

    profile = PACKAGED.model_copy(update={"sizing": PACKAGED.sizing.model_copy(update={"device_query_s": 17.25})})
    monkeypatch.setattr(subprocess, "run", run)
    process_cache(None, profile.sizing.device_query_s)
    with contextlib.suppress(ValueError):  # a host that shows no CPU model refuses the grant, after the probe
        hardware_of(profile, "cpu")
    assert UnitResolver(profile).hardware is not None
    assert host_hardware(profile.sizing.device_query_s) is host_hardware(17.25)
    assert seen and set(seen) == {17.25}
