"""`launch`: a launched process inherits the declared families of ambient variables, never the Python-path ones."""

import pytest
from senselab_core.runtime.launch import ISOLATED, launch, launch_environment

pytestmark = pytest.mark.logic

FAMILIES, NEVER = launch_environment()


def _member(family: str) -> str:
    found = FAMILIES[family]
    return min(found.names) if found.names else found.prefixes[0] + "SENSELAB_PROBE"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_each_family_passes_and_only_recorded_ones_reach_the_digest(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A variable of each family reaches the process; it changes the digest exactly when its family is recorded."""
    name = _member(family)
    monkeypatch.delenv(name, raising=False)
    before = launch("/env/bin/python")
    monkeypatch.setenv(name, "senselab-probe")
    after = launch("/env/bin/python")
    assert after.variables[name] == "senselab-probe"
    assert (after.digest != before.digest) is FAMILIES[family].recorded
    assert (after.measured.get(name) == "senselab-probe") is FAMILIES[family].measured


def test_only_thread_counts_are_measured() -> None:
    """A measure holds the thread counts a launch sets; device selection is never measured."""
    assert FAMILIES["threads"].measured and not FAMILIES["gpu"].measured
    started = launch("/env/bin/python", {"CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "4"})
    assert started.measured["OMP_NUM_THREADS"] == "4" and "CUDA_VISIBLE_DEVICES" not in started.measured


@pytest.mark.parametrize(
    "name", ["PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX", "OTHER_X"]
)
def test_python_path_and_unlisted_variables_never_pass(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Python-path variables, and any variable no family names, never reach a launched process."""
    monkeypatch.setenv(name, "/caller")
    started = launch("/env/bin/python")
    assert started.variables.get(name) == ISOLATED.get(name)


def test_declared_variables_pass_and_are_recorded() -> None:
    """A backend's declared engine variables are set over the ambient ones and enter the digest."""
    plain, declared = launch("/env/bin/python"), launch("/env/bin/python", {"VLLM_USE_X": "1"})
    assert declared.variables["VLLM_USE_X"] == "1" and declared.digest != plain.digest


def test_a_worker_runs_isolated_with_no_import_path_of_the_caller() -> None:
    """Every worker runs `-I -m`; no launch carries the caller's import path."""
    assert launch("/env/bin/python").module("m", "a") == ["/env/bin/python", "-I", "-m", "m", "a"]
