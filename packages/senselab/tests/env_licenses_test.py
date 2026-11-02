"""The packaged env license records: every locked package has one, and the provider refuses what it cannot cover."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml
from senselab._api.packaged import LOCK_RECORDS
from senselab_core.kernel.declare import REGISTRY, Env
from senselab_core.testing.lock_licenses import HOST_ENV, lock_records_from

ROOT = Path(__file__).resolve().parents[3]
LOCKS = ROOT / "packages/senselab/src/senselab/data/env_locks"
_spec = importlib.util.spec_from_file_location("check_licenses", ROOT / "scripts" / "check_licenses.py")
assert _spec and _spec.loader
lic = importlib.util.module_from_spec(_spec)
sys.modules["check_licenses"] = lic
_spec.loader.exec_module(lic)
RECORDS = yaml.safe_load(lic.ENV_LICENSES.read_text())


@pytest.mark.contract
@pytest.mark.parametrize("env", sorted(p.stem for p in LOCKS.glob("*.txt")))
def test_every_locked_package_has_a_record_at_its_pinned_version(env: str) -> None:
    """A lock and its records cover the same packages at the same versions, so a bump without a re-read fails."""
    pins = lic.lock_pins((LOCKS / f"{env}.txt").read_text())
    have = {name: entry["version"] for name, entry in RECORDS["envs"][env].items()}
    assert have == pins


@pytest.mark.contract
def test_the_host_records_cover_the_root_lock_closure() -> None:
    """The host env records are the pins of the senselab closure in `uv.lock`, own packages apart."""
    have = {name: entry["version"] for name, entry in RECORDS["envs"][HOST_ENV].items()}
    assert have == lic.host_pins(ROOT / "uv.lock")


@pytest.mark.contract
def test_every_registered_backend_has_records_for_its_env() -> None:
    """Each backend in the registry resolves to an env with records, and the provider derives from them."""
    for task in REGISTRY.tasks():
        for backend in REGISTRY.backends(task.name):
            assert LOCK_RECORDS(backend), (task.name, backend.name)


@pytest.mark.logic
def test_the_provider_refuses_a_wrong_format_and_an_env_with_no_records() -> None:
    """A document of another format, or a backend whose env has no records, raises rather than passing empty."""
    with pytest.raises(ValueError, match="format"):
        lock_records_from("format: env-licenses@0\nenvs: {}\n")
    provider = lock_records_from("format: env-licenses@1.0\nenvs: {}\n")
    backend = next(b for t in REGISTRY.tasks() for b in REGISTRY.backends(t.name) if b.env == Env.host())
    with pytest.raises(ValueError, match="no license records for env 'host'"):
        provider(backend)
