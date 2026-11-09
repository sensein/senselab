"""A record another user can read carries the torch build, never the package index URL it was installed from."""

import os
import sys
from pathlib import Path

import pytest
from planted_objects_fixture import identity_leaks, store_context, store_key
from senselab_core.kernel.declare import Env
from senselab_core.kernel.identity import ExecutionContext
from senselab_core.runtime.envs import EnvHandle, HostCuda, Uv, ensure_env, torch_build
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.measures import MeasureEnv, MeasureRoots, MeasureUnit, Provenance, env_entry
from senselab_core.runtime.profiles import Envs
from senselab_core.runtime.storage import StorePolicy
from senselab_core.runtime.store import STORE_DIR, ResultStore
from senselab_core.testing.calibration_samples import UNIT, observation

MIRROR = "https://user:secret@mirror.internal/whl/cu128"
NEEDLES = (MIRROR, "secret", "mirror.internal")
FAKE_UV = Uv((sys.executable, str(Path(__file__).with_name("fake_uv_fixture.py"))))
LEASE = JobLease(stale_after_s=30.0, renew_fraction=0.01, attribute_delay_s=0)


def _handle(index: str | None, sha: str = "a" * 64) -> EnvHandle:
    return EnvHandle(python="/env/bin/python", python_version="3.12", lock_sha=sha, wheel_set="w", torch_index=index)


def _context(handle: EnvHandle) -> ExecutionContext:
    context: ExecutionContext = store_context().model_copy(update=handle.context_fields())
    return context


def _built(tmp_path: Path, index: str) -> EnvHandle:
    lock = tmp_path / "demo.txt"
    lock.write_text(
        "# senselab subprocess-venv lock: demo\n# python: 3.12\n# torch: torch==2.8.0 torchaudio==2.8.0\n"
        f"# torch-index: cu128\nnumpy==2.1.0 \\\n    --hash=sha256:{'0' * 64}\n"
    )
    return ensure_env(
        Env.isolated("demo"),
        lock,
        tmp_path / "envs",
        settings=Envs(server_start_margin_s=1.0, torch_index=index),
        lease=LEASE,
        uv=FAKE_UV,
        host_cuda=HostCuda((12, 8), "nvidia-smi"),
    )


@pytest.mark.logic
@pytest.mark.parametrize(
    ("index", "build"),
    [
        ("https://download.pytorch.org/whl/cu128", "cu128"),
        ("https://download.pytorch.org/whl/cpu", "cpu"),
        ("https://download.pytorch.org/whl/rocm6.2/", "rocm6.2"),
        ("https://download.pytorch.org/whl/xpu", "xpu"),
        (MIRROR, "cu128"),
        ("https://user:secret@mirror.internal/simple", "other"),
    ],
)
def test_torch_build_is_the_tag_of_the_index(index: str, build: str) -> None:
    """The build is the index URL's last path segment when it is a build tag, else `other`; never the URL."""
    assert torch_build(index) == build


@pytest.mark.logic
def test_context_identity_uses_the_torch_build() -> None:
    """Two index URLs of one build are one context; another build is another."""
    one = _context(_handle("https://download.pytorch.org/whl/cu128"))
    mirrored = _context(_handle(MIRROR))
    other = _context(_handle("https://download.pytorch.org/whl/cu126"))
    assert one.driver == {"torch_build": "cu128"}
    assert one.fingerprint() == mirrored.fingerprint() != other.fingerprint()


@pytest.mark.contract
def test_store_dir_is_store6() -> None:
    """The context record changed shape, so entries live under a new store directory."""
    assert STORE_DIR == "store@6"


@pytest.mark.contract
def test_no_index_url_in_any_readable_record(tmp_path: Path) -> None:
    """With a userinfo mirror as the index, the store and the measures roots hold neither it nor its parts."""
    handle = _built(tmp_path, MIRROR)
    assert handle.torch_index == MIRROR
    store, shared = tmp_path / "store", tmp_path / "measures"
    ResultStore(store).put(store_key(), {"y": 1}, _context(handle))
    entry = env_entry({**handle.context_fields(), "platform": "linux", "numerics": {"device": "cpu"}, "threads": {}})
    unit = MeasureUnit.model_validate(
        {**UNIT.model_dump(), "env": MeasureEnv.model_validate({**entry, "decoder": None}).model_dump()}
    )
    provenance = Provenance.model_validate(
        {"job": "1", "run": "r", "senselab": {"version": "2.0.0", "commit": "a" * 40}, "profile": "d" * 64}
        | {"hardware": UNIT.hardware.model_dump()}
    )
    roots = MeasureRoots((shared,), StorePolicy.load_default(), "group", os.geteuid(), stale_after_s=60.0)
    roots.put_observations(unit, "p1", [observation(probe=None, item="i")], provenance)
    assert any(STORE_DIR in str(p) for p in store.rglob("*")) and any(shared.rglob("*.json"))
    assert identity_leaks(store, tmp_path, NEEDLES) == [] and identity_leaks(shared, tmp_path, NEEDLES) == []
    assert (tmp_path / "envs").is_dir() and any(MIRROR in p.read_text() for p in (tmp_path / "envs").glob("*.json"))
