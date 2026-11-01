"""The execution profile: packaged profiles, merging, the source allowlist, and `resolve_execution`."""

import platform
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo
from senselab_core.runtime.keys import build_key
from senselab_core.runtime.profiles import (
    ExecutionProfile,
    NoBackend,
    NoDevice,
    UnknownProfileKey,
    UnlistedSource,
    resolve_execution,
)
from senselab_core.testing.contracts import BASE, base_call

TASK = TaskSpec(
    name="t", version="1", kind="transform", inputs={"payload": "json"}, outputs={"value": "sha256"}, params=TaskParams
)
NO_GPU = HardwareInfo(cpu_count=1, memory_bytes=1, gpus=(), apple_silicon=False, platform="linux-x86_64")
GPU = HardwareInfo(
    cpu_count=1,
    memory_bytes=1,
    gpus=(GpuInfo(name="H100", memory_bytes=1, driver="1"),),
    apple_silicon=False,
    platform="linux-x86_64",
)


def _backend(name: str, stack: str, devices: tuple[str, ...] = ("cpu",), env: Env | None = None) -> BackendSpec:
    return BackendSpec(
        task="t",
        name=name,
        version="1",
        stack=stack,
        env=env or Env.host(),
        devices=devices,
        batch_invariant=True,
        worker="senselab_worker.fake",
        load_stub={},
    )


def _registry(*backends: BackendSpec) -> Registry:
    registry = Registry()
    registry.add_task(TASK)
    for b in backends:
        registry.add_backend(b)
    return registry


def _entry(model_id: str = "org/m") -> CatalogueEntry:
    return CatalogueEntry.model_validate(
        {
            "id": model_id,
            "revision": "a" * 40,
            "tasks": ["t"],
            "backends": ["b1", "b2"],
            "stacks": ["one", "two"],
            "license": {
                "expression": "MIT",
                "source": "card",
                "verified": {"date": "2026-11-01", "how": "hf-card"},
                "gated": False,
            },
        }
    )


CAT = Catalogue(entries=(_entry(),))


def _profile(**override: Any) -> ExecutionProfile:  # noqa: ANN401
    return ExecutionProfile.load("laptop", override)


@pytest.mark.logic
def test_stack_preference_picks_backend() -> None:
    """The profile's stack order alone decides the backend; the task and its spec are the same."""
    registry = _registry(_backend("b1", "one"), _backend("b2", "two"))
    picks = {
        order: resolve_execution(TASK, _entry(), _profile(stack_preferences={"t": list(order)}), NO_GPU, registry, CAT)
        for order in (("one", "two"), ("two", "one"))
    }
    assert picks[("one", "two")].backend.name == "b1" and picks[("two", "one")].backend.name == "b2"
    assert "stack 'two'" in picks[("two", "one")].explain() and "device cpu" in picks[("two", "one")].explain()
    with pytest.raises(NoBackend, match="available"):
        resolve_execution(TASK, _entry(), _profile(stack_preferences={"t": ["three"]}), NO_GPU, registry, CAT)


@pytest.mark.logic
def test_profile_digest_not_in_key() -> None:
    """Two profiles that differ give different digests and placements that build the same key."""
    registry = _registry(_backend("b1", "one"))
    plain, other = _profile(), _profile(retry={"attempts": 5}, network="offline")
    assert plain.canonical_digest() != other.canonical_digest()
    keys = []
    for profile in (plain, other):
        placement = resolve_execution(TASK, _entry(), profile, NO_GPU, registry, CAT)
        call, _ = base_call(TASK, placement.backend)
        keys.append(build_key(call, placement, BASE).digest())
    assert keys[0] == keys[1]


@pytest.mark.logic
def test_gpu_only_without_gpu_fails() -> None:
    """A CUDA-only backend on a node without a GPU fails naming the node; with a GPU it runs there."""
    registry = _registry(_backend("b1", "one", devices=("cuda",)))
    with pytest.raises(NoDevice, match=platform.node()):
        resolve_execution(TASK, _entry(), _profile(), NO_GPU, registry, CAT)
    assert resolve_execution(TASK, _entry(), _profile(), GPU, registry, CAT).device == "cuda"


@pytest.mark.logic
def test_device_follows_profile_order() -> None:
    """A backend that runs on CPU and CUDA takes the profile's first listed device the host has."""
    registry = _registry(_backend("b1", "one", devices=("cpu", "cuda")))
    assert resolve_execution(TASK, _entry(), _profile(), GPU, registry, CAT).device == "cuda"
    assert resolve_execution(TASK, _entry(), _profile(devices=["cpu", "cuda"]), GPU, registry, CAT).device == "cpu"


@pytest.mark.logic
def test_unknown_profile_key_suggests(tmp_path: Path) -> None:
    """A misspelt key, top level or nested, is refused with the closest known key."""
    with pytest.raises(UnknownProfileKey, match="did you mean 'retry'"):
        _profile(retr={"attempts": 2})
    with pytest.raises(UnknownProfileKey, match="'retry.attempt'.*did you mean 'attempts'"):
        _profile(retry={"attempt": 2})
    path = tmp_path / "site.yaml"
    path.write_text("model_source: [org]\n")
    with pytest.raises(UnknownProfileKey, match="did you mean 'model_sources'"):
        ExecutionProfile.load("laptop", path)


@pytest.mark.logic
def test_unlisted_source_refused() -> None:
    """A model from an organisation outside `model_sources` is refused; the default is the catalogue's."""
    registry = _registry(_backend("b1", "one"))
    with pytest.raises(UnlistedSource, match="'org'"):
        resolve_execution(TASK, _entry(), _profile(model_sources=["other"]), NO_GPU, registry, CAT)
    assert resolve_execution(TASK, _entry(), _profile(model_sources=["org"]), NO_GPU, registry, CAT)
    rogue = _entry("rogue/m")
    with pytest.raises(UnlistedSource, match="'rogue'"):
        resolve_execution(TASK, rogue, _profile(), NO_GPU, registry, CAT)
    assert resolve_execution(TASK, _entry("org/n"), _profile(), NO_GPU, registry, CAT)


@pytest.mark.logic
def test_retry_policy_read_from_profile(tmp_path: Path) -> None:
    """The retry policy is the profile's `retry`, deep-merged over the packaged one."""
    assert (ExecutionProfile.load().retry.attempts, ExecutionProfile.load().retry.first_backoff_s) == (3, 1.0)
    path = tmp_path / "site.yaml"
    path.write_text("retry:\n  attempts: 6\n")
    merged = ExecutionProfile.load("orcd", path).retry
    assert (merged.attempts, merged.first_backoff_s) == (6, 1.0)
    assert not (Path(__file__).parents[2] / "src/senselab_core/runtime/data/retry_policy.yaml").exists()


@pytest.mark.logic
@pytest.mark.parametrize("name", ["laptop", "orcd"])
def test_packaged_profile_loads_with_a_derivation_per_key(name: str) -> None:
    """Each packaged profile validates, and `load` refuses one whose key lacks a derivation."""
    assert ExecutionProfile.load(name).name == name


@pytest.mark.logic
def test_env_root_expands_scratch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The orcd roots sit under `$SCRATCH`; an unset variable is an error, not a literal path."""
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    assert ExecutionProfile.load("orcd").env_root("isolated") == tmp_path / "senselab-2x-venvs"
    monkeypatch.delenv("SCRATCH")
    with pytest.raises(ValueError, match="not set"):
        ExecutionProfile.load("orcd").env_root("isolated")
    registry = _registry(_backend("b1", "one", env=Env.isolated("x")))
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    placed = resolve_execution(TASK, _entry(), ExecutionProfile.load("orcd"), NO_GPU, registry, CAT)
    assert placed.env_root == str(tmp_path / "senselab-2x-venvs") and "isolated:x" in placed.explain()


@pytest.mark.logic
def test_retry_block_keeps_its_format_identity() -> None:
    """The profile's `retry` is a `retry-policy@1.0` value whose digest is fixed."""
    retry = ExecutionProfile.load().retry
    assert retry.format == "retry-policy@1.0"
    assert retry.canonical_digest() == "dbab4021d94897ef0d33b44d2bae468b549f250744218db112bd209ea0850457"
