"""The execution profile: packaged profiles, merging, the source allowlist, and `resolve_execution`."""

import platform
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry
from senselab_core.runtime.execution_settings import LeaseBoundError
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo
from senselab_core.runtime.keys import build_key
from senselab_core.runtime.profiles import (
    ExecutionProfile,
    NoBackend,
    NoDevice,
    ProfileFormatRetired,
    UnknownProfileKey,
    UnlistedSource,
    resolve_execution,
)
from senselab_core.runtime.resources import ClassMeasure, Measured, ResourceRow, ResourceTable
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
def test_batch_size_is_sized_from_the_measure_and_not_in_key() -> None:
    """The batch fills the memory headroom beside the resident model; 1 unmeasured or unknown; never in the key."""
    registry = _registry(_backend("b1", "one"))
    measure = ClassMeasure(
        load_s=2.5,
        claim_s=0.01,
        resident_bytes=1024,
        peak_memory_bytes=1280,
        s_per_input_s=0.1,
        input_s_per_item=5.0,
        measured=Measured(sha="1" * 40, jobs=("1",)),
    )
    row = ResourceRow(id="org/m", revision="a" * 40, download_bytes=None, classes={"cpu": measure})
    measured = Catalogue(entries=CAT.entries, resource_tables=(ResourceTable(task="t", backend="b1", rows=(row,)),))
    host = HardwareInfo(cpu_count=1, memory_bytes=10_000, gpus=(), apple_silicon=False, platform="linux-x86_64")
    placements = [
        resolve_execution(TASK, _entry(), _profile(), host, registry, CAT),
        resolve_execution(TASK, _entry(), _profile(), host, registry, measured),
        resolve_execution(TASK, _entry(), _profile(), None, registry, measured),
    ]
    batch = placements[1].batch_size
    room = _profile().sizing.memory_headroom * host.memory_bytes - measure.resident_bytes
    per_item = measure.peak_memory_bytes - measure.resident_bytes
    assert batch * per_item <= room < (batch + 1) * per_item
    assert [p.batch_size for p in placements] == [1, batch, 1] and placements[1].record().batch_size == batch > 1
    call, _ = base_call(TASK, placements[0].backend)
    keys = [build_key(call, p, BASE) for p in placements]
    assert len({k.digest() for k in keys}) == 1 and all("batch_size" not in str(k.model_dump()) for k in keys)
    assert len({p.backend_params.model_dump_json() for p in placements}) == 1 and "batch" not in str(call.params)


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
def test_packaged_profile_comments_every_key_and_holds_no_derivations(name: str) -> None:
    """Each packaged profile validates, carries a `#` comment above every top-level key and no `derivations` key."""
    assert ExecutionProfile.load(name).name == name
    path = Path(__file__).parents[2] / "src/senselab_core/runtime/data/profiles" / f"{name}.yaml"
    lines = path.read_text().splitlines()
    keys = [(i, line.split(":")[0]) for i, line in enumerate(lines) if line[:1].isalpha()]
    assert "derivations" not in {k for _, k in keys}
    uncommented = [k for i, k in keys if i == 0 or not lines[i - 1].startswith("# ")]
    assert not uncommented, f"keys without a comment line: {uncommented}"
    with pytest.raises(UnknownProfileKey, match="derivations"):
        ExecutionProfile.load(name, {"derivations": {"network": "prose"}})


@pytest.mark.logic
def test_env_root_expands_home_and_refuses_unset_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The orcd roots sit under `~/orcd/scratch` unless `SENSELAB_ENV_ROOT` names one; an unset variable is an error."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.setenv("SENSELAB_ENV_ROOT", str(tmp_path / "group"))
    assert ExecutionProfile.load("orcd").env_root("isolated") == tmp_path / "group"
    assert ExecutionProfile.load("orcd").env_root("server") == tmp_path / "group" / "servers"
    monkeypatch.delenv("SENSELAB_ENV_ROOT")
    root = tmp_path / "orcd" / "scratch" / "senselab-2x-venvs"
    assert ExecutionProfile.load("orcd").env_root("isolated") == root
    assert ExecutionProfile.load("orcd").env_root("server") == root / "servers"
    with pytest.raises(ValueError, match="not set"):
        ExecutionProfile.load("orcd", {"env_roots": {"isolated": "$SENSELAB_UNSET_ROOT/v"}}).env_root("isolated")
    registry = _registry(_backend("b1", "one", env=Env.isolated("x")))
    placed = resolve_execution(TASK, _entry(), ExecutionProfile.load("orcd"), NO_GPU, registry, CAT)
    assert placed.env_root == str(root) and "isolated:x" in placed.explain()


@pytest.mark.logic
def test_retry_block_keeps_its_format_identity() -> None:
    """The profile's `retry` is a `retry-policy@1.0` value whose digest is fixed."""
    retry = ExecutionProfile.load().retry
    assert retry.format == "retry-policy@1.0"
    assert retry.canonical_digest() == "dbab4021d94897ef0d33b44d2bae468b549f250744218db112bd209ea0850457"


@pytest.mark.logic
def test_profile1_refused(tmp_path: Path) -> None:
    """An `execution-profile@1.0` or `@2.0` override or file is refused as retired, naming the rewrite; 3.0 ships."""
    for retired in ("execution-profile@1.0", "execution-profile@2.0"):
        with pytest.raises(ProfileFormatRetired, match="execution-profile@3.0"):
            ExecutionProfile.load("laptop", {"format": retired})
    path = tmp_path / "site.yaml"
    path.write_text("format: execution-profile@1.0\nnetwork: offline\n")
    with pytest.raises(ProfileFormatRetired, match="retired format execution-profile@1.0"):
        ExecutionProfile.load("orcd", path)
    assert {ExecutionProfile.load(n).format for n in ("laptop", "orcd")} == {"execution-profile@3.0"}


@pytest.mark.logic
def test_resource_class_follows_device_order_and_host() -> None:
    """The class is the first of the first listed device the backend declares and, judged here, the host has."""
    registry = _registry(_backend("b1", "one", devices=("cpu", "cuda")))
    on_gpu = resolve_execution(TASK, _entry(), _profile(), GPU, registry, CAT)
    assert (on_gpu.resource_class, on_gpu.device, on_gpu.partition) == ("gpu", "cuda", None)
    assert resolve_execution(TASK, _entry(), _profile(), NO_GPU, registry, CAT).resource_class == "cpu"
    orcd = ExecutionProfile.load("orcd")
    scheduled = resolve_execution(TASK, _entry(), orcd, None, registry, CAT)
    assert (scheduled.resource_class, scheduled.partition) == ("gpu:h100", "${GPU_PARTITION}")
    other_gpu = GPU.__class__(**{**GPU.__dict__, "gpus": (GpuInfo(name="A100", memory_bytes=1, driver="1"),)})
    assert resolve_execution(TASK, _entry(), orcd, other_gpu, registry, CAT).resource_class == "gpu"
    cpu_only = ExecutionProfile.load("orcd", {"devices": ["cpu"]})
    assert resolve_execution(TASK, _entry(), cpu_only, other_gpu, registry, CAT).resource_class == "cpu"


@pytest.mark.logic
def test_resource_class_names_match_their_device() -> None:
    """A class named for one device but declaring another is refused, as is a misspelt class key."""
    with pytest.raises(ValueError, match="not named for their device"):
        _profile(resource_classes={"gpu:h100": {"device": "cuda", "max_workers": 1}})
    with pytest.raises(UnknownProfileKey, match="resource_classes.cpu.max_worker.*did you mean 'max_workers'"):
        _profile(resource_classes={"cpu": {"max_worker": 2}})
    with pytest.raises(UnknownProfileKey, match="'pool.quantum'"):
        _profile(pool={"quantum": 2})


@pytest.mark.logic
@pytest.mark.parametrize("name", ["laptop", "orcd"])
def test_each_packaged_lease_beats_inside_its_attribute_delay(name: str) -> None:
    """Each packaged profile loads: its beat interval plus its attribute delay stays below its stale bound."""
    lease = ExecutionProfile.load(name).lease
    assert 2 * lease.renew_fraction * lease.stale_after_s + lease.attribute_delay_s < lease.stale_after_s


@pytest.mark.logic
def test_a_lease_whose_beat_cannot_outrun_the_attribute_delay_is_refused() -> None:
    """2 × 0.33 × 300 + 102 ≥ 300: a lease one beat late could look stale, so load refuses it, naming the values."""
    with pytest.raises(LeaseBoundError, match=r"0\.33.*300.*102"):
        ExecutionProfile.load("orcd", {"lease": {"attribute_delay_s": 102}})
    assert ExecutionProfile.load("orcd", {"lease": {"attribute_delay_s": 101}}).lease.attribute_delay_s == 101
