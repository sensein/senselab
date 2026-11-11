"""The execution profile: packaged profiles, merging, the source allowlist, and `resolve_execution`."""

import importlib.util
import os
import platform
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel.declare import BackendSpec, Env, Registry, ServerConfig, TaskParams, TaskSpec
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry
from senselab_core.runtime.execution_settings import LeaseBoundError
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo
from senselab_core.runtime.keys import build_key
from senselab_core.runtime.profiles import (
    ExecutionProfile,
    IncompleteProfile,
    NoBackend,
    NoDevice,
    ProfileFormatRetired,
    UnfilledProfile,
    UnknownProfileKey,
    UnlistedSource,
    packaged_profiles,
    resolve_execution,
)
from senselab_core.testing.contracts import BASE, base_call

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)

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
def test_a_placement_records_batch_one_whatever_the_hardware() -> None:
    """A placement never sizes a batch, a device batch against host memory least of all: it records batch 1."""
    registry = _registry(_backend("b1", "one", devices=("cpu", "cuda")))
    big = HardwareInfo(cpu_count=1, memory_bytes=2**40, gpus=GPU.gpus, apple_silicon=False, platform="linux-x86_64")
    placements = [
        resolve_execution(TASK, _entry(), _profile(), hardware, registry, CAT) for hardware in (big, NO_GPU, None)
    ]
    assert [p.batch_size for p in placements] == [1] * len(placements)
    assert placements[0].device == "cuda" and placements[1].device == "cpu"


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
    merged = ExecutionProfile.load(CLUSTER, path).retry
    assert (merged.attempts, merged.first_backoff_s) == (6, 1.0)
    assert not (Path(__file__).parents[2] / "src/senselab_core/runtime/data/retry_policy.yaml").exists()


@pytest.mark.logic
@pytest.mark.parametrize("name", ["laptop", "slurm"])
def test_packaged_profile_comments_every_key_and_holds_no_derivations(name: str) -> None:
    """Each packaged profile validates, carries a `#` comment above every top-level key and no `derivations` key."""
    assert _packaged(name).name == name
    path = Path(__file__).parents[2] / "src/senselab_core/runtime/data/profiles" / f"{name}.yaml"
    lines = path.read_text().splitlines()
    keys = [(i, line.split(":")[0]) for i, line in enumerate(lines) if line[:1].isalpha()]
    assert "derivations" not in {k for _, k in keys}
    uncommented = [k for i, k in keys if i == 0 or not lines[i - 1].startswith("# ")]
    assert not uncommented, f"keys without a comment line: {uncommented}"
    with pytest.raises(UnknownProfileKey, match="derivations"):
        ExecutionProfile.load(name, {**FILLED.get(name, {}), "derivations": {"network": "prose"}})


@pytest.mark.logic
def test_env_root_expands_home_and_refuses_unset_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`SENSELAB_ENV_ROOT` names the fixture's roots; unset is an error; `${VAR:-fallback}` falls back."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.setenv("SENSELAB_ENV_ROOT", str(tmp_path / "group"))
    assert ExecutionProfile.load(CLUSTER).env_root("isolated") == tmp_path / "group"
    assert ExecutionProfile.load(CLUSTER).env_root("server") == tmp_path / "group" / "servers"
    monkeypatch.delenv("SENSELAB_ENV_ROOT")
    with pytest.raises(ValueError, match="not set"):
        ExecutionProfile.load(CLUSTER).env_root("isolated")
    root = tmp_path / "fallback"
    fallback = {"env_roots": {"isolated": f"${{SENSELAB_ENV_ROOT:-{root}}}"}}
    assert ExecutionProfile.load(CLUSTER, fallback).env_root("isolated") == root
    with pytest.raises(ValueError, match="not set"):
        ExecutionProfile.load(CLUSTER, {"env_roots": {"isolated": "$SENSELAB_UNSET_ROOT/v"}}).env_root("isolated")
    registry = _registry(_backend("b1", "one", env=Env.isolated("x")))
    placed = resolve_execution(TASK, _entry(), ExecutionProfile.load(CLUSTER), NO_GPU, registry, CAT)
    assert "isolated:x" in placed.explain() and str(tmp_path) not in placed.record().model_dump_json()


@pytest.mark.logic
def test_retry_block_keeps_its_format_identity() -> None:
    """The profile's `retry` is a `retry-policy@1.0` value whose digest is fixed."""
    retry = ExecutionProfile.load().retry
    assert retry.format == "retry-policy@1.0"
    assert retry.canonical_digest() == "dbab4021d94897ef0d33b44d2bae468b549f250744218db112bd209ea0850457"


@pytest.mark.logic
def test_profile1_refused(tmp_path: Path) -> None:
    """An `execution-profile@1.0`, `@2.0` or `@3.0` override or file is refused as retired, naming the rewrite."""
    for retired in ("execution-profile@1.0", "execution-profile@2.0", "execution-profile@3.0"):
        with pytest.raises(ProfileFormatRetired, match="execution-profile@4.0"):
            ExecutionProfile.load("laptop", {"format": retired})
    path = tmp_path / "site.yaml"
    path.write_text("format: execution-profile@1.0\nnetwork: offline\n")
    with pytest.raises(ProfileFormatRetired, match="retired format execution-profile@1.0"):
        ExecutionProfile.load(CLUSTER, path)
    assert {ExecutionProfile.load(n).format for n in ("laptop", CLUSTER)} == {"execution-profile@4.0"}
    assert _packaged("slurm").format == "execution-profile@4.0"


@pytest.mark.logic
def test_a_profile3_file_is_refused_naming_the_two_derived_keys(tmp_path: Path) -> None:
    """A site file copied at `execution-profile@3.0`, with probe_items and resolve_s, is refused naming both."""
    path = tmp_path / "site.yaml"
    path.write_text("format: execution-profile@3.0\ncalibration:\n  probe_items: 200\n  resolve_s: 1800\n")
    with pytest.raises(ProfileFormatRetired, match=r"probe_items and calibration\.resolve_s, which are now derived"):
        ExecutionProfile.load(CLUSTER, path)


@pytest.mark.logic
def test_resource_class_follows_device_order_and_host() -> None:
    """The class is the first of the first listed device the backend declares and, judged here, the host has."""
    registry = _registry(_backend("b1", "one", devices=("cpu", "cuda")))
    on_gpu = resolve_execution(TASK, _entry(), _profile(), GPU, registry, CAT)
    assert (on_gpu.resource_class, on_gpu.device, on_gpu.partition) == ("gpu", "cuda", None)
    assert resolve_execution(TASK, _entry(), _profile(), NO_GPU, registry, CAT).resource_class == "cpu"
    cluster = ExecutionProfile.load(CLUSTER)
    scheduled = resolve_execution(TASK, _entry(), cluster, None, registry, CAT)
    assert (scheduled.resource_class, scheduled.partition) == ("gpu:h100", "${GPU_PARTITION}")
    other_gpu = GPU.__class__(**{**GPU.__dict__, "gpus": (GpuInfo(name="A100", memory_bytes=1, driver="1"),)})
    assert resolve_execution(TASK, _entry(), cluster, other_gpu, registry, CAT).resource_class == "gpu"
    cpu_only = ExecutionProfile.load(CLUSTER, {"devices": ["cpu"]})
    assert resolve_execution(TASK, _entry(), cpu_only, other_gpu, registry, CAT).resource_class == "cpu"


@pytest.mark.logic
def test_a_server_backend_takes_a_service_class_first() -> None:
    """A server backend takes a `svc` class before any other; a host backend never does; without one, as usual."""
    host = _backend("b1", "one", devices=("cuda",))
    server = BackendSpec.model_validate(
        {**host.model_dump(), "env": Env.server("e"), "server": ServerConfig(settings={})}
    )
    cluster = ExecutionProfile.load(CLUSTER)
    placed = resolve_execution(TASK, _entry(), cluster, None, _registry(server), CAT)
    assert (placed.resource_class, placed.device) == ("svc:h100", "cuda")
    assert resolve_execution(TASK, _entry(), cluster, GPU, _registry(server), CAT).resource_class == "svc:h100"
    assert resolve_execution(TASK, _entry(), cluster, None, _registry(host), CAT).resource_class == "gpu:h100"
    assert resolve_execution(TASK, _entry(), _profile(), GPU, _registry(server), CAT).resource_class == "gpu"


@pytest.mark.logic
def test_resource_class_names_match_their_device() -> None:
    """A class named for one device but declaring another is refused, as is a misspelt class key."""
    with pytest.raises(ValueError, match="not named for their device"):
        _profile(resource_classes={"gpu:h100": {"device": "cuda", "max_workers": 1}})
    with pytest.raises(UnknownProfileKey, match="resource_classes.cpu.max_worker.*did you mean 'max_workers'"):
        _profile(resource_classes={"cpu": {"max_worker": 2}})
    with pytest.raises(UnknownProfileKey, match="'pool.quantum'"):
        _profile(pool={"quantum": 2})


FILLED: dict[str, dict[str, Any]] = {
    "slurm": {
        "env_roots": {"isolated": "/scratch/envs", "server": "/scratch/envs/servers"},
        "resource_classes": {
            "cpu": {"cpus": 2, "mem_bytes": 2**30, "time_s": 3600, "partition": "part", "max_workers": 2},
            "gpu": {
                "cpus": 2,
                "mem_bytes": 2**30,
                "time_s": 3600,
                "partition": "part",
                "gres": "gpu:1",
                "max_workers": 1,
            },
        },
    }
}


def _packaged(name: str) -> ExecutionProfile:
    return ExecutionProfile.load(name, FILLED.get(name, {}))


@pytest.mark.logic
def test_senselab_ships_only_generic_profiles() -> None:
    """The packaged profiles are the local one and the Slurm template; a site's profile is the user's own file."""
    assert packaged_profiles() == ("laptop", "slurm")
    with pytest.raises(FileNotFoundError, match="neither a packaged profile"):
        ExecutionProfile.load("orcd")
    with pytest.raises(FileNotFoundError, match="not a profile file"):
        ExecutionProfile.load("missing.yaml")


@pytest.mark.logic
def test_a_partial_file_passed_as_a_profile_says_to_use_an_override(tmp_path: Path) -> None:
    """One clear error, not a list of missing fields, and `load(name, file)` is the layering that works."""
    partial = tmp_path / "partial.yaml"
    partial.write_text("format: execution-profile@4.0\nretry:\n  attempts: 6\n  first_backoff_s: 1.0\n")
    with pytest.raises(IncompleteProfile, match=r"must be complete.*override"):
        ExecutionProfile.load(partial)
    assert ExecutionProfile.load("laptop", partial).retry.attempts == 6


@pytest.mark.logic
def test_the_slurm_template_refuses_to_load_until_every_placeholder_is_filled() -> None:
    """Each `<fill: ...>` is named with what goes there; filling them, by override or in a copy, loads it."""
    with pytest.raises(
        UnfilledProfile, match=r"resource_classes\.cpu\.partition \(Slurm partition for CPU workers\)"
    ) as caught:
        ExecutionProfile.load("slurm")
    assert {"env_roots.isolated", "resource_classes.gpu.gres"} <= set(caught.value.unfilled)
    partial = {"resource_classes": {"cpu": FILLED["slurm"]["resource_classes"]["cpu"]}}
    with pytest.raises(UnfilledProfile) as still:
        ExecutionProfile.load("slurm", partial)
    assert all(not key.startswith("resource_classes.cpu") for key in still.value.unfilled)
    loaded = _packaged("slurm")
    assert loaded.substrate.kind == "slurm" and loaded.resource_classes["gpu"].gres == "gpu:1"


@pytest.mark.logic
def test_a_profile_file_is_a_whole_profile_and_a_copy_of_the_template_loads_when_filled(tmp_path: Path) -> None:
    """A path loads that file; a user's filled copy of the template is a site profile, named by its file."""
    template = resources.files("senselab_core.runtime").joinpath("data", "profiles", "slurm.yaml").read_text()
    mine = tmp_path / "mysite.yaml"
    mine.write_text(template.replace("name: slurm", "name: mysite"))
    with pytest.raises(UnfilledProfile):
        ExecutionProfile.load(mine)
    assert ExecutionProfile.load(str(mine), FILLED["slurm"]).name == "mysite"


@pytest.mark.logic
@pytest.mark.parametrize("name", ["laptop", "slurm"])
def test_each_packaged_lease_beats_inside_its_attribute_delay(name: str) -> None:
    """Each packaged profile loads: its beat interval plus its attribute delay stays below its stale bound."""
    lease = _packaged(name).lease
    assert 2 * lease.renew_fraction * lease.stale_after_s + lease.attribute_delay_s < lease.stale_after_s


@pytest.mark.logic
def test_a_lease_whose_beat_cannot_outrun_the_attribute_delay_is_refused() -> None:
    """2 × 0.33 × 300 + 102 ≥ 300: a lease one beat late could look stale, so load refuses it, naming the values."""
    with pytest.raises(LeaseBoundError, match=r"0\.33.*300.*102"):
        ExecutionProfile.load(CLUSTER, {"lease": {"attribute_delay_s": 102}})
    assert ExecutionProfile.load(CLUSTER, {"lease": {"attribute_delay_s": 101}}).lease.attribute_delay_s == 101


@pytest.mark.logic
def test_the_test_profile_is_the_site_file_untouched_or_the_fixture_with_a_temp_env_root(
    profile_loader: Callable[..., ExecutionProfile], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`SENSELAB_TEST_PROFILE` is loaded as it is; without it the fixture's roots go to a temp dir, env unchanged."""
    monkeypatch.delenv("SENSELAB_ENV_ROOT", raising=False)
    monkeypatch.delenv("SENSELAB_TEST_PROFILE", raising=False)
    before = dict(os.environ)
    fixture = profile_loader()
    assert fixture.name == "cluster" and fixture.env_root("isolated").is_dir()
    assert not str(fixture.env_root("isolated")).startswith(str(Path.home()))
    assert fixture.env_root("server") == fixture.env_root("isolated") / "servers"
    assert profile_loader({"retry": {"attempts": 5}}).retry.attempts == 5
    assert dict(os.environ) == before
    monkeypatch.setenv("SENSELAB_ENV_ROOT", str(tmp_path / "given"))
    assert profile_loader().env_root("isolated") == tmp_path / "given"
    site = tmp_path / "site.yaml"
    site.write_text(
        CLUSTER.read_text().replace("name: cluster", "name: mysite").replace("${SENSELAB_ENV_ROOT}", "/site/envs")
    )
    monkeypatch.setenv("SENSELAB_TEST_PROFILE", str(site))
    loaded = profile_loader()
    assert loaded.name == "mysite" and loaded.env_root("isolated") == Path("/site/envs")


@pytest.mark.logic
def test_a_site_test_is_skipped_without_a_site_profile(
    site_profile: Callable[..., ExecutionProfile], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `SENSELAB_TEST_PROFILE` set, `site_profile` is the same loader (the unset case skips at setup)."""
    assert site_profile().name


@pytest.mark.logic
def test_a_run_selecting_model_or_gpu_tests_needs_a_site_profile() -> None:
    """The marker expression names a model or GPU marker positively, or it does not."""
    spec = importlib.util.spec_from_file_location("packages_conftest", CLUSTER.parents[4] / "conftest.py")
    assert spec is not None and spec.loader is not None
    root = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root)
    selects_site_tiers = root.selects_site_tiers

    for chosen in ("model", "limits and model", "gpu", "(env or limits) and model", "large_model"):
        assert selects_site_tiers(chosen), chosen
    for other in (
        "",
        "logic",
        "limits and not model",
        "env and not slow and not model",
        "(logic or contract) and not model",
    ):
        assert not selects_site_tiers(other), other
