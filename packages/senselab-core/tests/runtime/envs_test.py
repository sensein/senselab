"""Tests for `senselab_core.runtime.envs`, with `uv` faked by `fake_uv_fixture.py` at its command line."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from senselab_core.kernel.declare import Env
from senselab_core.kernel.identity import ExecutionContext
from senselab_core.runtime.envs import (
    CPU_INDEX,
    LOCK_DIGEST_TAG,
    CudaCompatibilityError,
    EnvHandle,
    EnvLockError,
    HostCuda,
    HostLockLacksBackend,
    LockInputs,
    ManagedPythonMissing,
    TorchIndex,
    Uv,
    compile_lock,
    detect_host_cuda,
    ensure_env,
    lock_digest,
    lock_identity_digest,
    pick_torch_index,
    read_lock_inputs,
)
from senselab_core.runtime.profiles import Envs, ExecutionProfile

pytestmark = pytest.mark.logic

FAKE_UV = Uv((sys.executable, str(Path(__file__).with_name("fake_uv_fixture.py"))))
CUDA_12_8 = HostCuda((12, 8), "nvidia-smi")
PROFILE_INDEX = "https://mirror.example/whl/cu126"
SETTINGS = Envs(lease_stale_after_s=30.0, server_start_margin_s=1.0)


def _lock(
    tmp_path: Path, *, python: str = "3.12", torch: str = "torch==2.8.0 torchaudio==2.8.0", body: str = ""
) -> Path:
    path = tmp_path / "locks" / "demo.txt"
    path.parent.mkdir(exist_ok=True)
    ceiling = "cu128" if torch != "none" else "none"
    path.write_text(
        f"# senselab subprocess-venv lock: demo\n# python: {python}\n# torch: {torch}\n# torch-index: {ceiling}\n"
        f"numpy==2.1.0 \\\n    --hash=sha256:{'0' * 64}\n{body}"
    )
    return path


def _calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.fixture
def uv_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The file the fake uv logs each call to."""
    log = tmp_path / "uv.log"
    monkeypatch.setenv("FAKE_UV_LOG", str(log))
    return log


def _ensure(tmp_path: Path, lock: Path, settings: Envs = SETTINGS) -> EnvHandle:
    return ensure_env(Env.isolated("demo"), lock, tmp_path / "envs", settings=settings, uv=FAKE_UV, host_cuda=CUDA_12_8)


def _smi(version: str) -> str:
    return f"| NVIDIA-SMI 570.86   Driver Version: 570.86   CUDA Version: {version}  |"


@pytest.mark.parametrize(
    ("smi", "nvcc", "ceiling", "tag"),
    [
        (_smi("12.9"), None, None, "cu128"),
        (_smi("12.8"), None, (13, 0), "cu128"),
        (_smi("13.0"), None, (13, 0), "cu130"),
        (_smi("12.9"), None, (12, 1), "cu121"),
        (_smi("12.4"), None, None, "cu124"),
        (_smi("11.8"), None, None, "cpu"),
        (None, "Cuda compilation tools, release 12.6, V12.6.85", None, "cu126"),
        ("garbled", "garbled", None, "cpu"),
        (None, None, None, "cpu"),
    ],
)
def test_cuda_index_routing(smi: str | None, nvcc: str | None, ceiling: tuple[int, int] | None, tag: str) -> None:
    """The probed CUDA, capped by the lock's ceiling, picks the newest index at or below it; no CUDA picks CPU."""
    outputs = {"nvidia-smi": smi, "nvcc": nvcc}

    def run(cmd: list[str]) -> str:
        out = outputs[cmd[0]]
        if out is None:
            raise FileNotFoundError(cmd[0])
        return out

    index = pick_torch_index(detect_host_cuda(run=run), max_cuda=ceiling)
    assert index.tag == tag
    assert index.url == (CPU_INDEX if tag == "cpu" else f"https://download.pytorch.org/whl/{tag}")
    assert index.source == "probe"


def test_profile_torch_index_overrides_probe(tmp_path: Path, uv_log: Path) -> None:
    """The profile's `envs.torch_index` replaces the probed index, and torch installs from it."""
    assert pick_torch_index(CUDA_12_8, override=PROFILE_INDEX) == TorchIndex(PROFILE_INDEX, "profile", "profile")
    handle = _ensure(
        tmp_path, _lock(tmp_path), Envs(lease_stale_after_s=30.0, server_start_margin_s=1.0, torch_index=PROFILE_INDEX)
    )
    (stage_one,) = [c for c in _calls(uv_log) if "--index-url" in c]
    assert stage_one[stage_one.index("--index-url") + 1] == PROFILE_INDEX
    assert handle.torch_index == PROFILE_INDEX


def test_torch_index_recorded_in_context(tmp_path: Path, uv_log: Path) -> None:
    """The env's lock, wheel set and torch index reach the execution context; a torch-free env records no index."""
    handle = _ensure(tmp_path, _lock(tmp_path))
    context = ExecutionContext(
        **handle.context_fields(),
        platform="linux-x86_64",
        numerics={"device_kind": "cuda"},
        decoder=None,
        host="node",
        protocol="p",
        model_source=None,
    )
    assert context.driver["torch_index"] == "https://download.pytorch.org/whl/cu128"
    assert context.env_lock == handle.lock_sha
    assert context.wheel_set == handle.wheel_set
    torch_free = ensure_env(
        Env.isolated("plain"), _lock(tmp_path, torch="none"), tmp_path / "envs", settings=SETTINGS, uv=FAKE_UV
    )
    assert torch_free.torch_index is None
    assert torch_free.context_fields()["driver"] == {}


_DRIVER = """
import sys
from pathlib import Path
from senselab_core.kernel.declare import Env
from senselab_core.runtime.envs import HostCuda, Uv, ensure_env
from senselab_core.runtime.profiles import ExecutionProfile
root, lock, fake = sys.argv[1:]
handle = ensure_env(Env.isolated("demo"), Path(lock), Path(root), settings=ExecutionProfile.load().envs,
                    uv=Uv((sys.executable, fake)),
                    host_cuda=HostCuda((12, 8), "nvidia-smi"))
print(handle.python)
"""


def test_concurrent_build_one_install(tmp_path: Path, uv_log: Path) -> None:
    """Two processes ensuring one env build it once and get the same interpreter."""
    lock = _lock(tmp_path)
    env = {**os.environ, "FAKE_UV_DELAY": "1", "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    argv = [sys.executable, "-c", _DRIVER, str(tmp_path / "envs"), str(lock), FAKE_UV.command[1]]
    procs = [subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in "ab"]
    outs = [p.communicate(timeout=45) for p in procs]
    assert all(p.returncode == 0 for p in procs), [err for _, err in outs]
    assert [c[0] for c in _calls(uv_log)].count("venv") == 1
    assert len({out.strip() for out, _ in outs}) == 1


def test_build_refuses_without_managed_interpreter(
    tmp_path: Path, uv_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no uv-managed interpreter, the build refuses, names `uv python install <ver>` and leaves nothing."""
    monkeypatch.setenv("FAKE_UV_UNMANAGED", "1")
    with pytest.raises(ManagedPythonMissing, match=r"uv python install 3\.14t") as info:
        _ensure(tmp_path, _lock(tmp_path, python="3.14t"))
    assert info.value.version == "3.14t"
    (venv,) = _calls(uv_log)
    assert venv[venv.index("--python-preference") + 1] == "only-managed"
    assert not [p for p in (tmp_path / "envs").iterdir() if p.is_dir()]


def test_recorded_env_names_managed_version(tmp_path: Path, uv_log: Path) -> None:
    """A free-threaded lock builds on the named managed version and the handle records it with `t`."""
    handle = _ensure(tmp_path, _lock(tmp_path, python="3.14t"))
    assert handle.python_version == "3.14.4t"
    venv = _calls(uv_log)[0]
    assert venv[:5] == ["venv", "--python", "3.14t", "--python-preference", "only-managed"]


def test_reuse_and_rebuild_on_changed_lock(tmp_path: Path, uv_log: Path) -> None:
    """An unchanged lock reuses the env; a changed lock rebuilds it and leaves the old build for running workers."""
    first = _ensure(tmp_path, _lock(tmp_path))
    assert _ensure(tmp_path, _lock(tmp_path)) == first
    assert [c[0] for c in _calls(uv_log)].count("venv") == 1
    changed = _ensure(tmp_path, _lock(tmp_path, body="scipy==1.14.0\n"))
    assert changed.lock_sha != first.lock_sha
    assert changed.wheel_set != first.wheel_set
    assert Path(first.python).exists()
    assert [c[0] for c in _calls(uv_log)].count("venv") == 2


def test_missing_wheel_is_cuda_compatibility_error(
    tmp_path: Path, uv_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torch pin the routed index lacks is a typed error naming the profile override."""
    monkeypatch.setenv("FAKE_UV_NO_WHEEL", "1")
    with pytest.raises(CudaCompatibilityError, match=r"torch==2\.8\.0") as info:
        _ensure(tmp_path, _lock(tmp_path))
    assert "envs.torch_index" in str(info.value)
    assert "SENSELAB_TORCH_INDEX_URL" not in str(info.value)


def test_worker_runtime_is_an_overlay(tmp_path: Path, uv_log: Path) -> None:
    """The worker runtime is copied into the env without a distribution, so the wheel set ignores it."""
    handle = _ensure(tmp_path, _lock(tmp_path))
    site = next((Path(handle.python).parent.parent / "lib").glob("python*/site-packages"))
    assert (site / "senselab_worker" / "launch.py").is_file()
    assert not list(site.glob("senselab_worker*.dist-info"))


def test_host_env_requires_a_managed_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A host env is this interpreter, refused when it is not uv-managed."""
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n[[package]]\nname = "numpy"\nversion = "2.1.0"\n')
    monkeypatch.setenv("FAKE_UV_PYTHON_DIR", str(Path(sys.base_prefix).resolve().parent))
    handle = ensure_env(Env.host(), lock, tmp_path, settings=SETTINGS, uv=FAKE_UV, requirements=["numpy"])
    assert handle.python == sys.executable
    assert handle.torch_index is None
    monkeypatch.setenv("FAKE_UV_PYTHON_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(ManagedPythonMissing, match="uv python install"):
        ensure_env(Env.host(), lock, tmp_path, settings=SETTINGS, uv=FAKE_UV, requirements=["numpy"])


def test_other_venv_failure_is_not_managed_python_missing(
    tmp_path: Path, uv_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `uv venv` failure that is not a missing interpreter surfaces as uv's own error, without the install hint."""
    monkeypatch.setenv("FAKE_UV_VENV_FAIL", "1")
    with pytest.raises(RuntimeError, match="Permission denied") as info:
        _ensure(tmp_path, _lock(tmp_path))
    assert not isinstance(info.value, ManagedPythonMissing)


def _plant_lease(tmp_path: Path, age_s: float) -> Path:
    lease = tmp_path / "envs" / "demo-cu128.lease"
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.write_text(json.dumps({"user": "u", "host": "h", "pid": 1, "token": "0" * 32}))
    then = time.time() - age_s
    os.utime(lease, (then, then))
    return lease


def test_profile_lease_bound_breaks_an_expired_build_lease(tmp_path: Path, uv_log: Path) -> None:
    """The profile's `lease_stale_after_s` reaches the lease: older is broken at once, younger is waited out."""
    assert ExecutionProfile.load("orcd").envs.lease_stale_after_s == ExecutionProfile.load().envs.lease_stale_after_s
    _plant_lease(tmp_path, age_s=120.0)
    started = time.monotonic()
    _ensure(tmp_path, _lock(tmp_path), Envs(lease_stale_after_s=60.0, server_start_margin_s=1.0))
    assert time.monotonic() - started < 20  # noqa: PLR2004 - far below the 60 s a live lease would hold it
    lease = _plant_lease(tmp_path, age_s=0.0)
    started = time.monotonic()
    _ensure(tmp_path, _lock(tmp_path, body="scipy==1.14.0\n"), Envs(lease_stale_after_s=1.0, server_start_margin_s=1.0))
    assert time.monotonic() - started >= 0.9  # noqa: PLR2004 - the young lease was held until it aged past 1 s
    assert not lease.exists()


def _uv_lock(tmp_path: Path, *packages: tuple[str, str], extras: dict[str, dict[str, list[str]]] | None = None) -> Path:
    path = tmp_path / "uv.lock"
    text = "version = 1\n"
    for name, version in packages:
        text += f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        if name in (extras or {}):
            text += "[package.optional-dependencies]\n"
            for extra, deps in extras[name].items():  # type: ignore[index]
                text += f"{extra} = [" + ", ".join(f'{{ name = "{d}" }}' for d in deps) + "]\n"
    path.write_text(text)
    return path


def _host(tmp_path: Path, lock: Path, requirements: list[str]) -> EnvHandle:
    return ensure_env(Env.host(), lock, tmp_path, settings=SETTINGS, uv=FAKE_UV, requirements=requirements)


def test_host_refused_when_lock_lacks_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A host env is refused unless the root lock covers every requirement of its backend, version included."""
    monkeypatch.setenv("FAKE_UV_PYTHON_DIR", str(Path(sys.base_prefix).resolve().parent))
    lock = _uv_lock(tmp_path, ("numpy", "2.1.0"), ("Some_Pkg", "1.0"), ("torch", "2.4.0"))
    assert _host(tmp_path, lock, ["numpy>=2", "some-pkg==1.0", "numpy"]).python == sys.executable
    for missing in ("absent-pkg>=1", "some-pkg==2.0", "torch>=2.8,<2.9", "numpy>=3", "pkg @ https://x.example/p.whl"):
        with pytest.raises(HostLockLacksBackend, match="Env.isolated") as info:
            _host(tmp_path, lock, ["numpy", missing])
        assert missing in str(info.value)


def test_host_marker_and_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A requirement whose marker is false here is skipped; an extra needs its locked dependencies."""
    monkeypatch.setenv("FAKE_UV_PYTHON_DIR", str(Path(sys.base_prefix).resolve().parent))
    lock = _uv_lock(tmp_path, ("a", "1.0"), ("b", "1.0"), extras={"a": {"x": ["b"], "y": ["c"]}})
    assert _host(tmp_path, lock, ["a", "absent; python_version < '3'", "a[x]"]).python == sys.executable
    with pytest.raises(HostLockLacksBackend):
        _host(tmp_path, lock, ["a[y]"])
    with pytest.raises(HostLockLacksBackend):
        _host(tmp_path, lock, ["a[nope]"])
    with pytest.raises(HostLockLacksBackend):
        _host(tmp_path, lock, ["absent; python_version >= '3'"])


def test_host_requires_requirements_and_valid_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A host env with no requirements, or an unparsable one, is refused outright."""
    monkeypatch.setenv("FAKE_UV_PYTHON_DIR", str(Path(sys.base_prefix).resolve().parent))
    lock = _uv_lock(tmp_path, ("a", "1.0"))
    for requirements in ([], ["not a requirement!"]):
        with pytest.raises(EnvLockError):
            _host(tmp_path, lock, requirements)


def _inputs(**changes: object) -> LockInputs:
    fields: dict[str, object] = {
        "name": "demo", "requirements": ("numpy>=2", "torch>=2"), "python": "3.12",
        "exclude_newer": "2026-10-02T00:00:00Z", "max_cuda": (13, 0), "quarantined": ("lightning!=2.6.2",),
        "base": ("numpy",),
    }  # fmt: skip
    return LockInputs(**{**fields, **changes})  # type: ignore[arg-type]


def test_lock_header_is_tagged_digest(tmp_path: Path, uv_log: Path) -> None:
    """A compiled lock carries `# identity-digest: <registered tag> <digest of its inputs>`, not a bare sha256."""
    from senselab_worker.digest import IDENTITY_TAGS

    inputs = _inputs()
    text = compile_lock(inputs, uv=FAKE_UV, check_torch_index=False)
    path = tmp_path / "demo.txt"
    path.write_text(text)
    tag, digest = lock_identity_digest(path)
    assert tag == LOCK_DIGEST_TAG and tag in IDENTITY_TAGS
    assert digest == lock_digest(inputs)
    assert "requirements-sha256" not in text and "# torch-index: cu130" in text
    assert "# exclude-newer: 2026-10-02T00:00:00Z" in text and "numpy==1.0.0" in text
    compiles = [c for c in _calls(uv_log) if c[:2] == ["pip", "compile"]]
    assert compiles and all(c[c.index("--python-preference") + 1] == "only-managed" for c in compiles)
    assert all(c[c.index("--exclude-newer") + 1] == inputs.exclude_newer for c in compiles)


@pytest.mark.parametrize(
    "change",
    [
        {"exclude_newer": "2026-10-03T00:00:00Z"},
        {"quarantined": ()},
        {"base": ()},
        {"requirements": ("numpy>=2",)},
        {"python": "3.13"},
        {"max_cuda": (12, 8)},
        {"excluded": ("pyaudio",)},
    ],
)
def test_every_resolution_input_changes_the_digest(change: dict[str, object]) -> None:
    """The digest covers each input of the resolution; the env's name is not one."""
    assert lock_digest(_inputs(**change)) != lock_digest(_inputs())
    assert lock_digest(_inputs(name="other")) == lock_digest(_inputs())


def test_lock_inputs_file_is_read(tmp_path: Path) -> None:
    """`read_lock_inputs` reads the packaged inputs and refuses another format."""
    root = Path(__file__).parents[4] / "packages/senselab/src/senselab/data/env_locks/inputs.yaml"
    inputs = read_lock_inputs(root)
    assert set(inputs) == {"crisperwhisper", "vllm-server", "pii-review"}
    assert inputs["vllm-server"].max_cuda == (13, 0) and inputs["pii-review"].quarantined
    bad = tmp_path / "inputs.yaml"
    bad.write_text("format: other@1.0\n")
    with pytest.raises(EnvLockError, match="format"):
        read_lock_inputs(bad)
