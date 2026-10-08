"""Tests for the install engine: env identity, staged installs from verified copies, pip check, smoke and R envs."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from packaging.markers import Marker

from senselab.utils.reproduce import env as envmod
from senselab.utils.reproduce import sandbox
from senselab.utils.reproduce.build import WheelResult
from senselab.utils.reproduce.locks import (
    ReproduceLockError,
    entry_version,
    file_sha256,
    lock_entries,
    requirements_digest_for,
    write_header,
)
from senselab.utils.reproduce.spec import EnvSpec, LockRef, PipCheckExpected, Profile, SdistSpec, TorchSpec

DATA = Path(__file__).parent / "data"
ENV_DIR = "envs/a"
UV = "/fake/uv"
CALLER_ENV = {"UV_CACHE_DIR": "/cache/uv", "UV_PYTHON_INSTALL_DIR": "/cache/py", "SSL_CERT_FILE": "/certs/ca.pem"}
ALLOWED_KEYS = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "SSL_CERT_FILE"}
PYPI = "https://pypi.org/simple"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
WHEEL_NAME = "python_speech_features-0.6-py3-none-any.whl"
PSF_URL = "https://files.example/python_speech_features-0.6.tar.gz"
PSF_SHA = "a0aebf746464bc929dc3162cb369d7ff967c398c5120ddf5fb40a65f01b92b11"
needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")


def _copy(base: Path, fixture: str, name: str) -> str:
    rel = f"{ENV_DIR}/{name}"
    (base / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / fixture, base / rel)
    return rel


def _lock(base: Path, rel: str) -> dict[str, str]:
    return {"lock": rel, "lock_sha256": file_sha256(base / rel)}


def _env(
    base: Path,
    *,
    source: str = "uv-export",
    torch: bool = True,
    sdist: bool = True,
    overlays: bool = True,
    body_fixture: str = "body_small.lock",
) -> dict[str, Any]:
    """Lay out an env under ``base`` and return its manifest ``env`` mapping."""
    body = _copy(base, body_fixture, "body.lock")
    env: dict[str, Any] = {
        "python": "3.11",
        "python_exact": "3.11.15",
        "torch": None,
        "body": {**_lock(base, body), "source": source},
        "exclude_newer": "2026-07-01",
        "max_cuda": "12.8",
    }
    if torch:
        fixture = "torch_small.lock" if source == "uv-export" else "torch_index_small.lock"
        env["torch"] = {"index_url": TORCH_INDEX, **_lock(base, _copy(base, fixture, "torch.lock"))}
    if sdist:
        patch = _copy(base, "sdist_small.diff", "pkg.diff")
        env["sdist_allow"] = [
            {
                "name": "python_speech_features",
                "url": PSF_URL,
                "sha256": PSF_SHA,
                "patches": [{"file": patch, "sha256": file_sha256(base / patch)}],
                "build_env": _lock(base, _copy(base, "psf_build.lock", "build.lock")),
            }
        ]
    if overlays:
        env["test"] = _lock(base, _copy(base, "test_small.lock", "test.lock"))
        tools = _copy(base, "test_small.lock", "notebook_tools.lock")
        (base / tools).write_text("# notebook tools\n" + (base / tools).read_text())
        env["notebook_tools"] = _lock(base, tools)
    if source == "uv-export":
        env["body"]["export_cmd"] = ["uv", "export", "--frozen", "--no-emit-project", "--no-dev"]
    else:
        req = _copy(base, "requirements_small.in", "requirements.in")
        config = _copy(base, "uv_config_small.toml", "uv.toml")
        overrides = _copy(base, "overrides_small.txt", "overrides.txt")
        env["body"].update(
            requirements_in=req,
            requirements_in_sha256=file_sha256(base / req),
            config_file=config,
            config_file_sha256=file_sha256(base / config),
            overrides=overrides,
            overrides_sha256=file_sha256(base / overrides),
        )
        env["first_resolve"] = {
            "file": _copy(base, "first_resolve_small.txt", "first_resolve.txt"),
            "sha256": file_sha256(DATA / "first_resolve_small.txt"),
        }
        digest = requirements_digest_for(EnvSpec.model_validate(env), base)
        (base / body).write_text(write_header((base / body).read_text(), {"requirements-sha256": digest}))
        env["body"]["lock_sha256"] = file_sha256(base / body)
    return env


def _uv_lock(tmp_path: Path) -> Path:
    path = tmp_path / "repo" / "uv.lock"
    path.parent.mkdir(parents=True)
    shutil.copyfile(DATA / "uv_small.lock", path)
    return path


def _marker_true(marker: str) -> bool:
    return not marker or Marker(marker).evaluate({"extra": ""})


class FakeUv:
    """Stands in for ``env._run``: records each call, emulates ``uv venv``, installs, freeze and check."""

    def __init__(self) -> None:
        """Start with no calls and a clean ``uv pip check``."""
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []
        self.read: dict[str, str] = {}
        self.installed: dict[str, list[str]] = {}
        self.check_output = "Checked 7 packages in 1ms\nAll installed packages are compatible\n"
        self.check_rc = 0
        self.drop_from_freeze: str | None = None
        self.sandboxed: list[list[str]] = []

    def __call__(self, argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        """Record ``argv`` and ``env`` and emulate uv; a sandbox-wrapped call is unwrapped and also noted."""
        if argv and argv[0] == sandbox.UNSHARE[0]:
            argv = argv[argv.index("-c") + 3 :]
            self.sandboxed.append(list(argv))
        self.calls.append((list(argv), env))
        out = ""
        rc = 0
        if argv[1] == "venv":
            created = Path(argv[-1])
            (created / "bin").mkdir(parents=True)
            (created / "bin" / "python").write_text("")
            self.installed[str(created)] = []
        elif argv[1:3] == ["pip", "install"]:
            venv = str(Path(argv[argv.index("--python") + 1]).parent.parent)
            if "-r" in argv:
                path = argv[argv.index("-r") + 1]
                text = Path(path).read_text()
                self.read[path] = text
                lines = [
                    f"{e.name}=={envmod.local_free(entry_version(e))}"
                    for e in lock_entries(text)
                    if _marker_true(e.marker)
                ]
            else:
                wheel = Path(argv[-1])
                self.read[str(wheel)] = hashlib.sha256(wheel.read_bytes()).hexdigest()
                lines = [f"{wheel.name.split('-')[0]} @ {wheel.as_uri()}"]
            self.installed[venv] += lines
        elif argv[1:3] == ["pip", "freeze"]:
            venv = str(Path(argv[argv.index("--python") + 1]).parent.parent)
            lines = sorted(set(self.installed[venv]))
            out = "\n".join(line for line in lines if line.split("=")[0].split(" ")[0] != self.drop_from_freeze)
        elif argv[1:3] == ["pip", "check"]:
            out, rc = self.check_output, self.check_rc
        return subprocess.CompletedProcess(argv, rc, out, "")

    def installs(self, venv: Path) -> list[list[str]]:
        """The ``uv pip install`` argvs aimed at ``venv``."""
        py = str(venv / "bin" / "python")
        return [a for a, _ in self.calls if a[1:3] == ["pip", "install"] and py in a]


class FakeWheels:
    """Stands in for ``env.ensure_wheel`` and ``env.reuse_wheel``: a wheelhouse and ``wheels.lock`` of demo wheels."""

    def __init__(self) -> None:
        """Start with no calls and nothing recorded; new builds hold ``content``."""
        self.calls: list[dict[str, Any]] = []
        self.content = b"demo wheel"
        self.recorded: dict[str, tuple[Path, str]] = {}

    def __call__(
        self,
        sdist: SdistSpec,
        base_dir: Path,
        wheelhouse: Path,
        wheels_lock: Path,
        torch_lock: TorchSpec | None = None,
        **kw: object,
    ) -> WheelResult:
        """Reuse the recorded wheel unless ``rebuild``; otherwise store ``content`` as a new wheel and record it."""
        self.calls.append({"sdist": sdist.name, "torch_lock": torch_lock, "wheels_lock": wheels_lock, **kw})
        if not kw.get("rebuild") and sdist.name in self.recorded:
            wheel, sha = self.recorded[sdist.name]
            return WheelResult(wheel, sha, file_sha256(wheels_lock), False)
        sha = hashlib.sha256(self.content).hexdigest()
        wheel = wheelhouse / sha / WHEEL_NAME
        wheel.parent.mkdir(parents=True, exist_ok=True)
        wheel.write_bytes(self.content)
        self.recorded[sdist.name] = (wheel, sha)
        wheels_lock.parent.mkdir(parents=True, exist_ok=True)
        wheels_lock.write_text("".join(f"{n} {s}\n" for n, (_, s) in sorted(self.recorded.items())))
        return WheelResult(wheel, sha, file_sha256(wheels_lock), True)

    def reuse(
        self, wheels_lock: Path, sdist: SdistSpec, wheelhouse: Path, torch_lock: object = None, **kw: object
    ) -> tuple[Path, str] | None:
        """The recorded wheel for ``sdist``, if any."""
        return self.recorded.get(sdist.name)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeUv, FakeWheels]:
    """Replace the uv runner, the wheel builder and the uv path; pin the caller's uv and CA variables."""
    uv, wheels = FakeUv(), FakeWheels()
    monkeypatch.setattr(envmod, "_run", uv)
    monkeypatch.setattr(envmod, "ensure_wheel", wheels)
    monkeypatch.setattr(envmod, "reuse_wheel", wheels.reuse)
    monkeypatch.setattr(envmod, "find_uv", lambda: UV)
    for key, value in CALLER_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PYTHONPATH", "/planted")
    monkeypatch.setenv("UV_INDEX_URL", "https://planted.example/simple")
    return uv, wheels


def _prefetch(tmp_path: Path, env: EnvSpec, **kw: Any) -> envmod.PrefetchResult:  # noqa: ANN401
    kw.setdefault("wheels_lock", tmp_path / "manifests" / ENV_DIR / "wheels.lock")
    return envmod.prefetch_env(
        env, tmp_path / "manifests", tmp_path / "cache", analysis="demo", wheelhouse=tmp_path / "wh", **kw
    )


# env_key and venv names


def _key_env(**changes: Any) -> EnvSpec:  # noqa: ANN401
    raw: dict[str, Any] = {
        "python": "3.11",
        "python_exact": "3.11.15",
        "torch": {"index_url": TORCH_INDEX, "lock": "t.lock", "lock_sha256": "1" * 64},
        "body": {"lock": "b.lock", "lock_sha256": "2" * 64, "source": "uv-export", "export_cmd": ["uv", "export"]},
        "sdist_allow": [
            {
                "name": "pkg",
                "url": "https://x/pkg-1.tar.gz",
                "sha256": "3" * 64,
                "patches": [{"file": "p.diff", "sha256": "4" * 64}],
                "build_env": {"lock": "be.lock", "lock_sha256": "5" * 64},
                "build_env_vars": {"MAX_JOBS": "4"},
                "build_modules": ["gcc/12"],
                "torch_cuda_arch_list": "8.0",
            }
        ],
        "exclude_newer": "2026-07-01",
        "wheels": {"lock": "w.lock", "lock_sha256": "6" * 64},
        "test": {"lock": "x.lock", "lock_sha256": "7" * 64},
        "max_cuda": "12.8",
    }
    for path, value in changes.items():
        node: Any = raw
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[int(key)] if key.isdigit() else node[key]
        node[keys[-1]] = value
    return EnvSpec.model_validate(raw)


def test_env_key_is_stable() -> None:
    """The key is a 64-hex digest that depends only on the env's contents."""
    key = envmod.env_key(_key_env())
    assert key == envmod.env_key(_key_env()) and len(key) == 64 and int(key, 16) >= 0


@pytest.mark.parametrize(
    "path,value",
    [
        ("torch.lock_sha256", "a" * 64),
        ("torch", None),
        ("body.lock_sha256", "a" * 64),
        ("sdist_allow.0.name", "other"),
        ("sdist_allow.0.url", "https://x/pkg-2.tar.gz"),
        ("sdist_allow.0.sha256", "a" * 64),
        ("sdist_allow.0.patches", [{"file": "p.diff", "sha256": "a" * 64}]),
        ("sdist_allow.0.build_env", {"lock": "be.lock", "lock_sha256": "a" * 64}),
        ("sdist_allow.0.build_env_vars", {"MAX_JOBS": "8"}),
        ("sdist_allow.0.build_modules", ["gcc/13"]),
        ("sdist_allow.0.torch_cuda_arch_list", "9.0"),
        ("python_exact", "3.11.14"),
        ("max_cuda", "12.9"),
    ],
)
def test_env_key_changes_with_each_input(path: str, value: Any) -> None:  # noqa: ANN401
    """Changing any single input changes the key."""
    assert envmod.env_key(_key_env(**{path: value})) != envmod.env_key(_key_env())


@pytest.mark.parametrize(
    "path,value",
    [
        ("wheels.lock_sha256", "a" * 64),
        ("test.lock_sha256", "a" * 64),
        ("torch.lock", "elsewhere/t.lock"),
        ("body.lock", "elsewhere/b.lock"),
        ("smoke_imports", ["numpy"]),
    ],
)
def test_env_key_ignores_outputs_and_file_locations(path: str, value: Any) -> None:  # noqa: ANN401
    """Wheel sha256s (outputs), overlays and lock file locations do not change the base key."""
    assert envmod.env_key(_key_env(**{path: value})) == envmod.env_key(_key_env())


def test_venv_names(tmp_path: Path) -> None:
    """Base venvs are ``repro-<analysis>-<key8>``; derived venvs add ``+<overlay sha8>``."""
    env = _key_env()
    key = envmod.env_key(env)
    assert envmod.venv_dir(tmp_path, "demo", env) == tmp_path / "venvs" / f"repro-demo-{key[:8]}"
    assert env.test is not None
    derived = envmod.derived_venv_dir(tmp_path, "demo", env, env.test)
    assert derived.name == f"repro-demo-{key[:8]}+{'7' * 8}"
    with pytest.raises(ValueError):
        envmod.venv_dir(tmp_path, "../x", env)


# prefetch_env with a fake uv


def test_prefetch_stage_argv_uv_export(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """The exact argv per stage; every lock is read from a verified copy outside the manifest repo."""
    uv, wheels = fake
    base = tmp_path / "manifests"
    env = EnvSpec.model_validate(_env(base))
    result = _prefetch(tmp_path, env, uv_lock=_uv_lock(tmp_path))
    venv = result.base.path
    py = str(venv / "bin" / "python")
    common = [UV, "pip", "install", "--no-config", "--python", py, "--no-deps"]
    calls = uv.installs(venv)
    assert [a[:7] for a in calls] == [common] * 3
    torch_argv, body_argv, wheel_argv = calls
    assert torch_argv[7:-1] == ["--require-hashes", "--only-binary", ":all:", "-r"]
    assert body_argv[7:-1] == ["--require-hashes", "--only-binary", ":all:", "--index-url", PYPI, "-r"]
    assert wheel_argv[7:-1] == [] and wheel_argv[-1].endswith(WHEEL_NAME)
    venv_calls = [a for a, _ in uv.calls if a[1] == "venv"]
    assert venv_calls[0] == [UV, "venv", "--no-config", "--python", "3.11.15", str(venv)]
    # Every -r and wheel path is a copy: outside the manifest repo and the wheelhouse, gone afterwards,
    # and holding the verified bytes.
    for path, text in uv.read.items():
        assert not Path(path).is_relative_to(base) and not Path(path).is_relative_to(tmp_path / "wh")
        assert not Path(path).exists()
    copies = {Path(p).name: t for p, t in uv.read.items()}
    assert copies["torch.lock"] == (base / ENV_DIR / "torch.lock").read_text()
    assert copies["body.lock"] == (base / ENV_DIR / "body.lock").read_text()
    assert copies[WHEEL_NAME] == hashlib.sha256(b"demo wheel").hexdigest()
    # Allowlisted env: nothing of the caller's beyond the uv dirs and CA bundle.
    for argv, run_env in uv.calls:
        assert run_env is not None and run_env["PATH"] == "/usr/bin:/bin"
        if argv in uv.sandboxed:
            assert argv[1:3] == ["pip", "freeze"] and set(run_env) == ALLOWED_KEYS | {
                "UV_OFFLINE",
                "UV_PYTHON_DOWNLOADS",
            }
        else:
            assert set(run_env) == ALLOWED_KEYS and run_env["UV_CACHE_DIR"] == "/cache/uv"
    assert wheels.calls[0]["python"] == "3.11.15" and wheels.calls[0]["expected_sha256"] is None
    assert result.wheels_lock_sha256 == file_sha256(base / ENV_DIR / "wheels.lock") and result.record
    assert result.built == ("python_speech_features",)


def test_prefetch_index_compiled_torch_uses_index(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """An index-compiled torch lock installs with ``--index-url <torch.index_url>``."""
    uv, _ = fake
    env = EnvSpec.model_validate(_env(tmp_path / "manifests", source="requirements.in", overlays=False))
    result = _prefetch(tmp_path, env)
    torch_argv = uv.installs(result.base.path)[0]
    assert torch_argv[7:-1] == ["--require-hashes", "--only-binary", ":all:", "--index-url", TORCH_INDEX, "-r"]
    assert Path(torch_argv[-1]).name == "torch.lock"


def test_prefetch_derived_venvs(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """Each overlay gets its own venv: the base stages, then the overlay with hashes and binaries only."""
    uv, _ = fake
    env = EnvSpec.model_validate(_env(tmp_path / "manifests"))
    result = _prefetch(tmp_path, env, uv_lock=_uv_lock(tmp_path))
    assert set(result.derived) == {"test", "notebook_tools"}
    for name, derived in result.derived.items():
        ref = getattr(env, name)
        assert derived.path == envmod.derived_venv_dir(tmp_path / "cache", "demo", env, ref)
        assert derived.overlay_sha256 == ref.lock_sha256
        calls = uv.installs(derived.path)
        assert len(calls) == 4
        assert calls[3][7:-1] == ["--require-hashes", "--only-binary", ":all:", "--index-url", PYPI, "-r"]
        assert Path(calls[3][-1]).name == f"{name}.lock"
        installed = json.loads((derived.path / "installed.json").read_text())
        assert "iniconfig" in installed["packages"]
    base_installed = json.loads((result.base.path / "installed.json").read_text())
    assert "iniconfig" not in base_installed["packages"]


def test_installed_json_and_freeze_check(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """``installed.json`` holds the parsed freeze, ``name @ url`` lines included, and equals the locks."""
    uv, _ = fake
    env = EnvSpec.model_validate(_env(tmp_path / "manifests", overlays=False))
    result = _prefetch(tmp_path, env, uv_lock=_uv_lock(tmp_path))
    data = json.loads((result.base.path / "installed.json").read_text())
    assert data["env_key"] == envmod.env_key(env)
    assert data["packages"]["python-speech-features"] == "0.6"
    assert data["packages"]["torch"] == "2.8.0" and data["packages"]["six"] == "1.17.0"
    assert any(line.startswith("python_speech_features @ file://") for line in data["freeze"])
    assert data["packages"] == result.base.packages


def test_freeze_mismatch_fails(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """A venv whose freeze lacks a locked package is refused and left without a completion marker."""
    uv, _ = fake
    uv.drop_from_freeze = "six"
    env = EnvSpec.model_validate(_env(tmp_path / "manifests", torch=False, sdist=False, overlays=False))
    with pytest.raises(ReproduceLockError, match="six"):
        _prefetch(tmp_path, env)
    assert not (envmod.venv_dir(tmp_path / "cache", "demo", env) / envmod.MARKER).exists()


def test_prefetch_is_idempotent(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """A complete venv is reused without any install; one without its marker is rebuilt from scratch."""
    uv, _ = fake
    env = EnvSpec.model_validate(_env(tmp_path / "manifests", torch=False, sdist=False))
    first = _prefetch(tmp_path, env)
    assert not first.base.reused
    n = len(uv.calls)
    second = _prefetch(tmp_path, env)
    assert second.base.reused and all(d.reused for d in second.derived.values())
    assert second.base.packages == first.base.packages
    assert [a for a, _ in uv.calls[n:] if a[1] in ("venv", "pip") and "install" in a or a[1] == "venv"] == []
    (first.base.path / envmod.MARKER).unlink()
    (first.base.path / "stale").write_text("x")
    third = _prefetch(tmp_path, env)
    assert not third.base.reused and not (first.base.path / "stale").exists()


def test_tampered_lock_refused_before_install(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """A lock edited after the manifest recorded its sha256 fails the integrity check; uv never runs."""
    uv, _ = fake
    base = tmp_path / "manifests"
    env = EnvSpec.model_validate(_env(base, torch=False, sdist=False, overlays=False))
    (base / ENV_DIR / "body.lock").write_text((base / ENV_DIR / "body.lock").read_text().replace("1.17.0", "1.16.0"))
    with pytest.raises(ReproduceLockError, match="sha256"):
        _prefetch(tmp_path, env)
    assert uv.calls == []


def test_lock_swapped_after_check_refused(
    tmp_path: Path, fake: tuple[FakeUv, FakeWheels], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock replaced between the integrity check and the copy is caught on the copy; uv never installs."""
    uv, _ = fake
    base = tmp_path / "manifests"
    env = EnvSpec.model_validate(_env(base, torch=False, sdist=False, overlays=False))
    real = envmod.check_integrity

    def check_then_swap(*args: Any, **kw: Any) -> dict[str, str]:  # noqa: ANN401
        verified = real(*args, **kw)
        lock = base / ENV_DIR / "body.lock"
        lock.write_text(lock.read_text().replace("1.17.0", "1.16.0"))
        return verified

    monkeypatch.setattr(envmod, "check_integrity", check_then_swap)
    with pytest.raises(ReproduceLockError, match="copy"):
        _prefetch(tmp_path, env)
    assert [a for a, _ in uv.calls if "install" in a] == []


def test_wheels_lock_from_manifest_is_checked(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """With ``env.wheels`` set, its path and sha256 go to ``ensure_wheel``; a second path is refused."""
    _, wheels = fake
    base = tmp_path / "manifests"
    raw = _env(base, torch=False, overlays=False)
    raw["wheels"] = _lock(base, _copy(base, "wheels_small.lock", "wheels.lock"))
    env = EnvSpec.model_validate(raw)
    envmod.prefetch_env(env, base, tmp_path / "cache", analysis="demo", wheelhouse=tmp_path / "wh")
    call = wheels.calls[0]
    assert call["wheels_lock"] == base / ENV_DIR / "wheels.lock"
    assert call["expected_sha256"] == raw["wheels"]["lock_sha256"] and call["torch_lock"] is None
    with pytest.raises(ValueError, match="wheels"):
        _prefetch(tmp_path, env, wheels_lock=tmp_path / "other.lock")


def test_sdist_without_wheels_lock_location_refused(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """An env with sdists, no ``env.wheels`` and no ``wheels_lock`` argument has nowhere to record wheels."""
    env = EnvSpec.model_validate(_env(tmp_path / "manifests", torch=False, overlays=False))
    with pytest.raises(ValueError, match="wheels"):
        _prefetch(tmp_path, env, wheels_lock=None)


def test_torch_lock_passed_to_wheel_builds(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """The env's torch lock goes to every sdist build, and the recorded sha256 chains across sdists."""
    _, wheels = fake
    base = tmp_path / "manifests"
    raw = _env(base, overlays=False)
    second = copy.deepcopy(raw["sdist_allow"][0])
    second["name"] = "python-speech-features-two"
    raw["sdist_allow"].append(second)
    env = EnvSpec.model_validate(raw)
    _prefetch(tmp_path, env, uv_lock=_uv_lock(tmp_path))
    assert all(c["torch_lock"] == env.torch for c in wheels.calls)
    first_lock = f"python_speech_features {hashlib.sha256(b'demo wheel').hexdigest()}\n"
    assert wheels.calls[0]["expected_sha256"] is None
    assert wheels.calls[1]["expected_sha256"] == hashlib.sha256(first_lock.encode()).hexdigest()


def _with_wheels(env: EnvSpec, base: Path) -> EnvSpec:
    """``env`` with ``env.wheels`` pointing at the current ``wheels.lock``, as the caller records it."""
    lock = base / ENV_DIR / "wheels.lock"
    return env.model_copy(update={"wheels": LockRef(lock=f"{ENV_DIR}/wheels.lock", lock_sha256=file_sha256(lock))})


def test_existing_wheels_lock_without_manifest_entry_refused(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """With ``env.wheels`` unset, a ``wheels_lock`` file that already exists is refused: record its sha256 first."""
    uv, _ = fake
    env = EnvSpec.model_validate(_env(tmp_path / "manifests", torch=False, overlays=False))
    first = _prefetch(tmp_path, env)
    assert first.record and first.wheels_lock_sha256 is not None
    with pytest.raises(ReproduceLockError, match="record its sha256"):
        _prefetch(tmp_path, env)
    again = _prefetch(tmp_path, _with_wheels(env, tmp_path / "manifests"), wheels_lock=None)
    assert again.base.reused and not again.record and again.wheels_lock_sha256 == first.wheels_lock_sha256


def test_changed_wheel_rebuilds_the_venv(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """A ``wheels.lock`` line now naming another wheel rebuilds the venv; its name does not change."""
    uv, wheels = fake
    base = tmp_path / "manifests"
    env = EnvSpec.model_validate(_env(base, torch=False, overlays=False))
    first = _prefetch(tmp_path, env)
    marker = json.loads((first.base.path / envmod.MARKER).read_text())
    assert marker["wheels"] == {"python_speech_features": hashlib.sha256(b"demo wheel").hexdigest()}
    assert json.loads((first.base.path / "installed.json").read_text())["wheels"] == marker["wheels"]
    env = _with_wheels(env, base)
    assert _prefetch(tmp_path, env, wheels_lock=None).base.reused
    wheels.content = b"rebuilt elsewhere"
    wheels(env.sdist_allow[0], base, tmp_path / "wh", base / ENV_DIR / "wheels.lock", rebuild=True)
    env = _with_wheels(env, base)
    second = _prefetch(tmp_path, env, wheels_lock=None)
    assert second.base.path == first.base.path and not second.base.reused
    new_sha = hashlib.sha256(b"rebuilt elsewhere").hexdigest()
    assert json.loads((second.base.path / envmod.MARKER).read_text())["wheels"] == {"python_speech_features": new_sha}
    assert uv.installs(second.base.path)[-1][-1].endswith(f"{new_sha}/{WHEEL_NAME}")


def test_rebuild_rebuilds_a_complete_venv(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """``rebuild=True`` rebuilds every venv and every wheel, even when all are complete."""
    _, wheels = fake
    base = tmp_path / "manifests"
    env = EnvSpec.model_validate(_env(base, torch=False))
    _prefetch(tmp_path, env)
    env = _with_wheels(env, base)
    result = _prefetch(tmp_path, env, wheels_lock=None, rebuild=True)
    assert not result.base.reused and not any(d.reused for d in result.derived.values())
    assert result.built == ("python_speech_features",) and wheels.calls[-1]["rebuild"] is True


def test_tampered_installed_json_rebuilds(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """An ``installed.json`` that no longer matches the marker's sha256 makes the venv incomplete."""
    env = EnvSpec.model_validate(_env(tmp_path / "manifests", torch=False, sdist=False, overlays=False))
    first = _prefetch(tmp_path, env)
    path = first.base.path / "installed.json"
    path.write_text(path.read_text().replace("1.17.0", "9.9.9"))
    assert not _prefetch(tmp_path, env).base.reused


def test_checks_recorded_only_with_a_profile(
    tmp_path: Path, fake: tuple[FakeUv, FakeWheels], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a profile no venv is marked checked; with one, both checks run once and are recorded."""
    ran: list[str] = []
    monkeypatch.setattr(envmod, "check_pip", lambda venv, env: ran.append(f"pip {venv.name}"))

    def fake_smoke(venv: Path, modules: list[str], profile: Profile) -> list[str]:
        ran.append(f"smoke {venv.name}")
        return []

    monkeypatch.setattr(envmod, "smoke", fake_smoke)
    raw = _env(tmp_path / "manifests", torch=False, sdist=False)
    raw.pop("notebook_tools")
    env = EnvSpec.model_validate(raw)
    plain = _prefetch(tmp_path, env)
    assert not plain.base.checked and not envmod.venv_checked(plain.base.path, env) and ran == []
    checked = _prefetch(tmp_path, env, profile=_profile(tmp_path))
    assert checked.base.checked and checked.derived["test"].checked and len(ran) == 4
    assert envmod.venv_checked(checked.base.path, env)
    assert env.test is not None and envmod.venv_checked(checked.derived["test"].path, env, env.test)
    marker = json.loads((checked.base.path / envmod.MARKER).read_text())
    assert marker["checks"] == {"digest": envmod.checks_digest(env), "pip_check_ok": True, "smoke_ok": True}
    again = _prefetch(tmp_path, env, profile=_profile(tmp_path))
    assert again.base.reused and again.base.checked and len(ran) == 4
    stricter = env.model_copy(update={"smoke_imports": ["numpy"]})
    assert not envmod.venv_checked(checked.base.path, stricter)
    _prefetch(tmp_path, stricter, profile=_profile(tmp_path))
    assert len(ran) == 8


def test_load_venv_reads_the_prefetched_venv_and_never_builds(
    tmp_path: Path, fake: tuple[FakeUv, FakeWheels], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_venv`` describes a prefetched venv from its marker; a missing or tampered one is None."""
    monkeypatch.setattr(envmod, "check_pip", lambda venv, env: None)
    monkeypatch.setattr(envmod, "smoke", lambda venv, modules, profile: [])
    raw = _env(tmp_path / "manifests", torch=False, sdist=False)
    raw.pop("notebook_tools")
    env = EnvSpec.model_validate(raw)
    path = envmod.venv_dir(tmp_path / "cache", "demo", env)
    assert envmod.load_venv(path, env) is None
    built = _prefetch(tmp_path, env, profile=_profile(tmp_path)).base
    loaded = envmod.load_venv(path, env)
    assert loaded is not None and loaded.checked and loaded.reused
    assert (loaded.env_key, loaded.overlay_sha256, loaded.packages) == (built.env_key, None, built.packages)
    stricter = env.model_copy(update={"smoke_imports": ["numpy"]})
    unchecked = envmod.load_venv(path, stricter)
    assert unchecked is not None and not unchecked.checked
    (path / envmod.INSTALLED).write_text("{}")
    assert envmod.load_venv(path, env) is None


def test_failed_check_leaves_venv_unchecked(
    tmp_path: Path, fake: tuple[FakeUv, FakeWheels], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A smoke failure raises and the marker keeps the checks false."""
    monkeypatch.setattr(envmod, "check_pip", lambda venv, env: None)
    monkeypatch.setattr(envmod, "smoke", lambda venv, modules, profile: ["numpy: ImportError"])
    env = EnvSpec.model_validate(_env(tmp_path / "manifests", torch=False, sdist=False, overlays=False))
    with pytest.raises(ReproduceLockError, match="smoke"):
        _prefetch(tmp_path, env, profile=_profile(tmp_path))
    assert not envmod.venv_checked(envmod.venv_dir(tmp_path / "cache", "demo", env), env)


# pip check

PIP_CHECK_OUTPUT = """Checked 9 packages in 1ms
Found 3 incompatibilities
The package `python-dateutil` requires `six>=1.5`, but it's not installed
The package `Foo_Bar` requires `numpy<2`, but `2.4.6` is installed
The package `decord` was built for a different platform
"""


def test_parse_pip_check() -> None:
    """Each incompatibility becomes a normalised (package, kind, detail) violation."""
    assert envmod.parse_pip_check(PIP_CHECK_OUTPUT) == [
        envmod.Violation("decord", "platform", "was built for a different platform"),
        envmod.Violation("foo-bar", "requirement", "requires numpy<2, but 2.4.6 is installed"),
        envmod.Violation("python-dateutil", "requirement", "requires six>=1.5, but it's not installed"),
    ]
    assert envmod.parse_pip_check("Checked 3 packages in 1ms\nAll installed packages are compatible\n") == []


def test_compare_pip_check() -> None:
    """Declared deviations must match exactly: extras and misses are both reported."""
    found = envmod.parse_pip_check(PIP_CHECK_OUTPUT)
    expected = [
        PipCheckExpected(package="decord", kind="platform", detail="was built for a different platform"),
        PipCheckExpected(package="Foo_Bar", kind="requirement", detail="requires numpy<2, but 2.4.6 is installed"),
        PipCheckExpected(
            package="python-dateutil", kind="requirement", detail="requires six>=1.5, but it's not installed"
        ),
    ]
    assert envmod.compare_pip_check(found, expected) == ([], [])
    unexpected, missing = envmod.compare_pip_check(found[1:], expected[:1] + expected[2:])
    assert unexpected == [found[1]] and missing == [envmod.Violation("decord", "platform", expected[0].detail)]
    platform_as_requirement = [PipCheckExpected(package="decord", kind="requirement", detail=expected[0].detail)]
    assert envmod.compare_pip_check(found[:1], platform_as_requirement)[0] == found[:1]


def test_pip_check_runs_uv(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """``pip_check`` runs ``uv pip check`` on the venv with the allowlisted env and parses its output."""
    uv, _ = fake
    uv.check_output, uv.check_rc = PIP_CHECK_OUTPUT, 1
    venv = tmp_path / "v"
    assert len(envmod.pip_check(venv)) == 3
    argv, run_env = uv.calls[-1]
    assert argv == [UV, "pip", "check", "--no-config", "--python", str(venv.absolute() / "bin" / "python")]
    assert uv.sandboxed[-1] == argv
    assert run_env is not None and set(run_env) == ALLOWED_KEYS | {"UV_OFFLINE", "UV_PYTHON_DOWNLOADS"}
    assert run_env["UV_OFFLINE"] == "1" and run_env["UV_CACHE_DIR"] != os.environ.get("UV_CACHE_DIR")
    uv.check_output, uv.check_rc = "error: no interpreter", 2
    with pytest.raises(ReproduceLockError):
        envmod.pip_check(venv)


def test_check_env_raises_on_undeclared(tmp_path: Path, fake: tuple[FakeUv, FakeWheels]) -> None:
    """``check_pip`` passes when the violations equal ``pip_check_expected`` and fails otherwise."""
    uv, _ = fake
    uv.check_output, uv.check_rc = PIP_CHECK_OUTPUT, 1
    found = envmod.parse_pip_check(PIP_CHECK_OUTPUT)
    env = _key_env(pip_check_expected=[v._asdict() for v in found])
    envmod.check_pip(tmp_path / "v", env)
    with pytest.raises(ReproduceLockError, match="decord"):
        envmod.check_pip(tmp_path / "v", _key_env(pip_check_expected=[v._asdict() for v in found[1:]]))


# smoke


def test_smoke_runs_each_import_in_the_sandbox(tmp_path: Path) -> None:
    """One sandboxed, offline ``python -c 'import m'`` per module; failures are returned by module."""
    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(argv: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((argv, env))
        assert cwd.is_dir() and list(cwd.iterdir()) == [] and cwd != Path.cwd()
        bad = argv[-1] == "import broken"
        return subprocess.CompletedProcess(argv, 1 if bad else 0, "", "ModuleNotFoundError" if bad else "")

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    failures = envmod.smoke(venv, ["numpy", "broken"], runner=runner)
    assert [f.split(":")[0] for f in failures] == ["broken"]
    for (argv, env), module in zip(calls, ["numpy", "broken"]):
        assert argv[: len(sandbox.UNSHARE)] == sandbox.UNSHARE
        assert argv[-3:] == [str(venv / "bin" / "python"), "-c", f"import {module}"]
        assert f"--bind {venv} {venv}" in argv[len(sandbox.UNSHARE) + 2]
        assert env["PATH"].startswith(f"{venv}/bin:") and env["HF_HUB_OFFLINE"] == "1"
        assert env["https_proxy"] == sandbox.PROXY and "PYTHONPATH" not in env
    with pytest.raises(ValueError):
        envmod.smoke(venv, ["os; import shutil"], runner=runner)


def test_smoke_timeout_is_a_fixed_error(tmp_path: Path) -> None:
    """An import that outlives the timeout raises ``E_SMOKE_TIMEOUT``."""

    def runner(argv: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, envmod.SMOKE_TIMEOUT)

    with pytest.raises(ReproduceLockError, match="E_SMOKE_TIMEOUT"):
        envmod.smoke(tmp_path / "venv", ["numpy"], runner=runner)


def test_smoke_ignores_modules_in_the_callers_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A module of the same name in the caller's working directory does not satisfy the sandboxed import."""
    if not sandbox.userns_available():
        pytest.skip("no user namespaces here")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "shadowmod.py").write_text("VALUE = 1\n")
    monkeypatch.chdir(caller)
    control = subprocess.run([str(venv / "bin" / "python"), "-c", "import shadowmod"], check=False)
    assert control.returncode == 0
    (failure,) = envmod.smoke(venv, ["shadowmod"])
    assert "No module named 'shadowmod'" in failure
    assert envmod.smoke(venv, ["json"]) == []


# R envs

R_DATE = datetime.date(2026, 5, 1)


def _profile(tmp_path: Path) -> Profile:
    return Profile.model_validate(
        {
            "roots": {"data": str(tmp_path / "data")},
            "results_root": str(tmp_path / "res"),
            "runs_root": str(tmp_path / "runs"),
            "slurm_logs_root": str(tmp_path / "logs"),
            "partitions": {"cpu": {"partition": "p", "cpus": 1, "mem": "1G", "time": "1:00:00"}},
            "build": "cpu",
            "r": {"module": "R/4.5.2", "version": "4.5.2"},
            "modules_init": "/etc/profile.d/modules.sh",
            "deny_patterns_file": str(tmp_path / "deny"),
            "canary_host": "1.1.1.1",
        }
    )


def _r_env(**changes: Any) -> EnvSpec:  # noqa: ANN401
    raw: dict[str, Any] = {
        "python": "3.11",
        "python_exact": "3.11.15",
        "torch": None,
        "body": {"lock": "b.lock", "lock_sha256": "2" * 64, "source": "uv-export", "export_cmd": ["uv", "export"]},
        "exclude_newer": "2026-07-01",
        "r_packages": [{"name": "lme4", "version": "1.1-37"}, {"name": "IRkernel", "version": "1.3.2"}],
        "r_snapshot": R_DATE.isoformat(),
        "r_lib": "{cache}/r/demo",
    }
    raw.update(changes)
    return EnvSpec.model_validate(raw)


class FakeR:
    """Stands in for the R runner: the module-env capture, the download, and the sandboxed offline install.

    ``bash`` holds the module-loading calls (capture and download); ``installs`` the unwrapped sandboxed
    install argvs with their env; ``sandboxed`` the raw install argvs.
    """

    def __init__(self, installed: dict[str, str]) -> None:
        """Report ``installed`` as the library contents."""
        self.installed = installed
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.bash: list[tuple[list[str], dict[str, str]]] = []
        self.installs: list[tuple[list[str], dict[str, str]]] = []
        self.sandboxed: list[list[str]] = []

    def __call__(self, argv: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
        """Record the call; write the files the generated R scripts would write.

        ``.libPaths()`` is emulated: ``R_LIBS``, then ``R_LIBS_USER`` when the env sets it and otherwise a user
        library under ``HOME`` (R's default), then R's own library.
        """
        self.calls.append((argv, env))
        if argv[0] == sandbox.UNSHARE[0]:
            self.sandboxed.append(argv)
            inner = argv[argv.index("-c") + 3 :]
            self.installs.append((inner, env))
            r_lib, r_script, out_dir, repo, venv, session = inner[4:10]
            assert Path(r_script).is_file() and Path(session).is_file() and repo.startswith("file://")
            out = Path(out_dir)
            (out / "installed.tsv").write_text("".join(f"{k}\t{v}\n" for k, v in self.installed.items()))
            (out / "namespaces.tsv").write_text("".join(f"{k}\t{v}\n" for k, v in self.installed.items()))
            user = env.get("R_LIBS_USER") or f"{env['HOME']}/R/x86_64-pc-linux-gnu-library/4.5"
            paths = list(dict.fromkeys([env["R_LIBS"], user, "/mod/r/lib64/R/library"]))
            (out / "lib_paths.txt").write_text("".join(f"{p}\n" for p in paths))
            (out / "session_info.txt").write_text("R version 4.5.2\nother attached packages: lme4\n")
            (out / "r_version.txt").write_text("R version 4.5.2 (2025-10-31)\n")
            if venv:
                kernel = Path(venv) / "share" / "jupyter" / "kernels" / "ir"
                kernel.mkdir(parents=True)
                (kernel / "kernel.json").write_text("{}")
            return subprocess.CompletedProcess(argv, 0, "", "")
        self.bash.append((argv, env))
        script = argv[4]
        if "env -0" in script:
            captured = "PATH=/mod/r/bin:/usr/bin:/bin\0LD_LIBRARY_PATH=/mod/r/lib:/mod/blas/lib\0HOME=/x\0"
            return subprocess.CompletedProcess(argv, 0, captured, "")
        _, download, repo, contrib = argv[6:10]
        assert Path(download).is_file() and repo.startswith(envmod.PPM_CRAN + "/")
        Path(contrib).mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_r_snapshot_url() -> None:
    """The repository is the Posit Package Manager snapshot of the manifest date."""
    assert envmod.r_snapshot_url(R_DATE) == "https://packagemanager.posit.co/cran/2026-05-01"


def test_prefetch_r_writes_session_and_module_env(tmp_path: Path) -> None:
    """A matching library writes SESSION.json and MODULE_ENV.json; a second call reuses them."""
    runner = FakeR({"lme4": "1.1-37", "IRkernel": "1.3.2", "Matrix": "1.7-3"})
    env, cache = _r_env(), tmp_path / "cache"
    result = envmod.prefetch_r(env, cache, _profile(tmp_path), runner=runner)
    r_lib = cache / "r" / "demo"
    assert result.r_lib == r_lib and not result.reused
    session = json.loads((r_lib / "SESSION.json").read_text())
    assert session["key"]["snapshot"] == "https://packagemanager.posit.co/cran/2026-05-01"
    assert session["installed"]["Matrix"] == "1.7-3" and session["lib_paths"][0] == str(r_lib)
    assert session["key"]["module"] == "R/4.5.2" and session["session_info"][0] == "R version 4.5.2"
    assert session["key"]["r_version"] == "4.5.2" and session["key"]["build_modules"] == []
    module_env = json.loads((r_lib / "MODULE_ENV.json").read_text())
    assert module_env == {
        "module": "R/4.5.2",
        "build_modules": [],
        "PATH": "/mod/r/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/mod/r/lib:/mod/blas/lib",
    }
    assert result.session_sha256 == file_sha256(r_lib / "SESSION.json")
    assert result.module_env_sha256 == file_sha256(r_lib / "MODULE_ENV.json")
    for argv, run_env in runner.bash:
        assert argv[:4] == ["bash", "--noprofile", "--norc", "-c"] and argv[5:7] == ["bash", "R/4.5.2"]
        assert argv[4].startswith('. /etc/profile.d/modules.sh && module load "$1"')
        assert set(run_env) <= {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE"}
    assert len(runner.bash) == 2 and "Rscript --vanilla" in runner.bash[1][0][4]
    ((install, install_env),) = runner.installs
    assert install[:3] == ["/bin/sh", "-c", envmod.R_INSTALL_SH] and "--vanilla" in install[2]
    assert install_env["R_LIBS"] == install_env["R_LIBS_USER"] == str(r_lib) and install_env["R_LIBS_SITE"] == ""
    assert install_env["PATH"] == "/mod/r/bin:/usr/bin:/bin" and install_env["LD_LIBRARY_PATH"].startswith("/mod/r")
    script = runner.sandboxed[0][runner.sandboxed[0].index("-c") + 1]
    assert f"--bind {r_lib} {r_lib}" in script and "ip link set lo up" in script
    assert session["lib_paths"] == [str(r_lib), "/mod/r/lib64/R/library"]
    n = len(runner.calls)
    again = envmod.prefetch_r(env, cache, _profile(tmp_path), runner=runner)
    assert again.reused and len(runner.calls) == n and again.session_sha256 == result.session_sha256


def test_prefetch_r_fails_unless_installed_equals_declared(tmp_path: Path) -> None:
    """A version other than the declared one, or a missing package, fails and writes no SESSION.json."""
    for installed in ({"lme4": "1.1-38", "IRkernel": "1.3.2"}, {"lme4": "1.1-37"}):
        cache = tmp_path / str(len(installed))
        with pytest.raises(ReproduceLockError, match="lme4|IRkernel"):
            envmod.prefetch_r(_r_env(), cache, _profile(tmp_path), runner=FakeR(installed))
        assert not (cache / "r" / "demo" / "SESSION.json").exists()


def test_prefetch_r_ir_notebook_installs_kernel_into_venv(tmp_path: Path) -> None:
    """For ``ir_notebook`` the kernel spec goes into the derived venv, whose ``bin`` leads ``PATH``."""
    runner = FakeR({"lme4": "1.1-37", "IRkernel": "1.3.2"})
    venv = tmp_path / "venv"
    envmod.prefetch_r(_r_env(), tmp_path / "c", _profile(tmp_path), kind="ir_notebook", venv=venv, runner=runner)
    ((install, _),) = runner.installs
    assert install[8] == str(venv) and 'PATH="$5/bin:$PATH"' in install[2]
    assert (venv / "share" / "jupyter" / "kernels" / "ir" / "kernel.json").is_file()
    kernels = venv / "share" / "jupyter" / "kernels"
    raw = runner.sandboxed[0]
    assert f"--bind {kernels} {kernels}" in raw[raw.index("-c") + 1]
    script = Path(install[5])
    assert not script.exists()  # the generated script lives in a removed temp dir
    with pytest.raises(ValueError, match="IRkernel"):
        envmod.prefetch_r(
            _r_env(r_packages=[{"name": "lme4", "version": "1.1-37"}]),
            tmp_path / "d",
            _profile(tmp_path),
            kind="ir_notebook",
            venv=venv,
            runner=runner,
        )
    with pytest.raises(ValueError, match="venv"):
        envmod.prefetch_r(_r_env(), tmp_path / "e", _profile(tmp_path), kind="ir_notebook", runner=runner)


def test_r_download_script_resolves_dependencies_and_writes_an_index() -> None:
    """The download script resolves recursive dependencies against the snapshot and writes ``PACKAGES``."""
    script = envmod.r_download_script(_r_env().r_packages)
    assert 'c("lme4", "IRkernel")' in script and "package_dependencies" in script and "recursive = TRUE" in script
    assert "download.packages" in script and "write_PACKAGES" in script and "install.packages" not in script


def test_r_script_lists_declared_packages() -> None:
    """The generated script installs and loads every declared package and writes the four outputs."""
    script = envmod.r_script(_r_env().r_packages)
    assert 'c("lme4", "IRkernel")' in script
    for name in ("installed.tsv", "lib_paths.txt", "session_info.txt", "r_version.txt", "installspec"):
        assert name in script
    with pytest.raises(ValueError):
        envmod.r_script(_r_env(r_packages=[{"name": 'x"); system("id', "version": "1"}]).r_packages)


def test_prefetch_r_needs_a_modules_init(tmp_path: Path) -> None:
    """Without ``profile.modules_init`` the R module cannot load: ``E_MODULES_INIT``, nothing removed."""
    cache = tmp_path / "c"
    (cache / "r" / "demo" / "keep").mkdir(parents=True)
    profile = _profile(tmp_path).model_copy(update={"modules_init": None})
    with pytest.raises(ReproduceLockError, match="E_MODULES_INIT"):
        envmod.prefetch_r(_r_env(), cache, profile, runner=FakeR({}))
    assert (cache / "r" / "demo" / "keep").is_dir()


def test_prefetch_r_needs_profile_module(tmp_path: Path) -> None:
    """Without ``profile.r`` there is no R to load."""
    profile = _profile(tmp_path).model_copy(update={"r": None})
    with pytest.raises(ValueError, match="profile.r"):
        envmod.prefetch_r(_r_env(), tmp_path / "c", profile, runner=FakeR({}))


def test_session_is_identical_across_fresh_prefetches(tmp_path: Path) -> None:
    """Two fresh prefetches, each with its own temp HOME, give byte-identical SESSION.json files."""
    shas = []
    homes = []
    for _ in range(2):
        runner = FakeR({"lme4": "1.1-37", "IRkernel": "1.3.2"})
        cache = tmp_path / "cache"
        session = cache / "r" / "demo" / "SESSION.json"
        session.unlink(missing_ok=True)
        result = envmod.prefetch_r(_r_env(), cache, _profile(tmp_path), runner=runner)
        assert not result.reused
        shas.append(result.session_sha256)
        homes.append(runner.installs[-1][1]["HOME"])
        assert runner.installs[-1][1]["R_LIBS_USER"] == str(cache / "r" / "demo")
    assert homes[0] != homes[1] and shas[0] == shas[1]


def test_prefetch_r_rebuilds_a_stale_library(tmp_path: Path) -> None:
    """A library whose SESSION.json is missing or has another key is emptied before the install."""
    cache = tmp_path / "cache"
    r_lib = cache / "r" / "demo"
    (r_lib / "oldpkg").mkdir(parents=True)
    envmod.prefetch_r(_r_env(), cache, _profile(tmp_path), runner=FakeR({"lme4": "1.1-37", "IRkernel": "1.3.2"}))
    assert not (r_lib / "oldpkg").exists() and (r_lib / "SESSION.json").is_file()
    (r_lib / "oldpkg").mkdir()
    newer = _r_env(r_snapshot="2026-06-01")
    runner = FakeR({"lme4": "1.1-37", "IRkernel": "1.3.2"})
    result = envmod.prefetch_r(newer, cache, _profile(tmp_path), runner=runner)
    assert not result.reused and not (r_lib / "oldpkg").exists()
    assert json.loads((r_lib / "SESSION.json").read_text())["key"]["snapshot"].endswith("2026-06-01")
    with pytest.raises(ReproduceLockError):
        envmod.prefetch_r(_r_env(), cache, _profile(tmp_path), runner=FakeR({"lme4": "0.1", "IRkernel": "1.3.2"}))
    assert not (r_lib / "SESSION.json").exists()


def test_prefetch_r_checks_the_r_version(tmp_path: Path) -> None:
    """The module must provide the R version the profile names."""
    profile = _profile(tmp_path)
    assert profile.r is not None
    other = profile.model_copy(update={"r": profile.r.model_copy(update={"version": "4.4.3"})})
    with pytest.raises(ReproduceLockError, match="4.4.3"):
        envmod.prefetch_r(_r_env(), tmp_path / "c", other, runner=FakeR({"lme4": "1.1-37", "IRkernel": "1.3.2"}))


def test_prefetch_r_loads_build_modules_first(tmp_path: Path) -> None:
    """``profile.r.build_modules`` load before the R module, for both the env capture and the install."""
    profile = _profile(tmp_path)
    assert profile.r is not None
    with_gcc = profile.model_copy(update={"r": profile.r.model_copy(update={"build_modules": ["gcc/12.2.0"]})})
    runner = FakeR({"lme4": "1.1-37", "IRkernel": "1.3.2"})
    result = envmod.prefetch_r(_r_env(), tmp_path / "c", with_gcc, runner=runner)
    for argv, _ in runner.bash:
        assert argv[4].startswith('. /etc/profile.d/modules.sh && module load gcc/12.2.0 "$1" && ')
    assert json.loads(result.module_env.read_text())["build_modules"] == ["gcc/12.2.0"]
    assert json.loads(result.session.read_text())["key"]["build_modules"] == ["gcc/12.2.0"]
    with pytest.raises(ValueError):
        envmod.r_fetch_script(["gcc; rm -rf /"], Path("/etc/profile.d/modules.sh"))


def test_r_env_vars_merge_order() -> None:
    """Module ``LD_LIBRARY_PATH`` entries come first, then native lib dirs; R_LIBS pins the library."""
    module_env = {"module": "R/4.5.2", "PATH": "/mod/r/bin:/usr/bin:/bin", "LD_LIBRARY_PATH": "/mod/r/lib:/nat/lib"}
    out = envmod.r_env_vars(
        module_env, "{cache}/r/demo", path_prefix=["{venv}/bin"], native_lib_dirs=["/nat/lib", "/ff/lib"]
    )
    assert out == {
        "PATH": "{venv}/bin:/mod/r/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/mod/r/lib:/nat/lib:/ff/lib",
        "R_LIBS": "{cache}/r/demo",
        "R_LIBS_USER": "{cache}/r/demo",
        "R_LIBS_SITE": "",
    }
    assert "LD_LIBRARY_PATH" not in envmod.r_env_vars({"PATH": "/usr/bin"}, "/r")


# slow: a real torch-free env


def _slow_env(base: Path, body_fixture: str, expected: list[dict[str, str]] | None = None) -> EnvSpec:
    raw = _env(base, torch=False, body_fixture=body_fixture)
    raw.pop("notebook_tools")
    sdist = raw["sdist_allow"][0]
    sdist.pop("patches")
    sdist["name"] = "python-speech-features"
    url, sha = _psf_from_uv_lock()
    sdist.update(url=url, sha256=sha)
    raw["smoke_imports"] = ["numpy", "dateutil", "six"]
    raw["pip_check_expected"] = expected or []
    return EnvSpec.model_validate(raw)


def _psf_from_uv_lock() -> tuple[str, str]:
    import tomllib

    for package in tomllib.loads((DATA / "uv_small.lock").read_text())["package"]:
        if package["name"] == "python-speech-features":
            return package["sdist"]["url"], package["sdist"]["hash"].removeprefix("sha256:")
    raise AssertionError("python-speech-features is not in uv_small.lock")


def _drop_entry(text: str, name: str) -> str:
    out, skip = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith(f"{name}=="):
            skip = True
        elif skip and not line.startswith(" "):
            skip = False
        if not skip:
            out.append(line)
    return "".join(out)


@pytest.mark.slow
@pytest.mark.network
@needs_userns
def test_real_env_pip_check_freeze_and_smoke(tmp_path: Path) -> None:
    """A real body + sdist + test-overlay env: clean pip check, freeze equals locks, smoke passes in the sandbox.

    Deleting one dependency line from a copy of the body (manifest sha256 updated) makes pip check report it.
    """
    base, cache, wheelhouse = tmp_path / "manifests", tmp_path / "cache", tmp_path / "wh"
    wheels_lock = base / ENV_DIR / "wheels.lock"
    env = _slow_env(base, "body_deps_small.lock")
    profile = _profile(tmp_path) if sandbox.userns_available() else None
    result = envmod.prefetch_env(
        env, base, cache, analysis="demo", wheelhouse=wheelhouse, wheels_lock=wheels_lock, profile=profile
    )
    assert result.built == ("python-speech-features",) and result.record
    assert result.base.checked == (profile is not None) == result.derived["test"].checked
    declared = {e.name: e.pin for e in lock_entries((DATA / "body_deps_small.lock").read_text())}
    assert result.base.packages == {**declared, "python-speech-features": "0.6"}
    assert envmod.installed_from_metadata(result.base.path) == result.base.packages
    assert envmod.pip_check(result.base.path) == []
    derived = result.derived["test"]
    assert derived.packages == {**result.base.packages, "iniconfig": "2.3.0"}
    assert envmod.installed_from_metadata(derived.path) == derived.packages
    assert envmod.pip_check(derived.path) == []
    if profile is None:
        pytest.skip("no user namespaces here: smoke not run")
    assert envmod.smoke(result.base.path, env.smoke_imports) == []
    # The sdist's wheel declares no requirements but imports scipy, which no lock holds: pip check
    # cannot see that, the sandboxed import can.
    (failure,) = envmod.smoke(result.base.path, ["python_speech_features"])
    assert failure.startswith("python_speech_features:") and "scipy" in failure
    assert envmod.smoke(derived.path, ["iniconfig"]) == [] and envmod.smoke(result.base.path, ["iniconfig"]) != []
    env = _with_wheels(env, base)
    again = envmod.prefetch_env(env, base, cache, analysis="demo", wheelhouse=wheelhouse, profile=profile)
    assert again.base.reused and again.base.checked and again.built == () and not again.record

    # C1 negative 2: a body with the six line deleted, manifest sha256 updated.
    neg_base = tmp_path / "neg"
    neg = _slow_env(neg_base, "body_deps_small.lock")
    body = neg_base / ENV_DIR / "body.lock"
    body.write_text(_drop_entry(body.read_text(), "six"))
    neg = neg.model_copy(update={"body": neg.body.model_copy(update={"lock_sha256": file_sha256(body)})})
    shutil.copyfile(wheels_lock, neg_base / ENV_DIR / "wheels.lock")
    neg = _with_wheels(neg, neg_base)
    neg_result = envmod.prefetch_env(neg, neg_base, cache, analysis="demo", wheelhouse=wheelhouse)
    assert neg_result.built == ()
    assert neg_result.base.path != result.base.path and "six" not in neg_result.base.packages
    violations = envmod.pip_check(neg_result.base.path)
    missing_six = envmod.Violation("python-dateutil", "requirement", "requires six>=1.5, but it's not installed")
    assert violations == [missing_six]
    with pytest.raises(ReproduceLockError, match="python-dateutil"):
        envmod.check_pip(neg_result.base.path, neg)
    declared_env = neg.model_copy(update={"pip_check_expected": [PipCheckExpected(**violations[0]._asdict())]})
    envmod.check_pip(neg_result.base.path, declared_env)
    shutil.rmtree(cache / "venvs")


@pytest.mark.slow
@pytest.mark.network
@needs_userns
def test_real_prefetch_on_nfs(nfs_dir: Path) -> None:
    """One real prefetch with the manifests, cache, wheelhouse and ``wheels.lock`` on NFS."""
    base = nfs_dir / "manifests"
    env = _slow_env(base, "body_deps_small.lock")
    result = envmod.prefetch_env(
        env,
        base,
        nfs_dir / "cache",
        analysis="demo",
        wheelhouse=nfs_dir / "wh",
        wheels_lock=base / ENV_DIR / "wheels.lock",
    )
    assert result.built == ("python-speech-features",) and result.record
    assert result.base.packages["python-speech-features"] == "0.6"
    assert envmod.pip_check(result.base.path) == []
    with envmod._flock(nfs_dir / "x.lock"):
        pass


def test_r_session_script_and_reader_share_prefetch_form(tmp_path: Path) -> None:
    """The run-time script records what ``r_script`` records; the reader drops the namespace block."""
    packages = _r_env().r_packages
    session, install = envmod.r_session_script(packages), envmod.r_script(packages)
    tail = session.split("pkgs <- c(", 1)[1].split("\n", 1)[1]
    assert install.endswith(tail) and "install.packages" not in session
    (tmp_path / "installed.tsv").write_text("lme4\t1.1-37\n")
    (tmp_path / "lib_paths.txt").write_text("/lib/a\n/r/lib\n")
    (tmp_path / "namespaces.tsv").write_text("Matrix\t1.7-3\nlme4\t1.1-37\n")
    (tmp_path / "session_info.txt").write_text(
        "R version 4.5.2\n\nother attached packages:\n[1] lme4_1.1-37\n\nloaded via a namespace:\n[1] tools\n"
    )
    (tmp_path / "r_version.txt").write_text("R version 4.5.2 (2025-10-31)\n")
    assert envmod.read_r_session(tmp_path) == {
        "installed": {"lme4": "1.1-37"},
        "namespaces": {"Matrix": "1.7-3", "lme4": "1.1-37"},
        "lib_paths": ["/lib/a", "/r/lib"],
        "r_version": "R version 4.5.2 (2025-10-31)",
        "session_info": ["R version 4.5.2", "other attached packages:", "[1] lme4_1.1-37"],
    }
    assert set(envmod.R_SESSION_FIELDS) == set(envmod.read_r_session(tmp_path))
