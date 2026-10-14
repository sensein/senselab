"""Tests for sdist builds: fetching, the build venv and argv, patches, ``wheels.lock`` and the batch script."""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from senselab.utils.reproduce import build, sandbox
from senselab.utils.reproduce.locks import ReproduceLockError, file_sha256, lock_entries
from senselab.utils.reproduce.spec import FileRef, LockRef, Profile, SdistSpec, TorchSpec
from tests.utils.reproduce.masks_fixture import _tool_masks as _tool_masks  # autouse, also under --noconftest

DATA = Path(__file__).parent / "data"
UV = "/fake/uv"
DATE = datetime.date(2026, 10, 7)
CALLER_ENV = {"UV_CACHE_DIR": "/cache/uv", "UV_PYTHON_INSTALL_DIR": "/cache/py", "SSL_CERT_FILE": "/certs/ca.pem"}
SETUP_PY = "from setuptools import setup\n\nsetup(name='demo', version='1.0', py_modules=['demo'])\n"
PATCH = "--- a/demo.py\n+++ b/demo.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")


def _sdist_bytes() -> bytes:
    """A tiny sdist: ``demo-1.0/{setup.py,demo.py}``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in (("setup.py", SETUP_PY), ("demo.py", "VALUE = 1\n")):
            data = text.encode()
            info = tarfile.TarInfo(f"demo-1.0/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _write(base: Path, rel: str, text: str) -> str:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return rel


def _setup(tmp_path: Path, patch: str | None = None, **fields: Any) -> tuple[SdistSpec, Path]:  # noqa: ANN401
    """A manifest repo with a build lock (and optional patch), and a local sdist served over ``file://``."""
    base = tmp_path / "manifests"
    lock = _write(base, "envs/a/build.lock", (DATA / "psf_build.lock").read_text())
    sdist = tmp_path / "upstream" / "demo-1.0.tar.gz"
    sdist.parent.mkdir(parents=True)
    sdist.write_bytes(_sdist_bytes())
    spec: dict[str, Any] = {
        "name": "demo",
        "url": sdist.as_uri(),
        "sha256": file_sha256(sdist),
        "build_env": {"lock": lock, "lock_sha256": file_sha256(base / lock)},
    }
    if patch is not None:
        rel = _write(base, "envs/a/demo.patch", patch)
        spec["patches"] = [{"file": rel, "sha256": file_sha256(base / rel)}]
    spec.update(fields)
    return SdistSpec.model_validate(spec), base


class FakeRun:
    """Stands in for ``build._run``: records each call; ``uv build`` writes a wheel into ``--out-dir``.

    The wheel's bytes are the source's ``demo.py`` plus the build's ``MAX_JOBS``, so build variables change its sha256.
    """

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []
        self.sources: list[str] = []

    def __call__(self, argv: list[str], env: dict[str, str] | None = None) -> None:
        """Record ``argv`` and ``env``; emulate the wheel build."""
        self.calls.append((list(argv), env))
        if "build" in argv:
            src = Path(argv[-1])
            self.sources.append((src / "demo.py").read_text())
            out = Path(argv[argv.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            jobs = (env or {}).get("MAX_JOBS", "").encode()
            (out / "demo-1.0-py3-none-any.whl").write_bytes(b"wheel:" + (src / "demo.py").read_bytes() + jobs)


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    """Replace the uv runner and version queries; pin the uv path and the caller's uv and CA variables."""
    fake = FakeRun()
    monkeypatch.setattr(build, "_run", fake)
    monkeypatch.setattr(build, "_capture", lambda argv, env=None: f"{Path(argv[-2]).name} 1.0")
    monkeypatch.setattr(build, "uv_executable", lambda: UV)
    for key, value in CALLER_ENV.items():
        monkeypatch.setenv(key, value)
    return fake


def _stored(wheelhouse: Path, content: bytes) -> Path:
    """Where a demo wheel with these bytes is stored."""
    return wheelhouse / hashlib.sha256(content).hexdigest() / "demo-1.0-py3-none-any.whl"


# fetch_sdist


def test_fetch_sdist_verifies_and_reuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A verified download is kept; a second fetch reuses it without opening the URL."""
    spec, _ = _setup(tmp_path)
    path = build.fetch_sdist(spec.url, spec.sha256, tmp_path / "dl")
    assert path == tmp_path / "dl" / "demo-1.0.tar.gz"
    assert file_sha256(path) == spec.sha256

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("downloaded again")

    monkeypatch.setattr(build.urllib.request, "urlopen", refuse)
    assert build.fetch_sdist(spec.url, spec.sha256, tmp_path / "dl") == path


def test_fetch_sdist_sha_mismatch_raises(tmp_path: Path) -> None:
    """A download whose sha256 differs from the manifest raises and leaves nothing behind."""
    spec, _ = _setup(tmp_path)
    with pytest.raises(ReproduceLockError, match="sha256"):
        build.fetch_sdist(spec.url, "0" * 64, tmp_path / "dl")
    assert not any((tmp_path / "dl").iterdir())


def test_fetch_sdist_replaces_a_corrupt_copy(tmp_path: Path) -> None:
    """A present file with the wrong sha256 is downloaded again."""
    spec, _ = _setup(tmp_path)
    dest = tmp_path / "dl"
    dest.mkdir()
    (dest / "demo-1.0.tar.gz").write_bytes(b"corrupt")
    assert file_sha256(build.fetch_sdist(spec.url, spec.sha256, dest)) == spec.sha256


# build_wheel


def test_build_wheel_argv_env_and_patch(tmp_path: Path, fake_run: FakeRun) -> None:
    """The build venv is made from the verified lock copy; the patch is applied; the wheel is stored by sha256."""
    spec, base = _setup(tmp_path, PATCH, build_env_vars={"MAX_JOBS": "4"}, torch_cuda_arch_list="8.0;9.0")
    work, wheelhouse = tmp_path / "work", tmp_path / "wheelhouse"
    wheel, sha = build.build_wheel(spec, base, wheelhouse, python="3.11.15", work_dir=work)
    assert wheel == _stored(wheelhouse, b"wheel:VALUE = 2\n4")
    assert sha == wheel.parent.name == file_sha256(wheel)
    assert fake_run.sources == ["VALUE = 2\n"]
    venv, py = work / "venv", str(work / "venv" / "bin" / "python")
    argvs = [argv for argv, _ in fake_run.calls]
    assert argvs[0] == [UV, "venv", "--no-config", "--python", "3.11.15", str(venv)]
    assert argvs[1] == [
        *(UV, "pip", "install", "--no-config", "--python", py, "--no-deps", "--require-hashes"),
        *("--only-binary", ":all:", "-r", str(work / "inputs" / "envs/a/build.lock")),
    ]
    inner = [
        *(UV, "build", "--no-config", "--wheel", "--no-build-isolation", "--python", py),
        *("--out-dir", str(work / "dist"), str(work / "src" / "demo-1.0")),
    ]
    assert argvs[2][: len(sandbox.UNSHARE)] == sandbox.UNSHARE and argvs[2][-len(inner) :] == inner
    assert f"--bind {venv} {venv}" in argvs[2][len(sandbox.UNSHARE) + 2]
    info = json.loads(wheel.with_name(wheel.name + ".build-info.json").read_text())
    assert info["python_exact"] == "3.11.15"
    assert info["uv"] == "uv 1.0" and info["gcc"] == "gcc 1.0"
    assert info["build_modules"] == [] and info["nvcc"] == "nvcc 1.0"
    assert sorted(p.name for p in wheel.parent.iterdir()) == [wheel.name, wheel.name + ".build-info.json"]


def test_build_env_is_an_allowlist(tmp_path: Path, fake_run: FakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every subprocess gets exactly the allowlisted keys; the build adds the declared, fixed and offline ones."""
    for key, value in {"PYTHONPATH": str(tmp_path / "fake"), "CFLAGS": "-DPLANTED", "CUDA_HOME": "/x"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("UV_INDEX_URL", "https://planted.example/simple")
    spec, base = _setup(tmp_path, build_env_vars={"MAX_JOBS": "4"}, torch_cuda_arch_list="8.0;9.0")
    work = tmp_path / "work"
    build.build_wheel(spec, base, tmp_path / "wheelhouse", python="3.11.15", work_dir=work)
    base_env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(work / "home"),
        "TMPDIR": str(work / "tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **CALLER_ENV,
    }
    for _, env in fake_run.calls[:2]:
        assert env == base_env
    proxy = "http://127.0.0.1:9"
    assert fake_run.calls[2][1] == {
        **base_env,
        "PATH": f"{work / 'venv' / 'bin'}:/usr/bin:/bin",
        "MAX_JOBS": "4",
        "SOURCE_DATE_EPOCH": "315532800",
        "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
        "UV_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
        "UV_CACHE_DIR": str(work / "tmp" / "uv-cache"),
        **{key: proxy for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")},
    }
    assert (work / "home").is_dir() and (work / "tmp").is_dir()


def test_build_env_vars_may_not_replace_fixed_keys(tmp_path: Path, fake_run: FakeRun) -> None:
    """A declared build variable that would replace an allowlisted or fixed one is refused."""
    for key in ("PATH", "SOURCE_DATE_EPOCH", "UV_OFFLINE", "TORCH_CUDA_ARCH_LIST"):
        spec, base = _setup(tmp_path / key, build_env_vars={key: "x"})
        with pytest.raises(ReproduceLockError, match=key):
            build.build_wheel(spec, base, tmp_path / key / "wh", python="3.11.15", work_dir=tmp_path / key / "w")


def test_build_wheel_with_torch_lock(tmp_path: Path, fake_run: FakeRun) -> None:
    """An index-compiled torch lock is installed into the build venv first, with its index URL."""
    spec, base = _setup(tmp_path)
    shutil.copyfile(DATA / "torch_index_small.lock", base / "envs/a/torch.lock")
    torch = TorchSpec(
        lock="envs/a/torch.lock",
        lock_sha256=file_sha256(base / "envs/a/torch.lock"),
        index_url="https://index.example/whl",
    )
    work = tmp_path / "work"
    build.build_wheel(spec, base, tmp_path / "wheelhouse", torch_lock=torch, python="3.11.15", work_dir=work)
    argvs = [argv for argv, _ in fake_run.calls]
    assert argvs[1][-4:] == ["--index-url", "https://index.example/whl", "-r", str(work / "inputs/envs/a/torch.lock")]
    assert argvs[2][-1] == str(work / "inputs" / "envs/a/build.lock")


def test_patch_sha_mismatch_raises_before_build(tmp_path: Path, fake_run: FakeRun) -> None:
    """A patch whose bytes differ from the manifest sha256 stops the build before any uv call or download."""
    spec, base = _setup(tmp_path, PATCH)
    (base / "envs/a/demo.patch").write_text(PATCH.replace("VALUE = 2", "VALUE = 3"))
    with pytest.raises(ReproduceLockError, match="demo.patch"):
        build.build_wheel(spec, base, tmp_path / "wheelhouse", python="3.11.15", work_dir=tmp_path / "work")
    assert fake_run.calls == []
    assert not (tmp_path / "work" / "sdist").exists()


def test_build_lock_sha_mismatch_raises_before_build(tmp_path: Path, fake_run: FakeRun) -> None:
    """A build lock edited after the manifest was written is refused."""
    spec, base = _setup(tmp_path)
    (base / "envs/a/build.lock").write_text((DATA / "psf_build.lock").read_text().replace("84.0.0", "84.0.1"))
    with pytest.raises(ReproduceLockError, match="build.lock"):
        build.build_wheel(spec, base, tmp_path / "wheelhouse", python="3.11.15", work_dir=tmp_path / "work")
    assert fake_run.calls == []


def test_patch_that_does_not_apply_raises(tmp_path: Path, fake_run: FakeRun) -> None:
    """A hashed patch that does not apply cleanly stops the build."""
    spec, base = _setup(tmp_path, PATCH.replace("-VALUE = 1", "-VALUE = 9"))
    with pytest.raises(ReproduceLockError, match="does not apply"):
        build.build_wheel(spec, base, tmp_path / "wheelhouse", python="3.11.15", work_dir=tmp_path / "work")
    assert fake_run.calls == []


def test_build_runs_in_the_sandbox() -> None:
    """The build argv is the sandbox prefix around ``uv build``, the venv read-only, no module loading inside."""
    w = Path("/w")
    writable = [w / "src", w / "dist", w / "home", w / "tmp"]
    argv = build.wheel_build_argv(
        UV, w / "venv/bin/python", w / "src/demo-1.0", w / "dist", venv=w / "venv", writable=writable
    )
    inner = [UV, "build", "--no-config", "--wheel", "--no-build-isolation", "--python", "/w/venv/bin/python"]
    inner += ["--out-dir", "/w/dist", "/w/src/demo-1.0"]
    assert argv[: len(sandbox.UNSHARE)] == sandbox.UNSHARE
    assert argv[len(sandbox.UNSHARE) :] == [sandbox.SH, "-c", argv[len(sandbox.UNSHARE) + 2], "sh", *inner]
    script = argv[len(sandbox.UNSHARE) + 2]
    assert "--bind /w/venv /w/venv" in script and "module" not in script


INIT = Path("/etc/profile.d/modules.sh")


def test_module_variables_reach_the_sandboxed_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Modules load outside the namespace (``in_modules`` around ``env -0``); their variables are passed as values."""
    spec, _ = _setup(tmp_path, build_modules=["gcc/12.2.0", "cuda/12.9.1"], build_env_vars={"MAX_JOBS": "4"})
    seen: list[list[str]] = []
    base = {"PATH": "/usr/bin:/bin", "HOME": "/w/home", "SHLVL": "1"}

    def capture(argv: list[str], env: dict[str, str]) -> str:
        seen.append(argv)
        assert env == base
        return "PATH=/m/gcc/bin:/m/cuda/bin:/usr/bin:/bin\0CUDA_HOME=/m/cuda\0HOME=/w/home\0SHLVL=2\0_=/usr/bin/env\0"

    monkeypatch.setattr(build, "_capture_env", capture)
    module_env = build.module_environment(spec, base, INIT)
    assert module_env == {"PATH": "/m/gcc/bin:/m/cuda/bin:/usr/bin:/bin", "CUDA_HOME": "/m/cuda"}
    (argv,) = seen
    assert argv[:4] == ["bash", "--noprofile", "--norc", "-c"]
    assert argv[4].startswith(". /etc/profile.d/modules.sh && module load gcc/12.2.0 cuda/12.9.1 && ")
    assert "if ! command -v nvcc >/dev/null; then" in argv[4] and "export CUDA_HOME=" in argv[4]
    assert argv[5:] == ["bash", "env", "-0"]
    env = build.build_environment(spec, base, Path("/w/venv"), module_env)
    assert env["PATH"] == "/w/venv/bin:/m/gcc/bin:/m/cuda/bin:/usr/bin:/bin" and env["CUDA_HOME"] == "/m/cuda"
    assert env["MAX_JOBS"] == "4" and env["UV_OFFLINE"] == "1"
    plain, _ = _setup(tmp_path / "plain")
    assert build.module_environment(plain, base) == {} and len(seen) == 1
    with pytest.raises(ReproduceLockError, match="E_MODULES_INIT"):
        build.module_environment(spec, base)


def test_nvcc_is_required_only_for_cuda_builds(tmp_path: Path) -> None:
    """A compiler-only module list loads without the nvcc guard; an arch list or a ``cuda/*`` module adds it."""
    gcc, _ = _setup(tmp_path / "gcc", build_modules=["gcc/12.2.0"])
    script = build.in_modules(gcc, ["true"], INIT)[4]
    assert script == '. /etc/profile.d/modules.sh && module load gcc/12.2.0 && exec "$@"'
    arch, _ = _setup(tmp_path / "arch", build_modules=["gcc/12.2.0"], torch_cuda_arch_list="8.0")
    assert "command -v nvcc" in build.in_modules(arch, ["true"], INIT)[4]
    assert build.needs_nvcc(arch) and not build.needs_nvcc(gcc)
    assert build.build_config_sha256(gcc, None, "3.11.15") != build.build_config_sha256(
        gcc.model_copy(update={"build_modules": ["gcc/13.1.0"]}), None, "3.11.15"
    )


def test_missing_nvcc_fails_the_wrapper(tmp_path: Path) -> None:
    """The wrapper's nvcc guard fails when no nvcc is on PATH (module init replaced by a no-op function)."""
    init = tmp_path / "init.sh"
    init.write_text("module() { :; }\n")
    spec, _ = _setup(tmp_path, build_modules=["cuda/1.0"])
    argv = build.in_modules(spec, ["true"], init)
    result = subprocess.run(argv, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "nvcc not on PATH" in result.stderr


def _tar_with_mode(path: Path, mode: int) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(path.as_posix())
        info.mode = mode
        tar.addfile(info, io.BytesIO(b""))
    return buffer.getvalue()


def _zip_with_mode(path: Path, mode: int) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo(path.as_posix())
        info.external_attr = mode << 16
        archive.writestr(info, b"")
    return buffer.getvalue()


@pytest.mark.parametrize(("make", "filename"), [(_tar_with_mode, "pkg-1.0.tar.gz"), (_zip_with_mode, "pkg-1.0.zip")])
def test_unpacked_modes_ignore_the_callers_umask(tmp_path: Path, make: Any, filename: str) -> None:  # noqa: ANN401
    """A member archived 0o664 unpacks as 0o660 (its mode under ``BUILD_UMASK``) whatever the process umask."""
    data = make(Path("pkg-1.0/a.py"), 0o664)
    old = os.umask(0o077)
    try:
        src = build._unpack(data, filename, tmp_path / "src")
    finally:
        os.umask(old)
    assert (src / "a.py").stat().st_mode & 0o777 == 0o660


# content-addressed wheelhouse


def test_wheelhouse_keeps_both_builds(tmp_path: Path, fake_run: FakeRun) -> None:
    """Two builds of one sdist with different build variables coexist; ``--rebuild`` repoints the line only."""
    lock, wheelhouse = tmp_path / "wheels.lock", tmp_path / "wheelhouse"
    spec1, base = _setup(tmp_path, build_env_vars={"MAX_JOBS": "1"})
    first = build.ensure_wheel(spec1, base, wheelhouse, lock, python="3.11.15", work_dir=tmp_path / "w1")
    spec2 = spec1.model_copy(update={"build_env_vars": {"MAX_JOBS": "2"}})
    with pytest.raises(ReproduceLockError, match="--rebuild"):
        build.ensure_wheel(spec2, base, wheelhouse, lock, python="3.11.15", work_dir=tmp_path / "w2")
    second = build.ensure_wheel(spec2, base, wheelhouse, lock, python="3.11.15", rebuild=True, work_dir=tmp_path / "w3")
    assert first.wheel != second.wheel
    assert first.wheel.is_file() and second.wheel.is_file()
    assert file_sha256(first.wheel) == first.wheel_sha256 != second.wheel_sha256 == file_sha256(second.wheel)
    line = build.read_wheels_lock(lock)["demo"]
    assert line.wheel_path == f"{second.wheel_sha256}/demo-1.0-py3-none-any.whl"
    assert line.build_config_sha256 == build.build_config_sha256(spec2, None, "3.11.15")


def test_wheelhouse_never_overwrites(tmp_path: Path, fake_run: FakeRun) -> None:
    """Other bytes already at the wheel's content address stop the build instead of being replaced."""
    spec, base = _setup(tmp_path)
    target = _stored(tmp_path / "wheelhouse", b"wheel:VALUE = 1\n")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"planted")
    with pytest.raises(ReproduceLockError, match="other bytes"):
        build.build_wheel(spec, base, tmp_path / "wheelhouse", python="3.11.15", work_dir=tmp_path / "w")
    assert target.read_bytes() == b"planted"


def test_ensure_wheel_removes_its_work_dir(tmp_path: Path, fake_run: FakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``work_dir`` the build runs in a temporary directory that is gone afterwards."""
    spec, base = _setup(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    result = build.ensure_wheel(spec, base, tmp_path / "wheelhouse", tmp_path / "wheels.lock", python="3.11.15")
    assert result.built and result.wheel.is_file()
    assert list(scratch.iterdir()) == []


# wheels.lock


def _line(**fields: Any) -> build.WheelLine:  # noqa: ANN401
    values: dict[str, Any] = {
        "name": "demo",
        "version": "1.0",
        "sdist_url": "https://files.example/demo-1.0.tar.gz",
        "sdist_sha256": "1" * 64,
        "build_env_sha256": "2" * 64,
        "patch_sha256s": (),
        "build_config_sha256": "3" * 64,
        "wheel_path": "4" * 64 + "/demo-1.0-py3-none-any.whl",
        "wheel_sha256": "4" * 64,
    }
    values.update(fields)
    return build.WheelLine(**values)


def test_wheels_lock_round_trip(tmp_path: Path) -> None:
    """Lines written are lines read back, sorted by name, under the column header; the write returns the sha256."""
    path = tmp_path / "wheels.lock"
    lines = [_line(name="zeta"), _line(patch_sha256s=("5" * 64, "6" * 64))]
    sha = build.write_wheels_lock(path, lines)
    assert sha == file_sha256(path)
    text = path.read_text().splitlines()
    assert text[0] == build.WHEELS_HEADER
    assert text[1].split()[0] == "demo" and text[2].split()[0] == "zeta"
    assert text[1].split()[5] == "5" * 64 + "," + "6" * 64
    assert text[2].split()[5] == "-"
    assert build.read_wheels_lock(path) == {"demo": lines[1], "zeta": lines[0]}
    assert build.read_wheels_lock(path, sha) == build.read_wheels_lock(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["wheels.lock", "wheels.lock.lock"]


def test_wheels_lock_checked_against_the_manifest(tmp_path: Path) -> None:
    """A ``wheels.lock`` whose sha256 differs from the manifest's, or that is missing, is refused."""
    path = tmp_path / "wheels.lock"
    build.write_wheels_lock(path, [_line()])
    with pytest.raises(ReproduceLockError, match="manifest"):
        build.read_wheels_lock(path, "0" * 64)
    spec, _ = _setup(tmp_path)
    with pytest.raises(ReproduceLockError, match="manifest"):
        build.reuse_wheel(path, spec, tmp_path, python="3.11.15", expected_sha256="0" * 64)
    with pytest.raises(ReproduceLockError, match="missing"):
        build.read_wheels_lock(tmp_path / "absent.lock", "0" * 64)


def test_wheels_lock_rejects_malformed_lines(tmp_path: Path) -> None:
    """A line without nine columns (e.g. the older seven-column layout) is refused."""
    path = tmp_path / "wheels.lock"
    shutil.copyfile(DATA / "wheels_small.lock", path)
    with pytest.raises(ReproduceLockError, match="columns"):
        build.read_wheels_lock(path)


def test_record_and_reuse(tmp_path: Path, fake_run: FakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first ensure builds and records; the second reuses the wheel without building."""
    spec, base = _setup(tmp_path, PATCH)
    lock, wheelhouse = tmp_path / "wheels.lock", tmp_path / "wheelhouse"
    result = build.ensure_wheel(spec, base, wheelhouse, lock, python="3.11.15", work_dir=tmp_path / "w1")
    assert result.built and result.wheels_lock_sha256 == file_sha256(lock)
    recorded = build.read_wheels_lock(lock, result.wheels_lock_sha256)["demo"]
    assert recorded == build.WheelLine(
        name="demo",
        version="1.0",
        sdist_url=spec.url,
        sdist_sha256=spec.sha256,
        build_env_sha256=spec.build_env.lock_sha256,
        patch_sha256s=(spec.patches[0].sha256,),
        build_config_sha256=build.build_config_sha256(spec, None, "3.11.15"),
        wheel_path=f"{result.wheel_sha256}/{result.wheel.name}",
        wheel_sha256=result.wheel_sha256,
    )

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("rebuilt a recorded wheel")

    monkeypatch.setattr(build, "build_wheel", refuse)
    again = build.ensure_wheel(
        spec, base, wheelhouse, lock, python="3.11.15", expected_sha256=result.wheels_lock_sha256
    )
    assert again == build.WheelResult(result.wheel, result.wheel_sha256, result.wheels_lock_sha256, built=False)
    with pytest.raises(ReproduceLockError, match="--rebuild"):
        build.ensure_wheel(spec, base, wheelhouse, lock, python="3.12.11")


def test_stage_wheel_copies_checked_bytes(tmp_path: Path) -> None:
    """The staged copy holds the recorded bytes; a wheel whose bytes changed is refused."""
    wheel = tmp_path / "wh" / "demo-1.0-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"wheel")
    staged = build.stage_wheel(wheel, hashlib.sha256(b"wheel").hexdigest(), tmp_path / "tmp")
    assert staged == tmp_path / "tmp" / wheel.name and staged.read_bytes() == b"wheel"
    with pytest.raises(ReproduceLockError, match="sha256"):
        build.stage_wheel(wheel, "0" * 64, tmp_path / "tmp2")


def test_reuse_refuses_a_changed_wheel(tmp_path: Path) -> None:
    """A wheelhouse wheel whose sha256 differs from its line raises instead of being installed or rebuilt."""
    spec, _ = _setup(tmp_path)
    wheelhouse = tmp_path / "wheelhouse"
    line = _line(
        sdist_url=spec.url,
        sdist_sha256=spec.sha256,
        build_env_sha256=spec.build_env.lock_sha256,
        build_config_sha256=build.build_config_sha256(spec, None, "3.11.15"),
    )
    (wheelhouse / line.wheel_path).parent.mkdir(parents=True)
    (wheelhouse / line.wheel_path).write_bytes(b"tampered")
    build.write_wheels_lock(tmp_path / "wheels.lock", [line])
    with pytest.raises(ReproduceLockError, match="sha256"):
        build.reuse_wheel(tmp_path / "wheels.lock", spec, wheelhouse, python="3.11.15")
    (wheelhouse / line.wheel_path).unlink()
    with pytest.raises(ReproduceLockError, match="missing"):
        build.reuse_wheel(tmp_path / "wheels.lock", spec, wheelhouse, python="3.11.15")
    build.write_wheels_lock(tmp_path / "wheels.lock", [line._replace(wheel_path="demo-1.0-py3-none-any.whl")])
    with pytest.raises(ReproduceLockError, match="<sha256>/<file>"):
        build.reuse_wheel(tmp_path / "wheels.lock", spec, wheelhouse, python="3.11.15")


def test_reuse_refuses_a_stale_line(tmp_path: Path) -> None:
    """A line recorded for other build inputs is refused, naming ``--rebuild``."""
    spec, _ = _setup(tmp_path)
    line = _line(sdist_url=spec.url, sdist_sha256=spec.sha256, build_env_sha256=spec.build_env.lock_sha256)
    build.write_wheels_lock(tmp_path / "wheels.lock", [line])
    with pytest.raises(ReproduceLockError, match="--rebuild"):
        build.reuse_wheel(tmp_path / "wheels.lock", spec, tmp_path, python="3.11.15")
    assert build.reuse_wheel(tmp_path / "absent.lock", spec, tmp_path, python="3.11.15") is None


def test_rebuild_replaces_the_line(tmp_path: Path, fake_run: FakeRun) -> None:
    """``rebuild`` builds even with a recorded line and replaces only that line."""
    spec, base = _setup(tmp_path)
    lock = tmp_path / "wheels.lock"
    other = _line(name="other")
    build.write_wheels_lock(lock, [other, _line(wheel_sha256="9" * 64)])
    with pytest.raises(ReproduceLockError, match="--rebuild"):
        build.ensure_wheel(spec, base, tmp_path / "wh", lock, python="3.11.15", work_dir=tmp_path / "w0")
    result = build.ensure_wheel(
        spec, base, tmp_path / "wh", lock, python="3.11.15", rebuild=True, work_dir=tmp_path / "w1"
    )
    lines = build.read_wheels_lock(lock)
    assert lines["other"] == other
    assert lines["demo"].wheel_sha256 == result.wheel_sha256
    assert lines["demo"].sdist_sha256 == spec.sha256


# render_build_job


def _profile(tmp_path: Path) -> Profile:
    return Profile.model_validate(
        {
            "roots": {"data": "/data/root"},
            "results_root": "/out/results",
            "runs_root": "/out/runs",
            "slurm_logs_root": "/out/slurm",
            "partitions": {
                "cpu": {"partition": "part-cpu", "cpus": 4, "mem": "8G", "time": "01:00:00"},
                "builder": {"partition": "part-build", "cpus": 16, "mem": "64G", "time": "00:15:00", "gres": "gpu:1"},
            },
            "build": "builder",
            "deny_patterns_file": "/etc/deny.txt",
            "canary_host": "192.0.2.1",
            "policy": {
                "min_cell": 5,
                "headline_rule": "balanced_accuracy_only",
                "require_deny_patterns": True,
                "require_canary": True,
                "confidential_provenance": True,
                "umask": "007",
            },
            "job_tmp_root": "/dev/shm",
            "modules_init": str(INIT),
        }
    )


def test_render_build_job(tmp_path: Path) -> None:
    """The batch script uses the build partition, pins uv and the driver, exports the arch list, logs under the root."""
    spec, _ = _setup(tmp_path, build_modules=["cuda/12.9.1"], torch_cuda_arch_list="8.0;9.0")
    text = build.render_build_job(
        _profile(tmp_path),
        spec,
        manifest=Path("/repo/m.yaml"),
        profile_path=Path("/repo/p.yaml"),
        step="train",
        date=DATE,
        driver_python="/opt/py/bin/python",
        uv="/opt/uv/bin/uv",
        environ={"UV_CACHE_DIR": "/cache/uv", "SENSELAB_REPRO_CACHE": "/cache/repro dir", "CFLAGS": "PLANTED"},
    )
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    for flag in (
        "--job-name=reproduce-build-demo",
        "--partition=part-build",
        "--cpus-per-task=16",
        "--mem=64G",
        "--time=00:15:00",
        "--gres=gpu:1",
        "--output=/out/slurm/2026-10-07/build-demo-%j.out",
        "--error=/out/slurm/2026-10-07/build-demo-%j.err",
        "--chdir=/out/slurm/2026-10-07",
    ):
        assert f"#SBATCH {flag}" in lines
    assert "#SBATCH --export=NONE" in lines
    assert lines[lines.index("set -euo pipefail") :] == [
        "set -euo pipefail",
        "export PATH=/usr/bin:/bin",
        "export UV_CACHE_DIR=/cache/uv",
        "export SENSELAB_REPRO_CACHE='/cache/repro dir'",
        "export REPRO_DRIVER_PYTHON=/opt/py/bin/python",
        "export REPRO_UV=/opt/uv/bin/uv",
        'export TMPDIR="/dev/shm/reproduce-build-${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
        "trap 'rm -rf \"$TMPDIR\"' EXIT",
        'export TORCH_CUDA_ARCH_LIST="8.0;9.0"',
        '"$REPRO_DRIVER_PYTHON" -P -m senselab.utils.reproduce build-one /repo/m.yaml'
        " --profile /repo/p.yaml --step train --sdist demo",
    ]
    assert "LD_LIBRARY_PATH" not in text and "PLANTED" not in text and "module" not in text


def test_rendered_prologue_runs_in_an_empty_environment(tmp_path: Path) -> None:
    """Everything before the driver call runs under ``env -i`` with ``set -u`` and exits 0, removing its TMPDIR."""
    spec, _ = _setup(tmp_path, build_modules=["gcc/12.2.0", "cuda/12.9.1"], torch_cuda_arch_list="8.0;9.0")
    text = build.render_build_job(
        _profile(tmp_path),
        spec,
        manifest=Path("/repo/m.yaml"),
        profile_path=Path("/repo/p.yaml"),
        step="train",
        date=DATE,
        driver_python="/opt/py/bin/python",
        uv="/opt/uv/bin/uv",
        environ={"UV_CACHE_DIR": "/cache/uv", "UV_PYTHON_INSTALL_DIR": "/cache/py"},
    )
    lines = text.splitlines()
    assert lines[-1].startswith('"$REPRO_DRIVER_PYTHON" ')
    job = f"pytest-{os.getpid()}"
    prologue = "\n".join([*lines[:-1], 'echo "$TMPDIR $REPRO_UV $TORCH_CUDA_ARCH_LIST"']) + "\n"
    result = subprocess.run(
        ["env", "-i", "PATH=/usr/bin:/bin", f"SLURM_JOB_ID={job}", "bash", "--noprofile", "--norc", "-c", prologue],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [f"/dev/shm/reproduce-build-{job}", "/opt/uv/bin/uv", "8.0;9.0"]
    assert not Path(f"/dev/shm/reproduce-build-{job}").exists()


def test_uv_executable_honours_repro_uv_under_a_minimal_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``REPRO_UV`` wins over ``PATH``; it must be an absolute path to an executable."""
    fake = tmp_path / "bin" / "uv"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o750)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("REPRO_UV", str(fake))
    assert build.uv_executable() == str(fake)
    for bad in ("uv", str(tmp_path / "absent")):
        monkeypatch.setenv("REPRO_UV", bad)
        with pytest.raises(ReproduceLockError, match="REPRO_UV"):
            build.uv_executable()
    fake.chmod(0o640)
    monkeypatch.setenv("REPRO_UV", str(fake))
    with pytest.raises(ReproduceLockError, match="REPRO_UV"):
        build.uv_executable()


def test_uv_executable_in_a_minimal_subprocess(tmp_path: Path) -> None:
    """A driver started with only the system ``PATH`` and ``REPRO_UV`` finds that uv."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("no uv on PATH")
        return
    code = "from senselab.utils.reproduce.build import uv_executable; print(uv_executable())"
    env = {"PATH": "/usr/bin:/bin", "REPRO_UV": uv, "HOME": str(tmp_path)}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == uv


def test_render_build_job_without_a_job_tmp_root(tmp_path: Path) -> None:
    """With no ``job_tmp_root`` in the profile the job sets no ``TMPDIR`` of its own."""
    spec, _ = _setup(tmp_path)
    profile = _profile(tmp_path).model_copy(update={"job_tmp_root": None})
    text = build.render_build_job(
        profile, spec, manifest=Path("/m.yaml"), profile_path=Path("/p.yaml"), step="s", date=DATE, uv="/u"
    )
    assert "TMPDIR" not in text and "/dev/shm" not in text


def test_render_build_job_without_cuda(tmp_path: Path) -> None:
    """A plain sdist renders no module load and no arch list; ``rebuild`` is passed through."""
    spec, _ = _setup(tmp_path)
    text = build.render_build_job(
        _profile(tmp_path),
        spec,
        manifest=Path("/repo/m.yaml"),
        profile_path=Path("/repo/p.yaml"),
        step="train",
        rebuild=True,
        date=DATE,
        driver_python="/opt/py/bin/python",
        uv="/opt/uv/bin/uv",
    )
    assert "module" not in text
    assert "TORCH_CUDA_ARCH_LIST" not in text
    assert text.splitlines()[-1].endswith("--sdist demo --rebuild")


def test_render_build_job_refuses_unsafe_values(tmp_path: Path) -> None:
    """Values that would break out of the script's quoting, and a relative uv, are refused."""
    with pytest.raises(ValidationError):
        _setup(tmp_path, build_modules=["cuda/12.9.1; rm -rf x"])
    spec, _ = _setup(tmp_path / "b", torch_cuda_arch_list='8.0"; echo "')
    with pytest.raises(ValueError, match="torch_cuda_arch_list"):
        build.render_build_job(
            _profile(tmp_path),
            spec,
            manifest=Path("/m.yaml"),
            profile_path=Path("/p.yaml"),
            step="s",
            date=DATE,
            uv="/u",
        )
    plain, _ = _setup(tmp_path / "c")
    with pytest.raises(ValueError, match="absolute"):
        build.render_build_job(
            _profile(tmp_path),
            plain,
            manifest=Path("/m.yaml"),
            profile_path=Path("/p.yaml"),
            step="s",
            date=DATE,
            uv="uv",
        )


# slow: a real pure-Python sdist build


def _psf_from_uv_lock() -> tuple[str, str]:
    """The python-speech-features sdist URL and sha256 from the ``uv_small.lock`` fixture."""
    import tomllib

    for package in tomllib.loads((DATA / "uv_small.lock").read_text())["package"]:
        if package["name"] == "python-speech-features":
            return package["sdist"]["url"], package["sdist"]["hash"].removeprefix("sha256:")
    raise AssertionError("python-speech-features is not in uv_small.lock")


def _spike_wheel_sha() -> str:
    """The wheel sha256 the spike recorded for python-speech-features 0.6 (``wheels_small.lock``, column 6)."""
    (line,) = [ln for ln in (DATA / "wheels_small.lock").read_text().splitlines() if ln.startswith("python-speech")]
    return line.split()[5]


@pytest.mark.slow
@pytest.mark.network
@needs_userns
def test_build_python_speech_features_matches_spike(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Building python_speech_features 0.6 from its real sdist reproduces the spike's wheel bit for bit.

    A broken ``setuptools`` on the caller's ``PYTHONPATH`` and a planted ``CFLAGS`` must not reach the build.
    """
    planted = tmp_path / "planted" / "setuptools"
    planted.mkdir(parents=True)
    (planted / "__init__.py").write_text("raise ImportError('planted setuptools reached the build')\n")
    monkeypatch.setenv("PYTHONPATH", str(planted.parent))
    monkeypatch.setenv("CFLAGS", "-DPLANTED_CFLAGS")
    url, sha = _psf_from_uv_lock()
    base = tmp_path / "manifests"
    base.mkdir()
    shutil.copyfile(DATA / "psf_build.lock", base / "psf_build.lock")
    assert all(e.hashes for e in lock_entries((base / "psf_build.lock").read_text()))
    spec = SdistSpec(
        name="python-speech-features",
        url=url,
        sha256=sha,
        build_env=LockRef(lock="psf_build.lock", lock_sha256=file_sha256(base / "psf_build.lock")),
    )
    lock = tmp_path / "wheels.lock"
    result = build.ensure_wheel(
        spec,
        base,
        tmp_path / "wheelhouse",
        lock,
        python="3.11.15",
        work_dir=tmp_path / "work",
        sdist_cache=tmp_path / "sdists",
    )
    wheel = result.wheel
    assert wheel.name == "python_speech_features-0.6-py3-none-any.whl"
    assert result.wheel_sha256 == _spike_wheel_sha() == file_sha256(wheel) == wheel.parent.name
    assert (
        build.read_wheels_lock(lock, result.wheels_lock_sha256)["python-speech-features"].wheel_sha256
        == wheel.parent.name
    )
    info = json.loads(wheel.with_name(wheel.name + ".build-info.json").read_text())
    assert info["uv"].startswith("uv ") and info["gcc"] != "unavailable"
    check = subprocess.run(
        [str(tmp_path / "work" / "venv" / "bin" / "python"), "-c", "import setuptools, wheel"],
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert check.returncode == 0


NET_SETUP_PY = """import socket
from setuptools import setup

try:
    socket.create_connection(("pypi.org", 443), timeout=5).close()
except OSError:
    pass
else:
    raise SystemExit("NETWORK REACHABLE FROM THE BUILD")
setup(name="demo", version="1.0", py_modules=["demo"])
"""


@pytest.mark.slow
@pytest.mark.network
def test_build_has_no_network(tmp_path: Path) -> None:
    """A setup.py that refuses to build when it can reach the network builds: the build runs in the namespace."""
    if not sandbox.userns_available():
        pytest.skip("no user namespaces here")
    import socket

    socket.create_connection(("pypi.org", 443), timeout=5).close()  # control: the host itself has network
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in (("setup.py", NET_SETUP_PY), ("demo.py", "VALUE = 1\n")):
            data = text.encode()
            info = tarfile.TarInfo(f"demo-1.0/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    sdist = tmp_path / "upstream" / "demo-1.0.tar.gz"
    sdist.parent.mkdir(parents=True)
    sdist.write_bytes(buffer.getvalue())
    base = tmp_path / "manifests"
    base.mkdir()
    shutil.copyfile(DATA / "psf_build.lock", base / "build.lock")
    spec = SdistSpec(
        name="demo",
        url=sdist.as_uri(),
        sha256=file_sha256(sdist),
        build_env=LockRef(lock="build.lock", lock_sha256=file_sha256(base / "build.lock")),
    )
    wheel, _ = build.build_wheel(spec, base, tmp_path / "wh", python="3.11.15", work_dir=tmp_path / "work")
    assert wheel.name == "demo-1.0-py3-none-any.whl"


def _hold_locks(directory: Path) -> None:
    """Take the wheels.lock sidecar lock shared, then exclusive, then read and record through it."""
    lock = directory / "env" / "wheels.lock"
    with build._locked(lock, shared=True):
        pass
    with build._locked(lock, shared=False):
        pass
    with build.file_lock(directory / "venv.lock", shared=True), build.file_lock(directory / "other.lock"):
        pass
    assert build.read_wheels_lock(lock) == {}
    build.write_wheels_lock(lock, [])
    assert build.read_wheels_lock(lock, file_sha256(lock)) == {}


def test_locks_take_shared_and_exclusive(tmp_path: Path) -> None:
    """Shared and exclusive locks on a sidecar created by the lock itself."""
    _hold_locks(tmp_path)


def test_locks_on_nfs(nfs_dir: Path) -> None:
    """On NFS ``flock`` is a POSIX lock: a shared lock needs a readable descriptor (EBADF otherwise)."""
    _hold_locks(nfs_dir)
