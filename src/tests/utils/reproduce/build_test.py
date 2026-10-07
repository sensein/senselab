"""Tests for sdist builds: fetching, the build venv and argv, patches, ``wheels.lock`` and the batch script."""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.reproduce import build
from senselab.utils.reproduce.locks import ReproduceLockError, file_sha256, lock_entries
from senselab.utils.reproduce.spec import FileRef, LockRef, Profile, SdistSpec, TorchSpec

DATA = Path(__file__).parent / "data"
UV = "/fake/uv"
DATE = datetime.date(2026, 10, 7)
CALLER_ENV = {"UV_CACHE_DIR": "/cache/uv", "UV_PYTHON_INSTALL_DIR": "/cache/py", "SSL_CERT_FILE": "/certs/ca.pem"}
SETUP_PY = "from setuptools import setup\n\nsetup(name='demo', version='1.0', py_modules=['demo'])\n"
PATCH = "--- a/demo.py\n+++ b/demo.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"


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
    monkeypatch.setattr(build, "find_uv", lambda: UV)
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
    assert argvs[2] == [
        *(UV, "build", "--no-config", "--wheel", "--no-build-isolation", "--python", py),
        *("--out-dir", str(work / "dist"), str(work / "src" / "demo-1.0")),
    ]
    info = json.loads(wheel.with_name(wheel.name + ".build-info.json").read_text())
    assert info["python_exact"] == "3.11.15"
    assert info["uv"] == "uv 1.0" and info["gcc"] == "gcc 1.0"
    assert info["modules"] == [] and "nvcc" not in info


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


def test_modules_wrap_the_build_without_dotfiles(tmp_path: Path) -> None:
    """With modules the build runs inside ``bash --noprofile --norc -c '. <init> && module load … && exec "$@"'``."""
    spec, _ = _setup(tmp_path, cuda_module="gcc/12.2.0 cuda/12.9.1")
    argv = build.wheel_build_argv(UV, spec, Path("/w/venv/bin/python"), Path("/w/src/demo-1.0"), Path("/w/dist"))
    inner = [UV, "build", "--no-config", "--wheel", "--no-build-isolation", "--python", "/w/venv/bin/python"]
    inner += ["--out-dir", "/w/dist", "/w/src/demo-1.0"]
    assert argv[:4] == ["bash", "--noprofile", "--norc", "-c"]
    assert argv[4].startswith(". /etc/profile.d/modules.sh && module load gcc/12.2.0 cuda/12.9.1 && ")
    assert "if ! command -v nvcc >/dev/null; then" in argv[4]
    assert "export CUDA_HOME=" in argv[4]
    assert argv[4].endswith('exec "$@"')
    assert argv[5:] == ["bash", *inner]
    plain, _ = _setup(tmp_path / "plain")
    assert (
        build.wheel_build_argv(UV, plain, Path("/w/venv/bin/python"), Path("/w/src/demo-1.0"), Path("/w/dist")) == inner
    )


def test_missing_nvcc_fails_the_wrapper(tmp_path: Path) -> None:
    """The wrapper's nvcc guard fails when no nvcc is on PATH (module init replaced by a no-op function)."""
    init = tmp_path / "init.sh"
    init.write_text("module() { :; }\n")
    spec, _ = _setup(tmp_path, cuda_module="cuda/1.0")
    argv = build.in_modules(spec, ["true"])
    argv[4] = argv[4].replace(build.MODULES_INIT, str(init))
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
        }
    )


def test_render_build_job(tmp_path: Path) -> None:
    """The batch script uses the build partition, loads the CUDA module, exports the arch list, logs under the root."""
    spec, _ = _setup(tmp_path, cuda_module="cuda/12.9.1", torch_cuda_arch_list="8.0;9.0")
    text = build.render_build_job(
        _profile(tmp_path),
        spec,
        manifest=Path("/repo/m.yaml"),
        profile_path=Path("/repo/p.yaml"),
        step="train",
        date=DATE,
        driver_python="/opt/py/bin/python",
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
    ):
        assert f"#SBATCH {flag}" in lines
    assert "#SBATCH --export=NONE" in lines
    assert lines[lines.index("set -euo pipefail") :] == [
        "set -euo pipefail",
        "export PATH=/usr/bin:/bin",
        "export UV_CACHE_DIR=/cache/uv",
        "export SENSELAB_REPRO_CACHE='/cache/repro dir'",
        "export REPRO_DRIVER_PYTHON=/opt/py/bin/python",
        'export TMPDIR="/dev/shm/reproduce-build-${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
        "trap 'rm -rf \"$TMPDIR\"' EXIT",
        ". /etc/profile.d/modules.sh",
        "module load cuda/12.9.1",
        'export TORCH_CUDA_ARCH_LIST="8.0;9.0"',
        '"$REPRO_DRIVER_PYTHON" -m senselab.utils.reproduce build-one /repo/m.yaml'
        " --profile /repo/p.yaml --step train --sdist demo",
    ]
    assert "LD_LIBRARY_PATH" not in text and "PLANTED" not in text


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
    )
    assert "module" not in text
    assert "TORCH_CUDA_ARCH_LIST" not in text
    assert text.splitlines()[-1].endswith("--sdist demo --rebuild")


def test_render_build_job_refuses_unsafe_values(tmp_path: Path) -> None:
    """Values that would break out of the script's quoting are refused."""
    spec, _ = _setup(tmp_path, cuda_module="cuda/12.9.1; rm -rf x", torch_cuda_arch_list="8.0")
    with pytest.raises(ValueError, match="cuda_module"):
        build.render_build_job(
            _profile(tmp_path), spec, manifest=Path("/m.yaml"), profile_path=Path("/p.yaml"), step="s", date=DATE
        )
    spec, _ = _setup(tmp_path / "b", torch_cuda_arch_list='8.0"; echo "')
    with pytest.raises(ValueError, match="torch_cuda_arch_list"):
        build.render_build_job(
            _profile(tmp_path), spec, manifest=Path("/m.yaml"), profile_path=Path("/p.yaml"), step="s", date=DATE
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
