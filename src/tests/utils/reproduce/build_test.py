"""Tests for sdist builds: fetching, the build venv and argv, patches, ``wheels.lock`` and the batch script."""

from __future__ import annotations

import datetime
import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.reproduce import build
from senselab.utils.reproduce.locks import ReproduceLockError, file_sha256, lock_entries
from senselab.utils.reproduce.spec import FileRef, LockRef, Profile, SdistSpec, TorchSpec

DATA = Path(__file__).parent / "data"
UV = "/fake/uv"
DATE = datetime.date(2026, 10, 7)
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
    """Stands in for ``build._run``: records each uv call; ``uv build`` writes a wheel into ``--out-dir``."""

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
            (out / "demo-1.0-py3-none-any.whl").write_bytes(b"wheel:" + (src / "demo.py").read_bytes())


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    """Replace the uv runner and pin the uv path."""
    fake = FakeRun()
    monkeypatch.setattr(build, "_run", fake)
    monkeypatch.setattr(build, "find_uv", lambda: UV)
    return fake


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
    """The build venv is made from the verified lock copy; the patch is applied; the build gets the fixed env."""
    spec, base = _setup(tmp_path, PATCH, build_env_vars={"MAX_JOBS": "4"}, torch_cuda_arch_list="8.0;9.0")
    work = tmp_path / "work"
    wheel, sha = build.build_wheel(spec, base, tmp_path / "wheelhouse", python="3.11.15", work_dir=work)
    assert wheel == tmp_path / "wheelhouse" / "demo-1.0-py3-none-any.whl"
    assert sha == hashlib.sha256(b"wheel:VALUE = 2\n").hexdigest() == file_sha256(wheel)
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
    env = fake_run.calls[2][1]
    assert env is not None
    assert env["SOURCE_DATE_EPOCH"] == "315532800"
    assert env["MAX_JOBS"] == "4"
    assert env["TORCH_CUDA_ARCH_LIST"] == "8.0;9.0"


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


def test_cuda_module_wraps_the_build_in_a_login_shell(tmp_path: Path) -> None:
    """With a CUDA module the build argv runs ``uv build`` inside ``bash -lc 'module load … && exec "$@"'``."""
    spec, _ = _setup(tmp_path, cuda_module="cuda/12.9.1")
    argv = build.wheel_build_argv(UV, spec, Path("/w/venv/bin/python"), Path("/w/src/demo-1.0"), Path("/w/dist"))
    inner = [UV, "build", "--no-config", "--wheel", "--no-build-isolation", "--python", "/w/venv/bin/python"]
    inner += ["--out-dir", "/w/dist", "/w/src/demo-1.0"]
    assert argv[:2] == ["bash", "-lc"]
    assert argv[2].startswith("module load cuda/12.9.1 && ")
    assert argv[2].endswith('exec "$@"')
    assert "CUDA_HOME" in argv[2]
    assert argv[3:] == ["bash", *inner]
    plain, _ = _setup(tmp_path / "plain")
    assert (
        build.wheel_build_argv(UV, plain, Path("/w/venv/bin/python"), Path("/w/src/demo-1.0"), Path("/w/dist")) == inner
    )


def test_unpacked_modes_ignore_the_callers_umask(tmp_path: Path) -> None:
    """A member archived 0o664 unpacks as 0o660 (its mode under ``BUILD_UMASK``) whatever the process umask."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("pkg-1.0/a.py")
        info.mode = 0o664
        tar.addfile(info, io.BytesIO(b""))
    archive = tmp_path / "pkg-1.0.tar.gz"
    archive.write_bytes(buffer.getvalue())
    old = os.umask(0o077)
    try:
        src = build._unpack(archive, tmp_path / "src")
    finally:
        os.umask(old)
    assert (src / "a.py").stat().st_mode & 0o777 == 0o660


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
        "wheel_file": "demo-1.0-py3-none-any.whl",
        "wheel_sha256": "4" * 64,
    }
    values.update(fields)
    return build.WheelLine(**values)


def test_wheels_lock_round_trip(tmp_path: Path) -> None:
    """Lines written are lines read back, sorted by name, under the column header."""
    path = tmp_path / "wheels.lock"
    lines = [_line(name="zeta"), _line(patch_sha256s=("5" * 64, "6" * 64))]
    build.write_wheels_lock(path, lines)
    text = path.read_text().splitlines()
    assert text[0] == build.WHEELS_HEADER
    assert text[1].split()[0] == "demo" and text[2].split()[0] == "zeta"
    assert text[1].split()[5] == "5" * 64 + "," + "6" * 64
    assert text[2].split()[5] == "-"
    assert build.read_wheels_lock(path) == {"demo": lines[1], "zeta": lines[0]}


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
    wheel, sha = build.ensure_wheel(spec, base, wheelhouse, lock, python="3.11.15", work_dir=tmp_path / "w1")
    recorded = build.read_wheels_lock(lock)["demo"]
    assert recorded == build.WheelLine(
        name="demo",
        version="1.0",
        sdist_url=spec.url,
        sdist_sha256=spec.sha256,
        build_env_sha256=spec.build_env.lock_sha256,
        patch_sha256s=(spec.patches[0].sha256,),
        build_config_sha256=build.build_config_sha256(spec, None),
        wheel_file=wheel.name,
        wheel_sha256=sha,
    )

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("rebuilt a recorded wheel")

    monkeypatch.setattr(build, "build_wheel", refuse)
    assert build.ensure_wheel(spec, base, wheelhouse, lock, python="3.11.15") == (wheel, sha)


def test_reuse_refuses_a_changed_wheel(tmp_path: Path) -> None:
    """A wheelhouse wheel whose sha256 differs from its line raises instead of being installed or rebuilt."""
    spec, _ = _setup(tmp_path)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "demo-1.0-py3-none-any.whl").write_bytes(b"tampered")
    line = _line(
        sdist_url=spec.url,
        sdist_sha256=spec.sha256,
        build_env_sha256=spec.build_env.lock_sha256,
        build_config_sha256=build.build_config_sha256(spec, None),
    )
    build.write_wheels_lock(tmp_path / "wheels.lock", [line])
    with pytest.raises(ReproduceLockError, match="sha256"):
        build.reuse_wheel(tmp_path / "wheels.lock", spec, wheelhouse)
    (wheelhouse / "demo-1.0-py3-none-any.whl").unlink()
    with pytest.raises(ReproduceLockError, match="missing"):
        build.reuse_wheel(tmp_path / "wheels.lock", spec, wheelhouse)


def test_reuse_refuses_a_stale_line(tmp_path: Path) -> None:
    """A line recorded for other build inputs (here a CUDA module added since) is refused, naming ``--rebuild``."""
    spec, _ = _setup(tmp_path)
    line = _line(sdist_url=spec.url, sdist_sha256=spec.sha256, build_env_sha256=spec.build_env.lock_sha256)
    build.write_wheels_lock(tmp_path / "wheels.lock", [line])
    with pytest.raises(ReproduceLockError, match="--rebuild"):
        build.reuse_wheel(tmp_path / "wheels.lock", spec, tmp_path)
    assert build.reuse_wheel(tmp_path / "absent.lock", spec, tmp_path) is None


def test_rebuild_replaces_the_line(tmp_path: Path, fake_run: FakeRun) -> None:
    """``rebuild`` builds even with a recorded line and replaces only that line."""
    spec, base = _setup(tmp_path)
    lock = tmp_path / "wheels.lock"
    other = _line(name="other")
    build.write_wheels_lock(lock, [other, _line(wheel_sha256="9" * 64)])
    with pytest.raises(ReproduceLockError, match="--rebuild"):
        build.ensure_wheel(spec, base, tmp_path / "wh", lock, python="3.11.15", work_dir=tmp_path / "w0")
    _, sha = build.ensure_wheel(
        spec, base, tmp_path / "wh", lock, python="3.11.15", rebuild=True, work_dir=tmp_path / "w1"
    )
    lines = build.read_wheels_lock(lock)
    assert lines["other"] == other
    assert lines["demo"].wheel_sha256 == sha
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
    assert "module load cuda/12.9.1" in lines
    assert 'export TORCH_CUDA_ARCH_LIST="8.0;9.0"' in lines
    assert "LD_LIBRARY_PATH" not in text
    assert lines[-1] == (
        "exec /opt/py/bin/python -m senselab.utils.reproduce build-one /repo/m.yaml"
        " --profile /repo/p.yaml --step train --sdist demo"
    )
    assert lines.index("module load cuda/12.9.1") < len(lines) - 1


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
    assert "module load" not in text
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
def test_build_python_speech_features_matches_spike(tmp_path: Path) -> None:
    """Building python_speech_features 0.6 from its real sdist reproduces the spike's wheel bit for bit."""
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
    wheel, wheel_sha = build.ensure_wheel(
        spec,
        base,
        tmp_path / "wheelhouse",
        lock,
        python="3.11.15",
        work_dir=tmp_path / "work",
        sdist_cache=tmp_path / "sdists",
    )
    assert wheel.name == "python_speech_features-0.6-py3-none-any.whl"
    assert wheel_sha == _spike_wheel_sha() == file_sha256(wheel)
    assert build.read_wheels_lock(lock)["python-speech-features"].wheel_sha256 == wheel_sha
    result = subprocess.run(
        [str(tmp_path / "work" / "venv" / "bin" / "python"), "-c", "import setuptools, wheel"], check=False
    )
    assert result.returncode == 0
