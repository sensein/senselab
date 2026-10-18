"""Tests for lock generation: the uv argv each builder runs, the files it writes, and the B3 round trip."""

from __future__ import annotations

import datetime
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from senselab.utils.reproduce import lockgen
from senselab.utils.reproduce.locks import (
    ReproduceLockError,
    check_integrity,
    file_sha256,
    lock_entries,
    normalize_name,
    parse_header,
    requirements_digest_for,
)
from senselab.utils.reproduce.spec import EnvSpec

DATA = Path(__file__).parent / "data"
ENV_DIR = "envs/a"
UV = "/fake/uv"
PLACEHOLDER = "00" * 32
EXCLUDE_NEWER = "2026-07-01T23:59:59Z"
FAMILY = ["nvidia-nvtx-cu12", "torch", "triton"]
BODY = ["annotated-doc", "certifi", "idna", "numpy", "packaging", "six"]
HELD_OUT = ["python-speech-features", "opencv-python"]
# numpy 2.4.6 with one of its body.lock hashes: a base name re-pinned with a different hash set.
NUMPY_SUBSET = "numpy==2.4.6 \\\n    --hash=sha256:0280e0356c0829a18d9de1cb7eee50ec22ca639878d7240307ca0943d73cd2c4\n"


def _no_emit(names: list[str]) -> list[str]:
    return [arg for name in names for arg in ("--no-emit-package", name)]


def _fixture(argv: list[str]) -> str:
    """The text uv would print for this builder: chosen by subcommand and input file."""
    if argv[1] == "export":
        return (DATA / ("test_small.lock" if "--only-group" in argv else "body_small.lock")).read_text()
    assert argv[1:3] == ["pip", "compile"]
    source = PurePosixPath(argv[3]).name
    if source == "torch.in":
        return (DATA / "torch_index_small.lock").read_text()
    if source == "requirements.in":
        return (DATA / ("body_small.lock" if "--generate-hashes" in argv else "first_resolve_small.txt")).read_text()
    assert source == "notebook_tools.in"
    return (DATA / "test_small.lock").read_text() + NUMPY_SUBSET


def _drop_no_emit(text: str, argv: list[str]) -> str:
    """Remove the requirement blocks uv's ``--no-emit-package`` would leave out."""
    dropped = {normalize_name(argv[i + 1]) for i, arg in enumerate(argv) if arg == "--no-emit-package"}
    out: list[str] = []
    keep = True
    for line in text.splitlines(keepends=True):
        if line[:1].isalnum():
            keep = normalize_name(re.split(r"[=@ ;\\]", line, maxsplit=1)[0]) not in dropped
        if keep:
            out.append(line)
    return "".join(out)


class FakeUv:
    """Stands in for ``subprocess.run``: records each call and prints a fixture chosen by its argv."""

    def __init__(self, returncode: int = 0) -> None:
        """Start with no calls."""
        self.calls: list[tuple[list[str], Path | None]] = []
        self.returncode = returncode

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        """Record ``argv`` and ``cwd``; return the fixture text as stdout."""
        self.calls.append((list(argv), kwargs.get("cwd")))
        stdout = _drop_no_emit(_fixture(argv), argv) if self.returncode == 0 else ""
        return subprocess.CompletedProcess(argv, self.returncode, stdout, "uv: resolution failed")


@pytest.fixture
def fake_uv(monkeypatch: pytest.MonkeyPatch) -> FakeUv:
    """Route every uv call through :class:`FakeUv` and pretend uv is on ``PATH``."""
    fake = FakeUv()
    monkeypatch.setattr(shutil, "which", lambda name: UV if name == "uv" else None)
    monkeypatch.setattr(lockgen.subprocess, "run", fake)
    return fake


def _copy(base: Path, fixture: str, name: str) -> str:
    rel = f"{ENV_DIR}/{name}"
    (base / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / fixture, base / rel)
    return rel


def _lock(rel: str) -> dict[str, str]:
    return {"lock": rel, "lock_sha256": PLACEHOLDER}


def _draft(base: Path, source: str = "requirements.in") -> dict[str, Any]:
    """A draft manifest ``env``: inputs in place and hashed, lock sha256s still placeholders."""
    (base / ENV_DIR).mkdir(parents=True, exist_ok=True)
    env: dict[str, Any] = {
        "python": "3.11",
        "python_exact": "3.11.9",
        "torch": {"index_url": "https://download.pytorch.org/whl/cu128", **_lock(f"{ENV_DIR}/torch.lock")},
        "body": {**_lock(f"{ENV_DIR}/body.lock"), "source": source},
        "sdist_allow": [
            {
                "name": "python_speech_features",
                "url": "https://files.example/python_speech_features-0.6.tar.gz",
                "sha256": "a0aebf746464bc929dc3162cb369d7ff967c398c5120ddf5fb40a65f01b92b11",
                "build_env": {"lock": _copy(base, "body_small.lock", "build.lock"), "lock_sha256": PLACEHOLDER},
            }
        ],
        "exclude_newer": "2026-07-01",
        "extra_no_emit": ["opencv-python"],
        "test": _lock(f"{ENV_DIR}/test.lock"),
        "notebook_tools": _lock(f"{ENV_DIR}/notebook_tools.lock"),
        "max_cuda": "12.8",
    }
    if source == "requirements.in":
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
        env["first_resolve"] = {"file": f"{ENV_DIR}/first_resolve.txt", "sha256": PLACEHOLDER}
    else:
        env["body"]["export_cmd"] = ["uv", "export"]
    return env


def _finalize(base: Path, raw: dict[str, Any]) -> EnvSpec:
    """Fill in every lock and resolve sha256 from the written files, as ``reproduce lock`` does."""
    refs = [raw[k] for k in ("torch", "body", "test", "notebook_tools") if raw.get(k)]
    for ref in [*refs, *(s["build_env"] for s in raw.get("sdist_allow", []))]:
        ref["lock_sha256"] = file_sha256(base / ref["lock"])
    if raw.get("first_resolve"):
        raw["first_resolve"]["sha256"] = file_sha256(base / raw["first_resolve"]["file"])
    return EnvSpec.model_validate(raw)


def _project(base: Path) -> Path:
    project = base / "repo"
    project.mkdir()
    shutil.copyfile(DATA / "uv_closure_small.lock", project / "uv.lock")
    return project


def _body(text: str) -> str:
    """The lock text after the reproduce header."""
    lines = text.splitlines(keepends=True)
    return "".join(lines[next(i for i, line in enumerate(lines) if not line.startswith("#")) :])


# Argv per builder


def test_first_resolve_argv(tmp_path: Path, fake_uv: FakeUv) -> None:
    """The first resolve is an unhashed universal compile of the body inputs."""
    env = EnvSpec.model_validate(_draft(tmp_path))
    argv = lockgen.first_resolve(env, tmp_path)
    assert argv == [
        UV, "pip", "compile", f"{ENV_DIR}/requirements.in",
        "--universal", "--no-annotate", "--no-header",
        "--python-version", "3.11", "--exclude-newer", EXCLUDE_NEWER, "--index-url", "https://pypi.org/simple",
        "--config-file", f"{ENV_DIR}/uv.toml", "--overrides", f"{ENV_DIR}/overrides.txt",
    ]  # fmt: skip
    assert fake_uv.calls == [(argv, tmp_path)]
    written = (tmp_path / ENV_DIR / "first_resolve.txt").read_text()
    assert parse_header(written)["python"] == "3.11"
    assert _body(written) == (DATA / "first_resolve_small.txt").read_text()


def test_torch_lock_from_index_argv(tmp_path: Path, fake_uv: FakeUv) -> None:
    """The torch pins come from the first resolve; only ``nvidia-*`` names are exempt from ``--exclude-newer``."""
    raw = _draft(tmp_path)
    _copy(tmp_path, "first_resolve_small.txt", "first_resolve.txt")
    argv = lockgen.torch_lock_from_index(EnvSpec.model_validate(raw), tmp_path)
    assert argv == [
        UV, "pip", "compile", f"{ENV_DIR}/torch.in",
        "--no-deps", "--generate-hashes", "--emit-index-url", "--no-header",
        "--python-version", "3.11", "--exclude-newer", EXCLUDE_NEWER,
        "--exclude-newer-package", "nvidia-nvtx-cu12=false",
        "--python-platform", "x86_64-manylinux_2_28", "--index-url", "https://download.pytorch.org/whl/cu128",
    ]  # fmt: skip
    assert fake_uv.calls == [(argv, tmp_path)]
    pins = "nvidia-nvtx-cu12==12.8.90\ntorch==2.8.0+cu128\ntriton==3.4.0\n"
    assert (tmp_path / ENV_DIR / "torch.in").read_text() == pins
    assert _body((tmp_path / ENV_DIR / "torch.lock").read_text()) == (DATA / "torch_index_small.lock").read_text()


def test_body_lock_from_requirements_argv_and_header(tmp_path: Path, fake_uv: FakeUv) -> None:
    """The body compile leaves out the torch family, the sdists and extra_no_emit, and records the digest."""
    raw = _draft(tmp_path)
    _copy(tmp_path, "first_resolve_small.txt", "first_resolve.txt")
    env = EnvSpec.model_validate(raw)
    argv = lockgen.body_lock_from_requirements(env, tmp_path)
    assert argv == [
        UV, "pip", "compile", f"{ENV_DIR}/requirements.in",
        "--universal", "--generate-hashes", "--no-header",
        "--python-version", "3.11", "--exclude-newer", EXCLUDE_NEWER, "--index-url", "https://pypi.org/simple",
        "--config-file", f"{ENV_DIR}/uv.toml", "--overrides", f"{ENV_DIR}/overrides.txt",
        *_no_emit(FAMILY + HELD_OUT),
    ]  # fmt: skip
    assert fake_uv.calls == [(argv, tmp_path)]
    header = parse_header((tmp_path / ENV_DIR / "body.lock").read_text())
    assert header["requirements-sha256"] == requirements_digest_for(env, tmp_path)
    assert header["command"].startswith("uv pip compile envs/a/requirements.in --universal")


def test_body_lock_from_uvlock_export_argv(tmp_path: Path, fake_uv: FakeUv) -> None:
    """The export leaves out the ``uv.lock`` closure of torch.lock's roots, the sdists and extra_no_emit."""
    project = _project(tmp_path)
    _copy(tmp_path, "torch_small.lock", "torch.lock")
    env = EnvSpec.model_validate(_draft(tmp_path, "uv-export"))
    argv = lockgen.body_lock_from_uvlock_export(env, tmp_path, project, extras=["qa"], groups=["eval"])
    assert argv == [
        UV, "export", "--frozen", "--no-emit-project", "--no-dev", "--no-header",
        "--extra", "qa", "--group", "eval", *_no_emit(FAMILY + HELD_OUT),
    ]  # fmt: skip
    assert fake_uv.calls == [(argv, project)]
    assert _body((tmp_path / ENV_DIR / "body.lock").read_text()) == (DATA / "body_small.lock").read_text()


def test_body_export_needs_a_torch_root(tmp_path: Path, fake_uv: FakeUv) -> None:
    """With a torch lock, the roots come from it; a torch lock without torch, torchvision or torchaudio is refused."""
    project = _project(tmp_path)
    _copy(tmp_path, "torch_small.lock", "torch.lock")
    lock = tmp_path / ENV_DIR / "torch.lock"
    lock.write_text("".join(b for b in re.split(r"(?m)^(?=\S)", lock.read_text()) if not b.startswith("torch ")))
    env = EnvSpec.model_validate(_draft(tmp_path, "uv-export"))
    with pytest.raises(ReproduceLockError, match="holds no torch, torchvision or torchaudio"):
        lockgen.body_lock_from_uvlock_export(env, tmp_path, project)
    assert fake_uv.calls == []


def test_overlay_test_lock_argv(tmp_path: Path, fake_uv: FakeUv) -> None:
    """The test overlay is the repository's dev group, without the names the base locks hold or hold out."""
    project = _project(tmp_path)
    _copy(tmp_path, "torch_small.lock", "torch.lock")
    _copy(tmp_path, "body_small.lock", "body.lock")
    env = EnvSpec.model_validate(_draft(tmp_path, "uv-export"))
    argv = lockgen.overlay_lock("test", env, tmp_path, project_dir=project)
    assert argv == [
        UV, "export", "--frozen", "--no-emit-project", "--only-group", "dev", "--no-header",
        *_no_emit(FAMILY + BODY + HELD_OUT),
    ]  # fmt: skip
    assert fake_uv.calls == [(argv, project)]
    names = {e.name for e in lock_entries((tmp_path / ENV_DIR / "test.lock").read_text())}
    assert names and not names & set(BODY)


def test_overlay_notebook_tools_argv(tmp_path: Path, fake_uv: FakeUv) -> None:
    """Notebook tools compile with the base locks as constraints and leave out every base name."""
    _copy(tmp_path, "torch_small.lock", "torch.lock")
    _copy(tmp_path, "body_small.lock", "body.lock")
    (tmp_path / ENV_DIR / "notebook_tools.in").write_text("nbconvert\nipykernel\n")
    env = EnvSpec.model_validate(_draft(tmp_path, "uv-export"))
    argv = lockgen.overlay_lock("notebook_tools", env, tmp_path, requirements=f"{ENV_DIR}/notebook_tools.in")
    assert argv == [
        UV, "pip", "compile", f"{ENV_DIR}/notebook_tools.in",
        "--universal", "--generate-hashes", "--no-header",
        "--python-version", "3.11", "--exclude-newer", EXCLUDE_NEWER, "--index-url", "https://pypi.org/simple",
        "--constraints", f"{ENV_DIR}/torch.lock", "--constraints", f"{ENV_DIR}/body.lock",
        *_no_emit(FAMILY + BODY + HELD_OUT),
    ]  # fmt: skip
    assert fake_uv.calls == [(argv, tmp_path)]
    names = {e.name for e in lock_entries((tmp_path / ENV_DIR / "notebook_tools.lock").read_text())}
    assert names and not names & set(BODY)


def test_overlay_needs_its_input(tmp_path: Path, fake_uv: FakeUv) -> None:
    """A test overlay needs the project; a notebook overlay needs its requirements file."""
    env = EnvSpec.model_validate(_draft(tmp_path, "uv-export"))
    with pytest.raises(ReproduceLockError, match="project_dir"):
        lockgen.overlay_lock("test", env, tmp_path)
    with pytest.raises(ReproduceLockError, match="requirements"):
        lockgen.overlay_lock("notebook_tools", env, tmp_path)
    assert fake_uv.calls == []


def test_missing_uv_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``uv`` on ``PATH`` the builders say so instead of failing inside subprocess."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ReproduceLockError, match="uv not found on PATH"):
        lockgen.first_resolve(EnvSpec.model_validate(_draft(tmp_path)), tmp_path)


def test_uv_failure_is_raised_with_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing uv call raises with uv's stderr and writes no lock."""
    monkeypatch.setattr(shutil, "which", lambda name: UV)
    monkeypatch.setattr(lockgen.subprocess, "run", FakeUv(returncode=2))
    with pytest.raises(ReproduceLockError, match="resolution failed"):
        lockgen.first_resolve(EnvSpec.model_validate(_draft(tmp_path)), tmp_path)
    assert not (tmp_path / ENV_DIR / "first_resolve.txt").exists()


# torch.lock from uv.lock


def test_torch_lock_from_uvlock_selects_the_linux_wheel_of_the_closure(tmp_path: Path) -> None:
    """The torch closure of ``uv.lock`` becomes URL-pinned, hashed lines, one linux x86_64 wheel each."""
    project = _project(tmp_path)
    env = EnvSpec.model_validate(_draft(tmp_path, "uv-export"))
    assert lockgen.torch_lock_from_uvlock(env, tmp_path, project / "uv.lock", ["torch"]) == []
    text = (tmp_path / ENV_DIR / "torch.lock").read_text()
    header = parse_header(text)
    assert (header["source"], header["roots"]) == ("uv.lock", "torch")
    assert header["uv-lock-sha256"] == file_sha256(project / "uv.lock")
    assert _body(text) == (DATA / "torch_small.lock").read_text()
    assert all(e.is_url and len(e.hashes) == 1 for e in lock_entries(text))


def test_torch_lock_from_uvlock_refuses_a_missing_wheel(tmp_path: Path) -> None:
    """No wheel for the env's Python is an error, not a silent skip."""
    project = _project(tmp_path)
    raw = _draft(tmp_path, "uv-export")
    raw["python"], raw["python_exact"] = "3.12", "3.12.11"
    with pytest.raises(ReproduceLockError, match="torch: 0 linux x86_64 wheels for cp312"):
        lockgen.torch_lock_from_uvlock(EnvSpec.model_validate(raw), tmp_path, project / "uv.lock", ["torch"])


# Round trips through check_integrity


def test_requirements_env_round_trip_passes_check_integrity(tmp_path: Path, fake_uv: FakeUv) -> None:
    """first_resolve, torch, body and notebook locks as written pass B3, header recompute included.

    The fake notebook compile re-pins numpy with a subset of the body's hashes, as a universal compile
    does; only the overlay's ``--no-emit-package`` of every base name keeps it out.
    """
    raw = _draft(tmp_path)
    raw["test"] = None
    (tmp_path / ENV_DIR / "notebook_tools.in").write_text("nbconvert\n")
    env = EnvSpec.model_validate(raw)
    lockgen.first_resolve(env, tmp_path)
    lockgen.torch_lock_from_index(env, tmp_path)
    lockgen.body_lock_from_requirements(env, tmp_path)
    lockgen.overlay_lock("notebook_tools", env, tmp_path, requirements=f"{ENV_DIR}/notebook_tools.in")
    verified = check_integrity(_finalize(tmp_path, raw), tmp_path)
    assert verified[raw["body"]["lock"]] == file_sha256(tmp_path / raw["body"]["lock"])


def test_requirements_env_with_a_stale_header_fails_check_integrity(tmp_path: Path, fake_uv: FakeUv) -> None:
    """Changing a compile input after the body was written is caught by the header recompute."""
    raw = _draft(tmp_path)
    raw["test"] = raw["notebook_tools"] = None
    env = EnvSpec.model_validate(raw)
    lockgen.first_resolve(env, tmp_path)
    lockgen.torch_lock_from_index(env, tmp_path)
    lockgen.body_lock_from_requirements(env, tmp_path)
    raw["extra_no_emit"] = ["opencv-python", "decord"]
    with pytest.raises(ReproduceLockError, match="header requirements-sha256"):
        check_integrity(_finalize(tmp_path, raw), tmp_path)


def test_uv_export_env_round_trip_passes_check_integrity(tmp_path: Path, fake_uv: FakeUv) -> None:
    """torch.lock from ``uv.lock``, the exported body and the test overlay pass B3 against the same ``uv.lock``."""
    project = _project(tmp_path)
    raw = _draft(tmp_path, "uv-export")
    raw["notebook_tools"] = None
    env = EnvSpec.model_validate(raw)
    lockgen.torch_lock_from_uvlock(env, tmp_path, project / "uv.lock", ["torch"])
    lockgen.body_lock_from_uvlock_export(env, tmp_path, project)
    lockgen.overlay_lock("test", env, tmp_path, project_dir=project)
    check_integrity(_finalize(tmp_path, raw), tmp_path, uv_lock=project / "uv.lock")


# Real uv


@pytest.mark.slow
@pytest.mark.network
def test_real_body_compile_from_pypi(tmp_path: Path) -> None:
    """A real compile of two pins from PyPI: every line hashed, header parses, B3 passes."""
    (tmp_path / ENV_DIR).mkdir(parents=True)
    req = tmp_path / ENV_DIR / "requirements.in"
    req.write_text("six==1.17.0\npackaging==25.0\n")
    raw: dict[str, Any] = {
        "python": "3.11",
        "python_exact": "3.11.9",
        "torch": None,
        "body": {
            **_lock(f"{ENV_DIR}/body.lock"),
            "source": "requirements.in",
            "requirements_in": f"{ENV_DIR}/requirements.in",
            "requirements_in_sha256": file_sha256(req),
        },
        "exclude_newer": datetime.date(2026, 10, 1).isoformat(),
    }
    env = EnvSpec.model_validate(raw)
    argv = lockgen.body_lock_from_requirements(env, tmp_path)
    assert "2026-10-01T23:59:59Z" in argv
    text = (tmp_path / ENV_DIR / "body.lock").read_text()
    entries = lock_entries(text)
    assert {(e.name, e.pin) for e in entries} == {("six", "1.17.0"), ("packaging", "25.0")}
    assert all(e.hashes for e in entries)
    assert parse_header(text)["requirements-sha256"] == requirements_digest_for(env, tmp_path)
    check_integrity(_finalize(tmp_path, raw), tmp_path)


def _notebook_overlay_passes(tmp_path: Path, raw: dict[str, Any], uv_lock: Path | None = None) -> None:
    (tmp_path / ENV_DIR / "notebook_tools.in").write_text("nbconvert\nipykernel\n")
    env = EnvSpec.model_validate(raw)
    lockgen.overlay_lock("notebook_tools", env, tmp_path, requirements=f"{ENV_DIR}/notebook_tools.in")
    entries = lock_entries((tmp_path / ENV_DIR / "notebook_tools.lock").read_text())
    assert {"nbconvert", "ipykernel"} <= {e.name for e in entries}
    assert all(e.hashes for e in entries)
    check_integrity(_finalize(tmp_path, raw), tmp_path, uv_lock=uv_lock)


@pytest.mark.slow
@pytest.mark.network
def test_real_notebook_overlay_on_a_uv_export_body(tmp_path: Path) -> None:
    """Notebook tools (nbconvert, ipykernel) compiled against a uv-export body and a uv.lock torch lock pass B3."""
    project = _project(tmp_path)
    raw = _draft(tmp_path, "uv-export")
    raw["test"], raw["exclude_newer"] = None, "2026-08-20"
    lockgen.torch_lock_from_uvlock(EnvSpec.model_validate(raw), tmp_path, project / "uv.lock", ["torch"])
    _copy(tmp_path, "body_small.lock", "body.lock")
    _notebook_overlay_passes(tmp_path, raw, project / "uv.lock")


@pytest.mark.slow
@pytest.mark.network
def test_real_notebook_overlay_on_a_requirements_body(tmp_path: Path) -> None:
    """Notebook tools (nbconvert, ipykernel) compiled against a real requirements.in body pass B3."""
    (tmp_path / ENV_DIR).mkdir(parents=True)
    req = tmp_path / ENV_DIR / "requirements.in"
    req.write_text("six==1.17.0\npackaging==25.0\n")
    raw: dict[str, Any] = {
        "python": "3.11",
        "python_exact": "3.11.9",
        "torch": None,
        "body": {
            **_lock(f"{ENV_DIR}/body.lock"),
            "source": "requirements.in",
            "requirements_in": f"{ENV_DIR}/requirements.in",
            "requirements_in_sha256": file_sha256(req),
        },
        "notebook_tools": _lock(f"{ENV_DIR}/notebook_tools.lock"),
        "exclude_newer": "2026-10-01",
    }
    lockgen.body_lock_from_requirements(EnvSpec.model_validate(raw), tmp_path)
    _notebook_overlay_passes(tmp_path, raw)
