"""Tests for ``senselab.utils.venv_lock``: the committed locks and how they are compiled."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from senselab.utils import venv_lock
from senselab.utils.venv_lock import (
    LOCK_DIR,
    VenvLockError,
    compile_lock,
    discover_backends,
    load_lock,
    parse_lock,
    requirements_sha256,
    resolved_packages,
    torch_only_cuda_packages,
)

_LOCKS = sorted(LOCK_DIR.glob("*.txt"))
_PIN = re.compile(r"^([A-Za-z0-9._-]+)(\[[^\]]*\])?\s*==\s*([^\s;\\]+)")
_ENTRY = re.compile(r"^([A-Za-z0-9._-]+)(\[[^\]]*\])?\s*(==|@)")


@pytest.fixture(scope="module")
def backends() -> dict[str, venv_lock.Backend]:
    """Every venv the code builds, read from its ``ensure_venv`` calls."""
    return discover_backends()


def _body(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line and not line.startswith(("#", " "))]


def test_every_venv_the_code_builds_has_a_current_lock(backends: dict[str, venv_lock.Backend]) -> None:
    """A changed requirement list without a regenerated lock fails here, not on a cluster node."""
    assert backends, "no ensure_venv calls were found"
    for name, backend in backends.items():
        load_lock(name, backend.requirements, backend.python, backend.max_cuda_version)


_DIGESTS_ON_HOST = """
import json, platform, sys, types
from senselab.utils.venv_lock import discover_backends, requirements_sha256
modules = sorted({m for b in discover_backends().values() for m in b.modules})
sys.platform, machine = sys.argv[1], sys.argv[2]
platform.machine = lambda: machine
platform.system = lambda: {"linux": "Linux", "darwin": "Darwin"}[sys.platform]
for name in modules:
    real = sys.modules[name]
    fresh = types.ModuleType(name)
    fresh.__dict__.update(__file__=real.__file__, __package__=real.__package__, __spec__=real.__spec__)
    sys.modules[name] = fresh
    exec(compile(open(real.__file__).read(), real.__file__, "exec"), fresh.__dict__)
print(json.dumps({n: requirements_sha256(b.requirements, b.python, b.max_cuda_version)
                  for n, b in discover_backends().items()}))
"""


@pytest.mark.parametrize("host", [("linux", "x86_64"), ("darwin", "arm64")])
def test_every_lock_digest_is_the_same_on_every_host(
    host: tuple[str, str], backends: dict[str, venv_lock.Backend]
) -> None:
    """A requirement list chosen by platform at import time breaks the lock on the other platform."""
    result = subprocess.run(
        [sys.executable, "-c", _DIGESTS_ON_HOST, *host], capture_output=True, text=True, check=True, timeout=600
    )
    there = json.loads(result.stdout.strip().splitlines()[-1])
    here = {n: requirements_sha256(b.requirements, b.python, b.max_cuda_version) for n, b in backends.items()}
    assert {n for n in here if there.get(n) != here[n]} == set()


def test_every_declared_requirement_is_in_its_lock_body(backends: dict[str, venv_lock.Backend]) -> None:
    """Stage 2 installs the body with --no-deps, so a requirement missing from it is never installed."""
    from packaging.requirements import Requirement

    missing: dict[str, list[str]] = {}
    for name, backend in backends.items():
        lines = _body(venv_lock.lock_path(name))
        body = {venv_lock.normalize_name(m.group(1)) for line in lines if (m := _ENTRY.match(line))}
        for spec in backend.requirements:
            wanted = venv_lock.normalize_name(Requirement(spec).name)
            if wanted not in venv_lock.TORCH_PACKAGES and wanted not in body:
                missing.setdefault(name, []).append(wanted)
    assert missing == {}


def test_no_lock_is_left_over_from_a_removed_venv(backends: dict[str, venv_lock.Backend]) -> None:
    """Every committed lock belongs to a venv some code still builds."""
    assert {path.stem for path in _LOCKS} == set(backends)


@pytest.mark.parametrize("path", _LOCKS, ids=lambda p: p.stem)
def test_no_lock_resolves_a_quarantined_release(path: Path) -> None:
    """Releases 2.6.2 and 2.6.3 of lightning / pytorch-lightning were withdrawn from PyPI as malware."""
    for line in _body(path):
        match = _PIN.match(line)
        if match and venv_lock.normalize_name(match.group(1)) in {"lightning", "pytorch-lightning"}:
            assert match.group(3) not in {"2.6.2", "2.6.3"}, line


@pytest.mark.parametrize("path", _LOCKS, ids=lambda p: p.stem)
def test_no_lock_falls_back_to_a_tokenizers_that_needs_a_rust_build(path: Path) -> None:
    """A tokenizers below 0.13 has no wheel for current Pythons; it marks a resolution gone back years."""
    for line in _body(path):
        match = _PIN.match(line)
        if match and venv_lock.normalize_name(match.group(1)) == "tokenizers":
            major, minor = (int(part) for part in match.group(3).split(".")[:2])
            assert (major, minor) >= (0, 13), line


@pytest.mark.parametrize("path", _LOCKS, ids=lambda p: p.stem)
def test_every_body_entry_is_pinned_and_hashed(path: Path) -> None:
    """Each body requirement is an exact pin with hashes, a hashed archive URL, or a VCS URL at a commit."""
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line or line.startswith(("#", " ")):
            continue
        hashed = i + 1 < len(lines) and "--hash=sha256:" in lines[i + 1]
        if " @ git+" in line:
            assert re.search(r"@[0-9a-f]{40}", line), f"{path.stem}: VCS requirement not fixed to a commit: {line}"
        elif " @ " in line:
            assert hashed, f"{path.stem}: archive URL without a hash: {line}"
        else:
            assert _PIN.match(line) and hashed, f"{path.stem}: not a hashed exact pin: {line}"


@pytest.mark.parametrize("path", _LOCKS, ids=lambda p: p.stem)
def test_torch_comes_from_the_header_never_the_body(path: Path) -> None:
    """The torch packages and their CUDA-variant dependencies are installed from the PyTorch index."""
    lock = parse_lock(path)
    names = {venv_lock.normalize_name(m.group(1)) for line in _body(path) if (m := _PIN.match(line))}
    assert not names & {"torch", "torchaudio"}
    for pin in lock.torch_pins:
        assert re.fullmatch(r"torch(audio)?==[0-9][0-9.]*", pin), pin
    if lock.torch_pins:
        assert not any(name.startswith("triton") for name in names)


def test_requirements_digest_ignores_order_and_tracks_every_input() -> None:
    """Reordering requirements keeps the digest; changing a spec, the Python or the CUDA cap does not."""
    base = requirements_sha256(["a==1", "b>=2"], "3.11")
    assert base == requirements_sha256(["b>=2", "a==1"], "3.11")
    assert base != requirements_sha256(["a==1", "b>=3"], "3.11")
    assert base != requirements_sha256(["a==1", "b>=2"], "3.12")
    assert base != requirements_sha256(["a==1", "b>=2"], "3.11", (12, 4))


_ANNOTATED = """\
sympy==1.14.0             # via pyannote-metrics, torch
torch==2.8.0              # via lightning, torchaudio, -r requirements.in
torchaudio==2.8.0         # via -r requirements.in
nvidia-cublas-cu12==12.8.4.1 ; sys_platform == 'linux'  # via nvidia-cudnn-cu12, torch
nvidia-cudnn-cu12==9.10.2.21 ; sys_platform == 'linux'  # via torch
nvidia-ml-py==13.0.0      # via nemo-toolkit
triton==3.4.0 ; sys_platform == 'linux'  # via torch
nemo-toolkit @ git+https://github.com/NVIDIA/NeMo.git@1688cc3d6a9ade854f544987810c53f605dc86fc  # via -r requirements.in
"""


def test_only_the_cuda_variant_packages_torch_alone_pulls_in_are_left_out() -> None:
    """nvidia-* reached only through torch goes; one another package needs stays, as does sympy."""
    packages = resolved_packages(_ANNOTATED)
    assert packages["nemo-toolkit"][0].startswith("git+https://")
    assert torch_only_cuda_packages(packages) == [
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
        "torch",
        "torchaudio",
        "triton",
    ]


def test_torch_is_bounded_down_until_every_index_serves_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A torch the CUDA or CPU index lacks is excluded and the resolution repeated."""
    compiles: list[str] = []

    def fake_compile(in_file: Path, python: str, constraints: Path, exclude_newer: object, **kwargs: object) -> str:
        bounded = "torch<2.9.0" in constraints.read_text()
        compiles.append(constraints.read_text())
        version = "2.8.0" if bounded else "2.9.0"
        return f"torch=={version}  # via -r requirements.in\nfoo==1.0  # via -r requirements.in\n"

    monkeypatch.setattr(venv_lock, "_compile", fake_compile)
    monkeypatch.setattr(venv_lock, "wheel_on_index", lambda pkg, version, tag, py, platform: version != "2.9.0")
    monkeypatch.setattr(venv_lock, "_uv", lambda: "uv")

    text = compile_lock("t", ["torch>=2.8", "foo"], "3.12", exclude_newer="2026-10-02T00:00:00Z")

    assert "# torch: torch==2.8.0" in text
    assert "# note: torch 2.9.0 skipped" in text
    assert all("lightning!=2.6.2,!=2.6.3" in c for c in compiles)


def test_no_torch_in_range_on_every_index_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When stepping down runs out, the compile fails instead of writing an uninstallable lock."""
    monkeypatch.setattr(venv_lock, "_compile", lambda *a, **k: "torch==2.4.1  # via -r requirements.in\n")
    monkeypatch.setattr(venv_lock, "wheel_on_index", lambda *a: False)
    with pytest.raises(VenvLockError, match="no torch within the requirements"):
        compile_lock("t", ["torch<2.5"], "3.11", max_steps=3)


def test_wheel_on_index_matches_version_python_and_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local-tagged CUDA wheel counts; another Python or platform does not."""
    wheels = [
        "torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
        "torch-2.8.0-cp312-none-macosx_11_0_arm64.whl",
    ]
    monkeypatch.setattr(venv_lock, "_index_wheels", lambda tag, package: wheels)
    assert venv_lock.wheel_on_index("torch", "2.8.0", "cu128", "3.11", venv_lock.LINUX_X86_64)
    assert not venv_lock.wheel_on_index("torch", "2.8.0", "cu128", "3.12", venv_lock.LINUX_X86_64)
    assert venv_lock.wheel_on_index("torch", "2.8.0", "cpu", "3.12", venv_lock.MACOS_ARM64)
    assert not venv_lock.wheel_on_index("torch", "2.8.1", "cu128", "3.11", venv_lock.LINUX_X86_64)


def test_the_newest_index_follows_the_cuda_cap() -> None:
    """An uncapped venv is checked against cu128; a capped one against its ceiling."""
    assert venv_lock.top_cuda_tag(None) == "cu128"
    assert venv_lock.top_cuda_tag((12, 4)) == "cu124"
    assert venv_lock.top_cuda_tag((12, 1)) == "cu121"
