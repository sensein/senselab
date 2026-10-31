"""Fixtures that fabricate wheels and throwaway tagged workspaces; none touches this repository's tags."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import subprocess
import tarfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
WORKSPACE_FILES = (
    ".gitignore",
    "pyproject.toml",
    "hatch_build_pins.py",
    "README.md",
    "LICENSE.md",
    "SECURITY.md",
    "packages/senselab-core/pyproject.toml",
    "packages/senselab-core/hatch_pin_loader.py",
    "packages/senselab-worker/pyproject.toml",
)
WORKSPACE_TREES = ("packages/senselab-core/src", "packages/senselab-worker/src", "src/senselab")
_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(root: Path, *args: str) -> str:
    """Run git inside the throwaway repository ``root`` with no user configuration."""
    assert REPO not in (root, *root.parents), "a throwaway repository must live outside the real one"
    env = {**os.environ, **_GIT_ENV}
    return subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True, text=True).stdout


def write_wheel(
    directory: Path, name: str, version: str, requires: Sequence[str] = (), modules: dict[str, str] | None = None
) -> Path:
    """Write a minimal wheel of ``name`` at ``version``, with Requires-Dist lines and module sources."""
    stem = f"{name.replace('-', '_')}-{version}"
    path = directory / f"{stem}-py3-none-any.whl"
    meta = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        *(f"Requires-Dist: {r}" for r in requires),
    ]
    files = {
        **(modules or {}),
        f"{stem}.dist-info/METADATA": "\n".join(meta) + "\n",
        f"{stem}.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = f"{stem}.dist-info/RECORD"
    lines = []
    for member, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        lines.append(f"{member},sha256={digest},{len(text.encode())}")
    files[record] = "\n".join([*lines, f"{record},,"]) + "\n"
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in files.items():
            archive.writestr(member, text)
    return path


def write_sdist(directory: Path, name: str, version: str) -> Path:
    """Write a minimal sdist of ``name`` at ``version`` holding only its PKG-INFO."""
    stem = f"{name.replace('-', '_')}-{version}"
    path = directory / f"{stem}.tar.gz"
    info = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(f"{stem}/PKG-INFO")
        member.size = len(info)
        archive.addfile(member, io.BytesIO(info))
    return path


@pytest.fixture
def fabricate_sdist(tmp_path: Path) -> Callable[..., Path]:
    """Return ``fabricate_sdist(name, version)``, writing an sdist beside the fabricated wheels."""
    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)
    return lambda *a, **k: write_sdist(dist, *a, **k)


@pytest.fixture
def fabricate(tmp_path: Path) -> Callable[..., Path]:
    """Return ``fabricate(name, version, requires=(), modules=None)``, writing a wheel under ``tmp_path/dist``."""
    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)
    return lambda *a, **k: write_wheel(dist, *a, **k)


@pytest.fixture
def tagged_workspace(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Path]:
    """Return ``make(tags, commits_after=0)``: a copy of the packaging inputs as a fresh git repo with those tags.

    Every tag sits on the first commit; ``commits_after`` further commits follow it.
    """

    def make(tags: Sequence[str], commits_after: int = 0) -> Path:
        root = tmp_path_factory.mktemp("workspace")
        listing = ["git", "ls-files", "-z", "--", *WORKSPACE_TREES]
        listed = subprocess.run(listing, cwd=REPO, check=True, capture_output=True, text=True).stdout.split("\0")
        for rel in (*WORKSPACE_FILES, *(p for p in listed if p)):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / rel).read_bytes())
        _git(root, "init", "-q", "-b", "main")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "first", "--no-gpg-sign")
        for tag in tags:
            _git(root, "tag", tag)
        for i in range(commits_after):
            _git(root, "commit", "-q", "--allow-empty", "-m", f"c{i}", "--no-gpg-sign")
        return root

    return make
