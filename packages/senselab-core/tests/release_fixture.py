"""Fixtures that fabricate wheels and throwaway tagged workspaces; none touches this repository's tags."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import subprocess
import tarfile
import tomllib
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


HATCHLING = next(
    r.split("==")[1]
    for r in tomllib.loads((REPO / "pyproject.toml").read_text())["build-system"]["requires"]
    if r.startswith("hatchling==")
)
"""The hatchling version every pyproject pins, whose output the mini artifacts reproduce."""


MINI_FILES = {
    ".gitignore": "__pycache__/\ndata/\n",
    "LICENSE.md": "license text\n",
    "hatch_build_pins.py": "PIN = 1\n",
    "pyproject.toml": (
        '[project]\nname = "senselab"\ndynamic = ["version", "dependencies"]\nrequires-python = ">=3.11"\n'
        'license = { file = "LICENSE.md" }\n'
        '[project.optional-dependencies]\nnlp = ["jiwer>=3.0"]\nall = ["senselab[nlp]"]\n'
        '[project.scripts]\nsenselab = "senselab.cli:main"\n'
        '[tool.hatch.metadata.hooks.custom]\npin = "senselab-core"\ndependencies = ["numpy>=1.23"]\n'
        '[tool.hatch.build]\nartifacts = ["/src/**/data/**"]\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/senselab"]\n'
        '[tool.hatch.build.targets.sdist]\nonly-include = ["src", "pyproject.toml", "LICENSE.md"]\n'
    ),
    "src/senselab/__init__.py": "\n",
    "src/senselab/cli.py": "def main():\n    return 0\n",
    "src/senselab/data/table.yaml": "a: 1\n",
    "src/tests/a_test.py": "def test_a():\n    pass\n",
    "packages/senselab-core/pyproject.toml": (
        '[project]\nname = "senselab-core"\ndynamic = ["version", "dependencies"]\nrequires-python = ">=3.11"\n'
        'license = "Apache-2.0"\n'
        '[tool.hatch.metadata.hooks.custom]\npin = "senselab-worker"\ndependencies = ["pyyaml"]\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/senselab_core"]\n'
        '[tool.hatch.build.targets.sdist.force-include]\n"../../hatch_build_pins.py" = "_shared_hatch_build_pins.py"\n'
    ),
    "packages/senselab-core/src/senselab_core/__init__.py": "\n",
    "packages/senselab-core/src/senselab_core/kernel/__init__.py": "\n",
    "packages/senselab-core/src/senselab_core/kernel/identity.py": "X = 1\n",
    "packages/senselab-core/tests/b_test.py": "def test_b():\n    pass\n",
    "packages/senselab-worker/pyproject.toml": (
        '[project]\nname = "senselab-worker"\ndynamic = ["version"]\nrequires-python = ">=3.10"\n'
        'license = "Apache-2.0"\ndependencies = ["numpy>=1.23"]\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/senselab_worker"]\n'
    ),
    "packages/senselab-worker/src/senselab_worker/__init__.py": "\n",
    "packages/senselab-worker/src/senselab_worker/canonical.py": "Y = 2\n",
}
"""A miniature of this repository's three distributions: one module, data file and test each, and their pyprojects."""


def mini_repo(root: Path) -> Path:
    """Write ``MINI_FILES`` under ``root`` and add them to a fresh git index (no commit, no tag); return ``root``."""
    for rel, text in MINI_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A", "--force")
    return root


def mini_metadata(name: str, version: str) -> str:
    """The core metadata hatchling writes for the ``mini_repo`` distribution ``name`` at ``version``."""
    lines = {
        "senselab-worker": ["License-Expression: Apache-2.0", "Requires-Python: >=3.10", "Requires-Dist: numpy>=1.23"],
        "senselab-core": [
            "License-Expression: Apache-2.0",
            "Requires-Python: >=3.11",
            "Requires-Dist: pyyaml",
            f"Requires-Dist: senselab-worker=={version}",
        ],
        "senselab": [
            "License-File: LICENSE.md",
            "Requires-Python: >=3.11",
            "Requires-Dist: numpy>=1.23",
            f"Requires-Dist: senselab-core=={version}",
            "Provides-Extra: all",
            "Requires-Dist: jiwer>=3.0; extra == 'all'",
            "Provides-Extra: nlp",
            "Requires-Dist: jiwer>=3.0; extra == 'nlp'",
        ],
    }[name]
    return "\n".join(["Metadata-Version: 2.5", f"Name: {name}", f"Version: {version}", *lines]) + "\n"


MINI_ROOTS = {
    "senselab": "",
    "senselab-core": "packages/senselab-core/",
    "senselab-worker": "packages/senselab-worker/",
}


def _mini_sources(repo: Path, name: str, kind: str) -> dict[str, bytes]:
    """The tracked files a hatch build of ``name`` copies into a ``kind`` artifact, by member path (no top dir)."""
    root = MINI_ROOTS[name]
    package = {"senselab": "senselab", "senselab-core": "senselab_core", "senselab-worker": "senselab_worker"}[name]
    out = {}
    for rel in MINI_FILES:
        if kind == "wheel" and rel.startswith(f"{root}src/{package}/"):
            out[rel.removeprefix(f"{root}src/")] = (repo / rel).read_bytes()
        elif kind == "sdist" and rel.startswith(root) and rel != ".gitignore":
            inner = rel.removeprefix(root)
            if name != "senselab" or inner.split("/")[0] in ("src", "pyproject.toml", "LICENSE.md"):
                out[inner] = (repo / rel).read_bytes()
    if kind == "sdist":
        out[".gitignore"] = (repo / ".gitignore").read_bytes()
        if name == "senselab-core":
            out["_shared_hatch_build_pins.py"] = (repo / "hatch_build_pins.py").read_bytes()
    return out


def write_mini_wheel(
    repo: Path, out: Path, name: str, version: str, edit: Callable[[dict[str, bytes]], None] = lambda m: None
) -> Path:
    """Write the wheel a hatch build of the ``mini_repo`` distribution ``name`` produces; ``edit`` alters its members.

    RECORD is written after ``edit``, so it lists the edited members with their hashes.
    """
    stem = f"{name.replace('-', '_')}-{version}"
    members = _mini_sources(repo, name, "wheel")
    members[f"{stem}.dist-info/METADATA"] = mini_metadata(name, version).encode()
    members[f"{stem}.dist-info/WHEEL"] = (
        f"Wheel-Version: 1.0\nGenerator: hatchling {HATCHLING}\nRoot-Is-Purelib: true\nTag: py3-none-any\n".encode()
    )
    if name == "senselab":
        members[f"{stem}.dist-info/entry_points.txt"] = b"[console_scripts]\nsenselab = senselab.cli:main\n"
        members[f"{stem}.dist-info/licenses/LICENSE.md"] = (repo / "LICENSE.md").read_bytes()
    edit(members)
    record = f"{stem}.dist-info/RECORD"
    rows = []
    for member, data in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{member},sha256={digest},{len(data)}")
    members[record] = ("\n".join([*rows, f"{record},,"]) + "\n").encode()
    path = out / f"{stem}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in members.items():
            archive.writestr(member, data)
    return path


def write_mini_sdist(
    repo: Path, out: Path, name: str, version: str, edit: Callable[[dict[str, bytes]], None] = lambda m: None
) -> Path:
    """Write the sdist a hatch build of the ``mini_repo`` distribution ``name`` produces; ``edit`` alters its members.

    Member names passed to ``edit`` are full tar names, top directory included.
    """
    stem = f"{name.replace('-', '_')}-{version}"
    members = {f"{stem}/{k}": v for k, v in _mini_sources(repo, name, "sdist").items()}
    members[f"{stem}/PKG-INFO"] = mini_metadata(name, version).encode()
    edit(members)
    path = out / f"{stem}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for member, data in members.items():
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def mini_dist(repo: Path, out: Path, version: str = "2.0.0") -> list[Path]:
    """Write all six artifacts of the ``mini_repo`` at ``version`` into ``out``; return them sorted."""
    out.mkdir(parents=True, exist_ok=True)
    for name in MINI_ROOTS:
        write_mini_wheel(repo, out, name, version)
        write_mini_sdist(repo, out, name, version)
    return sorted(out.iterdir())


@pytest.fixture
def mini(tmp_path: Path) -> Path:
    """A ``mini_repo`` under ``tmp_path/repo``."""
    (tmp_path / "repo").mkdir()
    return mini_repo(tmp_path / "repo")
