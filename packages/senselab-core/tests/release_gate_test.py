"""The release gate refuses what cannot be released, and the build writes only what can be installed."""

from __future__ import annotations

import functools
import http.server
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import tomllib
import types
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from packaging.version import Version

pytest_plugins = ["release_fixture"]

REPO = Path(__file__).resolve().parents[3]


def build(root: Path, out: Path, *args: str) -> list[Path]:
    """``uv build`` in ``root`` into ``out``; return the files it wrote."""
    subprocess.run(["uv", "build", "--out-dir", str(out), *args], cwd=root, check=True, capture_output=True, text=True)
    return sorted(out.iterdir())


_spec = importlib.util.spec_from_file_location("release_gate", REPO / "scripts" / "release_gate.py")
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
sys.modules["release_gate"] = gate
_spec.loader.exec_module(gate)

PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())
RELEASE_FLOOR = PYPROJECT["tool"]["senselab"]["release"]["RELEASE_FLOOR"]
FLOOR = Version(RELEASE_FLOOR)


def _release(fabricate: Callable[..., Path], version: str, **mod: dict[str, Any]) -> list[Path]:
    """Fabricate the three distributions at ``version``, each sibling pinned exactly."""
    return [
        fabricate("senselab-worker", version, **mod.get("worker", {})),
        fabricate("senselab-core", version, [f"senselab-worker=={version}"], **mod.get("core", {})),
        fabricate("senselab", version, [f"senselab-core=={version}"], **mod.get("root", {})),
    ]


@pytest.mark.logic
def test_release_floor_is_a_final_version_in_the_root_pyproject() -> None:
    """The floor is declared once, in the root pyproject, and the workflow does not restate it."""
    assert not FLOOR.is_devrelease and not FLOOR.is_postrelease and FLOOR.local is None
    assert "RELEASE_FLOOR" not in (REPO / ".github/workflows/publish.yaml").read_text()


@pytest.mark.logic
@pytest.mark.parametrize(
    "v", ["2.0.0", "2.0.1", "2.1.0rc1", "2.1.0b2", "3.0.0a1", "10.0.0", "2.0.0a0", "2.0.0a1", "2.0.0rc1"]
)
def test_gate_accepts_a_releasable_version(v: str, fabricate: Callable[..., Path]) -> None:
    """Final versions and tag-derived pre-releases at or above the floor pass the version and closure checks."""
    wheels = _release(fabricate, v)
    gate.check_versions(wheels, FLOOR)
    gate.check_closure(wheels)


@pytest.mark.logic
@pytest.mark.parametrize(
    "v",
    [
        "0.1.dev3",
        "2.0.0+g1234",
        "1.9.0",
        "2.0.0.dev1",
        "2.0.1.dev7",
        "1.4.0a12.dev89",
        "2.0.0.post1",
        "1.9.0rc1",
        "1.0.0a1",
        "2.0.0+local.1",
        "0.0.0",
        "1!1.0.0",
        "1!2.0.0",
        "2.0",
        "2.0.0.0",
        "2.0.0rc1.post1",
    ],
)
def test_a29_release_gate_refuses_version(v: str, fabricate: Callable[..., Path]) -> None:
    """A dev, local or below-floor version is refused by the version check."""
    with pytest.raises(gate.GateError):
        gate.check_versions(_release(fabricate, v), FLOOR)


@pytest.mark.logic
def test_a29_release_gate_refuses_wheels_that_disagree(fabricate: Callable[..., Path]) -> None:
    """Three wheels at two versions are refused even when each is releasable alone."""
    wheels = [
        fabricate("senselab-worker", "2.0.0"),
        fabricate("senselab-core", "2.0.1"),
        fabricate("senselab", "2.0.1"),
    ]
    with pytest.raises(gate.GateError, match="disagree"):
        gate.check_versions(wheels, FLOOR)


@pytest.mark.logic
@pytest.mark.parametrize("absent", gate.DISTRIBUTIONS)
def test_a11_release_gate_refuses_missing_sibling(absent: str, fabricate: Callable[..., Path]) -> None:
    """Dropping any one of the three wheels is refused, whichever it is."""
    wheels = [w for w in _release(fabricate, "2.0.0") if not w.name.startswith(absent.replace("-", "_") + "-")]
    assert len(wheels) == 2
    with pytest.raises(gate.GateError, match=absent):
        gate.check_closure(wheels)


@pytest.mark.logic
def test_a11_release_gate_refuses_the_root_wheel_alone(fabricate: Callable[..., Path]) -> None:
    """The wheel a bare ``uv build`` writes is refused."""
    with pytest.raises(gate.GateError):
        gate.check_closure([fabricate("senselab", "2.0.0", ["senselab-core==2.0.0"])])


@pytest.mark.logic
@pytest.mark.parametrize(
    "pin",
    [
        "senselab-worker>=2.0.0",
        "senselab-worker==2.0.1",
        "senselab-worker",
        "senselab-worker~=2.0",
        "senselab-worker==2.0.*",
        "senselab-worker===2.0.0",
        'senselab-worker==2.0.0; python_version < "3"',
        'senselab-worker==2.0.0; extra == "full"',
        "senselab-worker[gpu]==2.0.0",
        "senselab-worker @ https://example.invalid/w.whl",
    ],
)
def test_a32_release_gate_refuses_a_sibling_requirement_that_is_not_the_exact_version(
    pin: str, fabricate: Callable[..., Path]
) -> None:
    """Only ``==<this version>`` satisfies a sibling requirement; a range admits another or a local build."""
    wheels = [
        fabricate("senselab-worker", "2.0.0"),
        fabricate("senselab-core", "2.0.0", [pin]),
        fabricate("senselab", "2.0.0", ["senselab-core==2.0.0"]),
    ]
    with pytest.raises(gate.GateError, match="senselab-core requires"):
        gate.check_closure(wheels)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("v", "reason"), [("2.0.0.dev1", "dev build"), ("1.9.0", "below"), ("2.0.0+g1", "local"), ("2.0.0.post1", "post")]
)
def test_main_refuses_and_names_the_reason(
    v: str,
    reason: str,
    fabricate: Callable[..., Path],
    fabricate_sdist: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` returns 1 with the reason on stderr before it creates any environment."""
    dist = _release(fabricate, v)[0].parent
    for name in gate.DISTRIBUTIONS:
        fabricate_sdist(name, v)
    assert gate.main(dist, repo=dist) == 1
    assert reason in capsys.readouterr().err


PY_TAG = f"{sys.version_info.major}{sys.version_info.minor}"
HERE_PYTHON = {name: f"{sys.version_info.major}.{sys.version_info.minor}" for name in gate.DISTRIBUTIONS}
OK_MODULES: dict[str, dict[str, dict[str, str]]] = {
    "worker": {"modules": {"senselab_worker/__init__.py": "\n"}},
    "core": {"modules": {"senselab_core/__init__.py": "\n"}},
    "root": {"modules": {"senselab/__init__.py": "\n"}},
}


@pytest.mark.env
@pytest.mark.network
def test_check_installs_installs_each_distribution_alone_and_imports_it(fabricate: Callable[..., Path]) -> None:
    """A dist whose three wheels import installs, one distribution per venv, from ``dist/`` paths."""
    wheels = _release(fabricate, "2.0.0", **OK_MODULES)
    gate.check_installs(gate.dist_files(wheels[0].parent), HERE_PYTHON)


@pytest.mark.env
@pytest.mark.network
def test_a11_check_installs_refuses_a_wheel_that_does_not_import(fabricate: Callable[..., Path]) -> None:
    """An installable dist whose root package raises on import is refused."""
    wheels = _release(
        fabricate,
        "2.0.0",
        **{**OK_MODULES, "root": {"modules": {"senselab/__init__.py": "raise ImportError('broken')\n"}}},
    )
    with pytest.raises(gate.GateError, match="broken"):
        gate.check_installs(gate.dist_files(wheels[0].parent), HERE_PYTHON)


@pytest.mark.env
@pytest.mark.network
@pytest.mark.parametrize("culprit", ["worker", "core", "root"])
def test_i1_check_installs_refuses_an_undeclared_dependency(culprit: str, fabricate: Callable[..., Path]) -> None:
    """A distribution importing a third-party package its metadata omits fails in its own venv, whoever it is."""
    modules = {**OK_MODULES, culprit: {"modules": {next(iter(OK_MODULES[culprit]["modules"])): "import numpy\n"}}}
    wheels = _release(fabricate, "2.0.0", **modules)
    with pytest.raises(gate.GateError, match="numpy"):
        gate.check_installs(gate.dist_files(wheels[0].parent), HERE_PYTHON)


@pytest.mark.logic
@pytest.mark.parametrize("name", gate.DISTRIBUTIONS)
def test_c1_release_gate_refuses_a_wheel_without_its_pin(name: str, fabricate: Callable[..., Path]) -> None:
    """A wheel whose requirements differ from the graph's, by omission or by an invented edge, is refused."""
    graph = {"senselab": ["senselab-core==2.0.0"], "senselab-core": ["senselab-worker==2.0.0"], "senselab-worker": []}
    # the worker has no pin to drop, so it gains one
    graph[name] = ["senselab-core==2.0.0"] if name == "senselab-worker" else []
    wheels = [fabricate(n, "2.0.0", graph[n]) for n in gate.DISTRIBUTIONS]
    with pytest.raises(gate.GateError, match="the graph requires"):
        gate.check_closure(wheels)


@pytest.mark.logic
@pytest.mark.parametrize("extra", ["senselab", "senselab-core", "senselab-worker"])
def test_c1_release_gate_refuses_a_sibling_the_graph_does_not_give(extra: str, fabricate: Callable[..., Path]) -> None:
    """A wheel requiring a sibling beyond its one declared edge is refused."""
    wheels = [
        fabricate("senselab-worker", "2.0.0"),
        fabricate("senselab-core", "2.0.0", ["senselab-worker==2.0.0"]),
        fabricate("senselab", "2.0.0", ["senselab-core==2.0.0", f"{extra}==2.0.0"]),
    ]
    with pytest.raises(gate.GateError, match="the graph requires"):
        gate.check_closure(wheels)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("tag", "prerelease", "ok"),
    [
        ("2.0.0", False, True),
        ("2.0.0", None, True),
        ("2.1.0rc1", True, True),
        ("2.1.0-rc.1", True, True),
        ("2.1.0-alpha.1", True, False),
        ("v2.1.0rc1", True, False),
        ("v2.0.0", False, False),
        ("2.1.0", False, False),
        ("2.0.0", True, False),
        ("2.1.0rc1", False, False),
        ("2.1.1", False, False),
        ("garbage", None, False),
    ],
)
def test_i2_release_gate_binds_the_version_to_the_tag_and_prerelease_flag(
    tag: str, prerelease: bool | None, ok: bool, fabricate: Callable[..., Path]
) -> None:
    """The tag, canonical or ``X.Y.Z-(alpha|beta|rc).N``, names the built version; the prerelease flag matches."""
    version = "2.1.0rc1" if tag.startswith("2.1.0") else "2.0.0"
    wheels = _release(fabricate, version)
    if ok:
        gate.check_versions(wheels, FLOOR, tag=tag, prerelease=prerelease)
    else:
        with pytest.raises(gate.GateError):
            gate.check_versions(wheels, FLOOR, tag=tag, prerelease=prerelease)


@pytest.mark.logic
def test_m4_release_gate_checks_every_file_in_dist(
    fabricate: Callable[..., Path], fabricate_sdist: Callable[..., Path]
) -> None:
    """A stale sdist, a duplicate, a foreign file or a missing sdist each refuse; the complete set passes."""
    wheels = _release(fabricate, "2.0.0")
    sdists = [fabricate_sdist(n, "2.0.0") for n in gate.DISTRIBUTIONS]
    dist = wheels[0].parent
    files = gate.dist_files(dist)
    gate.check_files(files)
    gate.check_versions(files, FLOOR)
    stale = fabricate_sdist("senselab", "1.9.0")
    with pytest.raises(gate.GateError, match="disagree"):
        gate.check_versions(gate.dist_files(dist), FLOOR)
    with pytest.raises(gate.GateError, match="exactly one sdist"):
        gate.check_files(gate.dist_files(dist))
    stale.unlink()
    sdists[0].unlink()
    with pytest.raises(gate.GateError, match="exactly one sdist"):
        gate.check_files(gate.dist_files(dist))
    sdists[0] = fabricate_sdist("senselab", "2.0.0")
    for foreign in ("notes.txt", "senselab-2.0.0.zip"):
        (dist / foreign).write_text("x")
        with pytest.raises(gate.GateError, match="neither wheel nor sdist"):
            gate.dist_files(dist)
        (dist / foreign).unlink()
    (dist / ".gitignore").write_text("*\n")
    assert gate.dist_files(dist) == files
    (dist / "sub").mkdir()
    with pytest.raises(gate.GateError, match="neither wheel nor sdist"):
        gate.dist_files(dist)


def _core_files(root: Path, out: Path) -> list[Path]:
    """Build core in ``root`` and return its wheel and sdist."""
    return [p for p in build(root, out, "--package", "senselab-core") if not p.name.startswith(".")]


@pytest.mark.env
@pytest.mark.network
def test_n3_clean_build_contents_are_all_tracked(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """Every file in a clean build's wheel and sdist maps to a tracked file or a named generated one."""
    root = tagged_workspace(["2.0.0"])
    gate.check_members(_core_files(root, tmp_path / "out"), root)


@pytest.mark.env
@pytest.mark.network
@pytest.mark.parametrize("where", ["info-exclude", "untracked", "nested-untracked"])
def test_n3_release_gate_refuses_a_shipped_file_git_does_not_track(
    where: str, tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """A stray file that reaches an artifact is refused however git hides it, including ``.git/info/exclude``."""
    root = tagged_workspace(["2.0.0"])
    src = root / "packages/senselab-core/src/senselab_core"
    (src / "sub").mkdir()
    stray = src / ("sub/leak.txt" if where == "nested-untracked" else "leak.txt")
    stray.write_text("x")
    if where == "info-exclude":
        (root / ".git/info/exclude").write_text("leak.txt\n")
    files = _core_files(root, tmp_path / "out")
    assert any("leak.txt" in n for p in files if p.suffix == ".whl" for n in zipfile.ZipFile(p).namelist())
    with pytest.raises(gate.GateError, match="leak.txt"):
        gate.check_members(files, root)


@pytest.mark.env
@pytest.mark.network
def test_n3_gitignored_file_does_not_ship_and_is_not_refused(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """A file ignored by ``.gitignore`` is dropped by the build, so the contents check has nothing to refuse."""
    root = tagged_workspace(["2.0.0"])
    (root / "packages/senselab-core/src/senselab_core/x.log").write_text("x")
    (root / ".gitignore").write_text("*.log\n")
    gate.check_members(_core_files(root, tmp_path / "out"), root)


def _rewrite_sdist(path: Path, edits: dict[str, bytes | tuple[str, str]]) -> None:
    """Rewrite the sdist at ``path``: bytes replace or add a member; ``(kind, target)`` adds a non-regular one."""
    with tarfile.open(path) as tar:
        top = tar.getnames()[0].split("/")[0]
        kept = [(m, tar.extractfile(m).read() if m.isreg() else None) for m in tar.getmembers()]  # type: ignore[union-attr]
    with tarfile.open(path, "w:gz") as out:
        for member, data in kept:
            replaced = next((v for k, v in edits.items() if member.name == f"{top}/{k}" and isinstance(v, bytes)), None)
            data = replaced if replaced is not None else data
            if data is not None:
                member.size = len(data)
                out.addfile(member, io.BytesIO(data))
            else:
                out.addfile(member)
        for name, edit in edits.items():
            if isinstance(edit, bytes):
                if not any(m.name == f"{top}/{name}" for m, _ in kept):
                    info = tarfile.TarInfo(f"{top}/{name}")
                    info.size = len(edit)
                    out.addfile(info, io.BytesIO(edit))
                continue
            kind, target = edit
            info = tarfile.TarInfo(f"{top}/{name}")
            info.type = {
                "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE,
                "fifo": tarfile.FIFOTYPE,
                "chr": tarfile.CHRTYPE,
            }[kind]
            info.linkname = target
            out.addfile(info)


def _rewrite_wheel(path: Path, edits: dict[str, bytes | str]) -> None:
    """Rewrite the wheel at ``path``: bytes replace or add a member; the string ``"symlink"`` adds a symlink entry."""
    with zipfile.ZipFile(path) as archive:
        kept = {n: archive.read(n) for n in archive.namelist()}
    with zipfile.ZipFile(path, "w") as out:
        for name, data in {**kept, **{k: v for k, v in edits.items() if isinstance(v, bytes)}}.items():
            out.writestr(name, data)
        for name, kind in edits.items():
            if kind == "symlink":
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                out.writestr(info, "/etc/passwd")


@pytest.mark.env
@pytest.mark.network
def test_m1_m2_m4_mutating_a_real_build_is_refused(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """Aliases are bound to content, extra dist-info or data dirs and non-regular entries are refused."""
    root = tagged_workspace(["2.0.0"])
    built = _core_files(root, tmp_path / "out")
    gate.check_members(built, root)
    wheel, sdist = (next(p for p in built if p.suffix == s) for s in (".whl", ".gz"))
    stem = "senselab_core-2.0.0"
    wheel_cases: dict[str, dict[str, bytes | str]] = {
        "second dist-info": {"evil-1.0.dist-info/entry_points.txt": b"[console_scripts]\nx=y:z\n"},
        "other-version dist-info": {"senselab_core-9.9.9.dist-info/entry_points.txt": b"x"},
        "data dir": {f"{stem}.data/scripts/run": b"#!/bin/sh\n"},
        "license not named": {f"{stem}.dist-info/licenses/pyproject.toml": b"x"},
        "symlink entry": {"senselab_core/link.py": "symlink"},
        "changed package file": {"senselab_core/py.typed": b"changed"},
        "unknown dist-info file": {f"{stem}.dist-info/extra.txt": b"x"},
    }
    for label, edits in wheel_cases.items():
        target = tmp_path / label.replace(" ", "_") / wheel.name
        target.parent.mkdir()
        shutil.copy(wheel, target)
        _rewrite_wheel(target, edits)
        with pytest.raises(gate.GateError, match="differ from the tracked sources"):
            gate.check_members([target], root)
    sdist_cases: dict[str, dict[str, bytes | tuple[str, str]]] = {
        "alias content": {"_shared_hatch_build_pins.py": b"print('evil')\n"},
        "gitignore content": {".gitignore": b"!*\n"},
        "symlink": {"link": ("symlink", "/etc/passwd")},
        "hardlink": {"hard": ("hardlink", "src/senselab_core/__init__.py")},
        "fifo": {"pipe": ("fifo", "")},
        "device": {"dev": ("chr", "")},
    }
    for label, sdist_edits in sdist_cases.items():
        target = tmp_path / f"s-{label.replace(' ', '_')}" / sdist.name
        target.parent.mkdir()
        shutil.copy(sdist, target)
        _rewrite_sdist(target, sdist_edits)
        with pytest.raises(gate.GateError, match="differ from the tracked sources"):
            gate.check_members([target], root)


@pytest.mark.env
@pytest.mark.network
def test_m1_an_untracked_gitignore_in_a_package_root_that_ships_is_refused(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """The sdist ``.gitignore`` must equal a tracked one: an untracked file with other content ships and fails."""
    root = tagged_workspace(["2.0.0"])
    (root / "packages/senselab-core/.gitignore").write_text("!*\n")
    files = _core_files(root, tmp_path / "out")
    with pytest.raises(gate.GateError, match="gitignore"):
        gate.check_members(files, root)


@pytest.mark.env
@pytest.mark.network
def test_m1_the_root_distribution_license_must_be_a_named_tracked_copy(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """The root wheel's license file passes when it is the tracked LICENSE.md and fails when its bytes differ."""
    root = tagged_workspace(["2.0.0"])
    files = [p for p in build(root, tmp_path / "out", "--package", "senselab") if not p.name.startswith(".")]
    gate.check_members(files, root)
    wheel = next(p for p in files if p.suffix == ".whl")
    names = zipfile.ZipFile(wheel).namelist()
    license_member = next(n for n in names if "/licenses/" in n)
    _rewrite_wheel(wheel, {license_member: b"forged"})
    with pytest.raises(gate.GateError, match="licenses"):
        gate.check_members([wheel], root)


@pytest.mark.env
@pytest.mark.network
def test_m2_sibling_verification_binds_content_not_version(fabricate: Callable[..., Path], tmp_path: Path) -> None:
    """A sibling installed at the right version but from other files than the dist wheel fails the check."""
    dist = fabricate("senselab-worker", "2.0.0", modules={"senselab_worker/__init__.py": "A = 1\n"}).parent
    genuine = gate.wheel_file_hashes(next(dist.glob("*.whl")))
    other = tmp_path / "other"
    other.mkdir()
    sys.modules["release_fixture"].write_wheel(
        other, "senselab-worker", "2.0.0", modules={"senselab_worker/__init__.py": "A = 2\n"}
    )
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "-q", "--python", HERE_PYTHON["senselab"], str(venv)], check=True)
    python = str(venv / "bin/python")
    subprocess.run(
        ["uv", "pip", "install", "-q", "--no-cache", "--python", python, str(next(other.glob("*.whl")))], check=True
    )
    run = lambda files: subprocess.run(  # noqa: E731
        [python, "-c", gate._VERIFY, json.dumps({"senselab-worker": {"version": "2.0.0", "files": files}})],
        capture_output=True,
        text=True,
    )
    assert run(gate.wheel_file_hashes(next(other.glob("*.whl")))).returncode == 0
    refused = run(genuine)
    assert refused.returncode != 0 and "not installed from dist" in refused.stderr


@pytest.mark.logic
def test_n3_contents_check_refuses_a_directory_that_is_not_a_repository(
    fabricate: Callable[..., Path], tmp_path: Path
) -> None:
    """Without a git work tree the contents cannot be checked, and the gate says so rather than passing."""
    wheels = _release(fabricate, "2.0.0")
    with pytest.raises(gate.GateError, match="not a git work tree"):
        gate.check_members(wheels, tmp_path)


@pytest.mark.env
@pytest.mark.network
def test_n1_check_installs_exercises_the_sibling_edge(fabricate: Callable[..., Path]) -> None:
    """A root that imports core but pins nothing fails in its own venv: siblings are reached only through the pin."""
    wheels = [
        fabricate("senselab-worker", "2.0.0", modules=OK_MODULES["worker"]["modules"]),
        fabricate("senselab-core", "2.0.0", ["senselab-worker==2.0.0"], modules=OK_MODULES["core"]["modules"]),
        fabricate("senselab", "2.0.0", [], modules=OK_MODULES["root"]["modules"]),
    ]
    with pytest.raises(gate.GateError, match="senselab_core"):
        gate.check_installs(gate.dist_files(wheels[0].parent), HERE_PYTHON)


@pytest.mark.env
@pytest.mark.network
def test_n2_a_dotfile_wheel_cannot_stand_in_for_the_checked_one(fabricate: Callable[..., Path]) -> None:
    """Install and checks share one listing, so a dot-named broken wheel is neither checked nor installed."""
    wheels = _release(fabricate, "2.0.0", **OK_MODULES)
    dist = wheels[0].parent
    (dist / ".senselab-2.0.0-py3-none-any.whl").write_bytes(b"not a wheel")
    gate.check_installs(gate.dist_files(dist), HERE_PYTHON)


def _first_party_cache_entries(cache: Path) -> list[str]:
    """Return every path under the uv cache ``cache`` whose name is one of the three distributions."""
    names = {*gate.DISTRIBUTIONS, *(n.replace("-", "_") for n in gate.DISTRIBUTIONS)}
    return sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if p.name.split("-2.")[0] in names)


@pytest.mark.env
@pytest.mark.network
def test_c4_a_cached_wheel_of_the_same_name_and_version_cannot_stand_in_for_dist(
    fabricate: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other bytes at the same name, version and dist path, already in the host's uv cache, are not installed."""
    cache = tmp_path / "uv-cache"
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))
    stale = {k: {"modules": {next(iter(v["modules"])): "STALE = True\n"}} for k, v in OK_MODULES.items()}
    dist = _release(fabricate, "2.0.0", **stale)[0].parent
    scratch = tmp_path / "scratch"
    subprocess.run(["uv", "venv", "-q", "--python", HERE_PYTHON["senselab"], str(scratch)], check=True)
    subprocess.run(
        ["uv", "pip", "install", "-q", "--python", str(scratch / "bin/python"), "--find-links", str(dist), "senselab"],
        check=True,
    )
    assert _first_party_cache_entries(cache), "the stale wheels did not reach the cache, so nothing is tested"
    for path in dist.iterdir():
        path.unlink()
    wheels = _release(fabricate, "2.0.0", **OK_MODULES)
    assert wheels[0].parent == dist
    gate.check_installs(gate.dist_files(dist), HERE_PYTHON)


@pytest.mark.env
@pytest.mark.network
def test_c4_the_gate_leaves_no_first_party_wheel_in_the_host_cache(
    fabricate: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a passing run the host's uv cache holds nothing of the three distributions for a later run to reuse."""
    cache = tmp_path / "uv-cache"
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))
    wheels = _release(fabricate, "2.0.0", **OK_MODULES)
    gate.check_installs(gate.dist_files(wheels[0].parent), HERE_PYTHON)
    assert _first_party_cache_entries(cache) == []


@pytest.mark.logic
def test_n4_floor_pythons_are_read_from_each_pyproject(tmp_path: Path) -> None:
    """Each distribution's floor is the lower bound of its own ``requires-python``; none is a literal in the gate."""
    for name, root in gate.DIST_ROOTS.items():
        (tmp_path / root).mkdir(parents=True, exist_ok=True)
        bound = {"senselab": "3.8", "senselab-core": "3.9", "senselab-worker": "3.13"}[name]
        (tmp_path / root / "pyproject.toml").write_text(f'[project]\nrequires-python = ">={bound},<3.99"\n')
    assert gate.floor_pythons(tmp_path) == {"senselab": "3.8", "senselab-core": "3.9", "senselab-worker": "3.13"}


@pytest.mark.logic
def test_n4_root_is_tested_at_its_declared_floor() -> None:
    """The real floors are those the pyprojects declare, so the root is import-tested at the lowest Python it allows."""
    floors = gate.floor_pythons()
    assert floors["senselab"] == "3.11" and floors["senselab-core"] == "3.11" and floors["senselab-worker"] == "3.10"


@pytest.mark.logic
def test_a11_publish_builds_every_package_and_gates_before_upload() -> None:
    """The job fetches tags, builds all packages, gates with the tag, and uploads only on a release event."""
    workflow = yaml.safe_load((REPO / ".github/workflows/publish.yaml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"release", "workflow_dispatch"} and triggers["release"] == {"types": ["published"]}
    steps = workflow["jobs"]["pypi-release"]["steps"]
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["fetch-depth"] == 0 and checkout["with"]["fetch-tags"] is True
    commands = [s.get("run", "") for s in steps]
    index = {
        key: next(i for i, c in enumerate(commands) if key in c)
        for key in ("uv build", "release_gate.py", "uv publish")
    }
    assert index["uv build"] < index["release_gate.py"] < index["uv publish"]
    assert "--all-packages" in commands[index["uv build"]]
    gate_command = commands[index["release_gate.py"]]
    gate_step = steps[index["release_gate.py"]]
    assert gate_step["env"]["REF_NAME"] == "${{ github.ref_name }}"
    assert gate_step["env"]["PRERELEASE"] == "${{ github.event.release.prerelease }}"
    assert '--tag "$REF_NAME"' in gate_command and '--prerelease "$PRERELEASE"' in gate_command
    publish = steps[index["uv publish"]]
    assert publish["if"] == "github.event_name == 'release'" and "dist/*" in publish["run"]
    assert all("if" not in s for s in steps[: index["uv publish"]] if "release_gate.py" in s.get("run", ""))


@pytest.mark.logic
def test_i3_a_ci_job_selects_the_core_env_tests() -> None:
    """Some tests.yaml job runs the ``env`` tier over the core tests, so the release regressions run in CI."""
    workflow = yaml.safe_load((REPO / ".github/workflows/tests.yaml").read_text())
    runs = [s.get("run", "") for job in workflow["jobs"].values() for s in job.get("steps", [])]
    assert any("packages/senselab-core/tests" in r and "-m env" in r.replace('"', "").replace("'", "") for r in runs)


@pytest.mark.logic
def test_a11_every_workspace_member_is_a_published_distribution() -> None:
    """``--all-packages`` builds the workspace members; the gate requires exactly those distributions."""
    assert PYPROJECT["tool"]["uv"]["workspace"]["members"] == ["packages/*"]
    names = {tomllib.loads(p.read_text())["project"]["name"] for p in (REPO / "packages").glob("*/pyproject.toml")}
    assert names | {PYPROJECT["project"]["name"]} == set(gate.DISTRIBUTIONS)


_TAGS = [
    (["2.0.0"], 0, True),
    (["2.1.0"], 0, True),
    (["2.1.0rc1"], 0, True),
    (["2.0.0"], 2, False),
    (["v2.0.0"], 0, False),
    ([], 0, False),
    (["1.9.0"], 0, False),
    (["2.0.0rc1"], 0, True),
    (["2.0.0a1"], 0, True),
    (["2.0.0.post1"], 0, False),
]


@pytest.mark.env
@pytest.mark.network
@pytest.mark.parametrize(("tags", "after", "releasable"), _TAGS)
def test_a29_version_from_the_tag_history_is_gated(
    tags: list[str], after: int, releasable: bool, tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """Whatever hatch-vcs writes for a tag history is releasable exactly when it is a floor-or-above tag build."""
    root = tagged_workspace(tags, after)
    (wheel,) = [
        p for p in build(root, tmp_path / "out", "--wheel", "--package", "senselab-worker") if p.suffix == ".whl"
    ]
    if releasable:
        gate.check_versions([wheel], FLOOR)
    else:
        with pytest.raises(gate.GateError):
            gate.check_versions([wheel], FLOOR)


@pytest.mark.env
@pytest.mark.network
def test_version_is_at_least_the_release_floor(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """A checkout tagged at the floor builds all three distributions at a version of at least the floor."""
    root = tagged_workspace([RELEASE_FLOOR])
    files = build(root, tmp_path / "out", "--all-packages")
    wheels = [p for p in files if p.suffix == ".whl"]
    assert len(wheels) == 3
    gate.check_versions(wheels, FLOOR)
    gate.check_closure(wheels)
    assert {gate._metadata(w)[1] for w in wheels} == {FLOOR}


@pytest.mark.env
@pytest.mark.network
def test_a32_sdists_are_self_contained_and_the_pin_is_exact(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """The core sdist carries the hook loader and the shared hook as regular files, and its wheel pins the worker."""
    root = tagged_workspace(["2.3.4"])
    files = build(root, tmp_path / "out", "--package", "senselab-core")
    (sdist,) = [p for p in files if p.name.endswith(".tar.gz")]
    (wheel,) = [p for p in files if p.suffix == ".whl"]
    with tarfile.open(sdist) as archive:
        members = {m.name.split("/", 1)[1]: m for m in archive.getmembers() if "/" in m.name}
    assert not any(m.issym() or m.islnk() for m in members.values())
    assert {"hatch_pin_loader.py", "_shared_hatch_build_pins.py"} <= set(members)
    with zipfile.ZipFile(wheel) as archive:
        meta = archive.read(next(n for n in archive.namelist() if n.endswith("METADATA"))).decode()
    assert "Requires-Dist: senselab-worker==2.3.4\n" in meta


def _load_hook(path: str, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: ANN401
    """Load the build hook at ``path`` with a stand-in for hatchling's plugin interface; return its module."""

    class MetadataHookInterface:
        PLUGIN_NAME = ""

        def __init__(self, root: str, config: dict[str, Any]) -> None:
            self.root, self.config = root, config

    for name in ("hatchling", "hatchling.metadata", "hatchling.metadata.plugin", "hatchling.metadata.plugin.interface"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["hatchling.metadata.plugin.interface"].MetadataHookInterface = MetadataHookInterface  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location(f"hook_{abs(hash(path))}", REPO / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOOKS = [
    ("hatch_build_pins.py", "pyproject.toml", "senselab-core"),
    ("packages/senselab-core/hatch_pin_loader.py", "packages/senselab-core/pyproject.toml", "senselab-worker"),
]


@pytest.mark.logic
@pytest.mark.parametrize(("path", "pyproject", "pin"), HOOKS)
@pytest.mark.parametrize("version", ["2.0.0+g1234", "2.0.0+local", "2.0.0+a.1", "1.0+1"])
def test_a32_pin_hook_refuses_a_local_segment(
    path: str, pyproject: str, pin: str, version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both the shared hook and core's loader raise rather than write ``==X+local``."""
    hook = _load_hook(path, monkeypatch).PinHook(str(REPO), {"pin": pin, "dependencies": ["numpy"]})
    with pytest.raises(ValueError, match="local"):
        hook.update({"version": version})


@pytest.mark.logic
@pytest.mark.parametrize(("path", "pyproject", "pin"), HOOKS)
@pytest.mark.parametrize("version", ["2.0.0", "2.0.0a3", "2.1.0rc1", "10.20.30", "1.4.0a12.dev90"])
def test_versions_are_pinned_equal(
    path: str, pyproject: str, pin: str, version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever the version, each distribution's hook, with its own declared config, pins its sibling exactly.

    This reads the hook's output, not the installed versions, which drift between commits in an editable workspace.
    """
    config = tomllib.loads((REPO / pyproject).read_text())["tool"]["hatch"]["metadata"]["hooks"]["custom"]
    assert config["pin"] == pin
    hook = _load_hook(path, monkeypatch).PinHook(str(REPO), config)
    out: dict[str, Any] = {"version": version}
    hook.update(out)
    assert out["dependencies"] == [*config["dependencies"], f"{pin}=={version}"]
    assert out["dependencies"].count(f"{pin}=={version}") == 1


@pytest.mark.env
@pytest.mark.network
def test_i2_two_release_tags_on_one_commit_build_the_older_version(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """With 2.0.0 and 2.1.0rc1 on one commit the build is 2.0.0, so a release for the rc is refused by its tag."""
    root = tagged_workspace(["2.0.0", "2.1.0rc1"])
    (wheel,) = [
        p for p in build(root, tmp_path / "out", "--wheel", "--package", "senselab-worker") if p.suffix == ".whl"
    ]
    gate.check_versions([wheel], FLOOR, tag="2.0.0", prerelease=False)
    with pytest.raises(gate.GateError, match="2.1.0rc1"):
        gate.check_versions([wheel], FLOOR, tag="2.1.0rc1", prerelease=True)


@pytest.mark.env
@pytest.mark.network
@pytest.mark.slow
def test_release_gate_installs_dist(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """Build every package from a tagged checkout; the gate installs each distribution at its floor Python."""
    root = tagged_workspace([RELEASE_FLOOR])
    build(root, root / "dist", "--all-packages")
    assert gate.main(root / "dist", tag=RELEASE_FLOOR, prerelease=True, repo=root) == 0


def _rows(wheel: Path) -> tuple[str, list[list[str]], dict[str, bytes]]:
    """Return the wheel's RECORD name, its rows and every member's bytes."""
    with zipfile.ZipFile(wheel) as archive:
        members = {i.filename: archive.read(i) for i in archive.infolist()}
    record = next(n for n in members if n.endswith(".dist-info/RECORD"))
    rows = [r.split(",") for r in members[record].decode().splitlines() if r]
    return record, rows, members


def _write(wheel: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members:
            archive.writestr(name, data)


def _mutate_record(wheel: Path, how: str) -> None:
    """Rewrite ``wheel`` so that its RECORD and its members disagree in the way ``how`` names."""
    record, rows, members = _rows(wheel)
    module = next(r for r in rows if r[0].endswith(".py"))
    if how == "row dropped":
        rows.remove(module)
    elif how == "row for an absent member":
        rows.insert(0, ["pkg/absent.py", module[1], module[2]])
    elif how == "wrong hash":
        module[1] = "sha256=" + "A" * 43
    elif how == "wrong size":
        module[2] = str(int(module[2]) + 1)
    elif how == "other algorithm":
        module[1] = module[1].replace("sha256=", "sha512=")
    elif how == "row repeated":
        rows.insert(0, list(module))
    elif how == "malformed row":
        rows.insert(0, ["only-a-path"])
    elif how == "record hashes itself":
        rows[-1] = [record, module[1], module[2]]
    lines = "\n".join(",".join(r) for r in rows) + "\n"
    items = [(n, d) for n, d in members.items() if n != record]
    if how == "member not in RECORD":
        items.append(("senselab_worker/extra.py", b"X = 1\n"))
    if how == "member repeated":
        items.append((module[0], members[module[0]]))
    _write(wheel, [*items, (record, lines.encode())])


RECORD_MUTATIONS = [
    "row dropped",
    "row for an absent member",
    "wrong hash",
    "wrong size",
    "other algorithm",
    "row repeated",
    "malformed row",
    "record hashes itself",
    "member not in RECORD",
    "member repeated",
]


@pytest.mark.contract
@pytest.mark.parametrize("how", RECORD_MUTATIONS)
def test_wheel_record_must_list_exactly_its_members_with_their_hashes(how: str, fabricate: Callable[..., Path]) -> None:
    """A RECORD listing a member too few or too many, or with another hash or size, is refused; the true one is not."""
    wheel = fabricate("senselab-worker", "2.0.0", modules={"senselab_worker/__init__.py": "A = 1\n"})
    assert gate.record_problems(wheel) == []
    with pytest.warns(UserWarning) if how == "member repeated" else _no_warning():
        _mutate_record(wheel, how)
    assert gate.record_problems(wheel) != []


class _no_warning:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _repo_declaring(tmp_path: Path, scripts: dict[str, str], extra: str = "") -> Path:
    """A repository whose worker pyproject declares ``scripts`` as ``[project.scripts]``, plus ``extra`` TOML."""
    root = tmp_path / "repo" / "packages" / "senselab-worker"
    root.mkdir(parents=True)
    lines = ["[project]", 'name = "senselab-worker"', "[project.scripts]"]
    lines += [f'{k} = "{v}"' for k, v in scripts.items()]
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n" + extra)
    return tmp_path / "repo"


@pytest.mark.contract
@pytest.mark.parametrize(
    ("shipped", "same"),
    [
        ("[console_scripts]\nrun = senselab_worker.cli:main\n", True),
        ("[console_scripts]\nrun=senselab_worker.cli:main\n", True),
        (None, False),
        ("", False),
        ("[console_scripts]\nrun = senselab_worker.cli:other\n", False),
        ("[console_scripts]\nRun = senselab_worker.cli:main\n", False),
        ("[console_scripts]\nrun = senselab_worker.cli:main\nevil = os:system\n", False),
        ("[console_scripts]\nrun = senselab_worker.cli:main\n[gui_scripts]\nrun = senselab_worker.cli:main\n", False),
        ("[gui_scripts]\nrun = senselab_worker.cli:main\n", False),
    ],
)
def test_entry_points_must_equal_the_pyproject_declaration(
    shipped: str | None, same: bool, fabricate: Callable[..., Path], tmp_path: Path
) -> None:
    """``entry_points.txt`` passes only when its groups and names map to exactly what the pyproject declares."""
    stem = "senselab_worker-2.0.0"
    modules = {"senselab_worker/__init__.py": "\n"}
    if shipped is not None:
        modules[f"{stem}.dist-info/entry_points.txt"] = shipped
    wheel = fabricate("senselab-worker", "2.0.0", modules=modules)
    repo = _repo_declaring(tmp_path, {"run": "senselab_worker.cli:main"})
    if same:
        gate.check_wheel_metadata([wheel], repo)
    else:
        with pytest.raises(gate.GateError, match="entry_points.txt"):
            gate.check_wheel_metadata([wheel], repo)


@pytest.mark.logic
def test_no_declared_entry_points_means_no_entry_points_file(fabricate: Callable[..., Path], tmp_path: Path) -> None:
    """With nothing declared, a wheel without ``entry_points.txt`` passes and one carrying a script is refused."""
    repo = _repo_declaring(tmp_path, {})
    gate.check_wheel_metadata(
        [fabricate("senselab-worker", "2.0.0", modules={"senselab_worker/__init__.py": ""})], repo
    )
    stem = "senselab_worker-2.0.0"
    evil = {f"{stem}.dist-info/entry_points.txt": "[console_scripts]\nx = os:system\n"}
    with pytest.raises(gate.GateError, match="entry_points.txt"):
        gate.check_wheel_metadata([fabricate("senselab-worker", "2.0.0", modules=evil)], repo)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("member", "source"),
    [
        ("senselab_worker-2.0.0.data/scripts/run-worker", "packages/senselab-worker/bin/run.sh"),
        ("senselab_worker-2.0.0.data/scripts/other", None),
        ("senselab_worker-2.0.0.data/purelib/run-worker", None),
        ("senselab_worker-2.0.0.data/data/run-worker", None),
        ("senselab_worker-9.9.9.data/scripts/run-worker", None),
    ],
)
def test_data_scripts_ship_only_as_the_hatch_config_declares(member: str, source: str | None, mini: Path) -> None:
    """A ``.data/scripts`` member is derived from its declared source; no other ``.data`` member is derived at all."""
    pyproject = mini / "packages/senselab-worker/pyproject.toml"
    pyproject.write_text(
        pyproject.read_text() + '[tool.hatch.build.targets.wheel.shared-scripts]\n"bin/run.sh" = "run-worker"\n'
    )
    (mini / "packages/senselab-worker/bin").mkdir()
    (mini / "packages/senselab-worker/bin/run.sh").write_text("#!/bin/sh\n")
    subprocess.run(["git", "add", "-A"], cwd=mini, check=True)
    wheel = Path("senselab_worker-2.0.0-py3-none-any.whl")
    expected = gate.expected_members(wheel, mini, gate.tracked_modes(mini))
    assert expected.get(member) == source and (source is not None) is (member in expected)


@pytest.mark.env
@pytest.mark.network
def test_a_real_build_passes_the_wheel_metadata_check(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """The RECORD and entry points a clean hatch build writes pass the checks."""
    root = tagged_workspace(["2.0.0"])
    built = _core_files(root, tmp_path / "out")
    gate.check_wheel_metadata(built, root)


UV_STEERING = [
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "UV_DEFAULT_INDEX",
    "UV_FIND_LINKS",
    "UV_CONFIG_FILE",
    "UV_INDEX_STRATEGY",
    "UV_PRERELEASE",
    "UV_CONSTRAINT",
    "UV_OVERRIDE",
    "UV_NO_INDEX",
    "UV_SOME_FUTURE_SETTING",
]


@pytest.mark.contract
@settings(max_examples=200, deadline=None)
@given(
    env=st.dictionaries(
        st.one_of(
            st.sampled_from(
                [
                    *UV_STEERING,
                    *sorted(gate.ENV_KEPT),
                    "PIP_INDEX_URL",
                    "VIRTUAL_ENV",
                    "PYTHONPATH",
                    "PYTHONHOME",
                    "PYTHONSTARTUP",
                    "GIT_INDEX_FILE",
                    "GIT_DIR",
                    "GIT_WORK_TREE",
                    "LD_PRELOAD",
                    "DYLD_INSERT_LIBRARIES",
                ]
            ),
            st.text(alphabet=st.characters(categories=["Lu", "Ll", "Nd"], include_characters="_"), min_size=1),
        ),
        st.text(max_size=8),
    )
)
def test_xg6_the_gate_environment_is_an_allowlist(env: dict[str, str]) -> None:
    """Only the variables ``ENV_KEPT`` names pass, unchanged; every other name, known or invented, is dropped."""
    assert gate.gate_environment(env) == {k: v for k, v in env.items() if k in gate.ENV_KEPT}
    assert not {"PYTHONPATH", "PYTHONHOME", "GIT_INDEX_FILE", "GIT_DIR", "VIRTUAL_ENV"} & gate.ENV_KEPT


@pytest.mark.logic
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("uv", "venv", "--python", "3.12", "v"), ("uv", "venv", "--no-config", "--python", "3.12", "v")),
        (("uv", "pip", "install", "x"), ("uv", "pip", "install", "--no-config", "x")),
        (("python", "-c", "pass"), ("python", "-c", "pass")),
    ],
)
def test_every_uv_call_of_the_gate_ignores_configuration(
    argv: tuple[str, ...], expected: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_run`` adds ``--no-config`` to each uv command and runs it in the gate environment."""
    seen: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def run(args: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        seen.append((tuple(args), kwargs["env"]))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setenv("UV_INDEX_URL", "http://127.0.0.1:9/simple")
    monkeypatch.setattr(gate.subprocess, "run", run)
    gate._run(*argv)
    assert seen[0][0] == expected and "UV_INDEX_URL" not in seen[0][1]


@pytest.mark.env
@pytest.mark.network
def test_an_index_serving_a_same_version_first_party_wheel_cannot_win(
    fabricate: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An impostor 2.0.0, tagged for this exact Python so it outranks dist's ``py3`` wheel, is never installed.

    It is offered by every uv setting of the caller: index URLs, find-links, a config file and the index strategy.
    """
    _release(fabricate, "2.0.0", **OK_MODULES)
    impostors = tmp_path / "impostors"
    project = tmp_path / "index" / "simple" / "senselab-worker"
    project.mkdir(parents=True)
    impostors.mkdir()
    for name, version in (("senselab-worker", "2.0.0"), ("senselab-core", "2.0.0")):
        modules = {f"{name.replace('-', '_')}/__init__.py": "IMPOSTOR = 1\n"}
        written = sys.modules["release_fixture"].write_wheel(impostors, name, version, modules=modules)
        wheel = written.rename(written.with_name(written.name.replace("-py3-", f"-py{PY_TAG}-")))
        shutil.copy(wheel, project / wheel.name)
    (project / "index.html").write_text("".join(f'<a href="{p.name}">{p.name}</a>\n' for p in project.glob("*.whl")))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "index"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/simple"
        config = tmp_path / "uv.toml"
        config.write_text(f'index-url = "{url}"\nfind-links = ["{impostors}"]\n')
        for key, value in {
            "UV_INDEX_URL": url,
            "UV_DEFAULT_INDEX": url,
            "UV_EXTRA_INDEX_URL": url,
            "UV_INDEX": url,
            "UV_FIND_LINKS": str(impostors),
            "UV_CONFIG_FILE": str(config),
            "UV_INDEX_STRATEGY": "unsafe-best-match",
        }.items():
            monkeypatch.setenv(key, value)
        gate.check_installs(gate.dist_files(tmp_path / "dist"), HERE_PYTHON)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class _SimpleIndex:
    """A PEP 503 index on localhost serving the given wheels; ``url`` is its ``/simple`` root."""

    def __init__(self, root: Path, wheels: list[Path]) -> None:
        for wheel in wheels:
            project = root / "simple" / wheel.name.split("-")[0].replace("_", "-")
            project.mkdir(parents=True, exist_ok=True)
            shutil.copy(wheel, project / wheel.name)
        for project in (root / "simple").iterdir() if (root / "simple").exists() else []:
            links = "".join(f'<a href="{p.name}">{p.name}</a>\n' for p in project.glob("*.whl"))
            (project / "index.html").write_text(links)
        (root / "simple").mkdir(parents=True, exist_ok=True)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/simple"

    def __enter__(self) -> _SimpleIndex:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


@pytest.mark.env
@pytest.mark.network
def test_third_party_requirements_come_from_the_index_argument_and_never_from_the_environment(
    fabricate: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``index_url`` is where third-party requirements resolve; ``UV_INDEX_URL`` and friends cannot substitute for it.

    The worker requires ``sl-gate-probe``, which only one local index serves. Named by ``index_url`` it installs;
    named only by the caller's environment, with ``index_url`` an index that lacks it, the install fails.
    """
    _release(fabricate, "2.0.0", **OK_MODULES)
    dist = tmp_path / "dist"
    worker = next(dist.glob("senselab_worker-*.whl"))
    worker.unlink()
    write = sys.modules["release_fixture"].write_wheel
    write(dist, "senselab-worker", "2.0.0", ["sl-gate-probe==1.0"], OK_MODULES["worker"]["modules"])
    deps = tmp_path / "deps"
    deps.mkdir()
    probe = write(deps, "sl-gate-probe", "1.0", modules={"sl_gate_probe/__init__.py": "\n"})
    files = gate.dist_files(dist)
    with _SimpleIndex(tmp_path / "serving", [probe]) as serving, _SimpleIndex(tmp_path / "empty", []) as empty:
        gate.check_installs(files, HERE_PYTHON, index_url=serving.url)
        for key in ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL", "UV_INDEX", "PIP_INDEX_URL"):
            monkeypatch.setenv(key, serving.url)
        monkeypatch.setenv("UV_FIND_LINKS", str(deps))
        with pytest.raises(gate.GateError, match="sl-gate-probe"):
            gate.check_installs(files, HERE_PYTHON, index_url=empty.url)


@pytest.mark.logic
def test_main_passes_the_index_argument_to_the_installs_and_the_sdist_builds(
    fabricate: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` hands ``index_url`` to ``check_installs`` and ``check_sdist_builds``; it defaults to PyPI."""
    wheels = _release(fabricate, "2.0.0", **OK_MODULES)
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(gate, "static_checks", lambda *a, **k: None)
    monkeypatch.setattr(gate, "check_installs", lambda files, pythons, *, index_url: seen.append(("i", index_url)))
    monkeypatch.setattr(gate, "check_sdist_builds", lambda files, *, index_url: seen.append(("s", index_url)))
    assert gate.main(wheels[0].parent, repo=REPO, pythons=HERE_PYTHON) == 0
    assert gate.main(wheels[0].parent, repo=REPO, pythons=HERE_PYTHON, index_url="http://mirror.invalid/simple") == 0
    mirror = "http://mirror.invalid/simple"
    assert seen == [("i", gate.PYPI), ("s", gate.PYPI), ("i", mirror), ("s", mirror)]


@pytest.mark.logic
def test_main_cannot_skip_the_member_check() -> None:
    """``repo`` is a required argument of ``main``: no call checks a dist without comparing it to its sources."""
    import inspect

    parameter = inspect.signature(gate.main).parameters["repo"]
    assert parameter.default is inspect.Parameter.empty and parameter.kind is inspect.Parameter.KEYWORD_ONLY


# --- H9f: the gate checks completeness (R-107, XG-4..XG-9) ----------------------------------------


def _mini_dist(mini: Path, version: str = "2.0.0") -> list[Path]:
    return sys.modules["release_fixture"].mini_dist(mini, mini.parent / "dist", version)


def _static(files: list[Path], mini: Path, *, tag: str | None = "2.0.0") -> None:
    gate.static_checks(files, mini, FLOOR, tag=tag, prerelease=None if tag is None else Version(tag).is_prerelease)


def _rewrite(
    mini: Path, name: str, kind: str, edit: Callable[[dict[str, bytes]], None], version: str = "2.0.0"
) -> Path:
    """Rewrite one mini artifact in place with ``edit`` applied to its members; RECORD follows the edit."""
    fixture = sys.modules["release_fixture"]
    writer = fixture.write_mini_wheel if kind == "wheel" else fixture.write_mini_sdist
    return writer(mini, mini.parent / "dist", name, version, edit)


@pytest.mark.logic
def test_xg_a_hatch_shaped_dist_of_the_mini_repo_passes_every_static_check(mini: Path) -> None:
    """The positive control: the six artifacts a hatch build writes pass names, versions, metadata and members."""
    _static(_mini_dist(mini), mini)


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["1!1.0.0", "1!2.0.0"])
def test_xg4_an_epoch_version_is_refused_whatever_the_tag(tag: str, fabricate: Callable[..., Path]) -> None:
    """A ``1!1.0.0`` build outranks every 2.x release, so its version and its tag are both refused."""
    wheels = _release(fabricate, tag)
    with pytest.raises(gate.GateError, match="epoch"):
        gate.check_versions(wheels, FLOOR, tag=tag, prerelease=False)
    with pytest.raises(gate.GateError, match="epoch"):
        gate.check_versions(_release(fabricate, "2.0.0"), FLOOR, tag=tag, prerelease=False)


_SEGMENTS = st.tuples(
    st.sampled_from(["", "1!", "0!"]),
    st.lists(st.integers(0, 12), min_size=1, max_size=4).map(lambda p: ".".join(map(str, p))),
    st.sampled_from(["", "a1", "b2", "rc3", "alpha1", "c1", "-rc.1", "pre2", "a"]),
    st.sampled_from(["", ".post1", "-1", ".dev0", ".dev3"]),
    st.sampled_from(["", "+g1234", "+local.1"]),
)


@pytest.mark.contract
@settings(max_examples=300, deadline=None)
@given(parts=_SEGMENTS)
def test_xg4_only_the_grammar_is_a_releasable_version(parts: tuple[str, ...]) -> None:
    """A version string is releasable exactly when it is ``X.Y.Z`` or ``X.Y.Z(a|b|rc)N`` written canonically."""
    text = "".join(parts)
    releasable = re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", text) is not None and str(Version(text)) == text
    assert (gate.version_problem(text) is None) is releasable


_TAG_SEGMENTS = st.tuples(
    st.sampled_from(["", "v", "V"]),
    st.sampled_from(["", "1!", "0!"]),
    st.lists(st.integers(0, 12), min_size=1, max_size=4).map(lambda p: ".".join(map(str, p))),
    st.sampled_from(
        ["", "a1", "b2", "rc3", "-alpha.1", "-beta.2", "-rc.3", "alpha1", "-rc1", "rc", "-a.1", "c1", "-alpha", ".rc.1"]
    ),
    st.sampled_from(["", ".post1", "-1", ".dev0", "-dev.3"]),
    st.sampled_from(["", "+g1234", "+local.1"]),
)


@pytest.mark.contract
@settings(max_examples=400, deadline=None)
@given(parts=_TAG_SEGMENTS)
def test_xg4_a_tag_is_canonical_or_the_repository_pre_release_spelling(parts: tuple[str, ...]) -> None:
    """A tag is accepted exactly when it is a releasable version string or ``X.Y.Z-(alpha|beta|rc).N``."""
    text = "".join(parts)
    canonical = re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", text) is not None and str(Version(text)) == text
    repository = re.fullmatch(r"\d+\.\d+\.\d+-(alpha|beta|rc)\.\d+", text) is not None
    assert (gate.tag_problem(text) is None) is (canonical or repository)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("tag", "version", "ok"),
    [
        ("2.0.0-alpha.3", "2.0.0a3", True),
        ("2.0.0-beta.1", "2.0.0b1", True),
        ("2.0.0-rc.2", "2.0.0rc2", True),
        ("2.0.0a3", "2.0.0a3", True),
        ("2.0.0-alpha.3", "2.0.0a4", False),
        ("2.0.0-alpha.3", "2.0.0", False),
        ("v2.0.0-alpha.3", "2.0.0a3", False),
        ("1!2.0.0-alpha.3", "2.0.0a3", False),
        ("2.0.0-alpha.3+g1", "2.0.0a3", False),
        ("2.0.0-alpha.3.post1", "2.0.0a3", False),
        ("2.0.0-alpha.3.dev0", "2.0.0a3", False),
    ],
)
def test_xg4_the_repository_tag_spelling_binds_to_the_built_version(
    tag: str, version: str, ok: bool, fabricate: Callable[..., Path]
) -> None:
    """``X.Y.Z-(alpha|beta|rc).N`` passes when it parses to exactly the built version; any other segment is refused."""
    wheels = _release(fabricate, version)
    if ok:
        gate.check_versions(wheels, FLOOR, tag=tag, prerelease=True)
    else:
        with pytest.raises(gate.GateError):
            gate.check_versions(wheels, FLOOR, tag=tag, prerelease=True)


@pytest.mark.logic
def test_root_imports_its_packages_and_subpackages_and_core_imports_every_module(mini: Path) -> None:
    """The root distribution imports each package to depth two; core and the worker import every module."""
    fixture = sys.modules["release_fixture"]
    deep = {"senselab/audio/__init__.py": b"", "senselab/audio/tasks/__init__.py": b"", "senselab/audio/x.py": b""}
    root = fixture.write_mini_wheel(mini, mini.parent, "senselab", "2.0.0", lambda m: m.update(deep))
    assert gate.import_targets(root) == ["senselab", "senselab.audio"]
    core = fixture.write_mini_wheel(mini, mini.parent, "senselab-core", "2.0.0")
    assert (
        gate.import_targets(core)
        == gate.wheel_modules(core)
        == [
            "senselab_core",
            "senselab_core.kernel",
            "senselab_core.kernel.identity",
        ]
    )


@pytest.mark.logic
def test_no_install_requests_an_extra(fabricate: Callable[..., Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra-only requirements are not installed and the own wheel is installed by plain path, with no extra."""
    seen: list[tuple[str, ...]] = []

    def run(args: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        seen.append(tuple(args))
        out = json.dumps({"sys_platform": "linux", "python_version": "3.11"}) if "-c" in args else ""
        return subprocess.CompletedProcess(args, 0, out, "")

    wheels = _release(fabricate, "2.0.0")
    dist = wheels[0].parent
    (dist / wheels[2].name).unlink()
    sys.modules["release_fixture"].write_wheel(
        dist, "senselab", "2.0.0", ["senselab-core==2.0.0", "numpy>=1", "jiwer>=3.0; extra == 'nlp'"]
    )
    monkeypatch.setattr(gate.subprocess, "run", run)
    gate.check_installs(gate.dist_files(dist), HERE_PYTHON)
    flat = [a for args in seen if args[0] == "uv" for a in args]
    assert "numpy>=1" in flat and not [a for a in flat if "jiwer" in a or "[" in a]


def _drop(member: str) -> Callable[[dict[str, bytes]], None]:
    def edit(members: dict[str, bytes]) -> None:
        del members[member]

    return edit


@pytest.mark.contract
def test_xg5_every_derived_member_is_required(mini: Path) -> None:
    """Dropping any one member of any artifact, its RECORD rewritten to match, is refused."""
    fixture = sys.modules["release_fixture"]
    files = _mini_dist(mini)
    dropped = 0
    for path in files:
        name = gate._metadata(path)[0]
        kind = "wheel" if path.suffix == ".whl" else "sdist"
        for member in [m for m, _, _ in gate.artifact_members(path) if not m.endswith("/RECORD")]:
            _rewrite(mini, name, kind, _drop(member))
            with pytest.raises(gate.GateError):
                _static(files, mini)
            dropped += 1
        (fixture.write_mini_wheel if kind == "wheel" else fixture.write_mini_sdist)(mini, path.parent, name, "2.0.0")
    _static(files, mini)
    assert dropped > 20


@pytest.mark.logic
@pytest.mark.parametrize(
    ("name", "member"),
    [
        ("senselab-core", "senselab_core/kernel/identity.py"),
        ("senselab-core", "senselab_core/__init__.py"),
        ("senselab-worker", "senselab_worker/canonical.py"),
        ("senselab", "senselab/data/table.yaml"),
    ],
)
def test_xg5_a_wheel_missing_a_tracked_package_file_is_refused(name: str, member: str, mini: Path) -> None:
    """A tracked file the build silently dropped (ignored, force-added, excluded) refuses the release by name."""
    files = _mini_dist(mini)
    _rewrite(mini, name, "wheel", _drop(member))
    with pytest.raises(gate.GateError, match=re.escape(f"{member} (missing)")):
        _static(files, mini)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("name", "kind", "member", "content"),
    [
        ("senselab-core", "wheel", "senselab_core/evil.py", b"import os\n"),
        ("senselab-core", "wheel", "senselab_core/pyproject.toml", None),
        ("senselab-worker", "wheel", "senselab_worker-2.0.0.dist-info/licenses/LICENSE.md", None),
        ("senselab", "wheel", "tests/a_test.py", None),
        ("senselab", "sdist", "senselab-2.0.0/hatch_build_pins.py", None),
        ("senselab-worker", "sdist", "senselab_worker-2.0.0/extra.txt", b"x"),
    ],
)
def test_xg5_a_member_the_config_does_not_select_is_refused(
    name: str, kind: str, member: str, content: bytes | None, mini: Path
) -> None:
    """Extra members are refused even when their bytes are a tracked file's."""
    files = _mini_dist(mini)
    data = content if content is not None else (mini / "packages/senselab-core/pyproject.toml").read_bytes()
    _rewrite(mini, name, kind, lambda m: m.__setitem__(member, data))
    with pytest.raises(gate.GateError, match="not a tracked file the config selects"):
        _static(files, mini)


@pytest.mark.logic
def test_xg5_an_unknown_hatch_setting_refuses_rather_than_ships_underived_members(mini: Path) -> None:
    """A build setting the derivation does not model is a refusal naming it."""
    files = _mini_dist(mini)
    pyproject = mini / "packages/senselab-core/pyproject.toml"
    pyproject.write_text(pyproject.read_text() + '[tool.hatch.build.targets.wheel.sources]\n"src" = ""\n')
    with pytest.raises(gate.GateError, match="wheel.sources"):
        _static(files, mini)


@pytest.mark.logic
def test_xg5_wheel_modules_are_the_py_files_of_packages(mini: Path) -> None:
    """Every ``.py`` under ``__init__``-bearing directories is a module; one in a data directory is not."""
    wheel = sys.modules["release_fixture"].write_mini_wheel(
        mini,
        mini.parent,
        "senselab",
        "2.0.0",
        lambda m: m.update(
            {"senselab/data/guard/sitecustomize.py": b"", "senselab/__main__.py": b"", "senselab/x.pyi": b""}
        ),
    )
    assert gate.wheel_modules(wheel) == ["senselab", "senselab.__main__", "senselab.cli"]


@pytest.mark.logic
def test_xg6_the_tracked_set_ignores_the_callers_git_environment(mini: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An alternate index exported as ``GIT_INDEX_FILE`` cannot make an untracked file count as tracked."""
    (mini / "evil.py").write_text("x\n")
    alternate = mini.parent / "alt-index"
    env = {**os.environ, "GIT_INDEX_FILE": str(alternate)}
    subprocess.run(["git", "read-tree", "HEAD"], cwd=mini, env=env, capture_output=True)
    subprocess.run(["git", "add", "-A", "--force"], cwd=mini, env=env, check=True)
    assert "evil.py" in subprocess.run(["git", "ls-files"], cwd=mini, env=env, capture_output=True, text=True).stdout
    for key, value in {"GIT_INDEX_FILE": str(alternate), "GIT_DIR": str(mini / ".git"), "GIT_WORK_TREE": "/"}.items():
        monkeypatch.setenv(key, value)
    assert "evil.py" not in gate.tracked_modes(mini)


@pytest.mark.logic
def test_xg6_every_subprocess_runs_isolated_in_the_allowlisted_environment(
    fabricate: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each command ``check_installs`` runs sees only ``ENV_KEPT`` variables; each interpreter runs with ``-I``."""
    for key in ("PYTHONPATH", "PYTHONHOME", "GIT_INDEX_FILE", "VIRTUAL_ENV", "UV_INDEX_URL", "SOME_RANDOM_VAR"):
        monkeypatch.setenv(key, "/planted")
    seen: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def run(args: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        seen.append((tuple(args), kwargs["env"]))
        out = json.dumps({"sys_platform": sys.platform, "python_version": "3.12"}) if "-c" in args else ""
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(gate.subprocess, "run", run)
    gate.check_installs(gate.dist_files(_release(fabricate, "2.0.0", **OK_MODULES)[0].parent), HERE_PYTHON)
    assert seen and all(set(env) <= gate.ENV_KEPT for _, env in seen)
    interpreters = [args for args, _ in seen if args[0] != "uv"]
    assert interpreters and all(args[1] == "-I" for args in interpreters)


_CORE_INFO = "senselab_core-2.0.0/PKG-INFO"
_WORKER_INFO = "senselab_worker-2.0.0/PKG-INFO"


def _append(member: str, text: bytes) -> Callable[[dict[str, bytes]], None]:
    return lambda m: m.update({member: m[member] + text})


def _copy(member: str, source: str) -> Callable[[dict[str, bytes]], None]:
    return lambda m: m.update({member: m[source]})


@pytest.mark.logic
@pytest.mark.parametrize(
    ("name", "edit", "match"),
    [
        ("senselab-core", lambda m: _sub(_CORE_INFO, b"worker==2.0.0", b"worker>=0")(m), "senselab-core requires"),
        ("senselab-core", lambda m: _append(_CORE_INFO, b"Requires-Dist: evilpkg\n")(m), "evilpkg"),
        ("senselab-worker", lambda m: _sub(_WORKER_INFO, b">=3.10", b">=3.6")(m), "Requires-Python"),
        (
            "senselab-core",
            lambda m: _copy("othertop/pyproject.toml", "senselab_core-2.0.0/pyproject.toml")(m),
            "outside",
        ),
    ],
)
def test_xg7_sdist_metadata_and_members_are_checked(
    name: str, edit: Callable[[dict[str, bytes]], None], match: str, mini: Path
) -> None:
    """An sdist's pins, extra requirements, Python range and top directory are held to the same rules as a wheel's."""
    files = _mini_dist(mini)
    _rewrite(mini, name, "sdist", edit)
    with pytest.raises(gate.GateError, match=match):
        _static(files, mini)


@pytest.mark.logic
@pytest.mark.parametrize(
    "rename",
    [
        {"senselab_core-2.0.0.tar.gz": "anything-else-7.tar.gz"},
        {
            "senselab_core-2.0.0.tar.gz": "senselab_worker-2.0.0.tar.gz",
            "senselab_worker-2.0.0.tar.gz": "senselab_core-2.0.0.tar.gz",
        },
        {"senselab_core-2.0.0.tar.gz": "Senselab_Core-2.0.0.tar.gz"},
        {"senselab_core-2.0.0.tar.gz": "senselab-core-2.0.0.tar.gz"},
        {"senselab_core-2.0.0-py3-none-any.whl": "senselab_core-2.0.0-py2.py3-none-any.whl"},
        {"senselab_core-2.0.0-py3-none-any.whl": "senselab_core-2.0.0-1-py3-none-any.whl"},
        {"senselab_core-2.0.0-py3-none-any.whl": "senselab_core-2.0.0-cp39-cp39-win32.whl"},
    ],
)  # fmt: skip
def test_xg7_xg8_file_names_are_bound_to_the_metadata(rename: dict[str, str], mini: Path) -> None:
    """A renamed, swapped or retagged file is refused: its name must be exactly the one its metadata gives."""
    files = _mini_dist(mini)
    dist = files[0].parent
    staged = {old: (dist / old).read_bytes() for old in rename}
    for old in rename:
        (dist / old).unlink()
    for old, new in rename.items():
        (dist / new).write_bytes(staged[old])
    with pytest.raises(gate.GateError):
        _static(gate.dist_files(dist), mini)


_WHEEL = "senselab_core-2.0.0.dist-info/WHEEL"
_META = "senselab_core-2.0.0.dist-info/METADATA"


def _sub(member: str, old: bytes, new: bytes) -> Callable[[dict[str, bytes]], None]:
    return lambda m: m.update({member: m[member].replace(old, new)})


@pytest.mark.logic
@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_sub(_WHEEL, b"Root-Is-Purelib: true", b"Root-Is-Purelib: false"), "Root-Is-Purelib"),
        (_sub(_WHEEL, b"Tag: py3-none-any", b"Tag: cp39-cp39-win32"), "Tag"),
        (_sub(_WHEEL, b"Tag: py3-none-any", b"Tag: py3-none-any\nTag: py2-none-any"), "Tag"),
        (_sub(_WHEEL, b"Wheel-Version: 1.0", b"Wheel-Version: 1.0\nWheel-Version: 2.0"), "Wheel-Version"),
        (_sub(_WHEEL, b"Tag:", b"Build: 1\nTag:"), "Build"),
        (_sub(_META, b"Version: 2.0.0", b"Version: 2.0.0\nVersion: 9.9.9"), "repeated"),
        (_sub(_META, b"Name: senselab-core", b"Name: senselab-core\nName: evil"), "repeated"),
        (_sub(_META, b">=3.11", b">=3.6"), "Requires-Python"),
        (_sub(_META, b"Requires-Dist: pyyaml", b"Requires-Dist: pyyaml\nRequires-Dist: evilpkg"), "evilpkg"),
        (_sub(_META, b"Requires-Dist: pyyaml", b"Requires-Dist: pyyaml\nRequires-Dist: pyyaml"), "2 times"),
        (_sub(_META, b"Requires-Dist: pyyaml", b"Requires-Dist: pyyaml>=0"), "not declared"),
        (_sub(_META, b"Requires-Dist: pyyaml", b"Provides-Extra: x"), "Provides-Extra|not in Requires-Dist"),
        (_sub(_META, b"Requires-Python", b"License-File: pyproject.toml\nRequires-Python"), "License-File"),
    ],
)
def test_xg8_wheel_and_metadata_files_are_parsed_strictly(
    edit: Callable[[dict[str, bytes]], None], match: str, mini: Path
) -> None:
    """``WHEEL`` must state the file name's one tag and a pure-Python root; METADATA exactly the pyproject."""
    files = _mini_dist(mini)
    _rewrite(mini, "senselab-core", "wheel", edit)
    with pytest.raises(gate.GateError, match=match):
        _static(files, mini)


@pytest.mark.logic
def test_xg8_root_extras_are_expanded_as_hatch_expands_them(mini: Path) -> None:
    """``all = ["senselab[nlp]"]`` contributes nlp's requirements under ``extra == "all"``; dropping one is refused."""
    files = _mini_dist(mini)
    meta = "senselab-2.0.0.dist-info/METADATA"
    _rewrite(mini, "senselab", "wheel", _sub(meta, b"Requires-Dist: jiwer>=3.0; extra == 'all'\n", b""))
    with pytest.raises(gate.GateError, match="not in Requires-Dist"):
        _static(files, mini)


@pytest.mark.logic
@pytest.mark.parametrize(
    "member",
    [
        "../pyproject.toml",
        "/pyproject.toml",
        "senselab-2.0.0/../README.md",
        "senselab-2.0.0/./pyproject.toml",
        "senselab-2.0.0//pyproject.toml",
        "senselab-2.0.0/SRC/senselab/__init__.py",
        "othertop/pyproject.toml",
        "senselab-2.0.0\\pyproject.toml",
    ],
)
def test_xg9_tar_members_are_normalised_relative_names_under_the_top(member: str, mini: Path) -> None:
    """``..``, absolute, dot, empty, backslash, outside-top and case-duplicate names are refused, whatever the bytes."""
    files = _mini_dist(mini)
    source = "senselab-2.0.0/src/senselab/__init__.py" if "SRC" in member else "senselab-2.0.0/pyproject.toml"
    _rewrite(mini, "senselab", "sdist", lambda m: m.__setitem__(member, m[source]))
    with pytest.raises(gate.GateError):
        _static(files, mini)


@pytest.mark.logic
@pytest.mark.parametrize("member", ["../x.py", "/x.py", "senselab/../x.py", "SENSELAB/__init__.py"])
def test_xg9_zip_members_are_normalised_too(member: str, mini: Path) -> None:
    """The same name rules hold for wheel members."""
    files = _mini_dist(mini)
    _rewrite(mini, "senselab", "wheel", lambda m: m.__setitem__(member, m["senselab/__init__.py"]))
    with pytest.raises(gate.GateError):
        _static(files, mini)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("line", "extras", "selected"),
    [
        ("numpy>=1.23", [], ["numpy>=1.23"]),
        ("jiwer>=3.0; extra == 'nlp'", [], []),
        ("jiwer>=3.0; extra == 'nlp'", ["nlp"], ["jiwer>=3.0"]),
        ("jiwer>=3.0; extra == 'nlp'", ["all"], []),
        ("torch>=2.8; sys_platform == 'linux'", [], ["torch>=2.8"]),
        ("torch>=2.8; sys_platform == 'win32'", ["all"], []),
        ("av[x]>=15; sys_platform == 'linux' and extra == 'video'", ["video"], ["av[x]>=15"]),
    ],
)
def test_requirements_are_selected_for_the_target_interpreter_and_extras(
    line: str, extras: list[str], selected: list[str]
) -> None:
    """Markers are evaluated against the venv's environment with each requested extra; the marker is then dropped."""
    from packaging.requirements import Requirement

    environment = {"sys_platform": "linux", "python_version": "3.11", "platform_machine": "x86_64"}
    assert gate.selected_requirements([Requirement(line)], environment, extras) == selected


@pytest.mark.env
@pytest.mark.network
def test_xg5_a_module_that_does_not_import_is_refused(fabricate: Callable[..., Path]) -> None:
    """A module the package's ``__init__`` never imports is imported by the gate, and its failure refuses."""
    modules = {
        **OK_MODULES,
        "worker": {"modules": {"senselab_worker/__init__.py": "\n", "senselab_worker/x.py": "import numpy\n"}},
    }
    wheels = _release(fabricate, "2.0.0", **modules)
    with pytest.raises(gate.GateError, match="numpy"):
        gate.check_installs(gate.dist_files(wheels[0].parent), HERE_PYTHON)


@pytest.mark.env
@pytest.mark.network
def test_xg6_pythonpath_cannot_supply_a_module_the_wheel_lacks(
    fabricate: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker wheel without a module its ``__init__`` imports is refused even with that module on ``PYTHONPATH``."""
    source = tmp_path / "source"
    (source / "senselab_worker").mkdir(parents=True)
    (source / "senselab_worker" / "__init__.py").write_text("\n")
    (source / "senselab_worker" / "canonical.py").write_text("X = 1\n")
    modules = {
        **OK_MODULES,
        "worker": {"modules": {"senselab_worker/__init__.py": "from senselab_worker import canonical\n"}},
    }
    wheels = _release(fabricate, "2.0.0", **modules)
    monkeypatch.setenv("PYTHONPATH", str(source))
    with pytest.raises(gate.GateError, match="canonical"):
        gate.check_installs(gate.dist_files(wheels[0].parent), HERE_PYTHON)


@pytest.mark.env
@pytest.mark.network
def test_xg7_each_sdist_builds_the_wheel_in_dist(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """A real core sdist rebuilds its wheel byte for byte; an sdist whose source differs from its wheel is refused."""
    root = tagged_workspace(["2.0.0"])
    files = _core_files(root, tmp_path / "out")
    worker = [p for p in build(root, tmp_path / "w", "--package", "senselab-worker") if not p.name.startswith(".")]
    gate.check_sdist_builds([*files, *worker])
    sdist = next(p for p in files if p.name.endswith(".tar.gz"))
    _rewrite_sdist(sdist, {"src/senselab_core/__init__.py": b"CHANGED = 1\n"})
    with pytest.raises(gate.GateError, match="differs"):
        gate.check_sdist_builds([*files, *worker])


@pytest.mark.env
@pytest.mark.network
def test_xg5_a_real_build_of_a_tagged_checkout_passes_every_static_check(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """``uv build --all-packages`` on a tagged copy of this tree yields exactly the derived members and metadata."""
    root = tagged_workspace([RELEASE_FLOOR])
    build(root, root / "dist", "--all-packages")
    gate.static_checks(gate.dist_files(root / "dist"), root, FLOOR, tag=RELEASE_FLOOR, prerelease=True)
