"""The release gate refuses what cannot be released, and the build writes only what can be installed."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tomllib
import types
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
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
    assert gate.main(dist) == 1
    assert reason in capsys.readouterr().err


HERE_PYTHON = {name: f"{sys.version_info.major}.{sys.version_info.minor}" for name in gate.DISTRIBUTIONS}
OK_MODULES: dict[str, dict[str, dict[str, str]]] = {
    "worker": {"modules": {"senselab_worker/__init__.py": "\n"}},
    "core": {"modules": {"senselab_core/__init__.py": "\n"}},
    "root": {"modules": {"senselab/__init__.py": "\n"}},
}


@pytest.mark.env
def test_check_installs_installs_each_distribution_alone_and_imports_it(fabricate: Callable[..., Path]) -> None:
    """A dist whose three wheels import installs, one distribution per venv, from ``dist/`` paths."""
    wheels = _release(fabricate, "2.0.0", **OK_MODULES)
    gate.check_installs(gate.dist_files(wheels[0].parent), HERE_PYTHON)


@pytest.mark.env
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
    """The built version must equal the normalised tag, and the release's prerelease flag must match it."""
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


@pytest.mark.env
def test_main_passes_a_complete_fabricated_dist(
    fabricate: Callable[..., Path], fabricate_sdist: Callable[..., Path]
) -> None:
    """Wheels and sdists of one releasable version, pinned and importable, pass end to end."""
    wheels = _release(fabricate, "2.0.0", **OK_MODULES)
    for name in gate.DISTRIBUTIONS:
        fabricate_sdist(name, "2.0.0")
    assert gate.main(wheels[0].parent, tag="2.0.0", prerelease=False, pythons=HERE_PYTHON) == 0


def _core_files(root: Path, out: Path) -> list[Path]:
    """Build core in ``root`` and return its wheel and sdist."""
    return [p for p in build(root, out, "--package", "senselab-core") if not p.name.startswith(".")]


@pytest.mark.env
def test_n3_clean_build_contents_are_all_tracked(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """Every file in a clean build's wheel and sdist maps to a tracked file or a named generated one."""
    root = tagged_workspace(["2.0.0"])
    gate.check_contents(_core_files(root, tmp_path / "out"), root)


@pytest.mark.env
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
        gate.check_contents(files, root)


@pytest.mark.env
def test_n3_gitignored_file_does_not_ship_and_is_not_refused(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """A file ignored by ``.gitignore`` is dropped by the build, so the contents check has nothing to refuse."""
    root = tagged_workspace(["2.0.0"])
    (root / "packages/senselab-core/src/senselab_core/x.log").write_text("x")
    (root / ".gitignore").write_text("*.log\n")
    gate.check_contents(_core_files(root, tmp_path / "out"), root)


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
def test_m1_m2_m4_mutating_a_real_build_is_refused(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """Aliases are bound to content, extra dist-info or data dirs and non-regular entries are refused."""
    root = tagged_workspace(["2.0.0"])
    built = _core_files(root, tmp_path / "out")
    gate.check_contents(built, root)
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
        with pytest.raises(gate.GateError, match="not copies of tracked files"):
            gate.check_contents([target], root)
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
        with pytest.raises(gate.GateError, match="not copies of tracked files"):
            gate.check_contents([target], root)


@pytest.mark.env
def test_m1_an_untracked_gitignore_in_a_package_root_that_ships_is_refused(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """The sdist ``.gitignore`` must equal a tracked one: an untracked file with other content ships and fails."""
    root = tagged_workspace(["2.0.0"])
    (root / "packages/senselab-core/.gitignore").write_text("!*\n")
    files = _core_files(root, tmp_path / "out")
    with pytest.raises(gate.GateError, match="gitignore"):
        gate.check_contents(files, root)


@pytest.mark.env
def test_m1_the_root_distribution_license_must_be_a_named_tracked_copy(
    tagged_workspace: Callable[..., Path], tmp_path: Path
) -> None:
    """The root wheel's license file passes when it is the tracked LICENSE.md and fails when its bytes differ."""
    root = tagged_workspace(["2.0.0"])
    files = [p for p in build(root, tmp_path / "out", "--package", "senselab") if not p.name.startswith(".")]
    gate.check_contents(files, root)
    wheel = next(p for p in files if p.suffix == ".whl")
    names = zipfile.ZipFile(wheel).namelist()
    license_member = next(n for n in names if "/licenses/" in n)
    _rewrite_wheel(wheel, {license_member: b"forged"})
    with pytest.raises(gate.GateError, match="licenses"):
        gate.check_contents([wheel], root)


@pytest.mark.env
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
        gate.check_contents(wheels, tmp_path)


@pytest.mark.env
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


SAFE_EXPRESSIONS = {"github.event.pull_request.number", "matrix.python-version"}


def _strings(node: Any) -> list[str]:  # noqa: ANN401
    """Return every ``run`` string and github-script ``script`` string among the steps under ``node``."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("run", "script") and isinstance(value, str):
                found.append(value)
            else:
                found += _strings(value)
    elif isinstance(node, list):
        for item in node:
            found += _strings(item)
    return found


def interpolation_offenders(directory: Path) -> list[str]:
    """Parse every workflow and composite action under ``directory``; list each ``${{ }}`` in a script not allowed.

    A script is inspected as one whole string, so an expression split over lines, or inside a literal or folded
    block scalar, is found. An opener with no closer is an offender too.
    """
    offenders = []
    for path in sorted([*directory.rglob("*.yml"), *directory.rglob("*.yaml")]):
        for script in _strings(yaml.safe_load(path.read_text())):
            found = re.findall(r"\$\{\{(.*?)\}\}", script, flags=re.S)
            offenders += [
                f"{path.name}: {' '.join(e.split())}" for e in found if " ".join(e.split()) not in SAFE_EXPRESSIONS
            ]
            if script.count("${{") != len(found):
                offenders.append(f"{path.name}: unterminated ${{{{")
    return offenders


@pytest.mark.logic
def test_n5_no_workflow_interpolates_an_expression_into_a_run_script() -> None:
    """Untrusted contexts reach shell only through ``env:``; scripts hold no ``${{ }}`` but numbers and the matrix."""
    assert interpolation_offenders(REPO / ".github") == []


@pytest.mark.logic
@pytest.mark.parametrize(
    ("script", "bad"),
    [
        ("run: echo ${{ github.head_ref }}", True),
        ("run: |\n        echo ${{\n          github.head_ref }}", True),
        ("run: |\n        echo hi\n        echo ${{ github.event.pull_request.title }}", True),
        ("run: >\n        echo\n        ${{ github.head_ref }}", True),
        ("run: >\n        echo ${{\n        github.head_ref\n        }}", True),
        ("run: echo ${{ github.head_ref", True),
        ("run: echo ${{ matrix.python-version }}", False),
        ("run: |\n        echo ${{\n          github.event.pull_request.number }}", False),
        ('run: echo "$HEAD"', False),
    ],
)
def test_m3_interpolation_guard_inspects_whole_scripts(script: str, bad: bool, tmp_path: Path) -> None:
    """The guard parses YAML and scans each script as a whole: multi-line, literal and folded forms are all caught."""
    (tmp_path / "w.yaml").write_text(f"jobs:\n  j:\n    steps:\n    - {script}\n")
    assert bool(interpolation_offenders(tmp_path)) is bad


@pytest.mark.logic
def test_m3_interpolation_guard_covers_composite_actions_and_github_script(tmp_path: Path) -> None:
    """A composite action's ``run`` and a github-script ``with.script`` are scripts too."""
    (tmp_path / "action.yml").write_text("runs:\n  steps:\n  - run: echo ${{ github.head_ref }}\n")
    (tmp_path / "w.yaml").write_text(
        "jobs:\n  j:\n    steps:\n    - uses: actions/github-script@v7\n"
        "      with:\n        script: x(${{ github.head_ref }})\n"
    )
    assert len(interpolation_offenders(tmp_path)) == 2


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


@pytest.mark.network
@pytest.mark.env
def test_release_gate_installs_dist(tagged_workspace: Callable[..., Path], tmp_path: Path) -> None:
    """Build every package from a tagged checkout; the gate installs each distribution at its floor Python."""
    root = tagged_workspace([RELEASE_FLOOR])
    build(root, root / "dist", "--all-packages")
    assert gate.main(root / "dist", tag=RELEASE_FLOOR, prerelease=True, repo=root) == 0
