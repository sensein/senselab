"""The release gate refuses what cannot be released, and the build writes only what can be installed."""

from __future__ import annotations

import importlib.util
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
    gate.check_installs(wheels[0].parent, HERE_PYTHON)


@pytest.mark.env
def test_a11_check_installs_refuses_a_wheel_that_does_not_import(fabricate: Callable[..., Path]) -> None:
    """An installable dist whose root package raises on import is refused."""
    wheels = _release(
        fabricate,
        "2.0.0",
        **{**OK_MODULES, "root": {"modules": {"senselab/__init__.py": "raise ImportError('broken')\n"}}},
    )
    with pytest.raises(gate.GateError, match="broken"):
        gate.check_installs(wheels[0].parent, HERE_PYTHON)


@pytest.mark.env
@pytest.mark.parametrize("culprit", ["worker", "core", "root"])
def test_i1_check_installs_refuses_an_undeclared_dependency(culprit: str, fabricate: Callable[..., Path]) -> None:
    """A distribution importing a third-party package its metadata omits fails in its own venv, whoever it is."""
    modules = {**OK_MODULES, culprit: {"modules": {next(iter(OK_MODULES[culprit]["modules"])): "import numpy\n"}}}
    wheels = _release(fabricate, "2.0.0", **modules)
    with pytest.raises(gate.GateError, match="numpy"):
        gate.check_installs(wheels[0].parent, HERE_PYTHON)


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


@pytest.mark.env
def test_m3_release_gate_refuses_untracked_files_under_a_package_root(tagged_workspace: Callable[..., Path]) -> None:
    """An untracked, unignored file under any package root refuses; ignored files and files elsewhere do not."""
    root = tagged_workspace(["2.0.0"])
    dist = root / "dist"
    dist.mkdir()
    gate.check_untracked(dist)
    (root / "notes.txt").write_text("x")
    gate.check_untracked(dist)
    (root / ".gitignore").write_text("*.generated\n")
    (root / "packages/senselab-core/src/senselab_core/x.generated").write_text("x")
    gate.check_untracked(dist)
    for rel in (
        "packages/senselab-core/src/senselab_core/stray.py",
        "packages/senselab-worker/stray.txt",
        "src/senselab/stray.py",
    ):
        (root / rel).write_text("x")
        with pytest.raises(gate.GateError, match="untracked"):
            gate.check_untracked(dist)
        (root / rel).unlink()


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
    assert "github.ref_name" in gate_command and "github.event.release.prerelease" in gate_command
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


@pytest.mark.logic
@pytest.mark.parametrize("path", ["hatch_build_pins.py", "packages/senselab-core/hatch_pin_loader.py"])
@pytest.mark.parametrize("version", ["2.0.0+g1234", "2.0.0+local", "2.0.0+a.1", "1.0+1"])
def test_a32_pin_hook_refuses_a_local_segment(path: str, version: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both the shared hook and core's loader raise rather than write ``==X+local``."""

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
    hook = module.PinHook(str(REPO), {"pin": "senselab-worker", "dependencies": ["numpy"]})
    with pytest.raises(ValueError, match="local"):
        hook.update({"version": version})
    metadata: dict[str, Any] = {"version": "2.0.0"}
    hook.update(metadata)
    assert metadata["dependencies"] == ["numpy", "senselab-worker==2.0.0"]


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
    assert gate.main(root / "dist", tag=RELEASE_FLOOR, prerelease=True) == 0
