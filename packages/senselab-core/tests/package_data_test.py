"""A real build of this tree passes every static release check but the version check.

The version check is left out because this tree is not on a release tag, so hatch-vcs writes a dev version. Every
other check is the release gate's own: the members of each artifact equal the tracked files its hatch config
selects (data files included), and the metadata equals each pyproject.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if "release_gate" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("release_gate", REPO / "scripts" / "release_gate.py")
    assert _spec and _spec.loader
    sys.modules["release_gate"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["release_gate"])
gate = sys.modules["release_gate"]
PACKAGES = {"senselab": "senselab", "senselab-core": "senselab_core", "senselab-worker": "senselab_worker"}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Build every workspace distribution, wheel and sdist, once."""
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(["uv", "build", "--all-packages", "--out-dir", str(out)], cwd=REPO, check=True, capture_output=True)
    return list(gate.dist_files(out))


@pytest.mark.env
@pytest.mark.network
@pytest.mark.parametrize(
    "check",
    ["check_names", "check_files", "check_closure", "check_metadata", "check_members", "check_wheel_metadata"],
)
def test_the_build_of_this_tree_passes_the_static_release_checks(built: list[Path], check: str) -> None:
    """Each static check of the release gate, the version check aside, passes on a build of this tree."""
    if check in ("check_metadata", "check_members", "check_wheel_metadata"):
        getattr(gate, check)(built, REPO)
    else:
        getattr(gate, check)(built)


@pytest.mark.env
@pytest.mark.network
def test_every_tracked_data_file_is_a_derived_member(built: list[Path]) -> None:
    """The member derivation the checks use includes every tracked ``data/`` file of each import package."""
    modes = gate.tracked_modes(REPO)
    for path in built:
        name = gate._metadata(path)[0]
        root = gate.DIST_ROOTS[name]
        prefix = "src/" if root == "." else f"{root}/src/"
        data = {p for p in modes if p.startswith(f"{prefix}{PACKAGES[name]}/") and "/data/" in p}
        sources = {s for s in gate.expected_members(path, REPO, modes).values() if s is not None}
        assert data and data <= sources
