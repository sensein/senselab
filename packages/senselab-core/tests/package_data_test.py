"""Every tracked file of a distribution's import package reaches its wheel, and its ``data/`` files its sdist.

The root ``.gitignore`` ignores ``data/``; hatch reads it relative to each package, where the re-includes
written for the repository root never match. Each package's ``[tool.hatch.build] artifacts`` puts the
data back, and this builds every distribution to check that it did.
"""

import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DISTRIBUTIONS = {"senselab": REPO, "senselab_core": REPO / "packages" / "senselab-core"}
DISTRIBUTIONS["senselab_worker"] = REPO / "packages" / "senselab-worker"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build every workspace distribution, wheel and sdist, once."""
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(["uv", "build", "--all-packages", "--out-dir", str(out)], cwd=REPO, check=True, capture_output=True)
    return out


def _tracked(project: Path, name: str) -> list[str]:
    """Paths relative to ``project/src`` of every file git tracks in the import package ``name``."""
    listed = subprocess.run(["git", "ls-files", f"src/{name}"], cwd=project, check=True, capture_output=True, text=True)
    return [line.removeprefix("src/") for line in listed.stdout.splitlines()]


@pytest.mark.contract
@pytest.mark.parametrize("name", sorted(DISTRIBUTIONS))
def test_every_tracked_package_file_is_in_the_wheel(built: Path, name: str) -> None:
    """The wheel holds every file tracked in the import package, ``data/`` files included."""
    (wheel,) = built.glob(f"{name}-*.whl")
    members = set(zipfile.ZipFile(wheel).namelist())
    tracked = _tracked(DISTRIBUTIONS[name], name)
    assert any("/data/" in f for f in tracked), f"{name}: no data files to check"
    assert [f for f in tracked if f not in members] == []


@pytest.mark.contract
@pytest.mark.parametrize("name", sorted(DISTRIBUTIONS))
def test_every_tracked_data_file_is_in_the_sdist(built: Path, name: str) -> None:
    """The sdist holds every file tracked under ``src/**/data/``, so a wheel built from it has them too."""
    (sdist,) = built.glob(f"{name}-*.tar.gz")
    members = {n.split("/", 1)[1] for n in tarfile.open(sdist).getnames() if "/" in n}
    data = [f"src/{f}" for f in _tracked(DISTRIBUTIONS[name], name) if "/data/" in f]
    assert [f for f in data if f not in members] == []
