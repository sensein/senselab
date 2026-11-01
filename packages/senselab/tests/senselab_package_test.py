"""The 2.x ``senselab`` package: where it resolves, what it imports and the command line it mounts."""

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[2]
FRAMEWORKS = ("torch", "transformers", "speechbrain", "pyannote", "ctranslate2", "vllm")
MODULES = ("senselab", "senselab.cli", "senselab_core.runtime.fake_backend")


@pytest.mark.logic
def test_senselab_resolves_to_packages() -> None:
    """``import senselab`` finds the 2.x package, never the 1.x pool under ``src/``."""
    import senselab

    module = Path(str(senselab.__file__)).resolve()
    source = PACKAGES / "senselab" / "src" / "senselab" / "__init__.py"
    assert PACKAGES.parent / "src" not in module.parents
    assert module == source or module.read_bytes() == source.read_bytes()


@pytest.mark.contract
@pytest.mark.parametrize("module", MODULES)
def test_senselab_imports_no_framework(module: str) -> None:
    """A fresh interpreter importing a 2.x module loads no model framework."""
    probe = (
        "import sys\n"
        f"import {module}\n"
        f"loaded = sorted(m for m in {FRAMEWORKS!r} if m in sys.modules)\n"
        "print(','.join(loaded))\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=True)
    assert done.stdout.strip() == ""


@pytest.mark.contract
def test_cli_mounts_data_fetch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``senselab.cli:main`` serves core's ``data fetch`` and reports a missing manifest as one line."""
    from senselab.cli import main

    assert main(["data", "fetch", str(tmp_path / "missing.yaml")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("senselab: error:")
    assert len(err.strip().splitlines()) == 1
