"""The adapter import guard is an allowlist: the standard library and ``senselab.utils.reproduce`` only."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from senselab.utils.reproduce import run
from tests.utils.reproduce.first_party_fixture import REPO, first_party_packages

SRC = str(REPO / "src")
ALLOWED_LINES = [
    "import json, pathlib, sys",
    "from senselab.utils.reproduce.metrics import METRIC_NAMES",
    "from senselab.utils.reproduce import metrics",
    "import senselab.utils.reproduce.digest",
    "import importlib; importlib.import_module('senselab.utils.reproduce.metrics')",
]


def _run_adapter(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    """Run ``ADAPTER_RUNNER`` on ``source`` as the driver does, without the sandbox around it."""
    adapter = tmp_path / "adapter.py"
    adapter.write_text(source)
    read, write = os.pipe()
    try:
        return subprocess.run(
            [sys.executable, "-I", "-B", "-c", run.ADAPTER_RUNNER, SRC, str(write), str(adapter), str(tmp_path)],
            pass_fds=(write,),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    finally:
        os.close(read)
        os.close(write)


@pytest.mark.logic
def test_the_workspace_has_the_expected_first_party_packages() -> None:
    """Discovery finds the packages of every distribution, so the parametrised refusals cover all of them."""
    assert {"senselab", "senselab_core", "senselab_worker"} <= set(first_party_packages())


@pytest.mark.logic
@pytest.mark.parametrize("package", first_party_packages())
@pytest.mark.parametrize(
    "form",
    [
        "import {p}",
        "from {p} import x",
        "import {p}.sub",
        "import importlib; importlib.import_module('{p}.sub')",
        "__import__('{p}')",
        "__import__('{p}.' + 'sub')",
    ],
)
def test_an_adapter_may_not_import_any_first_party_package(tmp_path: Path, package: str, form: str) -> None:
    """Every first-party top-level package, by every import form, ends the run with the adapter-import exit."""
    outcome = _run_adapter(tmp_path, form.format(p=package) + "\n")
    assert outcome.returncode == run.EXIT_ADAPTER_IMPORT, outcome.stderr
    assert "E_ADAPTER_IMPORT" in outcome.stderr


@pytest.mark.logic
@pytest.mark.parametrize("module", ["numpy", "yaml", "pydantic", "requests"])
def test_an_adapter_may_not_import_a_third_party_package(tmp_path: Path, module: str) -> None:
    """The allowlist is the standard library plus the reproduce package, so a third-party import is refused too."""
    outcome = _run_adapter(tmp_path, f"import {module}\n")
    assert outcome.returncode == run.EXIT_ADAPTER_IMPORT, outcome.stderr


@pytest.mark.logic
@pytest.mark.parametrize("line", ALLOWED_LINES)
def test_an_adapter_may_import_the_stdlib_and_the_reproduce_package(tmp_path: Path, line: str) -> None:
    """Standard-library modules and ``senselab.utils.reproduce`` modules import without a violation."""
    outcome = _run_adapter(tmp_path, f"{line}\nimport json, os, re, collections, dataclasses\n")
    assert outcome.returncode == 0, outcome.stderr
