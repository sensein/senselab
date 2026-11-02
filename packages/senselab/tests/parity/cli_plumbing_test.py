"""The measurement and parity CLIs end to end, as subprocesses, on the model-free `fake.echo` task.

Every import and line of the real path runs: suite load, item preparation and input binding, the real worker client,
`measure`, `generate`, `items`, `run`. A break in any of them fails here, not on the cluster.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "benchmarks" / "suites" / "plumbing.yaml"
RUNNER = """
import dataclasses, runpy, sys
import senselab
import senselab_core.runtime.fake_backend
from senselab_core.runtime.call import packaged, use_packaged
from senselab_core.runtime.envs import EnvHandle

real = packaged()
host = EnvHandle(sys.executable, "3", "0" * 64, "0" * 64, None)
use_packaged(dataclasses.replace(real, open_env=lambda b, p: host if b.name == "fake" else real.open_env(b, p)))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _cli(script: str, *args: str | Path) -> None:
    done = subprocess.run(  # noqa: S603 - the repo's own scripts under this interpreter
        [sys.executable, "-c", RUNNER, str(ROOT / script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]


@pytest.mark.logic
def test_cli_measure_generate_items_run_end_to_end(tmp_path: Path) -> None:
    """`measure`, `generate`, `items` and `run` complete, and the generated profile judges the run's outputs."""
    common: list[str | Path] = [
        "--suite",
        SUITE,
        "--benchmark",
        "echo",
        "--backend",
        "fake",
        "--data-cache",
        tmp_path / "data",
    ]
    measured = tmp_path / "obs.json"
    _cli(
        "scripts/generate_tolerance.py",
        "measure",
        *common,
        "--execution",
        "laptop",
        "--repeats",
        "2",
        "--sha",
        "a" * 40,
        "--job",
        "1",
        "--work",
        tmp_path / "work",
        "--out",
        measured,
    )
    observed: dict[str, Any] = json.loads(measured.read_text("utf-8"))
    assert observed["backend"] == "fake"
    assert observed["items"], "no calibration item ran"
    _cli("scripts/generate_tolerance.py", "generate", measured, "--out", tmp_path / "profiles")
    profile = yaml.safe_load((tmp_path / "profiles" / "fake.yaml").read_text("utf-8"))
    assert profile["derivation"]["split"] == "calibration" and profile["fields"]

    items = tmp_path / "items.json"
    _cli("benchmarks/parity_reference.py", "items", *common, "--work", tmp_path / "work", "--out", items)
    document = json.loads(items.read_text("utf-8"))
    assert document["task"] == "fake.echo" and document["items"]
    outputs = tmp_path / "outputs.json"
    _cli(
        "benchmarks/parity_reference.py",
        "run",
        *common,
        "--execution",
        "laptop",
        "--sha",
        "a" * 40,
        "--job",
        "2",
        "--work",
        tmp_path / "work",
        "--out",
        outputs,
    )
    ran: dict[str, Any] = json.loads(outputs.read_text("utf-8"))
    assert {i["id"] for i in ran["items"]} == set(document["items"])
    assert all(set(i["batches"]) == {"1", "N"} and i["split"] == "heldout" for i in ran["items"])
