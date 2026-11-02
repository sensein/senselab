"""`regenerate_equivalence.py sample` and `regenerate` end to end, as subprocesses, on the plumbing suite.

The fake task is declared only when `senselab` registers its packaged tasks (the runner's `use_packaged` hook), as
`diarize` is, so a script that never imports `senselab` fails here with `UnknownTask`, not on the cluster.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from senselab_core.runtime.equivalence import parse_equivalence

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "benchmarks" / "suites" / "plumbing.yaml"
RUNNER = """
import dataclasses, importlib, runpy, sys
import senselab_core.runtime.call as call
from senselab_core.runtime.envs import EnvHandle

host = EnvHandle(sys.executable, "3", "0" * 64, "0" * 64, None)
declare = call.use_packaged


def use_packaged(packaged):
    importlib.import_module("senselab_core.runtime.fake_backend")
    opener = packaged.open_env
    return declare(dataclasses.replace(packaged, open_env=lambda b, p: host if b.name == "fake" else opener(b, p)))


call.use_packaged = use_packaged
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _cli(*args: str | Path) -> str:
    done = subprocess.run(  # noqa: S603 - the repo's own script under this interpreter
        [sys.executable, "-c", RUNNER, str(ROOT / "scripts" / "regenerate_equivalence.py"), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


@pytest.mark.logic
def test_cli_sample_twice_then_regenerate(tmp_path: Path) -> None:
    """`sample` executes every held-out item, a second `sample` reuses them all, and `regenerate` writes the table."""
    store = tmp_path / "store"

    def sample(reuse: str, verify: str) -> str:
        return _cli(
            "sample",
            "--suite",
            SUITE,
            "--benchmark",
            "echo",
            "--execution",
            "laptop",
            "--store",
            store,
            "--reuse",
            reuse,
            "--verify",
            verify,
            "--data-cache",
            tmp_path / "data",
            "--work",
            tmp_path / "work",
        )

    first = sample("equivalent", "0")
    assert first.startswith("echo: ") and " 0 reused, 0 verified" in first and not first.startswith("echo: 0 executed")
    second = sample("tolerant", "0.1")
    assert second.startswith("echo: 0 executed") and "0 disagreeing" in second
    table = tmp_path / "equivalence.yaml"
    assert _cli("regenerate", store, "--out", table).strip() == str(table)
    assert parse_equivalence(table.read_text("utf-8")) == {}
