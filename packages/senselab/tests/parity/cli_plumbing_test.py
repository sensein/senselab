"""The parity CLI end to end, as subprocesses, on the model-free `fake.echo` task.

Every import and line of the real path runs: suite load, item preparation and input binding, the real worker client,
`items`, `run`, and the comparison of the run's outputs. A break in any of them fails here, not on the cluster.
"""

import importlib
import importlib.util
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


def _cli(script: str, *args: str | Path, ok: bool = True) -> str:
    done = subprocess.run(  # noqa: S603 - the repo's own scripts under this interpreter
        [sys.executable, "-c", RUNNER, str(ROOT / script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert (done.returncode == 0) == ok, done.stderr[-3000:]
    return done.stderr


@pytest.mark.logic
def test_cli_items_run_compare_end_to_end(tmp_path: Path) -> None:
    """`items` and `run` complete, and the run's alone and one-request outputs agree under the packaged thresholds."""
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
    assert (ran["format"], ran["flatten"]) == ("parity-outputs@4.0", "flatten@2")
    assert _parity().load_outputs(outputs).items
    stale = tmp_path / "stale-outputs.json"
    stale.write_text(json.dumps({**ran, "format": "parity-outputs@1.0"}), "utf-8")
    with pytest.raises(_parity().ParityError, match="re-?generate it") as raised:
        _parity().load_outputs(stale)
    assert str(stale) in str(raised.value)
    assert {i["id"] for i in ran["items"]} == set(document["items"])
    assert all(set(i["batches"]) == {"1", "N"} and i["split"] == "heldout" for i in ran["items"])
    config = tmp_path / "run.yaml"
    config.write_text("thresholds:\n  embedding:\n    atol: 2\n", "utf-8")
    decided = tmp_path / "comparison.json"
    _cli("benchmarks/parity_reference.py", "compare", outputs, "--config", config, "--out", decided)
    record: dict[str, Any] = json.loads(decided.read_text("utf-8"))
    thresholds = importlib.import_module("senselab_core.runtime.compare").Thresholds
    assert record["format"] == "parity-comparison@1.0" and record["against"] == "batch"
    assert record["mismatches"] == {}
    assert record["thresholds"] == thresholds.load(config).canonical_digest() != thresholds.load().canonical_digest()


def _parity() -> Any:  # noqa: ANN401
    """The parity script, loaded once in this process."""
    if "parity_reference_cli_test" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "parity_reference_cli_test", ROOT / "benchmarks" / "parity_reference.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["parity_reference_cli_test"]


@pytest.mark.logic
def test_cli_items_records_the_task_default_model(tmp_path: Path) -> None:
    """`items` for a backend that declares no model (speech_to_text `transformers`) records the task's default."""
    suite = tmp_path / "asr.yaml"
    entry = {"name": "asr", "adapter": "echo_payloads", "items": 12, "task": "speech_to_text", "metric": "none"}
    suite.write_text(yaml.safe_dump({"format": "benchmark-suite@1.0", "name": "asr", "benchmarks": [entry]}))
    items = tmp_path / "items.json"
    _cli(
        "benchmarks/parity_reference.py",
        "items",
        "--suite",
        suite,
        "--benchmark",
        "asr",
        "--backend",
        "transformers",
        "--data-cache",
        tmp_path / "data",
        "--work",
        tmp_path / "work",
        "--out",
        items,
    )
    importlib.import_module("senselab")
    packaged = importlib.import_module("senselab_core.runtime.call").packaged
    default = packaged().catalogue.default("speech_to_text")
    document = json.loads(items.read_text("utf-8"))
    assert document["model"] == {"id": default.id, "revision": default.revision}
