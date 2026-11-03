"""`scripts/regenerate_equivalence.py regenerate`: the table written from the evidence in result stores."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel.identity import member_key
from senselab_core.runtime.equivalence import EquivalenceEvidence, EvidenceSide, parse_equivalence
from senselab_core.runtime.store import ResultStore

ROOT = Path(__file__).resolve().parents[4]


def _script() -> Any:  # noqa: ANN401
    spec = importlib.util.spec_from_file_location(
        "regenerate_equivalence_under_test", ROOT / "scripts" / "regenerate_equivalence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.logic
def test_regenerate_reads_every_store(tmp_path: Path) -> None:
    """Evidence split across two stores, one per context's job, joins into one class; no store gives no class."""
    cpu, gpu = {"device_kind": "cpu"}, {"device_kind": "cuda"}
    for n, (a, b) in enumerate(((cpu, gpu), (gpu, cpu))):
        ResultStore(tmp_path / str(n)).add_evidence(
            EquivalenceEvidence(
                task="t",
                backend="b",
                key=str(n) * 64,
                stored=EvidenceSide(fingerprint="a" * 64, numerics=a, env_lock="l"),
                current=EvidenceSide(fingerprint="b" * 64, numerics=b, env_lock="l"),
                mismatched=(),
                thresholds="c" * 64,
                run=f"slurm-{n}",
            )
        )
    script = _script()
    out = script.regenerate([tmp_path / "0", tmp_path / "1"], tmp_path / "equivalence.yaml")
    expected = {"b": {"class-0": frozenset({member_key(cpu, "l"), member_key(gpu, "l")})}}
    assert parse_equivalence(out.read_text()) == expected
    script.main(["regenerate", "--out", str(tmp_path / "empty.yaml")])
    assert parse_equivalence((tmp_path / "empty.yaml").read_text()) == {}
