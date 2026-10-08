"""The triage.tsv script: decision tables and a run tree in, the TSV and its column dictionary out."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pyarrow.parquet as pq
import pytest

from senselab.audio.workflows.triage import triage_table
from tests.audio.workflows.triage.triage_table_test import BIDS, HEADER, _tables

SCRIPT = Path(__file__).parents[3] / "scripts" / "triage_table.py"


def _script() -> ModuleType:
    """The script, imported from its path."""
    spec = importlib.util.spec_from_file_location("triage_table_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_writes_the_table_and_reports_its_counts(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Three recordings in, three rows and the dictionary out, with the verdict and release counts."""
    decisions, evidence = _tables(tmp_path)
    pq.write_table(decisions, tmp_path / "d.parquet")
    pq.write_table(evidence, tmp_path / "e.parquet")
    out = tmp_path / "pub" / triage_table.TABLE_NAME
    argv = ["--decisions", str(tmp_path / "d.parquet"), "--evidence", str(tmp_path / "e.parquet")]
    argv += ["--run-root", str(tmp_path / "out"), "--bids-root", str(tmp_path / BIDS), "--out", str(out)]
    assert _script().main(argv) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == HEADER and len(lines) == 4
    assert (out.parent / triage_table.COLUMNS_NAME).exists()
    printed = capsys.readouterr().out
    assert "3 rows" in printed
    assert "verdict {'discard': 1, 'pass': 1, 'review': 1}" in printed
    assert "release {'': 1, 'as_is': 1, 'redacted': 1}" in printed
