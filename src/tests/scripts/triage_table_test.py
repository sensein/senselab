"""The triage.tsv script: decision tables and a run tree in, the TSV and its column dictionary out."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pyarrow as pa
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


def _argv(tmp_path: Path, out: Path) -> list[str]:
    argv = ["--decisions", str(tmp_path / "d.parquet"), "--evidence", str(tmp_path / "e.parquet")]
    return argv + ["--run-root", str(tmp_path / "out"), "--bids-root", str(tmp_path / BIDS), "--out", str(out)]


def _manifest(tmp_path: Path, decisions: pa.Table) -> Path:
    path = tmp_path / "manifest.jsonl"
    lines = []
    for r in decisions.to_pylist():
        source = tmp_path / BIDS / r["participant"] / r["session"] / "audio" / f"{r['stem']}.wav"
        lines.append(json.dumps({"stem": Path(r["run_dir"]).name, "source": str(source)}) + "\n")
    path.write_text("".join(lines))
    return path


def test_an_unreadable_recording_takes_its_source_and_family_from_the_manifest_and_stem(tmp_path: Path) -> None:
    """A store with no recording stream: source path from the manifest, family from the stem, so no empty cell."""
    decisions, evidence = _tables(tmp_path)
    rows = decisions.to_pylist()
    (tmp_path / "out" / rows[2]["run_dir"] / "run" / "store.jsonl").write_text("")
    rows[2]["declared_family"] = None
    decisions = pa.Table.from_pylist(rows, schema=decisions.schema)
    pq.write_table(decisions, tmp_path / "d.parquet")
    pq.write_table(evidence, tmp_path / "e.parquet")
    out = tmp_path / "pub" / triage_table.TABLE_NAME
    assert _script().main(_argv(tmp_path, out)) == 2, "no source path and no manifest fails the build"
    assert not out.exists()
    assert _script().main([*_argv(tmp_path, out), "--manifest", str(_manifest(tmp_path, decisions))]) == 0
    lines = [line.split("\t") for line in out.read_text(encoding="utf-8").splitlines()]
    row = dict(zip(lines[0], lines[3]))
    assert row["source_path"].endswith(f"{rows[2]['stem']}.wav") and row["task_family"] == "prolonged-vowel"
    assert len({dict(zip(lines[0], line))["source_path"] for line in lines[1:]}) == 3


def test_a_recording_missing_from_or_repeated_in_the_tables_fails_the_build(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A manifest run with no row, or a row twice, writes nothing and exits 2."""
    decisions, evidence = _tables(tmp_path)
    pq.write_table(evidence, tmp_path / "e.parquet")
    out = tmp_path / "pub" / triage_table.TABLE_NAME
    manifest = _manifest(tmp_path, decisions)
    pq.write_table(decisions.slice(1), tmp_path / "d.parquet")
    assert _script().main([*_argv(tmp_path, out), "--manifest", str(manifest)]) == 2
    assert "1 manifest runs have no row" in capsys.readouterr().err
    pq.write_table(pa.concat_tables([decisions, decisions.slice(0, 1)]), tmp_path / "d.parquet")
    assert _script().main([*_argv(tmp_path, out), "--manifest", str(manifest)]) == 2
    err = capsys.readouterr().err
    assert "1 stems appear more than once" in err and "1 source paths are held by more than one row" in err
    assert not out.exists()
