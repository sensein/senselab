"""The recording-vectors writer script: a merged file keeps the data dictionary its shards carry."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pyarrow.parquet as pq
import pytest

from senselab.audio.workflows.triage import recording_vectors as rv

SCRIPT = Path(__file__).parents[3] / "scripts" / "triage_recording_vectors.py"


def _script() -> ModuleType:
    """The writer script, imported from its path."""
    spec = importlib.util.spec_from_file_location("triage_recording_vectors", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_merged_file_carries_the_dictionary(tmp_path: Path) -> None:
    """Merging shards keeps the dictionary they carry."""
    script = _script()
    for index in range(2):
        script.write_private(rv.schema().empty_table(), tmp_path / f"recording_vectors.{index:03d}.parquet")
    out = tmp_path / "merged" / "recording_vectors.parquet"
    script.merge(tmp_path, out)
    metadata = pq.read_schema(out).metadata
    assert json.loads(metadata[rv.DICTIONARY_KEY]) == json.loads(json.dumps(rv.dictionary_document()))


def test_shards_from_different_dictionaries_are_refused(tmp_path: Path) -> None:
    """Shards written with different dictionaries are not merged."""
    script = _script()
    table = rv.schema().empty_table()
    script.write_private(table, tmp_path / "recording_vectors.000.parquet")
    other = table.replace_schema_metadata({**table.schema.metadata, rv.DICTIONARY_KEY: b"{}"})
    script.write_private(other, tmp_path / "recording_vectors.001.parquet")
    with pytest.raises(SystemExit, match="different data dictionaries"):
        script.merge(tmp_path, tmp_path / "merged" / "recording_vectors.parquet")
