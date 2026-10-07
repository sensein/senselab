"""The recording-vectors data dictionary: one entry per column, each pointing at code that exists."""

import ast
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from senselab.audio.workflows.triage import recording_vectors as rv
from senselab.audio.workflows.triage.nodes import review
from senselab.audio.workflows.triage.nodes.gates import CONFORMANCE_GATES, FLAG_GATES, GATE_SPECS
from senselab.audio.workflows.triage.residue import ALIGNED, FREE, SYLLABLE, VOCAL
from senselab.audio.workflows.triage.vocabulary import (
    BRANCH_ROUTE_STATES,
    DISCARD_GROUNDS,
    FILE_ROUTE_STATES,
    Release,
    Triage,
)
from senselab.text.tasks.pii_detection.redaction_review import ORIGINAL_STATES, REDACTION_STATES, SPEAKER_STATES

REPO = Path(__file__).parents[5]


def _entries() -> dict[str, dict[str, Any]]:
    """The dictionary, keyed by column."""
    return {entry["name"]: entry for entry in rv.dictionary()}


def test_every_column_has_exactly_one_entry_in_schema_order() -> None:
    """The dictionary names every schema column once, in the schema's order."""
    names = [entry["name"] for entry in rv.dictionary()]
    assert names == [field.name for field in rv.schema()]
    assert len(names) == len(set(names))


def test_every_entry_states_every_field() -> None:
    """No entry leaves a field empty."""
    for entry in rv.dictionary():
        for key in ("dtype", *rv.DICTIONARY_FIELDS):
            assert entry.get(key), f"{entry['name']} lacks {key}"
        assert isinstance(entry["source"], list) and entry["source"], entry["name"]


def test_the_dtype_is_the_one_the_schema_writes() -> None:
    """An entry's dtype is the schema's, not a copy that can drift."""
    types = {field.name: str(field.type) for field in rv.schema()}
    for entry in rv.dictionary():
        assert entry["dtype"] == types[entry["name"]]


def test_a_column_the_schema_lacks_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry for a column the writer does not emit fails the load."""
    source = dict(rv._dictionary_source())
    source["columns"] = {**source["columns"], "no_such_column": dict(source["columns"]["participant"])}
    monkeypatch.setattr(rv, "_dictionary_source", lambda: source)
    with pytest.raises(ValueError, match="no_such_column"):
        rv.dictionary()


def test_a_column_the_dictionary_lacks_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A column the writer emits with no entry fails the load."""
    source = dict(rv._dictionary_source())
    source["columns"] = {k: v for k, v in source["columns"].items() if k != "wave_peak"}
    monkeypatch.setattr(rv, "_dictionary_source", lambda: source)
    with pytest.raises(ValueError, match="wave_peak"):
        rv.dictionary()


# ------------------------------------------------------------------------------ the sources resolve


@lru_cache(maxsize=None)
def _definitions(path: Path) -> frozenset[str]:
    """Every function and class in one file, by bare name and by ``Class.method``."""
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}.{child.name}")
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return frozenset(names)


def test_every_source_names_code_that_exists() -> None:
    """Every ``source`` names a function, class or module constant in an existing file."""
    root = REPO / rv.dictionary_document()["source_root"]
    unresolved = []
    for entry in rv.dictionary():
        for source in entry["source"]:
            path, _, qualname = source.partition(":")
            file = root / path
            if not qualname or not file.is_file() or qualname not in _definitions(file):
                unresolved.append(f"{entry['name']}: {source}")
    assert not unresolved, unresolved


def test_the_first_source_is_the_extractor() -> None:
    """Every entry leads with the extractor code that writes the column."""
    for entry in rv.dictionary():
        assert entry["source"][0].startswith("recording_vectors.py:"), entry["name"]


# ------------------------------------------------------------------------------ the families agree


def test_the_measurement_members_are_the_measurements_and_their_kinds() -> None:
    """The measurement family covers exactly the writer's measurements, each with its kind."""
    members = rv._dictionary_source()["families"]["measurement"]["members"]
    assert set(members) == set(rv.MEASUREMENTS)
    kinds = {
        **dict.fromkeys(rv.SCALAR_MEASUREMENTS, "scalar"),
        **dict.fromkeys(rv.VECTOR_MEASUREMENTS, "vector"),
        **dict.fromkeys(rv.MATRIX_MEASUREMENTS, "matrix"),
        **dict.fromkeys(rv.CATEGORICAL_MEASUREMENTS, "categorical"),
    }
    assert {name: spec["kind"] for name, spec in members.items()} == kinds


def test_the_gate_members_read_what_the_registry_says() -> None:
    """The gate family's readings and directions are the registry's."""
    members = rv._dictionary_source()["families"]["gate"]["members"]
    assert list(members) == list(rv.GATE_NAMES)
    for name, spec in members.items():
        assert spec.get("reading") == GATE_SPECS[name].reading, name
        assert spec["op"] == GATE_SPECS[name].op, name
        assert spec["kind"] == ("located" if GATE_SPECS[name].reading is None else "applied"), name


def test_a_located_gate_is_described_by_its_bound_alone() -> None:
    """A located gate is never applied by the fold, so its one column is the resolved bound."""
    applied = {name for names in CONFORMANCE_GATES.values() for name in names} | set(FLAG_GATES)
    assert not {name for name, spec in GATE_SPECS.items() if spec.reading is None} & applied
    entries = _entries()
    for name, spec in GATE_SPECS.items():
        if spec.reading is None:
            entry = entries[f"gate_{name}_bound"]
            assert 'gates.bounds["' + name + '"]' in entry["computation"], name
            assert "nodes/gates.py:load_gate_bounds" in entry["source"], name
            assert f"gate_{name}" not in entries and f"gate_{name}_passed" not in entries, name


def test_the_byte_codes_the_entries_state_are_the_writers() -> None:
    """The classifier and word-outcome codes the block entries list are the writer's tuples."""
    entries = _entries()
    classifiers = ", ".join(f"{index} {name}" for index, name in enumerate(rv.CLASSIFIERS))
    assert f"{classifiers}, {rv.UNKNOWN_CODE} any other" in entries["span_labels"]["description"]
    outcomes = ", ".join(f"{index} {name}" for index, name in enumerate(rv.WORD_OUTCOMES))
    assert f"{outcomes}, {rv.UNKNOWN_CODE} any other" in entries["asr_words"]["description"]
    for name, (low, high) in rv.SQUIM_RANGES:
        assert f"[{low:g}, {high:g}]" in entries["span_squim"]["description"], name


def test_carrier_rejected_lists_every_name_voice_can_write() -> None:
    """The listed values are exactly the gate names and criteria VOICE passes to ``Rejection``."""
    tree = ast.parse((REPO / "src/senselab/audio/workflows/triage/nodes/voice.py").read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    written = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Rejection" and len(node.args) > 1:
            gate = node.args[1]
            if isinstance(gate, ast.Constant):
                written.add(gate.value)
            elif isinstance(gate, ast.Name):
                written.add(constants[gate.id])
    assert set(_entries()["m_carrier_rejected"]["values"]) == written


# ------------------------------------------------------------------------------ the vocabularies


@pytest.mark.parametrize(
    ("column", "vocabulary"),
    [
        ("verdict", [member.value for member in Triage]),
        ("release", [member.value for member in Release]),
        ("discard_ground", list(DISCARD_GROUNDS)),
        ("route_state", list(FILE_ROUTE_STATES)),
        ("route_airway", list(BRANCH_ROUTE_STATES)),
        ("route_speech", list(BRANCH_ROUTE_STATES)),
        ("route_voice", list(BRANCH_ROUTE_STATES)),
        ("residue_method", [SYLLABLE, VOCAL, ALIGNED, FREE]),
        (
            "llm_status",
            [review.CLEAN, review.FLAGGED, review.ABSENT, review.DISABLED, review.NOTHING_TO_READ],
        ),
        ("llm_redaction_judgment", list(REDACTION_STATES)),
        ("llm_original_judgment", list(ORIGINAL_STATES)),
        ("llm_speakers", list(SPEAKER_STATES)),
    ],
)
def test_a_closed_vocabulary_is_the_one_the_code_declares(column: str, vocabulary: list[str]) -> None:
    """A column's listed values are the constants the code declares."""
    assert sorted(_entries()[column]["values"]) == sorted(vocabulary)


# ------------------------------------------------------------------------------ the file carries it


def test_the_schema_carries_the_dictionary_as_json() -> None:
    """The schema's metadata holds the expanded dictionary."""
    metadata = rv.schema().metadata
    document = json.loads(metadata[rv.DICTIONARY_KEY])
    assert document == json.loads(json.dumps(rv.dictionary_document()))
    assert document["schema_version"] == rv.SCHEMA_VERSION
    assert [c["name"] for c in document["columns"]] == [field.name for field in rv.schema()]


def test_a_written_file_is_self_describing(tmp_path: Path) -> None:
    """A parquet written on the schema carries the dictionary in its footer."""
    path = tmp_path / "recording_vectors.000.parquet"
    pq.write_table(rv.schema().empty_table(), path)
    metadata = pq.read_schema(path).metadata
    assert metadata[rv.VERSION_KEY] == str(rv.SCHEMA_VERSION).encode()
    columns = json.loads(metadata[rv.DICTIONARY_KEY])["columns"]
    assert {c["name"] for c in columns} == {field.name for field in rv.schema()}


def test_to_table_keeps_the_dictionary() -> None:
    """A table built from rows keeps the dictionary in its schema."""
    table = rv.to_table([])
    assert isinstance(table, pa.Table)
    assert rv.DICTIONARY_KEY in table.schema.metadata
