"""``triage.tsv``: one flat row per recording, over the decision tables.

The decision, its decisive evidence and the acquisition readings come from ``triage_decisions`` and
``triage_evidence`` (:mod:`~senselab.audio.workflows.triage.decision_tables`); the run tree supplies only
the source path, the recording's duration and which task-audio files exist. The columns and their
order are ``data/triage_table.yaml``.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import yaml

from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.recording_vectors import RUN_SUBDIR, STORE_NAME
from senselab.utils import fastio

COLUMNS_PATH = Path(__file__).parent / "data" / "triage_table.yaml"
TABLE_NAME = "triage.tsv"
COLUMNS_NAME = "triage.columns.yaml"
NO_BRANCH = "none"
LIST_SEP = ";"
EVIDENCE_SEP = "; "
ARROW = "→"
TASK_AUDIO_COLUMNS = {
    "task_plain": "task_audio_plain",
    "task_enhanced": "task_audio_enhanced",
    "task_redacted": "task_audio_redacted",
}
LEVEL_ITEM = "capture.level_rel_db"
ACTIVE_ITEM = "capture.plain_active_s"
INTERFERENCE_PREFIX = "interference_in_task:"
FAULT_PREFIX = "fault_in_task:"
FOUND_ITEMS = ("breaths_found", "cough_onsets", "ddk_events_against_instructed", "ddk_events_found")
INSTRUCTED_ITEMS = ("breaths_found", "coughs_against_instructed", "ddk_events_against_instructed")
_SCALAR_ITEMS = frozenset({LEVEL_ITEM, ACTIVE_ITEM, *FOUND_ITEMS, *INSTRUCTED_ITEMS})


@functools.cache
def column_dictionary() -> dict[str, Any]:
    """The column dictionary.

    Returns:
        ``data/triage_table.yaml``, parsed.
    """
    return dict(yaml.safe_load(COLUMNS_PATH.read_text()))


def columns() -> list[str]:
    """The column names, in order.

    Returns:
        Every ``columns[].name`` of ``data/triage_table.yaml``.
    """
    return [str(column["name"]) for column in column_dictionary()["columns"]]


def branch_of(family: str | None) -> str:
    """The branch that owns a declared family.

    Args:
        family: The declared family.

    Returns:
        ``AIRWAY``, ``SPEECH`` or ``VOICE``, else ``none``.
    """
    owners = [branch for branch, families in EXPECTATIONS.items() if family and family in families]
    return owners[0] if owners else NO_BRANCH


def _number(value: float) -> str:
    """A number, compact: at most four significant digits, no exponent."""
    if not math.isfinite(value):
        return str(value)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if abs(value) >= 1000:
        return str(round(value))
    text = f"{value:.4g}"
    return text if "e" not in text else f"{value:.6f}".rstrip("0").rstrip(".")


def compact(value: Any) -> str:  # noqa: ANN401 -- a reading is any JSON value
    """A reading as compact text.

    Args:
        value: A decoded JSON value.

    Returns:
        Lowercase booleans, compact numbers, strings as they are, ``null`` for None and anything
        else as compact JSON.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _number(float(value))
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _decoded(text: str | None) -> Any:  # noqa: ANN401 -- a stored reading is any JSON value
    """A JSON-encoded reading, decoded; the text itself where it is not JSON."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


def evidence_text(item: Mapping[str, Any]) -> str:
    """One evidence item as ``name=value<op>threshold→effect``.

    Args:
        item: A ``triage_decisions.evidence`` item; ``value`` and ``threshold`` are JSON.

    Returns:
        The item; without ``<op>threshold`` where it has no comparison or no threshold.
    """
    text = f"{item.get('name')}={compact(_decoded(item.get('value')))}"
    if item.get("comparison") and item.get("threshold") is not None:
        text += f"{item['comparison']}{compact(_decoded(item['threshold']))}"
    return f"{text}{ARROW}{item.get('effect')}"


def _clean(text: str) -> str:
    """Text with tabs and line breaks made spaces."""
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def field_text(value: Any) -> str:  # noqa: ANN401 -- a cell is any scalar or list
    """One TSV field.

    Args:
        value: The cell.

    Returns:
        Empty for None, lists joined with ``;``, numbers compact, and never a tab or line break.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return LIST_SEP.join(field_text(v) for v in value if v is not None)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if math.isnan(value) else _clean(compact(round(value, 3)))
    return _clean(str(value))


def acquisition_by_stem(evidence: pa.Table) -> dict[str, dict[str, Any]]:
    """The acquisition and task-count readings of every recording.

    Args:
        evidence: ``triage_evidence``.

    Returns:
        By stem: ``level_rel_session_db``, ``active_s``, ``interference_in_task``, ``fault_in_task``,
        ``task_events_found`` and ``task_events_instructed``; keys without a reading are absent.
    """
    names = evidence.column("name")
    keep = pc.or_(
        pc.is_in(names, value_set=pa.array(sorted(_SCALAR_ITEMS))),
        pc.or_(pc.starts_with(names, INTERFERENCE_PREFIX), pc.starts_with(names, FAULT_PREFIX)),
    )
    subset = evidence.filter(keep).select(["stem", "name", "value", "threshold"]).to_pylist()
    items: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in subset:
        items[row["stem"]][row["name"]] = row
    return {stem: acquisition(by_name) for stem, by_name in items.items()}


def acquisition(by_name: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One recording's acquisition and task-count readings.

    Args:
        by_name: The recording's evidence rows, by item name.

    Returns:
        The readings, as :func:`acquisition_by_stem` describes.
    """

    def value(name: str) -> Any:  # noqa: ANN401 -- a decoded reading
        row = by_name.get(name)
        return None if row is None else _decoded(row.get("value"))

    def threshold(name: str) -> Any:  # noqa: ANN401 -- a decoded bound
        row = by_name.get(name)
        return None if row is None else _decoded(row.get("threshold"))

    found = next((value(n) for n in FOUND_ITEMS if value(n) is not None), None)
    instructed = next((threshold(n) for n in INSTRUCTED_ITEMS if threshold(n) is not None), None)
    interference = sorted(n[len(INTERFERENCE_PREFIX) :] for n in by_name if n.startswith(INTERFERENCE_PREFIX))
    faults = sorted(f"{n[len(FAULT_PREFIX) :]}:{compact(value(n))}" for n in by_name if n.startswith(FAULT_PREFIX))
    return {
        "level_rel_session_db": value(LEVEL_ITEM),
        "active_s": value(ACTIVE_ITEM),
        "interference_in_task": interference,
        "fault_in_task": faults,
        "task_events_found": found,
        "task_events_instructed": instructed,
    }


def recording_stream(store_path: Path) -> dict[str, Any] | None:
    """ADMIT's ``recording`` stream, read off the head of a store.

    Args:
        store_path: A ``run/store.jsonl``.

    Returns:
        The first ``recording`` stream entity, or None where the store is unreadable or has none.
    """
    try:
        with store_path.open() as handle:
            for line in handle:
                if '"recording"' not in line:
                    continue
                record = json.loads(line)
                attributes = record.get("attributes") or {}
                if record.get("prov_type") == "stream" and attributes.get("name") == "recording":
                    return record
    except (OSError, ValueError):
        return None
    return None


def run_tree_facts(run_dir: str, audio_paths: Sequence[str], run_root: Path, bids_root: Path) -> dict[str, Any]:
    """What only the run tree holds: the source path, the recording's duration, the task audio present.

    Args:
        run_dir: The run directory, relative to ``run_root``.
        audio_paths: ``triage_decisions.audio_paths``, relative to ``run_root``.
        run_root: The scan root the decision tables were built over.
        bids_root: The BIDS root the source path is given relative to.

    Returns:
        ``source_path``, ``recording_duration_s`` and one ``task_audio_*`` key per cut that exists.
    """
    facts: dict[str, Any] = {}
    stream = recording_stream(run_root / run_dir / RUN_SUBDIR / STORE_NAME)
    if stream is not None:
        path = (stream.get("attributes") or {}).get("path")
        if path:
            source = Path(str(path))
            facts["source_path"] = (
                str(source.relative_to(bids_root)) if source.is_relative_to(bids_root) else str(source)
            )
        extent = stream.get("extent")
        if extent:
            facts["recording_duration_s"] = float(extent[1])
    for relative in audio_paths:
        column = TASK_AUDIO_COLUMNS.get(Path(relative).stem)
        if column is not None and os.path.exists(run_root / relative):
            facts[column] = relative
    return facts


def rows(
    decisions: pa.Table,
    evidence: pa.Table,
    run_root: Path,
    bids_root: Path,
    *,
    threads: int = fastio.DEFAULT_THREADS,
) -> Iterator[dict[str, Any]]:
    """One ``triage.tsv`` row per decision.

    Args:
        decisions: ``triage_decisions``.
        evidence: ``triage_evidence``.
        run_root: The scan root the decision tables were built over.
        bids_root: The BIDS root.
        threads: Run-tree reads in flight.

    Yields:
        One mapping per recording, keyed by :func:`columns`.
    """
    by_stem = acquisition_by_stem(evidence)
    records = decisions.to_pylist()
    version = column_dictionary()["version"]

    def facts(record: Mapping[str, Any]) -> dict[str, Any]:
        return run_tree_facts(str(record["run_dir"]), record.get("audio_paths") or (), run_root, bids_root)

    for record, tree in zip(records, fastio.ordered_map(facts, records, threads=threads)):
        yield row(record, by_stem.get(str(record["stem"]), {}), tree, version)


def row(
    decision: Mapping[str, Any], readings: Mapping[str, Any], tree: Mapping[str, Any], version: int
) -> dict[str, Any]:
    """One recording's row.

    Args:
        decision: Its ``triage_decisions`` row.
        readings: Its :func:`acquisition` readings.
        tree: Its :func:`run_tree_facts`.
        version: This table's schema version.

    Returns:
        The row, keyed by :func:`columns`.
    """
    verdict = decision.get("verdict")
    return {
        "participant_id": decision.get("participant"),
        "session_id": decision.get("session"),
        "task": decision.get("task"),
        "task_family": decision.get("declared_family"),
        "branch": branch_of(decision.get("declared_family")),
        "source_path": tree.get("source_path"),
        "verdict": verdict,
        "release": None if verdict == "discard" else decision.get("release"),
        "release_reason": None if verdict == "discard" else decision.get("release_reason"),
        "reason": decision.get("reason"),
        "reasons": list(decision.get("reasons") or ()),
        "run_status": decision.get("run_status"),
        "missing": list(decision.get("missing") or ()),
        "recording_duration_s": tree.get("recording_duration_s"),
        "task_start_s": decision.get("extent_start_s"),
        "task_end_s": decision.get("extent_end_s"),
        "task_duration_s": decision.get("extent_duration_s"),
        "task_events_found": readings.get("task_events_found"),
        "task_events_instructed": readings.get("task_events_instructed"),
        "annotations": list(decision.get("annotations") or ()),
        "decisive_evidence": EVIDENCE_SEP.join(evidence_text(item) for item in decision.get("evidence") or ()),
        "level_rel_session_db": readings.get("level_rel_session_db"),
        "active_s": readings.get("active_s"),
        "interference_in_task": list(readings.get("interference_in_task") or ()),
        "fault_in_task": list(readings.get("fault_in_task") or ()),
        "task_audio_plain": tree.get("task_audio_plain"),
        "task_audio_enhanced": tree.get("task_audio_enhanced"),
        "task_audio_redacted": tree.get("task_audio_redacted"),
        "figure": decision.get("figure_path"),
        "pipeline_commit": decision.get("commit"),
        "config_hash": decision.get("config_hash"),
        "schema_version": version,
    }


def write_tsv(table_rows: Iterable[Mapping[str, Any]], path: Path) -> int:
    """Write the rows as ``triage.tsv``, and the column dictionary beside it as ``triage.columns.yaml``.

    Args:
        table_rows: Rows keyed by :func:`columns`.
        path: The TSV's path.

    Returns:
        How many rows were written.
    """
    names = columns()
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, lineterminator="\n")
        writer.writerow(names)
        for table_row in table_rows:
            writer.writerow([field_text(table_row.get(name)) for name in names])
            written += 1
    (path.parent / COLUMNS_NAME).write_text(COLUMNS_PATH.read_text())
    return written
