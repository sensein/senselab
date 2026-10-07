"""The decision tables: one row per recording (``triage_decisions``) and one per evidence item (``triage_evidence``).

Both read a run root's store the way :mod:`~senselab.audio.workflows.triage.recording_vectors` does,
from the fold's own record, so neither can disagree with the decision. ``triage_decisions`` carries
the decision and the decisive evidence items; ``triage_evidence`` every item the fold weighed.
``specs/20261007-task-events-in-background/design.md`` ("Decision table") holds the vocabulary.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from senselab.audio.workflows.triage.recording_vectors import (
    RUN_SUBDIR,
    STORE_NAME,
    StoreView,
    _extent_columns,
    identity,
    read_store,
    stem_of,
)

SCHEMA_VERSION = 1
DECISIONS_NAME = "triage_decisions"
EVIDENCE_NAME = "triage_evidence"
FIGURE_PATH = Path("summary") / "summary.pdf"
TASK_AUDIO = ("task_plain", "task_enhanced", "task_redacted")

EVIDENCE_ITEM = pa.struct(
    [
        pa.field("name", pa.string()),
        pa.field("value", pa.string()),
        pa.field("unit", pa.string()),
        pa.field("comparison", pa.string()),
        pa.field("threshold", pa.string()),
        pa.field("effect", pa.string()),
    ]
)
"""One evidence item in ``triage_decisions.evidence``; ``value`` and ``threshold`` are JSON."""


@cache
def evidence_vocabulary() -> dict[str, Any]:
    """The evidence item vocabulary.

    Returns:
        ``data/decision_evidence.yaml``, parsed.
    """
    text = resources.files("senselab.audio.workflows.triage.data").joinpath("decision_evidence.yaml").read_text()
    return dict(yaml.safe_load(text))


def item_group(name: str) -> str:
    """The family group an evidence item belongs to.

    Args:
        name: The item's name.

    Returns:
        Its group in ``data/decision_evidence.yaml``: by name, by ``<prefix>:``, else the flag group.
    """
    vocabulary = evidence_vocabulary()
    for group, items in vocabulary["groups"].items():
        if name in items:
            return str(group)
    prefix, sep, _ = name.partition(":")
    if sep and prefix in vocabulary["prefixes"]:
        return str(vocabulary["prefixes"][prefix]["group"])
    return str(vocabulary["flags"]["group"])


def _json(value: Any) -> str | None:  # noqa: ANN401 -- a reading is any scalar
    """A reading or bound as JSON, None where there is none."""
    return None if value is None else json.dumps(value, sort_keys=True, default=str)


def _provenance(view: StoreView) -> dict[str, Any]:
    """The commit and configuration the fold was written under.

    Args:
        view: The store.

    Returns:
        ``commit``, the last activity's recorded commit (a replay or refold marker), and ``config_hash``,
        the last VERDICT activity's; None where the store records neither.
    """
    commit = next(
        (a["parameters"].get("commit") for a in reversed(view.activities) if a["parameters"].get("commit")), None
    )
    config = next(
        (
            a["parameters"].get("config_hash")
            for a in reversed(view.activities)
            if a.get("node") == "VERDICT" and a["parameters"].get("config_hash")
        ),
        None,
    )
    return {"commit": None if commit is None else str(commit), "config_hash": None if config is None else str(config)}


def _audio_paths(view: StoreView, run_root: Path, root: Path) -> list[str]:
    """The task-extent audio a reviewer listens to, else the recording itself.

    Args:
        view: The store.
        run_root: The run root.
        root: The scan root paths are given relative to.

    Returns:
        Each live ``task_*`` stream's path, or the ``recording`` stream's where none was cut.
    """
    streams = [view.last("stream", name=name) for name in TASK_AUDIO]
    found = [s for s in streams if s is not None and s.attributes.get("path")]
    if not found:
        recording = view.last("stream", name="recording")
        found = [recording] if recording is not None and recording.attributes.get("path") else []
    paths = []
    for stream in found:
        path = Path(str(stream.attributes["path"]))
        if not path.is_absolute():
            path = run_root / RUN_SUBDIR / path
        paths.append(str(path.relative_to(root)) if path.is_relative_to(root) else str(path))
    return paths


def decision_rows(run_root: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """One recording's decision row and its evidence rows.

    Args:
        run_root: The directory holding ``run/store.jsonl``.
        root: The scan root paths are given relative to.

    Returns:
        ``(decision, evidence)``, or None where the store is unreadable or holds no fold.
    """
    try:
        view = read_store(run_root / RUN_SUBDIR / STORE_NAME)
    except OSError:
        return None
    fold = view.last("verdict", node="VERDICT")
    if fold is None:
        return None
    decision = fold.attributes
    stem = stem_of(run_root)
    participant, session, task = identity(stem)
    recording = view.last("stream", name="recording")
    duration_s = recording.extent[1] if recording and recording.extent else None
    extent = _extent_columns(view, duration_s)
    start, end = extent.get("task_extent_start_s"), extent.get("task_extent_end_s")
    items = [dict(entry) for entry in decision.get("evidence") or () if isinstance(entry, Mapping)]
    figure = run_root / FIGURE_PATH
    row = {
        "participant": participant,
        "session": session,
        "task": task,
        "declared_family": decision.get("declared_family"),
        "stem": stem,
        "run_dir": str(run_root.relative_to(root)) if run_root.is_relative_to(root) else str(run_root),
        "verdict": decision.get("triage"),
        "release": decision.get("release"),
        "reason": decision.get("reason"),
        "reasons": [str(r) for r in decision.get("reason_keys") or ()],
        "run_status": decision.get("run_status"),
        "missing": [str(m) for m in decision.get("missing") or ()],
        "extent_start_s": start,
        "extent_end_s": end,
        "extent_duration_s": None if start is None or end is None else round(float(end) - float(start), 3),
        "annotations": [str(a) for a in decision.get("annotation_keys") or ()],
        "evidence": [
            {
                "name": str(entry.get("name")),
                "value": _json(entry.get("value")),
                "unit": entry.get("unit"),
                "comparison": entry.get("comparison"),
                "threshold": _json(entry.get("threshold")),
                "effect": entry.get("effect"),
            }
            for entry in items
            if entry.get("decisive")
        ],
        "figure_path": (str(figure.relative_to(root)) if figure.exists() and figure.is_relative_to(root) else None),
        "audio_paths": _audio_paths(view, run_root, root),
        **_provenance(view),
        "schema_version": SCHEMA_VERSION,
    }
    evidence = [
        {
            "stem": stem,
            "declared_family": decision.get("declared_family"),
            "verdict": decision.get("triage"),
            "group": item_group(str(entry.get("name"))),
            "name": str(entry.get("name")),
            "value": _json(entry.get("value")),
            "unit": entry.get("unit"),
            "comparison": entry.get("comparison"),
            "threshold": _json(entry.get("threshold")),
            "effect": entry.get("effect"),
            "decisive": bool(entry.get("decisive")),
            "schema_version": SCHEMA_VERSION,
        }
        for entry in items
    ]
    return row, evidence


def decisions_schema() -> pa.Schema:
    """The ``triage_decisions`` schema.

    Returns:
        One field per column.
    """
    text = pa.string()
    return pa.schema(
        [
            pa.field("participant", text),
            pa.field("session", text),
            pa.field("task", text),
            pa.field("declared_family", text),
            pa.field("stem", text),
            pa.field("run_dir", text),
            pa.field("verdict", text),
            pa.field("release", text),
            pa.field("reason", text),
            pa.field("reasons", pa.list_(text)),
            pa.field("run_status", text),
            pa.field("missing", pa.list_(text)),
            pa.field("extent_start_s", pa.float64()),
            pa.field("extent_end_s", pa.float64()),
            pa.field("extent_duration_s", pa.float64()),
            pa.field("annotations", pa.list_(text)),
            pa.field("evidence", pa.list_(EVIDENCE_ITEM)),
            pa.field("figure_path", text),
            pa.field("audio_paths", pa.list_(text)),
            pa.field("commit", text),
            pa.field("config_hash", text),
            pa.field("schema_version", pa.int32()),
        ]
    )


def evidence_schema() -> pa.Schema:
    """The ``triage_evidence`` schema.

    Returns:
        One field per column.
    """
    text = pa.string()
    return pa.schema(
        [
            pa.field("stem", text),
            pa.field("declared_family", text),
            pa.field("verdict", text),
            pa.field("group", text),
            pa.field("name", text),
            pa.field("value", text),
            pa.field("unit", text),
            pa.field("comparison", text),
            pa.field("threshold", text),
            pa.field("effect", text),
            pa.field("decisive", pa.bool_()),
            pa.field("schema_version", pa.int32()),
        ]
    )


def tables(run_roots: Iterable[Path], root: Path) -> tuple[pa.Table, pa.Table]:
    """Both tables over the run roots.

    Args:
        run_roots: The run roots to read.
        root: The scan root paths are given relative to.

    Returns:
        ``(triage_decisions, triage_evidence)``.
    """
    decisions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for run_root in run_roots:
        rows = decision_rows(run_root, root)
        if rows is None:
            continue
        decisions.append(rows[0])
        evidence.extend(rows[1])
    return (
        pa.Table.from_pylist(decisions, schema=decisions_schema()),
        pa.Table.from_pylist(evidence, schema=evidence_schema()),
    )


def write_tables(decisions: pa.Table, evidence: pa.Table, out_dir: Path) -> dict[str, Path]:
    """Write both tables as parquet, and ``triage_decisions`` as TSV with its lists as JSON.

    Args:
        decisions: ``triage_decisions``.
        evidence: ``triage_evidence``.
        out_dir: The directory to write into.

    Returns:
        The written paths, by name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        DECISIONS_NAME: out_dir / f"{DECISIONS_NAME}.parquet",
        f"{DECISIONS_NAME}_tsv": out_dir / f"{DECISIONS_NAME}.tsv",
        EVIDENCE_NAME: out_dir / f"{EVIDENCE_NAME}.parquet",
    }
    pq.write_table(decisions, paths[DECISIONS_NAME])
    pq.write_table(evidence, paths[EVIDENCE_NAME])
    flat = flat_for_tsv(decisions)
    with paths[f"{DECISIONS_NAME}_tsv"].open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, escapechar="\\")
        writer.writerow(flat.column_names)
        for row in zip(*(flat.column(name).to_pylist() for name in flat.column_names)):
            writer.writerow(["" if value is None else value for value in row])
    return paths


def flat_for_tsv(table: pa.Table) -> pa.Table:
    """The table with every list or struct column as compact JSON text, so it writes as TSV.

    Args:
        table: A decision table.

    Returns:
        The same columns, nested ones as JSON strings.
    """
    columns: dict[str, Sequence[Any]] = {}
    for name in table.column_names:
        column = table.column(name)
        if pa.types.is_list(column.type) or pa.types.is_struct(column.type):
            columns[name] = [
                None if value is None else json.dumps(value, separators=(",", ":"), sort_keys=True)
                for value in column.to_pylist()
            ]
        else:
            columns[name] = column.to_pylist()
    return pa.table(columns)
