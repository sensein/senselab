"""``triage.tsv``: one flat row per recording, over the decision tables.

The decision, its decisive evidence and the acquisition readings come from ``triage_decisions`` and
``triage_evidence`` (:mod:`~senselab.audio.workflows.triage.decision_tables`); the run tree supplies the
source path, the recording's duration, which task-audio files exist and the whole-file readings its store
holds (:func:`store_readings`); the recording vectors supply the whole-file SNR, noise floor and SQUIM
(:data:`VECTOR_COLUMNS`). The columns and their order are ``data/triage_table.yaml``.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import yaml

from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.recording_vectors import RUN_SUBDIR, STORE_NAME, StoreView, read_store
from senselab.audio.workflows.triage.routing_analysis.families import task_family, task_id_of
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
LEXICAL_ITEM = "lexical_words"
SPEECH_IN_TASK_ITEM = "speech_in_task"
INTERFERENCE_PREFIX = "interference_in_task:"
FAULT_PREFIX = "fault_in_task:"
FOUND_ITEMS = ("breaths_found", "cough_onsets", "ddk_events_against_instructed", "ddk_events_found")
INSTRUCTED_ITEMS = ("breaths_found", "coughs_against_instructed", "ddk_events_against_instructed")
_SCALAR_ITEMS = frozenset({LEVEL_ITEM, ACTIVE_ITEM, LEXICAL_ITEM, SPEECH_IN_TASK_ITEM, *FOUND_ITEMS, *INSTRUCTED_ITEMS})
VECTOR_COLUMNS = {
    "q_plain_snr_db": "snr_db",
    "q_plain_floor_dbfs": "noise_floor_dbfs",
    "q_plain_squim_pesq": "squim_pesq",
    "q_plain_squim_stoi": "squim_stoi",
    "q_plain_squim_si_sdr": "squim_si_sdr",
}
"""``recording_vectors`` column to ``triage.tsv`` column: the whole-file plain-stream readings."""
SQUIM_COLUMNS = ("squim_pesq", "squim_stoi", "squim_si_sdr")
NO_SQUIM_BRANCHES = ("AIRWAY",)
"""Branches whose families SQUIM, a speech-quality estimate, does not describe: their SQUIM columns are empty."""
WITHHELD = "withheld"


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
        "lexical_words_n": value(LEXICAL_ITEM),
        "speech_in_task_n": value(SPEECH_IN_TASK_ITEM),
    }


def _number_or_none(value: Any) -> float | None:  # noqa: ANN401 -- a stored reading is any type
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _merged_length(spans: Iterable[tuple[float, float]]) -> float:
    total, cursor = 0.0, -math.inf
    for start, end in sorted(spans):
        start = max(start, cursor)
        if end > start:
            total += end - start
        cursor = max(cursor, end)
    return total


def _cohort_columns(reading: Mapping[str, Any] | None) -> dict[str, Any]:
    """The speaker columns of a ``cohort_reading``."""
    block = dict((reading or {}).get("other_speaker") or {})
    measured = block.get("status") == "measured"

    def cosine(pool: str) -> float | None:
        record = block.get(pool) if measured else None
        return _number_or_none(record.get("cosine")) if isinstance(record, Mapping) else None

    consistency = dict(((reading or {}).get("checks") or {}).get("session_speaker_consistency") or {})
    value = next(
        (
            c.get("value")
            for c in consistency.get("comparisons") or ()
            if c.get("name") == "session_vs_other_sessions_cosine"
        ),
        None,
    )
    return {
        "speaker_target_cosine": cosine("target"),
        "speaker_non_task_cosine": cosine("non_task"),
        "session_consistency_cosine": _number_or_none(value),
        "enrollment_source": block.get("enrollment_source") if reading else None,
    }


def store_readings(view: StoreView, extent: tuple[float, float] | None) -> dict[str, Any]:
    """The whole-file readings ``triage.tsv`` takes from one store.

    Args:
        view: The store.
        extent: The task extent, for the masked seconds inside it, or None.

    Returns:
        ``session_background_dbfs``, ``speech_over_residual_db``, ``clipped_s``, ``dropout_s``, the ASR
        columns, ``off_task_words_n`` (the ledger's non-task speech words, where a ledger stands),
        ``masked_s`` (the final masks' merged seconds inside the extent, where a ledger stands) and the
        speaker columns. A key is None where the store holds no reading.
    """
    out: dict[str, Any] = {}
    floor = view.last("measurement", name="session_floor")
    background = (floor.attributes.get("session_background") or {}) if floor is not None else {}
    out["session_background_dbfs"] = _number_or_none(background.get("level_db"))
    residual = view.last("measurement", name="speech_residual")
    over = (residual.attributes.get("foreground_minus_residual_db") or {}) if residual is not None else {}
    out["speech_over_residual_db"] = _number_or_none(over.get("whole")) if isinstance(over, Mapping) else None
    raw = view.last("measurement", name="disruptions_file")
    out["clipped_s"] = _number_or_none(raw.attributes.get("clipped_s")) if raw is not None else None
    out["dropout_s"] = _number_or_none(raw.attributes.get("dropout_s")) if raw is not None else None
    transcript = view.last("measurement", name="consensus_transcript")
    if transcript is not None:
        outcomes = dict(transcript.attributes.get("outcomes") or {})
        words = sum(int(v) for v in outcomes.values() if isinstance(v, int))
        out["asr_agreement"] = round(int(outcomes.get("agreement") or 0) / words, 4) if words else None
        single = transcript.attributes.get("single_hypothesis")
        sources = transcript.attributes.get("n_sources")
        out["asr_single_hypothesis"] = single if single is not None else (None if sources is None else sources < 2)
        out["asr_degenerate_n"] = transcript.attributes.get("degenerate_n")
    ledger = view.last("measurement", name="pii_ledger")
    if ledger is not None:
        out["off_task_words_n"] = len(ledger.attributes.get("non_task_speech_ids") or ())
        masks = [
            (float(m["start_s"]), float(m["end_s"]))
            for m in ledger.attributes.get("final_masks") or ()
            if isinstance(m, Mapping) and m.get("start_s") is not None and m.get("end_s") is not None
        ]
        if extent is not None:
            masks = [(max(a, extent[0]), min(b, extent[1])) for a, b in masks if min(b, extent[1]) > max(a, extent[0])]
        out["masked_s"] = round(_merged_length(masks), 3)
    cohort = view.last("measurement", name="cohort_reading")
    out.update(_cohort_columns(None if cohort is None else cohort.attributes))
    return out


def manifest_sources(path: Path) -> dict[str, str]:
    """The run manifest's source per run directory name.

    Args:
        path: A JSONL manifest, one ``{"stem": <run directory name>, "source": <audio path>}`` per line.

    Returns:
        ``{run directory name: source path}``.

    Raises:
        ValueError: Where a run directory name appears twice.
    """
    sources: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name = str(record["stem"])
        if name in sources:
            raise ValueError(f"manifest names run directory {name} twice")
        sources[name] = str(record["source"])
    return sources


def _relative(path: str, bids_root: Path) -> str:
    source = Path(path)
    return str(source.relative_to(bids_root)) if source.is_relative_to(bids_root) else str(source)


def run_tree_facts(
    run_dir: str,
    audio_paths: Sequence[str],
    run_root: Path,
    bids_root: Path,
    source: str | None = None,
    extent: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """What only the run tree holds: the source path, the duration, the task audio present, the store's readings.

    Args:
        run_dir: The run directory, relative to ``run_root``.
        audio_paths: ``triage_decisions.audio_paths``, relative to ``run_root``.
        run_root: The scan root the decision tables were built over.
        bids_root: The BIDS root the source path is given relative to.
        source: The manifest's source for this run, used where the store records no ``recording`` stream
            (ADMIT found the file unreadable).
        extent: The task extent, for the masked seconds inside it.

    Returns:
        ``source_path``, ``recording_duration_s``, one ``task_audio_*`` key per cut that exists, and
        :func:`store_readings`.
    """
    facts: dict[str, Any] = {}
    try:
        view = read_store(run_root / run_dir / RUN_SUBDIR / STORE_NAME)
    except OSError:
        view = None
    stream = None if view is None else view.last("stream", name="recording")
    if stream is not None:
        path = stream.attributes.get("path")
        if path:
            facts["source_path"] = _relative(str(path), bids_root)
        if stream.extent:
            facts["recording_duration_s"] = float(stream.extent[1])
    if view is not None:
        facts.update(store_readings(view, extent))
    if "source_path" not in facts and source:
        facts["source_path"] = _relative(source, bids_root)
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
    sources: Mapping[str, str] | None = None,
    vectors: pa.Table | None = None,
) -> Iterator[dict[str, Any]]:
    """One ``triage.tsv`` row per decision.

    Args:
        decisions: ``triage_decisions``.
        evidence: ``triage_evidence``.
        run_root: The scan root the decision tables were built over.
        bids_root: The BIDS root.
        threads: Run-tree reads in flight.
        sources: :func:`manifest_sources`, the source of a run whose store records none.
        vectors: ``recording_vectors``, read for :data:`VECTOR_COLUMNS` by stem; None leaves them empty.

    Yields:
        One mapping per recording, keyed by :func:`columns`.
    """
    by_stem = acquisition_by_stem(evidence)
    by_vector = vector_readings(vectors)
    records = decisions.to_pylist()
    version = column_dictionary()["version"]

    def facts(record: Mapping[str, Any]) -> dict[str, Any]:
        run_dir = str(record["run_dir"])
        source = (sources or {}).get(Path(run_dir).name)
        start, end = record.get("extent_start_s"), record.get("extent_end_s")
        extent = None if start is None or end is None else (float(start), float(end))
        return run_tree_facts(run_dir, record.get("audio_paths") or (), run_root, bids_root, source, extent)

    for record, tree in zip(records, fastio.ordered_map(facts, records, threads=threads)):
        stem = str(record["stem"])
        yield row(record, {**by_stem.get(stem, {}), **by_vector.get(stem, {})}, tree, version)


def vector_readings(vectors: pa.Table | None) -> dict[str, dict[str, Any]]:
    """The whole-file readings ``triage.tsv`` takes from the recording vectors.

    Args:
        vectors: ``recording_vectors`` (at least ``stem`` and the :data:`VECTOR_COLUMNS` it holds), or None.

    Returns:
        By stem, the readings keyed by their ``triage.tsv`` column; empty where no table was given.
    """
    if vectors is None:
        return {}
    present = [name for name in VECTOR_COLUMNS if name in vectors.column_names]
    table = vectors.select(["stem", *present]).to_pylist()
    return {str(r["stem"]): {VECTOR_COLUMNS[name]: _number_or_none(r.get(name)) for name in present} for r in table}


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
    family = decision.get("declared_family") or task_family(task_id_of(str(decision.get("stem") or "")))
    branch = branch_of(family)
    reasons = list(decision.get("reasons") or ())
    annotations = list(decision.get("annotations") or ())
    duration = tree.get("recording_duration_s")
    task_s = decision.get("extent_duration_s")
    released = verdict != "discard" and decision.get("release") != WITHHELD
    masked = tree.get("masked_s") if released else None
    off_task = tree.get("off_task_words_n")
    if off_task is None:
        off_task = readings.get("speech_in_task_n")
    squim = {name: None if branch in NO_SQUIM_BRANCHES else readings.get(name) for name in SQUIM_COLUMNS}
    return {
        "participant_id": decision.get("participant"),
        "session_id": decision.get("session"),
        "task": decision.get("task"),
        "task_family": family,
        "branch": branch,
        "source_path": tree.get("source_path"),
        "verdict": verdict,
        "release": None if verdict == "discard" else decision.get("release"),
        "release_reason": None if verdict == "discard" else decision.get("release_reason"),
        "reason": decision.get("reason"),
        "reasons": reasons,
        "issues_n": len(set(reasons)) + len(set(annotations)),
        "run_status": decision.get("run_status"),
        "missing": list(decision.get("missing") or ()),
        "recording_duration_s": duration,
        "task_start_s": decision.get("extent_start_s"),
        "task_end_s": decision.get("extent_end_s"),
        "task_duration_s": task_s,
        "task_fraction": None if not duration or task_s is None else round(float(task_s) / float(duration), 4),
        "task_events_found": readings.get("task_events_found"),
        "task_events_instructed": readings.get("task_events_instructed"),
        "annotations": annotations,
        "decisive_evidence": EVIDENCE_SEP.join(evidence_text(item) for item in decision.get("evidence") or ()),
        "level_rel_session_db": readings.get("level_rel_session_db"),
        "active_s": readings.get("active_s"),
        "interference_in_task": list(readings.get("interference_in_task") or ()),
        "fault_in_task": list(readings.get("fault_in_task") or ()),
        "snr_db": readings.get("snr_db"),
        "noise_floor_dbfs": readings.get("noise_floor_dbfs"),
        "session_background_dbfs": tree.get("session_background_dbfs"),
        "speech_over_residual_db": tree.get("speech_over_residual_db"),
        **squim,
        "clipped_s": tree.get("clipped_s"),
        "dropout_s": tree.get("dropout_s"),
        "lexical_words_n": readings.get("lexical_words_n"),
        "off_task_words_n": off_task,
        "asr_agreement": tree.get("asr_agreement"),
        "asr_single_hypothesis": tree.get("asr_single_hypothesis"),
        "asr_degenerate_n": tree.get("asr_degenerate_n"),
        "masked_s": masked,
        "masked_fraction": None if masked is None or not task_s else round(float(masked) / float(task_s), 4),
        "speaker_target_cosine": tree.get("speaker_target_cosine"),
        "speaker_non_task_cosine": tree.get("speaker_non_task_cosine"),
        "session_consistency_cosine": tree.get("session_consistency_cosine"),
        "enrollment_source": tree.get("enrollment_source"),
        "task_audio_plain": tree.get("task_audio_plain"),
        "task_audio_enhanced": tree.get("task_audio_enhanced"),
        "task_audio_redacted": tree.get("task_audio_redacted"),
        "figure": decision.get("figure_path"),
        "pipeline_commit": decision.get("commit"),
        "config_hash": decision.get("config_hash"),
        "schema_version": version,
    }


def coverage_problems(
    decisions: pa.Table, table_rows: Sequence[Mapping[str, Any]], expected: Iterable[str] | None = None
) -> list[str]:
    """What stops the table being one row per recording.

    Args:
        decisions: ``triage_decisions``, read for each row's run directory and stem.
        table_rows: The rows :func:`rows` built from it, in its order.
        expected: The run directory names the run was asked for (the manifest), or None.

    Returns:
        One line per problem: a stem or run directory that appears twice, a run asked for and absent or
        present and not asked for, a row with no source path, a source path held by two rows. Empty
        where there are none.
    """
    problems: list[str] = []
    stems = [str(s) for s in decisions.column("stem").to_pylist()]
    names = [Path(str(d)).name for d in decisions.column("run_dir").to_pylist()]
    for label, values in (("stem", stems), ("run directory", names)):
        repeated = sorted(value for value, n in Counter(values).items() if n > 1)
        if repeated:
            problems.append(f"{len(repeated)} {label}s appear more than once, e.g. {repeated[:3]}")
    if expected is not None:
        wanted, have = set(expected), set(names)
        if wanted - have:
            problems.append(f"{len(wanted - have)} manifest runs have no row, e.g. {sorted(wanted - have)[:3]}")
        if have - wanted:
            problems.append(f"{len(have - wanted)} rows are not in the manifest, e.g. {sorted(have - wanted)[:3]}")
    empty = [stem for stem, r in zip(stems, table_rows) if not r.get("source_path")]
    if empty:
        problems.append(f"{len(empty)} rows have no source_path, e.g. {empty[:3]}")
    paths = Counter(str(r["source_path"]) for r in table_rows if r.get("source_path"))
    shared = sorted(path for path, n in paths.items() if n > 1)
    if shared:
        problems.append(f"{len(shared)} source paths are held by more than one row")
    return problems


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
