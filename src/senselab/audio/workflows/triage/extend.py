"""Reading, writing and re-exporting a finished triage run, for the drivers that extend one.

A run root holds ``run/store.jsonl`` and ``prov/``. This module carries the four operations every
extend driver needs over that layout — derive the root from a path inside it, read the store under
the run's own id, replace the store atomically, re-export BEP028 — plus manifest reading and
slicing for a Slurm array.

Supersession is here too: a driver recomputing a measurement the store already carries retires the
old one with an invalidation edge, never a deletion. So is the outcome vocabulary a driver records
per derivation, and the rule separating a derivation that *cannot apply* from one that *failed* —
:data:`UNAVAILABLE`, :class:`DerivationOutcome` and :func:`attempt_derivation`.

:func:`replay_decisions` re-decides rather than adds: it retires every live decision the nodes from
TAXONOMY on made, then runs them again over the PREPROCESS output the store already holds. It reads
under :func:`replay_run_id` so that what it writes cannot collide with what it retires, and writes
a marker :func:`find_replay_marker` reads to make a second pass a no-op.

See ``specs/20260912-extend-reprocessed-outputs/design.md`` and
``specs/20260922-replay-decisions-over-a-finished-corpus/design.md``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.classification.huggingface import AudioTooShortForAST
from senselab.audio.tasks.classification.yamnet import SpanTooShortForYAMNet
from senselab.audio.tasks.features_extraction.ppg import PpgsPosteriorgramUnavailable
from senselab.audio.tasks.phonation.api import F0RangeUnavailable
from senselab.audio.tasks.speech_to_text.crisperwhisper import CrisperWhisperDecoderPositionsExceeded
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.consensus import (
    Rebracketed,
    rebracket,
    render_transcript,
    vocabulary_key,
    word_attributes,
    word_from_attributes,
)
from senselab.audio.workflows.triage.enrollment import Enrollment
from senselab.audio.workflows.triage.nodes.common import (
    BranchResult,
    capture_environments,
    describe_exception,
    find_branch_report,
    find_measurement,
    find_verdict,
    live_entities,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.nodes.preprocess import NODE as PREPROCESS_NODE
from senselab.audio.workflows.triage.nodes.preprocess import (
    SpeakerDiarizationUnavailable,
    WithdrawnClip,
    write_withdrawn_clips,
)
from senselab.audio.workflows.triage.nodes.quality import (
    CLIP_AMPLITUDE_MEASUREMENT,
    CLIP_LEVELS,
    CONTEST_VERB,
    CONTRADICTED_CLIP,
    UNCLIPPED_LOUDER_N,
    clip_spans,
    quality,
)
from senselab.audio.workflows.triage.nodes.redact import settle_release
from senselab.audio.workflows.triage.nodes.report import report
from senselab.audio.workflows.triage.nodes.review import (
    LLM_REVIEW_MEASUREMENT,
    PROMPT_VERSION,
    READ_STATES,
    reading_key,
)
from senselab.audio.workflows.triage.nodes.review import NODE as REVIEW_NODE
from senselab.audio.workflows.triage.nodes.taxonomy import NODE as TAXONOMY_NODE
from senselab.audio.workflows.triage.nodes.taxonomy import _write_consensus_taxonomy
from senselab.audio.workflows.triage.nodes.verdict import NODE as VERDICT_NODE
from senselab.audio.workflows.triage.nodes.verdict import verdict
from senselab.audio.workflows.triage.run import (
    LOG_FILE,
    REPORT_NODE,
    NodeOutcome,
    _attempt,
    _attempt_artifacts,
    drive_decisions,
)
from senselab.audio.workflows.triage.vocabulary import GRAPH_ORDER, QUALITY, REDACTION_LLM_ANNOTATION
from senselab.utils.prov_bep028 import to_bep028_graph, write_bep028_files
from senselab.utils.prov_store import Entity, ProvStore
from senselab.utils.subprocess_venv import record_venv_use

RUN_SUBDIR = "run"
STORE_FILE = "store.jsonl"
PROV_SUBDIR = "prov"
PROV_LABEL = "triage"
SLICES_SUBDIR = "slices"
CONSENSUS_TAXONOMY = "consensus_taxonomy"
CONSENSUS_TRANSCRIPT = "consensus_transcript"
REBRACKET = "rebracket"
WORD_SUPERSEDED = "word_superseded"
WITHDRAW_CLIPS = "withdraw_contradicted_clips"
CLIP_SPAN_SUPERSEDED = "clip_span_superseded"
CLIP_CONTEST_SUPERSEDED = "clip_contest_superseded"
QUALITY_REPORT_SUPERSEDED = "quality_report_superseded"
PRAAT_MEASUREMENT_SUPERSEDED = "praat_features_superseded"
DIARIZATION_MEASUREMENT_SUPERSEDED = "diarization_superseded"
ONOMATOPOEIC_TOKENS_KEY = "words.onomatopoeic_tokens"
ASR_MEASURE = "asr"

SOURCE_STREAM = "recording"
"""The stream QUALITY's clip spans were detected on, and the one it names its findings about."""

REPLAY_NODE = "REPLAY"
"""The node name a replay's own activities carry: the retirements and the marker."""

REFOLD_NODE = "REFOLD"
"""The node name a re-fold's own activities carry: the retirement and the marker."""

REPLAY_MARKER_STEP = "decisions_replayed"
"""The step of the marker activity a replay writes last, carrying its config hash and commit."""

REFOLD_MARKER_STEP = "verdict_refolded"
"""The step of the marker activity a re-fold writes last, carrying its config hash and commit."""

DECISION_SUPERSEDED = "decision_superseded"
"""The step of the activity retiring one decision a replay is about to make again."""

READING_CARRIED_STEP = "reading_carried_forward"
"""The step of the REVIEW activity that re-attaches a reading a replay retired to the replayed store."""

READING_SUPERSEDED_STEP = "reading_superseded"
"""The step retiring the replay's own unread annotation once a carried reading stands in its place."""

REVIEW_NONE = "none"
"""No answered reading stood before the replay, so there was nothing to keep."""

REVIEW_READ = "read"
"""The replay's own REVIEW answered, so the earlier reading was replaced by a new one."""

REVIEW_CARRIED = "carried"
"""The earlier reading read exactly what the replayed store holds, and was carried forward."""

REVIEW_NEEDS_REREAD = "needs_reread"
"""The earlier reading read something the replay changed, and the replay did not read it again."""

REVIEW_STATES = (REVIEW_NONE, REVIEW_READ, REVIEW_CARRIED, REVIEW_NEEDS_REREAD)

REPLAYED_NODES: tuple[str, ...] = GRAPH_ORDER[GRAPH_ORDER.index(TAXONOMY_NODE) :]
"""The nodes a replay re-runs: the graph from TAXONOMY on, PREPROCESS and ADMIT read off disk."""

_DECISION_REASON = "the decision was replayed over this run's stored PREPROCESS output"
_READING_SUPERSEDED_REASON = "the reading carried forward from before the replay stands in its place"

_CLIP_SPAN_REASON = f"{CONTRADICTED_CLIP}: an unclipped sample is louder than this span's own level"
_CLIP_AMPLITUDE_REASON = "its per-span levels name clip spans withdrawn as contradicted"
_CLIP_CONTEST_REASON = "the clip span it contests was withdrawn; there is no span left to contest"
_QUALITY_REPORT_REASON = "it counts contests of clip spans that have since been withdrawn"
_WORD_REASON = "the token is in words.onomatopoeic_tokens; this reading spells it unbracketed"
_TRANSCRIPT_REASON = "its words were re-flagged against words.onomatopoeic_tokens"

OK = "ok"
"""A derivation that ran and wrote what it derives."""

ERROR = "error"
"""A row on which something failed. The only status that makes an array task exit nonzero."""

SKIPPED = "skipped"
"""A row whose store was left exactly as it was found, every derivation having landed."""

PRESENT = "present"
"""The store already carries this derivation's output, so it was not recomputed."""

CURRENT = "current"
"""The store already holds this derivation's own answer; recomputing it moved nothing."""

REWRITTEN = "rewritten"
"""This derivation replaced a reading the store held, retiring the old one."""

ABSENT = "absent"
"""This derivation has nothing in the store to work from, or cannot apply to this recording.

The bare word is the first case; the second reads ``absent: <Class>: <message>``.
"""

UNAVAILABLE: tuple[type[BaseException], ...] = (
    AudioTooShortForAST,
    CrisperWhisperDecoderPositionsExceeded,
    F0RangeUnavailable,
    PpgsPosteriorgramUnavailable,
    SpanTooShortForYAMNet,
    SpeakerDiarizationUnavailable,
)
"""Every typed absence a derivation may raise: each says this recording has no such thing to derive.

:func:`attempt_derivation` records one under :data:`ABSENT` and never as a failure.
"""


@dataclass(frozen=True)
class DerivationOutcome:
    """What one derivation did to one store, and whether anything failed.

    Attributes:
        detail: What the slice log records under the derivation's own name — one of the outcome
            words above, ``absent: <reason>``, or the failure's class and message.
        failed: Whether this derivation failed; one that could not apply did not.
    """

    detail: str
    failed: bool


def attempt_derivation(call: Callable[[], str]) -> DerivationOutcome:
    """Run one derivation over a store, separating what cannot apply from what failed.

    Args:
        call: The derivation, already bound to its store and configuration, returning the outcome
            word to record for it.

    Returns:
        The outcome. A typed absence from :data:`UNAVAILABLE` is recorded under :data:`ABSENT` and
        is not a failure; any other ``OSError``, ``ValueError`` or ``LookupError`` is.
    """
    try:
        return DerivationOutcome(call(), failed=False)
    except UNAVAILABLE as error:
        return DerivationOutcome(f"{ABSENT}: {describe_exception(error)}", failed=False)
    except (OSError, ValueError, LookupError) as error:
        return DerivationOutcome(describe_exception(error), failed=True)


def read_manifest(path: Path, *, required: Sequence[str] = ("stem",)) -> list[dict[str, Any]]:
    """Read a corpus manifest into rows.

    Args:
        path: The JSONL file, one object per line.
        required: Keys every row must carry a truthy value for.

    Returns:
        One dict per non-blank line, in file order.

    Raises:
        ValueError: If a line is not a JSON object, or is missing one of ``required``.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            missing = [key for key in required if not payload.get(key)]
            if missing:
                raise ValueError(f"{path}:{number} carries no {' or '.join(missing)}")
            rows.append(payload)
    return rows


def take_slice(rows: Sequence[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    """The stride of a manifest one array task owns.

    Args:
        rows: Every manifest row.
        index: This task's 0-based index.
        count: How many tasks the array has.

    Returns:
        ``rows[index::count]``.

    Raises:
        ValueError: If ``count`` is not positive, or ``index`` is outside it.
    """
    if count < 1:
        raise ValueError(f"--slice-count must be at least 1, got {count}")
    if not 0 <= index < count:
        raise ValueError(f"--slice-index must be in [0, {count}), got {index}")
    return list(rows[index::count])


def _duration(seconds: float) -> str:
    """A short human duration: ``45s``, ``12m``, ``3h05m``."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


class SliceLog:
    """One array task's row log, written a row at a time, with a progress line per row.

    The file is opened for writing when the slice starts, so a resubmitted slice begins an empty
    log: every row it owns is processed again, a finished one coming back ``present``, and the log
    ends as one record per row of this run. Each record is flushed and fsynced as it lands, so a
    running or crashed slice's finished rows are on disk.
    """

    def __init__(self, path: Path, *, slice_index: int, slice_count: int, total: int, workers: int = 1) -> None:
        """Open the log and truncate it.

        Args:
            path: The ``.jsonl`` row log.
            slice_index: This task's 0-based index, for the progress line.
            slice_count: How many tasks the array has.
            total: How many rows this task owns.
            workers: How many rows run at once, for the remaining-time estimate.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._prefix = f"[slice {slice_index}/{slice_count}]"
        self._total = total
        self._workers = max(1, workers)
        self._seconds: list[float] = []
        self._lock = threading.Lock()
        self._handle = path.open("w", encoding="utf-8")

    def add(self, record: dict[str, Any], seconds: float) -> None:
        """Append one row's record and print where the slice stands.

        Args:
            record: The row's outcome record.
            seconds: How long the row took.
        """
        with self._lock:
            self._handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._seconds.append(seconds)
            done = len(self._seconds)
            median = statistics.median(self._seconds)
            eta = _duration(median * (self._total - done) / self._workers)
            print(
                f"{self._prefix} {done}/{self._total} {record.get('status')} {seconds:.1f}s "
                f"(median {median:.1f}s, eta {eta})",
                flush=True,
            )

    def close(self) -> None:
        """Close the file."""
        self._handle.close()


def logged(
    rows: Sequence[dict[str, Any]],
    one: Callable[[dict[str, Any]], dict[str, Any]],
    log: SliceLog | None,
    *,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run ``one`` over each row, appending each outcome to ``log`` as it lands.

    Args:
        rows: The rows this task owns. Each row is handed to exactly one call of ``one``.
        one: Produces one row's outcome record; called from ``workers`` threads at once when
            ``workers`` is above 1.
        log: Where each record is appended, in the order rows finish, or None to only collect them.
        workers: How many rows run at once.

    Returns:
        One outcome record per row, in the order of ``rows``.

    Raises:
        Exception: Whatever ``one`` raised; rows not yet started are not run.
    """

    def timed(row: dict[str, Any]) -> tuple[dict[str, Any], float]:
        started = time.monotonic()
        return one(row), time.monotonic() - started

    if workers <= 1:
        out: list[dict[str, Any]] = []
        for row in rows:
            record, seconds = timed(row)
            out.append(record)
            if log is not None:
                log.add(record, seconds)
        return out
    held: list[dict[str, Any] | None] = [None] * len(rows)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(timed, row): index for index, row in enumerate(rows)}
        for future in as_completed(futures):
            record, seconds = future.result()
            held[futures[future]] = record
            if log is not None:
                log.add(record, seconds)
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [record for record in held if record is not None]


def batches(rows: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Split rows into fixed-size batches, the last one as short as it needs to be.

    Args:
        rows: The rows to split.
        size: Rows per batch.

    Yields:
        Each batch, in order.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError(f"--batch-size must be at least 1, got {size}")
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def run_root_of(stream_path: Path) -> Path:
    """The run root holding one recording's store, from the path of a stream inside it.

    Args:
        stream_path: ``<run_root>/run/streams/<stream>``.

    Returns:
        ``<run_root>``.

    Raises:
        ValueError: If the path is not that shape.
    """
    parents = stream_path.parents
    if len(parents) < 3 or parents[0].name != "streams" or parents[1].name != RUN_SUBDIR:
        raise ValueError(f"{stream_path} is not <run_root>/{RUN_SUBDIR}/streams/<stream>; no run root to extend")
    return parents[2]


def read_store(run_root: Path, *, run_id: str | None = None) -> ProvStore:
    """Read one run's store, by default under the run's own id so re-derived ids match the run's.

    Args:
        run_root: The run root.
        run_id: The id to read under. Defaults to the run root's own name. A replay passes
            :func:`replay_run_id` so that what it writes takes ids of its own.

    Returns:
        The store.

    Raises:
        FileNotFoundError: If the run holds no store.
    """
    store_path = run_root / RUN_SUBDIR / STORE_FILE
    if not store_path.is_file():
        raise FileNotFoundError(f"no store at {store_path}")
    return ProvStore.read_jsonl(store_path, run_id=run_id if run_id is not None else run_root.name)


def write_store(store: ProvStore, run_root: Path) -> Path:
    """Replace one run's store atomically, so a killed task never leaves a truncated one.

    Args:
        store: The merged store.
        run_root: The run root.

    Returns:
        The store's path.
    """
    store_path = run_root / RUN_SUBDIR / STORE_FILE
    partial = store_path.with_suffix(store_path.suffix + ".partial")
    store.write_jsonl(partial)
    partial.replace(store_path)
    return store_path


def export_prov(store: ProvStore, run_root: Path) -> list[Path]:
    """Re-export the BEP028 files from the merged store, so ``prov/`` agrees with it.

    Args:
        store: The merged store.
        run_root: The run root ``prov/`` is a directory of.

    Returns:
        The files written.
    """
    graph = to_bep028_graph(store, label=PROV_LABEL)
    return write_bep028_files(graph, run_root / PROV_SUBDIR, label=PROV_LABEL)


def supersede(store: ProvStore, entity_id: str, *, node: str, step: str, reason: str, software: str) -> str:
    """Retire one entity in favour of a replacement already written beside it.

    The store is append-only, so the record is not removed but marked ``wasInvalidatedBy`` an
    activity of its own — what ``live_entities`` and ``find_measurement`` filter on — separate from
    the activity that generated the replacement.

    Args:
        store: The provenance store.
        entity_id: The entity no longer to be read as what it was.
        node: The node whose output is being retired.
        step: The step name the retirement goes under.
        reason: Why the record is no longer current.
        software: The agent answerable for the retirement.

    Returns:
        The invalidating activity's id.
    """
    activity = store.activity(node=node, step=step, parameters={"superseded": entity_id, "reason": reason})
    store.was_associated_with(activity, software)
    store.used(activity, entity_id)
    store.was_invalidated_by(entity_id, activity)
    return activity


def rewrite_consensus_taxonomy(store: ProvStore, config: TriageConfig) -> str | None:
    """Recompute ``consensus_taxonomy`` from the stored per-span scores, retiring the older reading.

    Consolidates ``span_yamnet`` and ``span_hear``'s stored ``raw_scores``, so no model runs. A
    store already current is recognised by the recomputed entity id matching the stored one, and a
    store carrying no ``consensus_taxonomy`` at all is left alone rather than given one.

    Args:
        store: The finished run's store, read under the run's own id.
        config: The triage configuration, read for the consolidation floor and the ontology profile.

    Returns:
        The measurement's id when this call retired an older reading, or None when the store
        carries no ``consensus_taxonomy``, or already carries this exact consolidation.

    Raises:
        ValueError: If the configured classifier-ontology profile fails validation.
    """
    before = find_measurement(store, CONSENSUS_TAXONOMY)
    if before is None:
        return None
    software = software_agent(store)
    written, _, _ = _write_consensus_taxonomy(store, config, software)
    if not written:
        return None
    [after] = written
    if before.id == after:
        return None
    supersede(
        store,
        before.id,
        node=TAXONOMY_NODE,
        step=f"{CONSENSUS_TAXONOMY}_superseded",
        reason="consolidated by label string; the classifiers are merged on AudioSet node identity",
        software=software,
    )
    return after


def rebracket_words(store: ProvStore, config: TriageConfig) -> str | None:
    """Re-flag one finished run's consensus words against the vocabulary, retiring the old reading.

    :func:`~senselab.audio.workflows.triage.consensus.rebracket` re-evaluates ``bracketed_form``
    over each stored word's verbatim readings; nothing is re-aligned and no timing moves. A word
    that reads back unchanged keeps its id and stays live; one whose surface or flag moved is a new
    entity derived from the old one, and the old one is retired, as is the
    ``consensus_transcript`` listing them, rewritten with the new ids.

    The spans the ASR proposer contributed are not recomputed; the ``rebracket`` measurement names
    them and records that they were not.

    Args:
        store: The finished run's store, read under the run's own id.
        config: The triage configuration, read for ``words.onomatopoeic_tokens``.

    Returns:
        The rewritten transcript's id, or None when the store carries no ``consensus_transcript``
        or no word's reading moved.

    Raises:
        ValueError: If a word carries no extent, if its attributes are not the set this writer
            emits, or if a column does not read back as the alignment recorded it.
    """
    consensus = find_measurement(store, CONSENSUS_TRANSCRIPT)
    if consensus is None:
        return None
    onomatopoeic = {vocabulary_key(str(token)) for token in (config.get(ONOMATOPOEIC_TOKENS_KEY) or [])}
    n_sources = int(consensus.attributes["n_sources"])
    stored = [store.get_entity(str(word_id)) for word_id in consensus.attributes["word_ids"]]
    stored = [entity for entity in stored if not store.is_invalidated(entity.id)]
    reread: list[Rebracketed] = []
    for entity in stored:
        if entity.extent is None:
            raise ValueError(f"word {entity.id} carries no extent; it was not written by this graph")
        result = rebracket(
            word_from_attributes(entity.attributes, entity.extent), onomatopoeic=onomatopoeic, n_sources=n_sources
        )
        if set(word_attributes(result.word)) != set(entity.attributes):
            raise ValueError(
                f"word {entity.id} carries {sorted(entity.attributes)}, not "
                f"{sorted(word_attributes(result.word))}; re-flagging would rewrite fields this pass never read"
            )
        reread.append(result)
    moved = {entity.id for entity, result in zip(stored, reread) if word_attributes(result.word) != entity.attributes}
    if not moved:
        return None

    software = software_agent(store)
    activity = store.activity(
        node=PREPROCESS_NODE, step=REBRACKET, parameters={ONOMATOPOEIC_TOKENS_KEY: sorted(onomatopoeic)}
    )
    store.was_associated_with(activity, software)
    store.used(activity, consensus.id)
    word_ids: list[str] = []
    for entity, result in zip(stored, reread):
        if entity.id not in moved:
            word_ids.append(entity.id)
            continue
        word_id = store.entity(prov_type="word", extent=entity.extent, attributes=word_attributes(result.word))
        store.was_generated_by(word_id, activity)
        store.was_attributed_to(word_id, software)
        store.was_derived_from(word_id, entity.id)
        supersede(store, entity.id, node=PREPROCESS_NODE, step=WORD_SUPERSEDED, reason=_WORD_REASON, software=software)
        word_ids.append(word_id)

    words = [result.word for result in reread]
    signal = str(consensus.attributes["signal"])
    written = write_measurement(
        store,
        activity,
        software,
        name=CONSENSUS_TRANSCRIPT,
        signal=signal,
        attributes={
            **consensus.attributes,
            "word_ids": word_ids,
            "text": render_transcript(words, strong=("", "")),
            "bracket_overrides_n": sum(result.bracket_overrides for result in reread),
        },
        derived_from=(consensus.id,),
        extent=consensus.extent,
    )
    supersede(
        store,
        consensus.id,
        node=PREPROCESS_NODE,
        step=f"{CONSENSUS_TRANSCRIPT}_superseded",
        reason=_TRANSCRIPT_REASON,
        software=software,
    )
    write_measurement(
        store,
        activity,
        software,
        name=REBRACKET,
        signal=signal,
        attributes={
            "n_words": len(stored),
            "n_rebracketed": len(moved),
            "n_lexical_before": sum(1 for entity in stored if not entity.attributes["bracketed"]),
            "n_lexical_after": sum(1 for word in words if not word.bracketed),
            "superseded_consensus_transcript": consensus.id,
            "asr_proposed_span_ids": [
                span.id for span in live_entities(store, "span") if span.attributes.get("measure") == ASR_MEASURE
            ],
            "asr_proposed_spans_recomputed": False,
        },
        derived_from=(written,),
    )
    return written


def replay_run_id(run_root: Path, config_hash: str, commit: str | None = None) -> str:
    """The run id a replay of this run under this configuration and code writes its own records under.

    Args:
        run_root: The finished run root.
        config_hash: The replaying configuration's hash.
        commit: The replaying code revision, or None where none is recorded.

    Returns:
        The run id, distinct from the run root's own name and from any other configuration's or
        revision's.
    """
    suffix = "" if commit is None else f"-{commit[:12]}"
    return f"{run_root.name}+replay-{config_hash}{suffix}"


def find_replay_marker(store: ProvStore, config_hash: str, commit: str | None = None) -> str | None:
    """The marker a replay under this configuration and code already wrote into this store, if any.

    Args:
        store: The run's store.
        config_hash: The replaying configuration's hash.
        commit: The replaying code revision. None matches a marker of any revision.

    Returns:
        The marker activity's id, or None when this store has not been replayed under both.
    """
    for activity in store.activities(REPLAY_NODE):
        if activity.step != REPLAY_MARKER_STEP or activity.parameters.get("config_hash") != config_hash:
            continue
        if commit is None or activity.parameters.get("commit") == commit:
            return activity.id
    return None


def live_decisions(store: ProvStore, nodes: Sequence[str] = REPLAYED_NODES) -> list[str]:
    """Every live entity one of the named nodes generated.

    Args:
        store: The run's store.
        nodes: The nodes whose entities to collect. Defaults to every replayed node.

    Returns:
        The entity ids, in the store's own order.
    """
    wanted = set(nodes)
    found: list[str] = []
    for entity in store.entities():
        if store.is_invalidated(entity.id):
            continue
        activity_id = store.generated_by(entity.id)
        if activity_id is None:
            continue
        try:
            node = store.get_activity(activity_id).node
        except KeyError:
            continue
        if node in wanted:
            found.append(entity.id)
    return found


def retire_decisions(store: ProvStore, entity_ids: Sequence[str], *, software: str) -> list[str]:
    """Retire each decision a replay is about to supersede.

    Args:
        store: The run's store.
        entity_ids: The entities to retire, as :func:`live_decisions` found them.
        software: The software agent's id.

    Returns:
        The invalidating activities' ids.
    """
    return [
        supersede(
            store,
            entity_id,
            node=REPLAY_NODE,
            step=DECISION_SUPERSEDED,
            reason=_DECISION_REASON,
            software=software,
        )
        for entity_id in entity_ids
    ]


@dataclass(frozen=True)
class ReplayOutcome:
    """What one replay did to one finished run.

    Attributes:
        retired: How many live decisions the replay superseded.
        states: The run state of each replayed node, keyed by node name.
        errors: The nodes that raised and what they raised, keyed by node name.
        released: REDACT's released pair, empty unless it cleared one.
        summary: REPORT's products, empty when REPORT itself raised.
        marker: The marker activity's id.
        review: What became of the reading REVIEW held before the replay, one of :data:`REVIEW_STATES`.
        review_why: Why a reading could not be carried forward, or None.
    """

    retired: int
    states: dict[str, str]
    errors: dict[str, str]
    released: dict[str, Path]
    summary: dict[str, Path]
    marker: str
    review: str = REVIEW_NONE
    review_why: str | None = None


def load_hint_builder(directory: Path) -> Callable[[Path], Any]:
    """The hint populator from a directory holding ``hints.py``.

    Args:
        directory: The directory holding ``hints.py``.

    Returns:
        Its ``build_hint``, called with the recording's path.

    Raises:
        FileNotFoundError: If the directory holds no importable ``hints.py``.
    """
    spec = importlib.util.spec_from_file_location("extend_hints", directory / "hints.py")
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"no importable hints.py in {directory}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder: Callable[[Path], Any] = module.build_hint
    return builder


def source_of(run_root: Path) -> Path:
    """The recording a finished run was over, from its own ``run.json``.

    Args:
        run_root: The run root.

    Returns:
        The recording's path.

    Raises:
        FileNotFoundError: If the run holds no log.
        ValueError: If the log names no source.
    """
    log_path = run_root / RUN_SUBDIR / LOG_FILE
    if not log_path.is_file():
        raise FileNotFoundError(f"no run log at {log_path}")
    source = json.loads(log_path.read_text()).get("source")
    if not source:
        raise ValueError(f"{log_path} names no source")
    return Path(str(source))


@dataclass(frozen=True)
class RefoldOutcome:
    """What one re-fold did to one finished run.

    Attributes:
        retired: How many live VERDICT entities the re-fold superseded.
        triage: The triage axis the new fold reached.
        release: The release axis the new fold reached.
        summary: REPORT's products, empty when REPORT itself raised.
        errors: The nodes that raised and what they raised, keyed by node name.
        marker: The marker activity's id.
    """

    retired: int
    triage: str
    release: str
    summary: dict[str, Path]
    errors: dict[str, str]
    marker: str


def _settle(store: ProvStore, config: TriageConfig, *, run_dir: Path, artifacts_dir: Path) -> None:
    """Bring the release directory in line with the live fold, where one stands.

    Args:
        store: The provenance store, after VERDICT.
        config: The triage configuration, for ``redaction.bleep_hz``.
        run_dir: The run directory sidecar paths resolve against.
        artifacts_dir: The release directory.
    """
    folded = find_verdict(store, VERDICT_NODE)
    if folded is not None:
        ground = folded.attributes.get("release_ground")
        settle_release(
            store,
            str(folded.attributes.get("release") or ""),
            None if ground is None else str(ground),
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            bleep_hz=config.get("redaction.bleep_hz"),
        )


def refold_verdict(
    store: ProvStore,
    config: TriageConfig,
    hint: AudioHints | None,
    *,
    run_dir: Path,
    artifacts_dir: Path,
    summary_dir: Path,
    commit: str | None = None,
) -> RefoldOutcome:
    """Decide the file again over a store a driver has just added a measurement to, and re-render it.

    The narrow counterpart to :func:`replay_decisions`: every other node's verdict and every branch
    report stand, and only VERDICT's own conclusion is retired and taken again. A driver that adds
    an input VERDICT reads — REVIEW's ``redaction_llm_annotation`` is the one this was written
    for — must call this, or the store ends up holding a reading the recorded decision never saw.

    The hint is not optional in practice. VERDICT reads it to score each branch's declaration, and
    ``fold_file_verdict`` treats an unresolvable declaration as a flag ground of its own, so
    re-folding without one turns every recording's triage axis to ``flag``. The caller builds it the
    same way the replay driver does.

    Args:
        store: The finished run's store, already carrying whatever the driver added.
        config: The triage configuration.
        hint: What the recording was declared to contain.
        run_dir: The run directory sidecar paths resolve against.
        artifacts_dir: The release directory, settled to the new release by :func:`settle_release`.
        summary_dir: Where REPORT's products go.
        commit: The code revision the re-fold ran under, recorded on the marker.

    Returns:
        What the re-fold did, including the marker it wrote last.
    """
    software = software_agent(store)
    held = live_decisions(store, nodes=(VERDICT_NODE,))
    outcomes: dict[str, NodeOutcome] = {}
    result = _attempt(outcomes, VERDICT_NODE, lambda: verdict(store, None, config, hint, run_dir=run_dir))
    # Fold first, then retire everything the fold did not itself write. The fresh ids are the ones
    # the fold returned, never the store's latest by position: a conclusion equal to a retired one
    # is re-minted, and a positional lookup then hands back the decision it superseded.
    fresh = set() if result is None else {result.verdict_entity_id, result.ledger_entity_id}
    folded = None if result is None else store.get_entity(result.verdict_entity_id)
    retired = retire_decisions(store, [entity_id for entity_id in held if entity_id not in fresh], software=software)
    _settle(store, config, run_dir=run_dir, artifacts_dir=artifacts_dir)
    summary = _attempt_artifacts(outcomes, REPORT_NODE, lambda: report(store, summary_dir, config, run_dir=run_dir))
    # The marker says which configuration and commit folded this store, and deliberately not how
    # many entities this particular invocation retired: that count is 1 on a fold that moved and 0
    # on one that did not, so carrying it would mint a different activity for two equivalent folds
    # and no store would ever settle. The retirements are on the store's own edges either way.
    marker = store.activity(
        node=REFOLD_NODE,
        step=REFOLD_MARKER_STEP,
        parameters={"config_hash": config.config_hash, "commit": commit},
    )
    store.was_associated_with(marker, software)
    return RefoldOutcome(
        retired=len(retired),
        triage="" if folded is None else str(folded.attributes.get("outcome") or ""),
        release="" if folded is None else str(folded.attributes.get("release") or ""),
        summary=summary,
        errors={node: outcome.error for node, outcome in outcomes.items() if outcome.error is not None},
        marker=marker,
    )


@dataclass(frozen=True)
class HeldReading:
    """An answered REVIEW reading as it stood before a replay retired it.

    Attributes:
        annotation: The ``redaction_llm_annotation`` measurement.
        rounds: The per-round ``redaction_llm_review`` measurements its activity wrote.
    """

    annotation: Entity
    rounds: tuple[Entity, ...]


def held_reading(store: ProvStore) -> HeldReading | None:
    """The live answered REVIEW reading, with the rounds that produced it.

    Args:
        store: The run's store.

    Returns:
        The reading, or None where the live annotation is absent or recorded no answer.
    """
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    if annotation is None or annotation.attributes.get("status") not in READ_STATES:
        return None
    activity = store.generated_by(annotation.id)
    rounds = tuple(
        entity
        for entity in store.entities("measurement")
        if entity.attributes.get("name") == LLM_REVIEW_MEASUREMENT
        and not store.is_invalidated(entity.id)
        and store.generated_by(entity.id) == activity
    )
    return HeldReading(annotation=annotation, rounds=rounds)


def carry_reading_forward(
    store: ProvStore,
    config: TriageConfig,
    hint: AudioHints | None,
    held: HeldReading | None,
    *,
    software: str,
) -> tuple[str, str | None]:
    """Re-attach a reading a replay retired, where the replayed store holds exactly what it read.

    The comparison is the reading's own result-cache key against the key the replayed store's texts
    and task context give under the same commit. A match writes a REVIEW activity that used the held
    annotation, copies of its rounds and of the annotation derived from them, and retires the
    replay's own unanswered annotation. Anything else leaves the replay's annotation standing.

    Args:
        store: The replayed store, after REVIEW ran and before VERDICT.
        config: The replaying configuration.
        hint: What the recording was declared to contain.
        held: :func:`held_reading` as it was before the replay retired anything.
        software: The software agent's id.

    Returns:
        ``(state, why)``: one of :data:`REVIEW_STATES`, and the reason where it is
        :data:`REVIEW_NEEDS_REREAD`.
    """
    if held is None:
        return REVIEW_NONE, None
    fresh = find_measurement(store, REDACTION_LLM_ANNOTATION)
    if fresh is not None and fresh.id != held.annotation.id and fresh.attributes.get("status") in READ_STATES:
        return REVIEW_READ, None
    attributes = dict(held.annotation.attributes)
    held_key = (attributes.get("result_cache") or {}).get("key")
    revision = attributes.get("revision")
    if attributes.get("prompt_version") != PROMPT_VERSION:
        return REVIEW_NEEDS_REREAD, f"the reading is prompt version {attributes.get('prompt_version')!r}"
    if not held_key or not revision:
        return REVIEW_NEEDS_REREAD, "the reading records no cache key, so what it read cannot be compared"
    try:
        key = reading_key(store, config, hint, str(revision))
    except ValueError as error:
        return REVIEW_NEEDS_REREAD, describe_exception(error)
    if key != held_key:
        return REVIEW_NEEDS_REREAD, "the transcript texts or task context the reading read have changed"
    activity = store.activity(
        node=REVIEW_NODE,
        step=READING_CARRIED_STEP,
        parameters={"carried_from": held.annotation.id, "key": key, "prompt_version": PROMPT_VERSION},
    )
    store.was_associated_with(activity, software)
    store.was_associated_with(
        activity, store.agent(agent_type="model", model_id=str(attributes.get("model_id")), commit_sha=str(revision))
    )
    store.used(activity, held.annotation.id)
    for round_ in held.rounds:
        copied = store.entity(
            prov_type="measurement", extent=None, attributes={**round_.attributes, "carried_from": round_.id}
        )
        store.was_generated_by(copied, activity)
        store.was_attributed_to(copied, software)
        store.was_derived_from(copied, round_.id)
    carried = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={**attributes, "result_cache": {"key": key, "hit": False, "carried_from": held.annotation.id}},
    )
    store.was_generated_by(carried, activity)
    store.was_attributed_to(carried, software)
    store.was_derived_from(carried, held.annotation.id)
    if fresh is not None and fresh.id != held.annotation.id:
        supersede(
            store,
            fresh.id,
            node=REVIEW_NODE,
            step=READING_SUPERSEDED_STEP,
            reason=_READING_SUPERSEDED_REASON,
            software=software,
        )
    return REVIEW_CARRIED, None


def replay_decisions(
    store: ProvStore,
    config: TriageConfig,
    hint: AudioHints | None,
    *,
    run_dir: Path,
    artifacts_dir: Path,
    summary_dir: Path,
    enrollment: Enrollment | None = None,
    commit: str | None = None,
) -> ReplayOutcome:
    """Re-decide a finished run from TAXONOMY, over the PREPROCESS output its store already holds.

    Retires every live decision a replayed node made, then runs TAXONOMY through REDACT, VERDICT and
    REPORT against the store and the sidecars under ``run_dir``. An answered REVIEW reading the
    replay's own REVIEW did not answer again is carried forward by :func:`carry_reading_forward`
    where its inputs are unchanged, and reported as :data:`REVIEW_NEEDS_REREAD` where they are not.
    The store must have been read under :func:`replay_run_id`, so what the replay writes takes ids
    distinct from what it retires. The caller persists the result with :func:`write_store` and
    :func:`export_prov`.

    See ``specs/20260922-replay-decisions-over-a-finished-corpus/design.md``.

    Args:
        store: The finished run's store, read under :func:`replay_run_id`.
        config: The replaying configuration.
        hint: What the recording was declared to contain, rebuilt by the caller.
        run_dir: The run directory sidecar paths resolve against. Must hold ``derivatives/`` and a
            writable ``streams/``.
        artifacts_dir: The release directory handed to REDACT.
        summary_dir: Where REPORT's products go.
        enrollment: The target speaker's enrollment, when the caller has one.
        commit: The code revision the replay ran under, recorded on the marker.

    Returns:
        What the replay did, including the marker it wrote last.
    """
    software = software_agent(store)
    held = held_reading(store)
    retired = retire_decisions(store, live_decisions(store), software=software)
    outcomes: dict[str, NodeOutcome] = {}
    with record_venv_use() as used_venvs:
        released = drive_decisions(
            store,
            config,
            hint,
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            outcomes=outcomes,
            enrollment=enrollment,
        )
        review_state, review_why = carry_reading_forward(store, config, hint, held, software=software)
        ran = {node: outcome.state for node, outcome in outcomes.items()}
        _attempt(outcomes, "VERDICT", lambda: verdict(store, None, config, hint, run_dir=run_dir, ran=ran))
    _settle(store, config, run_dir=run_dir, artifacts_dir=artifacts_dir)
    capture_environments(store, used_venvs)
    summary = _attempt_artifacts(outcomes, REPORT_NODE, lambda: report(store, summary_dir, config, run_dir=run_dir))
    marker = store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={"config_hash": config.config_hash, "commit": commit, "retired": len(retired)},
    )
    store.was_associated_with(marker, software)
    return ReplayOutcome(
        retired=len(retired),
        states={node: outcome.state.value for node, outcome in outcomes.items()},
        errors={node: outcome.error for node, outcome in outcomes.items() if outcome.error is not None},
        released=dict(released),
        summary=dict(summary),
        marker=marker,
        review=review_state,
        review_why=review_why,
    )


def extend_quality(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> BranchResult | None:
    """Run QUALITY over a finished run, whose graph pass never reached it.

    QUALITY reads stored outputs only — PREPROCESS's clip spans over the ``recording`` stream and
    the ``clip_amplitude`` measurement beside them. Nothing is retired, and a store already
    carrying a live QUALITY verdict is left alone.

    Args:
        store: The finished run's store, read under the run's own id.
        config: The triage configuration, read for ``quality.clip_contradiction_margin``.
        run_dir: The run directory. QUALITY opens nothing under it.

    Returns:
        QUALITY's result, or None when the store already carries a live QUALITY verdict.

    Raises:
        ValueError: If ``quality.clip_contradiction_margin`` is unmeasured.
        LookupError: If the store holds no live ``recording`` stream, or holds clip spans over it
            with no ``clip_amplitude`` measurement to read them against.
    """
    if find_branch_report(store, QUALITY) is not None:
        return None
    return quality(store, SOURCE_STREAM, config, run_dir=run_dir)


def _retire_quality_findings(store: ProvStore, withdrawn: set[str], *, software: str) -> list[str]:
    """Retire QUALITY's contests of clip spans a caller has just withdrawn, and the verdict counting them.

    Args:
        store: The provenance store.
        withdrawn: The ids of the clip spans withdrawn in this pass.
        software: The agent answerable for the retirement.

    Returns:
        The retired contest assertions' ids; empty leaves the verdict alone.
    """
    contests = [
        entity
        for entity in live_entities(store, "assertion")
        if entity.attributes.get("verb") == CONTEST_VERB
        and entity.attributes.get("reason") == CONTRADICTED_CLIP
        and withdrawn.intersection(store.derived_from(entity.id))
    ]
    for contest in contests:
        supersede(
            store,
            contest.id,
            node=QUALITY,
            step=CLIP_CONTEST_SUPERSEDED,
            reason=_CLIP_CONTEST_REASON,
            software=software,
        )
    verdict = find_branch_report(store, QUALITY)
    if contests and verdict is not None:
        supersede(
            store,
            verdict.id,
            node=QUALITY,
            step=QUALITY_REPORT_SUPERSEDED,
            reason=_QUALITY_REPORT_REASON,
            software=software,
        )
    return [contest.id for contest in contests]


def withdraw_contradicted_clips(store: ProvStore, config: TriageConfig, *, signal: str = SOURCE_STREAM) -> str | None:
    """Retire a finished run's clip spans an unclipped sample of the same signal is louder than.

    The comparison is read from the stored ``clip_amplitude`` measurement rather than from samples:
    a span whose own level, keyed by its id under ``clip_levels``, falls below ``unclipped_peak`` by
    more than ``quality.clip_contradiction_margin`` of that level is withdrawn. A span the
    measurement carries no level for is left alone, no audio is opened, and no span but a clip span
    is touched.

    Each withdrawal is an ``assertion`` in
    :func:`~senselab.audio.workflows.triage.nodes.preprocess.write_withdrawn_clips`'s vocabulary,
    derived from the span it retires; the span is superseded, the ``clip_amplitude`` measurement is
    rewritten over the survivors and superseded, and QUALITY's ``contest`` assertions over a
    withdrawn span are superseded with it, as is the QUALITY verdict counting them. One round, not
    a fixpoint loop. See ``specs/20260912-quality-clip-consistency/design.md``.

    Args:
        store: The finished run's store, read under the run's own id.
        config: The triage configuration, read for ``quality.clip_contradiction_margin``.
        signal: The stream name the clip spans were detected on.

    Returns:
        The rewritten ``clip_amplitude`` measurement's id, or None — writing nothing — when the
        store carries no live clip span over ``signal``, or none the stored amplitudes contradict.

    Raises:
        ValueError: If ``quality.clip_contradiction_margin`` is unmeasured.
        LookupError: If the store carries clip spans over ``signal`` with no ``clip_amplitude``
            measurement stated over it.
    """
    spans = clip_spans(store, signal)
    if not spans:
        return None
    margin = float(config.require("quality.clip_contradiction_margin"))
    measurement = find_measurement(store, CLIP_AMPLITUDE_MEASUREMENT)
    if measurement is None or measurement.attributes.get("signal") != signal:
        raise LookupError(
            f"no live {CLIP_AMPLITUDE_MEASUREMENT!r} measurement over {signal!r}, but the store carries "
            "clip spans over it; this pass reads no audio. A finished run gains the measurement from "
            "scripts/extend_clip_amplitudes.py"
        )

    amplitudes = measurement.attributes
    levels = amplitudes.get(CLIP_LEVELS) or {}
    louder_counts = amplitudes.get(UNCLIPPED_LOUDER_N) or {}
    peak = amplitudes.get("unclipped_peak")
    peak_time_s = amplitudes.get("unclipped_peak_time_s")
    guard = int(amplitudes.get("edge_guard_samples") or 0)

    contradicted: list[tuple[str, WithdrawnClip]] = []
    if peak is not None and peak_time_s is not None:
        for span in spans:
            level, extent = levels.get(span.id), span.extent
            if level is None or extent is None or float(peak) <= float(level) * (1.0 + margin):
                continue
            start_s, end_s = extent
            contradicted.append(
                (
                    span.id,
                    WithdrawnClip(
                        extent=(float(start_s), float(end_s)),
                        clip_level=float(level),
                        louder_amplitude=float(peak),
                        louder_time_s=float(peak_time_s),
                        louder_samples_n=int(louder_counts.get(span.id, 0)),
                    ),
                )
            )
    if not contradicted:
        return None

    software = software_agent(store)
    activity = store.activity(
        node=PREPROCESS_NODE,
        step=WITHDRAW_CLIPS,
        parameters={
            "signal": signal,
            "clip_contradiction_margin": margin,
            "clip_edge_guard_samples": guard,
            "clip_spans_n": len(spans),
            "withdrawn_n": len(contradicted),
        },
    )
    store.was_associated_with(activity, software)
    store.used(activity, measurement.id)
    for span in spans:
        store.used(activity, span.id)

    for span_id, withdrawn in contradicted:
        write_withdrawn_clips(
            store,
            activity,
            software,
            withdrawn=(withdrawn,),
            signal=signal,
            margin=margin,
            derived_from=(span_id,),
        )
        supersede(
            store,
            span_id,
            node=PREPROCESS_NODE,
            step=CLIP_SPAN_SUPERSEDED,
            reason=_CLIP_SPAN_REASON,
            software=software,
        )

    retired = {span_id for span_id, _ in contradicted}
    _retire_quality_findings(store, retired, software=software)
    survivors = [span.id for span in spans if span.id not in retired]
    written = write_measurement(
        store,
        activity,
        software,
        name=CLIP_AMPLITUDE_MEASUREMENT,
        signal=signal,
        attributes={
            **amplitudes,
            "clip_spans_n": len(survivors),
            CLIP_LEVELS: {span_id: levels[span_id] for span_id in survivors if span_id in levels},
            UNCLIPPED_LOUDER_N: {span_id: louder_counts[span_id] for span_id in survivors if span_id in louder_counts},
        },
        derived_from=(measurement.id, *survivors),
        extent=measurement.extent,
    )
    supersede(
        store,
        measurement.id,
        node=PREPROCESS_NODE,
        step=f"{CLIP_AMPLITUDE_MEASUREMENT}_superseded",
        reason=_CLIP_AMPLITUDE_REASON,
        software=software,
    )
    return written
