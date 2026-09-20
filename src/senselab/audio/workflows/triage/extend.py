"""Reading, writing and re-exporting a finished triage run, for the drivers that extend one.

A run root holds ``run/store.jsonl`` and ``prov/``. This module carries the four operations every
extend driver needs over that layout — derive the root from a path inside it, read the store under
the run's own id, replace the store atomically, re-export BEP028 — plus manifest reading and
slicing for a Slurm array.

Supersession is here too: a driver recomputing a measurement the store already carries retires the
old one with an invalidation edge, never a deletion. So is the outcome vocabulary a driver records
per derivation, and the rule separating a derivation that *cannot apply* from one that *failed* —
:data:`UNAVAILABLE`, :class:`DerivationOutcome` and :func:`attempt_derivation`.

See ``specs/20260912-extend-reprocessed-outputs/design.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

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
from senselab.audio.workflows.triage.nodes.common import (
    BranchResult,
    describe_exception,
    find_branch_report,
    find_measurement,
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
from senselab.audio.workflows.triage.nodes.taxonomy import NODE as TAXONOMY_NODE
from senselab.audio.workflows.triage.nodes.taxonomy import _write_consensus_taxonomy
from senselab.audio.workflows.triage.vocabulary import QUALITY
from senselab.utils.prov_bep028 import to_bep028_graph, write_bep028_files
from senselab.utils.prov_store import ProvStore

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


def read_store(run_root: Path) -> ProvStore:
    """Read one run's store under the run's own id, so re-derived entity ids match the run's.

    Args:
        run_root: The run root.

    Returns:
        The store, with ``run_id`` set to the run root's own name.

    Raises:
        FileNotFoundError: If the run holds no store.
    """
    store_path = run_root / RUN_SUBDIR / STORE_FILE
    if not store_path.is_file():
        raise FileNotFoundError(f"no store at {store_path}")
    return ProvStore.read_jsonl(store_path, run_id=run_root.name)


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
