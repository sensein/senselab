"""Reading, writing and re-exporting a finished triage run, for the drivers that extend one.

A run root holds ``run/store.jsonl`` and ``prov/``. Every extend driver needs the same four
operations over that layout — derive the root from a path inside it, read the store under the run's
own id, replace the store atomically, re-export BEP028 — and the layout is the workflow's, not any
one driver's, so it is stated once here rather than copied per script.

Manifest reading and slicing live here for the same reason: a Slurm array shards a JSONL the same
way whatever it is extending.

Supersession lives here too. A driver that recomputes a measurement the store already carries has to
retire the old one, or the store asserts two readings of the same thing; the store is append-only, so
retiring is an invalidation edge and never a deletion. The design is in
``specs/20260912-extend-reprocessed-outputs/design.md``.

So does the outcome vocabulary every driver records per derivation, and the rule that separates a
derivation which *cannot apply* to a recording from one that *failed*: :data:`UNAVAILABLE`,
:class:`DerivationOutcome` and :func:`attempt_derivation`. One rule, in one place, rather than a
per-derivation list of words a driver treats as determinate.
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
    NodeResult,
    describe_exception,
    find_measurement,
    find_verdict,
    live_entities,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.nodes.preprocess import NODE as PREPROCESS_NODE
from senselab.audio.workflows.triage.nodes.quality import quality
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
ONOMATOPOEIC_TOKENS_KEY = "words.onomatopoeic_tokens"
ASR_MEASURE = "asr"

SOURCE_STREAM = "recording"
"""The stream QUALITY's clip spans were detected on, and the one it names its findings about."""

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

The bare word is the first case. The second carries the typed absence that said so, as
``absent: <Class>: <message>``.
"""

UNAVAILABLE: tuple[type[BaseException], ...] = (
    AudioTooShortForAST,
    CrisperWhisperDecoderPositionsExceeded,
    F0RangeUnavailable,
    PpgsPosteriorgramUnavailable,
    SpanTooShortForYAMNet,
)
"""Every typed absence a derivation may raise: each says this recording has no such thing to derive.

A derivation raising one of these has answered; :func:`attempt_derivation` records it under
:data:`ABSENT` and the row's status stays what the other derivations made it.
"""


@dataclass(frozen=True)
class DerivationOutcome:
    """What one derivation did to one store, and whether anything failed.

    Attributes:
        detail: What the slice log records under the derivation's own name — one of the outcome
            words above, ``absent: <reason>``, or the failure's class and message.
        failed: Whether this derivation failed. A derivation that could not apply did not.
    """

    detail: str
    failed: bool


def attempt_derivation(call: Callable[[], str]) -> DerivationOutcome:
    """Run one derivation over a store, separating what cannot apply from what failed.

    Args:
        call: The derivation, already bound to its store and configuration, returning the outcome
            word to record for it.

    Returns:
        The outcome. A typed absence from :data:`UNAVAILABLE` is recorded under :data:`ABSENT` with
        its reason and is not a failure; any other ``OSError``, ``ValueError`` or ``LookupError`` is
        recorded with its class and message and is.
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

    The store is append-only, so a superseded record is never removed: it is marked
    ``wasInvalidatedBy`` an activity of its own, which is what ``live_entities`` and
    ``find_measurement`` filter on. The retiring activity is separate from the one that generated
    the replacement: the replacement's own activity records what its computation ran with, and a
    reader arriving at the retired record needs the reason instead.

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

    The measurement consolidates ``span_yamnet`` and ``span_hear``'s ``raw_scores``, both of which
    the store already holds, so no model runs. The current writer merges its classifiers by AudioSet
    node identity rather than by label string and names each row's own spellings, so a store written
    by the string-matching writer holds a different consolidation of the same scores.

    Whether a store is already current is decided by recomputing and comparing entity ids, not by
    probing the attributes for a key the new form happens to carry: an id is a digest over the
    attributes, so an equal id is the store already holding exactly this reading.

    A store carrying no ``consensus_taxonomy`` at all is left alone rather than given one. There is
    nothing to make current there, and a consolidation written beside none of TAXONOMY's other
    outputs would assert that the node ran.

    Args:
        store: The finished run's store, read under the run's own id.
        config: The triage configuration, read for the consolidation floor and the ontology profile.

    Returns:
        The measurement's id when this call retired an older reading, or None when the store carries
        no ``consensus_taxonomy`` to make current, or already carries this exact consolidation.

    Raises:
        ValueError: If the configured classifier-ontology profile fails validation.
    """
    before = find_measurement(store, CONSENSUS_TAXONOMY)
    if before is None:
        return None
    software = software_agent(store)
    written = _write_consensus_taxonomy(store, config, software)
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

    Each stored word carries every recognizer's own reading of its column verbatim, so
    :func:`~senselab.audio.workflows.triage.consensus.rebracket` re-evaluates ``bracketed_form``
    over those readings and returns the column re-read. Nothing is re-aligned and no timing moves:
    the alignment key a token groups on is invariant under bracketing.

    A word whose attributes read back unchanged keeps its id and stays live. A word whose surface or
    flag moved is a new entity derived from the old one, and the old one is retired. The
    ``consensus_transcript`` listing them is retired and rewritten with the new ids, its rendered
    text and its bracket-override count; every other field of it is the alignment's and is carried
    through verbatim.

    The spans the ASR proposer contributed are not recomputed. They keep their ``wasDerivedFrom``
    edge to the retired transcript, which is what says which reading proposed them, and the
    ``rebracket`` measurement names them and records that they were not.

    Args:
        store: The finished run's store, read under the run's own id.
        config: The triage configuration, read for ``words.onomatopoeic_tokens``.

    Returns:
        The rewritten transcript's id when this call re-flagged anything, or None when the store
        carries no ``consensus_transcript``, or no word's reading moved.

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


def extend_quality(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> NodeResult | None:
    """Run QUALITY over a finished run, whose graph pass never reached it.

    QUALITY reads stored outputs only, so the finished run holds every input it takes: PREPROCESS's
    clip spans over the ``recording`` stream and the ``clip_amplitude`` measurement beside them.
    Nothing is retired — the run carries no QUALITY verdict to replace — and the verdict written
    here names, in its ``preceded_by``, the nodes that had actually concluded when it was reached.

    A store already carrying a live QUALITY verdict is left alone. The recomputation would mint the
    activity, the assertions and the verdict it already holds, so a second pass is a set-union
    no-op either way; skipping is what makes it free.

    Args:
        store: The finished run's store, read under the run's own id.
        config: The triage configuration, read for ``quality.clip_contradiction_margin``.
        run_dir: The run directory, for the shared node shape. QUALITY opens nothing under it.

    Returns:
        QUALITY's result, or None when the store already carries a live QUALITY verdict.

    Raises:
        ValueError: If ``quality.clip_contradiction_margin`` is unmeasured.
        LookupError: If the store holds no live ``recording`` stream, or holds clip spans over it
            with no ``clip_amplitude`` measurement to read them against.
    """
    if find_verdict(store, QUALITY) is not None:
        return None
    return quality(store, SOURCE_STREAM, config, run_dir=run_dir)
