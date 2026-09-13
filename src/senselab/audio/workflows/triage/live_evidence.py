"""The live provenance store read as routing evidence, so the ruleset can run inside the graph.

``routing_analysis`` reduces a finished ``store.jsonl`` to a
:class:`~senselab.audio.workflows.triage.routing_analysis.features.RecordingFeatures` and evaluates
:mod:`~senselab.audio.workflows.triage.routing_analysis.ruleset` over it. TAXONOMY holds the same
store in memory, mid-run, with no file yet. This module is the one place the two meet: it hands the
live store to the store's own writer and the result to the analysis reader, so both paths reduce the
same bytes with the same code and no feature path has two definitions.

The evaluation carries no declaration. TAXONOMY reads no hint, and a BIDS stem's ``task-`` id is a
declaration, so ``task_id`` and ``family`` are empty here and
:attr:`~senselab.audio.workflows.triage.routing_analysis.ruleset.RouteEvaluation.declared` with
them: ``routed``, ``state``, ``unavailable``, ``flags`` and ``gate_outcomes`` are the fields this
path fills.

``specs/20260912-ruleset-in-pipeline/design.md`` holds the reasoning: why the serialisation rather
than a second reader, what it costs, and what stage 2 replaces it with.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.label_membership import LabelMembership
from senselab.audio.workflows.triage.routing_analysis.features import (
    RecordingFeatures,
    extract_features,
    onomatopoeic_vocabulary,
    span_label_memberships,
)
from senselab.audio.workflows.triage.routing_analysis.ruleset import (
    RouteEvaluation,
    Ruleset,
    evaluate_routes,
    load_ruleset,
)
from senselab.utils.prov_store import ProvStore

EVIDENCE_PREFIX = "routing-evidence-"
"""Prefix of the transient store serialisation, written beside the run's own store and removed."""

EVIDENCE_SUFFIX = ".jsonl"
"""Suffix of that serialisation, so the reader opens what it expects."""

RECORDING_STREAM = "recording"
"""The stream entity ADMIT writes, whose path names the recording."""

UNDECLARED = ""
"""The task id and family of an evaluation that read no declaration, which every one here did."""

EMPTINESS_SOURCE = "stream_peak_max"
"""The feature source the emptiness bypass reads, which no gate names and every evaluation uses."""


def required_sources(ruleset: Ruleset) -> tuple[str, ...]:
    """Every feature source one ruleset reads, in sorted order.

    The gates are declarative, so what a given configuration consumes is knowable without reading
    the whole feature surface. A caller that must narrow the reduction reads this rather than
    enumerating :class:`~senselab.audio.workflows.triage.routing_analysis.features.RecordingFeatures`.

    Args:
        ruleset: The loaded ruleset.

    Returns:
        The first element of every gate's feature path, plus :data:`EMPTINESS_SOURCE`, each once.
    """
    sources = {str(gate.feature[0]) for gate in ruleset.gates.values()}
    sources.add(EMPTINESS_SOURCE)
    return tuple(sorted(sources))


def recording_stem(store: ProvStore) -> str:
    """The recording's file stem, off the ``recording`` stream entity ADMIT wrote.

    Args:
        store: The provenance store.

    Returns:
        The stem of the path the latest live ``recording`` stream names, or ``""`` when no such
        stream is in the store. The stem is an identifier only; the ``task-`` id it carries is a
        declaration and is not read into the evaluation.
    """
    found = [
        entity
        for entity in store.entities("stream")
        if entity.attributes.get("name") == RECORDING_STREAM and not store.is_invalidated(entity.id)
    ]
    if not found:
        return ""
    return Path(str(found[-1].attributes.get("path") or "")).stem


def read_live_features(
    store: ProvStore,
    *,
    run_dir: Path,
    memberships: Mapping[str, LabelMembership],
    onomatopoeic: frozenset[str],
    stem: str,
) -> RecordingFeatures:
    """Reduce a live store to the routing evidence, through the analysis reader.

    The serialisation is written under ``run_dir`` and not elsewhere: a measurement naming a sidecar
    names it relative to the store's own directory, and the reader resolves it against the file it
    was handed. It is removed whether or not the reduction succeeded.

    Args:
        store: The provenance store, as the run holds it.
        run_dir: The run directory the store's sidecar paths are relative to.
        memberships: Which labels a span carries, per classifier, from
            :func:`~senselab.audio.workflows.triage.routing_analysis.features.span_label_memberships`.
        onomatopoeic: The token vocabulary a consensus word is matched against, from
            :func:`~senselab.audio.workflows.triage.routing_analysis.features.onomatopoeic_vocabulary`.
        stem: The recording's stem, for the record's own id.

    Returns:
        The evidence record, carrying no declaration: ``task_id`` and ``family`` are
        :data:`UNDECLARED`.
    """
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 — closed below; the path outlives the handle
        dir=run_dir, prefix=EVIDENCE_PREFIX, suffix=EVIDENCE_SUFFIX, delete=False
    )
    handle.close()
    path = Path(handle.name)
    try:
        store.write_jsonl(path)
        return extract_features(
            path,
            stem=stem,
            run_root=str(run_dir.parent),
            task_id=UNDECLARED,
            family=UNDECLARED,
            memberships=memberships,
            onomatopoeic=onomatopoeic,
        )
    finally:
        path.unlink(missing_ok=True)


def evaluate_live_routes(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> RouteEvaluation:
    """Route one recording from the store the graph is still writing.

    Args:
        store: The provenance store, holding PREPROCESS's derivatives and TAXONOMY's own summaries.
            Every gate's evidence is written by one of those two, so this is callable from the end
            of TAXONOMY and not before it.
        config: The resolved triage configuration, read for ``taxonomy.ruleset`` and for the
            ``windows.<classifier>`` membership rule.
        run_dir: The run directory the store's sidecar paths are relative to.

    Returns:
        The evaluation.

    Raises:
        ValueError: When the ruleset or the membership rule cannot be loaded from the configuration.
    """
    ruleset = load_ruleset(config)
    features = read_live_features(
        store,
        run_dir=run_dir,
        memberships=span_label_memberships(config),
        onomatopoeic=onomatopoeic_vocabulary(config),
        stem=recording_stem(store),
    )
    return evaluate_routes(features, ruleset)


def route_attributes(evaluation: RouteEvaluation, ruleset: Ruleset) -> dict[str, Any]:
    """One evaluation as the attributes of the measurement that records it.

    Args:
        evaluation: What the ruleset made of the recording.
        ruleset: The ruleset it was evaluated under, for the sources it reads.

    Returns:
        The attributes. ``authoritative`` is False for as long as ``kind_state`` decides what runs;
        ``error`` is None, and the failed shape :func:`failed_route_attributes` returns carries the
        same keys so one reader handles both.
    """
    return {
        "authoritative": False,
        "error": None,
        "state": evaluation.state.value,
        "routed": list(evaluation.routed),
        "gate_outcomes": {name: outcome.value for name, outcome in evaluation.gate_outcomes.items()},
        "unavailable": {branch: list(names) for branch, names in evaluation.unavailable.items()},
        "flags": {branch: list(names) for branch, names in evaluation.flags.items()},
        "sources": list(required_sources(ruleset)),
        "stem": evaluation.stem,
    }


def failed_route_attributes(error: str) -> dict[str, Any]:
    """The same attributes for an evaluation that could not be made.

    Args:
        error: The failure, as :func:`~senselab.audio.workflows.triage.nodes.common.describe_exception`
            renders it.

    Returns:
        The attributes, with ``state`` None and ``error`` set. Those two are the discriminator: a
        reader that finds a null ``state`` has no routing to compare, not a recording nothing routed.
    """
    return {
        "authoritative": False,
        "error": error,
        "state": None,
        "routed": [],
        "gate_outcomes": {},
        "unavailable": {},
        "flags": {},
        "sources": [],
        "stem": "",
    }
