"""TAXONOMY — what the classifiers say is in the recording, consolidated from PREPROCESS's scores.

It runs no model, reads no hint, localises nothing and decides nothing. It writes two measurements
per classifier that produced scores: the whole-file label-score distribution, with no threshold
applied, and the file-level ``consensus_taxonomy`` that resolves every classifier's per-span labels
onto the AudioSet ontology node each denotes. ROUTING, FIGURE and REPORT read them.

It measures; ``routing`` decides. ``specs/20260912-ruleset-in-pipeline/design.md`` records what this
node used to fold and why that fold was removed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.workflows.triage.classifier_ontology import (
    PROFILE_PATH_KEY,
    canonical_names,
    resolved_profile_path,
)
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    find_measurements,
    software_agent,
    write_measurement,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

NODE = "TAXONOMY"

SUMMARISED_CLASSIFIERS = ("yamnet", "ast", "hear")


@dataclass(frozen=True)
class TaxonomyResult(NodeResult):
    """TAXONOMY's result.

    Attributes:
        classifiers: The classifiers whose per-span scores reached the consolidation, in name order.
        n_labels: How many ontology nodes the consolidation kept.
    """

    classifiers: tuple[str, ...]
    n_labels: int


def _label_score_distribution(store: ProvStore, classifier: str, run_dir: Path) -> dict[str, Any] | None:
    """Every label's score distribution over the whole recording, with no threshold applied.

    Args:
        store: The provenance store, holding PREPROCESS's verbatim score measurement.
        classifier: ``"yamnet"``, ``"ast"`` or ``"hear"``.
        run_dir: Where PREPROCESS wrote the classifier's windows.

    Returns:
        ``{n_windows, win_length_s, hop_s, labels, element_id}`` where ``labels`` maps a label to
        ``{peak, median, n_windows}``, ordered by descending peak; or None when the measurement or
        its sidecar is absent.
    """
    raw = find_measurement(store, f"{classifier}_scores")
    if raw is None:
        return None
    path = raw.attributes.get("path")
    if not path:
        return None
    sidecar = run_dir / str(path)
    if not sidecar.is_file():
        return None
    windows = json.loads(sidecar.read_text())
    per_label: dict[str, list[float]] = {}
    for window in windows:
        for pair in label_scores(window):
            for label, score in pair.items():
                per_label.setdefault(str(label), []).append(float(score))
    labels = {
        label: {"peak": float(max(scores)), "median": float(np.median(scores)), "n_windows": len(scores)}
        for label, scores in per_label.items()
    }
    return {
        "n_windows": len(windows),
        "win_length_s": raw.attributes.get("win_length_s"),
        "hop_s": raw.attributes.get("hop_s"),
        "labels": dict(sorted(labels.items(), key=lambda item: -item[1]["peak"])),
        "element_id": raw.id,
    }


PER_SPAN_CLASSIFIERS: dict[str, str] = {"yamnet": "span_yamnet", "hear": "span_hear"}
"""Each per-span classifier and the measurement PREPROCESS writes for it."""


def _per_span_label_scores(store: ProvStore, measurement_name: str) -> dict[str, dict[str, float]]:
    """One classifier's per-span label scores, reduced to the best score per label in each span.

    Args:
        store: The provenance store.
        measurement_name: ``"span_yamnet"`` or ``"span_hear"``.

    Returns:
        ``{span_id: {label: score}}``, read from ``raw_scores`` — the model's own output, written
        whatever the configuration says. No labelling threshold takes part.
    """
    by_span: dict[str, dict[str, float]] = {}
    for measurement in find_measurements(store, measurement_name):
        span_id = measurement.attributes.get("span_id")
        if span_id is None:
            continue
        slot = by_span.setdefault(str(span_id), {})
        for label, score in (measurement.attributes.get("raw_scores") or {}).items():
            slot[str(label)] = max(slot.get(str(label), 0.0), float(score))
    return by_span


def _on_ontology_nodes(
    per_span: dict[str, dict[str, float]], identity: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    """Rewrite each span's labels onto the ontology node each denotes, best score per node.

    Args:
        per_span: :func:`_per_span_label_scores`' result, in the classifier's own vocabulary.
        identity: :func:`~senselab.audio.workflows.triage.classifier_ontology.canonical_names`'
            map. A label it does not name keeps its own spelling.

    Returns:
        ``{span_id: {AudioSet display name: score}}``, the two spellings of one node folded to one
        entry carrying the higher score.
    """
    resolved: dict[str, dict[str, float]] = {}
    for span_id, scores in per_span.items():
        slot = resolved.setdefault(span_id, {})
        for label, score in scores.items():
            name = identity.get(label, label)
            slot[name] = max(slot.get(name, 0.0), float(score))
    return resolved


def _spellings_by_node(per_span: dict[str, dict[str, float]], identity: Mapping[str, str]) -> dict[str, list[str]]:
    """Which of one classifier's own label spellings reached each ontology node.

    Args:
        per_span: :func:`_per_span_label_scores`' result, in the classifier's own vocabulary.
        identity: The same map :func:`_on_ontology_nodes` reads.

    Returns:
        ``{AudioSet display name: the classifier's own spellings}``, each list sorted.
    """
    gathered: dict[str, set[str]] = {}
    for scores in per_span.values():
        for label in scores:
            gathered.setdefault(identity.get(label, label), set()).add(label)
    return {name: sorted(spellings) for name, spellings in gathered.items()}


def _consolidate(per_span: dict[str, dict[str, float]], floor: float | None) -> dict[str, dict[str, float]]:
    """One classifier's per-span scores consolidated to one row per label over the whole file.

    Args:
        per_span: :func:`_per_span_label_scores`' result.
        floor: A label whose peak falls under this is dropped, or ``None`` to keep every label.

    Returns:
        ``{label: {peak, median, n_spans}}``, over the spans that carry the label.
    """
    gathered: dict[str, list[float]] = {}
    for scores in per_span.values():
        for label, score in scores.items():
            gathered.setdefault(label, []).append(float(score))
    consolidated: dict[str, dict[str, float]] = {}
    for label, across_spans in gathered.items():
        peak = max(across_spans)
        if floor is not None and peak < floor:
            continue
        consolidated[label] = {
            "peak": peak,
            "median": float(np.median(across_spans)),
            "n_spans": float(len(across_spans)),
        }
    return consolidated


def _write_consensus_taxonomy(
    store: ProvStore, config: TriageConfig, software: str
) -> tuple[list[str], tuple[str, ...], int]:
    """Consolidate the per-span labels into one file-level taxonomy, for downstream to read.

    Every classifier's labels are resolved onto the AudioSet ontology node each denotes before they
    are consolidated, so a row is one node rather than one spelling: HeAR ``Throat Clear`` and
    YAMNet ``Throat clearing`` are one row reaching ``n_classifiers: 2``, and the row is named by
    the ontology. A label the profile does not name keeps its own spelling and its own row.

    Resolution is onto the label's mapped node alone, never its subtree, so HeAR ``Cough`` and
    YAMNet ``Throat clearing`` stay two rows. Two of one classifier's own labels landing on one node
    — HeAR ``Cough`` and ``Baby Cough`` both denote AudioSet ``Cough`` — contribute one entry to
    ``peak_by_classifier`` and are both named in ``labels_by_classifier``.

    Args:
        store: The provenance store.
        config: The run's configuration, read for the consolidation floor and the ontology profile.
        software: This node's software agent.

    Returns:
        The ids written, the classifiers that contributed, and how many ontology nodes the
        consolidation kept. The ids and the classifiers are empty when no per-span classifier
        produced scores, so "no consensus" and "a consensus over nothing" stay distinguishable.

    Raises:
        ValueError: If the configured classifier-ontology profile fails validation.
    """
    floor = config.get("taxonomy.consolidation_floor")
    consolidation_floor = None if floor is None else float(floor)
    identity = canonical_names(config.get(PROFILE_PATH_KEY))
    by_classifier: dict[str, dict[str, dict[str, float]]] = {}
    spellings: dict[str, dict[str, list[str]]] = {}
    read_ids: list[str] = []
    for classifier, measurement_name in PER_SPAN_CLASSIFIERS.items():
        measurements = list(find_measurements(store, measurement_name))
        if not measurements:
            continue
        read_ids.extend(measurement.id for measurement in measurements)
        per_span = _per_span_label_scores(store, measurement_name)
        spellings[classifier] = _spellings_by_node(per_span, identity)
        by_classifier[classifier] = _consolidate(_on_ontology_nodes(per_span, identity), consolidation_floor)
    if not by_classifier:
        return [], (), 0

    labels: dict[str, dict[str, Any]] = {}
    for classifier, consolidated in by_classifier.items():
        for label, stats in consolidated.items():
            row = labels.setdefault(label, {"label": label, "classifiers": {}, "labels_by_classifier": {}})
            row["classifiers"][classifier] = stats
            row["labels_by_classifier"][classifier] = spellings[classifier][label]
    ranked = sorted(
        labels.values(),
        key=lambda row: (-max(float(s["peak"]) for s in row["classifiers"].values()), str(row["label"])),
    )
    for row in ranked:
        peaks = {name: float(stats["peak"]) for name, stats in row["classifiers"].items()}
        row["peak"] = max(peaks.values())
        row["peak_by_classifier"] = peaks
        row["n_classifiers"] = len(peaks)
        row["classifiers"] = sorted(peaks)
        row["labels_by_classifier"] = dict(sorted(row["labels_by_classifier"].items()))

    activity = store.activity(
        node=NODE,
        step="consensus_taxonomy",
        parameters={
            "consolidation_floor": consolidation_floor,
            "classifiers": sorted(by_classifier),
            "ontology_profile": str(resolved_profile_path(config.get(PROFILE_PATH_KEY)).name),
        },
    )
    store.was_associated_with(activity, software)
    for element_id in read_ids:
        store.used(activity, element_id)
    written = write_measurement(
        store,
        activity,
        software,
        name="consensus_taxonomy",
        signal="plain",
        attributes={
            "labels": ranked,
            "n_labels": len(ranked),
            "classifiers": sorted(by_classifier),
            "consolidation_floor": consolidation_floor,
        },
        derived_from=tuple(read_ids),
        extent=None,
    )
    return [written], tuple(sorted(by_classifier)), len(ranked)


def _write_label_summaries(store: ProvStore, run_dir: Path, software: str) -> list[str]:
    """Write one whole-file label-score summary per classifier that produced scores.

    Args:
        store: The provenance store.
        run_dir: Where PREPROCESS wrote the score sidecars.
        software: This node's software agent.

    Returns:
        The ids written, for the node's view. A classifier whose scores are absent contributes
        nothing rather than an empty summary, so a missing summary and an all-zero one stay
        distinguishable.
    """
    written: list[str] = []
    for classifier in SUMMARISED_CLASSIFIERS:
        distribution = _label_score_distribution(store, classifier, run_dir)
        if distribution is None:
            continue
        element_id = str(distribution.pop("element_id"))
        activity = store.activity(node=NODE, step=f"{classifier}_label_summary", parameters={"classifier": classifier})
        store.was_associated_with(activity, software)
        store.used(activity, element_id)
        written.append(
            write_measurement(
                store,
                activity,
                software,
                name=f"{classifier}_label_summary",
                signal="plain",
                attributes={"classifier": classifier, **distribution},
                derived_from=(element_id,),
                extent=None,
            )
        )
    return written


def taxonomy(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> TaxonomyResult:
    """Consolidate what the classifiers said about the recording, from the store alone.

    Args:
        store: The provenance store, holding PREPROCESS's derivatives.
        source: The stream every element it writes names, ``"plain"``.
        config: The triage configuration, read for the consolidation floor and the ontology profile.
        hint: Accepted for the shared node shape and **not read**. A measurement that reads the
            declaration cannot disagree with it.
        run_dir: Where PREPROCESS wrote each classifier's verbatim
            ``derivatives/<classifier>_scores.json`` — the sidecars this node reads. It writes none
            of its own.

    Returns:
        The verdict, each classifier's whole-file label summary and the consensus taxonomy as the
        view, and what the consolidation had to work from.
    """
    software = software_agent(store)
    view = _write_label_summaries(store, run_dir, software)
    consensus, classifiers, n_labels = _write_consensus_taxonomy(store, config, software)
    view += consensus

    activity = store.activity(node=NODE, step="conclude", parameters={"stream": source})
    store.was_associated_with(activity, software)
    for element_id in view:
        store.used(activity, element_id)

    if classifiers:
        outcome = Outcome.PASS
        why = f"consolidated {n_labels} label(s) from " + ", ".join(classifiers)
    else:
        outcome = Outcome.FLAG
        why = "no per-span classifier produced scores; there was nothing to consolidate"

    verdict_id, verdict = write_verdict(
        store,
        activity,
        software,
        node=NODE,
        outcome=outcome,
        kind=None,
        why=why,
        detail={"classifiers": list(classifiers), "n_labels": n_labels},
    )
    view.append(verdict_id)
    return TaxonomyResult(
        verdict=verdict,
        view=tuple(view),
        verdict_entity_id=verdict_id,
        classifiers=classifiers,
        n_labels=n_labels,
    )
