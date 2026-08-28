"""AIRWAY — re-evaluate PREPROCESS's candidate spans with HeAR, then confirm or contest them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.health_acoustics.hear import (
    HEAR_MODEL_ID,
    HEAR_REVISION,
    HEAR_WINDOW_SECONDS,
    span_to_hear_buffer,
)
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    find_measurements,
    live_entities,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "AIRWAY"


def _inside_certified_silence(span: Entity, silence_windows: list[dict[str, Any]] | None) -> bool | None:
    """Whether every silence-graded window overlapping the span was certified silent.

    Args:
        span: The span being labelled.
        silence_windows: PREPROCESS's graded windows, or None when it graded none.

    Returns:
        True or False when at least one graded window overlaps, and None when the question has no
        answer here.
    """
    if silence_windows is None:
        return None
    start, end = span.extent or (0.0, 0.0)
    overlapping = [w for w in silence_windows if float(w["start"]) < end and float(w["end"]) > start]
    if not overlapping:
        return None
    return all(bool(w["is_silence"]) for w in overlapping)


def _windows_covering(store: ProvStore, classifier: str, extent: tuple[float, float]) -> list[Entity]:
    """Every one of this classifier's stored windows overlapping the extent, oldest first.

    Args:
        store: The provenance store.
        classifier: ``"hear"`` or ``"yamnet"``.
        extent: The extent to cover.

    Returns:
        The per-window measurement entities PREPROCESS wrote, filtered to those that overlap.
    """
    return [
        window
        for window in find_measurements(store, f"{classifier}_window")
        if window.extent is not None and window.extent[0] < extent[1] and window.extent[1] > extent[0]
    ]


def _confident_labels(window: dict[str, Any], default_threshold: float, thresholds: dict[str, float]) -> list[str]:
    """Return the labels whose individual scores clear the configured threshold."""
    labels: list[str] = []
    for pair in window["label_scores"]:
        for label, score in pair.items():
            if float(score) >= float(thresholds.get(str(label), default_threshold)):
                labels.append(str(label))
    return labels


def _span_hear_input(audio: Audio, extent: tuple[float, float]) -> Audio:
    """Return an isolated candidate for HeAR, preserving a long candidate's internal windows."""
    start, end = extent
    if end - start <= HEAR_WINDOW_SECONDS:
        return span_to_hear_buffer(audio, start, end)
    start_sample = int(round(start * audio.sampling_rate))
    end_sample = int(round(end * audio.sampling_rate))
    return Audio(waveform=audio.waveform[..., start_sample:end_sample].clone(), sampling_rate=audio.sampling_rate)


def _hear_window_extent(
    candidate_extent: tuple[float, float], raw_window: dict[str, Any]
) -> tuple[float, float]:
    """Place a native detector window on the recording timeline.

    A short candidate is embedded in an isolated two-second buffer, so its only detector result
    describes the candidate itself. A long candidate is passed through unchanged and HeAR returns
    one or more native windows relative to that candidate; those windows must be offset to the
    source recording rather than all being written over the parent span.
    """
    start, end = candidate_extent
    if end - start <= HEAR_WINDOW_SECONDS:
        return candidate_extent
    return start + float(raw_window["start"]), start + float(raw_window["end"])


def _is_transcribed(store: ProvStore, extent: tuple[float, float]) -> bool:
    """Whether a consensus word overlaps this span, which makes it transcribed content.

    An ``event`` entity — a bracketed or onomatopoeic non-word — does not make a span transcribed.

    Args:
        store: The provenance store.
        extent: The span's extent.

    Returns:
        True when at least one live ``word`` entity overlaps.
    """
    return any(
        word.extent is not None and word.extent[0] < extent[1] and word.extent[1] > extent[0]
        for word in live_entities(store, "word")
    )


def _labels_of(window: Entity) -> list[str]:
    """The label set a stored window carries.

    Args:
        window: A ``<classifier>_window`` measurement entity.

    Returns:
        The labels, as strings; empty when the window retained none.
    """
    return [str(label) for label in window.attributes.get("labels") or []]


def _gate(config: TriageConfig, hint: AudioHints | None) -> tuple[float, float | None]:
    """Resolve this branch's K and its near-gate band.

    Args:
        config: The triage configuration.
        hint: What the recording was declared to contain; its ``metadata["task"]`` selects the gate.

    Returns:
        The K in dB and the near-gate band in dB, which is None while unmeasured.
    """
    task = str(hint.metadata.get("task")) if hint is not None and hint.metadata.get("task") else None
    by_task = config.get("airway.k_db_by_task") or {}
    for_task = by_task.get(task) if task is not None else None
    branch = config.get("airway.k_db", config.require("spans.k_db.airway"))
    k_db = float(branch if for_task is None else for_task)
    band = config.get("airway.k_margin_db")
    return k_db, None if band is None else float(band)


def _contest_labels(config: TriageConfig) -> set[str]:
    """The YAMNet labels that may contest a HeAR label, refused when they are also airway evidence.

    Args:
        config: The triage configuration.

    Returns:
        The declared contest labels; empty while the key is null.

    Raises:
        ValueError: If the declared set intersects ``taxonomy.audioset_airway_labels``.
    """
    contest_labels = {str(label) for label in (config.get("airway.contest_labels") or [])}
    airway_evidence = {str(label) for label in config.require("taxonomy.audioset_airway_labels")}
    overlap = contest_labels & airway_evidence
    if overlap:
        raise ValueError(
            f"airway.contest_labels and taxonomy.audioset_airway_labels must be disjoint; "
            f"{sorted(overlap)} appear in both, so the same label would be airway evidence and a "
            "contest of airway evidence"
        )
    return contest_labels


def airway(  # noqa: C901 — the branch's four steps, in order
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> NodeResult:
    """Label, confirm and contest the spans PREPROCESS proposed at the airway K.

    Args:
        store: The provenance store, holding PREPROCESS's spans and window classifications.
        source: The store-held stream the spans were proposed over, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain; read for the task's gate only. A
            declaration this branch's measurements contradict is named by VERDICT's fold.
        run_dir: The run directory the stream sidecar is relative to.

    Returns:
        The verdict, the view over the spans and assertions touched, and a null figure path.

    Raises:
        ValueError: If ``airway.contest_labels`` intersects ``taxonomy.audioset_airway_labels``.
    """
    labels_of_interest = [str(label) for label in config.require("airway.labels_of_interest")]
    confirmation_map = {
        str(hear_label): {str(v) for v in yamnet_labels}
        for hear_label, yamnet_labels in config.require("airway.confirmation_map").items()
    }
    contest_labels = _contest_labels(config)
    k_db, k_margin_db = _gate(config, hint)

    software = software_agent(store)
    stream_id, audio = resolve_stream(store, run_dir, source)

    spans = [e for e in live_entities(store, "span") if e.attributes.get("k_db") == k_db]
    spans.sort(key=lambda e: e.extent or (0.0, 0.0))
    silence = find_measurement(store, "silence")
    silence_windows = silence.attributes.get("windows") if silence is not None else None

    if not spans:
        no_contrast = find_measurement(store, "spans_no_contrast")
        at_this_k = no_contrast is not None and no_contrast.attributes.get("k_db") == k_db
        reason = "PREPROCESS reported no_contrast at this K" if at_this_k else "no span was proposed at this K"
        activity = store.activity(node=NODE, step="classify", parameters={"k_db": k_db, "n_spans": 0})
        store.was_associated_with(activity, software)
        store.used(activity, stream_id)
        if at_this_k and no_contrast is not None:
            store.used(activity, no_contrast.id)
        outcome, why = Outcome.FAIL, reason
        verdict_id, verdict = write_verdict(
            store,
            activity,
            software,
            node=NODE,
            outcome=outcome,
            kind="airway",
            why=why,
            detail={
                "labelled_n": 0,
                "by_label": {},
                "contested_n": 0,
                "near_gate_n": 0,
                "merged_n": 0,
                "k_db": k_db,
                "flags": [why] if outcome is Outcome.FLAG else [],
            },
        )
        return NodeResult(verdict=verdict, view=(verdict_id,), verdict_entity_id=verdict_id)

    # Step 1 — re-evaluate every eligible candidate. Whole-file HeAR windows are deliberately not
    # evidence here: a short buffered input contains the proposed span and silence only.
    classify = store.activity(
        node=NODE,
        step="classify",
        parameters={"k_db": k_db, "labels_of_interest": labels_of_interest, "n_spans": len(spans)},
    )
    store.was_associated_with(classify, software)
    hear_agent = store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION)
    store.was_associated_with(classify, hear_agent)
    store.used(classify, stream_id)
    for span in spans:
        store.used(classify, span.id)
    if silence is not None:
        store.used(classify, silence.id)

    default_threshold = float(config.require("windows.hear.default_threshold"))
    thresholds = {
        str(label): float(value) for label, value in (config.get("windows.hear.label_thresholds") or {}).items()
    }
    labels_by_span: dict[str, list[tuple[str, str, list[str]]]] = {}
    by_label: dict[str, int] = {}
    near_gate_n = 0
    merged_n = 0
    flags: list[str] = []
    for span in spans:
        extent = span.extent or (0.0, 0.0)
        if _is_transcribed(store, extent):
            continue
        input_audio = _span_hear_input(audio, extent)
        reevaluated = detect_health_acoustic_events([input_audio], hop_length=HEAR_WINDOW_SECONDS, top_k=None)[0]
        members: dict[str, list[str]] = {}
        for raw_window in reevaluated:
            labels = _confident_labels(raw_window, default_threshold, thresholds)
            window_extent = _hear_window_extent(extent, raw_window)
            window_id = store.entity(
                prov_type="measurement",
                extent=window_extent,
                attributes={
                    "name": "hear_span_window",
                    "classifier": "hear",
                    "signal": source,
                    "span_id": span.id,
                    "labels": labels,
                    "scores": {label: score for pair in raw_window["label_scores"] for label, score in pair.items()},
                    "input_window_s": HEAR_WINDOW_SECONDS,
                    "isolated_span": True,
                },
            )
            store.was_generated_by(window_id, classify)
            store.was_attributed_to(window_id, hear_agent)
            store.was_derived_from(window_id, span.id)
            for label in labels:
                if label in labels_of_interest:
                    members.setdefault(label, []).append(window_id)
        if not members:
            continue
        merged_proposals = int(span.attributes.get("merged_proposals", 1))
        merged_n += merged_proposals
        margin = float(span.attributes["peak_over_floor_db"]) - k_db
        if k_margin_db is not None and margin <= k_margin_db:
            near_gate_n += 1
            flags.append(f"labelled span at {extent[0]:.2f}s sits {margin:.1f} dB over the gate")
        for label, window_ids in sorted(members.items()):
            attributes: dict[str, Any] = {
                "verb": "label",
                "label": label,
                "hear_window_ids": window_ids,
                "in_certified_silence": _inside_certified_silence(span, silence_windows),
                "merged_proposals": merged_proposals,
            }
            if k_margin_db is not None:
                attributes["margin_over_k_db"] = margin
            assertion_id = store.entity(prov_type="assertion", extent=span.extent, attributes=attributes)
            store.was_generated_by(assertion_id, classify)
            store.was_attributed_to(assertion_id, software)
            store.was_derived_from(assertion_id, span.id)
            for window_id in window_ids:
                store.used(classify, window_id)
                store.was_derived_from(assertion_id, window_id)
            labels_by_span.setdefault(span.id, []).append((assertion_id, label, window_ids))
            by_label[label] = by_label.get(label, 0) + 1

    # Step 2 — YAMNet answers the independently re-evaluated candidate over the same span extent.
    confirm_activity = store.activity(node=NODE, step="confirm", parameters={"contest_labels": sorted(contest_labels)})
    store.was_associated_with(confirm_activity, software)
    contested_n = 0
    answers: list[str] = []
    for span in spans:
        for assertion_id, label, window_ids in labels_by_span.get(span.id, []):
            confirms: list[tuple[str, str, str]] = []
            contests: list[tuple[str, str, str]] = []
            colocated: list[str] = []
            for hear_window_id in window_ids:
                hear_extent = store.get_entity(hear_window_id).extent or (0.0, 0.0)
                for yamnet_window in _windows_covering(store, "yamnet", hear_extent):
                    colocated.append(yamnet_window.id)
                    store.used(confirm_activity, yamnet_window.id)
                    for yamnet_label in _labels_of(yamnet_window):
                        if yamnet_label in confirmation_map.get(label, set()):
                            confirms.append((yamnet_window.id, yamnet_label, hear_window_id))
                        elif yamnet_label in contest_labels:
                            contests.append((yamnet_window.id, yamnet_label, hear_window_id))
            for verb, found in (("confirm", confirms), ("contest", contests)):
                if not found:
                    continue
                answer_id = store.entity(
                    prov_type="assertion",
                    extent=span.extent,
                    attributes={
                        "verb": verb,
                        "label": label,
                        "yamnet_labels": [yamnet_label for _, yamnet_label, _ in found],
                        "yamnet_window_ids": [window_id for window_id, _, _ in found],
                        "hear_window_ids": [window_id for _, _, window_id in found],
                    },
                )
                store.was_generated_by(answer_id, confirm_activity)
                store.was_attributed_to(answer_id, software)
                store.was_derived_from(answer_id, assertion_id)
                for window_id, _, _ in found:
                    store.was_derived_from(answer_id, window_id)
                answers.append(answer_id)
            if contests:
                contested_n += 1
                named = sorted({yamnet_label for _, yamnet_label, _ in contests})
                flags.append(f"{label} contested by {named} co-located in the same HeAR window")
            if not confirms and not contests:
                answer_id = store.entity(
                    prov_type="assertion",
                    extent=span.extent,
                    attributes={
                        "verb": "abstain",
                        "label": label,
                        "colocated_windows_n": len(colocated),
                        "hear_window_ids": window_ids,
                    },
                )
                store.was_generated_by(answer_id, confirm_activity)
                store.was_attributed_to(answer_id, software)
                store.was_derived_from(answer_id, assertion_id)
                answers.append(answer_id)

    # Step 3 — lexical contamination over the airway-labelled interval only.
    interval_id: str | None = None
    flag_id: str | None = None
    concluding = confirm_activity
    if labels_by_span:
        labelled_extents = [store.get_entity(span_id).extent or (0.0, 0.0) for span_id in labels_by_span]
        interval = (min(e[0] for e in labelled_extents), max(e[1] for e in labelled_extents))
        lexical = store.activity(node=NODE, step="lexical", parameters={"interval": list(interval)})
        store.was_associated_with(lexical, software)
        concluding = lexical
        interval_id = store.entity(
            prov_type="interval", extent=interval, attributes={"name": "airway_labelled_interval"}
        )
        store.was_generated_by(interval_id, lexical)
        store.was_attributed_to(interval_id, software)
        contaminating: list[str] = []
        for word in live_entities(store, "word"):
            word_start, word_end = word.extent or (0.0, 0.0)
            if word_start < interval[1] and word_end > interval[0]:
                store.used(lexical, word.id)
                contaminating.append(word.id)
        if contaminating:
            flag_id = store.entity(
                prov_type="assertion",
                extent=interval,
                attributes={"verb": "flag", "reason": "lexical_contamination", "word_ids": contaminating},
            )
            store.was_generated_by(flag_id, lexical)
            store.was_attributed_to(flag_id, software)
            store.was_derived_from(flag_id, interval_id)
            flags.append("lexical_contamination")

    # Step 4 — the outcome. An absence is a fail; the fold names any declaration it contradicts.
    if not labels_by_span:
        why = "spans exist but none carries a label of interest"
        outcome = Outcome.FAIL
    elif flags:
        outcome, why = Outcome.FLAG, "; ".join(flags)
    else:
        outcome = Outcome.PASS
        why = "at least one span carries a label of interest and nothing contests it"

    verdict_id, verdict = write_verdict(
        store,
        concluding,
        software,
        node=NODE,
        outcome=outcome,
        kind="airway",
        why=why,
        detail={
            "labelled_n": len(labels_by_span),
            "by_label": by_label,
            "contested_n": contested_n,
            "near_gate_n": near_gate_n,
            "merged_n": merged_n,
            "k_db": k_db,
            "flags": flags,
        },
    )

    view = (
        [span.id for span in spans]
        + [assertion_id for entries in labels_by_span.values() for assertion_id, _, _ in entries]
        + answers
        + ([interval_id] if interval_id else [])
        + ([flag_id] if flag_id else [])
        + [verdict_id]
    )
    return NodeResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id)
