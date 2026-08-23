"""AIRWAY — interpret PREPROCESS's spans. It proposes nothing: it labels, confirms and contests.

HeAR classifies each whole span placed in a silent buffer of exactly the model's window, via
``span_to_hear_buffer``; YAMNet confirms from its own native windows by coverage, never from a
padded span; ASR words are read for presence only. A hint changes only what an absence means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.health_acoustics.hear import HEAR_MODEL_ID, HEAR_REVISION, span_to_hear_buffer
from senselab.audio.tasks.plotting.plotting import plot_aligned_panels
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "AIRWAY"

# The recognizer whose word entities carry lexical evidence: PREPROCESS's CrisperWhisper default.
CRISPERWHISPER_ID = "nyralabs/CrisperWhisper2.0_turbo"


@dataclass(frozen=True)
class AirwayResult(NodeResult):
    """AIRWAY's result.

    Attributes:
        figure_path: The aligned figure, or None when rendering failed. An artifact, not store
            content.
    """

    figure_path: Path | None


def _hint_declares_airway(hint: AudioHints | None, labels_of_interest: list[str]) -> bool:
    """Whether the caller declared airway content (decision N18)."""
    if hint is None:
        return False
    declared = {tag.lower() for tag in hint.may_contain}
    return bool(declared & ({label.lower() for label in labels_of_interest} | {"airway"}))


def _inside_certified_silence(span: Entity, silence_windows: list[dict[str, Any]] | None) -> bool | None:
    """Whether every silence-graded window overlapping the span was certified silent (N17)."""
    if silence_windows is None:
        return None
    start, end = span.extent or (0.0, 0.0)
    overlapping = [w for w in silence_windows if float(w["start"]) < end and float(w["end"]) > start]
    if not overlapping:
        return None
    return all(bool(w["is_silence"]) for w in overlapping)


def _max_score(windows: list[dict[str, Any]], label: str) -> float:
    """The label's highest score across these windows."""
    best = 0.0
    for window in windows:
        for pair in window.get("label_scores", []):
            score = pair.get(label)
            if score is not None and float(score) > best:
                best = float(score)
    return best


def _best_of_interest(windows: list[dict[str, Any]], labels_of_interest: list[str]) -> dict[str, float]:
    """Each label of interest's highest score over these windows."""
    return {label: _max_score(windows, label) for label in labels_of_interest}


def airway(  # noqa: C901 — the branch's four steps, in order
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> AirwayResult:
    """Label, confirm and contest the spans PREPROCESS proposed at the airway K.

    Args:
        store: The provenance store, holding PREPROCESS's spans and derivatives.
        source: The store-held stream name HeAR's buffers are cut from, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain; read only to condition an absence.
        run_dir: The run directory; the figure goes under ``figures/``.

    Returns:
        The verdict, the view over the spans and assertions touched, and the figure path.

    Raises:
        ValueError: If ``hear.placement`` names an unimplemented placement.
    """
    software = software_agent(store)
    stream_id, plain = resolve_stream(store, run_dir, source)
    sr = int(plain.sampling_rate)

    k_db = float(config.require("spans.k_db.airway"))
    labels_of_interest = [str(label) for label in config.require("airway.labels_of_interest")]
    label_floor = float(config.require("hear.label_floor"))
    window_s = float(config.require("hear.window_s"))
    placement = str(config.require("hear.placement"))
    if placement != "centre":
        raise ValueError(f"hear.placement {placement!r} is not implemented; only 'centre' is")
    coverage_threshold = float(config.require("yamnet.coverage_threshold"))
    confirmation_map = {
        str(hear_label): {str(v) for v in yamnet_labels}
        for hear_label, yamnet_labels in config.require("airway.confirmation_map").items()
    }

    spans = [e for e in store.entities("span") if e.attributes.get("k_db") == k_db]
    spans.sort(key=lambda e: e.extent or (0.0, 0.0))
    hint_declares = _hint_declares_airway(hint, labels_of_interest)
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
        if hint_declares:
            outcome = Outcome.FLAG
            why = reason + "; a hint declares airway content not found"
        else:
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
                "flags": [why] if outcome is Outcome.FLAG else [],
            },
        )
        return AirwayResult(verdict=verdict, view=(verdict_id,), verdict_entity_id=verdict_id, figure_path=None)

    # Step 1 — HeAR labels each span: the whole span, buffered by span_to_hear_buffer; a span the
    # function refuses (longer than the window) is scanned over its own audio instead.
    hear_agent = store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION)
    classify = store.activity(
        node=NODE,
        step="classify",
        parameters={
            "k_db": k_db,
            "labels_of_interest": labels_of_interest,
            "label_floor": label_floor,
            "window_s": window_s,
            "placement": placement,
            "n_spans": len(spans),
        },
    )
    store.was_associated_with(classify, hear_agent)
    store.used(classify, stream_id)
    for span in spans:
        store.used(classify, span.id)
    if silence is not None:
        store.used(classify, silence.id)

    buffered: list[tuple[Entity, Audio]] = []
    sliding: list[tuple[Entity, Audio]] = []
    for span in spans:
        start, end = span.extent or (0.0, 0.0)
        try:
            buffered.append((span, span_to_hear_buffer(plain, start, end, placement=placement)))
        except ValueError:  # the function refuses a span longer than the window (N14)
            segment = plain.waveform[:, int(start * sr) : int(end * sr)]
            sliding.append((span, Audio(waveform=segment, sampling_rate=sr)))

    scored: list[tuple[Entity, dict[str, float], str]] = []
    if buffered:
        outputs = detect_health_acoustic_events([audio for _, audio in buffered], hop_length=window_s)
        for (span, _), windows in zip(buffered, outputs):
            scored.append((span, _best_of_interest(windows, labels_of_interest), "buffered"))
    if sliding:
        outputs = detect_health_acoustic_events([audio for _, audio in sliding])
        for (span, _), windows in zip(sliding, outputs):
            scored.append((span, _best_of_interest(windows, labels_of_interest), "sliding"))

    label_ids: dict[str, str] = {}
    span_labels: dict[str, str] = {}
    by_label: dict[str, int] = {}
    for span, scores, input_kind in scored:
        best_label = max(scores, key=lambda label: scores[label])
        if scores[best_label] < label_floor:
            continue
        assertion_id = store.entity(
            prov_type="assertion",
            extent=span.extent,
            attributes={
                "verb": "label",
                "label": best_label,
                "score": scores[best_label],
                "scores": scores,
                "input": input_kind,
                "in_certified_silence": _inside_certified_silence(span, silence_windows),
            },
        )
        store.was_generated_by(assertion_id, classify)
        store.was_attributed_to(assertion_id, hear_agent)
        store.was_derived_from(assertion_id, span.id)
        label_ids[span.id] = assertion_id
        span_labels[span.id] = best_label
        by_label[best_label] = by_label.get(best_label, 0) + 1

    # Step 2 — YAMNet answers each label from its own native windows, by coverage.
    yamnet_meas = find_measurement(store, "yamnet_windows")
    yamnet_windows: list[dict[str, Any]] | None = None
    if yamnet_meas is not None:
        yamnet_windows = json.loads((run_dir / yamnet_meas.attributes["path"]).read_text())
    yamnet_agent = store.agent(
        agent_type="model",
        model_id="https://tfhub.dev/google/yamnet/1",
        unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
    )
    confirm_activity = store.activity(node=NODE, step="confirm", parameters={"coverage_threshold": coverage_threshold})
    store.was_associated_with(confirm_activity, yamnet_agent)
    if yamnet_meas is not None:
        store.used(confirm_activity, yamnet_meas.id)

    contested_n = 0
    flags: list[str] = []
    answers: list[str] = []
    for span in spans:
        label_id = label_ids.get(span.id)
        if label_id is None:
            continue
        start, end = span.extent or (0.0, 0.0)
        overlapping = (
            [w for w in yamnet_windows if float(w["start"]) < end and float(w["end"]) > start]
            if yamnet_windows is not None
            else []
        )
        coverage_counts: dict[str, int] = {}
        for window in overlapping:
            for pair in window.get("label_scores", []):
                for label, score in pair.items():
                    if float(score) >= coverage_threshold:
                        coverage_counts[label] = coverage_counts.get(label, 0) + 1
        if not coverage_counts:
            attributes: dict[str, Any] = {
                "verb": "abstain",
                "best_coverage": 0.0,
                "n_windows": len(overlapping),
            }
        else:
            winner = max(
                coverage_counts,
                key=lambda label: (coverage_counts[label], _max_score(overlapping, label)),
            )
            verb = "confirm" if winner in confirmation_map.get(span_labels[span.id], set()) else "contest"
            attributes = {
                "verb": verb,
                "winner": winner,
                "coverage": coverage_counts[winner] / len(overlapping),
                "n_windows": len(overlapping),
                "mapped_to": span_labels[span.id],
            }
            if verb == "contest":
                contested_n += 1
                flags.append(f"yamnet contests {span_labels[span.id]} with {winner}")
        answer_id = store.entity(prov_type="assertion", extent=span.extent, attributes=attributes)
        store.was_generated_by(answer_id, confirm_activity)
        store.was_attributed_to(answer_id, yamnet_agent)
        store.was_derived_from(answer_id, label_id)
        store.was_derived_from(answer_id, span.id)
        answers.append(answer_id)

    # Step 3 — lexical contamination over the airway-labelled interval only.
    interval_id: str | None = None
    flag_id: str | None = None
    if span_labels:
        labelled_extents = [store.get_entity(span_id).extent or (0.0, 0.0) for span_id in span_labels]
        interval = (min(e[0] for e in labelled_extents), max(e[1] for e in labelled_extents))
        lexical = store.activity(node=NODE, step="lexical", parameters={"interval": list(interval)})
        store.was_associated_with(lexical, software)
        interval_id = store.entity(
            prov_type="interval", extent=interval, attributes={"name": "airway_labelled_interval"}
        )
        store.was_generated_by(interval_id, lexical)
        store.was_attributed_to(interval_id, software)
        contaminating: list[str] = []
        for word in store.entities("word"):
            if word.attributes.get("recognizer") != CRISPERWHISPER_ID:
                continue
            text = str(word.attributes.get("text") or "")
            if text.startswith("[") and text.endswith("]"):
                continue
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

    # Step 4 — the outcome. A hint conditions only what an absence means.
    if not span_labels:
        why = "no span carries a label of interest"
        if hint_declares:
            why += "; a hint declares airway content not found"
        flags.append(why)
        outcome = Outcome.FLAG
    elif flags:
        outcome, why = Outcome.FLAG, "; ".join(flags)
    else:
        outcome = Outcome.PASS
        why = "at least one span carries a label of interest and nothing contests it"

    verdict_id, verdict = write_verdict(
        store,
        classify,
        software,
        node=NODE,
        outcome=outcome,
        kind="airway",
        why=why,
        detail={"labelled_n": len(span_labels), "by_label": by_label, "contested_n": contested_n, "flags": flags},
    )

    figure_path: Path | None = None
    try:
        figure_path = _render_figure(store, plain, spans, span_labels, silence_windows, run_dir, config)
    except Exception:  # noqa: BLE001 — the figure is an artifact; failing to draw it changes no verdict
        figure_path = None

    view = (
        [span.id for span in spans]
        + list(label_ids.values())
        + answers
        + ([interval_id] if interval_id else [])
        + ([flag_id] if flag_id else [])
        + [verdict_id]
    )
    return AirwayResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, figure_path=figure_path)


def _render_figure(
    store: ProvStore,
    plain: Audio,
    spans: list[Entity],
    span_labels: dict[str, str],
    silence_windows: list[dict[str, Any]] | None,
    run_dir: Path,
    config: TriageConfig,
) -> Path:
    """One aligned figure: plain waveform, envelope with floor, spans, silence, spectrogram."""
    panels: list[dict[str, Any]] = [{"type": "waveform"}]
    envelope = find_measurement(store, "energy_envelope")
    if envelope is not None:
        sidecar = np.load(run_dir / envelope.attributes["path"])
        rate = int(envelope.attributes["sampling_rate"])
        stride = max(1, int(rate * float(config.require("gammatone.hop_s"))))
        times = (np.arange(len(sidecar["envelope_dbfs"])) / rate)[::stride]
        panels.append(
            {
                "type": "features",
                "data": [
                    (
                        times.tolist(),
                        sidecar["envelope_dbfs"][::stride].tolist(),
                        "envelope dBFS (pre-emphasised)",
                        "tab:blue",
                    ),
                    (times.tolist(), sidecar["floor_dbfs"][::stride].tolist(), "floor dBFS", "tab:gray"),
                ],
            }
        )
    segments = [
        {
            "label": span_labels.get(span.id, "unlabelled"),
            "start": (span.extent or (0.0, 0.0))[0],
            "end": (span.extent or (0.0, 0.0))[1],
        }
        for span in spans
    ]
    if segments:
        panels.append({"type": "segments", "segments": segments})
    if silence_windows:
        panels.append(
            {
                "type": "segments",
                "segments": [
                    {"label": "Silence" if w["is_silence"] else "sound", "start": w["start"], "end": w["end"]}
                    for w in silence_windows
                ],
            }
        )
    panels.append({"type": "spectrogram", "mel": False})
    figure = plot_aligned_panels(plain, panels, title="AIRWAY")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    path = run_dir / "figures" / "airway.png"
    figure.savefig(path)
    return path
