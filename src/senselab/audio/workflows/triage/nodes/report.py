"""REPORT — one summary and one summary JSON per file, on every file and every outcome.

Runs last, after VERDICT, and is the only node that writes no elements: it reads the whole store and
renders it. A rendering is not evidence, so nothing downstream reads either product to learn a fact
the store does not already hold. Both products carry element ids, which are a join key back into the
store, so they sit beside it and never under the release tree. The design is in
``specs/20260817-triage-workflow-dag/report.md``.
"""

from __future__ import annotations

import json
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from matplotlib.figure import Figure

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.plotting.plotting import (
    MIN_FIGURE_HEIGHT_IN,
    TEXT_PANEL_INCHES_PER_LINE,
    plot_aligned_panels,
)
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import find_measurement, live_entities, resolve_stream
from senselab.audio.workflows.triage.vocabulary import BRANCHES, GRAPH_ORDER
from senselab.utils.prov_store import Entity, ProvStore

NODE = "REPORT"
SUMMARY_STEM = "summary"
FORMATS = ("png", "pdf")

_CONDITIONED_STREAM = "plain"
_SOURCE_STREAM = "recording"
_CLASSIFIERS = ("yamnet", "ast", "hear")
_UNLABELLED = "unlabelled"
_UNKNOWN = "—"
_UNSCANNED = "[unscanned]"
_SHA_LENGTH = 40
_BLOCK_COLUMNS = 168
_TITLE_COLUMNS = 96
_SHOWN_DECIMALS = 4
_TOP_CATEGORIES = 6
_TITLE_SEPARATOR = " · "
_TASK_PREFIX = "task-"
_RUN_STAMP = re.compile(r"^(\d{4})(\d{2})(\d{2})-\d{6}(?:-\d+)?$")

_ABSENCE_BY_CLASS = {
    "ValueError": "unfitted (a config key it reads is null)",
    "LookupError": "unavailable (a derivative it reads is absent)",
}
_ABSENCE_ERRORED = "errored"
_NO_AXIS = "no time axis: the store holds no readable stream, so there is no shared axis to draw over"
_ENVELOPE_AXIS = "envelope dBFS"

_LANE_SOURCE = {
    "envelope": "energy_envelope",
    "spans (dB over floor)": "spans",
    "phonation": "phonation_spans",
    "yamnet labels": "yamnet_windows",
    "ast labels": "ast_windows",
    "hear labels": "hear_windows",
    "words": "consensus_transcript",
    "airway": "spans",
}
_LANE_BRANCH = {"speech spans": "SPEECH", "airway": "AIRWAY", "voice": "VOICE", "redacted": "REDACT"}
_LANES = (
    "envelope",
    "spans (dB over floor)",
    "phonation",
    "yamnet labels",
    "ast labels",
    "hear labels",
    "speech spans",
    "words",
    "airway",
    "voice",
    "redacted",
)

_BRANCH_MEASURES = {
    "AIRWAY": ("labelled_n", "contested_n", "near_gate_n", "merged_n", "k_db"),
    "SPEECH": ("speaker_count", "words_n", "speech_s", "nontarget_speech_s"),
    "VOICE": ("spans_n", "phonation_s", "longest_span_s"),
}


class ReportRenderError(RuntimeError):
    """The summary could not be drawn, after the JSON was already written.

    The JSON is the product a consumer reads and it is complete by the time the renderer runs, so a
    drawing failure must not take it down with it. The artifacts written before the failure travel on
    the exception; the runner records the failure and keeps them.

    Attributes:
        artifacts: What was written before the renderer raised.
    """

    def __init__(self, message: str, artifacts: dict[str, Path]) -> None:
        """Carry the message and the artifacts that survived."""
        super().__init__(message)
        self.artifacts = dict(artifacts)


def _assertions_by_source(store: ProvStore) -> dict[str, list[Entity]]:
    """Every live assertion, indexed by the element it was derived from.

    Built once per report. The two readers below ask "what was asserted over this word / this span",
    which the store answers only in the forward direction, so without an index each of them walks
    every assertion for every element it renders — cubic in the size of a transcript.

    Args:
        store: The provenance store.

    Returns:
        ``{source element id: [assertion, ...]}``, in write order.
    """
    index: dict[str, list[Entity]] = {}
    for assertion in live_entities(store, "assertion"):
        for source_id in store.derived_from(assertion.id):
            index.setdefault(source_id, []).append(assertion)
    return index


def _scan_state(store: ProvStore) -> tuple[bool, str]:
    """Whether a PII scan covered this transcript, and what to say when it did not.

    The marking is what redacts, so a page that renders unmarked words verbatim is trusting the
    absence of a marking. That absence has two causes and they are not the same: SPEECH scanned and
    found nothing, or nobody scanned at all — because routing declined the branch, because SPEECH
    raised, or because every detector failed. REDACT already refuses to release on this distinction
    (N15); the summary must respect it too, since it is written beside the store and read by people.

    Args:
        store: The provenance store.

    Returns:
        ``(trustworthy, reason)``. ``reason`` is empty when the scan ran and every required detector
        was attempted.
    """
    scan = find_measurement(store, "pii_scan")
    if scan is None:
        return False, "no pii scan is in the store; SPEECH did not run or did not reach its scan"
    failed = [str(name) for name in (scan.attributes.get("failed") or [])]
    missing = [str(name) for name in (scan.attributes.get("missing") or [])]
    if failed or missing:
        parts = []
        if failed:
            parts.append(f"detectors failed: {', '.join(sorted(failed))}")
        if missing:
            parts.append(f"required detectors were not attempted: {', '.join(sorted(missing))}")
        return False, "the pii scan is incomplete (" + "; ".join(parts) + ")"
    return True, ""


def _redacted_text(marks: dict[str, list[Entity]], word: Entity, *, scanned: bool = True) -> str:
    """A word's renderable text: its category placeholder when the scan marked it, else the word.

    The store holds PII by design and the report carries element ids, so the report is not a released
    artifact — but no matched text may appear in it either way.

    Args:
        marks: :func:`_assertions_by_source`'s index, so the answer costs one dictionary lookup
            rather than a walk over every assertion in the store.
        word: A ``word`` entity.
        scanned: Whether a complete PII scan stands behind the markings. When it does not, every word
            is withheld: an unmarked word is only evidence of cleanliness if something looked.

    Returns:
        ``"[<CATEGORY>]"`` when a live ``pii`` label assertion is derived from this word,
        ``"[unscanned]"`` when no complete scan covered the transcript, else the word's own text.
    """
    for assertion in marks.get(word.id, ()):
        attributes = assertion.attributes
        if attributes.get("verb") == "label" and attributes.get("label") == "pii":
            return f"[{attributes.get('category')}]"
    return str(word.attributes.get("text") or "") if scanned else _UNSCANNED


def _words(store: ProvStore) -> list[Entity]:
    """The consensus words in the order PREPROCESS fused them.

    Args:
        store: The provenance store.

    Returns:
        Live ``word`` entities sorted by their recorded index, then by their extent.
    """
    return sorted(
        live_entities(store, "word"),
        key=lambda word: (int(word.attributes.get("index") or 0), word.extent or (0.0, 0.0)),
    )


def _transcript(store: ProvStore, marks: dict[str, list[Entity]]) -> str:
    """The consensus transcript with every marked word replaced by its category.

    Args:
        store: The provenance store.
        marks: :func:`_assertions_by_source`'s index.

    Returns:
        The redacted transcript, empty when the store holds no consensus words, and every word
        withheld when no complete PII scan covered it.
    """
    scanned, _ = _scan_state(store)
    return " ".join(_redacted_text(marks, word, scanned=scanned) for word in _words(store))


def _envelope_spans(store: ProvStore) -> list[Entity]:
    """PREPROCESS's envelope spans — the ones carrying a peak over the rolling floor.

    Args:
        store: The provenance store.

    Returns:
        The spans, earliest first.
    """
    spans = [span for span in live_entities(store, "span") if "peak_over_floor_db" in span.attributes]
    return sorted(spans, key=lambda span: span.extent or (0.0, 0.0))


def _spans_of_family(store: ProvStore, family: str, *, voice: bool | None = None) -> list[Entity]:
    """Spans of one family, optionally only those VOICE re-timed or only those PREPROCESS proposed.

    ``onset_kind`` is what tells the two phonation populations apart: VOICE writes it and PREPROCESS
    does not.

    Args:
        store: The provenance store.
        family: The span family.
        voice: True for VOICE's spans only, False for PREPROCESS's only, None for both.

    Returns:
        The spans, earliest first.
    """
    found = [span for span in live_entities(store, "span") if span.attributes.get("family") == family]
    if voice is not None:
        found = [span for span in found if ("onset_kind" in span.attributes) is voice]
    return sorted(found, key=lambda span: span.extent or (0.0, 0.0))


def _airway_labels(store: ProvStore, marks: dict[str, list[Entity]], span: Entity) -> list[str]:
    """The airway labels a span carries, from AIRWAY's own label assertions over it.

    The generating activity's node is what makes this AIRWAY's answer rather than any label another
    node might one day assert over the same span.

    Args:
        store: The provenance store, read for each assertion's generating activity.
        marks: :func:`_assertions_by_source`'s index.
        span: An envelope span.

    Returns:
        The labels, sorted. Empty when AIRWAY labelled nothing over the span.
    """
    labels = set()
    for assertion in marks.get(span.id, ()):
        if assertion.attributes.get("verb") != "label":
            continue
        activity_id = store.generated_by(assertion.id)
        if activity_id is None or store.get_activity(activity_id).node != "AIRWAY":
            continue
        labels.add(str(assertion.attributes["label"]))
    return sorted(labels)


def _segments(entries: Iterable[tuple[tuple[float, float], str]]) -> list[dict[str, Any]]:
    """Segment dicts for a ``segments`` panel.

    Args:
        entries: ``(extent, label)`` pairs.

    Returns:
        The segment specifications, one per pair.
    """
    return [{"label": label, "start": float(extent[0]), "end": float(extent[1])} for extent, label in entries]


def _lane(name: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One ``segments`` panel, or none at all when the store held nothing for it.

    Args:
        name: The lane's name, drawn as the panel's y-label.
        entries: The lane's segments.

    Returns:
        A one-element list holding the panel, or an empty list.
    """
    return [{"type": "segments", "segments": entries, "name": name}] if entries else []


def _window_lane(store: ProvStore, classifier: str) -> list[dict[str, Any]]:
    """One classifier's label lane: one segment per window that carried a label set.

    Args:
        store: The provenance store.
        classifier: ``yamnet``, ``ast`` or ``hear``.

    Returns:
        The lane, or an empty list when the classifier's window fold is absent from the store.
    """
    entries = []
    for window in live_entities(store, "measurement"):
        if window.attributes.get("name") != f"{classifier}_window" or window.extent is None:
            continue
        labels = [str(label) for label in (window.attributes.get("labels") or [])]
        if labels:
            entries.append((window.extent, ", ".join(labels)))
    return _lane(f"{classifier} labels", _segments(entries))


def _label_counts(store: ProvStore, classifier: str) -> dict[str, int]:
    """How many windows each label of one classifier fired on.

    Args:
        store: The provenance store.
        classifier: ``yamnet``, ``ast`` or ``hear``.

    Returns:
        ``{label: windows}``, descending by count.
    """
    counts: dict[str, int] = {}
    for window in live_entities(store, "measurement"):
        if window.attributes.get("name") != f"{classifier}_window":
            continue
        for label in window.attributes.get("labels") or []:
            counts[str(label)] = counts.get(str(label), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _envelope_curves(
    store: ProvStore, run_dir: Path, config: TriageConfig
) -> list[tuple[np.ndarray, np.ndarray, str, str]] | None:
    """PREPROCESS's energy envelope and its rolling floor, decimated to the spectrogram's hop.

    Args:
        store: The provenance store.
        run_dir: Where the envelope's sidecar path resolves against.
        config: The triage configuration, read for the decimation stride only.

    Returns:
        ``[(times, values, label, colour), ...]`` for the two curves, or None when the derivative is
        absent from the store or its sidecar has been moved away.
    """
    envelope = find_measurement(store, "energy_envelope")
    if envelope is None:
        return None
    path = Path(str(envelope.attributes.get("path") or ""))
    path = path if path.is_absolute() else run_dir / path
    if not path.is_file():
        return None
    loaded = np.load(path)
    rate = float(envelope.attributes.get("sampling_rate") or 1.0)
    stride = max(1, int(rate * float(config.require("spectrogram.hop_ms")) / 1000.0))
    times = np.arange(0, len(loaded["envelope_dbfs"]), stride) / rate
    return [
        (times, loaded["envelope_dbfs"][::stride], "envelope dBFS", "steelblue"),
        (times, loaded["floor_dbfs"][::stride], "floor dBFS", "firebrick"),
    ]


def _panels(
    store: ProvStore, marks: dict[str, list[Entity]], run_dir: Path, config: TriageConfig
) -> tuple[list[dict[str, Any]], set[str]]:
    """The summary's layers on one shared time axis, drawn from whatever the store holds.

    A layer whose derivative is absent is omitted; nothing raises for want of one, because
    report.md requires a product on every outcome including a file ADMIT refused. Which layers were
    omitted, and why, is what the ABSENT block says — an omitted lane must never read as a measured
    absence.

    Args:
        store: The provenance store.
        marks: :func:`_assertions_by_source`'s index.
        run_dir: Where sidecar paths resolve against.
        config: The triage configuration, read for the envelope decimation stride only.

    Returns:
        The panel specifications for ``plot_aligned_panels``, and the names of the declared lanes
        they drew. The two are not the same set: the envelope shares the waveform's row and is
        named by that row's right-hand scale rather than by a panel ``name`` of its own.
    """
    waveform: dict[str, Any] = {"type": "waveform"}
    panels: list[dict[str, Any]] = [waveform]
    drawn: set[str] = set()
    scanned, _ = _scan_state(store)

    envelope = _envelope_curves(store, run_dir, config)
    if envelope is not None:
        waveform["twin"] = {"name": _ENVELOPE_AXIS, "data": envelope}
        drawn.add("envelope")

    spans = _envelope_spans(store)
    panels += _lane(
        "spans (dB over floor)",
        _segments(
            (span.extent, f"{float(span.attributes['peak_over_floor_db']):.0f} dB")
            for span in spans
            if span.extent is not None
        ),
    )
    panels += _lane(
        "phonation",
        _segments(
            (span.extent, f"{span.attributes.get('member')}/{span.attributes.get('production')}")
            for span in _spans_of_family(store, "phonation", voice=False)
            if span.extent is not None
        ),
    )
    for classifier in _CLASSIFIERS:
        panels += _window_lane(store, classifier)
    panels += _lane(
        "speech spans",
        _segments(
            (
                span.extent,
                f"{span.attributes.get('attributed_to') or 'unattributed'}"
                + (" nontarget" if span.attributes.get("nontarget") else ""),
            )
            for span in _spans_of_family(store, "speech")
            if span.extent is not None
        ),
    )
    panels += _lane(
        "words",
        _segments(
            (word.extent, _redacted_text(marks, word, scanned=scanned))
            for word in _words(store)
            if word.extent is not None
        ),
    )
    panels += _lane(
        "airway",
        _segments(
            (span.extent, ", ".join(_airway_labels(store, marks, span)) or _UNLABELLED)
            for span in spans
            if span.extent is not None
        ),
    )
    panels += _lane(
        "voice",
        _segments(
            (span.extent, f"{span.attributes.get('member')}/{span.attributes.get('onset_kind')} onset")
            for span in _spans_of_family(store, "phonation", voice=True)
            if span.extent is not None
        ),
    )
    panels += _lane(
        "redacted",
        _segments(
            (span.extent, str(span.attributes.get("category")))
            for span in live_entities(store, "span")
            if span.attributes.get("name") == "redaction" and span.extent is not None
        ),
    )
    panels.append({"type": "spectrogram"})
    return panels, drawn | {str(panel["name"]) for panel in panels if "name" in panel}


def _verdict_entities(store: ProvStore) -> dict[str, Entity]:
    """The latest live verdict entity per node, keyed by node name.

    Args:
        store: The provenance store.

    Returns:
        ``{node: entity}`` over every node that concluded.
    """
    latest: dict[str, Entity] = {}
    for entity in store.entities("verdict"):
        if not store.is_invalidated(entity.id):
            latest[str(entity.attributes.get("node"))] = entity
    return latest


def _steps(store: ProvStore) -> dict[str, dict[str, Any]]:
    """Per-step summary fields, each naming the element ids behind it.

    Args:
        store: The provenance store.

    Returns:
        ``{step: {**verdict detail, "element_ids": [...]}}`` over every node that wrote a verdict.
        The ids are the verdict entity itself and every **live** entity the node's activities
        generated, which is what makes any number in the summary traceable to the assertion that
        produced it. An invalidated entity is left out under the store's shared read rule: it is a
        join key to something the graph has withdrawn, and citing it would credit a claim to
        evidence that no longer stands.
    """
    by_node = _verdict_entities(store)
    generated: dict[str, list[str]] = {}
    for entity in store.entities():
        activity_id = store.generated_by(entity.id)
        if activity_id is None or store.is_invalidated(entity.id):
            continue
        generated.setdefault(store.get_activity(activity_id).node, []).append(entity.id)
    ordered = sorted(by_node, key=lambda node: GRAPH_ORDER.index(node) if node in GRAPH_ORDER else len(GRAPH_ORDER))
    return {
        node: {
            **{key: value for key, value in by_node[node].attributes.items() if key != "node"},
            "element_ids": sorted({by_node[node].id, *generated.get(node, [])}),
        }
        for node in ordered
    }


def _branches(store: ProvStore) -> dict[str, dict[str, Any]]:
    """ROUTING's decision per branch, joined to what the branch concluded.

    Args:
        store: The provenance store.

    Returns:
        ``{branch: {will_run, forced_by_hint, kind_state, why, verdict, flags}}``. Empty when ROUTING
        never ran, which is a graph in which no branch was ever asked.
    """
    decisions: dict[str, Entity] = {}
    for entity in store.entities("branch_decision"):
        if not store.is_invalidated(entity.id):
            decisions[str(entity.attributes["branch"])] = entity
    concluded = _verdict_entities(store)
    branches: dict[str, dict[str, Any]] = {}
    for branch, decision in decisions.items():
        verdict_entity = concluded.get(branch)
        branches[branch] = {
            "will_run": bool(decision.attributes.get("will_run")),
            "forced_by_hint": bool(decision.attributes.get("forced_by_hint")),
            "kind_state": decision.attributes.get("kind_state"),
            "raw_state": decision.attributes.get("raw_state"),
            "why": decision.attributes.get("why"),
            "verdict": None if verdict_entity is None else verdict_entity.attributes.get("outcome"),
            "flags": [] if verdict_entity is None else list(verdict_entity.attributes.get("flags") or []),
            "element_ids": sorted({decision.id} | ({verdict_entity.id} if verdict_entity is not None else set())),
        }
    return branches


def _senselab_commit() -> tuple[str | None, str | None]:
    """The senselab commit the run was made at, or the reason it could not be resolved.

    A ref is never returned: recording a ref where a commit belongs makes the provenance confidently
    wrong, which is worse than recording nothing.

    Returns:
        ``(commit_sha, unresolved_reason)`` with exactly one of the two set.
    """
    root = Path(__file__).resolve().parents[5]
    try:
        found = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as error:  # noqa: BLE001 — any failure to ask git is the same unresolved state
        return None, f"git could not be run: {type(error).__name__}"
    sha = found.stdout.strip()
    if found.returncode != 0 or len(sha) != _SHA_LENGTH:
        return None, "the installed senselab is not inside a git work tree; no commit to resolve"
    return sha, None


def _provenance(store: ProvStore, config: TriageConfig, run_id: str) -> dict[str, Any]:
    """The run's provenance, embedded rather than referenced.

    Args:
        store: The provenance store, read for its Agent records.
        config: The triage configuration.
        run_id: The run's id.

    Returns:
        ``{config_hash, config, commit, models, run_id, started, ended}``. Every model agent appears
        with its resolved commit or its ``unresolved_reason`` — never a bare ref.
    """
    models: list[dict[str, Any]] = []
    for agent in store.agents("model"):
        activities = [store.get_activity(activity_id) for activity_id in store.associations_of(agent.id)]
        models.append(
            {
                "agent_id": agent.id,
                "model_id": agent.model_id,
                "revision": agent.commit_sha,
                "unresolved_reason": agent.unresolved_reason,
                "task": sorted({activity.step for activity in activities if activity.step is not None}),
                "node": sorted({activity.node for activity in activities}),
            }
        )
    stamps = [(activity.started, activity.ended) for activity in store.activities()]
    started = sorted(stamp for stamp, _ in stamps if stamp is not None)
    ended = sorted(stamp for _, stamp in stamps if stamp is not None)
    commit, commit_unresolved_reason = _senselab_commit()
    return {
        "config_hash": config.config_hash,
        "config": config.values,
        "commit": commit,
        "commit_unresolved_reason": commit_unresolved_reason,
        "software": sorted({agent.version for agent in store.agents("software") if agent.version is not None}),
        "models": sorted(models, key=lambda model: (str(model["model_id"]), str(model["agent_id"]))),
        "run_id": run_id,
        "started": started[0] if started else None,
        "ended": ended[-1] if ended else None,
    }


def _admitted_path(store: ProvStore) -> str | None:
    """The path ADMIT was handed, from its own activity parameters.

    A file ADMIT refused has no stream entity, so this is the only place its name survives — and a
    refusal page that cannot name the file it refused is of no use to whoever has to find it.

    Args:
        store: The provenance store.

    Returns:
        The path, or None when ADMIT has no activity.
    """
    for activity in store.activities("ADMIT"):
        found = activity.parameters.get("audio_file")
        if found:
            return str(found)
    return None


def _file(store: ProvStore) -> dict[str, Any]:
    """What the recording is, from ADMIT's stream entity, falling back to what it was handed.

    Args:
        store: The provenance store.

    Returns:
        ``{path, duration_s, sample_rate, channels}``. Only ``path`` survives a refusal; the other
        three are None, because nothing measured them.
    """
    found = [entity for entity in live_entities(store, "stream") if entity.attributes.get("name") == _SOURCE_STREAM]
    if not found:
        return {"path": _admitted_path(store), "duration_s": None, "sample_rate": None, "channels": None}
    entity = found[-1]
    return {
        "path": entity.attributes.get("path") or _admitted_path(store),
        "duration_s": None if entity.extent is None else float(entity.extent[1]),
        "sample_rate": entity.attributes.get("sampling_rate"),
        "channels": entity.attributes.get("channels"),
    }


def _shown(value: Any) -> str:  # noqa: ANN401 — anything a store attribute can hold
    """One value as the page shows it: an em dash where the store holds nothing.

    Only the rendering is rounded. Every figure the page shows is in the JSON at full precision, and
    the JSON is what a consumer reads; "phonation_s: 11.979999999999999" on a page is a binary
    float's repr leaking into a document a human is meant to judge the run by.

    Args:
        value: The value.

    Returns:
        Its text, ``"—"`` when it is None, and a float rounded to four decimals.
    """
    if value is None:
        return _UNKNOWN
    if isinstance(value, float):
        return str(round(value, _SHOWN_DECIMALS))
    return str(value)


def _absences(store: ProvStore) -> dict[str, tuple[str, str]]:
    """Why each PREPROCESS derivative is missing, keyed by the derivative's name.

    PREPROCESS records ``"Class: first line"``. The reading is by class — ``config.require`` raises
    ``ValueError`` for a key nobody has measured, a block whose input is missing raises
    ``LookupError``, anything else is a genuine failure — and the message is what names *which* key
    or *which* input, so it is carried through to the page rather than dropped at the colon.

    Args:
        store: The provenance store.

    Returns:
        ``{derivative: (reading, what PREPROCESS recorded)}``. Empty when PREPROCESS never concluded.
    """
    verdict_entity = _verdict_entities(store).get("PREPROCESS")
    absent = (verdict_entity.attributes.get("absent") or {}) if verdict_entity is not None else {}
    return {
        str(name): (_ABSENCE_BY_CLASS.get(str(raised).split(":", 1)[0], _ABSENCE_ERRORED), str(raised))
        for name, raised in sorted(absent.items())
    }


def _lane_absences(store: ProvStore, drawn: set[str]) -> list[tuple[str, str]]:
    """Each declared lane the page did not draw, with why, as far as the store can say.

    Args:
        store: The provenance store.
        drawn: The names of the lanes that were drawn.

    Returns:
        ``[(lane, reason), ...]`` for every declared lane not in ``drawn``.
    """
    absences = _absences(store)
    branches = _branches(store)
    out: list[tuple[str, str]] = []
    for lane in _LANES:
        if lane in drawn:
            continue
        derivative = _LANE_SOURCE.get(lane)
        branch = _LANE_BRANCH.get(lane)
        if derivative is not None and derivative in absences:
            reading, raised = absences[derivative]
            out.append((lane, f"PREPROCESS/{derivative} {reading} [{raised}]"))
        elif branch is not None and branch in branches and not branches[branch]["will_run"]:
            out.append((lane, f"{branch} did not run: {branches[branch]['why']}"))
        elif branch is not None and _verdict_entities(store).get(branch) is None:
            out.append((lane, f"{branch} wrote no verdict"))
        else:
            out.append((lane, "nothing in the store for it"))
    return out


def _verdict(store: ProvStore) -> dict[str, Any]:
    """The file verdict's own detail, or a mapping of Nones when VERDICT never concluded.

    Args:
        store: The provenance store.

    Returns:
        The verdict entity's attributes minus ``node``, with ``element_id`` naming the entity.
    """
    entity = _verdict_entities(store).get("VERDICT")
    if entity is None:
        return {
            "triage": None,
            "release": None,
            "discard_ground": None,
            "reasons": [],
            "kinds": {},
            "screened": {},
            "agreement": {},
            "hints": {},
            "ran": {},
            "element_id": None,
        }
    return {
        **{key: value for key, value in entity.attributes.items() if key != "node"},
        "element_id": entity.id,
    }


def _categories(store: ProvStore) -> dict[str, dict[str, int]]:
    """The label sets each classifier fired, ranked by how many windows carried them.

    Args:
        store: The provenance store.

    Returns:
        ``{classifier: {label: windows}}``, only for classifiers whose windows are in the store.
    """
    found = {classifier: _label_counts(store, classifier) for classifier in _CLASSIFIERS}
    return {classifier: counts for classifier, counts in found.items() if counts}


def _provenance_line(provenance: dict[str, Any]) -> str:
    """One line naming the run's identity: its config, its commit and how its models resolved.

    The full model list stays in the JSON. What belongs on the page is enough for a reviewer to say
    "this page came from that configuration at that commit" without opening anything else.

    Args:
        provenance: :func:`_provenance`'s mapping.

    Returns:
        The line.
    """
    models = provenance["models"]
    resolved = sum(1 for model in models if model["revision"] is not None)
    commit = provenance["commit"] or f"unresolved: {provenance['commit_unresolved_reason']}"
    return (
        f"provenance: config {provenance['config_hash']}  senselab {commit}  "
        f"models: {resolved} at a resolved commit, {len(models) - resolved} with a reason"
    )


def _run_label(run_id: str) -> str:
    """The run's short label: the task token it names and the date it was made on.

    The runner mints a run id as ``<file stem>_<utc stamp>``, and a corpus stem is BIDS-shaped often
    enough to carry a ``task-`` entity. Neither is guaranteed, so each part is taken when it is there
    and dropped when it is not.

    Args:
        run_id: The store's run id.

    Returns:
        ``"<task token> · <YYYY-MM-DD>"``, falling back to the stem when no ``task-`` entity is in it
        and to the whole run id when no stamp is either.
    """
    tokens = run_id.split("_")
    stamped = _RUN_STAMP.match(tokens[-1]) if tokens else None
    stem = tokens[:-1] if stamped is not None else tokens
    task = next((token for token in stem if token.startswith(_TASK_PREFIX)), None)
    parts = [task or "_".join(stem) or run_id]
    if stamped is not None:
        parts.append("-".join(stamped.group(1, 2, 3)))
    return _TITLE_SEPARATOR.join(parts)


def _title(run_id: str, verdict: dict[str, Any]) -> str:
    """The page's title: what was recorded, when, and what the graph decided about it.

    Args:
        run_id: The store's run id, read for its task token and its date.
        verdict: :func:`_verdict`'s mapping.

    Returns:
        The title, wrapped over as many lines as it needs.
    """
    line = _TITLE_SEPARATOR.join(
        (_run_label(run_id), f"triage: {_shown(verdict['triage'])}", f"release: {_shown(verdict['release'])}")
    )
    return "\n".join(textwrap.wrap(line, width=_TITLE_COLUMNS, break_long_words=True, break_on_hyphens=False) or [line])


def _wrapped(lines: Iterable[str]) -> list[str]:
    """Every block line folded to the block width, keeping its indent and its blank separators.

    Args:
        lines: The block lines.

    Returns:
        The folded lines. A line that is blank, or already inside the width, is returned unchanged.
    """
    out: list[str] = []
    for line in lines:
        if not line.strip() or len(line) <= _BLOCK_COLUMNS:
            out.append(line)
            continue
        indent = " " * (len(line) - len(line.lstrip(" ")) + 2)
        out += textwrap.wrap(
            line,
            width=_BLOCK_COLUMNS,
            subsequent_indent=indent,
            break_long_words=True,
            break_on_hyphens=False,
        )
    return out


def _ran_line(verdict: dict[str, Any]) -> str:
    """One line saying what each node did, so a node that raised is not read as one never asked.

    Args:
        verdict: :func:`_verdict`'s mapping.

    Returns:
        The line, in graph order, with any node the fold names but the graph does not appended.
    """
    ran = verdict.get("ran") or {}
    if not ran:
        return "ran: —  (VERDICT wrote no run state)"
    ordered = [node for node in GRAPH_ORDER if node in ran] + [node for node in ran if node not in GRAPH_ORDER]
    return "ran: " + "  ".join(f"{node}:{ran[node]}" for node in ordered)


def _blocks(  # noqa: C901 — one independent block per step, as report.md asks
    store: ProvStore, marks: dict[str, list[Entity]], drawn: set[str], provenance: dict[str, Any]
) -> list[str]:
    """The per-step blocks that accompany the shared axis.

    Every line goes through the PII rule: no matched text, in the blocks any more than in the lanes.

    Args:
        store: The provenance store.
        marks: :func:`_assertions_by_source`'s index, built once per report and shared with the lanes.
        drawn: The names of the lanes the page drew, so the rest can be reported as absent rather
            than silently missing.
        provenance: :func:`_provenance`'s mapping, summarised onto one line.

    Returns:
        The lines, folded to the block width, in report.md's order: the run id, the file and its
        provenance, what ran, the branch decisions
        with each branch's conclusion, flags and measurements, TAXONOMY's classification beside the
        resolved kinds and the hint reading for each, the top categories, the spans, what was NOT
        drawn and why, the transcript, and the verdict — with REDACT's outcome shown whatever the
        triage axis says.
    """
    steps = _steps(store)
    verdict = _verdict(store)
    lines: list[str] = []

    described = _file(store)
    lines.append(f"run: {_shown(provenance['run_id'])}")
    lines.append(f"file: {_shown(described['path'])}")
    lines.append(
        f"duration: {_shown(described['duration_s'])} s  "
        f"rate: {_shown(described['sample_rate'])} Hz  channels: {_shown(described['channels'])}"
    )
    lines.append(_provenance_line(provenance))
    lines.append(_ran_line(verdict))

    branches = _branches(store)
    lines.append("")
    lines.append("BRANCHES")
    if not branches:
        lines.append("  routing did not run; no branch was asked")
    for branch in sorted(branches, key=lambda name: BRANCHES.index(name) if name in BRANCHES else len(BRANCHES)):
        decision = branches[branch]
        lines.append(
            f"  {branch}: will_run={decision['will_run']} forced_by_hint={decision['forced_by_hint']} "
            f"kind_state={decision['kind_state']} why={decision['why']}"
        )
        lines.append(f"    verdict: {_shown(decision['verdict'])}")
        detail = steps.get(branch, {})
        measured = [f"{key}={_shown(detail[key])}" for key in _BRANCH_MEASURES.get(branch, ()) if key in detail]
        if measured:
            lines.append("    measured: " + "  ".join(measured))
        for flag in decision["flags"]:
            lines.append(f"    flag: {flag}")

    lines.append("")
    lines.append("TAXONOMY")
    screened = verdict.get("screened") or (steps.get("TAXONOMY", {}).get("kinds") or {})
    resolved = verdict.get("kinds") or {}
    agreement = verdict.get("agreement") or {}
    hints = verdict.get("hints") or {}
    if not screened:
        lines.append("  TAXONOMY did not classify this recording")
    for kind in sorted(screened):
        lines.append(
            f"  {kind}: screened={screened[kind]} resolved={_shown(resolved.get(kind))} "
            f"agreement={_shown(agreement.get(kind))} hint={_shown(hints.get(kind))}"
        )

    lines.append("")
    lines.append("TOP CATEGORIES")
    categories = _categories(store)
    if not categories:
        lines.append("  no window label set is in the store")
    for classifier, counts in categories.items():
        top = list(counts.items())[:_TOP_CATEGORIES]
        marker = f" (top {_TOP_CATEGORIES} of {len(counts)})" if len(counts) > _TOP_CATEGORIES else ""
        lines.append(f"  {classifier}{marker}: " + ", ".join(f"{label} ({n})" for label, n in top))
    airway = steps.get("AIRWAY", {}).get("by_label") or {}
    if airway:
        lines.append("  airway labels: " + ", ".join(f"{label} ({n})" for label, n in sorted(airway.items())))

    lines.append("")
    lines.append("SPANS")
    lines.append(f"  envelope spans: {len(_envelope_spans(store))}")
    lines.append(f"  phonation spans: {len(_spans_of_family(store, 'phonation', voice=False))}")
    lines.append(f"  speech spans: {len(_spans_of_family(store, 'speech'))}")
    lines.append(f"  voice spans: {len(_spans_of_family(store, 'phonation', voice=True))}")

    lines.append("")
    lines.append("ABSENT (a lane not drawn is not a measured absence)")
    absences = _absences(store)
    if absences:
        by_reading: dict[str, list[str]] = {}
        for derivative, (reading, raised) in absences.items():
            by_reading.setdefault(reading, []).append(f"{derivative} [{raised}]")
        for reading in sorted(by_reading):
            lines.append(f"  {reading}: " + ", ".join(sorted(by_reading[reading])))
    else:
        lines.append("  PREPROCESS reports no absent derivative")
    lane_absences = _lane_absences(store, drawn)
    if lane_absences:
        for lane, reason in lane_absences:
            lines.append(f"  lane not drawn — {lane}: {reason}")
    else:
        lines.append("  every declared lane was drawn")

    lines.append("")
    scanned, why_not = _scan_state(store)
    heading = (
        "TRANSCRIPT (marked words rendered as their category)"
        if scanned
        else "TRANSCRIPT (WITHHELD — every word rendered as [unscanned])"
    )
    lines.append(heading)
    if not scanned:
        lines.append(f"  {why_not}; an unscanned transcript is not a clean one")
    transcript = _transcript(store, marks)
    if not transcript:
        lines.append("  no consensus transcript is in the store")
    lines += ["  " + wrapped for wrapped in textwrap.wrap(transcript, width=_BLOCK_COLUMNS)]

    lines.append("")
    lines.append("VERDICT")
    lines.append(
        f"  triage: {_shown(verdict['triage'])}   release: {_shown(verdict['release'])}   "
        f"discard_ground: {_shown(verdict.get('discard_ground'))}"
    )
    redact = steps.get("REDACT")
    if redact is None:
        lines.append("  redact: did not run")
    else:
        lines.append(
            f"  redact: {redact.get('outcome')} — {redact.get('why')} "
            f"(redactions_n={_shown(redact.get('redactions_n'))})"
        )
    for reason in verdict.get("reasons") or []:
        lines.append(
            f"  reason: {reason.get('node')} {reason.get('outcome')} [{reason.get('kind')}] {reason.get('why')}"
        )
    if not (verdict.get("reasons") or []):
        lines.append("  no node contributed a reason")
    return _wrapped(lines)


def report(
    store: ProvStore, summary_dir: Path, config: TriageConfig, *, run_dir: Path | None = None
) -> dict[str, Path]:
    """Render one summary and one summary JSON from the store, writing nothing back to it.

    Both products are emitted on every file and every outcome, including a file ADMIT refused, where
    they say that and nothing else. Both carry element ids, so ``summary_dir`` belongs beside the
    store and never under the release tree.

    Args:
        store: The provenance store, read in full and never written.
        summary_dir: Where the two products go. Created if it does not exist.
        config: The triage configuration, read for ``report.format`` and the envelope stride.
        run_dir: Where the store's sidecar paths resolve against. Defaults to ``summary_dir``'s
            parent, which is the run root the two trees are siblings in.

    Returns:
        ``{"summary": <path>, "json": <path>}``.

    Raises:
        ValueError: If ``report.format`` names a form other than ``png`` or ``pdf``. A typo must not
            fall through to a silent default.
        ReportRenderError: If the summary could not be drawn. The JSON is written first and is
            complete by then, so it travels on the exception rather than being lost with the page.
    """
    fmt = str(config.require("report.format"))
    if fmt not in FORMATS:
        raise ValueError(f"report.format is {fmt!r}; the declared forms are {', '.join(FORMATS)}")
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_dir = Path(run_dir) if run_dir is not None else summary_dir.parent

    marks = _assertions_by_source(store)
    verdict = _verdict(store)
    provenance = _provenance(store, config, store.run_id)
    payload = {
        "file": _file(store),
        "verdict": verdict,
        "branches": _branches(store),
        "steps": _steps(store),
        "transcript": {"text": _transcript(store, marks), "words_n": len(_words(store))},
        "categories": _categories(store),
        "provenance": provenance,
    }
    json_path = summary_dir / f"{SUMMARY_STEM}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    title = _title(store.run_id, verdict)
    summary_path = summary_dir / f"{SUMMARY_STEM}.{fmt}"
    try:
        _render(store, marks, resolved_run_dir, config, provenance, title, summary_path, fmt)
    except Exception as error:  # noqa: BLE001 — any drawing failure keeps the product already written
        raise ReportRenderError(
            f"the summary could not be drawn ({type(error).__name__}: {error}); the JSON was written",
            {"json": json_path},
        ) from error
    return {"summary": summary_path, "json": json_path}


def _render(  # noqa: PLR0913 — every argument is one thing the page needs and none has a default
    store: ProvStore,
    marks: dict[str, list[Entity]],
    run_dir: Path,
    config: TriageConfig,
    provenance: dict[str, Any],
    title: str,
    path: Path,
    fmt: str,
) -> None:
    """Draw the summary in the declared form, over the shared time axis when there is a stream.

    A file ADMIT refused has no conditioned stream, so there is no axis to share; the blocks are the
    whole product, and they say so — including which lanes are missing and why.

    Args:
        store: The provenance store.
        marks: :func:`_assertions_by_source`'s index.
        run_dir: Where sidecar paths resolve against.
        config: The triage configuration.
        provenance: :func:`_provenance`'s mapping, summarised onto the page's second line.
        title: The figure's title, carrying the decision.
        path: Where the rendered summary goes.
        fmt: ``pdf`` for two pages — the panels, then the blocks — or ``png`` for one image
            carrying both.
    """
    from matplotlib import pyplot
    from matplotlib.backends.backend_pdf import PdfPages

    audio = _stream(store, run_dir)
    panels, drawn = ([], set[str]()) if audio is None else _panels(store, marks, run_dir, config)
    blocks = _blocks(store, marks, drawn, provenance)

    if fmt == "pdf":
        lanes = _text_figure([_NO_AXIS], title) if audio is None else plot_aligned_panels(audio, panels, title=title)
        with PdfPages(path) as pages:
            for figure in (lanes, _text_figure(blocks, title)):
                pages.savefig(figure, bbox_inches="tight")
                pyplot.close(figure)
        return

    if audio is None:
        figure = _text_figure(blocks, title)
    else:
        panels.append({"type": "text", "lines": blocks})
        figure = plot_aligned_panels(audio, panels, title=title)
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)


def _text_figure(lines: list[str], title: str) -> Figure:
    """One text-only figure, tall enough for every line it carries.

    Args:
        lines: The lines, drawn monospaced from the top left.
        title: The figure's title.

    Returns:
        The figure, not yet saved and not yet closed.
    """
    from matplotlib import pyplot

    height = max(MIN_FIGURE_HEIGHT_IN, TEXT_PANEL_INCHES_PER_LINE * len(lines))
    figure = pyplot.figure(figsize=(14.0, height))
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)
    figure.suptitle(title)
    return figure


def _stream(store: ProvStore, run_dir: Path) -> Audio | None:
    """The conditioned stream the layers are drawn over, or None when the store holds no readable one.

    Args:
        store: The provenance store.
        run_dir: Where sidecar paths resolve against.

    Returns:
        The audio, or None. A recording ADMIT refused leaves none, and so does a run whose sidecar
        tree has been moved away from its store.
    """
    for name in (_CONDITIONED_STREAM, _SOURCE_STREAM):
        try:
            _, audio = resolve_stream(store, run_dir, name)
            if audio.waveform.shape[0] != 1:
                audio = Audio(waveform=audio.waveform.mean(dim=0, keepdim=True), sampling_rate=audio.sampling_rate)
            return audio
        except Exception:  # noqa: BLE001, S112 — an unreadable stream is one absent layer, not a failed report
            continue
    return None
