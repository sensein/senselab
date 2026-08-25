"""REPORT — one summary and one summary JSON per file, on every file and every outcome.

Runs last, after VERDICT, and is the only node that writes no elements: it reads the whole store and
renders it. A rendering is not evidence, so nothing downstream reads either product to learn a fact
the store does not already hold. Both products carry element ids, which are a join key back into the
store, so they sit beside it and never under the release tree. The design is in
``specs/20260817-triage-workflow-dag/report.md``.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.plotting.plotting import plot_aligned_panels
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import find_measurement, live_entities, resolve_stream
from senselab.utils.prov_store import Entity, ProvStore

NODE = "REPORT"
SUMMARY_STEM = "summary"
FORMATS = ("png", "pdf")

_CONDITIONED_STREAM = "plain"
_SOURCE_STREAM = "recording"
_GRAPH_ORDER = ("ADMIT", "PREPROCESS", "TAXONOMY", "routing", "AIRWAY", "SPEECH", "VOICE", "REDACT", "VERDICT")
_BRANCHES = ("AIRWAY", "SPEECH", "VOICE")
_CLASSIFIERS = ("yamnet", "ast", "hear")
_UNLABELLED = "unlabelled"
_SHA_LENGTH = 40
_BLOCK_COLUMNS = 168
_TOP_CATEGORIES = 6


def _redacted_text(store: ProvStore, word: Entity) -> str:
    """A word's renderable text: its category placeholder when the scan marked it, else the word.

    The store holds PII by design and the report carries element ids, so the report is not a released
    artifact — but no matched text may appear in it either way.

    Args:
        store: The provenance store.
        word: A ``word`` entity.

    Returns:
        ``"[<CATEGORY>]"`` when a live ``pii`` label assertion is derived from this word, else the
        word's own text.
    """
    for assertion in live_entities(store, "assertion"):
        attributes = assertion.attributes
        if attributes.get("verb") != "label" or attributes.get("label") != "pii":
            continue
        if word.id in store.derived_from(assertion.id):
            return f"[{attributes.get('category')}]"
    return str(word.attributes.get("text") or "")


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


def _transcript(store: ProvStore) -> str:
    """The consensus transcript with every marked word replaced by its category.

    Args:
        store: The provenance store.

    Returns:
        The redacted transcript, empty when the store holds no consensus words.
    """
    return " ".join(_redacted_text(store, word) for word in _words(store))


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


def _airway_labels(store: ProvStore, span: Entity) -> list[str]:
    """The airway labels a span carries, from the assertions derived from it.

    Args:
        store: The provenance store.
        span: An envelope span.

    Returns:
        The labels, sorted. Empty when nothing labelled the span.
    """
    labels = {
        str(assertion.attributes["label"])
        for assertion in live_entities(store, "assertion")
        if assertion.attributes.get("verb") == "label"
        and assertion.attributes.get("label") != "pii"
        and span.id in store.derived_from(assertion.id)
    }
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


def _panels(store: ProvStore, run_dir: Path, config: TriageConfig) -> list[dict[str, Any]]:
    """The summary's layers on one shared time axis, drawn from whatever the store holds.

    A layer whose derivative is absent is omitted; nothing raises for want of one, because
    report.md requires a product on every outcome including a file ADMIT refused.

    Args:
        store: The provenance store.
        run_dir: Where sidecar paths resolve against.
        config: The triage configuration, read for the envelope decimation stride only.

    Returns:
        Panel specifications for ``plot_aligned_panels``.
    """
    panels: list[dict[str, Any]] = [{"type": "waveform"}]

    envelope = find_measurement(store, "energy_envelope")
    if envelope is not None:
        path = Path(str(envelope.attributes.get("path") or ""))
        path = path if path.is_absolute() else run_dir / path
        if path.is_file():
            loaded = np.load(path)
            rate = float(envelope.attributes.get("sampling_rate") or 1.0)
            stride = max(1, int(rate * float(config.require("spectrogram.hop_ms")) / 1000.0))
            times = np.arange(0, len(loaded["envelope_dbfs"]), stride) / rate
            panels.append(
                {
                    "type": "features",
                    "style": "line",
                    "data": [
                        (times, loaded["envelope_dbfs"][::stride], "envelope dBFS", "steelblue"),
                        (times, loaded["floor_dbfs"][::stride], "floor dBFS", "firebrick"),
                    ],
                }
            )

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
        _segments((word.extent, _redacted_text(store, word)) for word in _words(store) if word.extent is not None),
    )
    panels += _lane(
        "airway",
        _segments(
            (span.extent, ", ".join(_airway_labels(store, span)) or _UNLABELLED)
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
    return panels


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
        The ids are the verdict entity itself and every entity the node's activities generated,
        which is what makes any number in the summary traceable to the assertion that produced it.
    """
    by_node = _verdict_entities(store)
    generated: dict[str, list[str]] = {}
    for entity in store.entities():
        activity_id = store.generated_by(entity.id)
        if activity_id is None:
            continue
        generated.setdefault(store.get_activity(activity_id).node, []).append(entity.id)
    ordered = sorted(by_node, key=lambda node: _GRAPH_ORDER.index(node) if node in _GRAPH_ORDER else len(_GRAPH_ORDER))
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


def _file(store: ProvStore) -> dict[str, Any]:
    """What the recording is, from ADMIT's stream entity.

    Args:
        store: The provenance store.

    Returns:
        ``{path, duration_s, sample_rate, channels}``, every field None when ADMIT wrote no stream.
    """
    found = [entity for entity in live_entities(store, "stream") if entity.attributes.get("name") == _SOURCE_STREAM]
    if not found:
        return {"path": None, "duration_s": None, "sample_rate": None, "channels": None}
    entity = found[-1]
    return {
        "path": entity.attributes.get("path"),
        "duration_s": None if entity.extent is None else float(entity.extent[1]),
        "sample_rate": entity.attributes.get("sampling_rate"),
        "channels": entity.attributes.get("channels"),
    }


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


def _blocks(store: ProvStore) -> list[str]:  # noqa: C901 — one independent block per step, as report.md asks
    """The per-step blocks that accompany the shared axis.

    Every line goes through the PII rule: no matched text, in the blocks any more than in the lanes.

    Args:
        store: The provenance store.

    Returns:
        The lines, in report.md's order: the branch decisions, each branch's conclusion and flags,
        TAXONOMY's classification beside the resolved kinds, and the verdict — with REDACT's outcome
        shown whatever the triage axis says.
    """
    lines: list[str] = []
    described = _file(store)
    lines.append(
        f"file: {described['path']}  duration: {described['duration_s']}s  "
        f"rate: {described['sample_rate']} Hz  channels: {described['channels']}"
    )

    branches = _branches(store)
    lines.append("")
    lines.append("BRANCHES")
    if not branches:
        lines.append("  routing did not run; no branch was asked")
    for branch in sorted(branches, key=lambda name: _BRANCHES.index(name) if name in _BRANCHES else len(_BRANCHES)):
        decision = branches[branch]
        lines.append(
            f"  {branch}: will_run={decision['will_run']} forced_by_hint={decision['forced_by_hint']} "
            f"kind_state={decision['kind_state']} why={decision['why']}"
        )
        lines.append(f"    verdict: {decision['verdict']}")
        for flag in decision["flags"]:
            lines.append(f"    flag: {flag}")

    steps = _steps(store)
    verdict = _verdict(store)
    lines.append("")
    lines.append("TAXONOMY")
    screened = verdict.get("screened") or (steps.get("TAXONOMY", {}).get("kinds") or {})
    resolved = verdict.get("kinds") or {}
    agreement = verdict.get("agreement") or {}
    if not screened:
        lines.append("  TAXONOMY did not classify this recording")
    for kind in sorted(screened):
        lines.append(
            f"  {kind}: screened={screened[kind]} resolved={resolved.get(kind)} agreement={agreement.get(kind)}"
        )

    lines.append("")
    lines.append("TOP CATEGORIES")
    categories = _categories(store)
    if not categories:
        lines.append("  no window label set is in the store")
    for classifier, counts in categories.items():
        top = list(counts.items())[:_TOP_CATEGORIES]
        lines.append("  " + classifier + ": " + ", ".join(f"{label} ({n})" for label, n in top))
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
    lines.append("TRANSCRIPT (marked words rendered as their category)")
    transcript = _transcript(store)
    if not transcript:
        lines.append("  no consensus transcript is in the store")
    lines += ["  " + wrapped for wrapped in textwrap.wrap(transcript, width=_BLOCK_COLUMNS)]

    lines.append("")
    lines.append("VERDICT")
    lines.append(
        f"  triage: {verdict['triage']}   release: {verdict['release']}   "
        f"discard_ground: {verdict.get('discard_ground')}"
    )
    redact = steps.get("REDACT")
    if redact is None:
        lines.append("  redact: did not run")
    else:
        lines.append(f"  redact: {redact.get('outcome')} — {redact.get('why')}")
    for reason in verdict.get("reasons") or []:
        lines.append(
            f"  reason: {reason.get('node')} {reason.get('outcome')} [{reason.get('kind')}] {reason.get('why')}"
        )
    if not (verdict.get("reasons") or []):
        lines.append("  no node contributed a reason")
    return lines


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
    """
    fmt = str(config.require("report.format"))
    if fmt not in FORMATS:
        raise ValueError(f"report.format is {fmt!r}; the declared forms are {', '.join(FORMATS)}")
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_dir = Path(run_dir) if run_dir is not None else summary_dir.parent

    verdict = _verdict(store)
    payload = {
        "file": _file(store),
        "verdict": verdict,
        "branches": _branches(store),
        "steps": _steps(store),
        "transcript": {"text": _transcript(store), "words_n": len(_words(store))},
        "categories": _categories(store),
        "provenance": _provenance(store, config, store.run_id),
    }
    json_path = summary_dir / f"{SUMMARY_STEM}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    title = f"triage: {verdict['triage']}   ·   release: {verdict['release']}   ·   run {store.run_id}"
    summary_path = summary_dir / f"{SUMMARY_STEM}.{fmt}"
    _render(store, resolved_run_dir, config, title, summary_path)
    return {"summary": summary_path, "json": json_path}


def _render(store: ProvStore, run_dir: Path, config: TriageConfig, title: str, path: Path) -> None:
    """Draw the summary, on the shared time axis when there is a stream and on prose alone when not.

    A file ADMIT refused has no conditioned stream, so there is no axis to share; the blocks are the
    whole product, and they say that.

    Args:
        store: The provenance store.
        run_dir: Where sidecar paths resolve against.
        config: The triage configuration.
        title: The figure's title, carrying the decision.
        path: Where the rendered summary goes.
    """
    from matplotlib import pyplot

    blocks = _blocks(store)
    audio = _stream(store, run_dir)
    if audio is None:
        figure = pyplot.figure(figsize=(14.0, max(4.0, 0.18 * len(blocks) * 1.8)))
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.01, 0.98, "\n".join(blocks), va="top", ha="left", family="monospace", fontsize=8)
        figure.suptitle(title)
    else:
        panels = _panels(store, run_dir, config)
        panels.append({"type": "text", "lines": blocks})
        figure = plot_aligned_panels(audio, panels, title=title)
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)


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
