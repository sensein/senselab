"""One review record per recording: its decision, every evidence item, overlays and a spectrogram.

The decision and evidence come from :func:`~senselab.audio.workflows.triage.decision_tables.decision_rows`,
so the page cannot disagree with the decision tables. A speech task's transcript, PII and redaction
view is supplied by the caller, which reads it the way the free-speech review page does.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from senselab.audio.workflows.triage.decision_tables import decision_rows, item_group
from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.recording_vectors import RUN_SUBDIR, STORE_NAME, StoreView, read_store
from senselab.audio.workflows.triage.review_page.spectrogram import pack, quantised_levels, settings

READINGS = ("airway_breath_reading", "airway_cough_reading", "voice_phonation_reading")
BACKGROUND_MODEL = "background_model"
BACKGROUND_SPEECH = "airway_background_speech"
STREAMS = ("recording", "plain", "enhanced", "residual", "task_plain", "task_enhanced", "task_redacted")
RELEASED_DIR = "released"
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3")

SpeechReader = Callable[[Path], "Mapping[str, Any] | None"]


def branch_of(family: str | None) -> str | None:
    """The branch that owns a declared family.

    Args:
        family: The declared family.

    Returns:
        ``AIRWAY``, ``SPEECH`` or ``VOICE``, or None for an undeclared or unknown family.
    """
    for branch, families in EXPECTATIONS.items():
        if family in families:
            return branch
    return None


def _span(value: Any, leading: bool = False) -> tuple[float, float] | None:  # noqa: ANN401 -- any shape
    """A ``(start, end)`` out of the span shapes the readings write, else None.

    Args:
        value: A ``{start_s, end_s}`` mapping, a ``[start, end]`` pair, or a ``[peak, start, end]`` burst.
        leading: Read the first two fields of a longer row as ``(start, end)``.

    Returns:
        The span, or None.
    """
    if isinstance(value, Mapping):
        start, end = value.get("start_s"), value.get("end_s")
    elif leading and isinstance(value, (list, tuple)) and len(value) >= 2:
        start, end = value[0], value[1]
    elif isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
        start, end = value[1], value[2]
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value[0], value[1]
    else:
        return None
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
        return None
    return round(float(start), 3), round(float(end), 3)


def _spans(values: Any, leading: bool = False) -> list[tuple[float, float]]:  # noqa: ANN401 -- any shape
    if not isinstance(values, (list, tuple)):
        return []
    return [span for span in (_span(v, leading) for v in values) if span is not None]


def _find(tree: Any, key: str) -> list[Any]:  # noqa: ANN401 -- a store attribute is any shape
    """Every value stored under ``key`` anywhere in a nested mapping."""
    found: list[Any] = []
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            if k == key:
                found.append(v)
            found.extend(_find(v, key))
    elif isinstance(tree, list):
        for v in tree:
            found.extend(_find(v, key))
    return found


EVENT_KEYS = ("bursts_s", "event_spans_s", "holds")


def overlays(view: StoreView) -> dict[str, list[list[Any]]]:
    """What the page draws over the spectrogram.

    Args:
        view: The store.

    Returns:
        ``events`` (the task readings' events), ``activity`` (the background reading's active regions)
        and ``issues`` (impulses, faults and background speech, each ``[start, end, kind]``).
    """
    events: list[list[Any]] = []
    for name in READINGS:
        reading = view.last("measurement", name=name)
        if reading is None:
            continue
        for key in EVENT_KEYS:
            for values in _find(reading.attributes, key):
                events.extend([s, e] for s, e in _spans(values))
    activity: list[list[Any]] = []
    issues: list[list[Any]] = []
    background = view.last("measurement", name=BACKGROUND_MODEL)
    if background is not None:
        attributes = background.attributes
        activity = [[s, e] for s, e in _spans(attributes.get("regions"))]
        issues.extend([s, e, "impulse"] for s, e in _spans(attributes.get("impulses")))
        faults = attributes.get("faults")
        if isinstance(faults, Mapping):
            for kind, values in faults.items():
                spans = _spans(values) if isinstance(values, list) else _spans([values])
                issues.extend([s, e, str(kind)] for s, e in spans)
    heard = view.last("measurement", name=BACKGROUND_SPEECH)
    if heard is not None:
        for key in ("speech_windows", "voice_runs"):
            issues.extend([s, e, "background_speech"] for s, e in _spans(heard.attributes.get(key), leading=True))
    return {"events": sorted(events), "activity": activity, "issues": sorted(issues)}


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def stream_paths(view: StoreView, run_root: Path, root: Path) -> dict[str, str]:
    """The audio a reviewer can listen to, by stream name, relative to the scan root.

    Args:
        view: The store.
        run_root: The run root.
        root: The scan root.

    Returns:
        Each stored stream's path, plus ``released`` where the run released an audio file.
    """
    out: dict[str, str] = {}
    for name in STREAMS:
        stream = view.last("stream", name=name)
        if stream is None or not stream.attributes.get("path"):
            continue
        path = Path(str(stream.attributes["path"]))
        if not path.is_absolute():
            path = run_root / RUN_SUBDIR / path
        out[name] = _relative(path, root)
    released = run_root / RELEASED_DIR
    if released.is_dir():
        audio = sorted(p for p in released.iterdir() if p.suffix in AUDIO_SUFFIXES)
        if audio:
            out["released"] = _relative(audio[0], root)
    return out


def spectrogram_of(view: StoreView, run_root: Path) -> tuple[str | None, str | None]:
    """The quantised spectrogram of the first stored stream named in the settings.

    Args:
        view: The store.
        run_root: The run root.

    Returns:
        ``(packed levels, stream name)``, or ``(None, None)`` where no stream decodes.
    """
    for name in settings()["spectrogram"]["stream_order"]:
        stream = view.last("stream", name=name)
        if stream is None or not stream.attributes.get("path"):
            continue
        path = Path(str(stream.attributes["path"]))
        if not path.is_absolute():
            path = run_root / RUN_SUBDIR / path
        try:
            samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        except (OSError, RuntimeError, sf.LibsndfileError):
            continue
        return pack(quantised_levels(np.asarray(samples).T, float(rate))), str(name)
    return None, None


def _value(text: str | None) -> Any:  # noqa: ANN401 -- an evidence value is any JSON scalar
    if text is None:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def review_record(run_root: Path, root: Path, speech: SpeechReader | None = None) -> dict[str, Any] | None:
    """One recording's review record.

    Args:
        run_root: The directory holding ``run/store.jsonl``.
        root: The scan root every path is given relative to.
        speech: Reads a SPEECH task's transcript view, or None to carry none.

    Returns:
        The record, or None where the store holds no fold.
    """
    rows = decision_rows(run_root, root)
    if rows is None:
        return None
    decision, evidence = rows
    view = read_store(run_root / RUN_SUBDIR / STORE_NAME)
    recording = view.last("stream", name="recording")
    spec, spec_stream = spectrogram_of(view, run_root)
    family = decision.get("declared_family")
    branch = branch_of(family)
    start, end = decision.get("extent_start_s"), decision.get("extent_end_s")
    return {
        "stem": decision["stem"],
        "participant": decision["participant"],
        "session": decision["session"],
        "task": decision["task"],
        "family": family,
        "branch": branch,
        "verdict": decision["verdict"],
        "release": decision["release"],
        "reason": decision["reason"],
        "reasons": decision["reasons"],
        "run_status": decision["run_status"],
        "missing": decision["missing"],
        "annotations": decision["annotations"],
        "extent": None if start is None or end is None else [start, end],
        "duration_s": recording.extent[1] if recording is not None and recording.extent else None,
        "evidence": [
            {
                "name": item["name"],
                "group": item.get("group") or item_group(item["name"]),
                "value": _value(item["value"]),
                "unit": item["unit"],
                "comparison": item["comparison"],
                "threshold": _value(item["threshold"]),
                "effect": item["effect"],
                "decisive": item["decisive"],
            }
            for item in evidence
        ],
        "spec": spec,
        "spec_stream": spec_stream,
        "overlay": overlays(view),
        "streams": stream_paths(view, run_root, root),
        "figure": decision["figure_path"],
        "speech": speech(run_root) if speech is not None and branch == "SPEECH" else None,
        "commit": decision["commit"],
        "config_hash": decision["config_hash"],
    }


def records(run_roots: Iterable[Path], root: Path, speech: SpeechReader | None = None) -> Iterable[dict[str, Any]]:
    """Every readable recording's review record.

    Args:
        run_roots: The run roots.
        root: The scan root.
        speech: The SPEECH transcript reader, or None.

    Yields:
        Each record, skipping stores with no fold.
    """
    for run_root in run_roots:
        record = review_record(run_root, root, speech)
        if record is not None:
            yield record
