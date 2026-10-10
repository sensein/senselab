"""The recording-level background, split around the session floor: own floor, SESSION, BACKGROUND.

Three steps, in this order, each writing one measurement every later reader takes from the store:

- :func:`write_own_floor` (PREPROCESS's ``background_floor`` block): the recording's own floor and
  level per band, from its ``plain`` and ``residual`` streams.
- :func:`write_speech_residual` and :func:`write_session_floor` (the ``SESSION`` node): the
  recording's residual background, foreground and quality; the floor of its BIDS session over its
  members' own floors; the session background over its speech tasks' residuals; and the recording's
  level against its session's. A run that sees no siblings records its own floor as the floor it
  uses (``floor_source: recording``) and its own reading as its session background's only member.
- :func:`write_background` (the ``BACKGROUND`` node): activity regions, impulses, hum and faults,
  read against the session-informed floor, with the arrays branches read in a sidecar.

The parameters are in ``data/background_model.yaml``; the design is
``specs/20261007-task-events-in-background/design.md``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from senselab.audio.workflows.triage.background_model import (
    BACKGROUND_MODEL,
    PLAIN_STREAM,
    RECORDING_STREAM,
    RESIDUAL_STREAM,
    _named_stream,
    background_model_parameters,
    measure_background,
    own_floor_of,
    session_floor_db,
    write_view_arrays,
)
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    lexical_words,
    path_attributes,
    software_agent,
    word_hull,
    write_measurement,
)
from senselab.audio.workflows.triage.routing_analysis.families import task_family, task_id_of
from senselab.audio.workflows.triage.session_background import (
    background_floor_for,
    session_background_of,
    session_background_parameters,
    speech_residual_of,
)
from senselab.utils.prov_store import ProvStore

OWN_FLOOR = "background_floor"
"""PREPROCESS's measurement of the recording's own floor and level."""

SESSION_FLOOR = "session_floor"
"""SESSION's measurement: the session floor, the session background and the recording's level against its session's."""

SPEECH_RESIDUAL = "speech_residual"
"""SESSION's per-recording measurement: the residual background, the foreground and the quality."""

ENHANCED_STREAM = "enhanced"
ENHANCED_YAMNET = "enhanced_yamnet_scores"

SESSION_NODE = "SESSION"
BACKGROUND_NODE = "BACKGROUND"
PREPROCESS_NODE = "PREPROCESS"

FLOOR_FROM_SESSION = "session"
FLOOR_FROM_RECORDING = "recording"

VIEW_SIDECAR = "derivatives/background_view.npz"
"""The arrays a branch reads the background through, beside the ``background_model`` measurement."""

_SESSION_KEY = re.compile(r"(sub-[^_]+)_(ses-[^_]+)")


def session_key(stem: str) -> str | None:
    """The BIDS session a recording belongs to, ``sub-<label>_ses-<label>``, or None.

    Args:
        stem: The recording's file stem.

    Returns:
        The participant and session labels, or None where the stem names no session.
    """
    found = _SESSION_KEY.search(stem)
    return f"{found.group(1)}_{found.group(2)}" if found else None


def _activity(store: ProvStore, node: str, step: str) -> tuple[str, str]:
    software = software_agent(store)
    activity = store.activity(node=node, step=step, parameters={"version": background_model_parameters()["version"]})
    store.was_associated_with(activity, software)
    return activity, software


def write_own_floor(store: ProvStore, run_dir: Path) -> str:
    """Write the recording's own floor and level (:func:`own_floor_of`), or the inputs it lacked.

    Args:
        store: The provenance store, read for the ``plain`` and ``residual`` streams.
        run_dir: The run directory their paths are relative to.

    Returns:
        The measurement's id.
    """
    activity, software = _activity(store, PREPROCESS_NODE, OWN_FLOOR)
    plain = _named_stream(store, run_dir, PLAIN_STREAM)
    attributes = (
        {"missing": [PLAIN_STREAM]}
        if plain is None
        else own_floor_of(plain, _named_stream(store, run_dir, RESIDUAL_STREAM))
    )
    return write_measurement(store, activity, software, name=OWN_FLOOR, signal=PLAIN_STREAM, attributes=attributes)


def session_attributes(own: Mapping[str, Any], members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What SESSION records for one recording, from its own floor and its session's members'.

    Args:
        own: The recording's own-floor record.
        members: Every member's own-floor record, this recording's included; empty where the run
            sees no siblings.

    Returns:
        ``floor_source`` (``session`` where the session floor could be read, else ``recording``),
        ``band_db`` (the session floor, or None), ``members_n`` and ``used_n``, ``level_rel_db``
        (the recording's active level less the session's, or None) and ``floor_rel_db`` (its own
        broadband floor less the session's, or None).
    """
    session = session_floor_db([dict(member) for member in members]) if members else None
    band = None if session is None else session["band_db"]
    if session is not None and list(session["band_edges_hz"]) != list(own.get("band_edges_hz") or []):
        band = None
    level_rel = None
    if session is not None and session["active_level_db"] is not None and own.get("active_level_db") is not None:
        level_rel = float(own["active_level_db"]) - float(session["active_level_db"])
    floor_rel = None
    if band is not None and own.get("own_db"):
        floor_rel = _broadband(own["own_db"]) - _broadband(band)
    return {
        "floor_source": FLOOR_FROM_SESSION if band is not None else FLOOR_FROM_RECORDING,
        "band_db": band,
        "members_n": 0 if session is None else int(session["members_n"]),
        "used_n": 0 if session is None else int(session["used_n"]),
        "level_rel_db": None if level_rel is None else round(level_rel, 2),
        "floor_rel_db": None if floor_rel is None else round(floor_rel, 2),
    }


def _broadband(band_db: Sequence[float]) -> float:
    return float(10.0 * np.log10(np.sum(10.0 ** (np.asarray(band_db, dtype=np.float64) / 10.0)) + 1e-12))


def _recording_stream_stem(store: ProvStore) -> str:
    found = [
        s
        for s in store.entities("stream")
        if s.attributes.get("name") == RECORDING_STREAM and not store.is_invalidated(s.id)
    ]
    return Path(str(found[-1].attributes.get("path") or "")).stem if found else ""


def declared_family(store: ProvStore) -> str | None:
    """The declared family off the ``recording`` stream's path.

    Args:
        store: The provenance store.

    Returns:
        The family of the stem's task id; None where no recording stream names one.
    """
    stem = _recording_stream_stem(store)
    return task_family(task_id_of(stem)) if stem else None


def _enhanced_speech(store: ProvStore, run_dir: Path) -> tuple[float | None, list[tuple[float, float]]]:
    """The enhanced stream's YAMNet speech: the highest speech-label score, and the windows reaching the bound."""
    import json  # noqa: PLC0415

    from senselab.audio.tasks.classification.label_scores import label_scores  # noqa: PLC0415
    from senselab.audio.workflows.triage.background_speech import background_speech_parameters  # noqa: PLC0415

    found = find_measurement(store, ENHANCED_YAMNET)
    path = run_dir / str(found.attributes.get("path") or "") if found is not None else None
    if path is None or not path.is_file():
        return None, []
    labels = background_speech_parameters().speech_labels
    least = float(session_background_parameters()["foreground"]["enhanced_speech_min"])
    best, windows = 0.0, []
    for window in json.loads(path.read_text()):
        scores = {key: float(value) for pair in label_scores(window) for key, value in pair.items()}
        score = max((scores.get(label, 0.0) for label in labels), default=0.0)
        best = max(best, score)
        if score >= least:
            windows.append((float(window.get("start", 0.0)), float(window.get("end", 0.0))))
    return best, windows


def speech_residual_attributes(store: ProvStore, run_dir: Path) -> dict[str, Any]:
    """The recording's :func:`~senselab.audio.workflows.triage.session_background.speech_residual_of` record.

    Args:
        store: The provenance store, read for the plain, enhanced and residual streams, the consensus words
            and the enhanced stream's YAMNet windows.
        run_dir: The run directory their paths are relative to.

    Returns:
        The record with the recording's ``stem``; ``missing`` naming the absent streams where any is.
    """
    from senselab.audio.workflows.triage.residue import is_non_lexical  # noqa: PLC0415

    names = (PLAIN_STREAM, ENHANCED_STREAM, RESIDUAL_STREAM)
    streams = {name: _named_stream(store, run_dir, name) for name in names}
    missing = [name for name in names if streams[name] is None]
    plain, enhanced, residual = streams[PLAIN_STREAM], streams[ENHANCED_STREAM], streams[RESIDUAL_STREAM]
    if missing or plain is None or enhanced is None or residual is None:
        return {"missing": missing}
    words = [
        word_hull(word)
        for word in lexical_words(store)
        if not is_non_lexical(str(word.attributes.get("text") or ""), vocal_task=True)
    ]
    best, windows = _enhanced_speech(store, run_dir)
    return {
        "stem": _recording_stream_stem(store),
        **speech_residual_of(
            plain,
            enhanced,
            residual,
            speech=sorted([w for w in words if w[1] > w[0]] + windows),
            lexical_words_n=len(words),
            enhanced_speech_max=best,
            family=declared_family(store),
        ),
    }


def write_speech_residual(store: ProvStore, run_dir: Path) -> dict[str, Any]:
    """Write the recording's residual background, foreground and quality (:func:`speech_residual_attributes`).

    Args:
        store: The provenance store.
        run_dir: The run directory the streams' paths are relative to.

    Returns:
        The record written.
    """
    software = software_agent(store)
    activity = store.activity(
        node=SESSION_NODE, step=SPEECH_RESIDUAL, parameters={"version": session_background_parameters()["version"]}
    )
    store.was_associated_with(activity, software)
    attributes = speech_residual_attributes(store, run_dir)
    write_measurement(store, activity, software, name=SPEECH_RESIDUAL, signal=RESIDUAL_STREAM, attributes=attributes)
    return attributes


def write_session_floor(
    store: ProvStore,
    members: Sequence[Mapping[str, Any]],
    *,
    session: str | None,
    background_members: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Write the recording's session floor and session background, or the input it lacked.

    Args:
        store: The provenance store, read for the recording's own floor.
        members: Every member's own-floor record, this recording's included; empty for a run that sees
            no siblings.
        session: The session key, recorded as read.
        background_members: Every member's :data:`SPEECH_RESIDUAL` record, this recording's included.

    Returns:
        The measurement's id. ``session_background`` is
        :func:`~senselab.audio.workflows.triage.session_background.session_background_of` over the members
        whose record is not ``missing``.
    """
    activity, software = _activity(store, SESSION_NODE, SESSION_FLOOR)
    own = find_measurement(store, OWN_FLOOR)
    background = session_background_of([dict(m) for m in background_members if not m.get("missing")])
    if own is None or own.attributes.get("missing"):
        attributes: dict[str, Any] = {"missing": [OWN_FLOOR], "session": session, "session_background": background}
    else:
        store.used(activity, own.id)
        attributes = {
            "session": session,
            **session_attributes(own.attributes, members),
            "session_background": background,
        }
    return write_measurement(store, activity, software, name=SESSION_FLOOR, signal=PLAIN_STREAM, attributes=attributes)


def write_background(store: ProvStore, run_dir: Path, *, clips: Sequence[tuple[float, float]]) -> str:
    """Write the recording-level background against its session-informed floor, with its view sidecar.

    The floor is the session background where ``data/session_background.yaml`` ``decides`` and the
    family's kind is one it decides for
    (:func:`~senselab.audio.workflows.triage.session_background.background_floor_for`); the measurement's
    ``floor_reference`` records which floor was used and any fallback.

    Args:
        store: The provenance store, read for the session floor and the streams.
        run_dir: The run directory the streams' paths are relative to.
        clips: The clip spans PREPROCESS kept.

    Returns:
        The ``background_model`` measurement's id. Where the session floor or the plain stream is
        absent it carries ``missing`` and nothing else, and no sidecar is written.
    """
    activity, software = _activity(store, BACKGROUND_NODE, BACKGROUND_MODEL)
    session = find_measurement(store, SESSION_FLOOR)
    plain = _named_stream(store, run_dir, PLAIN_STREAM)
    missing = [
        name
        for name, there in (
            (SESSION_FLOOR, session is not None and not session.attributes.get("missing")),
            (PLAIN_STREAM, plain is not None),
        )
        if not there
    ]
    if missing or session is None or plain is None:
        return write_measurement(
            store, activity, software, name=BACKGROUND_MODEL, signal=PLAIN_STREAM, attributes={"missing": missing}
        )
    store.used(activity, session.id)
    band = session.attributes.get("band_db")
    edges = [float(e) for e in background_model_parameters()["band_edges_hz"] if e <= plain[1] / 2.0]
    background, reference = background_floor_for(
        declared_family(store), session.attributes.get("session_background"), edges
    )
    reading = measure_background(
        plain,
        residual=_named_stream(store, run_dir, RESIDUAL_STREAM),
        recording=_named_stream(store, run_dir, RECORDING_STREAM),
        clips=clips,
        session_db=None if band is None else np.asarray(band, dtype=np.float64),
        background_db=background,
    )
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    write_view_arrays(run_dir / VIEW_SIDECAR, reading)
    attributes = {
        **reading.record(),
        "floor_source": session.attributes.get("floor_source"),
        "floor_reference": reference,
        "view": path_attributes(VIEW_SIDECAR, run_dir),
    }
    return write_measurement(
        store, activity, software, name=BACKGROUND_MODEL, signal=PLAIN_STREAM, attributes=attributes
    )
