"""The recording-level background, split around the session floor: own floor, SESSION, BACKGROUND.

Three steps, in this order, each writing one measurement every later reader takes from the store:

- :func:`write_own_floor` (PREPROCESS's ``background_floor`` block): the recording's own floor and
  level per band, from its ``plain`` and ``residual`` streams.
- :func:`write_session_floor` (the ``SESSION`` node): the floor of the recording's BIDS session,
  over its members' own floors, and the recording's level against its session's. A run that sees
  no siblings records its own floor as the floor it uses (``floor_source: recording``).
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
    path_attributes,
    software_agent,
    write_measurement,
)
from senselab.utils.prov_store import ProvStore

OWN_FLOOR = "background_floor"
"""PREPROCESS's measurement of the recording's own floor and level."""

SESSION_FLOOR = "session_floor"
"""SESSION's measurement: the session floor and the recording's level against its session's."""

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


def write_session_floor(store: ProvStore, members: Sequence[Mapping[str, Any]], *, session: str | None) -> str:
    """Write the recording's session floor, or the input it lacked.

    Args:
        store: The provenance store, read for the recording's own floor.
        members: Every member's own-floor record, this recording's included; empty for a run that sees
            no siblings.
        session: The session key, recorded as read.

    Returns:
        The measurement's id.
    """
    activity, software = _activity(store, SESSION_NODE, SESSION_FLOOR)
    own = find_measurement(store, OWN_FLOOR)
    if own is None or own.attributes.get("missing"):
        attributes: dict[str, Any] = {"missing": [OWN_FLOOR], "session": session}
    else:
        store.used(activity, own.id)
        attributes = {"session": session, **session_attributes(own.attributes, members)}
    return write_measurement(store, activity, software, name=SESSION_FLOOR, signal=PLAIN_STREAM, attributes=attributes)


def write_background(store: ProvStore, run_dir: Path, *, clips: Sequence[tuple[float, float]]) -> str:
    """Write the recording-level background against its session-informed floor, with its view sidecar.

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
    reading = measure_background(
        plain,
        residual=_named_stream(store, run_dir, RESIDUAL_STREAM),
        recording=_named_stream(store, run_dir, RECORDING_STREAM),
        clips=clips,
        session_db=None if band is None else np.asarray(band, dtype=np.float64),
    )
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    write_view_arrays(run_dir / VIEW_SIDECAR, reading)
    attributes = {
        **reading.record(),
        "floor_source": session.attributes.get("floor_source"),
        "view": path_attributes(VIEW_SIDECAR, run_dir),
    }
    return write_measurement(
        store, activity, software, name=BACKGROUND_MODEL, signal=PLAIN_STREAM, attributes=attributes
    )
