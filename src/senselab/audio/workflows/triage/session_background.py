"""The session background read off the residual of the session's speech tasks, and the recording's quality.

Per recording, :func:`speech_residual_of` reads the plain, enhanced and residual streams in the band
frames of :mod:`~senselab.audio.workflows.triage.background_model`: whether the enhanced stream carries
speech (lexical consensus words, or enhanced-stream YAMNet speech), the residual's level per band and
broadband, the foreground (the enhanced stream over speech frames), and the quality differences
foreground less residual and plain less residual over the whole file, speech frames and non-speech
frames. Per session, :func:`session_background_of` summarises the readings of its lexical speech tasks
that carry speech. Every parameter is in ``data/session_background.yaml``; the design is
``specs/20261007-task-events-in-background/design.md`` ("Session background from speech-task residuals").
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from senselab.audio.workflows.triage.background_model import BandFrames, background_model_parameters, band_frames
from senselab.audio.workflows.triage.routing_analysis.families import (
    AIRWAY_ELICITING,
    LEXICAL_SPEECH,
    SYLLABLE_REPETITION,
    VOICE_ELICITING,
)

SESSION_BACKGROUND_PATH = Path(__file__).parent / "data" / "session_background.yaml"

NO_FOREGROUND = "not applicable: no foreground"
"""The quality measure of a recording whose enhanced stream carries no speech."""

NO_SPEECH_TASK = "no qualifying speech task"
"""Why a session has no session background: none of its lexical tasks' enhanced streams carries speech."""

SPEECH_RESIDUAL_SOURCE = "speech_residual"
"""The floor source BACKGROUND records where the session background is the floor."""

FRAME_SETS = ("whole", "speech", "non_speech")

KINDS: dict[str, frozenset[str]] = {
    "lexical_speech": LEXICAL_SPEECH,
    "airway": AIRWAY_ELICITING,
    "voice": VOICE_ELICITING,
    "syllable_repetition": SYLLABLE_REPETITION,
}
"""The declared kinds ``decides_kinds`` names, by their families."""

Signal = tuple[np.ndarray, int]
Span = tuple[float, float]


@functools.cache
def session_background_parameters() -> dict[str, Any]:
    """The parameters of ``data/session_background.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(SESSION_BACKGROUND_PATH.read_text()) or {})


def kind_of(family: str | None) -> str | None:
    """The declared kind of a family, as ``decides_kinds`` names it.

    Args:
        family: The declared family, or None.

    Returns:
        ``lexical_speech``, ``airway``, ``voice`` or ``syllable_repetition``; None for any other family.
    """
    return next((kind for kind, families in KINDS.items() if family in families), None)


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def _difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else _round(a - b)


def _level(series: np.ndarray, mask: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(series[mask], percentile)) if mask.any() else None


def speech_mask(times_s: np.ndarray, spans: Sequence[Span]) -> np.ndarray:
    """The frames whose centre lies in a speech span.

    Args:
        times_s: Each frame's centre.
        spans: The speech spans.

    Returns:
        A boolean mask over the frames.
    """
    mask = np.zeros(len(times_s), dtype=bool)
    for start, end in spans:
        mask |= (times_s >= start) & (times_s <= end)
    return mask


def _aligned(frames: Sequence[BandFrames]) -> list[BandFrames]:
    n = min(len(f.times_s) for f in frames)
    return [BandFrames(f.times_s[:n], f.band_db[:n], f.level_db[:n]) for f in frames]


def speech_residual_of(
    plain: Signal,
    enhanced: Signal,
    residual: Signal,
    *,
    speech: Sequence[Span],
    lexical_words_n: int,
    enhanced_speech_max: float | None,
    family: str | None,
    p: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One recording's residual background, foreground and quality.

    Args:
        plain: The plain stream.
        enhanced: The enhanced stream.
        residual: The residual stream.
        speech: Where the recording holds speech: the lexical words' hulls and the enhanced-stream YAMNet
            speech windows.
        lexical_words_n: The consensus lexical words.
        enhanced_speech_max: The highest speech-label score of any enhanced-stream YAMNet window, or None
            where none was stored.
        family: The declared family, or None.
        p: The parameters; ``data/session_background.yaml`` when None.

    Returns:
        ``family`` and ``kind``; ``foreground``, whether the enhanced stream carries speech (at least
        ``foreground.lexical_words_min`` lexical words or a window reaching ``foreground.enhanced_speech_min``,
        over at least ``foreground.speech_frames_min_s`` of speech frames); ``member``, whether the reading
        counts toward its session background (a lexical speech task with foreground); ``speech_s`` and
        ``non_speech_s``; ``band_edges_hz`` and ``residual_band_db``, the residual's per-band level over its
        live frames; ``residual_db``, ``plain_db`` and ``enhanced_db``, each stream's level over
        :data:`FRAME_SETS` (``residual_db`` also its spread percentiles); ``foreground_db``, the enhanced
        level over speech frames; ``foreground_minus_residual_db``, that less the residual level over each
        frame set; ``plain_minus_residual_db``, the per-frame plain-over-residual level's ``level.percentile``
        over each frame set; and ``not_applicable``, :data:`NO_FOREGROUND` where there is no foreground, in
        which case every foreground and speech-frame value is None.
    """
    q = p or session_background_parameters()
    bg = background_model_parameters()
    pct = float(q["level"]["percentile"])
    plain_f, enhanced_f, residual_f = _aligned([band_frames(s, bg) for s in (plain, enhanced, residual)])
    hop = float(bg["hop_s"])
    live = plain_f.level_db >= float(bg["digital_floor_dbfs"])
    in_speech = speech_mask(plain_f.times_s, speech) & live
    out_speech = ~in_speech & live
    heard = lexical_words_n >= int(q["foreground"]["lexical_words_min"]) or (
        enhanced_speech_max is not None and enhanced_speech_max >= float(q["foreground"]["enhanced_speech_min"])
    )
    foreground = heard and in_speech.sum() * hop >= float(q["foreground"]["speech_frames_min_s"])
    sets = {"whole": live, "speech": in_speech if foreground else np.zeros_like(live), "non_speech": out_speech}
    residual_live = residual_f.level_db >= float(bg["digital_floor_dbfs"])
    lo, hi = (float(v) for v in q["level"]["spread_percentiles"])

    def levels(frames: BandFrames) -> dict[str, float | None]:
        return {name: _round(_level(frames.level_db, mask, pct)) for name, mask in sets.items()}

    residual_db: dict[str, float | None] = levels(residual_f)
    residual_db["p_low"] = _round(_level(residual_f.level_db, live, lo))
    residual_db["p_high"] = _round(_level(residual_f.level_db, live, hi))
    fg = _level(enhanced_f.level_db, sets["speech"], pct) if foreground else None
    diff = plain_f.level_db - residual_f.level_db
    rate = plain[1]
    return {
        "family": family,
        "kind": kind_of(family),
        "lexical_words_n": int(lexical_words_n),
        "enhanced_speech_max": _round(enhanced_speech_max, 3),
        "foreground": bool(foreground),
        "member": bool(foreground and family in LEXICAL_SPEECH),
        "speech_s": round(float(in_speech.sum() * hop), 3),
        "non_speech_s": round(float(out_speech.sum() * hop), 3),
        "band_edges_hz": [float(e) for e in bg["band_edges_hz"] if e <= rate / 2.0],
        "residual_band_db": [
            round(float(v), 2)
            for v in (
                np.percentile(residual_f.band_db[residual_live], pct, axis=0)
                if residual_live.any()
                else np.full(residual_f.band_db.shape[1], float(bg["digital_floor_dbfs"]))
            )
        ],
        "residual_db": residual_db,
        "plain_db": levels(plain_f),
        "enhanced_db": levels(enhanced_f),
        "foreground_db": _round(fg),
        "foreground_minus_residual_db": {name: _difference(fg, residual_db[name]) for name in FRAME_SETS},
        "plain_minus_residual_db": {name: _round(_level(diff, mask, pct)) for name, mask in sets.items()},
        "not_applicable": None if foreground else NO_FOREGROUND,
    }


def session_background_of(readings: Sequence[Mapping[str, Any]], p: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The session background and quality over the readings of one BIDS session's recordings.

    Args:
        readings: Each member's :func:`speech_residual_of` record, carrying its ``stem``.
        p: The parameters; ``data/session_background.yaml`` when None.

    Returns:
        ``source`` (:data:`SPEECH_RESIDUAL_SOURCE`, or None where the session has none) and ``fallback``
        (:data:`NO_SPEECH_TASK` where it has none, else None); ``candidates_n``, the lexical speech tasks
        read, and ``members``, the stems of those with foreground and matching bands; ``band_edges_hz`` and
        ``band_db``, the ``session.statistic`` of the members' residual band levels; ``level_db``, the
        statistic of their whole-file residual levels; ``spread_db``, the members' interquartile range of
        that level and the statistic of their within-recording spread percentiles; and ``quality``, the
        statistic of the members' foreground, foreground less residual and plain less residual, or
        :data:`NO_FOREGROUND` where the session has no member.
    """
    q = p or session_background_parameters()
    reducer = {"median": np.median, "mean": np.mean}[q["session"]["statistic"]]
    candidates = [r for r in readings if r.get("kind") == "lexical_speech"]
    qualifying = [r for r in candidates if r.get("member")]
    edges = max((tuple(r["band_edges_hz"]) for r in qualifying), key=len, default=())
    members = [r for r in qualifying if tuple(r["band_edges_hz"]) == edges]
    record: dict[str, Any] = {
        "version": int(q["version"]),
        "candidates_n": len(candidates),
        "members": [str(r.get("stem") or "") for r in members],
        "members_n": len(members),
        "band_edges_hz": list(edges),
    }
    if len(members) < int(q["session"]["members_min"]):
        return {
            **record,
            "source": None,
            "fallback": NO_SPEECH_TASK,
            "band_db": None,
            "level_db": None,
            "spread_db": None,
            "quality": {"not_applicable": NO_FOREGROUND},
        }

    def stat(values: Sequence[float | None]) -> float | None:
        held = [float(v) for v in values if v is not None]
        return _round(float(reducer(held))) if held else None

    levels = [float(r["residual_db"]["whole"]) for r in members if r["residual_db"].get("whole") is not None]
    quartiles = np.percentile(levels, [25.0, 75.0]) if levels else None
    return {
        **record,
        "source": SPEECH_RESIDUAL_SOURCE,
        "fallback": None,
        "band_db": [round(float(v), 2) for v in reducer(np.asarray([r["residual_band_db"] for r in members]), axis=0)],
        "level_db": stat(levels),
        "spread_db": {
            "members_iqr": None if quartiles is None else _round(float(quartiles[1] - quartiles[0])),
            "p_low": stat([r["residual_db"].get("p_low") for r in members]),
            "p_high": stat([r["residual_db"].get("p_high") for r in members]),
        },
        "quality": {
            "foreground_db": stat([r.get("foreground_db") for r in members]),
            "foreground_minus_residual_db": {
                name: stat([r["foreground_minus_residual_db"].get(name) for r in members]) for name in FRAME_SETS
            },
            "plain_minus_residual_db": {
                name: stat([r["plain_minus_residual_db"].get(name) for r in members]) for name in FRAME_SETS
            },
            "not_applicable": None,
        },
    }


def background_floor_for(
    family: str | None,
    session_background: Mapping[str, Any] | None,
    band_edges_hz: Sequence[float],
    p: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """The floor BACKGROUND reads a recording against in place of the session floor, and why.

    Args:
        family: The recording's declared family.
        session_background: SESSION's ``session_background`` record, or None.
        band_edges_hz: The bands the recording's frames are read in.
        p: The parameters; ``data/session_background.yaml`` when None.

    Returns:
        ``(band_db, reference)``: the session background per band where ``decides`` is true, the family's
        kind is in ``decides_kinds``, and the session has a background in the same bands; else None.
        ``reference`` records ``decides``, ``kind``, ``used`` (:data:`SPEECH_RESIDUAL_SOURCE` or
        ``session_floor``) and ``fallback``, the reason the session floor stands where the switch is on and
        the kind is one it decides for.
    """
    q = p or session_background_parameters()
    kind = kind_of(family)
    decides = bool(q["decides"])
    applies = decides and kind in set(q["decides_kinds"])
    reference: dict[str, Any] = {"decides": decides, "kind": kind, "used": "session_floor", "fallback": None}
    if not applies:
        return None, reference
    band = None if session_background is None else session_background.get("band_db")
    if band is None:
        reference["fallback"] = (session_background or {}).get("fallback") or NO_SPEECH_TASK
        return None, reference
    if list(session_background.get("band_edges_hz") or []) != [float(e) for e in band_edges_hz]:  # type: ignore[union-attr]
        reference["fallback"] = "bands differ"
        return None, reference
    reference["used"] = SPEECH_RESIDUAL_SOURCE
    return np.asarray(band, dtype=np.float64), reference
