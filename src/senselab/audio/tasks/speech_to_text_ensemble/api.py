"""Transcript ensemble fusion: ROVER-style time-slot voting over word streams.

Model-independent and dependency-free (stdlib only). Callers provide the word
streams (``{system_id: [{"text", "start", "end", "confidence"?}, ...]}``) and an
optional per-system ``weights`` map — e.g. senselab's model-family weights so
that several checkpoints of one architecture don't masquerade as independent
witnesses, or accuracy-derived weights from a validation set.
"""

from __future__ import annotations

import math
import re
from typing import Any

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s']")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_word(text: str) -> str:
    cleaned = _PUNCTUATION_PATTERN.sub(" ", text.lower())
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def iter_word_leaves(node: Any) -> list[dict[str, Any]]:  # noqa: ANN401 — recursive JSON walk
    """Deepest (text, start, end) leaves of a serialized ``ScriptLine`` tree = words.

    Operates on dicts/lists (JSON artifacts or ``ScriptLine.model_dump()``); for
    ``ScriptLine`` *instances* use :meth:`ScriptLine.iter_leaves`.
    """
    out: list[dict[str, Any]] = []
    if isinstance(node, list):
        for item in node:
            out.extend(iter_word_leaves(item))
        return out
    if not isinstance(node, dict):
        return out
    chunks = node.get("chunks")
    if isinstance(chunks, list) and chunks:
        for c in chunks:
            out.extend(iter_word_leaves(c))
        return out
    text, start, end = node.get("text"), node.get("start"), node.get("end")
    if text and start is not None and end is not None:
        try:
            word = {"text": str(text).strip(), "start": float(start), "end": float(end)}
        except (TypeError, ValueError):
            return out
        score = node.get("score")
        if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0:
            word["confidence"] = float(score)
        if word["text"]:
            out.append(word)
    return out


def load_calibrator(profile: Any) -> Any:  # noqa: ANN401 — callable | None
    """Build a confidence calibrator from a profile dict.

    Supported shapes: ``{"type": "logistic", "a": float, "b": float}`` applies
    ``sigmoid(a · logit(c) + b)``; ``{"type": "piecewise", "x": [...], "y": [...]}``
    linearly interpolates. ``None``/missing → no calibration.
    """
    if not profile:
        return None
    kind = str(profile.get("type") or "")
    if kind == "logistic":
        a, b = float(profile["a"]), float(profile["b"])

        def _logistic(c: float) -> float:
            c = min(1.0 - 1e-6, max(1e-6, float(c)))
            z = a * math.log(c / (1.0 - c)) + b
            return round(1.0 / (1.0 + math.exp(-z)), 4)

        return _logistic
    if kind == "piecewise":
        xs = [float(v) for v in profile["x"]]
        ys = [float(v) for v in profile["y"]]
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("piecewise calibration profile needs matching x/y with >= 2 knots")

        def _piecewise(c: float) -> float:
            c = float(c)
            if c <= xs[0]:
                return round(ys[0], 4)
            for i in range(1, len(xs)):
                if c <= xs[i]:
                    t = (c - xs[i - 1]) / max(1e-9, xs[i] - xs[i - 1])
                    return round(ys[i - 1] + t * (ys[i] - ys[i - 1]), 4)
            return round(ys[-1], 4)

        return _piecewise
    raise ValueError(f"unknown calibration profile type: {kind!r}")


def fuse_word_streams(
    word_streams: dict[str, list[dict[str, Any]]],
    *,
    weights: dict[str, float] | None = None,
    slot_overlap: float = 0.3,
    slot_mid_tol_s: float = 0.15,
    winner_margin: float = 0.66,
    alternate_min_share: float = 0.15,
    speaker_at: Any = None,  # noqa: ANN401 — callable (t) -> str | None
    p_voice_at: Any = None,  # noqa: ANN401 — callable (t) -> float | None
    calibrator: Any = None,  # noqa: ANN401 — callable (c) -> c' | None
) -> list[dict[str, Any]]:
    """Fuse per-system word streams into one consensus word list (ROVER-lite).

    Words across systems are grouped into time slots (time-overlap fraction ≥
    ``slot_overlap`` or midpoint distance ≤ ``slot_mid_tol_s``); each slot votes
    on the normalized text with ``weights[system] × word confidence``. The
    winner's confidence is ``vote share × mean member confidence × coverage``,
    where coverage penalizes slots that only a subset of systems witnessed
    (abstention is evidence). Alternates are recorded when the winner's share is
    below ``winner_margin``. Optional ``speaker_at``/``p_voice_at`` lookups
    attribute speakers and flag low-presence words; ``calibrator`` maps raw
    confidences through a calibration profile.

    Returns time-ordered ``[{text, start, end, confidence, coverage, sources,
    alternates, flags, speaker?}]``.
    """
    weights = weights or {}
    entries = []
    for system, words in word_streams.items():
        for w in words:
            entries.append({**w, "model": system, "mid": (w["start"] + w["end"]) / 2.0})
    entries.sort(key=lambda e: (e["mid"], e["start"], e["model"]))

    slots: list[dict[str, Any]] = []
    for e in entries:
        placed = False
        for slot in slots:
            if e["model"] in slot["models"]:
                continue
            ov = min(slot["end"], e["end"]) - max(slot["start"], e["start"])
            frac = ov / max(1e-9, min(slot["end"] - slot["start"], e["end"] - e["start"]))
            if frac >= slot_overlap or abs(e["mid"] - slot["mid"]) <= slot_mid_tol_s:
                slot["members"].append(e)
                slot["models"].add(e["model"])
                n = len(slot["members"])
                slot["start"] = sum(m["start"] for m in slot["members"]) / n
                slot["end"] = sum(m["end"] for m in slot["members"]) / n
                slot["mid"] = (slot["start"] + slot["end"]) / 2.0
                placed = True
                break
        if not placed:
            slots.append(
                {"start": e["start"], "end": e["end"], "mid": e["mid"], "members": [e], "models": {e["model"]}}
            )
    slots.sort(key=lambda s: (s["start"], s["end"]))

    ensemble_weight = sum(weights.get(m, 1.0) for m in word_streams) or 1.0
    fused: list[dict[str, Any]] = []
    for slot in slots:
        tally: dict[str, dict[str, Any]] = {}
        total_w = 0.0
        for m in slot["members"]:
            key = _normalize_word(m["text"]) or m["text"].lower()
            wt = weights.get(m["model"], 1.0) * float(m.get("confidence", 1.0))
            total_w += wt
            t = tally.setdefault(key, {"weight": 0.0, "models": [], "display": m["text"], "confs": []})
            t["weight"] += wt
            t["models"].append(m["model"])
            if "confidence" in m:
                t["confs"].append(m["confidence"])
        if not tally or total_w <= 0:
            continue
        ranked = sorted(tally.items(), key=lambda kv: (-kv[1]["weight"], kv[0]))
        win_key, win = ranked[0]
        share = win["weight"] / total_w
        member_conf = sum(win["confs"]) / len(win["confs"]) if win["confs"] else 1.0
        coverage = min(1.0, sum(weights.get(m, 1.0) for m in slot["models"]) / ensemble_weight)
        raw_conf = share * member_conf * coverage
        word = {
            "text": win["display"],
            "start": round(slot["start"], 4),
            "end": round(slot["end"], 4),
            "confidence": round(calibrator(raw_conf), 4) if calibrator else round(raw_conf, 4),
            "coverage": round(coverage, 4),
            "sources": sorted(set(win["models"])),
            "alternates": [],
            "flags": (["single_source"] if len(slot["models"]) == 1 else []),
        }
        if share < winner_margin:
            for _key, alt in ranked[1:]:
                alt_share = alt["weight"] / total_w
                if alt_share >= alternate_min_share:
                    word["alternates"].append(
                        {"text": alt["display"], "share": round(alt_share, 4), "models": sorted(set(alt["models"]))}
                    )
        mid = slot["mid"]
        if speaker_at is not None:
            word["speaker"] = speaker_at(mid)
        if p_voice_at is not None:
            pv = p_voice_at(mid)
            if pv is not None and pv < 0.5:
                word["flags"].append("low_presence")
        fused.append(word)
    return fused
