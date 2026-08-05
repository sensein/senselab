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
from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.floors import MIN_EVIDENCE_WEIGHT

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s']")
_WHITESPACE_PATTERN = re.compile(r"\s+")

MIN_CORROBORATION = MIN_EVIDENCE_WEIGHT
"""Floor on a word's corroboration weight, so an uncorroborated word is attenuated, not erased.

Same reasoning as ``rounds.MIN_REGIONAL_TRUST``, ``support.SUPPORT_FLOOR`` and
``influence.effective_weight``'s ``min_gate``: non-corroboration is the weakest signal in the
system, it is measured asymmetrically (presence buckets are coarser than word boundaries), and a
quiet or overlapped speaker produces exactly the pattern it fires on. The dissenting word may be
the only record that something was said.

This used to restate ``0.05`` rather than import it, on the ground that a model-independent task
API must not learn about the workflow's presence axis. The ground was sound and the conclusion did
not follow: what is imported is
:mod:`~senselab.audio.workflows.audio_analysis.floors`, a module with no imports of its own, no
knowledge of any axis, and one constant in it. Restating the number bought no independence — it is
the *argument* that is shared, and a copy of a shared argument is just an unwatched place for it to
change.
"""


def _corroboration_of(member: Any, floor: float) -> float | None:  # noqa: ANN401 — word dicts are duck-typed
    """A member's measured corroboration, clamped and floored, or ``None`` if it was never measured.

    ``None`` is deliberately distinct from ``0.0``: a factor nobody gathered must not discount
    anything, or a run with no corroboration source would condemn every word at once.
    """
    raw = member.get("corroboration")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return max(floor, min(1.0, value))


def _normalize_word(text: str) -> str:
    cleaned = _PUNCTUATION_PATTERN.sub(" ", text.lower())
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def _temporal_agreement(
    members: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None, int]:
    """How well the independent timing sources agree on *where* this word is (D-27).

    Returns ``(onset_confidence, offset_confidence, n_timing_sources)`` — **per edge, not pooled**.
    A word can be agreed on at its start and disputed at its end, and that is a different finding
    from one whose whole span is uncertain: the first localises a boundary, the second does not
    localise the word. Pooling them with a ``max`` reported the worse edge and named it "temporal
    confidence", so a reader could not tell which edge was in doubt or whether both were.

    Each is ``None`` for a single timing source: one opinion cannot corroborate itself, and
    reporting 1.0 there is the manufactured certainty this split exists to remove.

    **Grouped by the timing model's identity, not by the model that produced the text and not by
    the *kind* of timing source.** Two transcripts timed by one aligner agree about onset by
    construction, and counting them as two agreeing opinions would turn a shared dependency into
    evidence.

    Keying on ``timestamp_source`` alone does not work, and the case it misses is the one that
    motivated this. ``TimestampSource`` is ``native | bundled_aligner | external_aligner`` — a
    *kind*. Qwen3-ASR's word times come from ``Qwen/Qwen3-ForcedAligner-0.6B`` shipped with it
    (``bundled_aligner``); Canary-Qwen carries no timings, so the workflow aligns it with **the
    same model** (``external_aligner``). Two labels, one aligner. Measured on
    ``english_conversation_higgs_audio_v2_20260805-034348``: their onsets are bit-identical across
    all 62 words (max onset difference 0.0000 s), while both differ from CrisperWhisper by the same
    0.032 s mean. So the key is ``timestamp_model`` when a member declares one, and the
    ``timestamp_source`` kind only as a fallback.

    A member declaring neither is treated as its own source, which is the conservative reading:
    unknown provenance is not shared provenance, and assuming otherwise would silently erase real
    corroboration.

    **Anchored to the word's own duration**, which is the only absolute reference a word carries:
    onsets disagreeing by a whole word-length mean the sources are not describing the same
    position.
    """
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for index, m in enumerate(members):
        # Identity first, kind second. A shared aligner reaches two members under two different
        # kinds (bundled vs external), so the kind cannot detect the sharing it exists to expose.
        timing_model = m.get("timestamp_model")
        key = str(timing_model or m.get("timestamp_source") or f"__member_{index}__")
        by_source.setdefault(key, []).append(m)
    if len(by_source) < 2:
        return None, None, len(by_source)

    # One opinion per timing source: members sharing an aligner are averaged, not counted twice.
    starts = [sum(float(m["start"]) for m in group) / len(group) for group in by_source.values()]
    ends = [sum(float(m["end"]) for m in group) / len(group) for group in by_source.values()]
    duration = max(1e-6, sum(float(m["end"]) - float(m["start"]) for m in members) / max(1, len(members)))

    def _edge(values: list[float]) -> float:
        return max(0.0, 1.0 - min(1.0, (max(values) - min(values)) / duration))

    return _edge(starts), _edge(ends), len(by_source)


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
        # Timing provenance travels with the word, because that is where the consumer needs it:
        # ``_temporal_agreement`` groups members by who produced their times, and a word that
        # arrives without it is counted as its own independent timing source.
        for field in ("timestamp_source", "timestamp_model"):
            value = node.get(field)
            if value:
                word[field] = str(value)
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
    min_corroboration: float = MIN_CORROBORATION,
    speaker_at: Any = None,  # noqa: ANN401 — callable (t) -> str | None
    calibrator: Any = None,  # noqa: ANN401 — callable (c) -> c' | None
) -> list[dict[str, Any]]:
    """Fuse per-system word streams into one consensus word list (ROVER-lite).

    Words across systems are grouped into time slots (time-overlap fraction ≥
    ``slot_overlap`` or midpoint distance ≤ ``slot_mid_tol_s``); each slot votes
    on the normalized text with ``weights[system] × word confidence ×
    corroboration``. Coverage penalizes slots that only a subset of
    systems witnessed (abstention is evidence). Optional ``speaker_at`` attributes
    speakers; ``calibrator`` maps raw confidences through a calibration profile.

    **A word is doubtful in two independent ways (D-27), and reports both.**

    - ``existence_confidence`` — was this said, and is this the text: ``share × coverage``, times
      the members' own confidence **when any member reports one**. It used to be
      ``share × member_conf × coverage`` with ``member_conf`` defaulting to 1.0, and every default
      recognizer reports no confidence at all — so the term was always 1.0 and 61 of 62 words on a
      real run came out at exactly 1.0. Absent is now absent: ``member_confidence`` is ``None`` and
      the term drops out rather than silently reading as certainty.
    - ``temporal_confidence`` — is it *here*: agreement between **timing sources** on the word's
      onset and offset, anchored to the word's own duration. ``None`` for a single timing source,
      which has nobody to agree with; a lone source reporting perfect temporal confidence is the
      same manufactured certainty in the other field.

    ``confidence`` remains, and is the joint — their product where both exist. The asr axis is a
    resampling of that joint onto the time grid, which is why the two parts have to survive
    separately as far as the axis: a word every model agrees on but times differently and a word
    the models disagree about are different findings, and the product alone cannot tell them apart.

    **Timing sources, not models.** Two transcripts timed by one aligner agree about onset by
    construction (Canary-Qwen is timed by Qwen's aligner), so members are grouped by
    ``timestamp_source`` before the spread is measured. Provenance is the only thing that can see
    this: an aligner is not a ``Source``, so source-closure intersection cannot (D-20).

    **Corroboration** is an optional per-word ``corroboration`` field in ``[0, 1]``: external
    evidence, measured by the caller, that something was said there. It enters in exactly two
    places — the vote weight and the coverage term — and nowhere else:

    - The **coverage** term is the load-bearing one. ``share`` is identically 1.0 for a one-member
      slot, so a factor entering only the vote weight is a provable no-op on precisely the case
      this mechanism exists for: a word one model produced where nothing corroborates it.
    - ``member_conf`` is deliberately **untouched**. It reports what the models said about
      themselves; folding corroboration into it would make one number mean two measurements, and
      would corrupt the field a calibration profile is fitted against.

    Absent or ``None`` corroboration means *unmeasured* and is treated as 1.0, never as 0.0 — a
    factor nobody gathered must not act as a discount.

    Alternates are tallied twice. The corroborated tally decides the winner, its ``share`` and its
    confidence; a second tally without corroboration decides *what gets recorded*. Attenuation may
    decide who wins; it may never decide who is written down, or a doubted reading disappears
    through the alternates gate as completely as deleting it would have.

    Args:
        word_streams: ``{system_id: [{"text", "start", "end", "confidence"?, "corroboration"?}]}``.
        weights: Optional per-system weight, e.g. model-family weights.
        slot_overlap: Minimum time-overlap fraction to join an existing slot.
        slot_mid_tol_s: Midpoint distance that also joins a slot.
        winner_margin: Winner share (uncorroborated tally) below which alternates are recorded.
        alternate_min_share: Minimum uncorroborated share for an alternate to be recorded.
        min_corroboration: Floor on the corroboration weight. See :data:`MIN_CORROBORATION`.
        speaker_at: Optional ``(t) -> speaker_id | None``.
        calibrator: Optional ``(confidence) -> calibrated``.

    Returns:
        Time-ordered ``[{text, start, end, confidence, coverage, corroboration,
        member_corroboration, sources, alternates, flags, speaker?}]``. ``corroboration`` is
        ``None`` when no member of the winning text was measured; consumers must not read that as
        zero.

    Raises:
        ValueError: If ``min_corroboration`` is not strictly positive. At zero both the vote weight
            and the coverage contribution vanish, so corroboration would delete a word rather than
            attenuate it — erasure reached through configuration instead of through code.
    """
    if float(min_corroboration) <= 0.0:
        raise ValueError(
            f"min_corroboration must be > 0; got {min_corroboration}. A zero floor removes the word "
            "from both the vote and the coverage term, which is deletion, not attenuation."
        )
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
        total_w_uncorroborated = 0.0
        coverage_mass = 0.0
        member_corroboration: dict[str, float | None] = {}
        for m in slot["members"]:
            key = _normalize_word(m["text"]) or m["text"].lower()
            corr = _corroboration_of(m, float(min_corroboration))
            member_corroboration[m["model"]] = corr
            system_weight = weights.get(m["model"], 1.0)
            base = system_weight * float(m.get("confidence", 1.0))
            wt = base * (1.0 if corr is None else corr)
            total_w += wt
            total_w_uncorroborated += base
            # Coverage over the *members* rather than the model names, so the corroboration of the
            # word each system actually produced reaches the abstention term. The denominator stays
            # the ensemble at full trust.
            coverage_mass += system_weight * (1.0 if corr is None else corr)
            t = tally.setdefault(
                key,
                {
                    "weight": 0.0,
                    "weight_uncorroborated": 0.0,
                    "models": [],
                    "display": m["text"],
                    "confs": [],
                    "corr_num": 0.0,
                    "corr_den": 0.0,
                },
            )
            t["weight"] += wt
            t["weight_uncorroborated"] += base
            t["models"].append(m["model"])
            if corr is not None:
                t["corr_num"] += base * corr
                t["corr_den"] += base
            if "confidence" in m:
                t["confs"].append(m["confidence"])
        if not tally or total_w <= 0 or total_w_uncorroborated <= 0:
            continue
        ranked = sorted(tally.items(), key=lambda kv: (-kv[1]["weight"], kv[0]))
        _win_key, win = ranked[0]
        share = win["weight"] / total_w
        share_uncorroborated = win["weight_uncorroborated"] / total_w_uncorroborated
        # Absent is absent (D-27). ``None`` when no member reported a confidence, so the term drops
        # out of the product instead of entering it as 1.0 — which is a claim of certainty made on
        # behalf of models that said nothing about themselves.
        member_conf = sum(win["confs"]) / len(win["confs"]) if win["confs"] else None
        coverage = min(1.0, coverage_mass / ensemble_weight)
        existence_conf = share * coverage * (1.0 if member_conf is None else member_conf)
        onset_conf, offset_conf, timing_sources = _temporal_agreement(slot["members"])
        # The pooled figure is the worse edge, kept for consumers that want one number — but the
        # edges are reported separately because they are separate findings, and a renderer colouring
        # one box by the pooled value cannot show which end is in doubt.
        edges = [c for c in (onset_conf, offset_conf) if c is not None]
        temporal_conf = min(edges) if edges else None
        raw_conf = existence_conf if temporal_conf is None else existence_conf * temporal_conf
        win_corr = (win["corr_num"] / win["corr_den"]) if win["corr_den"] > 0 else None
        word = {
            "text": win["display"],
            "start": round(slot["start"], 4),
            "end": round(slot["end"], 4),
            "confidence": round(calibrator(raw_conf), 4) if calibrator else round(raw_conf, 4),
            "existence_confidence": round(existence_conf, 4),
            "temporal_confidence": None if temporal_conf is None else round(temporal_conf, 4),
            "onset_confidence": None if onset_conf is None else round(onset_conf, 4),
            "offset_confidence": None if offset_conf is None else round(offset_conf, 4),
            "member_confidence": None if member_conf is None else round(member_conf, 4),
            "timing_sources": timing_sources,
            "coverage": round(coverage, 4),
            "corroboration": None if win_corr is None else round(win_corr, 6),
            "member_corroboration": {
                model: (None if value is None else round(value, 6))
                for model, value in sorted(member_corroboration.items())
            },
            "sources": sorted(set(win["models"])),
            "alternates": [],
            "flags": (["single_source"] if len(slot["models"]) == 1 else []),
        }
        # Gated on the uncorroborated tally: whether a reading is *recorded* must not depend on how
        # far it was corroborated, or attenuation restores the erasure through the alternates gate.
        if share_uncorroborated < winner_margin:
            for _key, alt in ranked[1:]:
                alt_share_uncorroborated = alt["weight_uncorroborated"] / total_w_uncorroborated
                if alt_share_uncorroborated >= alternate_min_share:
                    alt_corr = (alt["corr_num"] / alt["corr_den"]) if alt["corr_den"] > 0 else None
                    word["alternates"].append(
                        {
                            "text": alt["display"],
                            "share": round(alt["weight"] / total_w, 4),
                            "share_uncorroborated": round(alt_share_uncorroborated, 4),
                            "models": sorted(set(alt["models"])),
                            "corroboration": None if alt_corr is None else round(alt_corr, 6),
                        }
                    )
        if speaker_at is not None:
            word["speaker"] = speaker_at(slot["mid"])
        fused.append(word)
    return fused
