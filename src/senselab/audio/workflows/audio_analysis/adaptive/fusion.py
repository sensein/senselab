"""Fusion: time-aligned word-slot voting → final outputs (FR-021/022, research.md D9).

Post-T050 the fusion *math* lives in the reusable task package
``senselab.audio.tasks.speech_to_text_ensemble`` (``fuse_word_streams`` /
``load_calibrator`` / ``iter_word_leaves`` are re-exported here for the loop's
callers); this module keeps the workflow-specific parts — artifact word-stream
collection, policy → weights/params translation, speaker & presence lookups
from the belief state, and the ``final/`` artifact writers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from senselab.audio.tasks.speech_to_text_ensemble import (  # noqa: F401 — re-exported for loop callers
    fuse_word_streams,
    iter_word_leaves,
    load_calibrator,
)
from senselab.audio.workflows.audio_analysis.adaptive.belief import bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.policy import family_weights

# ── word-stream extraction ───────────────────────────────────────────────


def collect_word_streams(
    asr_by_model: dict[str, dict[str, Any]],
    align_by_model: dict[str, dict[str, Any]],
    *,
    purged_spans: list[tuple[float, float, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Per-model timestamped word lists; alignment result wins for text-only models.

    ``purged_spans`` = (start, end, model) spans adjudicated as hallucinations —
    those words are excluded from fusion (C10 consequence).
    """
    streams: dict[str, list[dict[str, Any]]] = {}
    for model, block in asr_by_model.items():
        source = block
        align = align_by_model.get(model)
        words = iter_word_leaves((source.get("result") if isinstance(source, dict) else None) or [])
        has_ts = bool(words)
        if (not has_ts) and isinstance(align, dict) and align.get("status") == "ok":
            words = iter_word_leaves(align.get("result") or [])
        if not words:
            continue
        words.sort(key=lambda w: (w["start"], w["end"]))
        if purged_spans:
            words = [
                w
                for w in words
                if not any(m == model and w["start"] < e and w["end"] > s for (s, e, m) in purged_spans)
            ]
        streams[model] = words
    return streams


# ── slot voting ──────────────────────────────────────────────────────────


def fuse_words(
    word_streams: dict[str, list[dict[str, Any]]],
    *,
    policy: dict[str, Any],
    speaker_at: Any = None,  # noqa: ANN401 — callable (t) -> str | None
    p_voice_at: Any = None,  # noqa: ANN401 — callable (t) -> float | None
    calibrator: Any = None,  # noqa: ANN401 — callable (c) -> c' | None
) -> list[dict[str, Any]]:
    """Policy-driven wrapper over the reusable transcript-ensemble task (T050).

    Translates the adaptive policy into the task API's explicit arguments —
    model-family weights (FR-008) and the fusion slot/margin parameters — and
    delegates the voting math to
    :func:`senselab.audio.tasks.speech_to_text_ensemble.fuse_word_streams`.
    """
    fus = policy["fusion"]
    return fuse_word_streams(
        word_streams,
        weights=family_weights(sorted(word_streams), policy),
        slot_overlap=float(fus["slot_overlap"]),
        slot_mid_tol_s=float(fus["slot_mid_tol_s"]),
        winner_margin=float(fus["winner_margin"]),
        alternate_min_share=float(fus["alternate_min_share"]),
        speaker_at=speaker_at,
        p_voice_at=p_voice_at,
        calibrator=calibrator,
    )


# ── lookups from belief state ────────────────────────────────────────────


def make_speaker_lookup(store: Any, state: Any, stream: str) -> Any:  # noqa: ANN401
    """(t) → majority unified cluster_id across active diarization votes at t."""
    rows = state.axis_rows(stream, "identity")

    def lookup(t: float) -> str | None:
        for row in rows:
            if row["start"] <= t < row["end"]:
                bk = bucket_key(row["start"], row["end"])
                counts: dict[str, int] = {}
                for source, payload in store.active_votes(stream, "identity", bk).items():
                    if source.startswith("__") or "::" in source:
                        continue
                    cid = payload.get("cluster_id")
                    if cid and cid not in ("SIL", "<silent>"):
                        counts[str(cid)] = counts.get(str(cid), 0) + 1
                if not counts:
                    return None
                return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return None

    return lookup


def make_p_voice_lookup(state: Any, stream: str) -> Any:  # noqa: ANN401
    """(t) → presence p_voice at t from the presence belief rows."""
    rows = state.axis_rows(stream, "presence")

    def lookup(t: float) -> float | None:
        best = None
        for row in rows:
            if row["start"] <= t < row["end"]:
                pv = row.get("p_voice")
                if pv is not None:
                    best = pv if best is None else max(best, pv)
        return best

    return lookup


# ── final artifact writers ───────────────────────────────────────────────


def build_final_outputs(
    *,
    out_dir: Path,
    words: list[dict[str, Any]],
    store: Any,  # noqa: ANN401
    state: Any,  # noqa: ANN401
    stream: str,
    policy: dict[str, Any],
    generated_from_round: int,
    refined_identity: dict[str, Any] | None = None,
    calibrated: bool = False,
    timestamps_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write final/{transcript,diarization}.json + final/presence.parquet; return transcript doc."""
    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)

    base_speaker_lookup = make_speaker_lookup(store, state, stream)
    if refined_identity is not None:
        from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import cluster_at

        def speaker_lookup(t: float) -> str | None:
            return cluster_at(refined_identity, t) or base_speaker_lookup(t)
    else:
        speaker_lookup = base_speaker_lookup
    p_voice_lookup = make_p_voice_lookup(state, stream)

    # transcript.json — segments rollup on speaker change or >0.5 s word gap.
    segments: list[dict[str, Any]] = []
    for w in words:
        if segments and w.get("speaker") == segments[-1]["speaker"] and w["start"] - segments[-1]["end"] <= 0.5:
            seg = segments[-1]
            seg["end"] = w["end"]
            seg["text"] += " " + w["text"]
            seg["min_word_confidence"] = min(seg["min_word_confidence"], w["confidence"])
        else:
            segments.append(
                {
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": w.get("speaker"),
                    "text": w["text"],
                    "min_word_confidence": w["confidence"],
                }
            )
    transcript = {
        "calibrated": calibrated,
        "policy_hash": policy.get("policy_hash"),
        "generated_from_round": generated_from_round,
        "stream": stream,
        "timestamps": timestamps_meta or {"timestamps_source": "member_vote"},
        "words": words,
        "segments": segments,
    }
    (final / "transcript.json").write_text(json.dumps(transcript, indent=2))

    # diarization.json — refined I2 segments when available (real boundary
    # confidences from change-point prominence); else merge identity buckets
    # by majority cluster where voiced.
    diar_segments: list[dict[str, Any]] = []
    clusters: dict[str, dict[str, Any]] = {}
    if refined_identity is not None:
        for seg in refined_identity["segments"]:
            diar_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "cluster_id": seg["cluster_id"],
                    "boundary_confidence": seg.get("boundary_confidence", {"start": 0.5, "end": 0.5}),
                }
            )
            c = clusters.setdefault(
                seg["cluster_id"], {"cluster_id": seg["cluster_id"], "total_speech_s": 0.0, "n_segments": 0}
            )
            c["total_speech_s"] = round(c["total_speech_s"] + (seg["end"] - seg["start"]), 6)
            c["n_segments"] += 1
    else:
        for row in state.axis_rows(stream, "identity"):
            mid = (row["start"] + row["end"]) / 2.0
            cid = speaker_lookup(mid)
            pv = p_voice_lookup(mid)
            if cid is None or (pv is not None and pv < 0.5):
                continue
            if (
                diar_segments
                and diar_segments[-1]["cluster_id"] == cid
                and row["start"] <= diar_segments[-1]["end"] + 1e-6
            ):
                diar_segments[-1]["end"] = row["end"]
            else:
                diar_segments.append(
                    {
                        "start": row["start"],
                        "end": row["end"],
                        "cluster_id": cid,
                        "boundary_confidence": {"start": 0.5, "end": 0.5},
                    }
                )
            c = clusters.setdefault(cid, {"cluster_id": cid, "total_speech_s": 0.0, "n_segments": 0})
            c["total_speech_s"] = round(c["total_speech_s"] + (row["end"] - row["start"]), 6)
        for seg in diar_segments:
            clusters[seg["cluster_id"]]["n_segments"] += 1
    diarization = {
        "clusters": sorted(clusters.values(), key=lambda c: c["cluster_id"]),
        "segments": diar_segments,
        "refined": refined_identity is not None,
    }
    (final / "diarization.json").write_text(json.dumps(diarization, indent=2))

    # presence.parquet — final presence belief.
    import pandas as pd

    pres_rows = [
        {
            "start": r["start"],
            "end": r["end"],
            "p_voice": r.get("p_voice"),
            "aggregated_uncertainty": r.get("aggregated_uncertainty"),
            "epistemic": r.get("epistemic"),
            "aleatoric_floor": r.get("aleatoric_floor"),
            "status": r.get("status"),
            "irreducible_reason": r.get("irreducible_reason"),
            "round": r.get("round"),
        }
        for r in state.axis_rows(stream, "presence")
    ]
    pd.DataFrame(pres_rows).to_parquet(final / "presence.parquet", index=False)
    return transcript
