"""Fusion: time-aligned word-slot voting → final outputs (FR-021/022, research.md D9).

Post-T050 the fusion *math* lives in the reusable task package
``senselab.audio.tasks.speech_to_text_ensemble`` (``fuse_word_streams`` /
``load_calibrator`` / ``iter_word_leaves`` are re-exported here for the loop's
callers); this module keeps the workflow-specific parts — artifact word-stream
collection, policy → weights/params translation, speaker & speech_presence lookups
from the belief state, and the ``final/`` artifact writers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from senselab.audio.tasks.speech_to_text_ensemble import (  # noqa: F401 — re-exported for loop callers
    fuse_word_streams,
    iter_word_leaves,
    load_calibrator,
)
from senselab.audio.workflows.audio_analysis.adaptive.belief import bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.policy import family_weights
from senselab.audio.workflows.audio_analysis.layout import belief_dir, final_dir

# ── word-stream extraction ───────────────────────────────────────────────


def collect_word_streams(
    asr_by_model: dict[str, dict[str, Any]],
    align_by_model: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Per-model timestamped word lists; alignment result wins for text-only models.

    This function removes nothing. It used to drop every word of a model overlapping a span P3 had
    adjudicated, which left no record anywhere downstream — and made a word's survival depend on
    whether the intervention had been admitted within budget. Doubt about a word is now carried as
    a measured weight on the word itself (``adaptive.corroboration.apply_corroboration``).
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
        streams[model] = words
    return streams


# ── slot voting ──────────────────────────────────────────────────────────


def fuse_words(
    word_streams: dict[str, list[dict[str, Any]]],
    *,
    policy: dict[str, Any],
    speaker_at: Any = None,  # noqa: ANN401 — callable (t) -> str | None
    calibrator: Any = None,  # noqa: ANN401 — callable (c) -> c' | None
) -> list[dict[str, Any]]:
    """Policy-driven wrapper over the reusable transcript-ensemble task (T050).

    Translates the adaptive policy into the task API's explicit arguments —
    model-family weights (FR-008) and the fusion slot/margin parameters — and
    delegates the voting math to
    :func:`senselab.audio.tasks.speech_to_text_ensemble.fuse_word_streams`.

    Per-word corroboration is read off the words themselves (stamped by
    :func:`~senselab.audio.workflows.audio_analysis.adaptive.corroboration.apply_corroboration`),
    not passed here: fusion must not consult the intervention log, or budget admission decides
    what reaches the transcript.
    """
    fus = policy["fusion"]
    return fuse_word_streams(
        word_streams,
        weights=family_weights(sorted(word_streams), policy),
        slot_overlap=float(fus["slot_overlap"]),
        slot_mid_tol_s=float(fus["slot_mid_tol_s"]),
        winner_margin=float(fus["winner_margin"]),
        alternate_min_share=float(fus["alternate_min_share"]),
        min_corroboration=float(fus["corroboration"]["min_corroboration"]),
        speaker_at=speaker_at,
        calibrator=calibrator,
    )


def rollup_segments(words: list[dict[str, Any]], *, min_corroboration: float) -> tuple[list[dict[str, Any]], list[int]]:
    """Readable utterance rollup, plus the indices of words withheld from it.

    A word is included iff its ``corroboration`` is ``None`` (unmeasured — absent is not zero) or
    at least ``min_corroboration``. Withheld words stay in ``words[]`` with their measurement and
    their sources; only the concatenated ``text`` omits them.

    This is the one decision that remains, and it is deliberately at the rendering layer. Keeping
    an uncorroborated word in the readable transcript would let it *win*: the deliverable would
    assert it and the text consumers downstream (PII, sentiment, summary) would ingest it. Dropping
    it from ``words[]`` would be the erasure this work removed. The split keeps the evidence
    inspectable, carrying the number that excluded it, and makes the exclusion re-decidable by
    re-reading one file — no model re-run.

    Args:
        words: Fused words, time-ordered.
        min_corroboration: Rollup threshold.

    Returns:
        ``(segments, withheld_word_indices)``.
    """
    segments: list[dict[str, Any]] = []
    withheld: list[int] = []
    for index, w in enumerate(words):
        corroboration = w.get("corroboration")
        if corroboration is not None and float(corroboration) < float(min_corroboration):
            withheld.append(index)
            continue
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
    return segments, withheld


# ── lookups from belief state ────────────────────────────────────────────


def make_speaker_lookup(store: Any, state: Any, stream: str) -> Any:  # noqa: ANN401
    """(t) → majority unified cluster_id across active diarization votes at t.

    The buckets come from the axis (one set, folded across passes); the *labels* come from one
    pass's votes, because a cluster label is a statement a model made about a pass and the
    transcript being attributed was built from that pass.
    """
    rows = state.axis_rows("speaker")

    def lookup(t: float) -> str | None:
        for row in rows:
            if row["start"] <= t < row["end"]:
                bk = bucket_key(row["start"], row["end"])
                counts: dict[str, int] = {}
                for source, payload in store.active_votes(stream, "speaker", bk).items():
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


def make_p_voice_lookup(state: Any) -> Any:  # noqa: ANN401
    """(t) → speech_presence p_voice at t from the speech_presence belief rows."""
    rows = state.axis_rows("speech_presence")

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


def attenuation_columns(row: dict[str, Any]) -> dict[str, Any]:
    """The three columns that make a withdrawal readable from a parquet row.

    Shared by the per-round belief files and ``final/`` so the two cannot drift into describing the
    same withdrawal differently — the complaint being answered here is precisely that a fact held
    in memory reached no file at all.

    ``n_attenuated_sources`` exists because ``n_sources`` cannot serve. Attenuation deliberately
    keeps the source contributing, so the count is identical either side of it and an attenuated
    bucket read exactly like an unanimous one. The other two are JSON because the payload is a
    per-source mapping and a list of provenance records; flattening either into columns would fix
    an arity the run does not have.

    Args:
        row: One belief row, as produced by ``VoteStore.reaggregate_bucket``.

    Returns:
        ``n_attenuated_sources`` / ``attenuated_sources`` / ``attenuation``. An unattenuated bucket
        gets ``0``, ``"{}"`` and ``"[]"`` rather than nulls, so "nothing was withdrawn here" is
        stated rather than inferred from a gap.
    """
    weights = row.get("attenuated_sources") or {}
    detail = row.get("attenuation") or []
    return {
        "n_attenuated_sources": len(weights),
        "attenuated_sources": json.dumps(weights, sort_keys=True, default=str),
        "attenuation": json.dumps(detail, default=str),
    }


def build_final_outputs(
    *,
    out_dir: Path,
    words: list[dict[str, Any]],
    store: Any,  # noqa: ANN401
    state: Any,  # noqa: ANN401
    stream: str,
    policy: dict[str, Any],
    generated_from_round: int,
    corroboration_provenance: dict[str, Any],
    refined_identity: dict[str, Any] | None = None,
    calibrated: bool = False,
    timestamps_meta: dict[str, Any] | None = None,
    language: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the final deliverables and return ``(transcript, diarization)``.

    Both documents come back rather than only the transcript, so a caller that needs the
    diarization does not have to read ``final/diarization.json`` off disk — which would make a
    deliverable an input to the stage standing next to the one that wrote it.
    """
    final = final_dir(out_dir)
    final.mkdir(parents=True, exist_ok=True)

    base_speaker_lookup = make_speaker_lookup(store, state, stream)
    if refined_identity is not None:
        from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import cluster_at

        def speaker_lookup(t: float) -> str | None:
            return cluster_at(refined_identity, t) or base_speaker_lookup(t)
    else:
        speaker_lookup = base_speaker_lookup
    p_voice_lookup = make_p_voice_lookup(state)

    # transcript.json — segments rollup on speaker change or >0.5 s word gap, minus the words
    # whose measured corroboration falls below the rendering threshold. They stay in `words[]`.
    segment_min = float((policy["fusion"]["corroboration"])["segment_min_corroboration"])
    segments, withheld = rollup_segments(words, min_corroboration=segment_min)
    corroboration_doc = {
        **corroboration_provenance,
        "segment_min_corroboration": segment_min,
        "n_words_withheld_from_segments": len(withheld),
        # Indices into `words[]`, so the rollup is reproducible as a pure function of `words[]`
        # plus one number — the exclusion can be re-decided without re-running a model.
        "withheld_word_indices": withheld,
    }
    transcript = {
        "calibrated": calibrated,
        "policy_hash": policy.get("policy_hash"),
        "generated_from_round": generated_from_round,
        "stream": stream,
        "language": language,
        "timestamps": timestamps_meta or {"timestamps_source": "member_vote"},
        "corroboration": corroboration_doc,
        "words": words,
        "segments": segments,
    }
    (final / "transcript.json").write_text(json.dumps(transcript, indent=2))

    # diarization.json — refined I2 segments when available (real boundary
    # confidences from change-point prominence); else merge speaker buckets
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
        for row in state.axis_rows("speaker"):
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
    # contracts/final-outputs.md: member_labels (refined cluster ↔ diar-model raw
    # labels via vote co-occurrence) + per-segment overlap flag (I4 posterior).
    speaker_rows = state.axis_rows("speaker")
    member_labels: dict[str, dict[str, set]] = {}
    for seg in diar_segments:
        labels_for_cluster = member_labels.setdefault(str(seg["cluster_id"]), {})
        for row in speaker_rows:
            mid = (row["start"] + row["end"]) / 2.0
            if not (seg["start"] <= mid < seg["end"]):
                continue
            bk = bucket_key(row["start"], row["end"])
            for source, payload in store.active_votes(stream, "speaker", bk).items():
                if source.startswith(("__", "embedding_")) or "::" in source:
                    continue
                raw_label = payload.get("speaker_label")
                if raw_label and raw_label not in ("SIL", "<silent>"):
                    labels_for_cluster.setdefault(source, set()).add(str(raw_label))
        seg["overlap"] = any(
            (row.get("overlap_posterior") or 0.0) >= 0.5
            for row in speaker_rows
            if seg["start"] <= (row["start"] + row["end"]) / 2.0 < seg["end"]
        )
    for cluster in clusters.values():
        cluster["member_labels"] = {
            model: sorted(labels) for model, labels in (member_labels.get(str(cluster["cluster_id"])) or {}).items()
        }
    diarization = {
        "clusters": sorted(clusters.values(), key=lambda c: c["cluster_id"]),
        "segments": diar_segments,
        "refined": refined_identity is not None,
    }
    (final / "diarization.json").write_text(json.dumps(diarization, indent=2))

    # RTTM sidecar for interop (contracts/final-outputs.md).
    audio_id = policy.get("rttm_audio_id") or "audio"
    rttm_lines = [
        f"SPEAKER {audio_id} 1 {seg['start']:.3f} {max(0.0, seg['end'] - seg['start']):.3f} "
        f"<NA> <NA> {seg['cluster_id']} <NA> <NA>"
        for seg in diar_segments
    ]
    (final / "diarization.rttm").write_text("\n".join(rttm_lines) + ("\n" if rttm_lines else ""))
    return transcript, diarization


def extract_final_estimates(out_dir: Path, round_index: int) -> tuple[Path, ...]:
    """Copy round ``round_index``'s estimates to ``final/estimates/`` — verbatim (D-17).

    This is what ``final/`` is: an extraction. The deliverable presence track used to be *rebuilt*
    here from the belief state, into ``L2/speech_presence.parquet``, with columns
    (``speech_presence_confidence``, ``overlap_posterior``) that no round carried — so the number
    a consumer acted on was not the number any round believed, and there was nowhere to look to
    see when it had been decided. The columns now live on the estimate row and this function only
    moves bytes.

    One artifact per active axis, from the axis declaration rather than a list here. The four
    per-axis declarations this replaces named ``speech_presence``, ``asr`` and
    ``background_mask`` — and *not* ``speaker``, so the deliverable set was itself a list of
    three axes with the fourth missing, which is the failure ``axes.AXES`` exists to make
    impossible.

    Args:
        out_dir: Run directory.
        round_index: The round to extract. The last one, always: an earlier round would make the
            deliverable a re-reading of the trajectory rather than the answer.

    Returns:
        The paths written, in axis order. An axis the round wrote nothing for is skipped and named
        by the conformance guard rather than silently absent.
    """
    import pyarrow.parquet as pq

    from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES
    from senselab.audio.workflows.audio_analysis.layout import estimates_dir

    source = estimates_dir(out_dir, round_index)
    dest = final_dir(out_dir) / "estimates"
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for axis in AXIS_NAMES:
        origin = source / f"{axis}.parquet"
        if not origin.exists():
            continue
        pq.write_table(pq.read_table(origin), dest / f"{axis}.parquet")
        written.append(dest / f"{axis}.parquet")
    return tuple(written)


def write_speaker_outputs(
    out_dir: Path,
    *,
    posterior: Any,  # noqa: ANN401 — SpeakerCountPosterior
    hypotheses: Sequence[Any],
    correspondence: Sequence[Any] = (),
    tracks: Sequence[Any] = (),
    profile_version: str = "",
    influence_profile: str = "",
    generated_from_round: int = 0,
) -> tuple[Path, Path]:
    """Write ``final/speakers.json`` and ``final/per_speaker_presence.parquet`` (T102).

    Where the docstring always said, and where they now go. Both were written to the ``L2`` root
    instead — flattened per-run quantities with no round to belong to — so ``final/`` carried no
    per-speaker output at all while two declarations for it sat unproduced, and every consumer
    reached into the belief tree for a deliverable.

    Replaces the single per-bucket speaker scalar rather than sitting beside it: two names
    for one quantity is how schemas rot, and nothing on the way to alpha needs backwards
    compatibility.

    Args:
        out_dir: Run directory.
        posterior: The speaker-count posterior.
        hypotheses: One entry per hypothesized speaker.
        correspondence: Source-label to hypothesis mappings.
        tracks: Per-speaker speech_presence rows.
        profile_version: Detection-margin profile in force.
        influence_profile: Influence profile in force.
        generated_from_round: Round the outputs were fused from.

    Returns:
        ``(speakers_json_path, speech_presence_parquet_path)``.
    """
    import pandas as pd

    final = final_dir(out_dir)
    final.mkdir(parents=True, exist_ok=True)

    doc = {
        "profile_version": profile_version,
        "influence_profile": influence_profile,
        "generated_from_round": generated_from_round,
        "count_posterior": posterior.to_json(),
        "speakers": [h.to_json() for h in hypotheses],
        "label_correspondence": [c.to_json() for c in correspondence],
    }
    speakers_path = final / "speakers.json"
    speakers_path.write_text(json.dumps(doc, indent=2) + "\n")

    columns = [
        "speaker_id",
        "start",
        "end",
        "speech_presence_confidence",
        "speech_presence_uncertainty",
        "overlap_with",
        "contributing_sources",
        "round",
        "resolution_kind",
    ]
    rows = [t.to_row() for t in tracks]
    frame = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame({c: [] for c in columns})
    speech_presence_path = final / "per_speaker_presence.parquet"
    frame.to_parquet(speech_presence_path, index=False)
    return speakers_path, speech_presence_path
