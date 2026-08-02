"""Intervention catalog (prototype) — contracts/interventions.md.

Implemented for real on artifacts + the content-addressable cache:

- ``S1_stream_election`` — per-region raw/enhanced election from belief evidence.
- ``P3_uncorroborated_speech_attenuation`` / ``C9_missed_speech`` — cross-signal
  repair over existing evidence (C10 / C9), degraded to the indicators present in
  the ingested run (no whisper ``no_speech_prob`` / PPG unless the run had them).
- ``U2_reserve_escalation`` — adds reserve ASR models by **cache replay**: the
  reserve model's whole-file result is read from ``analyze_audio``'s
  content-addressable cache (same audio signature ⇒ same waveform), windowed to
  the region's buckets with the *same* harvest math the comparator uses
  (``harvest_asr_votes``), and merged as region-scoped votes.

- ``U1_region_reasr`` — live region re-ASR (HF whisper pipeline pool, enhanced stream
  regenerated on demand with recorded raw fallback).
- ``I1_boundary_refinement`` / ``I2_recluster`` — speaker repair from per-window
  embeddings (stored artifacts, or live fine-hop via ``backends.embed_windows``).
- ``I4_overlap_detection`` — segmentation-3.0 per-class posteriors (gated model;
  guards to ``next_actions`` without a token).

Still deferred: ``P2_fine_posteriors`` and v2's ``U4_overlap_separation``
(contracts/interventions.md).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.belief import Vote, bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.policy import family_weights, model_family
from senselab.audio.workflows.audio_analysis.adaptive.regions import region_buckets
from senselab.audio.workflows.audio_analysis.aggregate import _normalize_transcript_for_wer
from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES, OVERLAP_INFORMED_AXES
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    _levenshtein,
    asr_alignment_score_in_window,
    asr_text_in_window,
    resolve_asr_result,
    whisper_bucket_avg_logprob,
)
from senselab.audio.workflows.audio_analysis.layout import perturbation_dir
from senselab.audio.workflows.audio_analysis.perturbations import IDENTITY_NAME
from senselab.audio.workflows.audio_analysis.speech_presence_link import directed_presence_vote
from senselab.audio.workflows.audio_analysis.support import evidence_weight_from_corroboration

# ── shared artifact/cache access ─────────────────────────────────────────


def load_outcomes_dir(run_dir: Path, stream: str, task_dir: str) -> dict[str, dict[str, Any]]:
    """Load ``perturbation_dir(run_dir, stream)/<task_dir>/*.json`` keyed by provenance.model_id.

    Each payload records its ``_file_stem`` — alignment files are keyed by the
    *aligner* model id in provenance but written under the parent ASR model's
    safe filename, so cross-task joins go through the stem.

    The path comes from :func:`~senselab.audio.workflows.audio_analysis.layout.perturbation_dir` rather
    than being rebuilt here. It was rebuilt as ``run_dir / stream / task_dir`` until the pass
    outputs moved under ``L1/``, at which point this returned ``{}`` on every run — silently, so
    the ASR fusion path received nothing and emitted an empty transcript with no error anywhere.

    A missing directory now warns. An empty result is a legitimate answer only when the stage did
    not run; when the directory itself is absent the caller is asking about a layout that does not
    exist, and returning ``{}`` makes those two indistinguishable — which is precisely how the
    drift above survived to the point of producing a transcript with no words.
    """
    out: dict[str, dict[str, Any]] = {}
    d = perturbation_dir(run_dir, stream) / task_dir
    if not d.is_dir():
        print(
            f"warn: no {task_dir!r} outcomes directory for stream {stream!r} at {d} — "
            "nothing will be loaded for this task",
            file=sys.stderr,
        )
        return out
    for f in sorted(d.glob("*.json")):
        try:
            payload = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        payload["_file_stem"] = f.stem
        model_id = ((payload.get("provenance") or {}).get("model_id")) or f.stem
        out[str(model_id)] = payload
    return out


def load_alignments_matched(
    run_dir: Path, stream: str, asr_by_model: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Alignment payloads re-keyed by their parent **ASR** model id (stem join)."""
    by_stem: dict[str, dict[str, Any]] = {}
    d = perturbation_dir(run_dir, stream) / "alignment"
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            try:
                by_stem[f.stem] = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
    out: dict[str, dict[str, Any]] = {}
    for model, block in asr_by_model.items():
        stem = block.get("_file_stem")
        if stem and stem in by_stem:
            out[model] = by_stem[stem]
    return out


def build_cache_index(cache_dir: Path | None) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Index cache entries by (audio_signature, task, model_id); values sorted by filename."""
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    if cache_dir is None or not Path(cache_dir).is_dir():
        return index
    for f in sorted(Path(cache_dir).glob("*.json")):
        try:
            if f.stat().st_size > 30_000_000:  # features blobs — never needed here
                continue
            payload = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        prov = payload.get("provenance") or {}
        sig, task = str(prov.get("audio_signature") or ""), str(prov.get("task") or "")
        model = str(prov.get("model_id") or "")
        if sig and task:
            payload["_cache_file"] = f.name
            index.setdefault((sig, task, model), []).append(payload)
    return index


def _pick_ok(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for e in entries:  # already filename-sorted → deterministic
        if e.get("status") == "ok" and e.get("result") is not None:
            return e
    return None


def _has_g2p() -> bool:
    try:
        return importlib.util.find_spec("g2p_en") is not None
    except (ImportError, ValueError):
        return False


def _spec_missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError, ModuleNotFoundError):
        return True


# ── S1: stream election (C5, FR-015) ────────────────────────────────────


def _rows_in_span(rows: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [r for r in rows if r["end"] > start and r["start"] < end]


def _action_stream(region: dict[str, Any] | None) -> str:
    """Which pass a rule should operate on for this region.

    A region has no stream — it is a span of the recording, proposed from an axis that folds across
    passes. What a rule needs is an *action target*: which audio to hand a model. ``S1`` elects one
    from per-signal evidence and records it as ``action_stream``; before it has run, or when no
    rule has asked, the raw pass is the default because it is the one that was recorded rather than
    produced.
    """
    return str((region or {}).get("action_stream") or IDENTITY_NAME)


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and v == v]
    return sum(clean) / len(clean) if clean else None


def _election_scores(region: dict[str, Any], ctx: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Score each pass over the region, from that pass's own votes and measurements.

    Every term is per (signal, pass) or a per-pass measurement — never a reading of an axis. An
    axis is a fold across passes and therefore has no per-pass value to compare; asking it for one
    is what this rule used to do, and it is the same category error the belief store had. What a
    pass genuinely owns is what its signals said and what its audio measured, which is also the
    only evidence that bears on "which pass should a model be re-run against".
    """
    from senselab.audio.workflows.audio_analysis.aggregate import speech_presence_p_voice  # noqa: PLC0415
    from senselab.audio.workflows.audio_analysis.degradation import (  # noqa: PLC0415
        SNR_PREFERENCE,
        clip_degradation,
        reverb_degradation,
        snr_degradation,
    )
    from senselab.audio.workflows.audio_analysis.fuse import per_signal_uncertainty  # noqa: PLC0415

    store = ctx["store"]
    lo, hi = float(region["core_start"]), float(region["core_end"])
    scores: dict[str, dict[str, float]] = {}
    w = ctx["policy"]["election"]["weights"]
    for stream in ctx["passes"]:
        presence_buckets = [bk for bk in store.vote_buckets(stream, "speech_presence") if bk[1] > lo and bk[0] < hi]
        p_values = [
            p
            for bk in presence_buckets
            if (
                p := speech_presence_p_voice(
                    store.active_votes(stream, "speech_presence", bk),
                    weights=store.evidence_weights(stream, "speech_presence", bk),
                )
            )
            is not None
        ]
        degradations: list[float] = []
        for bk in presence_buckets:
            meta = store.row_meta.get((stream, "speech_presence", bk)) or {}
            snr_name = next((n for n in SNR_PREFERENCE if meta.get(n) is not None), None)
            terms = [
                snr_degradation(meta.get(snr_name)) if snr_name else None,
                reverb_degradation(meta.get("c50_brouhaha_db")),
                clip_degradation(meta.get("proportion_clipped")),
            ]
            measured = [float(t) for t in terms if t is not None]
            if measured:
                degradations.append(max(measured))
        doubts = [
            v
            for bk in store.vote_buckets(stream, "asr")
            if bk[1] > lo and bk[0] < hi
            for v in per_signal_uncertainty({"votes": store.active_votes(stream, "asr", bk)}).values()
        ]
        p_conf = _mean(list(p_values)) or 0.0
        quality = 1.0 - (_mean(list(degradations)) or 0.0)
        agree = 1.0 - (_mean(list(doubts)) or 0.0)
        total = w["speech_presence_conf"] * p_conf + w["quality"] * quality + w["asr_agreement"] * agree
        scores[stream] = {
            "speech_presence_conf": round(p_conf, 6),
            "quality": round(quality, 6),
            "asr_agreement": round(agree, 6),
            "total": round(total, 9),
        }
    return scores


def _s1_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or len(ctx["passes"]) < 2:
        return False, {}
    if region["region_id"] in ctx["elections"]:
        return False, {}
    return True, {"streams": list(ctx["passes"])}


def _s1_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    region = cand["region"]
    scores = _election_scores(region, ctx)
    # Ties go to the identity: it is the recording, and a transform has to *earn* the election.
    elected = max(scores, key=lambda s: (scores[s]["total"], s == IDENTITY_NAME))
    guard_fired = False
    if elected != IDENTITY_NAME:
        # Transform-artifact guard (degraded): reject a transformed stream when it claims speech
        # text where the recording itself has none *and* the recording's speech_presence is
        # confidently silent — enhancement can synthesize speech-like energy.
        raw_text = _region_text(ctx, IDENTITY_NAME, region)
        enh_text = _region_text(ctx, elected, region)
        raw_pres = _mean([scores[IDENTITY_NAME]["speech_presence_conf"]]) if IDENTITY_NAME in scores else None
        if enh_text and not raw_text and (raw_pres is not None and raw_pres < 0.2):
            guard_fired = True
            elected = IDENTITY_NAME
    election = {
        "region_id": region["region_id"],
        "scores": scores,
        "elected": elected,
        "guard_fired": guard_fired,
    }
    ctx["elections"][region["region_id"]] = election
    # An action target, not an index on the belief: it names the pass a later rule should re-run a
    # model against, and no axis changes because of it.
    region["action_stream"] = elected
    return {"election": election, "touched": {}}


def _region_text(ctx: dict[str, Any], stream: str, region: dict[str, Any]) -> str:
    texts = []
    lo, hi = float(region["core_start"]), float(region["core_end"])
    for bk in ctx["store"].vote_buckets(stream, "asr"):
        if bk[1] <= lo or bk[0] >= hi:
            continue
        for source, payload in ctx["store"].active_votes(stream, "asr", bk).items():
            if not source.startswith("__") and payload.get("text"):
                texts.append(str(payload["text"]))
    return " ".join(texts).strip()


# ── P3 / C9: adjudication over existing evidence (C10 / C9) ─────────────


def _presence_pool(ctx: dict[str, Any], stream: str) -> list[str]:
    """Independent presence voters for ``stream``, derived once per run and cached on ``ctx``."""
    from senselab.audio.workflows.audio_analysis.adaptive.corroboration import (  # noqa: PLC0415
        independent_presence_pool,
    )

    cache = ctx.setdefault("_presence_pools", {})
    if stream not in cache:
        pool, rejected = independent_presence_pool(ctx["store"], stream)
        cache[stream] = pool
        ctx.setdefault("_presence_pools_rejected", {})[stream] = rejected
    return list(cache[stream])


def _adjudication_candidates(ctx: dict[str, Any], stream: str) -> list[dict[str, Any]]:
    """ASR speech claims in buckets where independent evidence measured little support.

    Keyed on measured corroboration rather than on the row's ``p_voice``. ``p_voice`` is a
    weighted mean over *all* presence voters including the indicted ASR, so the source partly
    protected itself and acting on it changed the very number that indicted it — a same-round
    feedback path. Reading the independent pool instead makes the trigger quantity and the
    attenuation degree one measurement.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.belief import (  # noqa: PLC0415
        UNCORROBORATED_SPEECH_CLAIM,
    )
    from senselab.audio.workflows.audio_analysis.support import bucket_corroboration  # noqa: PLC0415

    adj = ctx["policy"]["adjudication"]
    pool = _presence_pool(ctx, stream)
    out = []
    for row in ctx["state"].axis_rows("speech_presence"):
        bk = bucket_key(row["start"], row["end"])
        votes = ctx["store"].active_votes(stream, "speech_presence", bk)
        corroboration = bucket_corroboration(votes, evidence_signals=pool)
        # Nothing measured here ⇒ nothing may be discounted. Absent is not zero.
        if corroboration is None or corroboration >= adj["corroboration_low"]:
            continue
        for source, payload in votes.items():
            if source.startswith("__") or source in pool:
                continue
            if source not in ctx["asr_model_ids"].get(stream, set()):
                continue
            if not payload.get("speaks"):
                continue
            if ctx["store"].has_evidence_weight_factor(
                stream, "speech_presence", bk, source, reason=UNCORROBORATED_SPEECH_CLAIM
            ):
                continue
            meta = row.get("meta") or {}
            nc = payload.get("native_confidence")
            indicators = {
                "low_native_confidence": nc is not None and float(nc) < adj["low_native_confidence"],
                "low_src_speech": (meta.get("src_speech") is not None)
                and float(meta.get("src_speech") or 1.0) < adj["low_src_speech"],
                "very_low_corroboration": corroboration < adj["corroboration_very_low"],
            }
            if sum(indicators.values()) >= adj["min_indicators"]:
                out.append(
                    {
                        "bucket": bk,
                        "source": source,
                        "indicators": indicators,
                        "corroboration": corroboration,
                        "evidence_sources": list(pool),
                    }
                )
    return out


def _p3_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    cands = {s: _adjudication_candidates(ctx, s) for s in ctx["passes"]}
    n = sum(len(v) for v in cands.values())
    return n > 0, {"n_candidates": n}


def _p3_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Withdraw weight from uncorroborated speech claims; the votes stay in the fold.

    The claim is attenuated on the presence bucket and on every asr bucket overlapping it, all by
    the *same* measurement taken on the presence bucket — which is what ``measured_on`` records.
    Nothing is removed: a quiet or overlapped speaker produces this exact signature, and deleting
    the only source that heard them leaves the bucket reporting confident silence with no trace to
    appeal to.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.belief import (  # noqa: PLC0415
        UNCORROBORATED_SPEECH_CLAIM,
    )

    adj = ctx["policy"]["adjudication"]
    floor = float(adj["min_evidence_weight"])
    touched: dict[str, set[tuple[float, float]]] = {}
    attenuated = []
    for stream in ctx["passes"]:
        for c in _adjudication_candidates(ctx, stream):
            measured_on = ("speech_presence", c["bucket"])
            records = ctx["store"].attenuate_source_in_bucket(
                stream,
                c["bucket"],
                c["source"],
                corroboration=c["corroboration"],
                evidence_sources=c["evidence_sources"],
                reason=UNCORROBORATED_SPEECH_CLAIM,
                round_idx=ctx["round_idx"],
                measured_on=measured_on,
                floor=floor,
            )
            for urow in _rows_in_span(ctx["state"].axis_rows("asr"), c["bucket"][0], c["bucket"][1]):
                ubk = bucket_key(urow["start"], urow["end"])
                records += ctx["store"].attenuate_source_in_bucket(
                    stream,
                    ubk,
                    c["source"],
                    corroboration=c["corroboration"],
                    evidence_sources=c["evidence_sources"],
                    reason=UNCORROBORATED_SPEECH_CLAIM,
                    round_idx=ctx["round_idx"],
                    measured_on=measured_on,
                    floor=floor,
                )
                touched.setdefault("asr", set()).add(ubk)
            touched.setdefault("speech_presence", set()).add(c["bucket"])
            attenuated.append(
                {
                    **c,
                    "stream": stream,
                    "votes_attenuated": len(records),
                    "evidence_weight": max(
                        (r["evidence_weight"] for r in records),
                        default=evidence_weight_from_corroboration(c["corroboration"], floor=floor),
                    ),
                    "axes": sorted({r["axis"] for r in records}),
                }
            )
    return {"attenuated": attenuated, "touched": touched}


def _c9_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    n = sum(len(_missed_speech_candidates(ctx, s)) for s in ctx["passes"])
    return n > 0, {"n_candidates": n}


def _missed_speech_candidates(ctx: dict[str, Any], stream: str) -> list[dict[str, Any]]:
    adj = ctx["policy"]["adjudication"]
    out = []
    for row in ctx["state"].axis_rows("speech_presence"):
        p_voice = row.get("p_voice")
        # C9's own lower bound. It used to borrow P3's threshold, so retuning one rule silently
        # moved the other; the numbers are unchanged, the coupling is not.
        if p_voice is None or not (adj["p_voice_missed_low"] <= p_voice < adj["p_voice_missed"]):
            continue
        bk = bucket_key(row["start"], row["end"])
        if "adjudicator/missed_speech" in ctx["store"].active_votes(stream, "speech_presence", bk):
            continue
        families = set()
        for urow in _rows_in_span(ctx["state"].axis_rows("asr"), bk[0], bk[1]):
            ubk = bucket_key(urow["start"], urow["end"])
            for source, payload in ctx["store"].active_votes(stream, "asr", ubk).items():
                if not source.startswith("__") and payload.get("text"):
                    families.add(model_family(source, ctx["policy"]))
        if len(families) >= 2:
            out.append({"bucket": bk, "families": sorted(families), "p_voice": p_voice})
    return out


def _c9_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    touched: dict[str, set[tuple[float, float]]] = {}
    added = []
    weight = float(ctx["policy"]["adjudication"]["missed_speech_weight"])
    for stream in ctx["passes"]:
        for c in _missed_speech_candidates(ctx, stream):
            ctx["store"].add_vote(
                Vote(
                    axis="speech_presence",
                    bucket=c["bucket"],
                    source="adjudicator/missed_speech",
                    stream=stream,
                    scope="file",
                    round=ctx["round_idx"],
                    payload={"speaks": True, "native_confidence": None, "weight": weight},
                    provenance={"families_agreeing": c["families"], "rule": "C9_missed_speech"},
                )
            )
            touched.setdefault("speech_presence", set()).add(c["bucket"])
            added.append({**c, "stream": stream})
    return {"added": added, "touched": touched}


# ── U2: reserve escalation via cache replay ──────────────────────────────


def _reserves_in_cache(ctx: dict[str, Any], stream: str) -> list[str]:
    sig = ctx["pass_sigs"].get(stream, "")
    found = []
    for model in ctx["policy"].get("reserve_asr_models") or []:
        if _pick_ok(ctx["cache_index"].get((sig, "asr", model), [])):
            found.append(model)
    return found


def _u2_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or region["axis"] != "asr":
        return False, {}
    stream = _action_stream(region)
    reserves = _reserves_in_cache(ctx, stream)
    if not reserves:
        return False, {"reason": "no_cached_reserves"}
    rows = _rows_in_span(ctx["state"].axis_rows("asr"), region["core_start"], region["core_end"])
    epi = _mean([r.get("epistemic_uncertainty") for r in rows])
    if epi is None or epi < ctx["policy"]["thresholds"]["theta_low"]:
        return False, {"epistemic": epi}
    return True, {"reserves": reserves, "epistemic": round(float(epi), 6), "stream": stream}


def _u2_gain(region: dict[str, Any], ctx: dict[str, Any], trigger: dict[str, Any]) -> float:
    return float(region["uncertainty_mass"]) * float(trigger.get("epistemic") or 0.0) * 10.0


def _u2_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    region, trigger = cand["region"], cand["trigger"]
    stream = trigger["stream"]
    sig = ctx["pass_sigs"][stream]
    run_dir: Path = ctx["run_dir"]

    asr_by_model = dict(load_outcomes_dir(run_dir, stream, "asr"))
    align_by_model = load_alignments_matched(run_dir, stream, asr_by_model)
    used_reserves: list[dict[str, Any]] = []
    for model in trigger["reserves"]:
        entry = _pick_ok(ctx["cache_index"].get((sig, "asr", model), []))
        if entry is None:
            continue
        asr_by_model[model] = entry
        # Text-only reserves need their cached alignment for word timestamps.
        parent_key = entry.get("cache_key") or (entry.get("provenance") or {}).get("cache_key")
        for (esig, task, amodel), align_entries in ctx["cache_index"].items():
            if esig != sig or task != "alignment":
                continue
            match = next(
                (
                    a
                    for a in align_entries
                    if a.get("status") == "ok"
                    and (a.get("provenance") or {}).get("parent_asr_cache_key")
                    and (a.get("provenance") or {}).get("parent_asr_cache_key") == parent_key
                ),
                None,
            )
            if match is not None:
                align_by_model[model] = match
                break
        used_reserves.append({"model": model, "cache_file": entry.get("_cache_file")})

    pass_summary_ext = {"duration_s": ctx["duration_s"], "asr": {"by_model": asr_by_model}}
    grid = BucketGrid(win_length=ctx["asr_grid"][0], hop_length=ctx["asr_grid"][1])

    pair_kind = "phoneme"
    if _has_g2p():
        from senselab.audio.workflows.audio_analysis.asr import harvest_asr_votes

        harvested = harvest_asr_votes(
            pass_summary=pass_summary_ext, grid=grid, ppg_block={}, alignment_by_model=align_by_model
        )
    else:
        pair_kind = "word"
        harvested = _harvest_word_level(pass_summary_ext, grid, align_by_model, ctx["duration_s"])

    rows = ctx["state"].axis_rows("asr")
    covered = region_buckets(region, rows)
    touched: dict[str, set[tuple[float, float]]] = {}
    n_votes = 0
    for entry in harvested:
        bk = bucket_key(entry["start"], entry["end"])
        if bk not in covered:
            continue  # merge-back midpoint rule (D2)
        for source, payload in entry["votes"].items():
            if isinstance(payload, dict) and source == "__pairwise_phoneme_distances__":
                payload = {**payload, "pair_distance_kind": pair_kind}
            ctx["store"].add_vote(
                Vote(
                    axis="asr",
                    bucket=bk,
                    source=source,
                    stream=stream,
                    scope=f"region:{region['region_id']}",
                    round=ctx["round_idx"],
                    payload=payload if isinstance(payload, dict) else {"value": payload},
                    provenance={"rule": "U2_reserve_escalation", "reserves": used_reserves},
                )
            )
            n_votes += 1
        touched.setdefault("asr", set()).add(bk)
    fam = family_weights(sorted([m for m in asr_by_model]), ctx["policy"])
    return {
        "reserves_used": used_reserves,
        "pair_distance_kind": pair_kind,
        "votes_added": n_votes,
        "family_weights": fam,
        "touched": touched,
    }


def _harvest_word_level(
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    alignment_by_model: dict[str, Any],
    duration_s: float,
) -> list[dict[str, Any]]:
    """g2p-free fallback: pairwise WORD-Levenshtein rate (the original FR-002 WER form).

    Same vote schema as ``harvest_asr_votes`` so ``aggregate_asr``
    consumes it unchanged; ``pair_distance_kind: "word"`` is recorded on the
    pair block by the caller.
    """
    from itertools import combinations

    asr_blocks = (pass_summary.get("asr") or {}).get("by_model") or {}
    resolved = {
        m: resolve_asr_result(b, alignment_by_model.get(m))
        for m, b in asr_blocks.items()
        if isinstance(b, dict) and b.get("status") == "ok"
    }
    out = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        votes: dict[str, Any] = {}
        seqs: dict[str, list[str]] = {}
        confs: dict[str, float] = {}
        for m, res in resolved.items():
            text = asr_text_in_window(res, start, end, fully_contained=True)
            alp = whisper_bucket_avg_logprob(res, start, end)
            ctc = asr_alignment_score_in_window(res, start, end)
            votes[m] = {"text": text, "phoneme_sequence": [], "avg_logprob": alp, "alignment_ctc_score": ctc}
            tokens = _normalize_transcript_for_wer(text or "").split()
            if tokens:
                seqs[m] = tokens
            if alp is not None:
                try:
                    confs[m] = max(0.0, min(1.0, math.exp(float(alp))))
                except (ValueError, OverflowError):
                    pass
        pairs = {}
        for a, b in combinations(sorted(seqs), 2):
            denom = max(len(seqs[a]), len(seqs[b]))
            if denom:
                pairs[f"{a}|{b}"] = min(1.0, _levenshtein(seqs[a], seqs[b]) / denom)
        votes["__pairwise_phoneme_distances__"] = {
            "pairs": pairs,
            "n_sources": len(seqs),
            "sources": sorted(seqs),
            "per_source_confidence": confs,
        }
        out.append({"start": start, "end": end, "votes": votes})
    return out


# ── U1: live region re-ASR ───────────────────────────────────────────────


def _u1_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or region["axis"] != "asr":
        return False, {}
    models = [m for m in (ctx["policy"].get("u1_asr_models") or []) if m not in ctx.get("live_asr_done", set())]
    if not models:
        return False, {"reason": "no_u1_models"}
    stream = _action_stream(region)
    rows = _rows_in_span(ctx["state"].axis_rows("asr"), region["core_start"], region["core_end"])
    epi = _mean([r.get("epistemic_uncertainty") for r in rows])
    if epi is None or epi < ctx["policy"]["thresholds"]["theta_low"]:
        return False, {"epistemic": epi}
    return True, {
        "models": models,
        "stream": stream,
        "epistemic": round(float(epi), 6),
        "crop": [region["crop_start"], region["crop_end"]],
    }


def _u1_guard(region: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    if _spec_missing("transformers") and not _senselab_asr_available():
        return "asr_backend_unavailable"
    if not ctx.get("input_audio"):
        return "input_audio_missing"
    return None


def _senselab_asr_available() -> bool:
    from senselab.audio.workflows.audio_analysis.adaptive.backends import senselab_transcribe_available

    return senselab_transcribe_available()


def _u1_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    from senselab.audio.workflows.audio_analysis.adaptive.audio_io import crop as crop_wav
    from senselab.audio.workflows.audio_analysis.adaptive.audio_io import get_stream_wav
    from senselab.audio.workflows.audio_analysis.adaptive.backends import transcribe_crop

    region, trigger = cand["region"], cand["trigger"]
    stream = trigger["stream"]
    wav, reason = get_stream_wav(ctx, stream)
    stream_fallback = None
    if wav is None and stream != IDENTITY_NAME:
        stream_fallback = reason
        stream = IDENTITY_NAME
        wav, reason = get_stream_wav(ctx, stream)
    if wav is None:
        raise RuntimeError(f"audio_unavailable: {reason}")

    crop_start, crop_end = float(region["crop_start"]), float(region["crop_end"])
    segment = crop_wav(wav, crop_start, crop_end)
    language = ctx["policy"].get("language")
    ran: list[dict[str, Any]] = []
    ext_blocks: dict[str, dict[str, Any]] = {}
    u1_backend = str(ctx["policy"].get("u1_backend", "auto"))
    for model in trigger["models"]:
        meta: dict[str, Any] = {}
        words, err = transcribe_crop(
            segment, model_id=model, offset_s=crop_start, language=language, meta=meta, backend=u1_backend
        )
        if words is None:
            ran.append({"model": model, "status": "failed", "error": err})
            continue
        ran.append({"model": model, "status": "ok", "n_words": len(words), **meta})
        ctx.setdefault("live_asr_done", set()).add(model)
        ctx.setdefault("live_asr_words", {}).setdefault(stream, {})[model] = words
        ext_blocks[model] = {
            "status": "ok",
            "result": [
                {
                    "text": " ".join(w["text"] for w in words),
                    "start": crop_start,
                    "end": crop_end,
                    "chunks": [{**w, "chunks": None} for w in words],
                }
            ],
        }
    if not ext_blocks:
        raise RuntimeError(f"all_u1_models_failed: {ran}")

    # Merge with the run's own ASR and re-harvest the covered buckets — same
    # path as U2 so votes/pairs stay schema-identical.
    asr_by_model = dict(load_outcomes_dir(ctx["run_dir"], stream, "asr"))
    align_by_model = load_alignments_matched(ctx["run_dir"], stream, asr_by_model)
    asr_by_model.update(ext_blocks)
    pass_summary_ext = {"duration_s": ctx["duration_s"], "asr": {"by_model": asr_by_model}}
    grid = BucketGrid(win_length=ctx["asr_grid"][0], hop_length=ctx["asr_grid"][1])
    pair_kind = "phoneme"
    if _has_g2p():
        from senselab.audio.workflows.audio_analysis.asr import harvest_asr_votes

        harvested = harvest_asr_votes(
            pass_summary=pass_summary_ext, grid=grid, ppg_block={}, alignment_by_model=align_by_model
        )
    else:
        pair_kind = "word"
        harvested = _harvest_word_level(pass_summary_ext, grid, align_by_model, ctx["duration_s"])
    rows = ctx["state"].axis_rows("asr")
    covered = region_buckets(region, rows)
    touched: dict[str, set[tuple[float, float]]] = {}
    n_votes = 0
    for entry in harvested:
        bk = bucket_key(entry["start"], entry["end"])
        if bk not in covered:
            continue
        for source, payload in entry["votes"].items():
            if isinstance(payload, dict) and source == "__pairwise_phoneme_distances__":
                payload = {**payload, "pair_distance_kind": pair_kind}
            ctx["store"].add_vote(
                Vote(
                    axis="asr",
                    bucket=bk,
                    source=source,
                    stream=stream,
                    scope=f"region:{region['region_id']}",
                    round=ctx["round_idx"],
                    payload=payload if isinstance(payload, dict) else {"value": payload},
                    provenance={"rule": "U1_region_reasr", "models": ran, "stream_fallback": stream_fallback},
                )
            )
            n_votes += 1
        touched.setdefault("asr", set()).add(bk)
    return {
        "models": ran,
        "stream": stream,
        "stream_fallback": stream_fallback,
        "pair_distance_kind": pair_kind,
        "votes_added": n_votes,
        "touched": touched,
    }


# ── I1 + I2: speaker repair from embeddings (stored artifacts or live) ─


def _get_identity_repair(ctx: dict[str, Any], stream: str) -> dict[str, Any] | None:
    """Compute (once per stream) the change-point + recluster repair."""
    cache = ctx.setdefault("_identity_repair", {})
    if stream in cache:
        return cache[stream]
    from senselab.audio.workflows.audio_analysis.adaptive.fusion import make_p_voice_lookup
    from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import repair_identity

    window_embeddings: dict[str, list[dict[str, Any]]] = {}
    emb_dir = perturbation_dir(ctx["run_dir"], stream) / "embeddings"
    if emb_dir.is_dir():
        for f in sorted(emb_dir.glob("*.json")):
            try:
                payload = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            windows = payload.get("windows") or []
            if len(windows) >= 4:
                window_embeddings[f.stem] = windows
    diar_boundaries: list[float] = []
    for block in load_outcomes_dir(ctx["run_dir"], stream, "diarization").values():
        res = block.get("result")
        if isinstance(res, list) and res:
            inner = res[0] if isinstance(res[0], list) else res
            for seg in inner:
                if isinstance(seg, dict):
                    for key in ("start", "end"):
                        v = seg.get(key)
                        if v is not None and 0.05 < float(v) < ctx["duration_s"] - 0.05:
                            diar_boundaries.append(float(v))
    repaired = None
    if window_embeddings:
        repaired = repair_identity(
            window_embeddings=window_embeddings,
            diar_boundaries=diar_boundaries,
            p_voice_at=make_p_voice_lookup(ctx["state"]),
            duration_s=ctx["duration_s"],
            policy=ctx["policy"],
        )
    cache[stream] = repaired
    return repaired


def _i1_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or region["axis"] != "speaker":
        return False, {}
    stream = _action_stream(region)
    if (stream, region["core_start"], region["core_end"]) in ctx.get("_i1_done", set()):
        return False, {}
    return True, {"crop": [region["crop_start"], region["crop_end"]], "stream": stream}


def _i1_guard(region: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    emb_dir = ctx["run_dir"] / _action_stream(region) / "embeddings"
    if not emb_dir.is_dir() or not any(emb_dir.glob("*.json")):
        if _spec_missing("torch") or _spec_missing("speechbrain"):
            return "embedding_backend_unavailable (no stored embeddings, no live backend)"
    return None


def _i1_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    region = cand["region"]
    stream = cand["trigger"]["stream"]
    repaired = _get_identity_repair(ctx, stream)
    if repaired is None:
        raise RuntimeError("identity_repair_no_embeddings")
    ctx.setdefault("_i1_done", set()).add((stream, region["core_start"], region["core_end"]))
    touched: dict[str, set[tuple[float, float]]] = {}
    n_votes = 0
    in_region = [c for c in repaired["change_points"] if region["crop_start"] <= c["time"] <= region["crop_end"]]
    for row in _rows_in_span(ctx["state"].axis_rows("speaker"), region["core_start"], region["core_end"]):
        bk = bucket_key(row["start"], row["end"])
        cps_here = [c for c in in_region if row["start"] <= c["time"] < row["end"]]
        if not cps_here:
            continue
        ctx["store"].add_vote(
            Vote(
                axis="speaker",
                bucket=bk,
                source="embedding_changepoint/consensus",
                stream=stream,
                scope="file",
                round=ctx["round_idx"],
                payload={
                    "change_point_times": [c["time"] for c in cps_here],
                    "change_point_confidence": max(c["confidence"] for c in cps_here),
                },
                provenance={"rule": "I1_boundary_refinement", "models": repaired["models_used"]},
            )
        )
        touched.setdefault("speaker", set()).add(bk)
        n_votes += 1
    return {
        "change_points_in_region": in_region,
        "votes_added": n_votes,
        "models": repaired["models_used"],
        "touched": touched,
    }


def _i2_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or region["axis"] != "speaker":
        return False, {}
    stream = _action_stream(region)
    if ctx.get("refined_identity", {}).get(stream):
        return False, {}  # once per stream per run
    return True, {"stream": stream}


def _i2_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import (
        cluster_at,
        cross_source_disagreement,
    )

    stream = cand["trigger"]["stream"]
    repaired = _get_identity_repair(ctx, stream)
    if repaired is None:
        raise RuntimeError("identity_repair_no_embeddings")
    ctx.setdefault("refined_identity", {})[stream] = repaired
    touched: dict[str, set[tuple[float, float]]] = {}
    n_votes = 0
    for row in ctx["state"].axis_rows("speaker"):
        bk = bucket_key(row["start"], row["end"])
        mid = (row["start"] + row["end"]) / 2.0
        cid = cluster_at(repaired, mid)
        prev_mid = mid - (row["end"] - row["start"])
        changed = cluster_at(repaired, prev_mid) != cid if prev_mid >= 0 else False
        ctx["store"].add_vote(
            Vote(
                axis="speaker",
                bucket=bk,
                source="embedding_recluster/consensus",
                stream=stream,
                scope="file",
                round=ctx["round_idx"],
                payload={
                    "speaker_label": cid or "SIL",
                    "cluster_id": cid or "SIL",
                    "speaker_changed_from_prev": changed,
                },
                provenance={"rule": "I2_recluster", "n_clusters": repaired["n_clusters"]},
            )
        )
        # Recompute cross-source label disagreement including the new voter
        # (overwrites the file-scope vote deterministically — same vote id).
        ids = []
        for source, payload in ctx["store"].active_votes(stream, "speaker", bk).items():
            if source.startswith("__") or "::" in source:
                continue
            c = payload.get("cluster_id")
            if c and c not in ("SIL", "<silent>"):
                ids.append(str(c))
        value = cross_source_disagreement(ids)
        if value is not None:
            ctx["store"].add_vote(
                Vote(
                    axis="speaker",
                    bucket=bk,
                    source="__cross_diar_label_disagreement__",
                    stream=stream,
                    scope="file",
                    round=ctx["round_idx"],
                    payload={"value": value, "n_sources": len(ids), "recomputed_by": "I2_recluster"},
                    provenance={"rule": "I2_recluster"},
                )
            )
        touched.setdefault("speaker", set()).add(bk)
        n_votes += 2
    return {
        "n_clusters": repaired["n_clusters"],
        "n_segments": len(repaired["segments"]),
        "change_points": len(repaired["change_points"]),
        "models": repaired["models_used"],
        "votes_added": n_votes,
        "touched": touched,
    }


# ── I4: overlap posteriors (gated live backend) ─────────────────────────


# ── P2: fine-grid speech_presence re-analysis ───────────────────────────────────


def _p2_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Fire when a speech_presence region's evidence is dominated by coarse voters.

    Two independent reasons, per contracts/interventions.md:

    1. **Coarse dominance** — at least half the active votes carry ``coarse=True``
       (sentence-level ASR, per-30 s Whisper no_speech, AST's 10.24 s window,
       YAMNet's 0.96 s window, ~1 s embedding silhouette). Those voters cast one
       identical vote across every bucket they span, so agreement among them is an
       artifact of their window size rather than evidence about this bucket.
    2. **Frame instability** — the speech_presence rows already report
       ``frame_dispersion`` from the round-1 posteriors; a high value means the
       bucket straddles an onset, which a finer grid can localize.
    """
    if region is None or region["axis"] != "speech_presence":
        return False, {}
    stream = _action_stream(region)
    rows = _rows_in_span(ctx["state"].axis_rows("speech_presence"), region["core_start"], region["core_end"])
    if not rows:
        return False, {}

    # Vote payloads live in the store, not on the belief row — the row only
    # carries `contributing_sources` (names). Reading the row here would silently
    # see zero votes and never fire.
    store = ctx["store"]
    coarse = active = 0
    instability: list[float] = []
    for row in rows:
        bk = bucket_key(row["start"], row["end"])
        for source, payload in (store.active_votes(stream, "speech_presence", bk) or {}).items():
            if source.startswith("__") or not isinstance(payload, dict):
                continue
            if payload.get("speaks") is None:
                continue
            active += 1
            if payload.get("coarse"):
                coarse += 1
        fi = (row.get("meta") or {}).get("frame_dispersion")
        if fi is not None:
            instability.append(float(fi))

    coarse_share = (coarse / active) if active else 0.0
    mean_instability = (sum(instability) / len(instability)) if instability else 0.0
    threshold = float(((ctx["policy"].get("speech_presence") or {}).get("coarse_share_threshold", 0.5)))
    fires = coarse_share >= threshold or mean_instability > 0.0
    return fires, {
        "stream": stream,
        "coarse_share": round(coarse_share, 4),
        "n_active_votes": active,
        "mean_frame_dispersion": round(mean_instability, 4),
        "reason": "coarse_dominance" if coarse_share >= threshold else "frame_dispersion",
    }


def _p2_guard(region: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    """Same prerequisites as any posterior re-analysis: the model, a token, audio."""
    import os  # noqa: PLC0415

    if _spec_missing("pyannote"):
        return "posteriors_unavailable (pyannote.audio not installed)"
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        return "posteriors_unavailable (HF token required for pyannote/segmentation-3.0)"
    if not ctx.get("input_audio"):
        return "input_audio_missing"
    return None


def _p2_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Re-run segmentation-3.0 on the crop and replace the region's speech_presence evidence.

    The replacement vote is scoped ``region:<id>`` so it supersedes the coarse
    round-1 voters over this span without deleting them — the store keeps both and
    the later scope wins, which is what makes the decision log auditable.

    Emits ``overlap_posterior`` on covered rows as a side effect, which is why the
    contract lets I4 run "light (reuses P2 output)" afterwards.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.audio_io import get_stream_wav  # noqa: PLC0415
    from senselab.audio.workflows.audio_analysis.adaptive.backends import overlap_posteriors  # noqa: PLC0415

    region = cand["region"]
    stream = cand["trigger"]["stream"]
    wav, reason = get_stream_wav(ctx, stream)
    if wav is None:
        raise RuntimeError(f"audio_unavailable: {reason}")
    post, err = overlap_posteriors(wav, span=(region["crop_start"], region["crop_end"]))
    if post is None:
        raise RuntimeError(err or "posteriors_failed")

    hop = float(post["frame_hop"])
    speech, overlap = post["speech"], post["overlap"]
    crop_start = float(region["crop_start"])

    def _mean_over(track: list[float], s: float, e: float) -> float | None:
        lo = max(0, int((s - crop_start) / hop))
        hi = min(len(track), int((e - crop_start) / hop) + 1)
        vals = track[lo:hi]
        return float(sum(vals) / len(vals)) if vals else None

    def _instability(s: float, e: float) -> float | None:
        lo = max(0, int((s - crop_start) / hop))
        hi = min(len(speech), int((e - crop_start) / hop) + 1)
        vals = speech[lo:hi]
        if len(vals) < 2:
            return None
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return float(min(1.0, 2.0 * (var**0.5)))

    rows = ctx["state"].axis_rows("speech_presence")
    covered = region_buckets(region, rows)
    touched: dict[str, set[tuple[float, float]]] = {}
    n_votes = 0
    for row in _rows_in_span(rows, region["core_start"], region["core_end"]):
        bk = bucket_key(row["start"], row["end"])
        if bk not in covered:
            continue  # merge-back midpoint rule (D2)
        p_speech = _mean_over(speech, row["start"], row["end"])
        if p_speech is None:
            continue
        ctx["store"].add_vote(
            Vote(
                axis="speech_presence",
                bucket=bk,
                source="frame_posterior_fine",
                stream=stream,
                scope=f"region:{region['region_id']}",
                round=ctx["round_idx"],
                payload={
                    # Directed through the shared builder rather than hand-shaped here. This vote
                    # is the *finest* presence evidence the loop can buy, and it was the last
                    # producer still reporting the raw posterior as its confidence: at p = 0.02
                    # `presence_probability` read the payload as P(speech) = 0.98, so the round of
                    # re-analysis bought to resolve a doubtful bucket asserted the opposite of what
                    # the detector measured.
                    **directed_presence_vote(p_speech),
                    "frame_mean": round(p_speech, 6),
                    "frame_dispersion": _instability(row["start"], row["end"]),
                    "coarse": False,
                },
                provenance={"rule": "P2_fine_posteriors", "frame_hop_s": hop},
            )
        )
        n_votes += 1
        ov = _mean_over(overlap, row["start"], row["end"])
        if ov is not None:
            row.setdefault("meta", {})["overlap_posterior"] = round(ov, 4)
            row["overlap_posterior"] = round(ov, 4)
        touched.setdefault("speech_presence", set()).add(bk)

    return {
        "frame_hop_s": hop,
        "frames": len(speech),
        "votes_added": n_votes,
        "mean_p_speech_in_core": (
            round(v, 4) if (v := _mean_over(speech, region["core_start"], region["core_end"])) is not None else None
        ),
        "n_classes": post.get("n_classes"),
        "touched": touched,
    }


def _i4_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or region["axis"] != "speaker":
        return False, {}
    co = [
        r
        for r in ctx.get("all_regions", [])
        if r["axis"] == "asr"
        and r.get("stream") == _action_stream(region)
        and min(r["core_end"], region["core_end"]) - max(r["core_start"], region["core_start"]) > 0
    ]
    return bool(co), {
        "co_located_asr_regions": [r["region_id"] for r in co],
        "stream": _action_stream(region),
    }


def _i4_guard(region: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    import os

    if _spec_missing("pyannote"):
        return "posteriors_unavailable (pyannote.audio not installed)"
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        return "posteriors_unavailable (HF token required for pyannote/segmentation-3.0)"
    if not ctx.get("input_audio"):
        return "input_audio_missing"
    return None


def _i4_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    from senselab.audio.workflows.audio_analysis.adaptive.audio_io import get_stream_wav
    from senselab.audio.workflows.audio_analysis.adaptive.backends import overlap_posteriors

    region = cand["region"]
    stream = cand["trigger"]["stream"]
    # Overlap is a property of the scene, not the processing stream: fall back
    # to the raw waveform when the region's stream can't be materialized, and
    # apply the posterior to every stream's rows over the span.
    wav, reason = get_stream_wav(ctx, stream)
    source_stream, stream_fallback = stream, None
    if wav is None and stream != IDENTITY_NAME:
        stream_fallback = reason
        source_stream = IDENTITY_NAME
        wav, reason = get_stream_wav(ctx, source_stream)
    if wav is None:
        raise RuntimeError(f"audio_unavailable: {reason}")
    post, err = overlap_posteriors(wav, span=(region["crop_start"], region["crop_end"]))
    if post is None:
        raise RuntimeError(err or "posteriors_failed")
    touched: dict[str, set[tuple[float, float]]] = {}
    hop, overlap = float(post["frame_hop"]), post["overlap"]

    def _mean_overlap(s: float, e: float) -> float | None:
        lo = max(0, int((s - region["crop_start"]) / hop))
        hi = min(len(overlap), int((e - region["crop_start"]) / hop) + 1)
        vals = overlap[lo:hi]
        return float(sum(vals) / len(vals)) if vals else None

    mean_over_core = _mean_overlap(region["core_start"], region["core_end"])
    for apply_stream in ctx["passes"]:
        for axis in OVERLAP_INFORMED_AXES:
            for row in _rows_in_span(ctx["state"].axis_rows(axis), region["core_start"], region["core_end"]):
                ov = _mean_overlap(row["start"], row["end"])
                if ov is None:
                    continue
                row.setdefault("meta", {})["overlap_posterior"] = round(ov, 4)
                row["overlap_posterior"] = round(ov, 4)
                touched.setdefault(axis, set()).add(bucket_key(row["start"], row["end"]))
    return {
        "source_stream": source_stream,
        "stream_fallback": stream_fallback,
        "frames": len(overlap),
        "mean_overlap_in_core": round(mean_over_core, 4) if mean_over_core is not None else None,
        "n_classes": post.get("n_classes"),
        "touched": touched,
    }


def _mass_gain(region: dict[str, Any], ctx: dict[str, Any], trigger: dict[str, Any]) -> float:
    return float(region["uncertainty_mass"]) if region else 0.0


def _n_candidates_gain(region: dict[str, Any], ctx: dict[str, Any], trigger: dict[str, Any]) -> float:
    return float(trigger.get("n_candidates") or 0)


RULES: list[dict[str, Any]] = [
    {
        "id": "S1_stream_election",
        "axes": list(AXIS_NAMES),
        "cost": "light",
        "trigger": _s1_trigger,
        "guard": None,
        "gain": _mass_gain,
        "execute": _s1_execute,
    },
    {
        "id": "P3_uncorroborated_speech_attenuation",
        "axes": [],  # stream-global, runs at most once per round
        "meta_axis": "speech_presence",
        "cost": "light",
        "trigger": _p3_trigger,
        "guard": None,
        "gain": _n_candidates_gain,
        "execute": _p3_execute,
    },
    {
        "id": "C9_missed_speech",
        "axes": [],
        "meta_axis": "speech_presence",
        "cost": "light",
        "trigger": _c9_trigger,
        "guard": None,
        "gain": _n_candidates_gain,
        "execute": _c9_execute,
    },
    {
        "id": "U2_reserve_escalation",
        "axes": ["asr"],
        "cost": "medium",
        "trigger": _u2_trigger,
        "guard": None,
        "gain": _u2_gain,
        "execute": _u2_execute,
    },
    {
        "id": "U1_region_reasr",
        "axes": ["asr"],
        "cost": "medium",
        "trigger": _u1_trigger,
        "guard": _u1_guard,
        "gain": _mass_gain,
        "execute": _u1_execute,
    },
    {
        "id": "I1_boundary_refinement",
        "axes": ["speaker"],
        "cost": "light",
        "trigger": _i1_trigger,
        "guard": _i1_guard,
        "gain": _mass_gain,
        "execute": _i1_execute,
    },
    {
        "id": "I2_recluster",
        "axes": ["speaker"],
        "cost": "light",
        "trigger": _i2_trigger,
        "guard": _i1_guard,  # same requirement: stored embeddings or live backend
        "gain": _mass_gain,
        "execute": _i2_execute,
    },
    {
        "id": "P2_fine_posteriors",
        "axes": ["speech_presence"],
        "cost": "medium",
        "trigger": _p2_trigger,
        "guard": _p2_guard,
        "gain": _mass_gain,
        "execute": _p2_execute,
    },
    {
        "id": "I4_overlap_detection",
        "axes": ["speaker"],
        "cost": "medium",
        "trigger": _i4_trigger,
        "guard": _i4_guard,
        "gain": _mass_gain,
        "execute": _i4_execute,
    },
]
