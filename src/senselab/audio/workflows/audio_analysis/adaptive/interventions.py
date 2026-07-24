"""Intervention catalog (prototype) — contracts/interventions.md.

Implemented for real on artifacts + the content-addressable cache:

- ``S1_stream_election`` — per-region raw/enhanced election from belief evidence.
- ``P3_hallucination_adjudication`` / ``C9_missed_speech`` — cross-signal repair
  over existing evidence (C10 / C9), degraded to the indicators present in the
  ingested run (no whisper ``no_speech_prob`` / PPG unless the run had them).
- ``U2_reserve_escalation`` — adds reserve ASR models by **cache replay**: the
  reserve model's whole-file result is read from ``analyze_audio``'s
  content-addressable cache (same audio signature ⇒ same waveform), windowed to
  the region's buckets with the *same* harvest math the comparator uses
  (``harvest_utterance_votes``), and merged as region-scoped votes.

- ``U1_region_reasr`` — live region re-ASR (HF whisper pipeline pool, enhanced stream
  regenerated on demand with recorded raw fallback).
- ``I1_boundary_refinement`` / ``I2_recluster`` — identity repair from per-window
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
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.belief import Vote, bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.policy import family_weights, model_family
from senselab.audio.workflows.audio_analysis.adaptive.regions import region_buckets
from senselab.audio.workflows.audio_analysis.aggregate import _normalize_transcript_for_wer
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    _levenshtein,
    asr_alignment_score_in_window,
    asr_text_in_window,
    resolve_asr_result,
    whisper_bucket_avg_logprob,
)

# ── shared artifact/cache access ─────────────────────────────────────────


def load_outcomes_dir(run_dir: Path, stream: str, task_dir: str) -> dict[str, dict[str, Any]]:
    """Load ``<run_dir>/<stream>/<task_dir>/*.json`` keyed by provenance.model_id.

    Each payload records its ``_file_stem`` — alignment files are keyed by the
    *aligner* model id in provenance but written under the parent ASR model's
    safe filename, so cross-task joins go through the stem.
    """
    out: dict[str, dict[str, Any]] = {}
    d = run_dir / stream / task_dir
    if not d.is_dir():
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
    d = run_dir / stream / "alignment"
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


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and v == v]
    return sum(clean) / len(clean) if clean else None


def _election_scores(region: dict[str, Any], ctx: dict[str, Any]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    w = ctx["policy"]["election"]["weights"]
    for stream in ctx["passes"]:
        pres = _rows_in_span(ctx["state"].axis_rows(stream, "presence"), region["core_start"], region["core_end"])
        utt = _rows_in_span(ctx["state"].axis_rows(stream, "utterance"), region["core_start"], region["core_end"])
        p_conf = _mean([r.get("p_voice") for r in pres]) or 0.0
        degr = _mean(
            [
                max(
                    float((r.get("meta") or {}).get("quality_snr") or 0.0),
                    float((r.get("meta") or {}).get("quality_clip") or 0.0),
                    float((r.get("meta") or {}).get("quality_reverb") or 0.0),
                )
                for r in pres
            ]
        )
        quality = 1.0 - (degr if degr is not None else 0.0)
        agree = 1.0 - (_mean([r.get("aggregated_uncertainty") for r in utt]) or 0.0)
        total = w["presence_conf"] * p_conf + w["quality"] * quality + w["utterance_agreement"] * agree
        scores[stream] = {
            "presence_conf": round(p_conf, 6),
            "quality": round(quality, 6),
            "utterance_agreement": round(agree, 6),
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
    elected = max(scores, key=lambda s: (scores[s]["total"], s == "raw_16k"))
    guard_fired = False
    if elected != "raw_16k":
        # Enhancement-artifact guard (degraded): reject the enhanced stream when
        # it claims speech text where the raw stream has none *and* raw presence
        # is confidently silent — SepFormer can synthesize speech-like energy.
        raw_text = _region_text(ctx, "raw_16k", region)
        enh_text = _region_text(ctx, elected, region)
        raw_pres = _mean(
            [
                r.get("p_voice")
                for r in _rows_in_span(
                    ctx["state"].axis_rows("raw_16k", "presence"), region["core_start"], region["core_end"]
                )
            ]
        )
        if enh_text and not raw_text and (raw_pres is not None and raw_pres < 0.2):
            guard_fired = True
            elected = "raw_16k"
    election = {
        "region_id": region["region_id"],
        "scores": scores,
        "elected": elected,
        "guard_fired": guard_fired,
    }
    ctx["elections"][region["region_id"]] = election
    region["elected_stream"] = elected
    return {"election": election, "touched": {}}


def _region_text(ctx: dict[str, Any], stream: str, region: dict[str, Any]) -> str:
    texts = []
    for row in _rows_in_span(ctx["state"].axis_rows(stream, "utterance"), region["core_start"], region["core_end"]):
        bk = bucket_key(row["start"], row["end"])
        for source, payload in ctx["store"].active_votes(stream, "utterance", bk).items():
            if not source.startswith("__") and payload.get("text"):
                texts.append(str(payload["text"]))
    return " ".join(texts).strip()


# ── P3 / C9: adjudication over existing evidence (C10 / C9) ─────────────


def _adjudication_candidates(ctx: dict[str, Any], stream: str) -> list[dict[str, Any]]:
    adj = ctx["policy"]["adjudication"]
    out = []
    for row in ctx["state"].axis_rows(stream, "presence"):
        p_voice = row.get("p_voice")
        if p_voice is None or p_voice >= adj["p_voice_hallucination"]:
            continue
        bk = bucket_key(row["start"], row["end"])
        votes = ctx["store"].active_votes(stream, "presence", bk)
        for source, payload in votes.items():
            if source.startswith(("__", "acoustic_", "frame_", "embedding_", "ast", "yamnet")):
                continue
            if source not in ctx["asr_model_ids"].get(stream, set()):
                continue
            if not payload.get("speaks"):
                continue
            meta = row.get("meta") or {}
            nc = payload.get("native_confidence")
            indicators = {
                "low_native_confidence": nc is not None and float(nc) < adj["low_native_confidence"],
                "low_src_speech": (meta.get("src_speech") is not None)
                and float(meta.get("src_speech") or 1.0) < adj["low_src_speech"],
                "very_low_p_voice": float(p_voice) < adj["p_voice_hallucination"] / 2.0,
            }
            if sum(indicators.values()) >= adj["min_indicators"]:
                out.append({"bucket": bk, "source": source, "indicators": indicators, "p_voice": p_voice})
    return out


def _p3_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    cands = {s: _adjudication_candidates(ctx, s) for s in ctx["passes"]}
    n = sum(len(v) for v in cands.values())
    return n > 0, {"n_candidates": n}


def _p3_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    touched: dict[tuple[str, str], set[tuple[float, float]]] = {}
    purged = []
    for stream in ctx["passes"]:
        for c in _adjudication_candidates(ctx, stream):
            n = ctx["store"].purge_source_in_bucket(
                stream, c["bucket"], c["source"], reason="hallucination_adjudicated", round_idx=ctx["round_idx"]
            )
            # Purge also hits utterance buckets overlapping this presence bucket.
            for urow in _rows_in_span(ctx["state"].axis_rows(stream, "utterance"), c["bucket"][0], c["bucket"][1]):
                ubk = bucket_key(urow["start"], urow["end"])
                n += ctx["store"].purge_source_in_bucket(
                    stream, ubk, c["source"], reason="hallucination_adjudicated", round_idx=ctx["round_idx"]
                )
                touched.setdefault((stream, "utterance"), set()).add(ubk)
            touched.setdefault((stream, "presence"), set()).add(c["bucket"])
            purged.append({**c, "stream": stream, "votes_purged": n})
    return {"purged": purged, "touched": touched}


def _c9_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    n = sum(len(_missed_speech_candidates(ctx, s)) for s in ctx["passes"])
    return n > 0, {"n_candidates": n}


def _missed_speech_candidates(ctx: dict[str, Any], stream: str) -> list[dict[str, Any]]:
    adj = ctx["policy"]["adjudication"]
    out = []
    for row in ctx["state"].axis_rows(stream, "presence"):
        p_voice = row.get("p_voice")
        if p_voice is None or not (adj["p_voice_hallucination"] <= p_voice < adj["p_voice_missed"]):
            continue
        bk = bucket_key(row["start"], row["end"])
        if "adjudicator/missed_speech" in ctx["store"].active_votes(stream, "presence", bk):
            continue
        families = set()
        for urow in _rows_in_span(ctx["state"].axis_rows(stream, "utterance"), bk[0], bk[1]):
            ubk = bucket_key(urow["start"], urow["end"])
            for source, payload in ctx["store"].active_votes(stream, "utterance", ubk).items():
                if not source.startswith("__") and payload.get("text"):
                    families.add(model_family(source, ctx["policy"]))
        if len(families) >= 2:
            out.append({"bucket": bk, "families": sorted(families), "p_voice": p_voice})
    return out


def _c9_execute(cand: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    touched: dict[tuple[str, str], set[tuple[float, float]]] = {}
    added = []
    weight = float(ctx["policy"]["adjudication"]["missed_speech_weight"])
    for stream in ctx["passes"]:
        for c in _missed_speech_candidates(ctx, stream):
            ctx["store"].add_vote(
                Vote(
                    axis="presence",
                    bucket=c["bucket"],
                    source="adjudicator/missed_speech",
                    stream=stream,
                    scope="file",
                    round=ctx["round_idx"],
                    payload={"speaks": True, "native_confidence": None, "weight": weight},
                    provenance={"families_agreeing": c["families"], "rule": "C9_missed_speech"},
                )
            )
            touched.setdefault((stream, "presence"), set()).add(c["bucket"])
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
    if region is None or region["axis"] != "utterance":
        return False, {}
    stream = region.get("elected_stream") or region.get("stream") or "raw_16k"
    reserves = _reserves_in_cache(ctx, stream)
    if not reserves:
        return False, {"reason": "no_cached_reserves"}
    rows = _rows_in_span(ctx["state"].axis_rows(stream, "utterance"), region["core_start"], region["core_end"])
    epi = _mean([r.get("epistemic") for r in rows])
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
    grid = BucketGrid(win_length=ctx["utterance_grid"][0], hop_length=ctx["utterance_grid"][1])

    pair_kind = "phoneme"
    if _has_g2p():
        from senselab.audio.workflows.audio_analysis.utterance import harvest_utterance_votes

        harvested = harvest_utterance_votes(
            pass_summary=pass_summary_ext, grid=grid, ppg_block={}, alignment_by_model=align_by_model
        )
    else:
        pair_kind = "word"
        harvested = _harvest_word_level(pass_summary_ext, grid, align_by_model, ctx["duration_s"])

    rows = ctx["state"].axis_rows(stream, "utterance")
    covered = region_buckets(region, rows)
    touched: dict[tuple[str, str], set[tuple[float, float]]] = {}
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
                    axis="utterance",
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
        touched.setdefault((stream, "utterance"), set()).add(bk)
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

    Same vote schema as ``harvest_utterance_votes`` so ``aggregate_utterance``
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
    if region is None or region["axis"] != "utterance":
        return False, {}
    models = [m for m in (ctx["policy"].get("u1_asr_models") or []) if m not in ctx.get("live_asr_done", set())]
    if not models:
        return False, {"reason": "no_u1_models"}
    stream = region.get("elected_stream") or region.get("stream") or "raw_16k"
    rows = _rows_in_span(ctx["state"].axis_rows(stream, "utterance"), region["core_start"], region["core_end"])
    epi = _mean([r.get("epistemic") for r in rows])
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
    if wav is None and stream != "raw_16k":
        stream_fallback = reason
        stream = "raw_16k"
        wav, reason = get_stream_wav(ctx, stream)
    if wav is None:
        raise RuntimeError(f"audio_unavailable: {reason}")

    crop_start, crop_end = float(region["crop_start"]), float(region["crop_end"])
    segment = crop_wav(wav, crop_start, crop_end)
    language = ctx["policy"].get("language")
    ran: list[dict[str, Any]] = []
    ext_blocks: dict[str, dict[str, Any]] = {}
    for model in trigger["models"]:
        words, err = transcribe_crop(segment, model_id=model, offset_s=crop_start, language=language)
        if words is None:
            ran.append({"model": model, "status": "failed", "error": err})
            continue
        ran.append({"model": model, "status": "ok", "n_words": len(words)})
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
    grid = BucketGrid(win_length=ctx["utterance_grid"][0], hop_length=ctx["utterance_grid"][1])
    pair_kind = "phoneme"
    if _has_g2p():
        from senselab.audio.workflows.audio_analysis.utterance import harvest_utterance_votes

        harvested = harvest_utterance_votes(
            pass_summary=pass_summary_ext, grid=grid, ppg_block={}, alignment_by_model=align_by_model
        )
    else:
        pair_kind = "word"
        harvested = _harvest_word_level(pass_summary_ext, grid, align_by_model, ctx["duration_s"])
    rows = ctx["state"].axis_rows(stream, "utterance")
    covered = region_buckets(region, rows)
    touched: dict[tuple[str, str], set[tuple[float, float]]] = {}
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
                    axis="utterance",
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
        touched.setdefault((stream, "utterance"), set()).add(bk)
    return {
        "models": ran,
        "stream": stream,
        "stream_fallback": stream_fallback,
        "pair_distance_kind": pair_kind,
        "votes_added": n_votes,
        "touched": touched,
    }


# ── I1 + I2: identity repair from embeddings (stored artifacts or live) ─


def _get_identity_repair(ctx: dict[str, Any], stream: str) -> dict[str, Any] | None:
    """Compute (once per stream) the change-point + recluster repair."""
    cache = ctx.setdefault("_identity_repair", {})
    if stream in cache:
        return cache[stream]
    from senselab.audio.workflows.audio_analysis.adaptive.fusion import make_p_voice_lookup
    from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import repair_identity

    window_embeddings: dict[str, list[dict[str, Any]]] = {}
    emb_dir = ctx["run_dir"] / stream / "embeddings"
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
            p_voice_at=make_p_voice_lookup(ctx["state"], stream),
            duration_s=ctx["duration_s"],
            policy=ctx["policy"],
        )
    cache[stream] = repaired
    return repaired


def _i1_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or region["axis"] != "identity":
        return False, {}
    stream = region.get("stream") or "raw_16k"
    if (stream, region["core_start"], region["core_end"]) in ctx.get("_i1_done", set()):
        return False, {}
    return True, {"crop": [region["crop_start"], region["crop_end"]], "stream": stream}


def _i1_guard(region: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    emb_dir = ctx["run_dir"] / (region.get("stream") or "raw_16k") / "embeddings"
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
    touched: dict[tuple[str, str], set[tuple[float, float]]] = {}
    n_votes = 0
    in_region = [c for c in repaired["change_points"] if region["crop_start"] <= c["time"] <= region["crop_end"]]
    for row in _rows_in_span(ctx["state"].axis_rows(stream, "identity"), region["core_start"], region["core_end"]):
        bk = bucket_key(row["start"], row["end"])
        cps_here = [c for c in in_region if row["start"] <= c["time"] < row["end"]]
        if not cps_here:
            continue
        ctx["store"].add_vote(
            Vote(
                axis="identity",
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
        touched.setdefault((stream, "identity"), set()).add(bk)
        n_votes += 1
    return {
        "change_points_in_region": in_region,
        "votes_added": n_votes,
        "models": repaired["models_used"],
        "touched": touched,
    }


def _i2_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or region["axis"] != "identity":
        return False, {}
    stream = region.get("stream") or "raw_16k"
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
    touched: dict[tuple[str, str], set[tuple[float, float]]] = {}
    n_votes = 0
    for row in ctx["state"].axis_rows(stream, "identity"):
        bk = bucket_key(row["start"], row["end"])
        mid = (row["start"] + row["end"]) / 2.0
        cid = cluster_at(repaired, mid)
        prev_mid = mid - (row["end"] - row["start"])
        changed = cluster_at(repaired, prev_mid) != cid if prev_mid >= 0 else False
        ctx["store"].add_vote(
            Vote(
                axis="identity",
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
        for source, payload in ctx["store"].active_votes(stream, "identity", bk).items():
            if source.startswith("__") or "::" in source:
                continue
            c = payload.get("cluster_id")
            if c and c not in ("SIL", "<silent>"):
                ids.append(str(c))
        value = cross_source_disagreement(ids)
        if value is not None:
            ctx["store"].add_vote(
                Vote(
                    axis="identity",
                    bucket=bk,
                    source="__cross_diar_label_disagreement__",
                    stream=stream,
                    scope="file",
                    round=ctx["round_idx"],
                    payload={"value": value, "n_sources": len(ids), "recomputed_by": "I2_recluster"},
                    provenance={"rule": "I2_recluster"},
                )
            )
        touched.setdefault((stream, "identity"), set()).add(bk)
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


def _i4_trigger(region: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if region is None or region["axis"] != "identity":
        return False, {}
    co = [
        r
        for r in ctx.get("all_regions", [])
        if r["axis"] == "utterance"
        and r.get("stream") == region.get("stream")
        and min(r["core_end"], region["core_end"]) - max(r["core_start"], region["core_start"]) > 0
    ]
    return bool(co), {
        "co_located_utterance_regions": [r["region_id"] for r in co],
        "stream": region.get("stream") or "raw_16k",
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
    if wav is None and stream != "raw_16k":
        stream_fallback = reason
        source_stream = "raw_16k"
        wav, reason = get_stream_wav(ctx, source_stream)
    if wav is None:
        raise RuntimeError(f"audio_unavailable: {reason}")
    post, err = overlap_posteriors(wav, span=(region["crop_start"], region["crop_end"]))
    if post is None:
        raise RuntimeError(err or "posteriors_failed")
    touched: dict[tuple[str, str], set[tuple[float, float]]] = {}
    hop, overlap = float(post["frame_hop"]), post["overlap"]

    def _mean_overlap(s: float, e: float) -> float | None:
        lo = max(0, int((s - region["crop_start"]) / hop))
        hi = min(len(overlap), int((e - region["crop_start"]) / hop) + 1)
        vals = overlap[lo:hi]
        return float(sum(vals) / len(vals)) if vals else None

    mean_over_core = _mean_overlap(region["core_start"], region["core_end"])
    for apply_stream in ctx["passes"]:
        for axis in ("identity", "utterance"):
            for row in _rows_in_span(
                ctx["state"].axis_rows(apply_stream, axis), region["core_start"], region["core_end"]
            ):
                ov = _mean_overlap(row["start"], row["end"])
                if ov is None:
                    continue
                row.setdefault("meta", {})["overlap_posterior"] = round(ov, 4)
                row["overlap_posterior"] = round(ov, 4)
                touched.setdefault((apply_stream, axis), set()).add(bucket_key(row["start"], row["end"]))
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
        "axes": ["presence", "identity", "utterance"],
        "cost": "light",
        "trigger": _s1_trigger,
        "guard": None,
        "gain": _mass_gain,
        "execute": _s1_execute,
    },
    {
        "id": "P3_hallucination_adjudication",
        "axes": [],  # stream-global, runs at most once per round
        "meta_axis": "presence",
        "cost": "light",
        "trigger": _p3_trigger,
        "guard": None,
        "gain": _n_candidates_gain,
        "execute": _p3_execute,
    },
    {
        "id": "C9_missed_speech",
        "axes": [],
        "meta_axis": "presence",
        "cost": "light",
        "trigger": _c9_trigger,
        "guard": None,
        "gain": _n_candidates_gain,
        "execute": _c9_execute,
    },
    {
        "id": "U2_reserve_escalation",
        "axes": ["utterance"],
        "cost": "medium",
        "trigger": _u2_trigger,
        "guard": None,
        "gain": _u2_gain,
        "execute": _u2_execute,
    },
    {
        "id": "U1_region_reasr",
        "axes": ["utterance"],
        "cost": "medium",
        "trigger": _u1_trigger,
        "guard": _u1_guard,
        "gain": _mass_gain,
        "execute": _u1_execute,
    },
    {
        "id": "I1_boundary_refinement",
        "axes": ["identity"],
        "cost": "light",
        "trigger": _i1_trigger,
        "guard": _i1_guard,
        "gain": _mass_gain,
        "execute": _i1_execute,
    },
    {
        "id": "I2_recluster",
        "axes": ["identity"],
        "cost": "light",
        "trigger": _i2_trigger,
        "guard": _i1_guard,  # same requirement: stored embeddings or live backend
        "gain": _mass_gain,
        "execute": _i2_execute,
    },
    {
        "id": "I4_overlap_detection",
        "axes": ["identity"],
        "cost": "medium",
        "trigger": _i4_trigger,
        "guard": _i4_guard,
        "gain": _mass_gain,
        "execute": _i4_execute,
    },
]
