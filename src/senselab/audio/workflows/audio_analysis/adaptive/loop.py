"""Orchestrator: ingest → belief → rounds of policy-ranked interventions → fusion.

Prototype entry point (``run_adaptive_loop``). Artifact-driven: round 1 is the
ingested analyze_audio run; rounds 2..K execute the intervention catalog with
budget + convergence semantics from the spec; the final round fuses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.belief import AXES, BeliefState, VoteStore
from senselab.audio.workflows.audio_analysis.adaptive.convergence import (
    apply_convergence_marks,
    build_convergence_report,
    round_summary,
)
from senselab.audio.workflows.audio_analysis.adaptive.fusion import (
    build_final_outputs,
    collect_word_streams,
    fuse_words,
    make_p_voice_lookup,
    make_speaker_lookup,
)
from senselab.audio.workflows.audio_analysis.adaptive.interventions import (
    RULES,
    _pick_ok,
    build_cache_index,
    load_alignments_matched,
    load_outcomes_dir,
)
from senselab.audio.workflows.audio_analysis.adaptive.policy import BudgetLedger, load_policy, plan_round
from senselab.audio.workflows.audio_analysis.adaptive.regions import propose_regions
from senselab.audio.workflows.audio_analysis.adaptive.types import AxisName, PlannedIntervention, Region
from senselab.audio.workflows.audio_analysis.layout import belief_dir, final_dir


def run_adaptive_loop(
    run_dir: Path,
    *,
    cache_dir: Path | None = None,
    policy_path: Path | None = None,
    out_dir: Path | None = None,
    max_rounds: int = 3,
    aggregator: str | None = None,
    harvests: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    policy_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the adaptive loop over an analyze_audio run.

    Two ingest paths, same loop:

    - **artifact-driven** (default): reads ``summary.json`` and the nine
      uncertainty parquets from ``run_dir``. This is what ``scripts/adaptive_loop.py``
      does over a finished run.
    - **in-process** (T040): the caller passes the ``PassHarvest`` objects it just
      produced via ``harvests`` (and usually ``summary``), skipping the parquet
      round-trip entirely.

    The in-process path cannot run the parity check, and says so rather than
    reporting a passing one: :meth:`VoteStore.parity_check` compares
    re-aggregation against the *stored* parquet values, which don't exist yet when
    the harvests are still in memory. A vacuous "0 mismatches" would be a
    misleading proof, so ``parity_check.status`` is ``"skipped"`` instead.

    Args:
        run_dir: The run directory. Still required for reading policy-adjacent
            artifacts and for output pathing, even on the in-process path.
        cache_dir: Cache directory for intervention re-runs.
        policy_path: Policy YAML; ``None`` uses the packaged default.
        out_dir: Where ``rounds/`` and ``final/`` are written; defaults to ``run_dir``.
        max_rounds: Total rounds including baseline. ``1`` = baseline only.
        aggregator: Sub-signal aggregator; inferred from the run when ``None``.
        harvests: Pass label → ``PassHarvest`` for the in-process path.
        policy_overrides: In-memory policy overrides (CLI flags), merged last.
        summary: Pre-loaded ``summary.json`` content; read from disk when ``None``.

    Returns:
        The loop's decision log.

    Raises:
        ValueError: If no completed passes can be determined.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir
    policy = load_policy(policy_path, policy_overrides)

    if summary is None:
        summary = json.loads((final_dir(run_dir) / "summary.json").read_text())
    passes = [pl for pl, ps in (summary.get("passes") or {}).items() if isinstance(ps, dict) and "duration_s" in ps]
    if not passes:
        raise ValueError(f"no completed passes in {run_dir}/summary.json")
    duration_s = float(summary["passes"][passes[0]]["duration_s"])
    pass_sigs = {pl: str(summary["passes"][pl].get("audio_signature") or "") for pl in passes}
    if aggregator is None:
        aggregator = _aggregator_from_run(run_dir) or "min"

    # ── round 1: ingest + parity (harvest/aggregate split proof) ────────
    t0 = time.time()
    parity: dict[str, Any]
    if harvests is not None:
        store = VoteStore.from_harvests({pl: h for pl, h in harvests.items() if pl in passes})
        parity = {
            "status": "skipped",
            "reason": "in-process harvests carry no stored parquet values to compare against",
        }
    else:
        store = VoteStore.from_run_dir(run_dir, passes)
        parity = store.parity_check(passes, aggregator=aggregator)
    state = BeliefState.from_store(store, passes, aggregator=aggregator)
    asr_grid = _grid_from_rows(state.axis_rows(passes[0], "asr"))
    theta_low = float(policy["thresholds"]["theta_low"])

    rounds_dir = belief_dir(out_dir) / "rounds"
    _write_round_belief(rounds_dir / "1", state, passes)
    (rounds_dir / "1" / "summary.json").write_text(
        json.dumps(
            {
                "round": 1,
                "ingested_from": str(run_dir),
                "parity_check": parity,
                "aggregator": aggregator,
                "uncertainty_mass": {
                    f"{s}/{a}": round(state.uncertainty_mass(s, a, theta_low), 6) for s in passes for a in AXES
                },
            },
            indent=2,
        )
    )

    input_audio = _resolve_input_audio(summary.get("input_audio"), run_dir)
    ctx: dict[str, Any] = {
        "store": store,
        "state": state,
        "policy": policy,
        "run_dir": run_dir,
        "cache_index": build_cache_index(cache_dir),
        "passes": passes,
        "pass_sigs": pass_sigs,
        "duration_s": duration_s,
        "asr_grid": asr_grid,
        "elections": {},
        "input_audio": input_audio,
        "asr_model_ids": {pl: set(load_outcomes_dir(run_dir, pl, "asr").keys()) for pl in passes},
    }

    ledger = BudgetLedger(policy)
    iterations: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    round_states: list[dict[str, Any]] = []
    touch_counts: dict[tuple[str, str, tuple[float, float]], int] = {}
    run_state = "max_rounds"

    # ── rounds 2..K ──────────────────────────────────────────────────────
    for round_idx in range(2, max_rounds + 1):
        ctx["round_idx"] = round_idx
        mass_before = {f"{s}/{a}": round(state.uncertainty_mass(s, a, theta_low), 6) for s in passes for a in AXES}
        regions: list[Region] = []
        for stream in passes:
            for axis in AXES:
                regions.extend(
                    propose_regions(
                        state.axis_rows(stream, axis),
                        axis=axis,
                        stream=stream,
                        policy=policy,
                        round_idx=round_idx,
                        duration_s=duration_s,
                    )
                )
        ctx["all_regions"] = regions
        admitted, not_admitted = plan_round(
            rules=RULES, regions=regions, ctx=ctx, ledger=ledger, policy=policy, round_idx=round_idx
        )

        fired: list[PlannedIntervention] = []
        for cand in admitted:
            rule = next(r for r in RULES if r["id"] == cand["rule"])
            entry = _iteration_entry(cand, round_idx)
            try:
                before_vals = _bucket_values(state)
                result = rule["execute"](cand, ctx)
                touched: dict[tuple[str, AxisName], set] = result.pop("touched", {})
                delta = {}
                # Sorted: these iterations accumulate into output, so a fixed order makes
                # byte-reproducibility structural rather than a property of dict insertion
                # order that a refactor could quietly break (FR-011f).
                for (stream, axis), buckets in sorted(touched.items()):
                    state.update_buckets(store, stream, axis, buckets, round_idx)
                    for bk in buckets:
                        key = (stream, axis, bk)
                        touch_counts[key] = touch_counts.get(key, 0) + 1
                    # Before/after means over the SAME (touched) bucket set.
                    befores = [v for bk in buckets if (v := before_vals.get((stream, axis, bk))) is not None]
                    after = _mean_over(state, stream, axis, buckets)
                    if befores and after is not None:
                        before = sum(befores) / len(befores)
                        delta[f"{stream}/{axis}"] = {
                            "mean_before": round(before, 6),
                            "mean_after": round(after, 6),
                            "delta": round(after - before, 6),
                            "n_buckets": len(buckets),
                        }
                entry.update({"status": "fired", "result": _json_safe_result(result), "delta": delta})
                cand["exec_status"] = "ok"
            except Exception as exc:  # noqa: BLE001 — failure envelope (D11)
                entry.update({"status": "failed", "error": repr(exc)})
                cand["exec_status"] = "failed"
            iterations.append(entry)
            fired.append(cand)
        for cand in not_admitted:
            iterations.append(_iteration_entry(cand, round_idx))

        budget_left = ledger.can_admit("medium") or ledger.can_admit("heavy")
        apply_convergence_marks(state, passes=passes, policy=policy, touch_counts=touch_counts, budget_left=budget_left)
        rs = round_summary(
            round_idx=round_idx,
            state=state,
            passes=passes,
            policy=policy,
            fired=fired,
            not_admitted=not_admitted,
            mass_before=mass_before,
            ledger=ledger,
        )
        round_summaries.append(rs)
        # State snapshot for non-convergence detection (FR-011e). Uncertainty mass plus the
        # bucket-status census is what "the same interpretation as last round" means here: two
        # interpretations trading places move the mass back and forth without either settling,
        # and that is invisible to the "nothing fired" stop below, which sees movement every round.
        round_states.append(
            {
                **{f"mass/{k}": v for k, v in sorted(rs["uncertainty_mass"]["after"].items())},
                **{f"status/{k}": v for k, v in sorted(rs["bucket_statuses"].items())},
            }
        )
        rd = rounds_dir / str(round_idx)
        _write_round_belief(rd, state, passes)
        (rd / "regions.json").write_text(json.dumps(regions, indent=2, default=str))
        (rd / "summary.json").write_text(json.dumps(rs, indent=2, default=str))
        _write_round_votes(rd, store, round_idx)

        if not fired:
            run_state = "converged" if not not_admitted else "no_runnable_interventions"
            break
    else:
        run_state = "max_rounds"

    # ── fusion round ─────────────────────────────────────────────────────
    # The consensus transcript comes from the stream whose FINAL asr
    # evidence is most self-consistent (lowest residual uncertainty mass);
    # region elections break ties. Enhancement can degrade ASR even when it
    # improves speech_presence/quality signals, so transcript fusion must not
    # inherit the speech_presence/quality-weighted election blindly.
    elected_streams = [e["elected"] for e in ctx["elections"].values()]
    fusion_stream = min(
        passes,
        key=lambda s: (
            round(state.uncertainty_mass(s, "asr", theta_low), 9),
            -elected_streams.count(s),
            s,
        ),
    )
    asr_by_model = load_outcomes_dir(run_dir, fusion_stream, "asr")
    align_by_model = load_alignments_matched(run_dir, fusion_stream, asr_by_model)
    # Reserve models pulled in by U2 join the fusion ensemble (cache replay).
    for entry in iterations:
        if entry["rule"] == "U2_reserve_escalation" and entry["status"] == "fired":
            for res in (entry.get("result") or {}).get("reserves_used", []):
                model = res["model"]
                cached = _pick_ok(ctx["cache_index"].get((pass_sigs[fusion_stream], "asr", model), []))
                if cached is not None:
                    asr_by_model.setdefault(model, cached)
                    for (sig, task, _m), aligns in ctx["cache_index"].items():
                        if sig == pass_sigs[fusion_stream] and task == "alignment":
                            m2 = next(
                                (
                                    a
                                    for a in aligns
                                    if (a.get("provenance") or {}).get("parent_asr_cache_key")
                                    == (cached.get("cache_key") or (cached.get("provenance") or {}).get("cache_key"))
                                ),
                                None,
                            )
                            if m2 is not None:
                                align_by_model.setdefault(model, m2)
    purged_spans = [
        (p["bucket"][0], p["bucket"][1], p["source"])
        for entry in iterations
        if entry["rule"] == "P3_hallucination_adjudication" and entry["status"] == "fired"
        for p in (entry.get("result") or {}).get("purged", [])
    ]
    word_streams = collect_word_streams(asr_by_model, align_by_model, purged_spans=purged_spans)
    # Live U1 words (already in file time) join the ensemble on their stream.
    for model, live_words in (ctx.get("live_asr_words", {}).get(fusion_stream) or {}).items():
        if live_words and model not in word_streams:
            word_streams[model] = sorted(live_words, key=lambda w: (w["start"], w["end"]))

    # Speaker attribution: refined speaker clusters (I2) win where available;
    # the vote-majority lookup is the fallback.
    refined = (ctx.get("refined_identity") or {}).get(fusion_stream)
    base_speaker_at = make_speaker_lookup(store, state, fusion_stream)
    if refined is not None:
        from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import cluster_at

        def speaker_at(t: float) -> str | None:
            return cluster_at(refined, t) or base_speaker_at(t)
    else:
        speaker_at = base_speaker_at
    from senselab.audio.workflows.audio_analysis.adaptive.fusion import load_calibrator

    calibrator = load_calibrator(policy.get("calibration_profile"))
    words = fuse_words(
        word_streams,
        policy=policy,
        speaker_at=speaker_at,
        p_voice_at=make_p_voice_lookup(state, fusion_stream),
        calibrator=calibrator,
    )

    # U3 (C8): consensus re-alignment for authoritative word timestamps —
    # guarded live aligner; fallback = weighted member timestamps.
    timestamps_meta: dict[str, Any] = {"timestamps_source": "member_vote"}
    fus_cfg = policy.get("fusion") or {}
    if words and str(fus_cfg.get("consensus_alignment", "auto")) == "auto":
        from senselab.audio.workflows.audio_analysis.adaptive.audio_io import get_stream_wav
        from senselab.audio.workflows.audio_analysis.adaptive.backends import consensus_align

        wav, wav_reason = get_stream_wav(ctx, fusion_stream)
        if wav is None:
            timestamps_meta["reason"] = wav_reason
        else:
            aligned, align_reason = consensus_align(
                wav, words, timeout_s=float(fus_cfg.get("consensus_alignment_timeout_s", 600.0))
            )
            if aligned is not None:
                for w, ts in zip(words, aligned):
                    w["start"], w["end"] = ts["start"], ts["end"]
                words.sort(key=lambda w: (w["start"], w["end"]))
                timestamps_meta = {"timestamps_source": "consensus_alignment_mms_fa"}
            else:
                timestamps_meta["reason"] = align_reason
    last_round = round_summaries[-1]["round"] if round_summaries else 1
    transcript_doc = build_final_outputs(
        out_dir=out_dir,
        words=words,
        store=store,
        state=state,
        stream=fusion_stream,
        policy=policy,
        generated_from_round=last_round,
        refined_identity=refined,
        calibrated=calibrator is not None,
        timestamps_meta=timestamps_meta,
        language=policy.get("language"),
    )

    # T032: LS final tracks + resolved-disagreements index (best-effort, additive).
    try:
        import json as _json

        from senselab.audio.workflows.audio_analysis.adaptive.ls_final import build_final_ls_bundle

        diarization_doc = _json.loads((out_dir / "final" / "diarization.json").read_text())
        build_final_ls_bundle(
            out_dir=out_dir,
            run_dir=run_dir,
            transcript=transcript_doc,
            diarization=diarization_doc,
            speech_presence_rows=state.axis_rows(fusion_stream, "speech_presence"),
            fusion_stream=fusion_stream,
            iterations=iterations,
        )
    except Exception as exc:  # noqa: BLE001 — sidecar must never fail the run
        print(f"warn: LS final bundle failed: {exc!r}")
    provenance = {
        "policy_hash": policy.get("policy_hash"),
        "aggregator": aggregator,
        "run_dir": str(run_dir),
        "fusion_stream": fusion_stream,
        "audio_backend": ctx.get("audio_backend") or None,  # loader per stream (T048 — never silent)
        "elapsed_s": round(time.time() - t0, 3),
    }
    report = build_convergence_report(
        state=state,
        passes=passes,
        policy=policy,
        rounds=round_summaries,
        ledger=ledger,
        iterations=iterations,
        run_state=run_state,
        provenance=provenance,
        round_states=round_states,
    )
    final = final_dir(out_dir)
    # Belief artifacts (posterior, speech_presence, convergence) are level 2; the deliverables
    # (transcript, diarization, timeline, summary) stay in final/. Different questions:
    # "what do we believe" is per bucket and per round, "what do we hand over" is one answer.
    belief = belief_dir(out_dir)
    final.mkdir(parents=True, exist_ok=True)
    belief.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)
    (belief / "convergence.json").write_text(json.dumps(report, indent=2, default=str))
    (belief / "iterations.json").write_text(
        json.dumps({"policy_hash": policy.get("policy_hash"), "entries": iterations}, indent=2, default=str)
    )
    # Summary figure. This lived only in scripts/adaptive_loop.py, so the
    # in-process path (T040) produced every artifact except the one a human
    # actually looks at. Best-effort: a plotting failure must not fail the loop.
    timeline_path: str | None = None
    try:
        from senselab.audio.workflows.audio_analysis.adaptive.plot import build_adaptive_timeline

        fig = build_adaptive_timeline(out_dir, title=run_dir.name)
        timeline_path = str(fig) if fig is not None else None
    except Exception as exc:  # noqa: BLE001 — sidecar
        print(f"warn: adaptive timeline plot failed: {exc!r}", file=sys.stderr)

    return {
        "run_state": run_state,
        # Why the loop stopped, after non-convergence detection has had its say. A caller reading
        # only ``run_state`` would see "converged" for a run that stopped because nothing more
        # would fire while its state was still trading places.
        "termination_reason": report["termination_reason"],
        "converged": report["converged"],
        "policy_hash": policy.get("policy_hash"),
        "timeline": timeline_path,
        "parity_check": parity,
        "rounds": len(round_summaries) + 1,
        "n_interventions_fired": sum(1 for e in iterations if e["status"] == "fired"),
        "n_words_fused": len(words),
        "fusion_stream": fusion_stream,
        "word_streams": word_streams,
        "out_dir": str(out_dir),
        "report": report,
    }


# ── helpers ──────────────────────────────────────────────────────────────


def _resolve_input_audio(recorded: str | None, run_dir: Path) -> str | None:
    """Resolve the run's input audio path, re-rooting when the run came from another machine.

    Tries the recorded absolute path first, then the last one/two path components
    relative to the repo root inferred from ``run_dir`` (…/artifacts/e2e_runs/<run>
    → repo). Returns None when nothing exists — audio-dependent rules then guard.
    """
    if not recorded:
        return None
    p = Path(recorded)
    if p.exists():
        return str(p)
    candidates = []
    for base in (run_dir.parent, run_dir.parent.parent, run_dir.parent.parent.parent):
        candidates.extend([base / Path(*p.parts[-2:]), base / p.name])
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _aggregator_from_run(run_dir: Path) -> str | None:
    dis = run_dir / "disagreements.json"
    if dis.exists():
        try:
            return (json.loads(dis.read_text()).get("config") or {}).get("aggregator")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _grid_from_rows(rows: list[dict[str, Any]]) -> tuple[float, float]:
    if not rows:
        return (1.0, 0.5)
    win = float(rows[0]["end"]) - float(rows[0]["start"])
    hop = (float(rows[1]["start"]) - float(rows[0]["start"])) if len(rows) > 1 else win
    return (round(win, 6), round(hop, 6) or win)


def _bucket_values(state: BeliefState) -> dict[tuple[str, str, tuple[float, float]], float | None]:
    """Per-bucket aggregated uncertainty snapshot (delta baselines use the touched set)."""
    out: dict[tuple[str, str, tuple[float, float]], float | None] = {}
    for (stream, axis), rows in sorted(state.rows.items()):
        for row in rows:
            out[(stream, axis, (round(row["start"], 6), round(row["end"], 6)))] = row.get("within_pass_uncertainty")
    return out


def _mean_over(state: BeliefState, stream: str, axis: str, buckets: set | None) -> float | None:
    vals = []
    for row in state.axis_rows(stream, axis):
        if buckets is not None and (round(row["start"], 6), round(row["end"], 6)) not in buckets:
            continue
        u = row.get("within_pass_uncertainty")
        if u is not None:
            vals.append(float(u))
    return sum(vals) / len(vals) if vals else None


def _iteration_entry(cand: PlannedIntervention, round_idx: int) -> dict[str, Any]:
    return {
        "intervention_id": cand.get("intervention_id")
        or f"{round_idx}_{cand['rule']}_{cand.get('region_id') or 'global'}",
        "round": round_idx,
        "rule": cand["rule"],
        "region_id": cand.get("region_id"),
        "axis": cand.get("axis"),
        "cost_class": cand.get("cost_class"),
        "priority": cand.get("priority"),
        "trigger": cand.get("trigger"),
        "status": cand.get("status"),
        "error": cand.get("error"),
    }


def _write_round_belief(round_dir: Path, state: BeliefState, passes: list[str]) -> None:
    import pandas as pd

    round_belief = round_dir / "belief"
    round_belief.mkdir(parents=True, exist_ok=True)
    for axis in AXES:
        rows = []
        for stream in passes:
            for r in state.axis_rows(stream, axis):
                rows.append(
                    {
                        "stream": stream,
                        "start": r["start"],
                        "end": r["end"],
                        "within_pass_uncertainty": r.get("within_pass_uncertainty"),
                        "p_voice": r.get("p_voice"),
                        "epistemic": r.get("epistemic"),
                        "aleatoric_floor": r.get("aleatoric_floor"),
                        "status": r.get("status"),
                        "irreducible_reason": r.get("irreducible_reason"),
                        "round": r.get("round"),
                        "n_sources": len(r.get("contributing_sources") or []),
                    }
                )
        if rows:
            pd.DataFrame(rows).to_parquet(round_belief / f"{axis}.parquet", index=False)


def _write_round_votes(round_dir: Path, store: VoteStore, round_idx: int) -> None:
    import pandas as pd

    votes = store.votes_added_in_round(round_idx)
    if votes:
        pd.DataFrame([v.to_record() for v in votes]).to_parquet(round_dir / "votes_added.parquet", index=False)


def _json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(result, default=str))
