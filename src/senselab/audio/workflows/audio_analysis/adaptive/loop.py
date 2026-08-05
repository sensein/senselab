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
from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.adaptive.belief import AXES, BeliefState, VoteStore
from senselab.audio.workflows.audio_analysis.adaptive.convergence import (
    apply_convergence_marks,
    build_convergence_report,
    round_summary,
)
from senselab.audio.workflows.audio_analysis.adaptive.fusion import (
    attenuation_columns,
    build_final_outputs,
    collect_word_streams,
    extract_final_estimates,
    fuse_words,
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
from senselab.audio.workflows.audio_analysis.estimates import estimate_frame
from senselab.audio.workflows.audio_analysis.io import merge_json
from senselab.audio.workflows.audio_analysis.layout import (
    belief_dir,
    derivatives_dir,
    estimates_dir,
    evidence_dir,
    final_dir,
    last_round,
    round_dir,
)
from senselab.audio.workflows.audio_analysis.perturbations import read_measurements, read_register


def run_adaptive_loop(
    run_dir: Path,
    *,
    cache_dir: Path | None = None,
    policy_path: Path | None = None,
    out_dir: Path | None = None,
    max_rounds: int = 3,
    aggregator: str | None = None,
    harvests: dict[str, Any] | None = None,
    unharvested_votes: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
    summary: dict[str, Any] | None = None,
    policy_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the adaptive loop over an analyze_audio run.

    Two ingest paths, same loop:

    - **artifact-driven** (default): reads ``L1/perturbations.json`` and the linked votes from
      ``L2/round0/votes/<axis>.parquet``. This is what ``scripts/adaptive_loop.py``
      does over a finished run.
    - **in-process** (T040): the caller passes the ``PassHarvest`` objects it just
      produced via ``harvests`` (and usually ``summary``), skipping the parquet
      round-trip entirely.

    Both paths now see the same evidence and both run :meth:`VoteStore.replay_check`. It used
    to be a *parity* check against ``within_pass_uncertainty`` on the L1 parquet, which the
    in-process path could not run at all — and which compared two different implementations of
    the fold, so a mismatch could not distinguish a missing input from a disagreement between
    them. The replay proves the property that actually matters: every value is re-derivable from
    the persisted votes plus the recorded decisions, so the store need not persist estimates.

    Args:
        run_dir: The run directory. Still required for reading policy-adjacent
            artifacts and for output pathing, even on the in-process path.
        cache_dir: Cache directory for intervention re-runs.
        policy_path: Policy YAML; ``None`` uses the packaged default.
        out_dir: Where ``rounds/`` and ``final/`` are written; defaults to ``run_dir``.
        max_rounds: Total rounds including baseline. ``1`` = baseline only.
        aggregator: Sub-signal aggregator; inferred from the run when ``None``.
        harvests: Pass label → ``PassHarvest`` for the in-process path.
        unharvested_votes: ``{axis → {perturbation → buckets}}`` for an active axis with no vote
            harvest. **No active axis needs it today**: all four declare a
            ``axes.HarvestSource``, ``background_mask`` included, so the mask arrives with the rest
            of the harvest — it used to be handed in here as one vote per mask *region*, which gave
            the loop a single bucket to fold where L2 had 1070. Kept because
            :meth:`VoteStore.from_harvests` raises for an axis it can read nowhere, and this is the
            remedy that error names — activating the declared-but-unbuilt ``task`` axis
            (``harvested=False``) would use it. The artifact path reads the same evidence out of
            ``L2/round/0/derivatives/votes/`` and needs nothing here.
        policy_overrides: In-memory policy overrides (CLI flags), merged last.
        summary: Pre-loaded ``{"input_audio": ..., "passes": {...}}`` index; read from
            ``L1/perturbations.json`` when ``None``.

    Returns:
        The loop's decision log.

    Raises:
        ValueError: If no completed passes can be determined.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir
    policy = load_policy(policy_path, policy_overrides)

    register = read_register(run_dir)
    if summary is None:
        # ``L1/perturbations.json``, not ``final/summary.json``. A deliverable that a later stage
        # reads to rebuild state is an intermediate wearing the wrong name; the index of what was
        # measured under which perturbation belongs beside the declaration of what they are.
        summary = {
            "input_audio": _register_source_audio(run_dir),
            "passes": read_measurements(run_dir),
        }
    passes = [pl for pl, ps in (summary.get("passes") or {}).items() if isinstance(ps, dict) and "duration_s" in ps]
    if not passes:
        raise ValueError(f"no completed perturbations in {run_dir}/L1/perturbations.json")
    duration_s = float(summary["passes"][passes[0]]["duration_s"])
    pass_sigs = {pl: str(summary["passes"][pl].get("audio_signature") or "") for pl in passes}
    if aggregator is None:
        aggregator = _aggregator_from_run(run_dir) or "min"

    # ── the baseline round: ingest + replay proof (values are re-derivable) ────────
    t0 = time.time()
    parity: dict[str, Any]
    if harvests is not None:
        store = VoteStore.from_harvests(
            {pl: h for pl, h in harvests.items() if pl in passes}, unharvested=unharvested_votes
        )
    else:
        store = VoteStore.from_run_dir(run_dir, passes)
    parity = store.replay_check(aggregator=aggregator)
    fused_parity = store.fused_parity(_fused_axes_from_run(run_dir), aggregator=aggregator)

    # The baseline is not a round of the loop's own — it *is* the round fusion already wrote, and
    # numbering it separately is what produced two trees whose "round 1" meant different things.
    # So the loop adopts fusion's index, writes only what it can add there (the replay and parity
    # proofs), and does not rewrite estimates it has just proven it agrees with.
    #
    # Resolved before the ingest fold rather than after, because the fold has to stamp it: the
    # state's rows record which round last recomputed them, and a baseline that names itself ``1``
    # while the tree calls it ``2`` is a row disagreeing with its own directory from the first
    # write onwards.
    baseline = _baseline_round(out_dir)
    state = BeliefState.from_store(store, aggregator=aggregator, round_index=baseline)
    asr_grid = _grid_from_rows(state.axis_rows("asr"))
    theta_low = float(policy["thresholds"]["theta_low"])

    (round_dir(out_dir, baseline)).mkdir(parents=True, exist_ok=True)
    _write_round_summary(
        out_dir,
        baseline,
        {
            "round": baseline,
            "ingested_from": str(run_dir),
            "replay_check": parity,
            "fused_parity": fused_parity,
            "aggregator": aggregator,
            "uncertainty_mass": {str(a): round(state.uncertainty_mass(a, theta_low), 6) for a in AXES},
        },
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
        # The declared set, so anything that has to *regenerate* a perturbation's audio
        # dispatches on its transform rather than on how its name is spelled.
        "perturbations": [p.to_json() for p in register],
        "asr_model_ids": {pl: set(load_outcomes_dir(run_dir, pl, "asr").keys()) for pl in passes},
    }

    ledger = BudgetLedger(policy)
    iterations: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    round_states: list[dict[str, Any]] = []
    touch_counts: dict[tuple[str, tuple[float, float]], int] = {}
    run_state = "max_rounds"

    # ── rounds 2..K ──────────────────────────────────────────────────────
    for round_idx in range(baseline + 1, baseline + max_rounds):
        ctx["round_idx"] = round_idx
        mass_before: dict[str, float] = {a: round(state.uncertainty_mass(a, theta_low), 6) for a in AXES}
        regions: list[Region] = []
        for axis in AXES:
            regions.extend(
                propose_regions(
                    state.axis_rows(axis),
                    axis=axis,
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
                touched: dict[AxisName, set] = result.pop("touched", {})
                delta = {}
                # Sorted: these iterations accumulate into output, so a fixed order makes
                # byte-reproducibility structural rather than a property of dict insertion
                # order that a refactor could quietly break (FR-011f).
                for axis, buckets in sorted(touched.items()):
                    state.update_buckets(store, axis, buckets, round_idx)
                    for bk in buckets:
                        key = (axis, bk)
                        touch_counts[key] = touch_counts.get(key, 0) + 1
                    # Before/after means over the SAME (touched) bucket set.
                    befores = [v for bk in buckets if (v := before_vals.get((axis, bk))) is not None]
                    after = _mean_over(state, axis, buckets)
                    if befores and after is not None:
                        before = sum(befores) / len(befores)
                        delta[axis] = {
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
        apply_convergence_marks(state, policy=policy, touch_counts=touch_counts, budget_left=budget_left)
        rs = round_summary(
            round_idx=round_idx,
            state=state,
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
        derivatives = derivatives_dir(out_dir, round_idx)
        derivatives.mkdir(parents=True, exist_ok=True)
        _write_round_belief(out_dir, round_idx, state)
        (derivatives / "regions.json").write_text(json.dumps(regions, indent=2, default=str))
        _write_round_summary(out_dir, round_idx, rs)
        _write_round_votes(derivatives, store, round_idx)
        # The round's own view of itself. Every round owes one — a trajectory a reader can only
        # see for the rounds fusion wrote is not a trajectory — and it is the *same* figure
        # ``fuse`` draws for its rounds, from the same declaration, so the two halves of a run's
        # rounds are comparable pictures rather than two conventions.
        _draw_round_timeline(out_dir, round_idx, state, duration_s=duration_s, title=run_dir.name)

        if not fired:
            run_state = "converged" if not not_admitted else "no_runnable_interventions"
            break
    else:
        run_state = "max_rounds"

    # ── fusion round ─────────────────────────────────────────────────────
    # The consensus transcript comes from the pass whose ASR *signals* are most self-consistent;
    # region elections break ties. Enhancement can degrade ASR even when it improves
    # speech_presence/quality signals, so transcript fusion must not inherit the
    # speech_presence/quality-weighted election blindly.
    #
    # Measured per (signal, pass) from the votes, not by comparing one pass's *axis* against
    # another's. An axis is a fold across passes, so it has no per-pass value to compare — asking
    # for one was the same category error the belief store had, surviving in the one place that
    # genuinely does have to choose a pass. What a pass owns is its signals' readings, and those
    # are exactly what a transcript is built from.
    per_pass_asr = {s: _asr_signal_doubt(store, s) for s in passes}
    elected_streams = [e["elected"] for e in ctx["elections"].values()]
    fusion_stream = min(
        passes,
        key=lambda s: (
            round(doubt, 9) if (doubt := per_pass_asr[s]) is not None else float("inf"),
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
    word_streams = collect_word_streams(asr_by_model, align_by_model)
    # Live U1 words (already in file time) join the ensemble on their stream.
    for model, live_words in (ctx.get("live_asr_words", {}).get(fusion_stream) or {}).items():
        if live_words and model not in word_streams:
            word_streams[model] = sorted(live_words, key=lambda w: (w["start"], w["end"]))

    # Per-word corroboration, measured against presence voters that are independent of ASR.
    # Stamped after the U2 cache-replay and U1 live merges so late-arriving streams are measured
    # on the same footing, and stamped for *every* word whether or not any intervention fired —
    # otherwise budget admission decides what survives into the transcript.
    from senselab.audio.workflows.audio_analysis.adaptive.corroboration import (
        apply_corroboration,
        independent_presence_pool,
        make_corroboration_lookup,
    )

    corr_cfg = (policy.get("fusion") or {}).get("corroboration") or {}
    pool, pool_rejected = independent_presence_pool(store, fusion_stream)
    if not pool:
        print(
            f"warn: no independent presence voter survived screening on stream {fusion_stream!r} — "
            "word corroboration is inert for this run (every word unmeasured)",
            file=sys.stderr,
        )
    word_streams, corr_prov = apply_corroboration(
        word_streams,
        make_corroboration_lookup(store, fusion_stream, pool=pool),
        exponent=float(corr_cfg["exponent"]),
        min_corroboration=float(corr_cfg["min_corroboration"]),
        pool=pool,
        rejected=pool_rejected,
    )

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
        calibrator=calibrator,
    )

    # U3 (C8): consensus re-alignment for authoritative word timestamps —
    # guarded live aligner; fallback = weighted member timestamps.
    timestamps_meta: dict[str, Any] = {"timestamps_source": "member_vote"}
    fus_cfg = policy.get("fusion") or {}
    if words and str(fus_cfg.get("consensus_alignment", "auto")) == "auto":
        from senselab.audio.workflows.audio_analysis.adaptive.audio_io import get_stream_wav
        from senselab.audio.workflows.audio_analysis.adaptive.backends import consensus_align
        from senselab.audio.workflows.audio_analysis.adaptive.fusion import rollup_segments

        wav, wav_reason = get_stream_wav(ctx, fusion_stream)
        if wav is None:
            timestamps_meta["reason"] = wav_reason
        else:
            # Only the words the rollup retains are handed to the aligner. Forced alignment places
            # every token it is given somewhere in the audio, so feeding it text the audio may not
            # contain lets its path optimisation drag the *neighbouring* words' timestamps with it.
            # Withheld words keep their member-vote timestamps.
            segment_min = float((fus_cfg.get("corroboration") or {})["segment_min_corroboration"])
            _segments, withheld = rollup_segments(words, min_corroboration=segment_min)
            withheld_set = set(withheld)
            retained = [i for i in range(len(words)) if i not in withheld_set]
            backend = str(fus_cfg.get("consensus_alignment_backend", "qwen"))
            aligned, align_reason = consensus_align(
                wav,
                [words[i] for i in retained],
                timeout_s=float(fus_cfg.get("consensus_alignment_timeout_s", 600.0)),
                backend=backend,
            )
            if aligned is not None:
                for index, ts in zip(retained, aligned):
                    words[index]["start"], words[index]["end"] = ts["start"], ts["end"]
                words.sort(key=lambda w: (w["start"], w["end"]))
                timestamps_meta = {
                    # Names the backend that actually ran rather than the one that used to be
                    # hard-coded here: a reader comparing published timings against the per-edge
                    # boundary agreement needs to know whether the published value came from inside
                    # the set of members whose spread is being reported, or from outside it.
                    "timestamps_source": f"consensus_alignment_{backend}",
                    "consensus_alignment_backend": backend,
                    "n_words_aligned": len(retained),
                    "n_words_on_member_timestamps": len(withheld_set),
                }
            else:
                timestamps_meta["reason"] = align_reason
    fused_from_round = round_summaries[-1]["round"] if round_summaries else 1
    transcript_doc, diarization_doc = build_final_outputs(
        out_dir=out_dir,
        words=words,
        store=store,
        state=state,
        stream=fusion_stream,
        policy=policy,
        generated_from_round=fused_from_round,
        corroboration_provenance=corr_prov,
        refined_identity=refined,
        calibrated=calibrator is not None,
        timestamps_meta=timestamps_meta,
        language=policy.get("language"),
    )

    # T032: LS final tracks + resolved-disagreements index (best-effort, additive).
    ls_report: dict[str, Any] = {}
    try:
        from senselab.audio.workflows.audio_analysis.adaptive.ls_final import build_final_ls_bundle

        ls_report = build_final_ls_bundle(
            out_dir=out_dir,
            run_dir=run_dir,
            transcript=transcript_doc,
            diarization=diarization_doc,
            speech_presence_rows=state.axis_rows("speech_presence"),
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
        policy=policy,
        rounds=round_summaries,
        ledger=ledger,
        iterations=iterations,
        run_state=run_state,
        provenance=provenance,
        round_states=round_states,
    )
    final = final_dir(out_dir)
    final.mkdir(parents=True, exist_ok=True)
    # final/decisions.json — how the run got to its answer: the trajectory, the reversals, the
    # stopping reason and every intervention entry. This was ``L2/convergence.json`` plus
    # ``L2/iterations.json``, two per-run documents flattened to the belief root, so ``final/``
    # carried no account of the run at all and the evaluator reached into ``L2/`` to reconstruct
    # one. Replaced outright rather than mirrored: one quantity in two places is one quantity that
    # can disagree with itself, and each round's own slice is already in its ``summary.json``.
    (final / "decisions.json").write_text(
        json.dumps(
            {"policy_hash": policy.get("policy_hash"), "convergence": report, "interventions": iterations},
            indent=2,
            default=str,
        )
    )
    # The deliverable axes: the last round's estimates, extracted verbatim.
    extract_final_estimates(out_dir, last_round(out_dir) or 0)
    # Summary figure. This lived only in scripts/adaptive_loop.py, so the
    # in-process path (T040) produced every artifact except the one a human
    # actually looks at. Best-effort: a plotting failure must not fail the loop.
    timeline_path: str | None = None
    try:
        from senselab.audio.workflows.audio_analysis.adaptive.plot import build_adaptive_timeline

        fig = build_adaptive_timeline(out_dir, title=run_dir.name, transcript=transcript_doc)
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
        # What the LS stage produced, so the driver can report where the bundle is without
        # probing final/ for it. A stage branching on a deliverable is treating it as state.
        "labelstudio": ls_report,
        "replay_check": parity,
        "fused_parity": fused_parity,
        "rounds": len(round_summaries) + 1,
        "n_interventions_fired": sum(1 for e in iterations if e["status"] == "fired"),
        "n_words_fused": len(words),
        "fusion_stream": fusion_stream,
        "word_streams": word_streams,
        "out_dir": str(out_dir),
        "report": report,
        # Handed back rather than left for a caller to read out of ``final/``: a driver that wants
        # to re-render the timeline with a ground-truth overlay needs the transcript, and the only
        # other way to get it is to open the deliverable this loop just wrote.
        "transcript": transcript_doc,
    }


# ── helpers ──────────────────────────────────────────────────────────────


def _register_source_audio(run_dir: Path) -> str | None:
    """The source recording recorded in ``L1/perturbations.json``, or ``None``."""
    try:
        payload = json.loads((evidence_dir(run_dir) / "perturbations.json").read_text())
    except (OSError, ValueError):
        return None
    source = payload.get("source_audio") if isinstance(payload, dict) else None
    return str(source) if source else None


def _resolve_input_audio(recorded: str | None, run_dir: Path) -> str | None:
    """Resolve the run's input audio path, re-rooting when the run came from another machine.

    Tries the recorded absolute path first, then the last one/two path components
    relative to the repo root inferred from ``run_dir`` (…/artifacts/analyze_audio/<run>
    → repo). Returns None when nothing exists — audio-dependent rules then guard.

    The root is inferred from the path *shape*, which is fragile: it walks a fixed number of
    parents rather than looking for a marker. It happens to work for the default output layout
    and would not for an arbitrary ``--out-dir``.
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
    # ``L2/disagreements.json``. The path here was the pre-L1/L2 flat one, so it never resolved
    # and every standalone run silently fell back to the "min" default — including runs whose
    # analyze_audio pass had been given a different aggregator on the command line.
    dis = belief_dir(run_dir) / "disagreements.json"
    if dis.exists():
        try:
            return (json.loads(dis.read_text()).get("config") or {}).get("aggregator")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _fused_axes_from_run(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """L2's round-0 axes, the oracle for :meth:`VoteStore.fused_parity`.

    Round 0 specifically, not the last round. The store ingests the round-0 votes and folds them
    once; L2's later rounds condition each axis on the *others*, which is evidence the store does
    not have, so comparing against them would skip every bucket as coupled and report a vacuous
    zero — the exact failure signature these checks exist to remove.

    Missing files yield an empty mapping and the report then says ``not_in_l2`` for every bucket,
    which is a finding rather than a pass.
    """
    import pandas as pd

    directory = estimates_dir(run_dir, 0)
    if not directory.is_dir():
        return {}
    return {
        str(path.stem): pd.read_parquet(path).to_dict("records")
        for path in sorted(directory.glob("*.parquet"))
        if path.stem in AXES
    }


def _asr_signal_doubt(store: VoteStore, stream: str) -> float | None:
    """Mean per-signal ASR uncertainty on one pass, over the buckets that pass reported.

    A per-*pass* quantity computed from per-pass votes, which is legitimate: it is what the pass's
    own transcribers said, never a reading of the axis. Returns ``None`` when no ASR signal spoke
    on this pass, which must not sort as "most confident".
    """
    from senselab.audio.workflows.audio_analysis.fuse import per_signal_uncertainty

    values: list[float] = []
    for bucket in store.vote_buckets(stream, "asr"):
        readings = per_signal_uncertainty({"votes": store.active_votes(stream, "asr", bucket)})
        values.extend(readings.values())
    return sum(values) / len(values) if values else None


def _grid_from_rows(rows: list[dict[str, Any]]) -> tuple[float, float]:
    if not rows:
        return (1.0, 0.5)
    win = float(rows[0]["end"]) - float(rows[0]["start"])
    hop = (float(rows[1]["start"]) - float(rows[0]["start"])) if len(rows) > 1 else win
    return (round(win, 6), round(hop, 6) or win)


def _bucket_values(state: BeliefState) -> dict[tuple[str, tuple[float, float]], float | None]:
    """Per-bucket axis snapshot (delta baselines use the touched set)."""
    out: dict[tuple[str, tuple[float, float]], float | None] = {}
    for axis, rows in sorted(state.rows.items()):
        for row in rows:
            out[(axis, (round(row["start"], 6), round(row["end"], 6)))] = row.get("uncertainty")
    return out


def _mean_over(state: BeliefState, axis: str, buckets: set | None) -> float | None:
    vals = []
    for row in state.axis_rows(axis):
        if buckets is not None and (round(row["start"], 6), round(row["end"], 6)) not in buckets:
            continue
        u = row.get("uncertainty")
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


def _baseline_round(out_dir: Path) -> int:
    """The round the loop ingests: the last one fusion wrote, or 0 when it wrote none.

    Adopted rather than assigned. The adaptive loop used to call its ingest "round 1" while the
    fusion loop called the same iteration "round 0", so the two trees disagreed about what any
    given number meant — and under one tree the collision is a round reading its own output.
    """
    return last_round(out_dir) or 0


def _write_round_summary(out_dir: Path, round_index: int, payload: dict[str, Any]) -> None:
    """Merge this loop's block into ``L2/round/<n>/summary.json``.

    Merged rather than written, because the round the loop adopts as its baseline is a round
    ``fuse`` also has an account of. Two facts about one round belong in one document; a second
    write erased the first, which is how the baseline round came to carry the loop's replay proof
    and none of fusion's fold log.
    """
    merge_json(round_dir(out_dir, round_index) / "summary.json", {"adaptive": payload})


def _write_round_belief(out_dir: Path, round_index: int, state: BeliefState) -> None:
    """Write one belief row per (axis, bucket) — the axis, which is already the fold.

    Nothing is collapsed here any more. The state holds one row per (axis, bucket) because that is
    what an axis is, so this writer transcribes rather than decides. It used to receive one row per
    (pass, axis, bucket) and elect the most doubtful pass, recording the winner as
    ``elected_stream`` — which is a per-pass axis with the index moved into the value, and left the
    loop itself still reasoning over two answers per bucket while the file showed one.

    ``contributing_passes`` survives and ``elected_stream`` does not, because they are different
    claims: the first says which passes fed the fold, the second says which pass's reading was
    taken *instead* of folding.

    The columns come from :data:`~senselab.audio.workflows.audio_analysis.estimates.ESTIMATE_COLUMNS`,
    the same declaration ``fuse`` writes through, because these rounds and fusion's rounds are one
    trajectory in one directory under one declared artifact.
    """
    round_belief = estimates_dir(out_dir, round_index)
    round_belief.mkdir(parents=True, exist_ok=True)
    for axis in AXES:
        rows = [
            {
                "start": r["start"],
                "end": r["end"],
                "uncertainty": r.get("uncertainty"),
                "epistemic_uncertainty": r.get("epistemic_uncertainty"),
                "confidence": r.get("confidence"),
                "variability": r.get("variability"),
                "triage_score": r.get("triage_score"),
                "p_voice": r.get("p_voice"),
                "aleatoric_floor": r.get("aleatoric_floor"),
                "aleatoric_floor_terms": (r.get("aleatoric_floor_policy") or {}).get("terms") or [],
                "status": r.get("status"),
                "irreducible_reason": r.get("irreducible_reason"),
                # The calibrated presence probability and the overlap posterior, on the round that
                # believes them. ``final/`` extracts a round, so a deliverable column no round
                # carries is a number computed at the wrong stage — and both of these were.
                "speech_presence_confidence": r.get("speech_presence_confidence", r.get("p_voice")),
                "overlap_posterior": r.get("overlap_posterior", (r.get("meta") or {}).get("overlap_posterior")),
                # Not ``round`` — that one the declaration stamps from the directory below. This
                # is the round that last recomputed the row, which for an untouched bucket is an
                # earlier one, and writing it as ``round`` is what made a round directory hold
                # rows claiming three different rounds.
                "last_refolded_round": r.get("last_refolded_round"),
                "n_sources": len(r.get("contributing_sources") or []),
                "contributing_signals": r.get("contributing_signals") or [],
                "contributing_passes": r.get("contributing_passes") or [],
                **attenuation_columns(r),
            }
            for r in sorted(state.axis_rows(axis), key=lambda x: (float(x["start"]), float(x["end"])))
        ]
        # Written even when the axis has no rows. "This round believes nothing about the mask"
        # and "this round was not asked about the mask" are different facts, and skipping the
        # file makes them the same one — which is how the fourth axis's estimates stopped at
        # round 2 while the run reported it settled. An empty table with the declared columns
        # says the first; an absent file says neither and is read as the second.
        estimate_frame(axis, rows, round_index=round_index).to_parquet(round_belief / f"{axis}.parquet", index=False)


def _draw_round_timeline(out_dir: Path, round_index: int, state: BeliefState, *, duration_s: float, title: str) -> None:
    """Draw ``L2/round/<n>/timeline.png``. Best-effort: a plot must not fail a round."""
    from senselab.audio.workflows.audio_analysis.l2_plot import build_round_timeline

    try:
        build_round_timeline(
            out_dir,
            round_index=round_index,
            axis_rows={axis: state.axis_rows(axis) for axis in AXES},
            duration_s=duration_s,
            title=f"{title} — L2 round {round_index}",
        )
    except Exception as exc:  # noqa: BLE001 — sidecar
        print(f"warn: round {round_index} timeline plot failed: {exc!r}", file=sys.stderr)


def _write_round_votes(derivatives: Path, store: VoteStore, round_idx: int) -> None:
    """The votes this round *added*, beside the estimates they moved."""
    import pandas as pd

    votes = store.votes_added_in_round(round_idx)
    if votes:
        pd.DataFrame([v.to_record() for v in votes]).to_parquet(derivatives / "votes_added.parquet", index=False)


def _json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(result, default=str))
