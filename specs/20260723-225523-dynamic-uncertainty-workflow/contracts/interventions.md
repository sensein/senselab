# Contract: Intervention catalog (v1)

Each rule declares: **trigger** (predicate over belief state), **guards**, **action**, **evidence
added** (votes), **cost class**, **expected_gain heuristic**. All parameters live in the policy file.
Rule ids are stable API (they appear in iterations.json and tests).

## S1_stream_election — per-region stream choice (C5)

- **Trigger**: region proposed on any axis; both streams have round-1 evidence; no election yet or
  covered evidence changed.
- **Action**: score streams per policy weights over region buckets: presence confidence, quality
  (1 − aggregated degradation), utterance agreement (1 − mean pairwise phoneme distance). Apply
  enhancement-artifact guard: enhanced cannot win if raw-side PPG non-silent fraction < guard_min_raw_ppg
  while enhanced-side claims speech, or squim STOI drops raw→enhanced.
- **Evidence**: `elected_stream` on region + covered BeliefRows; StreamElection record.
- **Cost**: light. **Gain**: prerequisite multiplier for U1/U2 (elected implicitly by those rules if
  not yet run).

## P2_fine_posteriors — fine-grid presence re-analysis

- **Trigger**: presence region where coarse voters dominate (share of `coarse=true` active votes ≥ 0.5)
  or `presence_uncertainty` driven by frame_instability.
- **Action**: re-run segmentation-3.0 + Brouhaha posteriors on the crop at fine hop (policy:
  `fine_hop_s: 0.01`); per-class posteriors retained (FR-016).
- **Evidence**: replacement frame-posterior votes (scope region), overlap_posterior on covered rows.
- **Cost**: medium. **Gain**: `presence epistemic × mass`.

## P3_hallucination_adjudication — ASR text where VAD says silence (C10)

- **Trigger**: buckets with active ASR `speaks=true` votes while frame-posterior p_voice < 0.2.
- **Guards**: none (light; runs on existing evidence).
- **Action**: score indicators — whisper `no_speech_prob ≥ 0.5` with tokens (flag exists,
  `presence.py:330`), `1 − exp(avg_logprob)` high, alignment CTC score low, PPG non-silent fraction
  low, `src_machine + src_environment + music mass` high. Verdict `hallucination` if ≥ 2 independent
  indicators (different families); verdict `missed_speech` if PPG active ∧ CTC high ∧ ≥ 2 ASR families
  agree on text.
- **Evidence**: hallucination → affected utterance/presence votes `status=purged_hallucination` (C10);
  missed_speech → add `adjudicator/missed_speech` presence vote `{speaks: true, weight: 0.5}` (C9).
- **Cost**: light. **Gain**: `n_affected_buckets × mean uncertainty`.

## U1_region_reasr — targeted re-transcription

- **Trigger**: utterance region, `epistemic ≥ theta_low` (disagreement-driven, not floor-driven).
- **Guards**: crop ≥ 1.0 s; elected stream audio available; region has ≥ 1 ASR family with active votes.
- **Action**: crop per contracts/region-reprocessing.md; run the round-1 ASR set on the elected stream
  crop (cached, FR-014); auto-align text-only outputs (existing aligner path).
- **Evidence**: region-scoped utterance+presence votes per model (shadow same model+stream, D5).
- **Cost**: medium (per model forward on crop; counted once per rule firing). **Gain**:
  `epistemic × uncertainty_mass × quality_gain_factor` (higher when elected stream ≠ round-1 dominant
  stream — new information likely).

## U2_reserve_escalation — add a dissenting model

- **Trigger**: utterance region still open after a U1 firing (delta < ε or disagreement persists).
- **Guards**: reserve pool non-empty; heavy budget available; reserve model's family not already
  majority in region.
- **Action**: run one reserve ASR model (policy `reserve_asr_models`, in order) on the elected-stream
  crop; align if text-only.
- **Evidence**: new-model region-scoped votes (coexist).
- **Cost**: heavy. **Gain**: `epistemic × mass × (1 − family_overlap)`.

## U3_consensus_realignment — authoritative word timestamps (C8)

- **Trigger**: fusion round; contested or converged utterance regions with a consensus text.
- **Action**: force-align consensus text over the region (Qwen aligner default, MMS fallback —
  script's existing backends).
- **Evidence**: `final` word timestamps; does not alter uncertainty rows.
- **Cost**: medium (batched once for all regions). **Gain**: fixed (fusion prerequisite).

## I1_boundary_refinement — snap disputed diarization boundaries (C6)

- **Trigger**: identity region containing ≥ 1 diar-model boundary where `__cross_diar_label_disagreement__`
  or change_inconsistency is a top sub-signal.
- **Guards**: ≥ 1 embedding model available; crop ≥ 2 × embedding window.
- **Action**: re-embed crop at fine hop (policy `identity_fine_hop_s: 0.1`); compute calibrated
  adjacent-cosine change-point trajectory (`calibrate_cosine_uncertainty`, `embeddings.py:666`);
  emit change-point evidence.
- **Evidence**: `embedding_changepoint/<model>` identity votes supporting the nearest model boundary
  (or contradicting all, which raises epistemic honestly); refined boundary candidates for fusion.
- **Cost**: medium. **Gain**: `identity epistemic × mass`.

## I4_overlap_detection — explain joint uncertainty (C7)

- **Trigger**: co-located identity + utterance regions (time-IoU ≥ 0.5), or identity region with
  rapid label alternation across models.
- **Guards**: per-class posteriors available (FR-016) — else fires P2 first (dependency declared).
- **Action**: compute overlap posterior = Σ multi-speaker powerset class probabilities over crop.
- **Evidence**: `overlap_posterior` on covered rows → `aleatoric_floor`; if floor explains residual,
  regions close as `irreducible: overlapping_speech`.
- **Cost**: light (reuses P2 output) or medium (posteriors not yet fine-grained).
- **Gain**: `joint mass × co-location`.

## U4_overlap_separation (v2, default off)

- **Trigger**: irreducible `overlapping_speech` utterance region; `--enable-overlap-separation`.
- **Action**: SepFormer separation on crop → per-source embedding match to unified clusters →
  per-source ASR (C11) → speaker-attributed region votes.
- **Cost**: heavy.

## Implementation notes (contract aligned with code, 2026-07-24 — spec T045)

- **U2 cost class is `medium`** (cache replay / one forward per crop), not heavy as originally
  drafted; the "reserve family not already majority" guard was not implemented (the family-weight
  aggregation already discounts intra-family agreement, making the guard redundant in practice).
- **`I2_recluster` is a catalog addition** beyond this contract (tasks.md T024a): change-point +
  diar-boundary segmentation, p_voice-weighted pooling, cross-model co-association consensus; it
  emits `embedding_recluster/consensus` votes and recomputes `__cross_diar_label_disagreement__`.
- **U3 runs as a fusion-stage step**, not a RULES entry — consistent with its trigger ("fusion
  round") but not budget-ledgered; its guard is the SIGALRM timeout + backend availability
  (`fusion.consensus_alignment` policy).
- **`max_region_rounds` is enforced via convergence marks** (per-bucket touch counts, FR-017)
  rather than as a plan-time admission cap; the observable behavior (a region stops being
  re-touched after N interventions without ε improvement) matches the contract's intent.
- **Backend pins**: `u1_backend` (`auto|senselab|pipeline`) and `audio_io_backend`
  (`auto|senselab|dsp`) select the primary/fallback paths explicitly; the backend used is recorded
  in `iterations.json` / `convergence.json → audio_backend` (never silent).

## Shared guards

- Region crops quantized to reporting grid before hashing (cache stability).
- A rule fires at most once per (region, round); `max_region_rounds` caps total firings per region
  across rounds (FR-017).
- Any rule may be disabled in policy; disabled rules never appear in plans (but their triggers are
  still evaluated for `next_actions` reporting).
