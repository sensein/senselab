# Sweep B — Computation candidates

Audited the computation layer of `src/senselab/audio/workflows/audio_analysis` (61 files that
import no senselab task) in four parallel batches, each reading its files in full — every
function, every numeric literal, every docstring — and cross-checking against
`specs/20260815-215106-analyze-audio-audit/candidates/sweep-a-prose.md` where a file's docstring
was already flagged `stale-or-false` there.

Batch assignments:
- Batch 1 (grid/fuse/core infra, 17 files) → B-1..B-4
- Batch 2 (speaker/identity/stats, 14 files) → B-5..B-10
- Batch 3 (background-scene/plot/misc, 17 files) → B-11..B-15
- Batch 4 (adaptive/ subpackage, 13 files) → B-16..B-22

---

## Batch 1: grid / fuse / core infra

### B-1
- kind: unfitted-threshold
- location: `src/senselab/audio/workflows/audio_analysis/fuse.py:559` (`derive_mask_from_axes`, `settled_below: float = 0.35`)
- defect: Whether a bucket counts as "settled" — and therefore becomes a `target_free` mask region that discounts speaker/other signals in later rounds — is gated on a bare `0.35` default with no measurement behind it. `keys.py`'s own docstring (`keys.py:144`) names this exact default as the paradigm case of the defect the codebase is supposed to have eliminated ("a derivative whose choice is not in its key is the `settled_below=0.35` default argument all over again").
- failure: A bucket where the presence axis (and background_mask, if present) reads uncertainty 0.34 becomes a `target_free` region with confidence `1-0.34=0.66`, attenuating any signal claiming a speaker there in round >= 1 via `rounds.regional_weights`; a bucket reading 0.36 — qualitatively indistinguishable — gets no region and no discount. `derive_mask_from_axes` is `fuse_axes`'s default `derive=` argument, wired into every real run through `write_final_uncertainty`/`fold_run_axes`; `run_config.py` exposes no override, so it cannot even be tuned per run.
- consumer: N/A
- target: N/A

### B-2
- kind: unfitted-threshold
- location: `src/senselab/audio/workflows/audio_analysis/fuse.py:831` (`fuse_axes`, `unsettled_above: float = 0.6`)
- defect: Whether an unsettled bucket is offered to the D-10 `remeasure` hook (and counted in the C4 "untried actions" convergence criterion) is gated on a bare `0.6` default with no derivation, and no caller in the package overrides it.
- failure: A bucket at uncertainty 0.61 is added to `_pending`, offered to `remeasure`, and counted as an untried action (blocking C4/convergence until spent); a bucket at 0.59 never contributes to that count — an arbitrary boundary that changes both what gets a second look and whether the loop reports as "converged."
- consumer: N/A
- target: N/A

### B-3
- kind: misnamed-statistic
- location: `src/senselab/audio/workflows/audio_analysis/aggregators.py:86-91` (`apply_aggregator`, `"disagreement_weighted"` branch)
- defect: The formula `(1 - mean_conf) * max_u` algebraically reduces to `mean(uncertainty) * max(uncertainty)` — a function of the *level* of uncertainty, not of *disagreement* (spread/variance) across sub-signals — despite the name and inline comment claiming it "surfaces buckets where many signals are slightly off rather than one wildly off."
- failure: Five sub-signals unanimously reporting uncertainty 0.9 (perfect agreement) score `0.9*0.9=0.81`; five sub-signals split four at 0.0 and one at 1.0 (textbook disagreement) score `mean=0.2, max=1.0 -> 0.2`. The aggregator ranks the unanimous-agreement bucket ~4x higher "disagreement" than the genuinely disagreeing one.
- consumer: `disagreements.py:75,94,118` ranks buckets in `disagreements.json` by `triage_score` desc — the column `apply_aggregator` populates via `fuse.fuse_axis`'s `aggregator` argument (`uncertainty.aggregator` in `data/run_config/default.yaml:236`, one of four selectable values). Selecting `disagreement_weighted` to prioritize genuinely contested buckets instead prioritizes unanimous maximal-doubt buckets.
- target: N/A

### B-4
- kind: promotion-candidate
- location: `src/senselab/audio/workflows/audio_analysis/level.py:129-289` (`apply_gain_db`, `integrated_lufs`, `loudness_range_lu`, `true_peak_dbtp`, `clipped_fraction`, `normalization_gain_db`, `peak_limited_gain_db`)
- defect: Pure, standards-grounded (BS.1770 / EBU Tech 3342) loudness/gain/clipping measurements over a raw `(waveform, sampling_rate)` pair, with no reference to grid/keys/contracts/rounds or any other workflow bookkeeping.
- failure: N/A
- consumer: N/A
- target: `senselab.audio.tasks.quality_control` (has `metrics.py`/`checks.py`, no LUFS/true-peak functions yet). Blocked only by `AudioVariant`, `GainCapExceededError`, `measure_variant`, `write_level_json` staying workflow-specific (variant names, `level.json` contract path, gain-cap-as-provenance policy) — the seven listed functions have no such entanglement and could move with a thin re-import left for workflow callers.

## Batch 2: speaker / identity / statistics

### B-5
- kind: unfitted-threshold
- location: `src/senselab/audio/workflows/audio_analysis/support.py:276` (`MIN_LOW_FRACTION`), used at `support.py:301-353` (`informative_evidence`)
- defect: `MIN_LOW_FRACTION = 0.02` gates whether an evidence signal is admitted into the speech-presence corroboration pool, but its own docstring says the only numbers ever offered as justification (503/697, 601/697, 0.500, 0.897, etc.) were measured while `native_confidence` was read undirected — a bug the docstring itself says makes the per-voter verdicts unusable "before they are cited again." The code still uses the constant to gate the same decision the disowned numbers were meant to justify.
- failure: A genuine VAD-like signal whose "no speech" fraction sits just under 0.02, or an acoustic-proxy signal that clears 0.02 only because of the undirected-confidence artifact, is silently included or excluded from evidence with no valid measurement behind the specific cutoff — either wrongly discounting every claimant in a run, or wrongly treating a flat proxy as discriminating evidence. Both directions are unverified pending re-measurement, exactly the state the docstring says must not be cited.
- consumer: `informative_evidence()` feeds `evidence_signal_names`/`signal_support()`, which sets `support` used by `speaker_identity.claims_from_perturbations` (-> `speaker_count_posterior`) and `reliability.measured_weights`'s `backing` factor — changing how much every speaker-count and speaker-identity claim is trusted.
- target: N/A

### B-6
- kind: unfitted-threshold
- location: `src/senselab/audio/workflows/audio_analysis/speaker_identity.py:121` (`multimodal_threshold: float = 0.15`)
- defect: The probability cutoff for declaring a speaker-count posterior "multimodal" has no stated derivation, unlike the well-derived numbers in `calibration.py`'s detection-margin ladder.
- failure: `is_multimodal` directly gates `SpeakerHypothesis.converged` (`speaker_identity.py:585`: `converged = not posterior.is_multimodal and doubt < 0.5`). A posterior `{2: 0.86, 3: 0.14}` reads unimodal (converged); `{2: 0.84, 3: 0.16}` reads multimodal (not converged) — a 2-point probability shift flips whether the adaptive loop believes the speaker-count question is settled, with no measurement supporting 0.15 over 0.10 or 0.20.
- consumer: `posterior.is_multimodal` surfaces in `summary.py`'s "(multi-modal)" label and `to_json()` for `final/speakers.json`; `converged` feeds the adaptive loop's stopping logic.
- target: N/A

### B-7
- kind: unfitted-threshold
- location: `src/senselab/audio/workflows/audio_analysis/speaker_identity.py:60` (`_SUPPORTED_THRESHOLD = 0.5`)
- defect: The cutoff at which a source's measured physical support counts as "corroborated by the audio" (`has_supported_evidence`) is an unexplained midpoint with no derivation, next to `signal_support`'s well-derived floor (`MIN_EVIDENCE_WEIGHT = 0.05`, derived in `floors.py`).
- failure: A source with `source_support == 0.49` reports `has_supported_evidence: false` in `final/speakers.json`; `0.50` reports `true` — a binary, actionable verdict resting on an unmeasured boundary.
- consumer: `SpeakerHypothesis.to_json()["has_supported_evidence"]`, written to `final/speakers.json` (`contracts.py:629`).
- target: N/A

### B-8
- kind: unearned-confidence
- location: `src/senselab/audio/workflows/audio_analysis/identity_binding.py:145`
- defect: `binding_agreement` returns `0.0` whenever `eligible == 0` (`bound=unbound=[]`), conflating "every diarizer explicitly rejected this speaker" (real disagreement, `eligible>0`, none bound) with "no diarizer had any spans to bind against, or every diarizer was already at capacity" (nothing was actually checked). The docstring only describes the first case; `identity_binding_test.py` never exercises `eligible == 0`.
- failure: `speaker_spans={"S2":[(10.0,11.0)]}` with two diarizers both already at declared capacity elsewhere (`is_censored_at(...)` True for both) produces `bound=[]`, `unbound=[]`, `censored=["toolA","toolB"]`, `eligible=0` -> `binding_agreement=0.0` — bit-for-bit identical to two uncensored diarizers unanimously rejecting S2, though the true evidentiary state is completely different.
- consumer: `per_speaker_presence()`'s `binding_agreement` field, exported via `__all__` for `final/`-style per-speaker output. Currently only exercised by `identity_binding_test.py` (not yet wired into `fuse.py`'s production path, which reads `bind_labels()`'s `assignment` directly instead), so the concrete run-time effect is UNVERIFIED, but the defect is live in the shipped, tested public function.
- target: N/A

### B-9
- kind: unearned-confidence
- location: `src/senselab/audio/workflows/audio_analysis/speaker.py:636-640` (`n_models = len(clusters)`, `share = len(sources) / n_models`), consumed by `speaker_identity.py:627` (`build_speech_presence_tracks`)
- defect: `per_speaker_tracks`'s `speech_presence_confidence` is the fraction of models present in `diar_ok` for a bucket that placed the speaker there. `diar_ok` (`speaker.py:213`) drops any model whose block `status != "ok"` — a diarizer that crashed or was never configured shrinks the denominator instead of being counted as an absent vote. The docstring handles "a model reporting silence" (stays in the denominator) but not "a model that never ran" (silently excluded).
- failure: A run where 3 of 4 configured diarizers fail leaves `diar_ok={"pyannote":...}`; every bucket where pyannote reports a speaker gets `speech_presence_confidence=1/1=1.0`, `uncertainty=_binary_entropy(1.0)=0.0` — byte-identical to all 4 diarizers unanimously agreeing. `final/per_speaker_presence.parquet` cannot distinguish four-way corroboration from one uncorroborated detector with three silently missing.
- consumer: `PerSpeakerPresenceTrack.to_row()` -> `final/per_speaker_presence.parquet` (`contracts.py:631`). The sibling `speaker_assignment` voter fixed this exact failure mode via `n_sources`/`source_outcomes` (comment at `speaker.py:552-555`); the fix was not carried to this per-speaker-track computation.
- target: N/A

### B-10
- kind: promotion-candidate
- location: `src/senselab/audio/workflows/audio_analysis/statistics.py:51` (`confidence`), `:81` (`variability`), `:112` (`entropy_uncertainty`), `:132` (`epistemic_uncertainty`)
- defect: Pure statistics over generic `Sequence[float]`/`Mapping[str, float]` inputs — weighted-vote probability, population standard deviation, normalized Shannon entropy, entropy mutual-information decomposition — with no reference to any audio-analysis type or workflow bookkeeping.
- failure: N/A
- consumer: N/A
- target: `senselab/utils/tasks/` (e.g. a new `uncertainty.py` beside `pooling.py`/`cca_cka.py`) — the codebase's own `project_mc_dropout_optional` note wants exactly this primitive elsewhere. Nothing blocks the move: the module is already stdlib-only, and is imported by `attribution.py`/`background_mask.py`/`fuse.py`/`harmonize.py`/`joint.py`/`occupancy.py`, never the reverse. Cost is six import-site renames.

## Batch 3: background-scene / speech-presence / plot

### B-11
- kind: unfitted-threshold
- location: `src/senselab/audio/workflows/audio_analysis/global_summary.py:195-213`
- defect: The PESQ (2.0/3.5), STOI (0.5/0.85), and SI-SDR (0.0/15.0) ramp bounds that turn `torchaudio_squim` quality scores into the run's "quality" claim uncertainty are asserted as "literature-derived acceptance thresholds" with no citation anywhere in the module, config, or specs tree — the same shape as the already-fixed HNR-ramp defect. The docstring itself contradicts the code: it states SI-SDR "below 5 dB poor," but the ramp's low anchor is `0.0`, not `5.0`.
- failure: A recording with PESQ = 2.6 (an ordinary, usable-speech "fair" MOS-equivalent quality, not "clean") yields `pesq_unc = (2.6-2.0)/(3.5-2.0)` inverted to `0.6`. Via `combined = max(...)` that alone can dominate `quality.uncertainty` and the run's headline `combined_uncertainty` — flagging an ordinary recording as failing the "high quality" claim on an unfitted boundary.
- consumer: `scripts/analyze_audio.py` stores `compute_run_global_summary(...)`'s `combined_uncertainty` as the run's bottom-line four-claim verdict (`summaries["global_uncertainty"]`).
- target: N/A

### B-12
- kind: unearned-confidence
- location: `src/senselab/audio/workflows/audio_analysis/disagreements.py:152`
- defect: `"high_uncertainty_rate": (high_count / total_rows) if total_rows else 0.0` reports `0.0` ("nothing was uncertain") whenever `total_rows == 0` — every axis's harvest/fuse step failed, or the run covered zero measurable buckets — collapsing "we could not measure anything" into "we measured, and it was all clean."
- failure: A run where harvesting fails for every axis produces `disagreements.json["totals"] = {"total_rows": 0, "high_uncertainty_rate": 0.0, ...}`.
- consumer: `scripts/check_layering.py:121` prints `d['totals']['high_uncertainty_rate']` directly as a regression metric against a stored baseline (was 0.9941) — a total-harvest-failure run prints `0.0000`, reading as a dramatic improvement rather than a broken run.
- target: N/A

### B-13
- kind: unfitted-threshold
- location: `src/senselab/audio/workflows/audio_analysis/noise_floor.py:294,358-373` (`recorder_margin_db`, default 3.0 dB)
- defect: This margin decides whether a band's `binding_floor` reads `"recorder"` (report nothing perceptual) or `"perceptual"`. `data/detection_margin/2026-07-29.json`'s `derivation` block documents every other margin (the 3/6/10 dB ladder, `prominence_ratio_db`) against cited ISO/ECMA sources or measured level-probe floors, but `recorder_margin_db` is absent from it; its only textual grounding (`specs/.../contracts/policy-profile.md:117`) is the qualitative phrase "within a few dB" from FR-021b, not a measured or cited number.
- failure: UNVERIFIED (plausible that a band floor 2.9 dB above the recorder floor is mislabeled "recorder-bound" versus one at 3.1 dB reading "perceptual," but no measured distribution of real recorder-vs-room gaps was available to construct a concrete misclassified case).
- consumer: `estimate_noise_floor` stamps `binding` onto every `NoiseFloorEstimate`, written to `noise_floor.parquet` and read back by `detect_stationary_sources`, which copies the field into every stationary-source finding.
- target: N/A

### B-14
- kind: promotion-candidate
- location: `src/senselab/audio/workflows/audio_analysis/acoustic.py:50-76,127-167` (`lufs_track`, `level_above_floor_track`)
- defect: Pure `(waveform, sampling_rate) -> (times, values)` numpy computations (BS.1770-style short-term loudness; broadband excess over a bias-corrected percentile floor) with zero reference to workflow bookkeeping types — only `math`/`numpy` and, in the caller, `pyloudnorm`.
- failure: N/A
- consumer: N/A
- target: `senselab/audio/tasks/features_extraction/` (new `loudness.py` beside `opensmile.py`/`torchaudio.py`, which has no LUFS/loudness support today). Nothing structural blocks the move beyond updating `speech_presence.py`'s import.

### B-15
- kind: promotion-candidate
- location: `src/senselab/audio/workflows/audio_analysis/occupancy.py:133-169` (`occupancy`, `_union_length`)
- defect: Clip a set of `(start, end)` intervals to a window and sum union length — generic interval algebra whose only workflow coupling is that `occupancy()`'s signature takes the workflow's `Spans`/`Span` dataclasses rather than plain tuples.
- failure: N/A
- consumer: N/A
- target: `senselab/utils/tasks/` (an interval/union-length helper). `_union_length` is already tuple-only and movable as-is; `occupancy()` would need a tuple-based signature with a one-line adapter kept in the workflow.

## Batch 4: adaptive/ subpackage

### B-16
- kind: unearned-confidence
- location: `src/senselab/audio/workflows/audio_analysis/adaptive/loop.py:315-317`
- defect: `run_state = "converged" if not not_admitted else "no_runnable_interventions"` is set whenever no intervention fired *and* none was even proposed/blocked that round — true whenever every open bucket's doubt sits in `[theta_low, theta_high)`, below the region-seed threshold `regions.propose_regions` requires — not only when buckets have actually reached `u <= theta_low` (the per-bucket definition of "converged" used everywhere else, e.g. `convergence.apply_convergence_marks:58`).
- failure: A recording whose speaker-axis buckets are all steady at doubt ~0.40 (above `theta_low=0.33`, below `theta_high=0.66`) never seeds a region (`regions.py:46` requires `_u(rows[i]) >= theta_high`), so round 2 has `regions=[]`; if P3/C9 also find no candidates, `admitted=[]`, `not_admitted=[]`, and the loop reports `run_state="converged"` after one round. `build_convergence_report` then reports `"converged": true` in the same document whose own `per_axis["speaker"]` block lists those buckets as `"open"` with nonzero `residual_mass` — a self-contradicting headline verdict, reachable with one round and no history for `detect_non_convergence` to override.
- consumer: `run_adaptive_loop`'s returned `"converged"`/`"termination_reason"` and `final/decisions.json`'s `convergence.converged` — the single top-level success flag a caller checks.
- target: N/A

### B-17
- kind: unearned-confidence
- location: `src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:198-244` (payloads in `repair_identity`), consumed via `adaptive/interventions.py:798-812` (`_i1_execute`) and `:850-865` (`_i2_execute`)
- defect: I1's added vote (`change_point_times`/`change_point_confidence`) and I2's added vote (`speaker_label`/`cluster_id`/`speaker_changed_from_prev`) carry no key `fuse.per_signal_uncertainty` recognizes (`_UNCERTAINTY_FIELDS`/`_LOGPROB_FIELDS`/`_CONFIDENCE_FIELDS`/`"speaks"` — fuse.py:54-77), so `fuse_axis` (called by `belief.VoteStore.reaggregate_bucket`, belief.py:810) never scores either vote — yet `belief.py:818`'s `contributing_sources` is built from raw `active_votes` keys and *does* include them.
- failure: A region repaired by I1/I2 each round shows `mean_after == mean_before` every round in `loop.py`'s own before/after accounting — real diarizer votes are what move the number, never these "new" votes — while `contributing_sources` in the same row (and `L2/round/<n>/estimates/speaker.parquet`, and the LabelStudio view via `labelstudio.py:751`) lists `embedding_changepoint/consensus` and `embedding_recluster/consensus` as if they had spoken toward the value. After `max_region_rounds` (2) of this no-op touching, `convergence.apply_convergence_marks:79-86` marks the region `irreducible: no_reduction_under_available_interventions` — a false verdict; the true reason is the vote-payload schema mismatch, not an aleatoric limit or absent working intervention.
- consumer: `convergence.build_convergence_report`'s `irreducible_regions[].reason`; `contributing_sources` read by `labelstudio.py:751` and every round's estimate parquet.
- target: N/A

### B-18
- kind: misnamed-statistic
- location: `src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:227-231` (`repair_identity`)
- defect: `seg["boundary_confidence"] = {"start": cp_conf.get(round(seg["start"],4), 0.5), "end": cp_conf.get(round(seg["end"],4), 0.5)}` — `cp_conf` only has entries for genuine embedding change-points; a segment edge from a diarizer-only cut or a voiced-span boundary silently gets the literal `0.5`, indistinguishable in the output from a measured prominence of 0.5.
- failure: `adaptive/fusion.py:296-297` writes this into `final/diarization.json` under a comment calling it "real boundary confidences from change-point prominence" for every segment — a diarizer-sourced cut (unmeasured, fabricated 0.5) and a barely-significant real change-point at e.g. measured confidence 0.05 (a weak but genuine detection just above `cp_floor=0.15`) are misordered: the fabricated value reads as *more* confident than the true, low-confidence measurement.
- consumer: `adaptive/fusion.py:296-297,324` -> `final/diarization.json`'s `segments[].boundary_confidence`.
- target: N/A

### B-19
- kind: unfitted-threshold
- location: `src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:939` (`_p2_trigger`)
- defect: `fires = coarse_share >= threshold or mean_instability > 0.0` — the second disjunct compares a continuous variance-derived quantity against exactly `0.0` with no configured or derived cutoff, contradicting the function's own docstring two lines above ("a **high** value means the bucket straddles an onset").
- failure: `mean_instability` (mean `frame_dispersion` over belief rows) is essentially always `>0.0` for real-valued frame posteriors, so this disjunct is nearly always true regardless of whether instability is actually "high." `P2_fine_posteriors` (cost class `medium`, capped at 24/run) fires far more often than the design intends, consuming medium budget that U1/U2 then lose to `deferred_budget`.
- consumer: `adaptive/policy.plan_round`'s budget admission for the `medium` cost class (shared with U1/U2).
- target: N/A

### B-20
- kind: misnamed-statistic
- location: `src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:14,196` (`_COST_WEIGHT`, `plan_round`) and `adaptive/interventions.py:502-503,1134-1139` (`_u2_gain`, `_mass_gain`, `_n_candidates_gain`)
- defect: `priority = gain / _COST_WEIGHT[cost_class]` is computed and sorted across every rule in one list, but `gain` is not one quantity: `_mass_gain` returns a bounded doubt-seconds `uncertainty_mass`, `_n_candidates_gain` (P3/C9) returns a raw unbounded count of triggering buckets, and `_u2_gain` returns `mass * epistemic * 10.0` — an arbitrary x10 with no derivation anywhere in the file or `default.yaml`. None is normalized to the others' scale.
- failure: On a recording with 50 uncorroborated-speech buckets, P3's `priority = 50/1 = 50` dwarfs a genuinely contested speaker region's bounded `_mass_gain` priority (typically `<1`), and U2's arbitrary x10 multiplier inflates its priority relative to an equal-mass U1 candidate of the same `medium` cost class — `priority`, the single number `plan_round` uses to rank and admit interventions, reflects which gain formula a rule happened to be assigned, not relative value across rule types.
- consumer: `adaptive/policy.plan_round`'s admission order and `final/iterations.json`'s `priority` field.
- target: N/A

### B-21
- kind: unearned-confidence
- location: `src/senselab/audio/workflows/audio_analysis/adaptive/convergence.py:64-78` (`apply_convergence_marks`) together with `adaptive/provenance.py:1-136` (entire module, dead)
- defect: `provenance.classify_resolution`/`RevisionRecord` exist to distinguish "uncertainty fell because independent evidence arrived" from "uncertainty fell because the loop overwrote its own prior value and then re-measured the overwrite," but (confirming sweep A's A-5) neither has any call site in `loop.py`/`interventions.py`/`belief.py`. `convergence.py:77`'s `improvement = float(prev_u) - float(last_u)` is exactly the undifferentiated raw delta the dead module's docstring warns against — it cannot tell a genuine-evidence uncertainty drop from a self-confirming re-score of a prior overwrite.
- failure: A rule whose recomputation re-derives a value from the same evidence a prior round already committed (e.g. a cached identity-repair result subsequently re-read by another rule against the same embeddings) can register `improvement >= epsilon` in `convergence.py` purely from re-scoring one prior computation, and `apply_convergence_marks` has no mechanism to classify that as `revision` rather than genuine progress toward `converged`.
- consumer: `convergence.apply_convergence_marks`'s "stalled" check and the resulting `converged`/`irreducible` status feeding `build_convergence_report`.
- target: N/A

### B-22
- kind: promotion-candidate
- location: `src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:125-149` (`_agglomerative_cosine`), `:46-50` (`_l2`), `:53-78` (`change_point_trajectory`)
- defect: Deterministic average-linkage agglomerative clustering on a cosine-distance matrix, L2-normalization, and an adjacent-window cosine-distance trajectory (fixed 3-tap smoothing) are generic numerical routines with zero dependency on `Region`/`VoteStore`/policy dicts — plain arrays and floats throughout.
- failure: N/A
- consumer: N/A
- target: `senselab/utils/tasks/` (e.g. beside `dimensionality_reduction.py`) for `_l2`/`_agglomerative_cosine`, or `senselab/audio/tasks/speaker_embeddings/` for `change_point_trajectory`. Blocked only by leading-underscore module-private naming and `noqa: ANN401` typing shortcuts — the functions take no adaptive-loop state.

---

## Checked and clean

**Batch 1 (grid/fuse/core infra):** `grid.py` (`BucketGrid` validation and `DEFAULT_TIME_GRID` checked; `eps=1e-9` is a float-rounding guard, not a decision gate), `resolution.py` (`NATIVE_RESOLUTION_S` are model-spec constants, not gates; `declared_resolution_s` confirmed to have zero production callers), `floors.py` (`MIN_EVIDENCE_WEIGHT=0.05` carries a stated derivation, verified against `fuse.py`'s weighting path), `__init__.py` (independently reconfirmed A-1's stale docstring; the lazy-export mechanism itself is correct), `types.py` (no numeric literals; `FusedAxis`/`SignalResult` field claims checked against `fuse.py`), `aggregators.py` (`min`/`mean`/`harmonic_mean` branches verified correctly named; only `disagreement_weighted` — B-3 — is a defect), `level.py` (every literal traced to a cited standard: BS.1770 Annex 2, EBU Tech 3342, full-scale-pin `0.9999`; `GainCapExceededError` raised, never silently clamped), `layout.py` (every path helper cross-checked against contracts.py's declared patterns), `stage_context.py` (`STAGE_VERSIONS` all currently 1; `_commit_sha_for`'s three-way branching matches its docstring; noted but not reported — a `PassPlan.mask_grid` docstring/prose mismatch with `grid.py`'s current default, prose territory not math; **fixed 2026-08-16 on `fix/extraction-axes-edge`**, together with the two other places that said the default grid is 0.5 s — `stages.stage_background_mask`'s `grid` argument doc and `background_mask_test.py`'s prose. No register row was filed for the three: this sweep saw the mismatch and classified it, so filing it now would record a second, contradictory verdict on a call already made rather than a new finding. The test whose *assertion* the same stale default had disarmed is a separate defect and is filed, as F-191), `stage_io.py` (no numeric literals; acyclicity argument checked against `keys.py`), `io.py` (no thresholds; `merge_json`'s shallow-merge behavior matches its docstring), `votes.py` (`DEFAULT_UTTERANCE_SCENE_COUPLING` carries a written rationale and is overridable; `mask_from_pvoice`/`intensity_mask` confirmed to have zero production callers, so not reported as a live threshold), `keys.py` (path-construction methods have no numeric literals; houses B-1's self-indicting comment), `run_config.py` (every `_build` fallback cross-checked identical to `default.yaml`; `snr_floor_db=10.0` is explicitly marked "UNDERIVED" in the YAML's own comment — transparent, not hidden), `contracts.py` (purely structural/declarative DSL; keyword sweep found no numeric literal gating a statistical verdict).

**Batch 2 (speaker/identity/statistics):** `speaker.py` (FR-007 embedding-calibration gating verified; `cluster_cosine_threshold=0.5` carries a stated rough derivation — EER for ECAPA on VoxCeleb — so not flagged), `speaker_identity.py` (empty/all-uncertain posterior fallbacks and `perturbation_uncertainty`'s >=2-point guard verified honest), `harmonize.py` (`MIN_CENTROID_SIMILARITY=0.5` module default confirmed never reached in production — the one live caller always passes an explicit derived value), `identity_binding.py` (Hungarian global matching and no-margin maximal-doubt default verified consistent), `joint.py` (window-lag span comparison and FR-007 null-guards verified), `statistics.py` (all four functions verified against their own stated definitions), `measurements.py` (pure serialization, `None`-vs-missing discipline enforced), `support.py` (direction-aware `presence_probability` and diarizer/ASR exclusion from evidence pool verified, aside from B-5), `influence.py` (`derived < independent` gate-ordering enforced via raised `ValueError`, not silently; module's own docstring argues against relocation), `reliability.py` (shared-bucket-key comparison and relative corroboration factor verified), `calibration.py` (the "good case" — every literal traces to a cited standard or is labeled provisional/measured), `degradation.py` (`DEFAULT_ANCHORS` explicitly labeled uncalibrated-but-documented; `None`-propagation verified, no `0.0` manufacturing), `clustering.py` (`cosine_threshold`/`cross_group_threshold` feed only plot-color assignment, confirmed by grep — cosmetic, not a scored verdict), `invariance.py` (relative-deviation scaling and `None`-when-unprobed behavior verified; three perturbation transforms confirmed output-preserving).

**Batch 3 (background-scene/speech-presence/plot):** `noise_floor.py` (margin ladder and `prominence_ratio_db=9.0` traced to cited ISO/ECMA sources; the "no published precedent" per-activity-stratum estimator confirmed dead code — never invoked with `target_active`; `recorder_margin_db` separately flagged as B-13), `sources.py` (margin ladder, flatness/pregain/floor-signature guards, excision-cost logic all check out; `discount_for_mask_uncertainty`'s undeived `0.3` confirmed dead code), `mask_harvest.py` (`TARGET_POLARITY` and asymmetric uncertainty treatment correctly reasoned; thresholds policy-sourced), `occupancy.py` (censoring-aware count-posterior model verified correct, aside from the promotion note B-15), `acoustic.py` (LUFS anchors justified inline; `loudness_confidence` confirmed dead code — no production caller), `signal.py` (bounded-unit checks and ref-vs-SHA distinction verified correct, no thresholds), `estimates.py` (`control_doubt` correctly returns `None`, not `0.0`, for missing/NaN confidence — the right-side case, not a defect), `disagreements.py` (ranking logic sound aside from B-12), `attribution.py` (entropy-based `speaker_assignment_doubt` matches its own worked example), `global_summary.py` (hallucination detection and PII `None`-vs-`0.0` handling verified correct, aside from B-11), `summary.py` (`_axis_summary` excludes unmeasured buckets from the mean rather than zero-filling), `speech_presence_link.py` (documents its own prior fixes — the removed unfitted HNR ramp and misused silhouette-as-presence-vote — with measured costs; current anchors echoed in `default.yaml`, `coarse_voter_weight=0.25` already flagged "UNDERIVED" there, not a new finding), `shapes.py` (pure type module, no thresholds), `rounds.py` (confidence-scaled trust withdrawal and convergence criteria reference `floors.py`'s derived `MIN_EVIDENCE_WEIGHT`, not a local literal), `plot.py`/`l2_plot.py` (embedded `cosine_threshold=0.5` only affects plot color grouping, not a reported verdict), `sampler.py` (dispatch and rescale logic verified correct; tightly coupled to workflow key/shape types, not a promotion candidate).

**Batch 4 (adaptive/ subpackage):** `convergence.py` (every function traced; `theta_low`/`epsilon`/`max_region_rounds` sourced from policy with inline derivation comments — findings B-16/B-21 arise from its interaction with loop.py/provenance.py, not from this file alone), `regions.py` (seed/expand/merge/rank pipeline verified against its own docstring; root cause of B-16 confirmed here), `ls_final.py` (`_conf_bin` tertile bins are display-only LabelStudio labels, not a verdict gate), `policy.py` (`load_policy`/`_validate_floors`/`BudgetLedger` verified sound; `_COST_WEIGHT` is the basis of B-20), `__init__.py` and `types.py` (no computation), `belief.py` (`VoteStore`/`replay_check`/`fused_parity` traced in full; `_attach_floor`'s anchor-based floor verified to floor at `None`, not `0.0`), `triage.py` (both functions confirmed to have zero production call sites — not reported as a math defect since they gate nothing reachable), `identity_repair.py` (source of B-17/B-18/B-22; `detect_change_points`'s threshold and `recluster_cosine_threshold` present but undocumented — not reported standalone, no additional concrete failure beyond B-17/B-18), `interventions.py` (every rule's trigger/guard/gain/execute traced; confirmed `_p3_execute` floors rather than deletes; source of B-19/B-20, plus the B-17 payload-schema mismatch), `loop.py` (full ingest/round/fusion/writer trace; source of B-16; also confirmed a live re-instance of belief.py's `n_sources`-fallback bug at line 806, folded into B-17 since it shares the same root cause and consumer), `provenance.py` (confirmed dead, matching sweep A's A-5; used as B-21's connection to a live convergence-math gap), `corroboration.py` (`None`-propagation and floored-exponent gate verified sound — well-guarded against the 0.0-vs-None trap it explicitly discusses).

### B-shadow
- kind: promotion-candidate
- location: src/senselab/audio/workflows/audio_analysis/types.py:1
- defect: The module is named `types`, which shadows the stdlib module of the same name for any
  Python process whose working directory is this package.
- failure: `cd src/senselab/audio/workflows/audio_analysis && python -c "import ast"` fails with
  `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types'` —
  an error naming `weakref`, not the real cause. Observed while writing this plan.
- target: Rename, or accept and document. Not a promotion in the layering sense; recorded here
  because it is a naming defect in the computation layer with no better home.
