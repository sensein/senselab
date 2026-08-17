# Reproduction verdicts — C-/D- raised findings (F-162, F-164..F-176)

Scope: the 15 findings whose `raised-by` is a `C-` or `D-` id in `candidates/deduped.md`, of which
14 survived the refutation gate (`verdicts/refutation.md`). F-163 (C-2) was REFUTED and is skipped
per instructions. Four of the remaining 14 are `SURVIVED-CORRECTED`; the corrected mechanism (not
the originally filed one) is what each entry below reproduces or names an experiment for.

---

### F-162
- outcome: DEMONSTRATED
- script: repro/F-162.py
- observed: `fuse_consensus_words(asr_resolved)` called exactly as `compute.py:433` calls it (no
  `policy=`) reports `provenance["slot_overlap"] = 0.3` and `provenance["slot_mid_tol_s"] = 0.15`
  even with a configured policy (`asr_slot_overlap=0.95`, `asr_slot_mid_tol_s=0.001`) bound in the
  same scope — the hardcoded literals are used regardless of config, should be 0.95/0.001 if the
  config reached the fold. Additionally, `inspect.signature(harvest_asr_votes)` (the harvester
  `_reharvest_asr` calls on the U1 live-re-ASR intervention route, `adaptive/interventions.py:595`)
  has no `policy` parameter at all — confirming the corrected mechanism from `verdicts/refutation.md`
  (config is dead on *every* call path, including U1, not merely "unreachable elsewhere" as
  originally filed).

### F-164
- outcome: LATENT
- script: none
- experiment: compute ECAPA/ResNet embeddings on a child/sibling-pair speaker-verification corpus
  with ground-truth same/different-speaker labels; derive the empirical EER-optimal cosine
  threshold for that population and measure the error rate of the fixed adult-derived
  `cluster_cosine_threshold=0.5` / `merge_threshold=0.55` (`compute.py:105,533`, `embeddings.py:324-325`)
  at that threshold; compare against the error rate on an adult VoxCeleb-style control at the same
  threshold. (Refutation narrowed the surviving claim to these two fixed literals; `same_speaker_floor`/
  `diff_speaker_floor` are already re-measured per embedder when clustering succeeds, so they are
  out of scope for this experiment.)

### F-165
- outcome: DEMONSTRATED
- script: repro/F-165.py
- observed: a synthetic bucket with populated votes (cross-diarizer disagreement, an embedder's
  J2 change-point entries at two boundary times, per-embedder cosine doubt, J1 `overlap_count`,
  per-diarizer labels) and zero ASR word coverage (`word_coverage=0.0`, state `target_active`, not
  `target_free`) has `bucket_dict["votes"]` replaced by `{}` under the real wordless gate — wiping
  every one of those measurements, not just the two word-dependent attribution voters
  (`speaker_assignment`, `target_activity`) the gate is meant to null.
- **superseded 2026-08-16 by `d8cb7449`** (the observation above is kept as the record of what ran
  against the code as audited, and is not rewritten). The fix exempts the word gate wherever the
  mask region state is `target_active` or `nontarget_active`, and `target_active` is precisely the
  state this fixture builds — chosen at the time only to clear the *first* gate. Rerun against
  `fix/f165-mask-aware-word-gate`, the script prints
  `gate-enabled bucket_dict['votes'] wiped to {}: False` and then
  `Could not reproduce the defect as specified.`, exiting 1. That non-reproduction is the fix
  reading correctly on the audit's own fixture, not a broken script.
- **the fixture is deliberately not changed**, though switching `target_active` → `indeterminate`
  would still reproduce and would restore a zero exit. Three reasons, in order. (1) These scripts are
  dated evidence of what was demonstrated against the code as it stood, not a maintained test suite;
  editing the input to keep the output green destroys the one thing the file is for. (2) The green
  exit would be bought with an equal-or-worse fixture: `indeterminate` describes production no better
  than `target_active` did. (3) The honest complication, which is the reason this cannot be settled
  by picking a state at all — **in production the defect persists for every bucket regardless of the
  fix**, because the mask's per-region table never reaches `harvest_speaker_votes` (F-187 in
  `../register.md`: `stages.py` writes `BackgroundMask.to_json()`, which emits counters only, so
  `mask_regions` is `[]` and `state` is always `None`, a state the gate still applies to). This
  script's `target_active` fixture therefore never described production either; it described a
  document shape production does not produce. Any fixture choice here demonstrates a synthetic-shape
  defect, so the choice is between one that reads the fix and one that hides it, and the first is
  more useful. A dated STATUS note now sits in the script's docstring saying so; no executable line
  was touched.

### F-166
- outcome: LATENT
- script: none
- experiment: feed Whisper (or the run's configured ASR model set) infant-cry, toddler-babble, and
  laughter clips alongside matched adult non-lexical vocal-sound controls (cough, laugh, sigh);
  for each clip, record `no_speech_prob` and the emitted transcript text; measure the rate of
  confidently-hallucinated transcriptions (`said_something=True` and `nsp < policy.no_speech_threshold`,
  i.e. `speaks=True` per `speech_presence_link.py:249-273`) per group, and compare the pediatric/
  non-lexical group's rate against the adult-control group's rate.

### F-167
- outcome: LATENT
- script: none
- experiment: run the diarizer set on ground-truth-labeled adult-adult, adult-child, and
  child-child dyad recordings of matched duration/turn structure; compute `modal_count` accuracy
  (agreement with the true speaker count) and posterior calibration (e.g. Brier score or ECE of
  `probabilities` against the true count) for `speaker_count_posterior`, stratified by age
  composition, and compare the three strata.

### F-168
- outcome: LATENT
- script: none
- experiment: run an annotated infant-cry or toddler free-vocalization recording through the
  background-scene/mask stage under each declared task label (including a hypothetical `cry`/
  `babble` task type, which does not currently exist in `target_event_types_by_task`); measure the
  fraction of ground-truth cry/babble time scored `target_free`/`nontarget_active` by
  `target_event_types_for` (`background_mask.py`), and compare against an adult breath-task control
  run the same way.

### F-169
- outcome: LATENT
- script: none
- experiment: record matched breath-task and conversational-task audio from the same device/room/
  session (so channel and noise floor are held constant); run `scene_degradation`
  (`degradation.py:129-172`, fixed `snr_clean_db=25.0`/`c50_clean_db=30.0` anchors, no `task_type`
  parameter) on both; compare `quality_snr`/`quality_reverb` uncertainty between the two, checking
  whether the breath-task recording (quiet capture is correct by design there) scores as
  substantially degraded purely from the SNR gap to the conversational anchor.

### F-170
- outcome: LATENT
- script: none
- experiment: run YAMNet's top-1 AudioSet label over a corpus stratified by speaker age band
  (adult conversational, child connected speech, toddler one-word utterances, infant cry/babble);
  measure the false "non-speech veto" rate per band (fraction of ground-truth-speech windows the
  `_speech_window_mask` veto — `compute.py:890-1009` — excludes from `_cluster_pass_speakers`) and
  compare each pediatric band's rate against the adult band's rate.

### F-171
- outcome: LATENT
- script: none
- experiment: on a corpus with ground-truth vocalization durations spanning infant cries (~<0.5s),
  toddler one-word utterances (~<1s), and adult conversational turns (~2s+), extract ECAPA
  embeddings under the fixed `embedding_window_s=2.0` (`compute.py:101,592-593`); measure
  same/different-speaker cosine separation (e.g. AUC or EER of the same/diff cosine distributions)
  as a function of true-vocalization-duration-to-window-occupancy ratio, and identify the
  occupancy fraction below which separation degrades past a usable operating point.

### F-172
- outcome: DEMONSTRATED
- script: repro/F-172.py
- observed: `compute_run_global_summary(n_speakers=2, ...)` returns
  `single_speaker["uncertainty"] = 1.0` under both `expects_speech=True` and `expects_speech=False`
  — the only exposed population/expectation knob governs solely the `n_speakers==0` branch, so
  there is no way to score a correct caregiver-mediated 2-speaker recording other than as maximal
  ("1.0") violation, identical to an unexpected-second-speaker failure case. Expected (if
  task-aware): a lower or `None` value for a task where 2 speakers is the correct paradigm.

### F-173
- outcome: LATENT
- script: none
- experiment: on a corpus with time-aligned ground truth containing at least one infant/toddler
  and one adult speaker per recording, run `detect_change_points`/`repair_identity`
  (`identity_repair.py:81-99,152-244`, fixed `min_segment_s=0.25`, `recluster_cosine_threshold=0.45`)
  and compute segment-purity / Adjusted Rand Index against ground truth separately for
  infant/toddler-attributed spans versus adult-attributed spans in the same recordings, compared
  against an adult-only conversational corpus run the same way.

### F-174
- outcome: LATENT
- script: none
- experiment: build or obtain infant-caregiver recordings with hand-labeled true co-vocalization
  spans (simultaneous, non-turn-taking crying/babbling with adult speech), plus a matched
  adult-adult corpus with labeled overlap spans; run I4 (`overlap_track_from_spans`,
  `adaptive/backends.py:237-278`, feeding `ALEATORIC_FLOOR_TERMS`/`_attach_floor` in
  `adaptive/belief.py:1099-1146`) on both, and compare `overlap_posterior` against ground truth per
  population — checking whether infant-caregiver co-vocalization reads as falsely low overlap
  because the diarizers individually fail to register the second (child) voice as a concurrent
  span.

### F-175
- outcome: LATENT
- script: none
- experiment: on recordings with labeled infant/toddler vocalization at low measured `p_voice`,
  alongside labeled adult speech at comparably low `p_voice`, measure how often
  `len(families) >= 2` (`interventions.py:430-473`, C9's ASR-family-agreement gate) fires inside
  each group's spans; compare the pediatric/non-lexical group's firing rate against the adult
  group's rate to quantify how often the correction mechanism is structurally inapplicable versus
  genuinely inapplicable (true silence).

### F-176
- outcome: LATENT
- script: none
- experiment: alongside the standard `transcript.wer`/`wer_normalized` computed by
  `adaptive/evaluate.py:85-130` (which excludes `untranscribed` GT spans from both ref and hyp),
  compute and report the fraction of total recording/GT duration falling in `untranscribed` spans,
  stratified by available speaker/age metadata; compare the current WER-only headline figure
  against a coverage-adjusted accuracy metric on a corpus with substantial infant/toddler
  untranscribable material versus an all-adult corpus, to quantify how much the headline WER
  silently over-certifies pipeline performance on the pediatric corpus.

---

Summary: 3 DEMONSTRATED (F-162, F-165, F-172), 11 LATENT (F-164, F-166, F-167, F-168, F-169, F-170,
F-171, F-173, F-174, F-175, F-176). All LATENT experiments name a corpus/property, a metric, and an
explicit comparison per the task's own bar; none could be executed here (no child/pediatric/
non-verbal-vocalization corpus is available in this environment), which is the expected outcome for
findings in this population-coverage class.

**This tally is the gate's outcome against the code as audited (2026-08-15), not a statement about
HEAD**, and it is left standing as that record. Since it was written, two of the three DEMONSTRATED
have been fixed — F-162 in `5c0b10a6`, F-165 in `d8cb7449` — and `repro/F-165.py` no longer
reproduces on its own fixture as a result (see its entry above, which carries the superseded note
and the reasoning). Read "3 DEMONSTRATED" as "3 were demonstrated", never as "3 are live"; each
entry above, not this line, is where a finding's current status is recorded.

---

# Reproduction verdicts — Sweep B computation findings (F-139..F-161, raised-by B-1..B-22 + B-shadow)

Scope: the 23 findings whose `raised-by` is a `B-` id in `candidates/deduped.md`, of which 22
survived the refutation gate (`verdicts/refutation.md`). F-151 (B-13) was REFUTED (`calibration.py:238`
cites ISO 1996-2:2017) and is skipped per instructions. No `SURVIVED-CORRECTED` entries in this
scope — every reproduction below demonstrates the mechanism exactly as the refutation gate verified
it. Every finding here executes real code against directly-constructed inputs; no model is loaded
and nothing is downloaded.

---

### F-139
- outcome: DEMONSTRATED
- script: repro/F-139.py
- observed: `derive_mask_from_axes` with `settled_below=0.35` (bare default, no caller in the
  package ever overrides it, absent from `default.yaml`) turns a bucket at uncertainty 0.34 into a
  `target_free` mask region (confidence 0.66) and a bucket at 0.36 into no region at all — a sharp
  cliff between two qualitatively identical buckets, on a value with no derivation.

### F-140
- outcome: DEMONSTRATED
- script: repro/F-140.py
- observed: `fuse_axes` with `unsettled_above=0.6` (same unfitted-default pattern, no override
  path) offers a bucket to the `remeasure` hook (and counts it toward C4 convergence) when its
  folded uncertainty is ~0.610, but not when it is ~0.584 — two raw signal readings 0.01 apart.

### F-141
- outcome: DEMONSTRATED
- script: repro/F-141.py
- observed: `apply_aggregator([0.9]*5, "disagreement_weighted")` = 0.81 versus
  `apply_aggregator([0.0,0.0,0.0,0.0,1.0], "disagreement_weighted")` = 0.20 — five signals in
  unanimous agreement score 4x *higher* "disagreement" than textbook 4-vs-1 disagreement, confirming
  the aggregator is a level statistic (`mean(u)*max(u)`), not a spread statistic, despite its name.

### F-142
- outcome: DEMONSTRATED (promotion-candidate)
- script: repro/F-142.py
- observed: an AST sweep of `level.py`'s imports shows zero coupling to any `audio_analysis` type
  (only numpy/json/math/pathlib/dataclasses + `senselab.utils` logging); `apply_gain_db`,
  `integrated_lufs`, `loudness_range_lu`, `true_peak_dbtp`, `clipped_fraction`,
  `normalization_gain_db` all ran correctly on a bare synthetic `(waveform, sampling_rate)` pair.

### F-143
- outcome: DEMONSTRATED
- script: repro/F-143.py
- observed: `informative_evidence` with `MIN_LOW_FRACTION=0.02` excludes a signal reporting "no
  speech" in 0 of 50 buckets (fraction 0.0) from the evidence pool, but includes the identical
  signal reporting it in exactly 1 of 50 buckets (fraction 0.02) — one bucket flips corroboration
  eligibility, on a threshold whose only prior numbers the docstring itself disowns.

### F-144
- outcome: DEMONSTRATED
- script: repro/F-144.py
- observed: `speaker_count_posterior` with `multimodal_threshold=0.15` reports `is_multimodal=False`
  at P(3)=0.1453 and `is_multimodal=True` at P(3)=0.1597 — a 0.0144 probability shift flips the
  posterior's `is_multimodal` flag, which gates `converged` in `final/speakers.json`.

### F-145
- outcome: DEMONSTRATED
- script: repro/F-145.py
- observed: `SpeakerHypothesis.has_supported_evidence` reads `False` at `source_support=0.49` and
  `True` at `0.50` — a 0.01 difference in a measured-but-unfitted quantity flips a boolean written
  to `final/speakers.json`, against `_SUPPORTED_THRESHOLD=0.5`'s unexplained midpoint.

### F-146
- outcome: DEMONSTRATED
- script: repro/F-146.py
- observed: `per_speaker_presence` reports `binding_agreement=0.0` identically for two diarizers
  already at capacity elsewhere (`eligible=0`, nothing was actually checked) and for two unbounded
  diarizers that explicitly reject the speaker (`eligible=2`, `bound=0`, genuine unanimous
  disagreement) — "unmeasured" and "measured and rejected" are bit-for-bit indistinguishable.

### F-147
- outcome: DEMONSTRATED (one of the three highest-value reproductions)
- script: repro/F-147.py
- observed: `per_speaker_tracks` reports `speech_presence_confidence=1.0` /
  `speech_presence_uncertainty=0.0` identically whether all 4 diarizers unanimously vote a speaker
  present, or 3 of 4 crashed (absent from `votes`, not "reported silence") and only 1 survivor
  voted — `n_models=len(clusters)` shrinks with the crash instead of registering it, exactly as
  claimed. The sibling `speaker_assignment` voter already carries `n_sources` to guard against this
  same collapse; the fix was not carried to `per_speaker_tracks`.

### F-148
- outcome: DEMONSTRATED (promotion-candidate)
- script: repro/F-148.py
- observed: `statistics.py` imports only `math`/`typing` (confirmed by AST sweep); `confidence`,
  `variability`, `entropy_uncertainty`, `epistemic_uncertainty` all ran correctly on plain
  `list[bool]`/`list[float]`/`dict[str, float]` inputs with zero `Region`/`VoteStore` coupling.

### F-149
- outcome: DEMONSTRATED
- script: repro/F-149.py
- observed: the real `_aggregate_quality` on `PESQ=2.6` (ordinary usable-speech quality, above the
  docstring's own stated "degraded < 2.5" cutoff) returns `pesq_uncertainty=0.6`, which becomes the
  headline `quality.uncertainty` — a "literature-derived" ramp with no citation anywhere in
  `data/`/`specs/` for its 2.0/3.5 anchors flags ordinary speech as 60%-of-the-way to maximally
  uncertain.

### F-150
- outcome: DEMONSTRATED
- script: repro/F-150.py
- observed: `build_disagreements_index` on a run where every axis produced zero rows (total
  harvest/fuse failure) reports `high_uncertainty_rate=0.0` — bit-for-bit the same value a clean
  run with no uncertain rows would report, and `scripts/check_layering.py` prints it directly
  against a stored 0.9941 baseline, reading a broken run as a dramatic improvement.

### F-152
- outcome: DEMONSTRATED (promotion-candidate)
- script: repro/F-152.py
- observed: `acoustic.py` imports only `math`/`numpy` (AST-confirmed); `lufs_track` and
  `level_above_floor_track` both ran correctly on a bare synthetic `(waveform, sampling_rate)` pair.

### F-153
- outcome: DEMONSTRATED (promotion-candidate)
- script: repro/F-153.py
- observed: `_union_length`'s own parameter type is plain `list[tuple[float, float]]` (no
  `Span`/`Spans`) and it computes the correct union length (2.5) on bare tuples; `occupancy`'s
  parameter list is the only thing in the file typed against `Spans` — confirming the generic core
  is separable from its workflow-typed wrapper exactly as claimed.

### F-154
- outcome: DEMONSTRATED (one of the three highest-value reproductions)
- script: repro/F-154.py
- observed: a belief-state bucket at doubt=0.40 (strictly between `theta_low=0.33` and
  `theta_high=0.66`) is confirmed to seed no region at all via the real `propose_regions`. Driving
  that state through the real `build_convergence_report` with `run_state="converged"` (as
  `loop.py:315-317` sets it when nothing fired and nothing was `not_admitted`) yields
  `report["converged"] = True` at the top level, while the same document's own
  `report["per_axis"]["speaker"] = {"buckets": 1, "open": 1, "residual_mass": 0.07}` — a
  self-contradicting headline verdict.

### F-155
- outcome: DEMONSTRATED
- script: repro/F-155.py
- observed: a vote entry under key `identity_repair_i1` carrying `change_point_times`/
  `change_point_confidence` (I1's real payload schema) is absent from the real
  `per_signal_uncertainty`'s output (`fuse_axis` never scores it), yet is present in
  `contributing_sources` computed exactly as `belief.py:818` computes it (every vote-source key
  present, regardless of whether it was scored) — confirming the schema mismatch lets an inert
  vote masquerade as a contributor.

### F-156
- outcome: DEMONSTRATED
- script: repro/F-156.py
- observed: the real `detect_change_points` on a trajectory with one strong and one genuinely weak
  change-point returns the weak one at `confidence=0.0333`. Applying `identity_repair.py:227-231`'s
  own `cp_conf.get(round(seg["start"], 4), 0.5)` expression to a segment edge with no matching
  detected change-point yields the fabricated fallback `0.5` — higher than the real, weak,
  measured detection, misordering true vs. fabricated confidence.

### F-157
- outcome: DEMONSTRATED
- script: repro/F-157.py
- observed: the real `_p2_trigger` fires (`fires=True`, `reason="frame_dispersion"`) on a region
  with `coarse_share=0.0` (nowhere near the 0.5 threshold) purely because `mean_instability=1e-9 >
  0.0` — confirming the `> 0.0` gate (rather than a graded high-value threshold, as the function's
  own docstring describes) fires on essentially any nonzero real-valued frame posterior.

### F-158
- outcome: DEMONSTRATED
- script: repro/F-158.py
- observed: the three real gain functions on comparable inputs — `_n_candidates_gain` (P3, light
  cost, 50 candidates) → priority 50.0; `_mass_gain` (a genuinely contested speaker region,
  medium cost, mass 0.8) → priority 0.2; `_u2_gain` (equal mass 0.8, epistemic 0.5, medium cost,
  x10 multiplier) → priority 1.0. P3's priority is 250x the contested speaker region's despite an
  arguably smaller true value, and U2 outranks an equal-mass speaker candidate 5x — confirming no
  shared normalization exists across gain families sorted in one list.

### F-159
- outcome: DEMONSTRATED (one of the three highest-value reproductions)
- script: repro/F-159.py
- observed: the real `classify_resolution(before=0.50, after=0.30, was_revised=True,
  independent_evidence=False)` correctly returns `"revision"` (self-confirmation, not a confidence
  gain). An AST sweep confirms zero call sites of `classify_resolution`/`RevisionRecord` anywhere
  in `adaptive/*.py` outside `provenance.py` itself. Feeding the identical 0.50→0.30 drop through
  the real `apply_convergence_marks` marks the bucket `status="converged"` outright — the
  strongest form of credited progress — because that function has no `was_revised`/
  `independent_evidence` input to consult at all.

### F-160
- outcome: DEMONSTRATED (promotion-candidate)
- script: repro/F-160.py
- observed: `identity_repair.py`'s only import from `audio_analysis` is a bare constant
  (`floors.MIN_EVIDENCE_WEIGHT`), confirmed by AST sweep; `_l2`, `_agglomerative_cosine`, and
  `change_point_trajectory` all ran correctly on plain numpy arrays / `list[dict]` with generic
  `vector`/`start_s`/`end_s` keys, with zero `Region`/`VoteStore`/policy coupling anywhere in the
  file.

### F-161
- outcome: DEMONSTRATED
- script: repro/F-161.py
- observed: `python -c "import ast"` run with cwd set to
  `src/senselab/audio/workflows/audio_analysis/` fails with
  `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types'`,
  naming this package's own `types.py` and blaming `weakref` in the traceback rather than the real
  cause — directly reproducing the shadowing (matches the mechanism already confirmed live in
  `verdicts/refutation.md`'s F-161 entry).

---

Summary: 22/22 DEMONSTRATED, 0 LATENT. Five of these (F-142, F-148, F-152, F-153, F-160) are
promotion-candidates demonstrated by AST import/signature sweep plus a live run on plain-typed
inputs, per the task's own bar for that class. F-151 (B-13) is out of scope, REFUTED per
`verdicts/refutation.md`.
