# Deduped findings — analyze_audio audit

Merged from `sweep-a-prose.md` (138 raw: 8 stale-or-false, 96 rationale-to-migrate, 34
restates-code), `sweep-b-computation.md` (23 raw: B-1..B-22 + B-shadow),
`sweep-c-orchestration.md` (2 raw: C-1, C-2), `sweep-d-assumptions.md` (13 raw: D-1..D-13).
Raw total: 176.

Two reclassifications were applied per reviewer instruction, not counted as merges:
- `A-122` (pii.py:261-268) and `A-111` (disagreements.py:23-29) are re-filed as
  `rationale-to-migrate` (each carries load-bearing rationale — a rejected-alternative design
  note and an L1/L2 separation principle, respectively) rather than `restates-code` as
  sweep-a-prose.md labeled them.
- `B-5`, `B-8`, `B-16`, `B-19` are emitted once each, sourced from their `B-` id only (Sweep D
  deliberately excluded them from its assumption population as population-neutral — no
  age/task-specific angle — rather than re-reporting them; see each entry's note).

After conservative merge review (detailed in "Cross-sweep patterns" below), **no two raw
candidates were found to share both a location and a fix** — the sweeps' own extensive
cross-checking against each other (Sweep B against A, Sweep C against A+B, Sweep D against B)
already absorbed the true duplicates before this pass. Several same-file, same-topic pairs
remain deliberately **separate** because they need different fixes (e.g. `A-4`/`A-98` at the
same `_fused_axis` location: one is a stale sentence to correct, the other is a delete-this-
scaffolding note to preserve). Merged count: **176** (0 merged away). See individual notes below
for the "possible duplicate, kept separate" calls.

---

## Layer: prose (Sweep A)

### F-1
- raised-by: A-1
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/__init__.py:1-6
- defect: stale-or-false — top-level docstring claims three uncertainty axes and a 5-row timeline; `axes.py`'s own docstring documents this exact count as the already-fixed defect ("any list of three axes is wrong"), and `AXES` now has four members (`background_mask`).
- failure: a reader of `help(audio_analysis)` or rendered docs does not look for `background_mask` in `compute_uncertainty_axes`'s output, `estimates/*.parquet`, or the timeline PNG.

### F-2
- raised-by: A-2
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/io.py:150
- defect: stale-or-false — docstring path `L2/round0/votes/<axis>.parquet` omits the real `derivatives/` segment and misspells `round/0` as `round0` (verified against `scripts/analyze_audio.py:830-833` and `contracts.py:566,842`).
- failure: a reader locating the linked-votes file from this docstring looks in (or declares a contract for) `L2/round0/votes/`, which does not exist.

### F-3
- raised-by: A-3
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/stage_context.py:91-94
- defect: stale-or-false — justifies `STAGE_VERSIONS` bump criteria by "wrapper-shaped output changes... classifiers attach phoneme labels," but no stage owning a `STAGE_VERSIONS` entry attaches phoneme labels; `ast`/`yamnet` do AudioSet scene classification only.
- failure: a contributor deciding whether to bump `STAGE_VERSIONS["ast"]`/`["yamnet"]` is pointed at the wrong justification and may skip the bump because the real change doesn't match the doc's description.

### F-4
- raised-by: A-4
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:718-719 (`_fused_axis`)
- defect: stale-or-false — states "the belief store ingests L1's per-pass axis folds"; `belief.py`'s own `VoteStore.from_run_dir` docstring says that path was removed and the store now ingests `L2/round/0/derivatives/votes/<axis>.parquet`.
- failure: a reader trying to explain why the belief-store line and the L2-fused overlay differ goes looking for a nonexistent L1 per-pass-axis-fold file instead of the real (simpler) reason.
- note: possible duplicate of F-98 (A-98, same function, remainder of the same paragraph) — kept separate per the prompt's own example: a stale sentence and a "delete this scaffolding" note need different fixes.

### F-5
- raised-by: A-5
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/provenance.py:1-22
- defect: stale-or-false — module docstring claims every state change in the loop is attributable per FR-011g via `RevisionRecord`/`classify_resolution`; repo-wide grep finds zero call sites for either in `loop.py`, `interventions.py`, or `belief.py`.
- failure: a reader believes every revision in a run is audit-traceable and does not check whether the mechanism is reachable — it is not, so nothing in a real run is attributed by it.
- note: possible duplicate of F-171 (B-21), which reports the live consequence (convergence math cannot distinguish genuine-evidence drops from self-confirming re-scores) of this same dead module — kept separate because fixing the docstring (F-5) does not by itself fix the missing convergence-math distinction (F-171); wiring `provenance.py` in for real would fix both.

### F-6
- raised-by: A-6
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/loop.py:3-6
- defect: stale-or-false — docstring says round 1 is always the ingested analyze_audio run and omits the in-process ingest path; contradicted by `_baseline_round`'s own docstring 730 lines later, which names "round 1 vs round 0" as the exact bug this design fixed.
- failure: a reader assumes fixed round numbering and disk-only ingestion, missing that the baseline round varies per run and that `run_adaptive_loop(run_dir, harvests=..., summary=...)` is a fully supported in-memory path — the one `analyze_audio.py` actually uses.

### F-7
- raised-by: A-7
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:22-24
- defect: stale-or-false — module docstring lists `P2_fine_posteriors` as "still deferred," but `_p2_trigger`/`_p2_guard`/`_p2_execute` are fully implemented and registered in the `RULES` table.
- failure: a reader believes P2 never fires and skips auditing its working coarse-dominance/frame-instability logic, or re-implements a rule that already exists.

### F-8
- raised-by: A-8
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:19-20
- defect: stale-or-false — describes `I4_overlap_detection` as using "segmentation-3.0 per-class posteriors," but `_i4_execute` calls `backends.overlap_track_from_spans`, which derives overlap from cross-diarizer spans instead (confirmed by that function's own docstring and a nearby comment in the same file).
- failure: a reader auditing I4 looks for a segmentation-3.0 per-class posterior extraction inside `_i4_execute`; it isn't there.

### F-9
- raised-by: A-9
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/grid.py:15-19
- defect: rationale-to-migrate — measured evidence (242/242/19/8 rows, zero shared bucket keys across four resolutions) for why one shared grid replaced four independent ones; destination: grid/fuse design (doc.md "one grid" section).
- failure: UNVERIFIED — documentation-placement finding, not a runtime defect.

### F-10
- raised-by: A-10
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/fuse.py:88-101 (`is_direction_only_claim`)
- defect: rationale-to-migrate — measured cost of the vote-folding fix (presence axis fusing 12 vs 8 signals depending on ASR set); destination: grid/fuse design (vote-folding).
- failure: UNVERIFIED.

### F-11
- raised-by: A-11
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/axes.py:281-334 (`IDENTITY_ONLY_AXES`)
- defect: rationale-to-migrate — measured 5x enhanced-vs-raw `words` voter reading; destination: per-speaker-identity-scene design (layered-architecture.md).
- failure: UNVERIFIED.

### F-12
- raised-by: A-12
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/stage_context.py:202-243 (`_commit_sha_for`)
- defect: rationale-to-migrate — three-outcome commit-resolution design rationale; destination: cache/provenance design (commit-SHA pinning rules).
- failure: UNVERIFIED.

### F-13
- raised-by: A-13
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/run_config.py:9-14
- defect: rationale-to-migrate — the "seventy flags, zero shared bucket keys" measurement behind the no-per-knob-flags design; destination: run-config design.
- failure: UNVERIFIED.

### F-14
- raised-by: A-14
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/contracts.py:1-58
- defect: rationale-to-migrate — "enumerating what is forbidden cannot terminate" rationale for the declare-what-is-permitted contracts design; destination: stage-contracts / D-17 summary.
- failure: UNVERIFIED.

### F-15
- raised-by: A-15
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/resolution.py:1-24
- defect: rationale-to-migrate — measured VAD saturation from collapsing 17ms frames onto 250ms buckets; destination: grid/fuse design (per-signal resolution).
- failure: UNVERIFIED.

### F-16
- raised-by: A-16
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/asr.py:1-27
- defect: rationale-to-migrate — history of four removed per-bucket ASR quantities; destination: asr axis design.
- failure: UNVERIFIED.
- note: possible duplicate of F-18/F-19 (A-18/A-19) — same historical narrative told a third and fourth time in the same file at different line ranges; kept separate (different code locations) but flagged in Cross-sweep patterns.

### F-17
- raised-by: A-17
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/asr.py:100-115 (`phoneme_similarity`)
- defect: rationale-to-migrate — g2p-fallback rationale ("letters are not sounds"); destination: asr axis design (grading/g2p).
- failure: UNVERIFIED.

### F-18
- raised-by: A-18
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/asr.py:293-320 (`resample_member_doubt`)
- defect: rationale-to-migrate — restates the module docstring's epistemic-uncertainty-was-zero framing; destination: asr axis design (consolidate with F-16).
- failure: UNVERIFIED.

### F-19
- raised-by: A-19
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/asr.py:426-456 (`harvest_asr_votes`)
- defect: rationale-to-migrate — third copy of the same consensus_words history in one file; destination: asr axis design (consolidate with F-16/F-18).
- failure: UNVERIFIED.

### F-20
- raised-by: A-20
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speaker.py:139-153 (`harvest_speaker_votes`)
- defect: rationale-to-migrate — measured "same-speaker-as-before" gate replacement (0.666 vs 0.168 doubt); destination: speaker attribution design (speaker-axis-attribution-design.md, already cross-referenced).
- failure: UNVERIFIED.

### F-21
- raised-by: A-21
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speaker_identity.py:1-25
- defect: rationale-to-migrate — validation-recording anecdote motivating embedding-clustering as a synthetic diarizer; destination: speaker attribution / per-speaker-identity design.
- failure: UNVERIFIED.

### F-22
- raised-by: A-22
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speaker_identity.py:300-308 (`source_kind_for`)
- defect: rationale-to-migrate — "5 speakers vs 2" anecdote, duplicated near-verbatim in influence.py, support.py, reliability.py (4 copies total); destination: influence/support/reliability weighting design (one canonical home removes three repeats).
- failure: UNVERIFIED.

### F-23
- raised-by: A-23
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/identity_binding.py:1-19
- defect: rationale-to-migrate — the three-id-namespace ("all once rendered as S0") rationale, near-verbatim repeated in harmonize.py and clustering.py (3 copies); destination: layered-architecture.md (D-19).
- failure: UNVERIFIED.

### F-24
- raised-by: A-24
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/embeddings.py:396-407
- defect: rationale-to-migrate — comment on deleted p_voice computation ("a silhouette coefficient is not a probability"); destination: l1-post-processing-register.md item 12, but the comment has no anchor once the dead code it sits on is deleted for real.
- failure: UNVERIFIED.
- note: possible duplicate of F-52 (A-52, speech_presence_link.py) — same silhouette-voter incident narrated at a second location with the same destination anchor; kept separate because each is a distinct comment in a distinct file that must be migrated/removed independently.

### F-25
- raised-by: A-25
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/clustering.py:104-133 (`assign_unified_clusters_with_seed_phase`)
- defect: rationale-to-migrate — two-threshold derivation (cross_group 0.75 vs cosine 0.5); destination: clustering/statistics design, beside calibration.py's derivation blocks.
- failure: UNVERIFIED.

### F-26
- raised-by: A-26
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/harmonize.py:1-23
- defect: rationale-to-migrate — "cross-model statement first guesses same person" framing; destination: layered-architecture.md (D-6).
- failure: UNVERIFIED.

### F-27
- raised-by: A-27
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/invariance.py:1-26
- defect: rationale-to-migrate — gain-scaling/background-detection cross-reference; destination: background-scene design (amplification finding).
- failure: UNVERIFIED.

### F-28
- raised-by: A-28
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/joint.py:1-29
- defect: rationale-to-migrate — "J1/J4 have moved" history, duplicating identity_binding.py's "What changes from J4" section; destination: layered-architecture.md (D-19/D-7).
- failure: UNVERIFIED.
- note: possible duplicate of F-23 (A-23) — same J4/segmentation-3.0-channel-ordering narrative; kept separate (different files, both need independent migration).

### F-29
- raised-by: A-29
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/statistics.py:1-36
- defect: rationale-to-migrate — "all called 'uncertainty'" naming-collision history; destination: layered-architecture.md or an estimator-taxonomy note.
- failure: UNVERIFIED.

### F-30
- raised-by: A-30
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/measurements.py:1-16
- defect: rationale-to-migrate — "frame_mean at a resolution the model never reported" / units:"mixed" history; destination: layered-architecture.md (D-18).
- failure: UNVERIFIED.

### F-31
- raised-by: A-31
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/support.py:276-298 (`MIN_LOW_FRACTION`)
- defect: rationale-to-migrate, flagged for content risk — the docstring cites specific measured numbers (503/697, 601/697, 0.500, 0.897) that it simultaneously disowns as taken under a since-fixed reading bug ("must be re-measured before they are cited again"); destination: support/reliability design, migrated with the numbers replaced or dropped, not carried forward as-is.
- failure: UNVERIFIED as a runtime defect (the constant's own live threshold behavior is F-135/B-5); this entry is specifically about the docstring citing numbers it disowns.

### F-32
- raised-by: A-32
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/influence.py:1-30
- defect: rationale-to-migrate — "pseudo-diarizer agreeing with itself is not corroboration" rule, duplicated in asr.py/speaker.py; destination: layered-architecture.md (D-21 rule 6).
- failure: UNVERIFIED.

### F-33
- raised-by: A-33
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/calibration.py:1-32
- defect: rationale-to-migrate — declared-and-unread field history (`temperature`, `token_entropy_reference_nats`); destination: layered-architecture.md or l1-post-processing-register.md.
- failure: UNVERIFIED.

### F-34
- raised-by: A-34
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/degradation.py:1-18
- defect: rationale-to-migrate — measured "clip((25-snr_db)/20,0,1) returns 0.0 in every bucket of every recording" L1/L2 boundary rationale; destination: layered-architecture.md (L1/L2 calibration boundary).
- failure: UNVERIFIED.
- note: same file as F-172 (D-6), which is a distinct (assumption-class) finding about the same anchor lacking task-conditioning — different fix, kept separate.

### F-35
- raised-by: A-35
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/reliability.py:1-22
- defect: rationale-to-migrate — "saturated embedding check outvoted unanimous diarizer agreement" incident, third independent telling alongside speaker.py/embeddings.py; destination: speaker attribution / clustering-statistics design.
- failure: UNVERIFIED.

### F-36
- raised-by: A-36
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:1-24
- defect: rationale-to-migrate — mask-semantics rationale (30 dB suppression baseline leaves residual foreground dominant); destination: background-scene design (mask semantics).
- failure: UNVERIFIED.

### F-37
- raised-by: A-37
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:244-255 (`_classify_bucket`)
- defect: rationale-to-migrate — measured 0.99/0.99 confidence-and-uncertainty collision producing one whole-file region; destination: background-scene design (mask classification).
- failure: UNVERIFIED.

### F-38
- raised-by: A-38
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:462-477 (`_speech_activity_by_bucket`)
- defect: rationale-to-migrate — boolean-collapse-of-1070-buckets rationale; destination: background-scene design (mask evidence).
- failure: UNVERIFIED.

### F-39
- raised-by: A-39
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/noise_floor.py:1-32
- defect: rationale-to-migrate — quantile-of-exponential-noise derivation (9.8 dB at p10) and explicit "no published precedent" caveat; destination: background-scene design (noise-floor estimation).
- failure: UNVERIFIED.

### F-40
- raised-by: A-40
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/noise_floor.py:165-197 (`estimate_band_floor_db`)
- defect: rationale-to-migrate — "p10+6dB cut discards two-thirds of exponential noise" derivation; destination: background-scene design (noise-floor estimation).
- failure: UNVERIFIED.

### F-41
- raised-by: A-41
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/noise_floor.py:376-410
- defect: rationale-to-migrate — ECMA-74/ISO 7779 prominence-ratio derivation; destination: background-scene design (stationary source detection).
- failure: UNVERIFIED.

### F-42
- raised-by: A-42
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sources.py:1-27
- defect: rationale-to-migrate — "amplifying a noise floor produces plausible fake environmental labels" rationale; destination: background-scene design (source detection guards).
- failure: UNVERIFIED.

### F-43
- raised-by: A-43
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sources.py:245-256
- defect: rationale-to-migrate — measured excision-vs-mixed-window comparison (0.705 vs 0.548); destination: background-scene design (excision routing).
- failure: UNVERIFIED.

### F-44
- raised-by: A-44
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/foreground.py:1-24
- defect: rationale-to-migrate — oracle-experiment rationale (30 dB suppression made present/absent background indistinguishable); destination: background-scene design (foreground suppression).
- failure: UNVERIFIED.
- note: possible duplicate of F-45 (A-45) — same oracle-experiment cited twice in one file; kept separate per source's own note.

### F-45
- raised-by: A-45
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/foreground.py:121-128 (`is_deep_enough_for`)
- defect: rationale-to-migrate — same oracle-experiment rationale reused for the depth-below-foreground comparison; destination: background-scene design (foreground suppression, consolidate with F-44).
- failure: UNVERIFIED.

### F-46
- raised-by: A-46
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sound_sources.py:31-48 (`AUDIOSET_SCORE_FUNCTION`)
- defect: rationale-to-migrate — softmax-vs-sigmoid class-competition rationale; destination: background-scene design (sound-source categorization).
- failure: UNVERIFIED.

### F-47
- raised-by: A-47
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sound_sources.py:90-106 (`window_label_mass`)
- defect: rationale-to-migrate — top-1-discards-evidence example (Music 0.40 / Speech 0.38); destination: speech-presence design (label mass vs top-1), duplicated verbatim in speech_presence.py.
- failure: UNVERIFIED.
- note: near-duplicate of F-48 (A-48) — same worked example in two files; kept separate, consolidate at destination.

### F-48
- raised-by: A-48
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speech_presence.py:1-41
- defect: rationale-to-migrate — same Music/Speech top-1 example as F-47; destination: speech-presence design (L1 evidence), consolidate with F-47.
- failure: UNVERIFIED.

### F-49
- raised-by: A-49
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:1-42
- defect: rationale-to-migrate — Jensen's-inequality argument for why two ASR confidence statistics differ; destination: speech-presence design (L1/L2 split).
- failure: UNVERIFIED.

### F-50
- raised-by: A-50
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:144-176 (`_abstaining_ramp`)
- defect: rationale-to-migrate — measured acoustic_hnr abstention behavior (mean 0.2675 doubt vs 0.0 elsewhere); destination: speech-presence design (signal abstention).
- failure: UNVERIFIED.

### F-51
- raised-by: A-51
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:327-349
- defect: rationale-to-migrate — removed `_link_hnr` banner, measured 8.12 dB median HNR below the "confidently voiced" anchor; destination: l1-post-processing-register.md item 10.
- failure: UNVERIFIED.

### F-52
- raised-by: A-52
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:415-445
- defect: rationale-to-migrate — removed `_silhouette_votes_by_bucket` banner, measured weight 1.0 (highest of 15) on the least informative voter; destination: l1-post-processing-register.md item 12.
- failure: UNVERIFIED.
- note: see F-24 duplicate note.

### F-53
- raised-by: A-53
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/quality.py:1-36
- defect: rationale-to-migrate — "both returned 0.0 in every bucket measured" L1/L2 boundary rationale; destination: quality/degradation design.
- failure: UNVERIFIED.

### F-54
- raised-by: A-54
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/quality.py:326-352 (`quality_series`)
- defect: rationale-to-migrate — units:"mixed" honesty and overlapping-window independence caveat; destination: quality/degradation design (D-20/D-25).
- failure: UNVERIFIED.

### F-55
- raised-by: A-55
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/pii.py:19-25
- defect: rationale-to-migrate — measured near-zero true-positive rate for Presidio's most-severe categories in pediatric/clinical voice data; destination: PII detection design.
- failure: UNVERIFIED.

### F-56
- raised-by: A-56
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/mask_harvest.py:1-24
- defect: rationale-to-migrate — "uncertainty was a property of there being one producer, not of the mask" rationale; destination: background-scene design (D-22).
- failure: UNVERIFIED.

### F-57
- raised-by: A-57
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/mask_harvest.py:37-52 (`TARGET_POLARITY`)
- defect: rationale-to-migrate — task-gated polarity rationale (breathing task: speech vote means target absence); destination: background-scene design (task-gated polarity).
- failure: UNVERIFIED.
- note: this is the same task-vocabulary gap D-5 (F-175) later reports from an assumption angle — kept separate (F-57 is prose rationale for existing polarity logic; F-175/D-5 is a gap in the task→target-event vocabulary the logic depends on).

### F-58
- raised-by: A-58
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/occupancy.py:1-22
- defect: rationale-to-migrate — "honest uncertainty is disagreement across models, not one model's confidence" rationale; destination: speaker/occupancy design (D-19).
- failure: UNVERIFIED.

### F-59
- raised-by: A-59
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/occupancy.py:68-79 (`capacity_for`)
- defect: rationale-to-migrate — "raising instead of the current design was tried and is wrong at this depth" rationale; destination: speaker/occupancy design (D-19).
- failure: UNVERIFIED.

### F-60
- raised-by: A-60
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/shapes.py:1-31
- defect: rationale-to-migrate — "forcing shapes through one tabular row" reduction-catalogue rationale; destination: L1 shapes / derivative design (D-18).
- failure: UNVERIFIED.

### F-61
- raised-by: A-61
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/shapes.py:148-159 (`Matrix`)
- defect: rationale-to-migrate — measured "1.0000 in 100% of frames on a half-silent clip" pooled-value example; destination: L1 shapes / derivative design.
- failure: UNVERIFIED.

### F-62
- raised-by: A-62
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/perturbations.py:1-26, 76-102
- defect: rationale-to-migrate — measured raw-vs-enhanced speaker-axis divergence (0.0 vs 0.398, averaging to a false 0.227); destination: perturbations/passes design (D-17).
- failure: UNVERIFIED.

### F-63
- raised-by: A-63
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sampler.py:1-27
- defect: rationale-to-migrate — measured provenance-describes-a-measurement-the-file-lacks example; destination: L2 derivative/sampler design (D-25).
- failure: UNVERIFIED.

### F-64
- raised-by: A-64
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/rounds.py:1-24
- defect: rationale-to-migrate — "regional trust attenuates the wrong claim without silencing the right ones" rationale, with a 5-speaker/4.9s worked example; destination: L2 fusion/rounds design.
- failure: UNVERIFIED.

### F-65
- raised-by: A-65
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/rounds.py:143-157
- defect: rationale-to-migrate — cycle-detection window derivation (p+1 rounds to detect a period-p cycle); destination: L2 fusion/rounds design (D-12).
- failure: UNVERIFIED.

### F-66
- raised-by: A-66
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/global_summary.py:52-59 (`PASS_FOLD`)
- defect: rationale-to-migrate — "not a minimum: raw/enhanced disagreement is evidence" rationale; destination: run summary/global aggregation design.
- failure: UNVERIFIED.

### F-67
- raised-by: A-67
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/summary.py:1-18
- defect: rationale-to-migrate — "not-measured treated as zero overstates certainty" rationale; destination: run summary/reporting design.
- failure: UNVERIFIED.

### F-68
- raised-by: A-68
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/plot.py:1-42
- defect: rationale-to-migrate — "a default argument decided the layer" naming/layering incident; destination: plotting/layering design.
- failure: UNVERIFIED.

### F-69
- raised-by: A-69
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/plot.py:270-278 (`_load_background_mask_rows`)
- defect: rationale-to-migrate — "written against the flat layout, matched nothing once passes moved under L1/" history; destination: plotting design/layout history.
- failure: UNVERIFIED.

### F-70
- raised-by: A-70
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/l1_plot.py:1-20
- defect: rationale-to-migrate — "diarizer stopped here / level fell here, neither alone tells the story" rationale; destination: plotting design (L1 evidence figure).
- failure: UNVERIFIED.

### F-71
- raised-by: A-71
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/l1_plot.py:171-196
- defect: rationale-to-migrate — signal-grouping/row-height design rationale (alphabetical order was unreadable); destination: plotting design (L1 evidence figure).
- failure: UNVERIFIED.

### F-72
- raised-by: A-72
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/l2_plot.py:1-13
- defect: rationale-to-migrate — "replaces mostly-empty chunked timeline PNGs" rationale; destination: plotting design (L2 round timeline).
- failure: UNVERIFIED.

### F-73
- raised-by: A-73
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/labelstudio.py:1-17
- defect: rationale-to-migrate — removed TextArea/coarse-grid history; destination: labelstudio/export design.
- failure: UNVERIFIED.

### F-74
- raised-by: A-74
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/labelstudio.py:652-667
- defect: rationale-to-migrate — "per-speaker presence labelled by speaker, not merged" rationale; destination: labelstudio/export design.
- failure: UNVERIFIED.

### F-75
- raised-by: A-75
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/__init__.py:5-9
- defect: rationale-to-migrate — lazy-import-strategy rationale (no torch/model backends at module level); destination: adaptive loop design (import/dependency strategy).
- failure: UNVERIFIED.

### F-76
- raised-by: A-76
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/audio_io.py:110-121
- defect: rationale-to-migrate — "used to hardcode two pass names" history; destination: adaptive loop design (audio_io/perturbation dispatch).
- failure: UNVERIFIED.

### F-77
- raised-by: A-77
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:191-193
- defect: rationale-to-migrate — stage-once/load-from-local-snapshot rationale to avoid per-file Hub HEAD (429 source under batch); destination: adaptive loop design (model loading/caching).
- failure: UNVERIFIED.

### F-78
- raised-by: A-78
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:200-213
- defect: rationale-to-migrate — "what is lost" reduction disclosure for segmentation-3.0; destination: policy/triage design (P2 rationale).
- failure: UNVERIFIED.

### F-79
- raised-by: A-79
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:296-310
- defect: rationale-to-migrate — "hardcoded to MMS_FA, D-1 moved Canary off MMS" history; destination: belief/fusion design (consensus alignment).
- failure: UNVERIFIED.

### F-80
- raised-by: A-80
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/types.py:3-23
- defect: rationale-to-migrate — TypedDict-vs-dataclass design rationale; destination: types/data-model design.
- failure: UNVERIFIED.

### F-81
- raised-by: A-81
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:65-74
- defect: rationale-to-migrate — "flag said background_mask was harvested, method enumerated three axes" mismatch history; destination: belief/fusion design.
- failure: UNVERIFIED.

### F-82
- raised-by: A-82
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:206-221
- defect: rationale-to-migrate — measured SNR-gate scope gap (gate reached round 0 only, final/ published 0.2267 vs round 0's 0.0487); destination: belief/fusion design (SNR gating).
- failure: UNVERIFIED.

### F-83
- raised-by: A-83
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:1108-1125
- defect: rationale-to-migrate — "every lookup missed, floor assigned 0.0 everywhere — the confident claim of no floor" incident; destination: belief/fusion design (aleatoric floor).
- failure: UNVERIFIED.
- note: same "0.0 conflates absence-of-measurement with a real zero" pattern as F-140 (B-8), F-141 (B-9), F-147 (B-12), F-148 (B-18) — see Cross-sweep patterns.

### F-84
- raised-by: A-84
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/corroboration.py:1-21
- defect: rationale-to-migrate — measured `acoustic_loudness`/`ast` corroboration pinning near 1.0 under max-pooling; destination: corroboration/presence design.
- failure: UNVERIFIED.

### F-85
- raised-by: A-85
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/evaluate.py:73-77
- defect: rationale-to-migrate — "used to reach into L2/ for intermediates, scoring a scorer" history; destination: evaluation design (L1/L2/final boundary).
- failure: UNVERIFIED.

### F-86
- raised-by: A-86
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:35-38
- defect: rationale-to-migrate — "dropped every word overlapping a P3-adjudicated span" history; destination: belief/fusion design (transcript fusion).
- failure: UNVERIFIED.

### F-87
- raised-by: A-87
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:378-389
- defect: rationale-to-migrate — "deliverable presence track used to be rebuilt here, diverging from the round's belief" history; destination: final-outputs design.
- failure: UNVERIFIED.

### F-88
- raised-by: A-88
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:430-438
- defect: rationale-to-migrate — "both written to L2 root instead of final/" history; destination: final-outputs design.
- failure: UNVERIFIED.

### F-89
- raised-by: A-89
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:35-43
- defect: rationale-to-migrate — "two bare 0.05 literals" naming/derivation history; destination: identity-repair design.
- failure: UNVERIFIED.

### F-90
- raised-by: A-90
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:54-62
- defect: rationale-to-migrate — "returned {} on every run, silently, once outputs moved under L1/" history; destination: adaptive loop design (artifact access).
- failure: UNVERIFIED.

### F-91
- raised-by: A-91
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:266-273
- defect: rationale-to-migrate — "used to concatenate per-bucket text, forcing the axis onto a 1.0s grid" history; destination: policy/triage design (S1 stream election).
- failure: UNVERIFIED.

### F-92
- raised-by: A-92
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:582-591
- defect: rationale-to-migrate — "environment without g2p_en silently measured a different quantity under the same column name" history; destination: policy/triage design (U1/U2 escalation).
- failure: UNVERIFIED.

### F-93
- raised-by: A-93
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:866-877
- defect: rationale-to-migrate — measured "published axis 0.288→0.608 while deliverable stayed 0.1196" gap that the attribution axis exists to remove; destination: identity-repair design.
- failure: UNVERIFIED.

### F-94
- raised-by: A-94
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/loop.py:583-593
- defect: rationale-to-migrate — "root inferred from path shape, fragile, works only for the default layout" caveat; destination: adaptive loop design.
- failure: UNVERIFIED.

### F-95
- raised-by: A-95
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/ls_final.py:75-80
- defect: rationale-to-migrate — "read final/ back out of the directory it was about to write" history; destination: LS-export design.
- failure: UNVERIFIED.

### F-96
- raised-by: A-96
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/ls_final.py:217-225
- defect: rationale-to-migrate — "three answers from one file, only one written down" history; destination: belief/fusion design.
- failure: UNVERIFIED.

### F-97
- raised-by: A-97
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:24-29
- defect: rationale-to-migrate — measured flat-vs-varying mask-derivative figure discrepancy; destination: visualization design.
- failure: UNVERIFIED.

### F-98
- raised-by: A-98
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:716-721 (`_fused_axis`)
- defect: rationale-to-migrate, self-marked deletion note — "this function is scaffolding for a defect, should be deleted rather than maintained"; destination: visualization design / belief-store cleanup backlog, worth preserving even after F-4's stale sentence is fixed.
- failure: UNVERIFIED.
- note: see F-4 duplicate note (same location, different fix).

### F-99
- raised-by: A-99
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:20-25
- defect: rationale-to-migrate — "configuration spread across a file and seventy flags" history; destination: policy/triage design.
- failure: UNVERIFIED.

### F-100
- raised-by: A-100
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:74-79
- defect: rationale-to-migrate — "a floor configurable to zero is not a floor" rationale for raising rather than clamping; destination: policy/triage design.
- failure: UNVERIFIED.

### F-101
- raised-by: A-101
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/provenance.py:6-16
- defect: rationale-to-migrate — "uncertainty can fall for two different reasons, indistinguishable in the number alone" rationale, valuable independent of F-5's wiring gap; destination: provenance/mutual-influence design.
- failure: UNVERIFIED.

### F-102
- raised-by: A-102
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/regions.py:20-24
- defect: rationale-to-migrate — "per-(pass,axis) proposal produced two overlapping regions for one ambiguity" history; destination: region-proposal design.
- failure: UNVERIFIED.

### F-103
- raised-by: A-103
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/triage.py:1-13
- defect: rationale-to-migrate — "continuous frame posteriors, never segmentized VAD" design rationale; destination: policy/triage design.
- failure: UNVERIFIED.

### F-104
- raised-by: A-104
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/acoustic.py (module docstring area)
- defect: rationale-to-migrate — LUFS-vs-percentile loudness rationale, sampled rather than quoted at length; destination: quality/degradation design (loudness measurement).
- failure: UNVERIFIED.

### F-105
- raised-by: A-122 (reclassified per reviewer instruction — was labeled restates-code in sweep-a-prose.md)
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/pii.py:261-268 (`report_to_dict`)
- defect: rationale-to-migrate — rejected-alternative design note explaining why a redundant per-span `perturbation` field was not carried, not a mechanical restatement of the dict comprehension below it; destination: PII detection design (audio_analysis adapter), alongside F-55.
- failure: UNVERIFIED — load-bearing rationale that must not be treated as a mechanical-deletion candidate.

### F-106
- raised-by: A-111 (reclassified per reviewer instruction — was labeled restates-code in sweep-a-prose.md)
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/disagreements.py:23-29 (`_row_summary`)
- defect: rationale-to-migrate — states an L1/L2 separation principle (fused row = how much/which signals carried doubt; evidence = the L1 per-signal measurement for the same bucket), not a mechanical restatement of the two-dict concatenation below it; destination: layered-architecture.md (L1/L2 boundary register).
- failure: UNVERIFIED — load-bearing rationale; could be shortened but must not be deleted outright as a "restates-code" item.

## Layer: prose — restates-code (Sweep A, remaining after F-105/F-106 reclassification)

### F-107
- raised-by: A-105
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/grid.py:24-31 (`BucketGrid`)
- defect: restates-code — attribute docstring ("Must be > 0") restates the `__post_init__` validation two lines below.
- failure: UNVERIFIED — no runtime effect; pure redundant documentation.

### F-108
- raised-by: A-106
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/aggregators.py:49-52
- defect: restates-code — Raises section restates the two `if` guard clauses immediately below.
- failure: UNVERIFIED.

### F-109
- raised-by: A-107
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/level.py:100-101 (`AudioVariant.to_json`)
- defect: restates-code — docstring describes exactly what the one-line body does.
- failure: UNVERIFIED.

### F-110
- raised-by: A-108
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/layout.py:87-89 (`evidence_dir`)
- defect: restates-code — path half restates the one-line function body; only the "nothing concluded" clause carries content already stated in the module docstring.
- failure: UNVERIFIED.

### F-111
- raised-by: A-109
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speaker.py:79-88 (`_cosine_similarity`, `_cos_dist`)
- defect: restates-code — both docstrings name what the four-line function bodies already make obvious.
- failure: UNVERIFIED.

### F-112
- raised-by: A-110
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/attribution.py:43-47 (`_binary_entropy`)
- defect: restates-code — three-line body is the textbook binary-entropy formula; docstring adds nothing beyond the name.
- failure: UNVERIFIED.

### F-113
- raised-by: A-112
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/signal.py:79-93 (`SignalProvenance.to_json`)
- defect: restates-code — one-line docstring on a method that is a dict literal of the dataclass fields.
- failure: UNVERIFIED.

### F-114
- raised-by: A-113
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/acoustic.py:50-56, 105-113 (`lufs_track`, `loudness_confidence_track`)
- defect: restates-code — both docstrings restate the return tuple / one-line composition already visible in the signature and body.
- failure: UNVERIFIED.
- note: same function (`lufs_track`) as F-181 (B-14)'s promotion-candidate; different fix (docstring rewrite vs. module relocation) — kept separate.

### F-115
- raised-by: A-114
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:89-92, 108-145 (five one-line property docstrings)
- defect: restates-code — each property docstring adds nothing past the one-line body.
- failure: UNVERIFIED.

### F-116
- raised-by: A-115
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:416-418 (`_overlaps`)
- defect: restates-code — exact restatement of the one-line boolean expression below it.
- failure: UNVERIFIED.

### F-117
- raised-by: A-116
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sources.py:269-272 (`ExcisedSegment.duration_s`)
- defect: restates-code — exact restatement of `max(0.0, self.end - self.start)`.
- failure: UNVERIFIED.

### F-118
- raised-by: A-117
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sources.py:189-191 (`is_quarantined`)
- defect: restates-code — close paraphrase of the one-line set-membership check.
- failure: UNVERIFIED.

### F-119
- raised-by: A-118
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sound_sources.py:64-72 (`_category_for`)
- defect: restates-code — mirrors the dict-lookup-plus-warn-once body with no added information.
- failure: UNVERIFIED.

### F-120
- raised-by: A-119
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speech_presence.py:73-87 (`_row_window_overlap`, `_mean_col`)
- defect: restates-code — both are direct restatements of the loop-and-filter bodies beneath them.
- failure: UNVERIFIED.

### F-121
- raised-by: A-120
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:129-141 (`_finite`, `_ramp`, `_mean`)
- defect: restates-code — pure restatement of trivial helper bodies.
- failure: UNVERIFIED.

### F-122
- raised-by: A-121
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/quality.py:194-202, 383-391 (`_finite_or_none`, `_as_optional_float`)
- defect: restates-code — two functions with near-duplicate bodies and near-duplicate docstrings; also a simplification candidate.
- failure: UNVERIFIED.

### F-123
- raised-by: A-123
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/occupancy.py:160-169 (`_union_length`)
- defect: restates-code — exact restatement of the sweep-line accumulation below it.
- failure: UNVERIFIED.
- note: same function as F-184 (B-15)'s promotion-candidate; different fix (docstring vs. relocation) — kept separate.

### F-124
- raised-by: A-124
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/shapes.py:220-238 (`LabelScore`, `Window`)
- defect: restates-code — trivial one-line restatements of two-field dataclasses.
- failure: UNVERIFIED.

### F-125
- raised-by: A-125
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/perturbations.py:131-166 (`is_identity`, `to_json`, `from_json`)
- defect: restates-code — each mirrors a one-line equality check or dict round-trip.
- failure: UNVERIFIED.

### F-126
- raised-by: A-126
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/sampler.py:78-81, 295-306 (`stats` property, `_reduce`)
- defect: restates-code — both restate the return value/branching in the one-liners below them.
- failure: UNVERIFIED.

### F-127
- raised-by: A-127
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/rounds.py:56-57 (`_overlaps`)
- defect: restates-code / undocumented duplicate — near-verbatim duplicate of `sources.py`'s `_overlaps` with no cross-reference; a reader of one module's rationale won't know the other copy exists.
- failure: UNVERIFIED.

### F-128
- raised-by: A-128
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/global_summary.py:20-37
- defect: restates-code — close paraphrase of the `if/elif` chain at lines 292-306, despite framing itself as semantics documentation.
- failure: UNVERIFIED.
- note: overlaps lines with F-173 (D-9)'s assumption finding about the same `n_speakers`/`single_speaker_uncertainty` logic; different fixes (shorten a redundant docstring vs. change the scoring to be task-aware) — kept separate.

### F-129
- raised-by: A-129
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/global_summary.py:160-174 (`_aggregate_quality`)
- defect: restates-code — re-derives, almost line for line, the `ramp(...)` calls and thresholds two lines below it.
- failure: UNVERIFIED.
- note: overlaps topic (not lines) with F-180 (B-11)'s unfitted-threshold finding on the same PESQ/STOI/SI-SDR ramps; different fixes (trim a redundant docstring vs. derive/cite the actual bounds) — kept separate.

### F-130
- raised-by: A-130
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/plot.py:57-77, 251-260
- defect: restates-code, mixed — mostly walks through the code rather than adding information, though the "why per-row midpoint, not derived count" clause is legitimate rationale.
- failure: UNVERIFIED.

### F-131
- raised-by: A-131
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/l1_plot.py:212-217 (`classify_signal`)
- defect: restates-code, mixed — first sentence restates the function name; the naming-convention rationale in the second sentence is legitimate.
- failure: UNVERIFIED.

### F-132
- raised-by: A-132
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/l2_plot.py:32-52 (`build_round_timeline`)
- defect: restates-code — Args section restates each parameter name; only the empty-figure-ambiguity sentence is load-bearing.
- failure: UNVERIFIED.

### F-133
- raised-by: A-133
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/labelstudio.py:112-140, 341-343
- defect: restates-code, mixed — pure string-formatting docstrings restate the f-string beneath them; the "no pass token" clause is legitimate one-line rationale.
- failure: UNVERIFIED.

### F-134
- raised-by: A-134
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/foreground.py (short-file sample)
- defect: restates-code — one minor restatement instance sampled from an otherwise mostly-rationale file.
- failure: UNVERIFIED.

### F-135
- raised-by: A-135
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:31-32 (`_to_audio`)
- defect: restates-code — one-line body says exactly this; nothing non-obvious is added.
- failure: UNVERIFIED.

### F-136
- raised-by: A-136
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/convergence.py:104 (`round_summary`)
- defect: restates-code — purely names the return artifact, self-evident from the function's own `return {...}`.
- failure: UNVERIFIED.

### F-137
- raised-by: A-137
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:4-22
- defect: restates-code — near line-by-line paraphrase of five functions' bodies; no more informative than reading them.
- failure: UNVERIFIED.

### F-138
- raised-by: A-138
- layer: prose
- location: src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:140-141 (`BudgetLedger`)
- defect: restates-code — "light is uncapped" is immediately visible from the class body two lines below; only the FR pointer adds anything.
- failure: UNVERIFIED.

---

## Layer: computation (Sweep B)

### F-139
- raised-by: B-1
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/fuse.py:559 (`derive_mask_from_axes`, `settled_below: float = 0.35`)
- defect: unfitted-threshold — whether a bucket becomes a `target_free` mask region (discounting speaker/other signals in later rounds) is gated on a bare, undecided default, matching `keys.py`'s own named paradigm case of this defect class.
- failure: a bucket at uncertainty 0.34 becomes `target_free` (confidence 0.66), attenuating later-round signals there; a bucket at 0.36 gets no region and no discount, despite being qualitatively indistinguishable. Not overridable via `run_config.py`.

### F-140
- raised-by: B-2
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/fuse.py:831 (`fuse_axes`, `unsettled_above: float = 0.6`)
- defect: unfitted-threshold — whether a bucket is offered to the D-10 `remeasure` hook and counted toward C4 convergence is gated on a bare, undecided default with no caller override.
- failure: a bucket at 0.61 is added to `_pending` and counted as an untried action (blocking convergence); a bucket at 0.59 never is.

### F-141
- raised-by: B-3
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/aggregators.py:86-91 (`"disagreement_weighted"` branch)
- defect: misnamed-statistic — `(1 - mean_conf) * max_u` algebraically reduces to `mean(uncertainty) * max(uncertainty)`, a function of the level of uncertainty, not of disagreement/spread, despite the name and inline comment.
- failure: five sub-signals unanimously at 0.9 score `0.81`; four at 0.0 and one at 1.0 (textbook disagreement) score `0.2` — the aggregator ranks unanimous agreement ~4x higher "disagreement" than genuine disagreement, and it's a selectable value in `default.yaml`'s `uncertainty.aggregator`.

### F-142
- raised-by: B-4
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/level.py:129-289 (`apply_gain_db`, `integrated_lufs`, `loudness_range_lu`, `true_peak_dbtp`, `clipped_fraction`, `normalization_gain_db`, `peak_limited_gain_db`)
- defect: promotion-candidate — pure BS.1770/EBU-Tech-3342-grounded loudness/gain/clipping math over a raw `(waveform, sampling_rate)` pair, with no workflow-bookkeeping coupling; target `senselab.audio.tasks.quality_control`.
- failure: UNVERIFIED — not a functional defect, a placement/reuse opportunity.

### F-143
- raised-by: B-5
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/support.py:276 (`MIN_LOW_FRACTION`), used at :301-353 (`informative_evidence`)
- defect: unfitted-threshold — `MIN_LOW_FRACTION=0.02` gates evidence-pool admission for speech-presence corroboration, but the only numbers ever offered to justify it were measured under a bug the docstring itself disowns ("must be re-measured before they are cited again").
- failure: a signal whose "no speech" fraction sits just under/over 0.02 is silently included/excluded from evidence with no valid measurement behind the cutoff, feeding `speaker_count_posterior` and `reliability.measured_weights`'s `backing` factor.
- note: cross-referenced (not re-reported) by Sweep D — excluded from its assumption population as population-neutral (no age/task-specific angle).

### F-144
- raised-by: B-6
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/speaker_identity.py:121 (`multimodal_threshold: float = 0.15`)
- defect: unfitted-threshold — probability cutoff for "multimodal" speaker-count posterior has no stated derivation, unlike calibration.py's well-derived detection-margin ladder.
- failure: `{2: 0.86, 3: 0.14}` reads unimodal/converged; `{2: 0.84, 3: 0.16}` reads multimodal/not-converged — a 2-point shift flips the adaptive loop's stopping decision, with no measurement behind 0.15 vs 0.10/0.20.

### F-145
- raised-by: B-7
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/speaker_identity.py:60 (`_SUPPORTED_THRESHOLD = 0.5`)
- defect: unfitted-threshold — the cutoff for "corroborated by the audio" (`has_supported_evidence`) is an unexplained midpoint, unlike `signal_support`'s derived floor (`MIN_EVIDENCE_WEIGHT=0.05`).
- failure: `source_support=0.49` reports `has_supported_evidence:false`; `0.50` reports `true`, in `final/speakers.json`.

### F-146
- raised-by: B-8
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/identity_binding.py:145 (`binding_agreement`)
- defect: unearned-confidence — returns `0.0` whenever `eligible==0`, conflating "every diarizer explicitly rejected this speaker" with "nothing was actually checked" (no spans to bind against, or capacity already reached); docstring only describes the first case.
- failure: two diarizers both already at capacity elsewhere produce `bound=[]`, `unbound=[]`, `eligible=0` → `binding_agreement=0.0`, bit-for-bit identical to two uncensored diarizers unanimously rejecting the speaker. Currently unwired into `fuse.py`'s production path (only `identity_binding_test.py` exercises it), so the concrete run-time effect is UNVERIFIED, but the defect is live in the shipped, tested public function.
- note: cross-referenced (not re-reported) by Sweep D — excluded from its assumption population as population-neutral.

### F-147
- raised-by: B-9
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/speaker.py:636-640 (`n_models`, `share`), consumed via speaker_identity.py:627
- defect: unearned-confidence — `speech_presence_confidence` is the fraction of *present* models placing a speaker in a bucket; a diarizer that crashed or never ran shrinks the denominator instead of counting as an absent vote (docstring handles "reports silence" but not "never ran").
- failure: 3 of 4 diarizers failing leaves one survivor; every bucket where it reports a speaker gets confidence 1.0/uncertainty 0.0 — indistinguishable from four-way unanimous agreement, in `final/per_speaker_presence.parquet`. The sibling `speaker_assignment` voter already fixed this exact failure mode via `n_sources`/`source_outcomes`; the fix wasn't carried here.

### F-148
- raised-by: B-10
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/statistics.py:51 (`confidence`), :81 (`variability`), :112 (`entropy_uncertainty`), :132 (`epistemic_uncertainty`)
- defect: promotion-candidate — pure statistics over generic `Sequence[float]`/`Mapping[str,float]` inputs (weighted-vote probability, population std-dev, normalized Shannon entropy, entropy mutual-information decomposition); target `senselab/utils/tasks/` (e.g. new `uncertainty.py`), matching the codebase's own `project_mc_dropout_optional` want.
- failure: UNVERIFIED — placement opportunity, not a functional defect; six import-site renames.

### F-149
- raised-by: B-11
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/global_summary.py:195-213
- defect: unfitted-threshold — PESQ/STOI/SI-SDR ramp bounds are asserted as "literature-derived acceptance thresholds" with no citation anywhere in module/config/specs; docstring itself contradicts the code (says SI-SDR "below 5 dB poor," ramp low anchor is 0.0).
- failure: PESQ=2.6 (ordinary usable-speech quality, not "clean") yields `pesq_unc=0.6` via inverted ramp math, which via `max(...)` can dominate `quality.uncertainty` and the run's headline `combined_uncertainty` — flagging an ordinary recording as failing "high quality" on an unfitted boundary.
- note: same family as the already-fixed HNR-ramp defect; see Cross-sweep patterns.

### F-150
- raised-by: B-12
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/disagreements.py:152
- defect: unearned-confidence — `high_uncertainty_rate` reports `0.0` ("nothing was uncertain") whenever `total_rows==0` (every axis's harvest/fuse failed), collapsing "we could not measure anything" into "we measured, and it was all clean."
- failure: a total-harvest-failure run produces `high_uncertainty_rate: 0.0`; `scripts/check_layering.py:121` prints this directly against a stored baseline (0.9941) — reading as a dramatic improvement rather than a broken run.

### F-151
- raised-by: B-13
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/noise_floor.py:294,358-373 (`recorder_margin_db`, default 3.0 dB)
- defect: unfitted-threshold — decides whether a band's `binding_floor` reads "recorder" vs "perceptual"; every other margin in `data/detection_margin/2026-07-29.json` is cited/derived, but this one's only textual grounding is a qualitative "within a few dB" phrase, not a measured or cited number.
- failure: UNVERIFIED (plausible misclassification near the 3 dB boundary; no measured recorder-vs-room-gap distribution was available to construct a concrete case). Stamped onto every `NoiseFloorEstimate` and copied into stationary-source findings.

### F-152
- raised-by: B-14
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/acoustic.py:50-76,127-167 (`lufs_track`, `level_above_floor_track`)
- defect: promotion-candidate — pure `(waveform, sampling_rate)->(times, values)` numpy computations (BS.1770-style short-term loudness; bias-corrected percentile floor excess), only `math`/`numpy`/`pyloudnorm`; target `senselab/audio/tasks/features_extraction/` (new `loudness.py`).
- failure: UNVERIFIED — placement opportunity.
- note: same function (`lufs_track`) as F-114 (A-113)'s restates-code docstring finding — different fix, kept separate.

### F-153
- raised-by: B-15
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/occupancy.py:133-169 (`occupancy`, `_union_length`)
- defect: promotion-candidate — generic interval algebra (clip intervals to a window, sum union length); only workflow coupling is the `Spans`/`Span` dataclass signature; target `senselab/utils/tasks/`.
- failure: UNVERIFIED — placement opportunity.
- note: same function (`_union_length`) as F-123 (A-123)'s restates-code docstring finding — different fix, kept separate.

### F-154
- raised-by: B-16
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/adaptive/loop.py:315-317
- defect: unearned-confidence — `run_state="converged"` is set whenever no intervention fired *and* none was even proposed (every open bucket's doubt in `[theta_low, theta_high)`, below the region-seed threshold), not only when buckets truly reached `u<=theta_low` — the per-bucket definition of "converged" used everywhere else.
- failure: a recording whose speaker-axis buckets sit steady at ~0.40 doubt (above theta_low=0.33, below theta_high=0.66) never seeds a region; with no P3/C9 candidates either, the loop reports `run_state="converged"` after one round while `per_axis["speaker"]` in the same document lists those buckets `"open"` with nonzero residual mass — a self-contradicting headline `final/decisions.json` verdict.
- note: cross-referenced (not re-reported) by Sweep D — excluded from its assumption population as population-neutral.

### F-155
- raised-by: B-17
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:198-244, consumed via adaptive/interventions.py:798-812 (I1) and :850-865 (I2); also folds in `loop.py:806`'s re-instance of the same root cause
- defect: unearned-confidence — I1/I2's added votes carry no key `fuse.per_signal_uncertainty` recognizes, so `fuse_axis` never scores them, yet `belief.py:818`'s `contributing_sources` (built from raw `active_votes` keys) lists them as if they had spoken toward the value.
- failure: a repaired region shows `mean_after==mean_before` every round (only real diarizer votes move the number) while `contributing_sources` — and `L2/round/<n>/estimates/speaker.parquet`, and the LabelStudio view — falsely lists the I1/I2 votes as contributing; after `max_region_rounds` of no-op touching, the region is marked `irreducible: no_reduction_under_available_interventions`, a false verdict whose true cause is the schema mismatch, not an aleatoric limit.

### F-156
- raised-by: B-18
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:227-231 (`repair_identity`)
- defect: misnamed-statistic — `boundary_confidence` falls back to the literal `0.5` for any segment edge lacking a genuine change-point (diarizer-only cut or voiced-span boundary), indistinguishable in the output from a measured prominence of 0.5.
- failure: `adaptive/fusion.py:296-297,324` writes this into `final/diarization.json` labeled "real boundary confidences from change-point prominence" — a fabricated 0.5 reads as *more* confident than a genuine, weak measured detection at e.g. 0.05, misordering true vs. fabricated confidence.

### F-157
- raised-by: B-19
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:939 (`_p2_trigger`)
- defect: unfitted-threshold — `fires = coarse_share >= threshold or mean_instability > 0.0` compares a continuous variance-derived quantity against exactly `0.0`, contradicting the function's own docstring ("a high value means the bucket straddles an onset").
- failure: `mean_instability` is essentially always `>0.0` for real-valued frame posteriors, so `P2_fine_posteriors` (cost class medium, capped 24/run) fires far more often than intended, consuming budget U1/U2 then lose to `deferred_budget`.
- note: cross-referenced (not re-reported) by Sweep D — excluded from its assumption population as population-neutral.

### F-158
- raised-by: B-20
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:14,196 and interventions.py:502-503,1134-1139
- defect: misnamed-statistic — `priority = gain / _COST_WEIGHT[cost_class]` is sorted across every rule in one list, but `gain` is not one quantity: bounded doubt-seconds (`_mass_gain`), a raw unbounded count (`_n_candidates_gain`), and an arbitrary x10-scaled product (`_u2_gain`) with no shared normalization.
- failure: on a recording with 50 uncorroborated-speech buckets, P3's `priority=50` dwarfs a genuinely contested speaker region's bounded `_mass_gain` priority (typically `<1`), and U2's x10 multiplier inflates its rank relative to an equal-mass U1 candidate — `priority` reflects which gain formula a rule happened to be assigned, not relative value across rule types.

### F-159
- raised-by: B-21
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/adaptive/convergence.py:64-78 (`apply_convergence_marks`), together with adaptive/provenance.py:1-136 (dead)
- defect: unearned-confidence — `improvement = float(prev_u) - float(last_u)` is the undifferentiated raw delta the dead `provenance.py` module's own docstring warns against; nothing distinguishes "uncertainty fell from independent evidence" from "uncertainty fell from the loop re-scoring its own prior overwrite."
- failure: a rule that re-derives a value from evidence a prior round already committed (e.g. a cached identity-repair result re-read by another rule against the same embeddings) can register `improvement>=epsilon` purely from re-scoring, with no mechanism to classify it as `revision` rather than genuine progress toward `converged`.
- note: possible duplicate of F-5 (A-5) — see that entry's note; both trace to the same dead `provenance.py`, but fixing the stale docstring alone would not fix this convergence-math gap.

### F-160
- raised-by: B-22
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:125-149 (`_agglomerative_cosine`), :46-50 (`_l2`), :53-78 (`change_point_trajectory`)
- defect: promotion-candidate — deterministic average-linkage clustering on a cosine-distance matrix, L2-normalization, and a fixed-smoothing adjacent-window cosine trajectory are generic numerical routines with zero dependency on `Region`/`VoteStore`/policy dicts; target `senselab/utils/tasks/` or `senselab/audio/tasks/speaker_embeddings/`.
- failure: UNVERIFIED — placement opportunity, blocked only by leading-underscore naming.

### F-161
- raised-by: B-shadow
- layer: computation
- location: src/senselab/audio/workflows/audio_analysis/types.py:1
- defect: naming defect — the module is named `types`, shadowing the stdlib module for any Python process whose working directory is this package.
- failure: `cd src/senselab/audio/workflows/audio_analysis && python -c "import ast"` fails with `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types'`, naming `weakref` rather than the real cause — observed while writing the sweep.

---

## Layer: orchestration (Sweep C)

### F-162
- raised-by: C-1
- layer: orchestration
- location: src/senselab/audio/workflows/audio_analysis/compute.py:433 (`harvest_pass`)
- defect: call-site-mismatch — `fuse_consensus_words(asr_resolved)` is called with no `policy=` argument, even though `harvest_pass` has `speech_presence_policy` bound in scope and uses it three lines earlier for the mask harvest; of the four total call sites of `fuse_consensus_words(`, this is the only one reachable in production, and it drops `policy`, while the two call sites that would honor `policy` (asr.py:287, asr.py:466) are structurally unreachable given the real caller always supplies `fused=`.
- failure: a user setting `linking.asr_slot_overlap`/`asr_slot_mid_tol_s` to anything other than the 0.3/0.15 defaults sees no change in the published `asr` or `speaker` axes; `word_doubt_provenance["slot_overlap"]` reports 0.3 regardless of the config — the "recorded value cannot drift from used value" guarantee is not actually exercised, because the recorded value is always the hardcoded default.

### F-163
- raised-by: C-2
- layer: orchestration
- location: src/senselab/audio/workflows/audio_analysis/compute.py:890-1009 (`_speech_window_mask`), mirrored at stages.py:763-806, sound_sources.py:193, background_mask.py:534
- defect: model-in-control-flow — whether an embedding-clustering window counts as "speech" is decided by a hardcoded backend-priority ladder keyed on the literal strings `"yamnet"`/`"ast"` ("YAMNet is authoritative when available"), not on anything measured about the two classifiers' relative confidence; the same closed two-classifier assumption is hardcoded at four call sites, and `PassPlan` only ever exposes `ast_model`/`yamnet_model` as named fields — unlike the ASR/diarization/embedding-model lists elsewhere in the package, which take arbitrary model-id lists through config.
- failure: on a window where AST's top-1 is `Speech` at high confidence and YAMNet's top-1 is `Music`/`Singing` (a documented YAMNet confusion on child/sung voices, named in the function's own docstring), the window is unconditionally marked non-speech and excluded from `cluster_pass_speakers`'s input, with no config knob to reweight or disable the YAMNet-first rule. Acknowledged/documented tradeoff, not a hidden bug.
- note: possible duplicate of F-176 (D-7), which reports the population-specific harm (YAMNet's child-voice miss rate) enabled by this same hardcoded ladder — kept separate because fixing the pluggability gap (a config knob for trust order) does not by itself fix the unvalidated population claim (D-7 wants the miss rate actually measured and a default that protects an unattended run); the two fixes are complementary, not identical.

---

## Layer: assumption (Sweep D)

### F-164
- raised-by: D-1
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/compute.py:105,533 (`cluster_cosine_threshold=0.5`), speaker.py:132-135 (`same_speaker_floor=0.30`, `diff_speaker_floor=0.70`), embeddings.py:262-325 (`merge_threshold=0.55`) and :611-657, plot.py `_cluster_speakers_by_embedding`
- defect: adult-speech-assumption — one cosine-similarity family (0.30/0.70 floor, 0.5-0.55 merge/cluster threshold) is applied uniformly to decide same-vs-different speaker, justified by ECAPA's EER on VoxCeleb (adult, largely broadcast/interview corpus); one function's own comment admits "different speakers with similar timbre (same gender+age, children, family resemblance) can sit at cos_sim ~0.30."
- failure: gates `_cluster_pass_speakers`'s synthetic diarization source, `harvest_speaker_votes`'s same/change-claim validation, and H2 harmonization, feeding `speaker_count_posterior` and `final/speakers.json`/`final/per_speaker_presence.parquet` with no population caveat carried into either artifact.
- experiment: compute the same embeddings on a child/sibling-pair speaker-verification corpus with ground truth, derive the empirical EER-optimal threshold and the 0.30/0.70 floor's error rate there, and compare against the adult VoxCeleb-derived rate.

### F-165
- raised-by: D-2
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/speaker.py:518-543 (`harvest_speaker_votes`, `fused_words` gate)
- defect: read-speech-assumption — "no ASR word landed in this bucket" is treated as "no speech to attribute," discarding the entire votes dict for the bucket (every diarizer/embedding/agreement/change-point vote, not just the attribution voters), on a justifying measurement made on a two-adult conversation.
- failure: in a caregiver-infant/toddler recording where the child vocalizes non-lexically (babble/cry/laugh), every bucket where only the child is active is silently zeroed out; `final/per_speaker_presence.parquet` emits no row for it — the child's speaking time disappears as if it never happened.
- experiment: on a caregiver+infant/toddler recording with ground-truth per-speaker activity (child producing only babble/cry/laugh), measure the fraction of the child's true speaking time dropped, versus the same measurement on an adult-adult control.

### F-166
- raised-by: D-3
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:249-287 (`_link_asr`, line 262)
- defect: adult-speech-assumption — Whisper's `no_speech_prob` head is trusted to flag hallucination only when it reads high, presuming the head is well-calibrated on the input; a known failure point for non-lexical vocal sound, where Whisper can emit confident, low-`nsp` fabricated words.
- failure: a confidently-hallucinated transcript over a crying/babbling bucket reports `speaks=True` with high confidence and nonzero `word_overlap_s` — exactly the condition that prevents D-2's (F-165's) wordless gate from firing, so a non-lexical vocalization ends up attributed to fabricated lexical content instead of recognized as unattributable non-speech vocal activity.
- experiment: feed Whisper infant-cry, toddler-babble, and laughter clips alongside matched adult non-speech vocal sounds (cough, laugh, sigh); measure the rate of confident (low-nsp) fabricated transcriptions per group.

### F-167
- raised-by: D-4
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/speaker_identity.py:469-492 (`evidence_from_passes`/`_distinct_speakers`), consumed via `speaker_count_posterior`/`SpeakerCountPosterior.to_json`
- defect: lifespan-gap — the run's "how many speakers" belief is built entirely from each diarizer's reported label count, weighted by cross-pass stability and physical support, neither of which encodes what population that diarizer's model was trained/validated on (overwhelmingly adult conversational/meeting corpora).
- failure: `modal_count`/`probabilities` is reported as the headline speaker-count verdict with no population caveat; a child-voice-specific diarizer failure (merging two children into one label, or splitting one child's variable voice into two) surfaces only as ordinary posterior uncertainty, indistinguishable from any other disagreement source.
- experiment: run the diarizers on ground-truth-labeled adult-adult, adult-child, and child-child dyad recordings; measure `modal_count` accuracy and posterior calibration stratified by age composition.

### F-168
- raised-by: D-5
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:42-54,191-211 (`TARGET_EVENT_LABELS`/`target_event_types_for`) and calibration.py:232-237, against data/audioset_source_map.json's own cry/babble entries
- defect: lifespan-gap — the "target's own activity" vocabulary for non-speech tasks covers only `speech`/`breath`/`cough`, with no `cry`/`babble` task type, even though the AudioSet map this same package loads already has labels for exactly those sounds.
- failure: for a cry- or babble-elicitation recording, `target_event_types_for` falls back to speech-only, so the child's target cry/babble scores `target_confidence≈0` and reads `target_free`, while `sound_sources.py`'s own map simultaneously scores the same audio's mass on `"Baby cry, infant cry"→people`, marking the same span `nontarget_active` — the recording's actual target vocalization is reported as background content, the exact misattribution FR-033a was built to prevent for breath/cough.
- experiment: run an annotated infant-cry or toddler free-vocalization recording through the mask/background-scene stage under each declared task label, and measure the fraction of ground-truth cry/babble time marked `target_free`/`nontarget_active` versus an adult breath-task control.

### F-169
- raised-by: D-6
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/degradation.py:33-44 (`DEFAULT_ANCHORS`, `snr_clean_db=25.0`/`c50_clean_db=30.0`) and :129-172 (`scene_degradation`, no `task_type` parameter)
- defect: read-speech-assumption — the "clean" reference point is fluent conversational/studio speech at ~25 dB+ SNR, applied identically regardless of target activity; `scene_degradation` takes no task-type input at all, unlike `background_mask.py`'s task-aware machinery elsewhere in the same package.
- failure: `quality_snr`/`quality_reverb` feed the run's headline quality verdict as one population-general number; a breath/cough/cry-elicitation task, where quiet capture is correct by design, scores as degraded purely from the SNR gap to the conversational anchor.
- experiment: record matched breath-task and conversational-task audio from the same device/room, run the quality/degradation stages on both, and check whether the breath-task recording scores as substantially degraded despite correctly capturing its intended quiet signal.
- note: same file as F-34 (A-34)'s prose rationale about the L1/L2 calibration boundary — different fix (task-conditioning the anchor vs. documenting why L1 must not hold the calibrated score) — kept separate.

### F-170
- raised-by: D-7
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/compute.py:890-1009 (`_speech_window_mask`)
- defect: adult-speech-assumption — YAMNet's top-1 AudioSet label is treated as authoritative ("veto, not a fallback ladder") for whether a window counts as speech before clustering; the function's own docstring names the child-voice-as-Music/Singing failure mode, with the stated mitigation being a manual per-run operator action, not a default that protects an unattended run.
- failure: the mask gates which windows reach `_cluster_pass_speakers` (merged into `diarization.by_model`, feeding speaker votes, `final/per_speaker_presence.parquet`, `final/speakers.json`); a dropped child/infant vocalization reads as "no evidence there" rather than "evidence rejected by an instrument with a known population-specific miss rate."
- experiment: run YAMNet's top-1 label over a corpus stratified by speaker age band (adult conversational, child connected speech, toddler one-word, infant cry/babble) and measure the false "non-speech" veto rate per band against the adult band's rate.
- note: possible duplicate of F-163 (C-2) — see that entry's note; complementary fixes, kept separate.

### F-171
- raised-by: D-8
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/compute.py:101,592-593 (`embedding_window_s: float = 2.0`, "ECAPA's recommended minimum")
- defect: adult-speech-assumption — speaker-embedding extraction uses a fixed 2.0s window sized to ECAPA's recommended minimum, derived from models benchmarked on sustained adult conversational/read speech where a single speaker reliably occupies 2+ continuous seconds.
- failure: a toddler's one-word utterance (commonly <1s), a single infant cry bout, or a babbling burst does not fill the window, so the embedding mixes the brief target vocalization with silence and adjacent content; feeds the same synthetic-diarization chain as F-170, with no signal that a given window's embedding rests on a partially-occupied span.
- experiment: on a corpus with ground-truth vocalization durations spanning infant cries, toddler one-word utterances, and adult conversational turns, measure same/different-speaker cosine separation as a function of window occupancy, and identify the duration below which separation degrades past usable.

### F-172
- raised-by: D-9
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/global_summary.py:29-37,289-306 (`n_speakers` semantics / `single_speaker_uncertainty`)
- defect: lifespan-gap — the headline "single speaker" compliance claim scores any recording with 2+ detected speakers as maximally violating (1.0), presuming a solo self-directed speaker paradigm; does not hold for caregiver-mediated pediatric elicitation, the standard paradigm for recording infants/toddlers, where a co-occurring adult is correct, not a defect.
- failure: `single_speaker_uncertainty` feeds `combined_uncertainty`, the run's bottom-line verdict; a technically clean caregiver-child recording is scored maximally noncompliant purely for correctly containing two speakers, indistinguishable from a recording that was supposed to be solo and wasn't.
- experiment: run `compute_run_global_summary` over caregiver-mediated pediatric elicitation recordings alongside solo adult self-report recordings and compare `single_speaker_uncertainty`/`combined_uncertainty` distributions.
- note: overlaps lines with F-128 (A-128)'s restates-code finding on the same docstring/if-elif chain — different fixes (shorten redundant prose vs. make the scoring task-aware) — kept separate.

### F-173
- raised-by: D-10
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:81-99 (`detect_change_points`), :152-244 (`repair_identity`; `min_segment_s=0.25`, `recluster_cosine_threshold=0.45`), consumed via adaptive/interventions.py I1/I2, landing in adaptive/fusion.py:250-330
- defect: adult-speech-assumption — the repair mechanism presumes speaker turns are recoverable as ≥250ms segments with a clean embedding-cosine-trajectory peak at boundaries, the behavior such models are trained/validated to exhibit on adult conversational corpora.
- failure: I2's output becomes `refined_identity`, written verbatim into `final/diarization.json`/`final/speakers.json` as the run's authoritative "who is speaking when"; individual infant cries/coos frequently run under 250ms and are merged into a neighboring segment by construction, with no field recording that validity was established only on adult speech.
- experiment: on a corpus with time-aligned ground truth spanning at least one infant/toddler and one adult per recording, run I1/I2 and compute segment-purity/ARI against ground truth separately for infant- vs. adult-attributed spans, compared against an adult-only conversational corpus.

### F-174
- raised-by: D-11
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:1059-1131 (I4), adaptive/belief.py:1099-1146 (`ALEATORIC_FLOOR_TERMS`/`_attach_floor`), adaptive/convergence.py:79-86
- defect: adult-speech-assumption — I4 derives `overlap_posterior` purely from inter-diarizer disagreement, presuming real overlap registers as disagreement among diarizers trained/validated mainly on adult multi-speaker corpora.
- failure: for infant-caregiver co-vocalization (rapid, non-turn-taking simultaneous crying/babbling with adult speech), a diarizer validated on adults may simply fail to segment the second voice, yielding a *low* `overlap_posterior` meaning "the model didn't see two speakers," not "no overlap" — feeding `aleatoric_floor` and the `irreducible_reason` written into `final/convergence.json` with no population caveat.
- experiment: build or use infant-caregiver recordings with hand-labeled true co-vocalization spans plus a matched adult-adult corpus with labeled overlap; run I4 on both and compare `overlap_posterior` against ground truth per population.

### F-175
- raised-by: D-12
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:430-473 (C9 `_missed_speech_candidates`/`_c9_execute`)
- defect: read-speech-assumption — C9 only proposes a corrective "speaks" vote in a low-p_voice bucket when `len(families)>=2` (at least two ASR families independently converge on word-level text), presuming lexical content that general-purpose ASR systems converge on.
- failure: for crying/babbling/other pre-lexical vocalization, ASR emits nothing, hallucinates, or disagrees entirely, so the `>=2 families agree` gate essentially never fires regardless of true vocalization present; a bucket can reach `status:"converged"` at low presence with no distinction between "converged because silence" and "converged because the correction mechanism is structurally inapplicable."
- experiment: on recordings with labeled infant/toddler vocalization at low measured p_voice, measure how often ≥2 ASR families agree on any word inside those spans versus inside labeled adult speech at comparably low p_voice.

### F-176
- raised-by: D-13
- layer: assumption
- location: src/senselab/audio/workflows/audio_analysis/adaptive/evaluate.py:85-87 (`transcribed`/`untranscribed` split), :117-130 (`_score_words`)
- defect: lifespan-gap — untranscribed ground-truth spans are excluded from WER on both sides (reasonable per-segment), but this leaves the certifying `transcript.wer` metric silently undefined for whatever an annotator declined to transcribe — disproportionately crying/babbling/non-lexical vocalization for non-adult recordings, not merely hard-to-hear speech.
- failure: `eval.json`'s `transcript.wer`/`wer_normalized` is the pipeline's headline accuracy figure with no record of the `untranscribed`-span fraction; a corpus with substantial untranscribable infant/toddler material reports the identical headline WER as an all-adult corpus, silently certifying "the pipeline works" for a population it never scored.
- experiment: alongside WER, compute and report the fraction of total recording/GT duration in `untranscribed` spans, stratified by available speaker/age metadata, and compare the current WER-only headline against a coverage-adjusted accuracy metric.

---

## Cross-sweep patterns

### Pattern 1: an unfitted numeric threshold gates a binary downstream verdict, same shape as the already-fixed HNR ramp
Seven locations, spanning both computation-layer defaults and assumption-layer population claims:
F-139 (`fuse.py:559`, `settled_below=0.35`), F-140 (`fuse.py:831`, `unsettled_above=0.6`), F-143/B-5
(`support.py:276`, `MIN_LOW_FRACTION=0.02`, numbers disowned by their own docstring), F-144
(`speaker_identity.py:121`, `multimodal_threshold=0.15`), F-145 (`speaker_identity.py:60`,
`_SUPPORTED_THRESHOLD=0.5`), F-151/B-13 (`noise_floor.py`, `recorder_margin_db=3.0`), F-149/B-11
(`global_summary.py`, PESQ/STOI/SI-SDR ramp bounds, contradicting its own docstring). Every one
gates a binary/categorical verdict (settled-or-not, multimodal-or-not, supported-or-not,
recorder-or-perceptual, acceptable-or-not) on a boundary with no cited derivation and no
run-config override — a repeated pattern more actionable than seven separate one-off fixes: a
single sweep to require every threshold in this module family to carry either a citation or a
`data/`-derivation file (per this repo's own stated convention) would close all seven at once.

### Pattern 2: a default value of `0.0`/`1.0`/absence is indistinguishable from "not measured"
Five+ locations conflate a genuine zero/saturated measurement with "nothing was actually checked":
F-83/A-83 (`belief.py`, aleatoric floor defaults to 0.0 on every lookup miss — "the confident claim
this audio imposes no floor"), F-146/B-8 (`identity_binding.py`, `binding_agreement=0.0` for
`eligible==0`, indistinguishable from unanimous rejection), F-147/B-9 (`speaker.py`,
`speech_presence_confidence=1.0` when only one of four diarizers ran, indistinguishable from 4-way
corroboration), F-150/B-12 (`disagreements.py`, `high_uncertainty_rate=0.0` when `total_rows==0`,
i.e. total harvest failure reads as "all clean"), F-156/B-18 (`identity_repair.py`,
`boundary_confidence` fabricates `0.5` for any non-measured segment edge, outranking genuine weak
measurements). This is the same class of bug the codebase's own `l1-post-processing-register.md`
already tracks for other signals (silhouette-as-probability, SNR floor saturating to 0.0) — it
recurs because "did we measure this" and "what did we measure" share one field in five different
places.

### Pattern 3: an adult/clean-corpus-derived anchor is applied with no population or task conditioning
Four+ locations apply a single numeric anchor fitted on adult conversational/studio speech
uniformly across every recording, with the mismatch surfacing only when a caregiver/child or
non-conversational task is involved: F-164/D-1 (0.30/0.70/0.5-0.55 cosine-similarity family, from
VoxCeleb), F-169/D-6 (`degradation.py`'s 25 dB SNR / 30 dB C50 "clean speech" anchor, no
`task_type` parameter), F-172/D-9 (`global_summary.py`'s solo-speaker assumption penalizing
correct caregiver-mediated recordings), F-171/D-8 (`compute.py`'s 2.0s ECAPA-minimum embedding
window, sized for sustained adult turns). `background_mask.py` has task-aware machinery a few
files away in the same package (F-168/D-5's `TARGET_EVENT_LABELS` gap notwithstanding) — the
degradation/global-summary/embedding path never adopted the same conditioning.

### Pattern 4: the same rationale/incident is narrated near-verbatim in 3+ files with no canonical home
Called out by Sweep A itself in several places, confirmed here as a genuine repeated-fix
opportunity rather than 3-4 independent findings: the "silhouette coefficient is not a probability"
incident (F-24/A-24 in embeddings.py, F-52/A-52 in speech_presence_link.py, both targeting
l1-post-processing-register.md item 12); the "5 speakers vs 2 diarizers reported" validation
anecdote (F-22/A-22 in speaker_identity.py, echoed in influence.py/support.py/reliability.py,
F-35/A-35's "third independent telling"); the "three id namespaces once rendered as S0" rationale
(F-23/A-23 in identity_binding.py, duplicated in harmonize.py and clustering.py, and again in
F-28/A-28's joint.py "What changes from J4" section); and asr.py's own triple-telling of its
consensus_words removal history (F-16/F-18/F-19). Each pair remains a separate finding (different
files each need independent migration/deletion), but a single canonical-home pass across these
four clusters would remove roughly seven redundant copies in one motion.

### Pattern 5: the hardcoded two-classifier (`"ast"`, `"yamnet"`) trust ladder recurs at four call sites
F-163/C-2 documents this as one finding spanning `compute.py:890-1009`, `stages.py:763-806`,
`sound_sources.py:193`, and `background_mask.py:534` — noted here only to flag that it is already
the largest single repeated-location finding in the merge (4 call sites edited identically to add
a third classifier or reorder trust), and that F-170/D-7 reports the population-specific harm this
same hardcoding enables without a config-level mitigation.
