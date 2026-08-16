# Sweep A — prose candidates

Audited `src/senselab/audio/workflows/audio_analysis` (81 files) in four parallel batches, each
reading its files in full (module docstring, every class/function docstring, every comment block).
`stale-or-false` was swept exhaustively across all 81 files; `restates-code` and
`rationale-to-migrate` were sampled — each batch's "Checked and clean" section says, per file,
what was actually read and cross-checked, not just glanced at.

Six of the eight `stale-or-false` candidates below were independently re-verified against the
current code (by reading the contradicting file pair directly) before being included here; the
other two (`A-4`, `A-5`) rest on the sweeping agent's own grep/read evidence, which is recorded in
each item's `failure` field.

Batch assignments:
- Batch 1 (grid/fuse/core infra, 21 files) → A-1, A-2, A-3, A-9..A-15, A-105..A-108
- Batch 2 (ASR/speaker/identity/stats, 21 files) → A-16..A-35, A-109..A-113
- Batch 3 (background scene/speech presence/plot, 21 files) → A-36..A-74, A-114..A-134
- Batch 4 (adaptive/ subpackage, 18 files) → A-4..A-8, A-75..A-104, A-135..A-138

---

## Class: stale-or-false (priority — exhaustive across all 81 files)

### A-1
- class: stale-or-false
- location: src/senselab/audio/workflows/audio_analysis/__init__.py:1-6
- quote: "Three-axis uncertainty workflow for analyze_audio outputs. ... emits three per-bucket uncertainty time series — `speech_presence`, `speaker`, and `asr` — plus a ranked `disagreements.json` index and a 5-row timeline plot."
- why: `axes.py`'s own module docstring documents this exact defect as already found and fixed: "There were three declarations and twenty-two literal tuples... **Any list of three axes is wrong**." `axes.AXES` has four active members (`background_mask` is the fourth, fused, written to `estimates/`, and drawn on the timeline). The package's top-level docstring — the first thing a reader or pdoc sees — still states the pre-fix count.
- failure: A reader of `help(audio_analysis)` or the rendered package docs believes the workflow tracks three axes and the timeline plot has 5 rows, and would not think to look for `background_mask` in `compute_uncertainty_axes`'s output, `estimates/*.parquet`, or the timeline PNG — reproducing, at the doc level, the exact bug axes.py says it fixed in the code.

### A-2
- class: stale-or-false
- location: src/senselab/audio/workflows/audio_analysis/io.py:150
- quote: "Write ``L2/round0/votes/<axis>.parquet`` — the linked evidence, at the vote level."
- why: Verified against the real write site (`scripts/analyze_audio.py:830-833`: `derivatives_dir(run_dir, 0) / "votes" / f"{axis_name}.parquet"`) and against `contracts.py`'s declared pattern `"L2/round/{n}/derivatives/votes/*.parquet"` (contracts.py:566, 842). The actual path has a `derivatives/` segment the docstring omits and spells the round segment `round/0`, not `round0`.
- failure: A reader locating the linked-votes file from this docstring alone looks in (or writes a new contract declaration for) `L2/round0/votes/`, which does not exist — exactly the kind of undeclared-path drift `contracts.py`'s own guard (D-17) exists to catch.

### A-3
- class: stale-or-false
- location: src/senselab/audio/workflows/audio_analysis/stage_context.py:91-94
- quote: "these numbers only need to move for *wrapper-shaped* output changes — mainly ``features`` (composes three backends into a row dict) and the classifiers (attach phoneme labels)."
- why: No stage owning a `STAGE_VERSIONS` entry attaches phoneme labels. `stage_scene` (keys `ast`/`yamnet`) runs AudioSet scene classification plus a same-grid agreement sidecar — nothing phoneme-related. The only phoneme-label code in the package (`g2p_phonemes`/`normalize_arpabet` in harvesters.py) is not cache-keyed via `STAGE_VERSIONS` and is unconnected to the `ast`/`yamnet` entries this sentence is attached to.
- failure: A contributor deciding whether an AudioSet-composition change warrants bumping `STAGE_VERSIONS["ast"]`/`["yamnet"]` is pointed at the wrong justification (phoneme labels) and may skip the bump because the real change doesn't match the doc's description.

### A-4
- class: stale-or-false
- location: src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:718-719 (`_fused_axis`)
- quote: "the belief store ingests L1's per-pass axis folds"
- why: `belief.py`'s own `VoteStore.from_run_dir` docstring (belief.py:288-294) states this path was removed: "It used to read ``L1/<pass>/uncertainty/<axis>.parquet`` — a per-pass axis fold, which is a quantity that cannot exist... Both are gone." A repo-wide search finds no current writer of that path; `from_run_dir` ingests votes from `L2/round/0/derivatives/votes/<axis>.parquet` instead. `plot.py`'s docstring describes the pre-fix mechanism in the present tense.
- failure: A reader trying to understand why the belief-store line and the L2-fused overlay differ goes looking for an L1 per-pass-axis-fold file as the store's data source and finds nothing — the real (current) reason the two differ is that the fold ran once for L2 and the store re-aggregates votes, which is a different and simpler story than the one this sentence tells.

### A-5
- class: stale-or-false
- location: src/senselab/audio/workflows/audio_analysis/adaptive/provenance.py:1-22 (module docstring)
- quote: "Every state change in a mutually-influencing loop is attributable: which signal caused it, in which round, at what weight, on what evidence... (FR-011g)."
- why: `RevisionRecord`, `classify_resolution`, and `revision_log_entry` are defined and unit-tested but have zero call sites in `loop.py`, `interventions.py`, or `belief.py` (verified by repo-wide grep — the only other hit is a comment in `corroboration.py` naming the function, not calling it). Nothing in the actual attenuate/overwrite code paths (`VoteStore.attenuate_source_in_bucket`, `_i2_execute`'s cluster-vote overwrite, `_p3_execute`'s attenuation) routes through this module.
- failure: A reader believes every revision in a run is already tagged and audit-traceable per FR-011g and does not think to check whether the self-confirmation guard is reachable from any production code path — it currently is not, so nothing in a real run is actually attributed by this mechanism.

### A-6
- class: stale-or-false
- location: src/senselab/audio/workflows/audio_analysis/adaptive/loop.py:3-6 (module docstring)
- quote: "Prototype entry point (``run_adaptive_loop``). Artifact-driven: round 1 is the ingested analyze_audio run; rounds 2..K execute the intervention catalog..."
- why: Contradicted by `_baseline_round`'s own docstring 730 lines later (loop.py:735-742): the ingested round's number is **adopted** (`last_round(out_dir) or 0`), never fixed at 1 — and that function explicitly names "round 1" vs "round 0" as the exact bug this design fixed: "the adaptive loop used to call its ingest 'round 1' while the fusion loop called the same iteration 'round 0'... under one tree the collision is a round reading its own output." The module docstring also omits the fully-supported in-process ingest path (`run_adaptive_loop(run_dir, harvests=..., summary=...)`) documented in the function's own Args block.
- failure: A reader assumes the ingested round is always numbered 1 and that the loop only runs over a finished artifact directory on disk, missing that the baseline round varies per run and that an in-memory `PassHarvest` ingest path exists and is what `analyze_audio.py` actually uses.

### A-7
- class: stale-or-false
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:22-24 (module docstring)
- quote: "Still deferred: ``P2_fine_posteriors`` and v2's ``U4_overlap_separation`` (contracts/interventions.md)."
- why: Verified directly — `_p2_trigger` (895-946), `_p2_guard` (949-959), and `_p2_execute` (962-1056) are fully implemented and `P2_fine_posteriors` is registered with all three hooks in the `RULES` table (1209-1216). It is not deferred; the module's "Implemented for real" bullet list above simply omits it.
- failure: A reader believes P2 never fires and skips auditing its (working, non-trivial) coarse-dominance/frame-instability logic, or attempts to re-implement a rule that already exists.

### A-8
- class: stale-or-false
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:19-20 (module docstring)
- quote: "``I4_overlap_detection`` — segmentation-3.0 per-class posteriors (gated model; guards to ``next_actions`` without a token)."
- why: `_i4_execute` (interventions.py:1087-1131) calls `backends.overlap_track_from_spans`, whose own docstring (backends.py:243-249) states overlap is derived from **cross-diarizer spans**, explicitly not from any single model's per-class channels — matching `_p2_execute`'s own comment nearby ("I4 now derives overlap from cross-diarizer spans instead of reusing this output"). Two descriptions of I4 inside the same file disagree.
- failure: A reader auditing I4 looks for a segmentation-3.0 model invocation and a per-class posterior extraction inside `_i4_execute`; neither exists there — the actual mechanism (cross-diarizer span overlap) is a different, and arguably better-corroborated, computation than the one described.

---

## Class: rationale-to-migrate (sampled — see each batch's "Checked and clean" for what was read in full vs. sampled)

### A-9
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/grid.py:15-19
- quote: "Measured before it was: the four axes carried 242 / 242 / 19 / 8 rows on 0.1/0.02, 0.1/0.02, 0.25/0.25 and 1.0/0.5 respectively, shared zero bucket keys, and the coupling between them therefore did nothing while reporting that it had run."
- destination: grid/fuse design (doc.md "one grid" section)

### A-10
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/fuse.py:88-101 (`is_direction_only_claim`)
- quote: "The cost was measured on a real run: with ``--asr-models openai/whisper-*`` the presence axis fused 12 signals, and on the *shipped* default ASR set only 8 — all three ASR models and both diarizers had silently left the axis..."
- destination: grid/fuse design (vote-folding)

### A-11
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/axes.py:281-334 (`IDENTITY_ONLY_AXES`)
- quote: "Measured on the 48 kHz validation clip its enhanced ``words`` voter read mean 0.0510 against raw's 0.0102 — 5x higher..."
- destination: per-speaker-identity-scene design (specs/20260728-221507-per-speaker-identity-scene/layered-architecture.md)

### A-12
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/stage_context.py:202-243 (`_commit_sha_for`)
- quote: "Three outcomes, therefore, three treatments: **Not a Hub id at all**... **A definitive 'there is no commit'**... **Anything else** — a 429, a network error, a ``GatedRepoError`` — propagates."
- destination: cache/provenance design (commit-SHA pinning rules)

### A-13
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/run_config.py:9-14
- quote: "Seventy of them existed, and the run recipes in the repo's own docs differed from one another only in flags whose right value a reader had no basis to pick... every cross-axis coupling in the pipeline ran against zero shared bucket keys."
- destination: run-config / no-per-knob-flags design

### A-14
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/contracts.py:1-58 (module docstring)
- quote: "Three rounds of guards were written against the violation last found, and each missed the next instance of the same class... **Enumerating what is forbidden cannot terminate. Declaring what is permitted does.**"
- destination: stage-contracts / D-17 design summary

### A-15
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/resolution.py:1-24
- quote: "A frame posterior at ~17 ms collapsed onto 250 ms buckets **saturates**. The VAD trace came out flat at 1.0 across a conversation with four clear pauses..."
- destination: grid/fuse design (per-signal resolution)

### A-16
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/asr.py:1-27
- quote: "Four things used to ride on every bucket beside it, and all four are gone: per-bucket text..., the pairwise phoneme distance..., avg_logprob / token_entropy / alignment_ctc_score..."
- destination: asr axis design

### A-17
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/asr.py:100-115 (`phoneme_similarity`)
- quote: "Falls back to exact match when g2p is unavailable or produces nothing for either side, not to grapheme overlap: letters are not sounds..."
- destination: asr axis design (grading/g2p)

### A-18
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/asr.py:293-320 (`resample_member_doubt`)
- quote: "epistemic_uncertainty was structurally 0.0 for this axis on every run... reliability.signal_stability weighted the fused series..."
- destination: asr axis design (duplicate of module docstring's framing — consolidate)

### A-19
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/asr.py:426-456 (`harvest_asr_votes`)
- quote: "This used to be a single consensus_words entry: 1 - existence_confidence, whose share term is the weighted mean of the recognizers' agreement..."
- destination: asr axis design (third copy of the same historical narrative in one file — consolidate)

### A-20
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/speaker.py:139-153 (`harvest_speaker_votes`)
- quote: "It asked 'was it the same speaker as before?' until 2026-08-05... it read 0.666 on a conversation whose per-speaker presence doubt was 0.168."
- destination: speaker attribution design (specs/20260728-221507-per-speaker-identity-scene/speaker-axis-attribution-design.md, already referenced two lines later)

### A-21
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/speaker_identity.py:1-25 (module docstring)
- quote: "on a validation recording two diarizers each reported one speaker for the whole clip while embedding clustering reported five distinct regions aligned to name boundaries."
- destination: speaker attribution / per-speaker-identity design

### A-22
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/speaker_identity.py:300-308 (`source_kind_for`)
- quote: "on one validation recording it reported five speakers where two 'independent' diarizers reported one, with re-examination suggesting it was the closer answer."
- destination: influence/support/reliability weighting design — same anecdote is duplicated near-verbatim in influence.py, support.py, and reliability.py (4 copies total); one canonical home would remove three repeats

### A-23
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/identity_binding.py:1-19 (module docstring)
- quote: "Three id namespaces stay distinct because all three once rendered as S0... What changes from J4. It bound S_k to segmentation-3.0's activation channels, which are permutation-arbitrary within each inference..."
- destination: layered-architecture.md (D-19) — near-verbatim repeated in harmonize.py and clustering.py module docstrings (3 copies)

### A-24
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/embeddings.py:396-407 (comment on deleted p_voice computation)
- quote: "A silhouette coefficient is a property of a chosen partition on a chosen metric, not a probability... contributing a near-constant ~0.44 doubt across every bucket while earning the highest fusion weight of fifteen signals precisely because it was near-constant."
- destination: l1-post-processing-register.md (item 12, already cited by name in the comment) — the fix is recorded correctly as a comment on now-absent code, but has no anchor once that code is deleted for real

### A-25
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/clustering.py:104-133 (`assign_unified_clusters_with_seed_phase`)
- quote: "Why two thresholds: cross_group_threshold (default 0.75) governs match-across-groups... cosine_threshold (default 0.5) governs other_items matching..."
- destination: clustering/statistics design (threshold derivations — belongs beside calibration.py's derivation blocks)

### A-26
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/harmonize.py:1-23 (module docstring)
- quote: "So any cross-model statement about speaker first guesses that two labels denote the same person. Treated as fact, that guess makes two models which were never correctly compared read as disagreeing..."
- destination: layered-architecture.md (D-6)

### A-27
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/invariance.py:1-26 (module docstring)
- quote: "Gain scaling. Changes no signal-to-noise ratio... This is the same measurement that reframed background detection away from amplification..."
- destination: background-scene design (amplification finding, cross-referenced rather than linked)

### A-28
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/joint.py:1-29 (module docstring)
- quote: "J1 and J4 have moved. The count posterior is now cross-diarizer spread... `segmentation-3.0` reports one activation per speaker, but the channel ordering is arbitrary within a window."
- destination: layered-architecture.md (D-19/D-7) — duplicates identity_binding.py's "What changes from J4" section (A-23)

### A-29
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/statistics.py:1-36 (module docstring)
- quote: "The codebase had been calling all of them 'uncertainty,' which is why a max-doubt fold, a Shannon entropy, and a max-minus-min spread all ended up in a column of that name."
- destination: layered-architecture.md or a dedicated "estimator taxonomy" note

### A-30
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/measurements.py:1-16 (module docstring)
- quote: "the guesses observed were frame_mean at a resolution the model never reported and six quantities under units: 'mixed'."
- destination: layered-architecture.md (D-18)

### A-31
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/support.py:276-298 (`MIN_LOW_FRACTION`)
- quote: "Measured over 697 buckets of a real recording, four of seven candidate evidence signals never once fell below 0.20 — acoustic_hnr (median 0.500)... Caveat: those figures were taken while the screen read native_confidence undirected... must be re-measured before they are cited again."
- destination: support/reliability weighting design — flagged for content risk, not stale-or-false: the docstring itself disowns the cited numbers two paragraphs after stating them, so a migration should replace them with re-measured figures or drop the specific numbers and keep only the qualitative property

### A-32
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/influence.py:1-30 (module docstring)
- quote: "A clustering-derived pseudo-diarizer agreeing with the embeddings it was computed from is not corroboration — it is one computation counted twice..."
- destination: layered-architecture.md (D-21 rule 6) — duplicated in asr.py/speaker.py framings of the same rule

### A-33
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/calibration.py:1-32 (module docstring)
- quote: "temperature and token_entropy_reference_nats currently reach no fold. Their only consumers were aggregate.aggregate_asr and aggregate.aggregate_speech_presence, which had no production caller and are deleted..."
- destination: layered-architecture.md or l1-post-processing-register.md (declared-and-unread fields)

### A-34
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/degradation.py:1-18 (module docstring)
- quote: "Holding them at L1 destroyed the underlying measurements: clip((25 − snr_db)/20, 0, 1) returned 0.0 in every bucket of every recording measured, because clean speech sits at 60–70 dB SNR against a 25 dB anchor."
- destination: layered-architecture.md (L1/L2 calibration boundary)

### A-35
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/reliability.py:1-22 (module docstring)
- quote: "which is exactly how a saturated embedding check came to outvote unanimous diarizer agreement on a real recording."
- destination: speaker attribution / clustering-statistics design (embedding calibration incident — third independent telling, alongside speaker.py and embeddings.py)

### A-36
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:1-24
- quote: "The mask marks regions free of **target activity**... Since a 30 dB suppression baseline was measured to leave residual foreground dominant, these regions may carry most of the trustworthy background evidence..."
- destination: background-scene design (mask semantics)

### A-37
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:244-255 (`_classify_bucket`)
- quote: "confidence 0.99 *and* uncertainty 0.99 committed to 'target active' on evidence that supported no verdict at all. Measured on a 21.5 s conversation, that asymmetry produced a single whole-file `target_active` region at uncertainty 0.9997."
- destination: background-scene design (mask classification)

### A-38
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:462-477 (`_speech_activity_by_bucket`)
- quote: "That boolean is what made the mask unable to be uncertain... correctly collapsed 1070 identical buckets into a single 21 s region reporting no doubt at all."
- destination: background-scene design (mask evidence)

### A-39
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/noise_floor.py:1-32 (module docstring)
- quote: "A ``q``-quantile of exponentially distributed noise power sits a calculable factor below the mean: about 9.8 dB for a tenth percentile... Estimating per activity stratum is the mitigation, and it has **no published precedent**: validate before relying on it."
- destination: background-scene design (noise-floor estimation)

### A-40
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/noise_floor.py:165-197 (`estimate_band_floor_db`)
- quote: "a tenth-percentile-plus-6 dB cut discards roughly two thirds of exponentially distributed noise, and re-taking a low quantile of the truncated remainder drives the estimate down every pass."
- destination: background-scene design (noise-floor estimation)

### A-41
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/noise_floor.py:376-410 (stationary-sources banner)
- quote: "Standards-grounded rather than invented: ECMA-74 / ISO 7779 define a discrete tone as prominent at a Prominence Ratio of about 9 dB..."
- destination: background-scene design (stationary source detection)

### A-42
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/sources.py:1-27 (module docstring)
- quote: "Amplifying a noise floor produces confident, plausible environmental labels — waterfall, water, gurgling, static — that are statistically indistinguishable from genuine broadband noise..."
- destination: background-scene design (source detection guards)

### A-43
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/sources.py:245-256 (excision routing banner)
- quote: "excising the quiet segment and classifying it alone beat every mixed-window variant (0.705 vs a best of 0.548), because one 10.24 s window spanning both halves couples them..."
- destination: background-scene design (excision routing)

### A-44
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/foreground.py:1-24 (module docstring)
- quote: "with 30 dB of suppression and the residual amplified to a healthy level, the reported result was *identical* whether a faint background source was present or entirely absent..."
- destination: background-scene design (foreground suppression)

### A-45
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/foreground.py:121-128 (`is_deep_enough_for`)
- quote: "The oracle experiment showed 30 dB of suppression leaving the residual foreground dominant over a background 30 dB down, so the comparison is against the source's own depth below the foreground..."
- destination: background-scene design (foreground suppression — duplicate of A-44 within one file)

### A-46
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/sound_sources.py:31-48 (`AUDIOSET_SCORE_FUNCTION`)
- quote: "Reading it through a softmax makes the classes compete, which suppresses secondary classes multiplicatively — so a background source at fixed underlying evidence gets a systematically smaller share..."
- destination: background-scene design (sound-source categorization)

### A-47
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/sound_sources.py:90-106 (`window_label_mass`)
- quote: "A window whose top label is ``Music`` at 0.40 with ``Speech`` second at 0.38 voted a confident *no speech* — discarding 0.38 of speech evidence."
- destination: speech-presence design (label mass vs top-1) — duplicated verbatim in speech_presence.py (A-48)

### A-48
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/speech_presence.py:1-41 (module docstring)
- quote: "Not ``top-1 ∈ labels``: the argmax discards several hundred scores, so a window topped by ``Music`` at 0.40 with ``Speech`` second at 0.38 used to read as a confident *no speech*."
- destination: speech-presence design (L1 evidence) — same example as A-47, consolidate at destination

### A-49
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:1-42 (module docstring)
- quote: "By Jensen's inequality that strictly exceeds ``exp(mean(avg_logprob))`` whenever the chunks disagree, so the two are different statistics and one of them had been picked silently."
- destination: speech-presence design (L1/L2 split)

### A-50
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:144-176 (`_abstaining_ramp`)
- quote: "Measured on a clean two-speaker conversation, ``acoustic_hnr`` contributed mean 0.2675 doubt (median 0.2574, max 0.5000) while all four diarizers, all three recognizers and the brouhaha VAD read exactly 0.0000."
- destination: speech-presence design (signal abstention)

### A-51
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:327-349 (removed `_link_hnr` banner)
- quote: "Measured on `english_conversation_higgs_audio_v2`, median HNR is **8.12 dB**: *below* the anchor that means 'confidently voiced'... Removing it takes presence doubt from 0.0250 to 0.0160..."
- destination: speech-presence design (l1-post-processing-register.md item 10 / HNR voter)

### A-52
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:415-445 (removed `_silhouette_votes_by_bucket` banner)
- quote: "It held weight **1.0**, the highest of all fifteen presence signals, while every informative voter sat at 0.78-0.91... The least informative voter earned the most weight."
- destination: speech-presence design (l1-post-processing-register.md item 12 / silhouette voter)

### A-53
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/quality.py:1-36 (module docstring)
- quote: "Both returned **0.0 in every bucket of every recording measured**, because clean speech sits at 60-70 dB SNR and 59.8 dB C50 — far above anchors chosen for conversational audio..."
- destination: quality/degradation design (L1/L2 split)

### A-54
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/quality.py:326-352 (`quality_series`)
- quote: "``units: 'mixed'`` was the honest admission of it... adjacent values share half their audio. A consumer that treats them as independent samples is wrong..."
- destination: quality/degradation design (native-resolution series, D-20/D-25)

### A-55
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/pii.py:19-25 (module docstring)
- quote: "in pediatric and clinical voice data, the nominally most severe Presidio categories (``US_SSN``, ``CREDIT_CARD``) have near-zero true-positive rate and are dominated by ASR digit hallucinations..."
- destination: PII detection design (audio_analysis adapter)

### A-56
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/mask_harvest.py:1-24 (module docstring)
- quote: "The mask's uncertainty was ``1 - confidence`` of a single derived judgement. That read as a property of the mask when it was a property of there being **one producer**..."
- destination: background-scene design (mask harvest / D-22)

### A-57
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/mask_harvest.py:37-52 (`TARGET_POLARITY`)
- quote: "In a breathing task the target is the breath, speech detection is silent through it, and a speech vote therefore indicates target **absence**... since AudioSet maps ``Breathing`` to ``people``, a mask built from voice activity alone reported the collected signal as a background source."
- destination: background-scene design (task-gated polarity)

### A-58
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/occupancy.py:1-22 (module docstring)
- quote: "What it produced was one model's internal confidence dressed as a distribution over speaker count... The honest uncertainty about 'how many speakers are active here' is the same as for every other axis: **disagreement across models.**"
- destination: speaker/occupancy design (D-19)

### A-59
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/occupancy.py:68-79 (`capacity_for`)
- quote: "Raising instead was tried and is wrong at this depth — one unlisted diarizer would kill the whole harvest, so a new model could not be trialled without a table edit first."
- destination: speaker/occupancy design (D-19)

### A-60
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/shapes.py:1-31 (module docstring)
- quote: "Forcing them through one tabular row is what produced every reduction the real-run audit found: a per-speaker probability matrix stored as its mean, 527 label scores stored as a hand-picked sum, a span set stored as a covered fraction..."
- destination: L1 shapes / derivative design (D-18)

### A-61
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/shapes.py:148-159 (`Matrix`)
- quote: "Storing the pooled value made that choice invisibly, and it is what returned ``1.0000`` in 100% of frames on a clip that was half digital silence."
- destination: L1 shapes / derivative design

### A-62
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/perturbations.py:1-26, 76-102 (module docstring; `SNR_GATED_TRANSFORMS`)
- quote: "the raw pass placed the speaker axis at exactly 0.0 in 179 of 190 buckets, the enhanced pass at 0.398 with only 51% zeros, and averaging the two published 0.227 — the diarizers agreed and the axis said otherwise..."
- destination: perturbations / passes design (D-17)

### A-63
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/sampler.py:1-27 (module docstring)
- quote: "``native_window_s: 0.0619, resolution_s: 0.0169`` recorded on a row spanning ``0.0 → 0.1``, provenance describing a measurement the file did not contain."
- destination: L2 derivative / sampler design (D-25)

### A-64
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/rounds.py:1-24 (module docstring)
- quote: "regional trust is how the same evidence attenuates the wrong claim without silencing the right ones" (referencing "the source which turned out to be right about the five named speakers on a 4.9 s recording")
- destination: L2 fusion / rounds design

### A-65
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/rounds.py:143-157 (`DEFAULT_MAX_ROUNDS`, `DEFAULT_CYCLE_WINDOW`)
- quote: "A cycle of period *p* only becomes visible once the window holds a repeat, which takes ``p + 1`` rounds; four therefore catches periods one through three."
- destination: L2 fusion / rounds design (D-12)

### A-66
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/global_summary.py:52-59 (`PASS_FOLD`)
- quote: "It is deliberately *not* a minimum: raw and enhanced are the same recording under a transform, so they are a perturbation sample whose disagreement is evidence — picking the lower-uncertainty one... discards exactly the information the second pass was run to obtain."
- destination: run summary / global aggregation design

### A-67
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/summary.py:1-18 (module docstring)
- quote: "Treating 'not measured' as zero would report a run as more certain than it was, which is the failure mode a summary is most likely to introduce."
- destination: run summary / reporting design

### A-68
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/plot.py:1-42 (module docstring)
- quote: "A collision between two conclusions is not fixed by moving one into the evidence layer. It is fixed by giving them different names... A default argument decided the layer."
- destination: plotting / layering design

### A-69
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/plot.py:270-278 (`_load_background_mask_rows`)
- quote: "This one was written against the flat layout and matched nothing for as long as passes have lived under ``L1/``, which reads exactly like a run with no mask."
- destination: plotting design / layout history

### A-70
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/l1_plot.py:1-20 (module docstring)
- quote: "'the diarizer stopped here' next to 'the level fell to -60 dBFS here' is usually the whole story, and neither row says it alone."
- destination: plotting design (L1 evidence figure)

### A-71
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/l1_plot.py:171-196 (`SIGNAL_GROUPS`, `_ROW_HEIGHT`)
- quote: "Alphabetical order interleaved a frame VAD, an acoustic proxy and a diarizer, which made the figure unreadable... A uniform height gave a binary on/off row the same space as a spectrogram..."
- destination: plotting design (L1 evidence figure)

### A-72
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/l2_plot.py:1-13 (module docstring)
- quote: "This replaces the chunked ``timeline_001.png`` / ``timeline_002.png`` output, whose panels were mostly empty: a fixed time window rarely lines up with where anything actually happened..."
- destination: plotting design (L2 round timeline)

### A-73
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/labelstudio.py:1-17 (module docstring)
- quote: "There was an ``uncertainty__asr__text`` TextArea rebuilding a per-bucket consensus from each model's bucketed transcript... the coarse one is what forced the asr axis onto a 1.0 s grid of its own."
- destination: labelstudio / export design

### A-74
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/labelstudio.py:652-667 (`attach_scene_context_tracks_to_ls`)
- quote: "Per-speaker speech_presence is labelled by speaker rather than merged, because knowing *who* is contested is the entire reason the speaker axis moved off a single scalar..."
- destination: labelstudio / export design

### A-75
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/__init__.py:5-9
- quote: "This subpackage keeps imports light on purpose: no torch / model backends are imported at module level... Interventions that need live model backends import them lazily inside their ``execute`` functions and degrade to ``blocked_guard`` when unavailable."
- destination: adaptive loop design (import/dependency strategy)

### A-76
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/audio_io.py:110-121 (`get_stream_wav`)
- quote: "It used to be a two-armed comparison against the two pass names of the day (``perturbations.py`` records which), so a third perturbation was an edit here — in a module that has no business knowing any perturbation's name."
- destination: adaptive loop design (audio_io / perturbation dispatch)

### A-77
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:191-193 (`_enhance`)
- quote: "Stage once (download-once via the heartbeat lock) + load from the local snapshot dir so SpeechBrain makes no per-file Hub HEAD (429 source under batch)."
- destination: adaptive loop design (model loading / caching)

### A-78
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:200-213 (`speech_posteriors`)
- quote: "**What is lost, stated because it is a real reduction.** ... under ``segmentation-3.0`` it bought both."
- destination: policy/triage design (P2 rationale)

### A-79
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:296-310 (`consensus_align`)
- quote: "This was hard-coded to torchaudio MMS_FA with no way to choose... D-1 moved Canary off MMS precisely so that word-boundary differences would 'reflect the models, not two different aligners'."
- destination: belief/fusion design (consensus alignment)

### A-80
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/types.py:3-23 (module docstring)
- quote: "**Why ``TypedDict`` and not dataclasses**, which is what tasks.md asked for: ... The actual defect class here is key typos and wrong value types, not mutation..."
- destination: types/data-model design

### A-81
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:65-74 (`_HARVEST_ACCESSORS`)
- quote: "the flag said ``background_mask`` was harvested, this method enumerated three axes in a literal tuple... the axis was rebuilt from one vote per mask *region* — 1070 buckets at round 0, one by round 4."
- destination: belief/fusion design

### A-82
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:206-221 (`snr_gate_from_run`)
- quote: "the gate reached round 0 only, and the loop's ungated re-aggregation folded the enhanced pass back in, so ``final/`` published 0.2267 on a recording whose round 0 read 0.0487."
- destination: belief/fusion design (SNR gating)

### A-83
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:1108-1125 (`_attach_floor`)
- quote: "Every lookup missed, the floor was assigned ``0.0`` on every bucket of every run, and ``0.0`` is the confident claim 'this audio imposes no floor'."
- destination: belief/fusion design (aleatoric floor)

### A-84
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/corroboration.py:1-21 (module docstring)
- quote: "measured: ``acoustic_loudness`` median 0.897, ``ast`` 0.728 over 697 buckets — pooled with max they pin corroboration near 1.0"
- destination: corroboration/presence design

### A-85
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/evaluate.py:73-77
- quote: "it used to reach into ``L2/`` for the presence track, the baseline round's uncertainty mass and the last round's speaker axis, and each of those was a scorer scoring an intermediate."
- destination: evaluation design (L1/L2/final boundary)

### A-86
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:35-38 (`collect_word_streams`)
- quote: "It used to drop every word of a model overlapping a span P3 had adjudicated... and made a word's survival depend on whether the intervention had been admitted within budget."
- destination: belief/fusion design (transcript fusion)

### A-87
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:378-389 (`extract_final_estimates`)
- quote: "the deliverable presence track used to be *rebuilt* here from the belief state... so the number a consumer acted on was not the number any round believed."
- destination: final-outputs design

### A-88
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:430-438 (`write_speaker_outputs`)
- quote: "Both were written to the ``L2`` root instead... so ``final/`` carried no per-speaker output at all while two declarations for it sat unproduced."
- destination: final-outputs design

### A-89
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:35-43 (`MIN_WINDOW_WEIGHT`)
- quote: "Two bare ``0.05`` literals used to sit inline here, doing the job of this constant without naming it or connecting it to the argument that sets it."
- destination: identity-repair design

### A-90
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:54-62 (`load_outcomes_dir`)
- quote: "It was rebuilt as ``run_dir / stream / task_dir`` until the pass outputs moved under ``L1/``, at which point this returned ``{}`` on every run — silently."
- destination: adaptive loop design (artifact access)

### A-91
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:266-273 (`_claims_words`)
- quote: "This used to concatenate each model's per-bucket ``text`` and test the string for emptiness... and that forced the axis onto a 1.0 s grid so a whole word could fit inside a bucket."
- destination: policy/triage design (S1 stream election)

### A-92
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:582-591 (`_reharvest_asr`)
- quote: "The gate used to route to ``_harvest_word_level``, a second harvester emitting pairwise *word*-Levenshtein distances in the same vote schema — so an environment without ``g2p_en`` produced an axis measuring a different quantity under the same column name."
- destination: policy/triage design (U1/U2 escalation)

### A-93
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:866-877 (`_i2_execute`)
- quote: "the published axis went 0.288 -> 0.608 while the deliverable still read 0.1196 — which is the defect the attribution axis exists to remove."
- destination: identity-repair design

### A-94
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/loop.py:583-593 (`_resolve_input_audio`)
- quote: "The root is inferred from the path *shape*, which is fragile: it walks a fixed number of parents rather than looking for a marker. It happens to work for the default output layout and would not for an arbitrary ``--out-dir``."
- destination: adaptive loop design

### A-95
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/ls_final.py:75-80 (`build_final_ls_bundle`)
- quote: "While it lived in ``final/`` this stage read it back out of the directory it was about to write... 'not found' being indistinguishable from 'no bundle'."
- destination: LS-export design

### A-96
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/ls_final.py:217-225 (`_final_belief_index`)
- quote: "This function used to apply its own most-doubtful collapse, ``adaptive.plot`` filtered to the fusion stream, and ``evaluate`` filtered to the transcript's — three answers from one file, only one of which was written down."
- destination: belief/fusion design

### A-97
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:24-29 (module docstring)
- quote: "on a run whose mask derivative is a single ``target_active`` region at uncertainty 0.0 the final figure showed one flat confident band while ``L2/round/<n>/timeline.png`` showed the same axis varying across 1070 buckets."
- destination: visualization design

### A-98
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:716-721 (`_fused_axis`, remainder after A-4's stale claim)
- quote: "**This function is scaffolding for a defect, and should be deleted rather than maintained.** ... Remove that fold and the belief store has nothing to read but L2's axes — one number, and no reason for this comparison to exist."
- destination: visualization design / belief-store cleanup backlog — self-marked deletion note worth preserving even after A-4 is fixed

### A-99
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:20-25 (`load_policy`)
- quote: "It lived at ``adaptive/policy/default.yaml`` beside a CLI that also carried model ids, grids and stage switches as flags, so a run's configuration was spread across a file and seventy arguments..."
- destination: policy/triage design

### A-100
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:74-79 (`_validate_floors`)
- quote: "A zero floor therefore restores erasure through configuration — silently, and everywhere at once. A floor that can be configured to zero is not a floor, which is why this raises rather than clamping."
- destination: policy/triage design

### A-101
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/provenance.py:6-16 (module docstring)
- quote: "uncertainty can fall for two completely different reasons... Both look identical in the number alone. A loop that cannot distinguish them converges on its own edits..."
- destination: provenance/mutual-influence design (pair with influence.py's own docs) — valuable independent of A-5's wiring gap

### A-102
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/regions.py:20-24 (`propose_regions`)
- quote: "Proposing per (pass, axis) produced two overlapping regions for one ambiguity, each spending budget separately, and made the intervention catalogue's target a property of which pass happened to look worse."
- destination: region-proposal design

### A-103
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/adaptive/triage.py:1-13 (module docstring)
- quote: "Design follows ``SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md``: the speech gate is driven by **continuous frame posteriors**... never segmentized VAD, whose hysteresis erases brief events."
- destination: policy/triage design

### A-104
- class: rationale-to-migrate
- location: src/senselab/audio/workflows/audio_analysis/acoustic.py (module docstring area)
- quote: (LUFS-vs-percentile loudness rationale; module docstring)
- destination: quality/degradation design (loudness measurement) — noted by the batch as legitimate rationale, sampled rather than quoted at length here

---

## Class: restates-code (sampled)

### A-105
- location: src/senselab/audio/workflows/audio_analysis/grid.py:24-31 (`BucketGrid` attributes)
- quote: "win_length: Bucket length in seconds. Must be > 0."
- why: Plainly restates the `__post_init__` validation two lines below with no added information.

### A-106
- location: src/senselab/audio/workflows/audio_analysis/aggregators.py:49-52 (Raises section)
- quote: "Raises: ValueError: If ``name`` is not a recognized aggregator, or ``weights`` is present with a different length than ``sub_signals``..."
- why: Restates the two `if` guard clauses immediately below in the function body.

### A-107
- location: src/senselab/audio/workflows/audio_analysis/level.py:100-101 (`AudioVariant.to_json`)
- quote: "Serialize, encoding non-finite loudness as ``None`` so the JSON stays valid."
- why: Describes exactly what the one-line body does.

### A-108
- location: src/senselab/audio/workflows/audio_analysis/layout.py:87-89 (`evidence_dir`)
- quote: "``<run>/L1`` — everything measured, nothing concluded."
- why: The path half restates the one-line function body; only "nothing concluded" carries content, already stated at length in the module docstring above.

### A-109
- location: src/senselab/audio/workflows/audio_analysis/speaker.py:79-88 (`_cosine_similarity`, `_cos_dist`)
- quote: "Cosine similarity between two equal-length vectors. Returns None on bad inputs."
- why: Both docstrings just name what the four-line function bodies already make obvious.

### A-110
- location: src/senselab/audio/workflows/audio_analysis/attribution.py:43-47 (`_binary_entropy`)
- quote: "Normalised Shannon entropy of a two-outcome split; 0 unanimous, 1 evenly split."
- why: The three-line body is the textbook binary-entropy formula; adds nothing beyond the function name.

### A-111
- location: src/senselab/audio/workflows/audio_analysis/disagreements.py:23-29 (`_row_summary`)
- quote: "The fused row says how much doubt there is and which signals carried it; evidence is the L1 per-signal measurement for the same bucket..."
- why: Legitimate rationale, but stated at length for a formatting helper that concatenates two dicts into a string — could be one sentence.

### A-112
- location: src/senselab/audio/workflows/audio_analysis/signal.py:79-93 (`SignalProvenance.to_json`)
- quote: "Serialise for the parquet sidecar and the run summary."
- why: One-line docstring on a method that is a dict literal of the dataclass fields.

### A-113
- location: src/senselab/audio/workflows/audio_analysis/acoustic.py:50-56, 105-113 (`lufs_track`, `loudness_confidence_track`)
- quote: "Returns: (times, lufs), floored at LUFS_FLOOR." / "Convenience: the LUFS track mapped through loudness_confidence."
- why: Both restate the return tuple / one-line composition the signature and body already convey.

### A-114
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:89-92, 108-145 (five one-line property docstrings)
- quote: "Region length in seconds." / "Total duration of target-free regions (FR-038)." / "Number of target-free regions."
- why: Each adds nothing past what the one-line property body already says.

### A-115
- location: src/senselab/audio/workflows/audio_analysis/background_mask.py:416-418 (`_overlaps`)
- quote: "True when two half-open intervals intersect."
- why: Exact restatement of the one-line boolean expression below it.

### A-116
- location: src/senselab/audio/workflows/audio_analysis/sources.py:269-272 (`ExcisedSegment.duration_s`)
- quote: "Segment length in seconds."
- why: Exact restatement of `max(0.0, self.end - self.start)`.

### A-117
- location: src/senselab/audio/workflows/audio_analysis/sources.py:189-191 (`is_quarantined`)
- quote: "Whether ``label`` is one an amplified noise floor is known to produce."
- why: Close paraphrase of the one-line set-membership check.

### A-118
- location: src/senselab/audio/workflows/audio_analysis/sound_sources.py:64-72 (`_category_for`)
- quote: "Resolve one AudioSet display name to a category, warning once if unmapped."
- why: Mirrors the function body (dict lookup + warn-once) with no added information.

### A-119
- location: src/senselab/audio/workflows/audio_analysis/speech_presence.py:73-87 (`_row_window_overlap`, `_mean_col`)
- quote: "Return the subset of feature rows whose window overlaps ``[start, end)``." / "Mean of column ``col`` across rows, ignoring None / non-numeric values."
- why: Both are direct restatements of the loop-and-filter bodies beneath them.

### A-120
- location: src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:129-141 (`_finite`, `_ramp`, `_mean`)
- quote: "Coerce an evidence field to a finite float, or ``None`` if it is not a usable number."
- why: Pure restatement of trivial helper bodies.

### A-121
- location: src/senselab/audio/workflows/audio_analysis/quality.py:194-202, 383-391 (`_finite_or_none`, `_as_optional_float`)
- quote: "Coerce to float, or ``None`` when absent or non-finite." (both, near-identically)
- why: Two functions with near-duplicate bodies and near-duplicate docstrings — a simplification candidate as well as a prose one.

### A-122
- location: src/senselab/audio/workflows/audio_analysis/pii.py:261-268 (`report_to_dict`)
- quote: "Every span in a ``PiiPassReport`` was scanned for the same pass, so rather than carrying a redundant per-span ``perturbation`` field..."
- why: Mostly restates the dict comprehension immediately below it.

### A-123
- location: src/senselab/audio/workflows/audio_analysis/occupancy.py:160-169 (`_union_length`)
- quote: "Total length covered by ``intervals``, counting overlap once."
- why: Exact restatement of the sweep-line accumulation below it.

### A-124
- location: src/senselab/audio/workflows/audio_analysis/shapes.py:220-238 (`LabelScore`, `Window`)
- quote: "One label and its score, in the classifier's own units." / "One analysis window's label scores, descending."
- why: Trivial one-line restatements of two-field dataclasses.

### A-125
- location: src/senselab/audio/workflows/audio_analysis/perturbations.py:131-166 (`is_identity`, `to_json`, `from_json`)
- quote: "Is this the untransformed recording?" / "The register entry for this perturbation." / "Rebuild a perturbation from its register entry."
- why: Each mirrors a one-line equality check or dict round-trip with no added information.

### A-126
- location: src/senselab/audio/workflows/audio_analysis/sampler.py:78-81, 295-306 (`stats` property, `_reduce`)
- quote: "``{'hits', 'misses'}`` — how much recomputation the cache avoided." / "``mean`` / ``max`` / ``min`` over the measured values, or ``None`` when there are none."
- why: Both restate the return value/branching in the one-liners below them.

### A-127
- location: src/senselab/audio/workflows/audio_analysis/rounds.py:56-57 (`_overlaps`)
- quote: (unnamed helper, no docstring; near-verbatim duplicate of `sources.py`'s `_overlaps` with no cross-reference)
- why: Duplicate logic across two modules with no shared reference — flagged because a reader of one module's rationale won't know the other copy exists.

### A-128
- location: src/senselab/audio/workflows/audio_analysis/global_summary.py:20-37 (module docstring, "n_speakers semantics")
- quote: "``n_speakers == 0`` → recording without anyone speaking... ``expects_speech=True`` says 'we expected a speaker; absence violates the single-speaker claim'."
- why: Close paraphrase of the `if/elif` chain at lines 292-306 — closer to restates-code than rationale despite its framing.

### A-129
- location: src/senselab/audio/workflows/audio_analysis/global_summary.py:160-174 (`_aggregate_quality`)
- quote: "PESQ (1–4.5): clean speech > 3.5; degraded < 2.5. Uncertainty rises below 3.5, saturating below 2.0."
- why: Re-derives, almost line for line, the `ramp(...)` calls and thresholds two lines below it.

### A-130
- location: src/senselab/audio/workflows/audio_analysis/plot.py:57-77, 251-260 (`_series_for`, `_pass_color_alpha`)
- quote: "Return ``(centers, values)`` for plotting one fused axis line in [0, 1]."
- why: Mostly walks through the code rather than adding new information (though the "why per-row midpoint, not derived count" clause is legitimate — mixed).

### A-131
- location: src/senselab/audio/workflows/audio_analysis/l1_plot.py:212-217 (`classify_signal`)
- quote: "Group a signal name by the kind of evidence it is."
- why: First sentence restates the function name; the second sentence (naming convention rationale) is legitimate — mixed, first half flagged.

### A-132
- location: src/senselab/audio/workflows/audio_analysis/l2_plot.py:32-52 (`build_round_timeline`)
- quote: "Draw one row per axis per fused quantity for a single round." (plus an Args section restating each parameter name)
- why: Adds little past the signature; the one load-bearing sentence (empty-figure ambiguity) is the only rationale in the block.

### A-133
- location: src/senselab/audio/workflows/audio_analysis/labelstudio.py:112-140, 341-343 (`_track_name`, `_signal_track_name`, `_scene_track_name`, `_new_region_id`)
- quote: "Track carrying one fused axis. No pass token — an axis has no pass."
- why: Pure string-formatting helpers whose docstrings restate the f-string beneath them (the "no pass token" clause is a legitimate one-line rationale attached to an otherwise trivial restatement).

### A-134
- location: src/senselab/audio/workflows/audio_analysis/foreground.py (short-file sample)
- quote: (one-line restatement of a return-tuple helper)
- why: Sampled per batch's methodology note — foreground.py is short and mostly rationale (A-44/A-45), with one minor restates-code instance.

### A-135
- location: src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:31-32 (`_to_audio`)
- quote: "Wrap a 1-D float32 numpy crop as a 16 kHz mono senselab ``Audio``."
- why: The one-line body says exactly this; nothing non-obvious is added.

### A-136
- location: src/senselab/audio/workflows/audio_analysis/adaptive/convergence.py:104 (`round_summary`)
- quote: "One ``rounds/<k>/summary.json`` payload."
- why: Purely names the return artifact; self-evident from the function's own `return {...}`.

### A-137
- location: src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:4-22 (module docstring, steps 1-5)
- quote: "1. Per embedding model, L2-normalize the per-window vectors and compute the adjacent-window cosine-distance trajectory ... 5. Output refined segments/clusters ..."
- why: A near line-by-line paraphrase of `change_point_trajectory`, `detect_change_points`, `_voiced_spans`, the pooling/clustering block, and `repair_identity`'s return — no more informative than reading the five functions.

### A-138
- location: src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:140-141 (`BudgetLedger`)
- quote: "Per-run intervention budget by cost class (FR-018). Light is uncapped."
- why: The class body two lines below (`self.caps = {"light": None, ...}`) makes "light is uncapped" immediately visible; the FR pointer is the only non-redundant part.

---

## Checked and clean

Every one of the 81 files was read in full by its assigned batch. Files below produced no flagged
candidate at all, or (where noted) produced candidates that are recorded above rather than implying
anything wrong with the rest of the file's prose.

**Batch 1 (grid/fuse/core infra):** `axes.py` (heaviest single prose file in this batch, entirely
load-bearing — see A-11; cross-checked `AXES`/`AXIS_NAMES`/`HARVESTED_AXES` derivations against
their actual comprehensions), `fuse.py` (verified `fuse_axis`/`fuse_axes`/`SnrGate` defaults against
docstrings — see A-10 for its one sampled rationale item), `votes.py` (PassHarvest field docstrings
checked against `compute.py`'s construction call — clean), `harvesters.py` (lazy-import claims
verified — clean), `keys.py` (Route/Operator/SignalKey path-arity logic checked against
`__post_init__` — clean), `types.py` (field lists checked, not claimed exhaustive so not flagged —
clean), `contracts.py` (STAGE_CONTRACTS/MODULE_STAGE/KNOWN_DEVIATIONS cross-checked against the
actual module list — see A-14), `stage_context.py` (STAGE_VERSIONS keys checked against `stages.py`
task strings — see A-3, A-12), `stage_io.py` (capability table verified line-by-line against
`Stage.is_round_scoped` — clean), `stages.py` (background-mask measurement cross-checked against
axes.py's claim — clean), `io.py` (all writers besides A-2 checked against actual column
lists/call sites — clean), `compute.py` ("no per-axis grid parameter" claim verified against the
actual signature — clean), `aggregate.py` (deletion claim verified — none of the three named
functions exist — clean), `aggregators.py` (default aggregator cross-checked against
`run_config.py` and `default.yaml` — see A-106), `run_config.py` (AGGREGATORS tuple duplication
checked, no drift — see A-13), `layout.py` (every helper checked against contracts.py's declared
patterns — see A-108), `resolution.py` (NATIVE_RESOLUTION_S entries checked against
`stage_context.py` defaults — see A-15), `floors.py` (positive-floor claim spot-verified against
`support.py`'s guard — clean), `__init__.py` (see A-1 — the one file in this batch with a
package-level stale claim), `level.py` (+10 dB cap claim cross-checked against `default.yaml` —
see A-107).

**Batch 2 (ASR/speaker/identity/stats):** all 21 files read fully; no `stale-or-false` survived —
every "used to X until <date>" framing checked (`asr.py`'s consensus_words history, `speaker.py`'s
same-speaker-as-before history, `embeddings.py`'s p_voice removal) correctly describes a *past*
mechanism, not a current one. Symbol references (`fuse.fuse_axis`, `HARVESTED_AXES`,
`AXIS_PRIORITY`, `per_signal_uncertainty`) all resolved to real current definitions. Numeric
defaults spot-checked: `cluster_cosine_threshold=0.5` (speaker.py), `INVARIANT_PERTURBATIONS`
constants (invariance.py), detection-margin "3/6/10 dB" ladder (calibration.py) — all consistent.
One item flagged for content risk rather than falsity: `support.py`'s `MIN_LOW_FRACTION` docstring
(A-31) cites measured numbers it simultaneously disowns as taken under a since-fixed reading bug.

**Batch 3 (background scene/speech presence/plot):** all 21 files read fully; no `stale-or-false`
confirmed — every count claim checked out: `sources.py`'s "3/6/10 dB ladder" against
`assign_tier`'s defaults, "four fabrication guards" against `screen_candidate`'s four reject
branches, `sound_sources.py`'s "AST + YAMNet" two-classifier claim against the actual
`("ast", "yamnet")` loop (not three), `quality.py`'s 0.5 s/0.25 s window against
`QUALITY_ANALYSIS_WIN_S`/`HOP_S`, `mask_harvest.py`'s three-source claim against `MASK_SOURCES`,
`shapes.py`'s "six shapes" against the `Measurement` union, `l2_plot.py`'s four-quantity claim
against `_QUANTITIES`. `plot.py`'s "6-row" layout claim was checked against the actual dynamic
row-index construction and found consistent (optional rows are documented as optional elsewhere in
the same docstring).

**Batch 4 (adaptive/ subpackage):** all 18 files read fully. Five stale-or-false candidates
surfaced (A-4 through A-8, all independently re-verified above). Everything else in this batch —
`backends.py`'s per-rule routing docstrings, `belief.py`'s `replay_check`/`fused_parity` mechanism
description (correctly describes the current, non-oracle design), `convergence.py`'s C1-C4
criteria, `corroboration.py`'s exclusion-of-ASR-voters claim, `evaluate.py`'s scoring methodology,
`fusion.py`'s four-axis claim, `identity_repair.py`'s five-step docstring (accurate, though flagged
restates-code — A-137), `ls_final.py`'s belief-vs-final placement rule, `policy.py`'s deep-merge and
floor-validation description, `regions.py`'s seed/expand/merge/rank pipeline, `triage.py`'s
windowing and SNR-floor math — checked against the code and found consistent.
</content>
