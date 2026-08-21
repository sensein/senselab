# Remediation decomposition: from `analyze_audio` (refiner) to task + workflow + triage graph

Companion to the audit (`summary.md`, `register.md`) and to `doc.md` /
`specs/20260728-221507-per-speaker-identity-scene/layered-architecture.md` (D-17's "pipeline is a
DAG of workflows" already argues the shape this document applies one level up: L1/L2/final are
workflows *inside* `analyze_audio`; this document asks which pieces should stop being inside it at
all).

**The architectural directive, restated:** signal extraction → a **task**
(`senselab/audio/tasks/` or `senselab/utils/tasks/`). A reusable multi-step chain that more than
one workflow wants → its own **workflow**, separate from whichever workflow first needed it.
`analyze_audio` (the refiner) keeps only what is genuinely about iteration: the adaptive loop, the
belief store, intervention rules, convergence. A second workflow, the **triage graph** (single-pass:
review flag, transcript, speaker count, PII, quality, task match, trim), is planned to reuse the
refiner's tasks and chains without inheriting its loop.

**A naming collision to fix before the triage graph ships.** `adaptive/triage.py` already exists —
it is the refiner's *round-0 cheap-signal gate* (`triage_decision`, `dsp_snr_series`, pure numpy,
called from `scripts/analyze_audio.py` before round 1 to decide whether enhancement is needed).
Its own docstring calls it "reusable... by ad-hoc analyses," which is true, but it must never be
confused with the planned single-pass **triage graph** workflow — the coincidence of the word
"triage" is not a signal of shared design. Rename this module (e.g. `adaptive/round0_gate.py`)
before a second "triage" concept ships into the same package.

---

## Inventory 1 — signal extraction that should become a task

**16 candidates**: the 5 the audit already flagged `promotion-candidate` (F-142, F-148, F-152,
F-153, F-160), plus 11 more found by reading every computation-layer file in `audio_analysis/` for
a function that takes audio (or a model's output) and returns a measurement with no dependency on
`Region`/`VoteStore`/`StageContext`/round state.

### Already flagged by the audit (recapped, not re-derived)

| id | file:line | extracts | blocker |
| --- | --- | --- | --- |
| F-142 | `level.py:129-289` | BS.1770/EBU-Tech-3342 loudness, gain, clipping (`apply_gain_db`, `integrated_lufs`, `loudness_range_lu`, `true_peak_dbtp`, `clipped_fraction`, `normalization_gain_db`, `peak_limited_gain_db`) | none — pure `(waveform, sr) -> value`; target `senselab.audio.tasks.quality_control` |
| F-148 | `statistics.py:51,81,112,132` | vote confidence, population std-dev, normalized Shannon entropy, entropy mutual-information decomposition | none — pure `Sequence[float]`/`Mapping[str,float]` in; target new `senselab/utils/tasks/uncertainty.py` (same ask as `project_mc_dropout_optional`) |
| F-152 | `acoustic.py:50-76,127-167` | BS.1770 short-term loudness track, bias-corrected percentile floor-excess track | none — pure `(waveform, sr)->(times, values)`; target `senselab/audio/tasks/features_extraction/loudness.py` |
| F-153 | `occupancy.py:133-169` | interval-algebra occupancy (clip + union length) | only the `Spans`/`Span` dataclass shape at the boundary; target `senselab/utils/tasks/` — **and see N-9 below: at least two duplicate reimplementations of the same interval algebra exist elsewhere, so this move is a dedup opportunity, not just a relocation** |
| F-160 | `adaptive/identity_repair.py:46-50,53-78,125-149` | L2-normalize, adjacent-window cosine trajectory, agglomerative cosine clustering | leading-underscore naming only; target `senselab/utils/tasks/` or `speaker_embeddings/` — **and see N-6: this is one of two independent speaker-clustering implementations in the package (the other is `embeddings.py`'s spectral/k-means path); both should land in the same target module** |

### New candidates found in this pass

| id | file:line | extracts | workflow-state deps | proposed task module | blocker |
| --- | --- | --- | --- | --- | --- |
| N-1 | `quality.py:156-191` (`_rolloff_hz`) | spectral roll-off frequency via raw `torch.stft` | none | `senselab/audio/tasks/features_extraction/` (spectral rolloff) | genuine gap in an otherwise well-factored L1 harvester — every *other* signal in `quality.py` already delegates to an existing task (`quality_control.metrics`, `scene_quality.brouhaha`); only this one measurement was written inline instead of calling one |
| N-2 | `invariance.py:65-125` (`perturb`, `invariance_score`) | output-preserving audio perturbations (gain/shift/DC-offset) + a graded invariance score from re-run answers | none — only couples to `floors.MIN_EVIDENCE_WEIGHT`, a repo-wide constant | `senselab/utils/tasks/` (pairs naturally with F-148's target `uncertainty.py`) | `perturb`'s three transforms likely duplicate existing `senselab.audio.tasks.data_augmentation` primitives — check for consolidation before adding a fourth gain/shift implementation. `probe_diarization_invariance` itself stays workflow-side (injects a `run_diarization` callable) |
| N-3 | `noise_floor.py:69-645` (nearly the whole file: `third_octave_bands`, `estimate_band_floor_db`, `resolvable_bands`, `estimate_noise_floor`, `band_excess_db`, `binding_floor`, `prominence_ratio_db`, `estimate_recorder_floor_db`, `detect_stationary_sources`, `foreground_background_ratio_db`, `cross_recording_baseline`) | bias-corrected per-band noise floor (ECMA-74 prominence), cross-recording equipment/room separation — the single largest missed candidate in the audit, exactly the DSP module `doc.md` documents by name | none from workflow state — only `calibration.quantile_bias_correction_db` (itself pure math) as an optional default | `senselab/audio/tasks/quality_control/noise_floor.py` | none structural — `load_detection_margin_profile` (workflow JSON schema) is pulled in only as a lazy default; a caller passing explicit thresholds bypasses it entirely, so this is an import/default-arg change, not a redesign |
| N-4 | `sources.py:134-150,471-527` (`spectral_flatness`, `modulation_depth`) | spectral flatness `[0,1]`, AM modulation depth | none — pure numpy | `senselab/audio/tasks/features_extraction/` | none. **Not** promotable from the same file: `assign_tier`, `screen_candidate`, `plan_excision`, `route_classifier` blend measurement with the 3/6/10 dB tier-ladder decision and margin-profile policy — correctly L2, stays in the workflow (Inventory 2). Also note: `screen_candidate`/`plan_excision` have **no callers anywhere in the pipeline today** (only their own tests) — designed and tested but not integrated |
| N-5 | `foreground.py:43-107` (`project_onto`, `suppression_depth_db`, `leakage_margin_db`) | suppression-depth measurement via signal projection (not level) — the reasoning `doc.md` cites for why amplification cannot fix a buried source | none — pure numpy in, float out | `senselab/audio/tasks/quality_control/` or a `speech_enhancement` metrics submodule | `ForegroundSuppression` (the dataclass wrapping these, with `.is_deep_enough_for`) mixes in a threshold decision — leave that at L2. Also note: `suppress_foreground`, the function that would call this in production, has **no callers anywhere in the pipeline today** — same unwired state as N-4 |
| N-6 | `embeddings.py:97-393,409-657` (`cluster_pass_speakers`, `_merge_close_clusters`, `_empirical_calibration_band`, `_sequential_calibration_band`, `_within_cluster_band`, `calibrate_cosine_uncertainty`) and `clustering.py:86-199` (`assign_unified_clusters_with_seed_phase`) | windowed-embedding spectral/k-means clustering with silhouette k-selection, per-embedder empirical same/different-speaker cosine calibration (the part F-164's refutation found *already correct*), cross-model centroid unification | none — plain `WindowEmbedding`/`np.ndarray`/dict in and out | `senselab/audio/tasks/speaker_embeddings/` | none structural — the strongest "already nearly a task" candidate found. **Flag: this is a second, independent speaker-clustering algorithm (spectral/k-means) from F-160's agglomerative one in `identity_repair.py` — both should land in the same target module, not as two parallel promotions that leave two competing clustering implementations in two different places** |
| N-7 | `harmonize.py:117-211,434-463` (`_overlap_seconds`, `_maximise`, `overlap_assignment`, `_cosine`, `centroid_assignment`, `_align_pair`) | segment-overlap Hungarian-style assignment, centroid-cosine assignment, Levenshtein sequence alignment — the primitives under "harmonising labels/transcripts across diarizers/ASR backends" | none — pure `Sequence[Segment-like]`/`Sequence[str]` in, mapping/alignment out | `senselab/utils/tasks/` (generic assignment/alignment primitives, not audio-specific once segments are `(start, end, label)` tuples) | `harmonize_speaker_labels`/`harmonize_from_diarization`/`harmonize_transcripts` (same file) are **not** promotable — they build confidence/uncertainty via `statistics.py` and are the chain-orchestration layer (Inventory 2) that calls these primitives across multiple diarizers/ASR models |
| N-8 | `adaptive/backends.py:237-279` (`overlap_track_from_spans`) | per-frame overlap decision from cross-diarizer spans (F-174's mechanism) | depends on `occupancy.spans_from_diarization`/`count_at` | same target as F-153 (`senselab/utils/tasks/`), once that move happens | blocked on F-153 moving first, not on anything of its own |
| N-9 | `adaptive/triage.py:21-155` (`triage_decision`, `dsp_snr_series`) | frame-posterior speech/enhancement gating decision + DSP SNR-per-frame series | none — confirmed "pure (numpy only)" by its own docstring | `senselab/audio/tasks/voice_activity_detection/` (`triage_decision`) and `quality_control/` (`dsp_snr_series`) | `dsp_snr_series` overlaps with `spectral_gating_snr_metric`/`peak_snr_from_spectral_metric` already used in `quality.py` — reconcile three SNR estimators rather than promote a fourth verbatim. **This is the same module flagged in the intro for the triage/triage-graph name collision — moving its functions to task modules does not make this module become or feed the triage graph** |
| N-10 | `attribution.py:50-103,106-160` (`speaker_assignment_doubt`, `word_coverage`) | entropy over a label-share mapping; interval-union word coverage | none | no new module — **both duplicate an existing promotion target**: `speaker_assignment_doubt` restates `statistics.entropy_uncertainty` (F-148's target), `word_coverage` restates `occupancy.py`'s interval algebra (F-153's target) | this is a consolidation opportunity, not a new destination — moving F-148/F-153 first and then having `attribution.py` import the promoted versions removes the duplication in the same step |

**Total inventory 1: 16.** The 3 highest-value promotions, ranked by how much duplicate or ad-hoc
reimplementation they prevent for a triage graph that needs its own speaker count, quality score,
and confidence statistics from day one with no import into the refiner package:

1. **N-6** (`embeddings.py`'s clustering + cosine calibration) — the triage graph cannot produce a
   speaker count without this, it is the cleanest of all 16 to move, and moving it forces the
   F-160 dedup (two clustering algorithms, one destination) to happen at the same time.
2. **F-142** (loudness/gain/clipping) — the "recording quality" triage output needs exactly this
   and nothing else new.
3. **F-148** (uncertainty statistics) — every one of the seven triage outputs that carries a
   confidence score needs `confidence`/`entropy_uncertainty`; today a triage graph would have to
   import `audio_analysis.statistics` directly to get a five-line entropy formula, coupling its
   dependency graph to the whole refiner package for that alone.

(**N-3**, `noise_floor.py`, is the single largest candidate by line count and is a close fourth —
held out of the top three only because the quality/scene chain it feeds, Inventory 2 below, is
already partly unwired in production, so promoting it alone does not unblock the triage graph's
"recording quality" output the way N-6/F-142/F-148 do.)

---

## Inventory 2 — reusable chains that should become their own workflow

### `stages.py` is already the shared single-pass extraction layer — for raw signals only

Direct reading of `stages.py` (836 lines) and `stage_context.py` confirms: the six `stage_*`
functions plus `run_pass` call tools (diarize, ASR, AST/YAMNet, features, alignment,
background_mask/sources) and return **plain dict fragments** — no `VoteStore`, no `Region`, no
round numbers anywhere in the file. `StageContext.perturbation` is an audio-variant label
(`raw`/`enhanced`), not an adaptive-loop round, and `PassPlan` makes every stage optional (empty
tuples/`None` = skip) because it doesn't run *axes* at all — only raw per-tool signals. This is
already exactly what a triage graph needs for extraction, and is already documented (`doc.md`,
"Importable pipeline") as callable in-process independent of the loop, for the same reason the
adaptive loop needed it.

**What is not yet reusable is one layer down.** Embedding/clustering, ASR-stream fusion,
harmonization/attribution, and quality→degradation anchoring all live inside `compute.harvest_pass`
(601 lines), which **is** axis-shaped: it returns a `PassHarvest` consumed by
`votes.link_pass`/`fuse.fuse_axis` — the round/`L2/round<N>` machinery. The four chains below are
presently mixed into `harvest_pass` alongside vote-bucket wrapping that is refiner-specific; each
chain's own functions, read directly, are already round-agnostic.

### The four chains

**1. window-and-embed-and-cluster**
- Steps: `stage_diarization`/scene stages (context) → windowed per-window embeddings
  (`embeddings.py`) → `_speech_window_mask` veto (`compute.py:890-1009`, YAMNet>AST>loudness) →
  `cluster_pass_speakers` (spectral/k-means + silhouette) → synthetic `spk*` diarization block.
- Who reuses it: refiner and triage — triage's "speaker count" output is exactly
  `n_speakers`/`best_silhouette`.
- Entanglement with refiner vocabulary: none in the chain itself (plain `Audio`/`pass_summary`
  dict/lists in, plain dict out); it is entangled only by being invoked *inside* `harvest_pass`.
- Carries forward, unmodified: **F-170** (YAMNet veto drops child/infant vocalizations before
  clustering ever sees them), **F-164** (fixed adult-derived cosine thresholds 0.5/0.55/0.30/0.70,
  verified-latent), **F-167** (the resulting speaker-count posterior carries no population
  caveat, verified-latent).
- Proposed module: `senselab/audio/workflows/speaker_clustering/`

**2. transcribe-then-align-then-fuse-across-backends**
- Steps: `stage_asr` → `stage_alignment` (`stages.py`) → `resolve_asr_result` (`harvesters.py`) →
  `fuse_consensus_words` → `speech_to_text_ensemble.fuse_word_streams` (phoneme-graded via
  `asr.phoneme_similarity`) — the fusion math itself is already a re-export from a `senselab.audio.
  tasks.speech_to_text_ensemble` task, confirming this chain is largely already reusable.
- Who reuses it: refiner and triage — triage's "transcript" output is exactly the fused word
  stream; the doubt-scoring wrapper (`harvest_asr_votes`, per-bucket vote dicts) stays with the
  refiner.
- Entanglement: `fuse_consensus_words`/`fuse_word_streams` are already round-agnostic
  (`Mapping[model→result]` in, plain `(words, provenance)` out); `harvest_asr_votes` is
  vote/bucket-shaped and belongs to the refiner, not this chain.
- Carries forward, unmodified: **F-162** (`harvest_pass`'s call to `fuse_consensus_words
  (asr_resolved)` at `compute.py:433` omits `policy=`, so `linking.asr_slot_overlap`/
  `asr_slot_mid_tol_s` config is decorative on the only reachable production call path — a triage
  graph lifting this chain inherits a config knob that reads as live but is not, unless fixed
  first). Adjacent (feeds the same transcript's reliability, different file): **F-166** (Whisper
  `no_speech_prob` over-trusted on non-lexical vocalization, verified-latent).
- Proposed module: `senselab/audio/workflows/transcript_fusion/` (or lift `fuse_word_streams`'s
  existing task home outward and add the alignment step beside it)

**3. the diarize-harmonise-attribute chain**
- Steps: `stage_diarization` → `harmonize_from_diarization`/`harmonize_speaker_labels` (overlap +
  centroid matching → harmonized `C*` cluster space) → `bind_labels`/`per_speaker_presence`
  (`identity_binding.py`, Hungarian match to the fused `S*` ids).
- Who reuses it: refiner and triage — triage's "speaker count" (the cross-diarizer harmonized
  count) and speaker identity context for the PII and review-flag outputs.
- Entanglement: all four functions are pure over plain segments/spans — no
  `StageContext`/`PassPlan`/`Region`/`VoteStore`. The only round-flavored caller found is
  `fuse._speaker_assignment`, which picks the `"raw"` pass out of a `harvests` mapping for the C2
  convergence check — that caller, not the chain, is refiner-specific. `attribution.py`'s
  doubt-scoring (`speaker_assignment_doubt`, `word_coverage`, `target_activity_doubt`) is a
  separate, refiner-only vote layer (feeding the `speaker` axis) that stays behind with
  `harvest_speaker_votes`.
- Carries forward: **F-165** (`fused_words`-empty bucket zeroes the *entire* votes dict, discarding
  non-lexical child vocalization — demonstrated), **F-173** (250 ms minimum segment/adult change-
  point assumption, verified-latent) via the identity-repair path this chain feeds into.
- Proposed module: `senselab/audio/workflows/speaker_identity/`

**4. quality/scene measurement** (largest, and what the "recording quality"/"trim regions" triage
outputs depend on almost entirely)
- Steps: Brouhaha frames → `harvest_quality_measurements` (`quality.py`, dB/Hz/proportion per
  bucket) → `degradation.py` (`snr_degradation`/`reverb_degradation`/`bandwidth_degradation`/
  `clip_degradation`, anchored via a calibration profile) — separately, `stage_background_mask`/
  `stage_background_sources` (`stages.py`) call `noise_floor.py` + `background_mask.build_mask` +
  `sound_sources.py`'s source-category mapping.
- Who reuses it: refiner and triage.
- Entanglement: `harvest_quality_measurements` is already round-agnostic; the anchoring functions
  in `degradation.py` are applied once in `votes.link_pass` (round-agnostic) and again per-round in
  `adaptive/belief.py`/`adaptive/interventions.py` — those two re-applications are refiner-only,
  the anchoring functions are not. The mask/noise-floor/sources stages are already wired through
  plain `StageContext`, confirming they are reusable as-is.
- **Important caveat for this chain specifically**: `sources.py`'s `screen_candidate`/
  `plan_excision` and `foreground.py`'s `suppress_foreground` have **no callers anywhere in the
  pipeline today** (only their own tests) — designed and tested, not integrated. A triage graph
  built against "what the quality/scene chain does today" must not assume this sub-chain runs; it
  would need to be wired in explicitly, or excluded and stated as excluded.
- Carries forward: **F-149** (PESQ/STOI/SI-SDR ramp contradicting its own docstring), **F-169**
  (fixed 25 dB/30 dB conversational-speech anchors, no `task_type`, verified-latent), **F-168**
  (background-mask task vocabulary has no cry/babble entries despite the AudioSet map already
  having the labels, verified-latent).
- Proposed module: `senselab/audio/workflows/scene_quality/`

---

## Inventory 3 — genuinely refiner-only (stays)

Confirmed by direct reading of every module under `adaptive/` plus root-level `rounds.py`.

| module | why it is iteration-specific, not reusable |
| --- | --- |
| `adaptive/loop.py` | the literal round loop (`for round_idx in range(baseline+1, baseline+max_rounds)`), `run_state` transitions, `touch_counts` keyed across rounds |
| `adaptive/belief.py` (`VoteStore`) | exists to hold votes with mutable `active|shadowed` status and weight-withdrawal (`attenuate_source_in_bucket`) that persists *across* rounds, plus `replay_check` proving round-to-round re-derivability — a single pass has no re-aggregation to replay |
| `adaptive/interventions.py` | every rule (S1, P2/P3, U1/U2, I1/I2, I4, C9) is defined by `trigger`/`guard`/`gain`/`execute` keyed to being admitted into *a* round via `policy.plan_round`; the proposal/admission machinery is the iteration-specific part, not the underlying model calls the rules wrap |
| `adaptive/convergence.py` | reads round-indexed `touch_counts`/`history` to decide `converged`/`irreducible`/`budget_exhausted` — undefined outside a multi-round context |
| `adaptive/regions.py` | `propose_regions` takes `round_idx`, produces ids `r<round>_<axis>_<idx>` — exists to say where the *next* round should intervene |
| `adaptive/policy.py` | `BudgetLedger` tracks spend across rounds; `plan_round` admits candidates against it — round-to-round budget state is the load-bearing part (`load_policy`/`family_weights` are reusable config plumbing, incidentally) |
| `adaptive/provenance.py` | **dead code** — F-5 confirmed: zero call sites for `RevisionRecord`/`classify_resolution` outside its own docstring and one test |
| `adaptive/triage.py` | refiner-only (round-0 gate before round 1) as a *module*; its two functions are Inventory-1 promotion candidates (N-9) precisely because they carry no round state of their own — it is refiner-only only in the sense that this particular call site (gating before round 1) belongs to the loop, not that the underlying computation does |
| `rounds.py` (root) | already correctly shared *below* `adaptive/`: imported by both `fuse.py` (the non-adaptive per-round fold) and `adaptive/convergence.py` — evidence that "operates across rounds" already lives at the right place for its two consumers |

**Not refiner-only, despite living under `adaptive/` today:**
- `adaptive/corroboration.py` — every function (`independent_presence_pool`, `corroboration_over_span`, `apply_corroboration`) is a pure measurement over one store/stream snapshot, no round index, no history.
- `adaptive/evaluate.py` — `evaluate_against_ground_truth` reads only `final/*` (transcript, diarization, parquets, `decisions.json`), explicitly documented as scoring "the deliverable and nothing else"; runs once, against any completed output tree regardless of round count.
- `adaptive/identity_repair.py`'s `repair_identity` — single-pass; takes `window_embeddings`, `diar_boundaries`, `p_voice_at`, `duration_s`, `policy` and nothing round-shaped. The refiner-only part lives *outside* this file, in `interventions.py`'s I1/I2 rules that wrap the call with region-scoped trigger/guard/priority, and in `loop.py:406` feeding the result back into belief revision for later rounds.
- `adaptive/fusion.py` — the math (`fuse_word_streams`) is already a re-export from a task module; what remains here (artifact collection, policy→weights translation, `final/` writers) is workflow glue invoked once per round, not inherently round-aware.

**The one-line test that falls out of this reading:** a module is refiner-only exactly when its
data or control flow is *keyed by round index or accumulates state across calls* — round-indexed
history, a budget ledger that must persist between invocations, or existing to decide "should there
be another round." Anything whose output is a pure function of one evidence snapshot is reusable
single-pass computation, regardless of which subpackage it currently sits under.

---

## The triage graph's shape

A single-pass composition, in dependency order: **lifted `stages.py`/`stage_context.py`** (raw
per-tool extraction, already round-agnostic) → the **four Inventory-2 chains** (speaker clustering,
transcript fusion, speaker identity/harmonization, quality/scene) → the **Inventory-1 task
promotions** those chains call into for their pure math (embeddings calibration, loudness,
uncertainty statistics, noise floor, assignment primitives) → seven small, task-specific output
builders, none of which loop, none of which touch `adaptive/`.

**What it needs that does not exist yet:** a package boundary. `stages.py`/`stage_context.py`
today live inside `senselab.audio.workflows.audio_analysis`, so a triage graph that wants only
`run_pass` still imports the refiner package — and with it `contracts.py`, `adaptive/`, and every
other file in the 91-file package it uses none of. The concrete fix: lift `stages.py` +
`stage_context.py` into a new shared location (e.g.
`senselab/audio/workflows/audio_analysis_extraction/`), with `audio_analysis` re-exporting
`run_pass`/`StageContext`/`PassPlan` from there for its own callers. The four chain modules
proposed in Inventory 2 sit beside it, each importable independently.

Per output, what it is built from and whether the audit found a defect in that dependency chain
(register's `graph_implication` column, reused directly):

| output | built from | audit defect in its dependency chain (graph_implication) |
| --- | --- | --- |
| human-review flag | `disagreements.py`'s `build_disagreements_index` (already single-pass; ranks by `triage_score`, no round parameter required to run once) | **F-150** (`disagreements.py:152`, `high_uncertainty_rate=0.0` on total harvest failure reads as a dramatic improvement, not a broken run) — **consumed** |
| transcript | chain 2, transcribe-then-align-then-fuse | **F-162** (`fuse_consensus_words`'s `policy=` dropped at the only reachable call site) — **consumed**; adjacent **F-166** (Whisper `no_speech_prob` over-trust) — **consumed** |
| speaker count | chains 1 and 3, window-and-embed-and-cluster + diarize-harmonise-attribute, plus `speaker_identity.speaker_count_posterior` | **F-144** (decorative `multimodal_threshold`), **F-145** (unexplained `_SUPPORTED_THRESHOLD`), **F-147** (`speech_presence_confidence` unearned on diarizer crash), **F-164** (adult-derived cosine thresholds, verified-latent), **F-167** (no population signal, verified-latent), **F-165** (wordless bucket zeroes all votes, demonstrated), **F-170** (YAMNet veto drops pediatric vocalization pre-clustering, verified-latent) — all **consumed** |
| PII | `pii.py`'s `detect_pii_in_pass` — already single-pass, no round coupling, a thin adapter over the standalone `senselab.text.tasks.pii_detection` task | none found — no register finding (F-1..F-176) touches `pii.py` |
| recording quality | chain 4, quality/scene measurement, plus `global_summary.py` | **F-149** (PESQ/STOI/SI-SDR ramp contradicts its own docstring) — **consumed**; **F-169** (25 dB/30 dB anchor, no `task_type`, verified-latent) — **consumed** |
| task match | `background_mask.py` (`task.type`-conditioned target vocabulary, part of chain 4) | **F-168** (no `cry`/`babble` task-type vocabulary despite the AudioSet map already having the labels, verified-latent) — **consumed** |
| trim regions | chain 4's `background_mask.py` `target_free` regions, `foreground.py`'s suppression depth | **F-172** (`single_speaker_uncertainty` scores any 2+-speaker recording maximally noncompliant, no task-aware exception, demonstrated) — **consumed**, insofar as trim decisions read the same mask/speaker-count machinery; also inherits chain 4's **unwired-suppression caveat** above (`suppress_foreground` has no production caller today, so a naive port would silently skip it) |

**One structural risk specific to this graph, not called out by any single finding:** every
"consumed" defect above was found and reasoned about *inside* the refiner, where the adaptive loop
re-runs and multiple diarizers/rounds partially dilute a single bad vote. A single-pass triage
graph has no such dilution — the same defect (e.g. F-144's decorative threshold, F-162's dropped
`policy=`) governs the *only* pass the graph gets, with no second chance. Config-wiring correctness
(does a declared knob actually reach its call site) therefore matters strictly more for the triage
graph than it did for the refiner, and should be re-verified for every promoted chain rather than
assumed carried over correctly.
