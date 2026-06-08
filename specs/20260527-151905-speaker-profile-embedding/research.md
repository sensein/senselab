# Phase 0 Research: Speaker Profile Embedding

**Feature**: Speaker Profile Embedding for analyze_audio
**Date**: 2026-05-27

This document resolves the open decisions (including the two items the spec deferred to planning: FR-017 brief-intrusion resolution and FR-018 model consensus) and records the reuse strategy against the existing `audio_analysis` workflow.

---

## R1. Cross-stage cache reuse vs. the wrapper-hash

**Decision**: The profile stage reuses the existing cache (`artifacts/analyze_audio_cache/`) at the **task level** (diarization, speaker embeddings, scene classification) by invoking the *same task functions with identical params*. To make hits actually carry over, the cached unit's key must **not** depend on `analyze_audio.py`'s script-source hash. We factor the per-file cache-key computation into a shared library helper whose "wrapper hash" is the **library module hash** (stable across both the `analyze_audio` script and `build_speaker_profile` script), and both stages call it.

**Rationale**: Today `cache_key` in `analyze_audio.py` folds in `wrapper_version_hash = sha256(analyze_audio.py source)`. A separate `build_speaker_profile.py` would have a different source hash → every entry would miss, defeating the user's stated goal ("running beforehand helps analyze_audio not re-run"). Keying on the library function that actually does the work (e.g., the diarization/embedding wrappers) makes the cache identity reflect the computation, not the calling script.

**Alternatives considered**:
- *Per-script wrapper hash (status quo)*: simplest, but no cross-stage reuse — rejected (contradicts FR-015).
- *Single merged CLI that does both*: avoids the issue but couples the stages and breaks the "profile is reusable input" model (FR-006) — rejected.
- *Ignore wrapper hash entirely*: loses the "wrapper behavior changed" safety the cache was designed for — rejected; we keep a wrapper hash but base it on the stable shared module.

**Validation hook**: an integration test runs `build_speaker_profile` then `analyze_audio` on the same file with the same params and asserts the diarization/embedding tasks report `cache: "hit"` on the second stage.

**Status (as of US2)**: partially delivered. The shared `cache.py` helper (`task_wrapper_hash`, keyed on each task's implementing modules) shipped in Phase 2 and `build_speaker_profile` uses it, but `analyze_audio.py` still keys every task on `sha256(analyze_audio.py source)` — so the two stages do **not** yet share entries and FR-015/R1 is not realized end-to-end. The swap was deliberately deferred until a second consumer existed; it now does. Finishing it (audit the task→module map, bump the cache schema version, swap `analyze_audio`'s keying, add the real cross-stage `cache: "hit"` test) is scheduled as **Phase 6 (T033–T036)**, sequenced after US3 so it doesn't collide with the US2/US3 edits to `analyze_audio.py`. Trade-off to keep in mind: the current script-source hash over-invalidates (re-runs unchanged tasks on any script edit, never shares across callers); the module-based hash fixes both but must enumerate every behavior-determining module per task to avoid under-invalidation.

---

## R2. Profile aggregation — dominant-cluster centroid across files

**Decision**: Pool per-window embeddings from **all** of a subject's files (all diarized speech windows, all speakers — diarization/`p_voice` used only to locate speech, not to assign identity per FR-002), then run the existing `cluster_pass_speakers` logic over the pooled set. The **profile = L2-normalized centroid of the dominant (largest, silhouette-coherent) cluster**, per embedding model. Minority clusters (other voices, noise) are discarded.

**Rationale**: `clustering.cluster_pass_speakers` already implements exactly the contamination-tolerant behavior we want — `min_cluster_size` outlier rejection, silhouette-gated multi-cluster vs single-cluster regimes, `_merge_close_clusters` to fold prosodic sub-clusters of one speaker, and `_empirical_calibration_band` for thresholds. Reusing it avoids a second, divergent clustering implementation. Cross-file pooling is the only new step: build one `list[WindowEmbedding]` spanning all files (tagged with source-file id for leave-one-file-out, R5).

**Alternatives considered**:
- *Mean of all window embeddings*: not robust — a persistent second voice biases the mean (rejected; matches spec FR-003 clustering decision).
- *Per-file profile then average*: loses cross-file outlier rejection; a fully-contaminated file pollutes its per-file centroid — rejected.
- *New GMM/iterative attention pooling*: more powerful but unjustified complexity vs. reusing validated clustering — deferred unless evaluation shows the spectral approach underperforms.

---

## R3. Multi-model / multi-timescale consensus (FR-018)

**Decision** (updated per 2026-05-27 clarification): Default consensus = **three models — ECAPA-TDNN + ResNet-TDNN + WavLM** (`microsoft/wavlm-base-plus-sv`). Build **one centroid per model**; at comparison time score each window against each model's centroid and combine by a **calibrated mean of per-model cosine-uncertainties** (each model calibrated with its own `_empirical_calibration_band`, so 192-D ECAPA / ResNet and 512-D WavLM never compare directly). Model list is **configurable** with a **single-model fallback** (ECAPA). Multi-*timescale* is realized by a **short-window (~0.5s)** detection pass (R4) alongside the long ~2s profile centroid. **X-Vector is dropped from the default** (weak, redundant with ECAPA family).

**WavLM integration**: WavLM SV is a HuggingFace `WavLMForXVector` model, but senselab's `extract_speaker_embeddings_from_audios` is **SpeechBrain-only** (raises `NotImplementedError` otherwise, `speaker_embeddings/api.py:82`). This feature therefore **extends the embedding backend** with a transformers WavLM path (FR-019). `transformers`/`torch` are already deps, so the addition is bounded. Default checkpoint `microsoft/wavlm-base-plus-sv`; configurable so a WavLM-Large SV checkpoint can be substituted **if one becomes available** (no official `wavlm-large-sv` is on the HuggingFace Hub as a `WavLMForXVector` — `microsoft/wavlm-large` is a headless backbone). If the WavLM backend fails to load, the consensus degrades to the available SpeechBrain models with a recorded reason (no abort).

**Update (2026-06): a strong WavLM-Large SV exists, but not as a drop-in.** Microsoft's UniSpeech repo (`downstreams/speaker_verification`) publishes fine-tuned WavLM-Large SV models — VoxCeleb1-O EER ≈ **0.43%** (fix-pretrain No) / 0.75% (fix-pretrain Yes), vs our `wavlm-base-plus-sv` at ~2–3% — i.e. the much stronger fine-grained discriminator the same-gender blind spot needs. Caveats blocking a checkpoint-id swap: (1) the official checkpoints are on OneDrive/Google Drive, but a **HF mirror exists** — `subatomicseer/wavlm-large-sv-ckpts` hosts `wavlm_large.pt` (1.26 GB backbone) + `wavlm_large_finetune.pth` (1.3 GB fine-tuned SV head), so a reproducible `hf_hub_download` (revision-pinnable) fetch is now possible, resolving the hosting friction; (2) still **UniSpeech-format torch pickles, not `transformers.WavLMForXVector`** — loaded via UniSpeech's `verification.py` + `s3prl` (WavLM-Large SSL → ECAPA-TDNN head), so it needs a **new backend** (good subprocess-venv candidate given s3prl's heavy/pinned deps), not just a config change; (3) **license/provenance**: the HF mirror is community-hosted and tagged **CC-BY-SA-3.0** (share-alike) while Microsoft's UniSpeech *code* is MIT — verify the relicense legitimacy + share-alike acceptability, and the `.pt`/`.pth` are pickles (load with care). Net: substantially more practical now (reproducible fetch solved); remaining cost is the new s3prl/UniSpeech backend + a licensing sign-off. Candidate for the deferred same-gender / stronger-discriminator work.

**Rationale**: Consensus value comes from **error decorrelation**, not raw accuracy. ECAPA and ResNet are both VoxCeleb-supervised with Fbank front-ends → partly correlated errors. WavLM is self-supervised (learned waveform front-end, 94k-hour diverse corpus) with **denoising/overlap-aware pretraining** directly relevant to other-voice and noisy-clinical detection → genuine decorrelation. Averaging *calibrated uncertainties* (not raw cosines, which live on different scales/dims) makes models commensurable.

**Alternatives considered**:
- *ECAPA + WavLM only (2-way)*: highest diversity per cost; rejected in favor of also keeping ResNet for added robustness (user choice).
- *SpeechBrain-only (ECAPA + ResNet)*: no new backend, but lower diversity (correlated errors) — rejected; WavLM's decorrelation is worth the bounded backend work.
- *All four incl. X-Vector*: diminishing returns, X-Vector weak/redundant — rejected.
- *Concatenate embeddings into one vector*: mixes incompatible metric spaces/dims — rejected for late (score-level) fusion.
- *Unofficial / self-fine-tuned WavLM-Large SV*: provenance risk for clinical data — deferred unless an official/validated checkpoint appears.
- *wav2vec-BERT 2.0 as a consensus member (2026-06-05, raised on PR #523)*: in a separate analysis w2v-BERT 2.0 reportedly outscored WavLM but **still under ECAPA**. Two caveats make it a *future candidate, not a swap*: (1) **transferability** — that ranking was on a different task; what matters here is speaker discrimination (esp. same-gender) + noise robustness, where our sweeps show WavLM's role is *decorrelated robustness*, not raw discrimination (ECAPA already wins discrimination). If w2v-BERT is below ECAPA it wouldn't lift discrimination — its only plausible role is **replacing WavLM as the decorrelated/robust member**, to be judged on the consensus tradeoff, not standalone accuracy. (2) **No SV head** — w2v-BERT 2.0 is a frame encoder with no standard speaker-verification head (unlike the drop-in `wavlm-base-plus-sv` `WavLMForXVector`), so it needs pooling or an SV-finetuned checkpoint = net-new backend, not a `--embedding-models` config change. For the same-gender gap specifically, **WavLM-Large SV (above) is the more direct lever.** Model set stays configurable, so swappable later if a usable SV-capable w2v-BERT checkpoint appears; current ECAPA+ResNet+WavLM results accepted for this iteration.

---

## R4. Brief-intrusion detection resolution (FR-017)

**Decision**: Run detection on **overlapping short windows (~0.5s hop, ~1s window — the existing `embeddings.py` default)** for temporal resolution, but compute each short window's embedding and compare it to the (long-window-built) profile. For intrusions shorter than the embedding window, rely on (a) the dense overlap and (b) corroboration from the existing **presence/identity axes** (a brief foreign voice depresses `p_voice`-consistency and raises identity uncertainty). Localization below ~1s is reported as **lower-confidence** rather than precise.

**Rationale**: This matches `embeddings.py`'s own documented trade-off ("1s window / 0.5s hop gives one embedding per 0.5s bucket … noisier below 1s but functional"). We get fine resolution for flagging while keeping the profile centroid built from longer, cleaner windows. Multi-timescale (R3) gives the short pass for detection and the long pass for the profile.

**Alternatives considered**:
- *Single long window only*: misses sub-1s intrusions entirely — rejected.
- *Sample-level / VAD-overlap detection*: out of scope for an embedding-based signal; the spec explicitly accepts coarse sub-1s localization.

---

## R5. Leave-one-file-out scoring (FR-012)

**Decision**: Tag each pooled window with its **source-file id**. When scoring recording *F*, recompute the dominant-cluster centroid **excluding all windows from *F***. For the common multi-file subject this is cheap (re-aggregate the cached pooled embeddings minus one file). For a **single-file subject**, fall back to **within-file holdout**: build the centroid from the file's windows excluding the window under test (and a small temporal guard band around it).

**Rationale**: Matches the spec's leave-one-file-out decision while remaining viable for single-file subjects (an edge case the spec calls out). Re-aggregation is O(clustering) on already-cached vectors, not a re-extraction.

**Alternatives considered**:
- *Single global profile, no exclusion*: inflates self-match (rejected per spec).
- *Precompute all leave-one-out centroids up front*: fine for small file counts; we compute lazily per scored file to bound memory.

---

## R6. Other-voice flag + threshold (FR-008)

**Decision**: Per scored window, compute consensus calibrated **identity uncertainty** vs. the leave-one-file-out profile (R3 + R5). Flag **likely-other-voice** when uncertainty exceeds an **adaptive per-subject threshold** derived from the profile's own within-cluster distribution via `_empirical_calibration_band` (with the literature fallback band `[0.30, 0.70]` when too few pairs). Comparison is **gated on speech presence**: windows with low `p_voice` get `status = "unavailable"` (N/A), not a flag.

**Rationale**: Per-subject calibration adapts to each subject's embedding spread (mic, room, voice), which a fixed global cosine threshold cannot. The calibration helpers already exist and are used by the identity axis, keeping the profile signal consistent with existing outputs.

**Alternatives considered**:
- *Fixed global cosine threshold*: simpler but ignores per-subject variance and SNR — rejected.
- *Supervised threshold*: no labels available — rejected.

---

## R7. Recording-quality indicator (FR-010)

**Decision**: Derive a per-recording target-speaker quality score from profile-consistency, combining: (a) fraction of speech-present duration whose windows match the profile (1 − other-voice rate), (b) mean within-profile cosine consistency on matched windows, and (c) the existing per-window **SQUIM** objective metrics (STOI/PESQ/SI-SDR) already computed by `analyze_audio`, restricted to matched windows. Reported as a normalized [0,1] indicator plus its components (not a single opaque number). **Surfaced by extending the existing per-pass `quality` claim** (symmetric with the `single_speaker` extension, per R10) — not a standalone indicator; its detail also appears in the per-pass `speaker_profile.json` sidecar.

**Refinement (US3 implementation)**: the headline `profile_target_quality` scalar is the mean of **(a) + (b) only** — both natively [0,1] and genuinely profile-relative. SQUIM **(c)** is reported as **raw means** on the matched windows (`profile_squim`) *alongside* the scalar rather than folded into it. Rationale: STOI/PESQ/SI-SDR live on different scales (≈[0,1] / [1,4.5] / dB), so folding them in would require inventing normalization anchors that duplicate the ones the existing `quality` claim already applies to whole-file SQUIM; and since `torchaudio_squim` is a whole-file score, "matched-window SQUIM" adds little beyond what the `quality` claim already carries. Keeping SQUIM raw avoids the magic-number anchors in the headline and avoids double-representing the same signal. (Decided with the user during implementation.)

**Rationale**: Reuses signals already on disk; "how cleanly is the *target* captured" is exactly target-matched-window consistency × objective quality. Exposing components keeps it auditable and avoids over-claiming a calibrated absolute.

**Alternatives considered**:
- *SQUIM alone*: doesn't account for target-voice dominance / contamination — rejected (the feature's point is target-specific quality).
- *Single black-box score*: less actionable; rejected in favor of components + summary.

---

## R8. Minimum-data & windowing policy (FR-005, FR-002)

**Decision**: Profile windows are the existing ≥~1s embedding windows; **sub-1s contiguous speech fragments are dropped** (the window grid in `_window_starts` already requires `duration ≥ window_s`). A profile is **confident** when the dominant cluster is built from **≥~20s aggregate** speech-present audio (target ~30s); below the floor → `confidence: "low"`; far below or no coherent cluster → decline (`confidence: "insufficient"`). Threshold configurable.

**Rationale**: Aligns the spec's aggregate (not continuous) minimum with the embedding model's window requirement; reuses existing grid behavior.

---

## R9. Profile artifact: format & storage

**Decision**: Persist as a **JSON** file (one per subject), containing: subject id, per-model centroid vectors, dominant-cluster metadata (n windows, aggregate seconds, silhouette, empirical calibration band), per-source-file usage record + per-file keep/drop decision and reason, confidence label, params/provenance (models, window/hop, senselab + schema version), and the source-file ids needed for leave-one-file-out. Stored under a user-given `--output` path; loadable by `analyze_audio --speaker-profile <path>`.

**Rationale**: JSON matches the existing per-task JSON output convention and is human-inspectable (FR-004 auditability, FR-006 reuse). Vectors are small (≤~192 floats × few models).

**Alternatives considered**:
- *Parquet / .npy*: better for large arrays, but the artifact is tiny and JSON keeps provenance readable — rejected.
- *Embedding stored in cache only*: not portable/reusable as an explicit input — rejected.

---

## R10. Integration point in analyze_audio

**Decision** (refined against existing output logic): Reuse `analyze_audio`'s existing structures rather than parallel ones.
- **Per-window (pre-aggregation)**: extend the **identity axis** — when `--speaker-profile` is supplied, emit per-bucket `model_votes["speaker_profile/<model>"]` + `"speaker_profile/consensus"` carrying `{similarity, other_voice_uncertainty, flag}` into the existing `identity.parquet`; write a per-pass `<pass>/speaker_profile.json` sidecar (matching the `embeddings/*.json` convention) for the verbose per-window list.
- **Per-recording (post-aggregation)**: extend the existing **`single_speaker` claim** in `global_uncertainty.by_pass[<pass>]` (`global_summary.py`) with profile sub-signals (other-voice fraction/duration, peak/p95 uncertainty, `profile_confidence`) and fold a profile uncertainty into its headline via the established intensity-weighted-mean / `max()` aggregation (`_mean_over_voice_buckets`). No new top-level summary object.
- `disagreements.json` ranking is automatic since the votes live in the identity parquet.

When absent, **all existing outputs are byte-for-byte unchanged** (FR-011, SC-006).

**Rationale**: `single_speaker` already answers "is there >1 speaker?" (`n_speakers`, `identity_axis_mean`); the profile is the *target-relative* refinement of exactly that claim, so a downstream gate reads one enriched claim instead of merging a second summary. Symmetrically, US3's target-quality rollup extends the existing **`quality`** claim (FR-010, R7) rather than standing alone.

**Rationale**: The identity axis already compares embeddings vs. diarization; the profile is "identity vs. a known reference," a natural extension that lands alongside existing per-bucket outputs for review (FR-009).

**Alternatives considered**:
- *Entirely separate output pipeline*: duplicates bucket/grid plumbing — rejected.
- *Replace identity axis*: breaking and unnecessary — rejected.

---

## R11. Wrong-speaker / ambiguity confidence (FR-014) & session preference (FR-013)

**Decision**: Take the **dominant cluster as target** (clinical trust). When the top-two clusters are **near-equal** (sizes and centroid separation below a margin), still build the profile from the larger cluster but set `confidence: "ambiguous"` and record the runner-up cluster's stats. **Session preference (FR-013)** is implemented as an optional `--prefer-session <id>` weighting that up-weights same-session windows in cluster selection/centroid; default is unweighted (use all files). Functions with or without session metadata.

**Rationale**: Matches the clarified clinical assumption and keeps session preference an optional refinement, not a hard dependency.

---

## R12. Build-time speech gate — lightweight, not full presence

**Decision**: The profile builder gates speech with a **best-available presence** signal: a cheap floor of diarization (all speakers) + a per-window speech mask (AST/YAMNet speech labels + loudness — promoting the existing `compute._speech_window_mask`, fed to `cluster_pass_speakers` as `is_speech_per_window`); it **opportunistically** folds in Whisper `no_speech_prob` / PPG voiced-fraction when those outputs are already cached, but **never triggers ASR/PPG solely to gate**.

**Rationale**: The full 11-voter `p_voice` (presence.py) requires the most expensive tasks (ASR) and the ~1.4 GB PPG venv. For *selecting windows to embed*, speech-vs-non-speech suffices and the clustering step already rejects other-speaker windows. Building runs *before* `analyze_audio`, so requiring `p_voice` at a cold cache would force ASR purely for gating. Comparison time (FR-008) is different — it runs inside `analyze_audio` where the full `p_voice` already exists, so it reuses it directly at no extra cost.

**Alternatives considered**:
- *Full `p_voice` at build time*: most robust gate but pays for ASR/PPG up front — rejected as default (allowed opportunistically when cached).
- *Loudness-only VAD*: too crude for quiet/whispered speech — rejected in favor of the scene+loudness mask.
- *Use confidence signals to **weight** the centroid (not just gate)*: promising (Whisper `avg_logprob`/`no_speech_prob`, PPG, SQUIM) — deferred to optional research (T028b), PPG opt-in.

---

## R13. Synthetic test fixtures (SpeechT5, committed)

**Decision**: Validation data is **synthetic and committed**. A one-time generator (`scripts/gen_synthetic_test_audio.py`, not in CI) uses **SpeechT5** (`microsoft/speecht5_tts`+`speecht5_hifigan`, revision-pinned, **MIT**) with 3 fixed CMU-Arctic x-vector speaker embeddings + a fixed seed to synthesize public-domain phonetically-rich text into 16 kHz mono FLAC clips + `manifest.json` under `src/tests/data_for_testing/synthetic/`. Tests load the committed clips and **deterministically compose** contamination / overlap / noise scenarios (T010b) for SC-002/003/004/005 and the T028 sweep.

**Rationale**: For these tests, **replicability ≫ perceptual quality** — embedding-based assertions need identical audio every run. A pinned local model is reproducible (revision + seed), offline (no API key/network/cost), and **license-clean for committing** (MIT). Committing the clips freezes speaker identity regardless of TTS version drift; mixing/noise are cheap deterministic ops done at test time. This also sidesteps the b2aivoice data-access restriction entirely.

**Alternatives considered**:
- *OpenAI TTS (tts-1 / gpt-4o-mini-tts)*: higher quality but non-pinnable (server-side drift), needs key/network/cost in the test path, and murkier output-licensing for committing — rejected for fixtures (fine for user-facing/demo audio elsewhere). Quality isn't the bottleneck here.
- *MMS/Coqui-XTTS as generator*: MMS is CC-BY-NC, XTTS non-commercial — rejected for committed assets on licensing grounds.
- *Generate live in tests*: non-deterministic across model versions, heavy deps, flaky CI — rejected.

---

## R14. Scope boundary — signal producer, not a triage gate

**Decision**: The speaker-profile feature is a **general signal producer that feeds `analyze_audio`** and stops at decision-ready outputs: per-window similarity/flags + identity-axis votes, and a recording-level rollup that **extends the existing `single_speaker` claim** with profile sub-signals (FR-020, see R10). It does **not** decide PASS vs. manual-review and embeds no operating-point policy.

**Rationale**: The downstream use case has an asymmetric cost (a multi-speaker recording must not pass through, while over-flagging is merely costly), which implies a recall-biased gate, fail-safe routing on low-confidence profiles, and likely an agentic reviewer weighing *all* `analyze_audio` measures (diarization speaker count, presence axis, SQUIM, ASR disagreement, and this profile signal) with a decision + rationale. That is a cross-signal orchestration concern with its own success criteria and guardrails — coupling it into this feature would tie a reusable capability to one dataset's policy. It is deferred to a **separate future spec** that consumes these outputs.

**Implication for outputs**: ensure the emitted signals are sufficient for any operating point — hence FR-020's fraction/duration/peak/p95 + echoed profile confidence (so the gate can fail-safe). No verdict field is produced here.

---

## Constants & Thresholds (surface explicitly in implementation)

All thresholds MUST be implemented as **named, documented, configurable** values (module-level constants or CLI/function params with a comment giving the value, its source, and whether it is an assumption to validate) — not buried magic numbers. Origin tags: **[reuse]** = existing senselab default (cite source), **[new]** = introduced here and to be validated empirically.

### Reused from existing code — keep as defaults, cite source

| Constant | Default | Source | Notes |
|----------|---------|--------|-------|
| `profile_window_s` / `profile_hop_s` | `2.0` / `1.0` | `embeddings.py` | Long windows for clean centroid embeddings. |
| `detect_window_s` / `detect_hop_s` | `1.0` / `0.5` | `embeddings.py` | Short windows for detection temporal resolution (R4). |
| `n_clusters_max` | `6` | `clustering.cluster_pass_speakers` | Max speaker clusters considered. |
| `min_windows_for_clustering` | `4` | `clustering.cluster_pass_speakers` | Below → single-cluster regime. |
| `coherent_silhouette_threshold` | `0.10` | `clustering.cluster_pass_speakers` | Multi- vs single-cluster gate. |
| `min_cluster_fraction` | `0.10` (floor 2 windows) | `clustering.cluster_pass_speakers` | Tiny-cluster = outlier, not speaker. |
| `merge_threshold` | `0.55` | `clustering._merge_close_clusters` | Collapse prosodic sub-clusters of one speaker. |
| `same_speaker_floor` / `diff_speaker_floor` | `0.30` / `0.70` | `clustering.calibrate_cosine_uncertainty` | Literature fallback band; empirical 75th/25th-percentile anchors override per profile. |
| `clustering_algorithm` | `"spectral"` (k-means fallback) | `clustering.cluster_pass_speakers` | — |

### New — placeholder/tuned, validate via empirical sweep

| Constant | Proposed default | Status | Notes |
|----------|------------------|--------|-------|
| `min_confident_speech_s` | `20.0` | **[new]** assumption (A2) | Below → `confidence="low"`. Validate against real per-subject durations. |
| `target_confident_speech_s` | `30.0` | **[new]** assumption (A2) | Target for `confidence="ok"`. |
| `AMBIGUITY_SHARE_RATIO` | `0.80` (provisional) | **[new]** validate (T028) | Flag `confidence="ambiguous"` when `runner_up_speech_s / dominant_speech_s ≥ 0.80`, evaluated only with ≥2 speech clusters (centroid separation is guaranteed by the no-merge-below-0.55 rule). Provisional: balanced ~50/50 → ambiguous, dominant ~85/15 → confident (R11). |
| `other_voice_threshold` | adaptive (from `_empirical_calibration_band`) | **[new]** assumption (A5) | Per-subject; optional fixed override via `--profile-other-voice-threshold`. |
| Consensus fusion weights | unweighted mean of calibrated per-model uncertainties | **[new]** assumption | ECAPA/ResNet/WavLM (R3); revisit weighting if one model dominates errors. |
| `min_contiguous_speech_s` (per contributing window) | `~1.0` | **[new]** TBD | Sub-1s fragments dropped/merged; exact drop-vs-merge policy TBD. |
| Sub-1s intrusion confidence boundary | localization low-confidence below `~1.0s` | **[new]** TBD (FR-017) | Quantify the boundary during implementation. |
| WavLM default checkpoint | `microsoft/wavlm-base-plus-sv` | **[new]** (FR-019) | Configurable; substitute WavLM-Large SV if one becomes available. |

## T028 sensitivity characterization (2026-06-03, GPU `scripts/slurm_speaker_profile_sweep.sh`)

Ran on the synthetic fixtures with the full ECAPA+ResNet+WavLM consensus. **These are characterizations, not tuned production values** — synthetic TTS data; not validated on real recordings.

**Degradation (pure-target recording, varying additive-noise SNR):** the embedding signals are robust to ~20 dB then degrade, but the degradation is **dominated by false other-voice flagging**, not graceful quality estimation:

| SNR dB | si_sdr | stoi | target_quality | match_fraction | consistency | false_other_voice |
|---:|---:|---:|---:|---:|---:|---:|
| clean | 28.4 | 1.00 | 0.98 | 1.00 | 0.96 | 0.00 |
| 20 | 20.2 | 0.97 | 0.93 | 0.97 | 0.89 | 0.03 |
| 15 | 15.3 | 0.93 | 0.86 | 0.91 | 0.81 | 0.09 |
| 10 | 10.2 | 0.87 | 0.77 | 0.83 | 0.71 | 0.17 |
| 5 | 5.1 | 0.81 | 0.61 | 0.57 | 0.64 | 0.43 |
| 0 | 0.1 | 0.74 | 0.43 | 0.26 | 0.60 | 0.74 |
| −5 | −4.8 | 0.67 | 0.00 | 0.00 | 0.00 | 1.00 |

**Implication (design):** noise makes the target's own speech read as "another voice" (false-OV 0.17→0.74 over 10→0 dB). So `target_match_fraction` / `target_quality` **conflate noise with other-voice** and are not a clean acoustic-quality proxy. The right dependency is **SQUIM → trustworthiness of the profile flags** (below ~10–15 dB the flags are unreliable and should be down-weighted), *not* embeddings → quality. SQUIM degrades cleanly/monotonically here, so it is the better acoustic-quality signal. Acting on this (gating/widening flag confidence by SQUIM, and reframing the US3 "quality" indicator as target *purity*) is a candidate refinement — deferred, consistent with the 2026-06-03 scope clarification.

**Per-model degradation breakout (GPU job 15404761):** the consensus false-other-voice explosion above is **almost entirely the Fbank models**. Standalone false-OV rate per model on the pure-target recording (each model thresholded alone at the 0.5 cutoff):

| SNR dB | ecapa fOV | resnet fOV | wavlm fOV | consensus fOV |
|---:|---:|---:|---:|---:|
| clean | 0.06 | 0.00 | 0.00 | 0.00 |
| 20 | 0.11 | 0.09 | 0.00 | 0.03 |
| 15 | 0.23 | 0.17 | 0.00 | 0.09 |
| 10 | 0.46 | 0.43 | 0.06 | 0.17 |
| 5 | 0.86 | 0.71 | 0.11 | 0.43 |
| 0 | 0.97 | 0.89 | 0.11 | 0.74 |
| −5 | 1.00 | 0.97 | 0.37 | 1.00 |

**WavLM is far more noise-robust than ECAPA/ResNet** (SSL, noise/overlap-aware pretraining — exactly the FR-018 decorrelation rationale). At 0 dB the Fbank models misflag ~90–97% of the target's own speech as other-voice while WavLM misflags ~11%.

**Multi-condition cross-validation (GPU job 15405479) — exposes a robustness↔discrimination trade-off.** Leave-one-out over speaker A's 3 long passages × 5 noise seeds confirms the false-OV ordering with tight variance (0 dB mean±std: ecapa 0.98±0.02, resnet 0.93±0.04, wavlm 0.20±0.05). But per-model *detection* of an overlaid intruder reveals the opposite weakness for WavLM:

| intruder | ecapa det/FP | resnet det/FP | wavlm det/FP | consensus |
|---|---|---|---|---|
| B (ksp, different timbre) | 0.93/0.05 | 0.89/0.01 | 0.81/0.00 | 0.89/0.01 |
| C (clb, similar timbre) | 0.85/0.05 | 0.67/0.01 | **0.00**/0.00 | 0.48/0.01 |

WavLM-base-plus-sv **completely misses the similar-timbre intruder** (0.00) while ECAPA catches 85%. So: **WavLM = noise-robust but weak fine-discrimination; ECAPA/ResNet = discriminative but noise-fragile.**

**Revised implication (supersedes the naive "up-weight WavLM"):** a static WavLM up-weight would fix noise but *break* the hardest/most-important case — a same-gender / similar-voice co-speaker. The right lever is **condition-dependent fusion** (trust WavLM under low SNR; trust the Fbank pair in clean audio for fine discrimination — i.e., SQUIM-gated weighting), and/or a stronger **WavLM-Large SV** checkpoint to lift discrimination (the FR-019 substitution note). The equal-weight consensus is a mediocre hedge at both extremes. All a candidate refinement, deferred per the 2026-06-03 clarification. Caveat: synthetic TTS, 3-speaker corpus (1 target subject + 2 others), additive Gaussian noise — magnitudes will differ on real data; the qualitative trade-off is mechanistically expected.

**3-subject leave-one-subject-out (GPU job 15406725; each of A/B/C as target, the other two as intruders) — generalizes the noise result and exposes a systemic same-gender blind spot.**

- *Noise robustness generalizes across target voices:* at 0 dB WavLM false-OV is 0.00–0.14 for every target (slt/ksp/clb) while ECAPA/ResNet are 0.90–1.00. Voice-independent.
- *Same-gender intruder detection is a systemic failure, not a WavLM quirk:* detection of an overlaid intruder by gender match —

  | target | intruder | gender | ecapa | resnet | wavlm | consensus |
  |---|---|---|---|---|---|---|
  | F(slt) | M(ksp) | diff | 0.89 | 0.89 | 0.89 | 0.89 |
  | F(slt) | F(clb) | SAME | 0.33 | 0.22 | 0.00 | **0.00** |
  | M(ksp) | F(slt) | diff | 0.89 | 0.89 | 0.67 | 0.89 |
  | M(ksp) | F(clb) | diff | 0.89 | 0.89 | 0.78 | 0.89 |
  | F(clb) | F(slt) | SAME | 0.44 | 0.00 | 0.00 | **0.00** |
  | F(clb) | M(ksp) | diff | 0.56 | 0.89 | 0.67 | 0.67 |

  Cross-gender intruders are detected well by all models (~0.67–0.89); the **same-gender (both-female slt↔clb) intruder is detected 0% by the consensus in both directions** (ECAPA the only model with any signal, ≤0.44; WavLM exactly 0.00).

**Headline implication:** detecting a **same-gender / similar-timbre co-speaker is a known blind spot** of this profile-similarity approach with the default models, and the consensus does not rescue it — not fixable by fusion weighting alone (WavLM contributes nothing there). It would need a stronger-discrimination embedding (WavLM-Large SV / hard-speaker-tuned) or a corroborating mechanism (diarization, relative-change). For the recall-biased downstream gate (a same-gender co-speaker must not pass), this is the key limitation to carry into the triage spec.

**7-speaker systematic confirmation (throwaway corpus, ASR/SQUIM-gated, not committed):** a larger run (the `Matthijs/cmu-arctic-xvectors` set only contains 7 CMU-Arctic speakers — 2F/5M — so the extra females requested weren't available; the gain is 5 males → 20 male-male same-gender pairs) reproduces the blind spot with real sampling. Every generated clip passed an ASR round-trip gate (WER≈0.00, STOI 1.0, PESQ ~4). Over 22 same-gender vs 20 cross-gender ordered pairs:

| pairing | ecapa | resnet | wavlm | consensus |
|---|---|---|---|---|
| same-gender | 0.52±0.25 | 0.30±0.25 | 0.01±0.02 | 0.16±0.20 |
| cross-gender | 0.71±0.19 | 0.69±0.16 | 0.51±0.19 | 0.67±0.17 |

Same-gender detection: consensus 0.16 vs 0.67 cross-gender — and crucially the equal-weight consensus (0.16) is **worse than ECAPA alone (0.52)** because averaging in WavLM's ~0 same-gender signal actively dilutes it. Noise robustness generalized too (0 dB false-OV: ecapa 0.93, resnet 0.85, wavlm 0.12 over 7 voices). So same-gender discrimination is fundamentally capped (~0.52 even for the best single model) and the equal-weight fusion is the worst of both axes; a stronger discriminator (WavLM-Large SV / hard-speaker-tuned) or corroborating mechanism is needed for the same-gender case. (Female-female is still only the slt/clb pair; male-male is now well-sampled.)

**Detection-window-length sweep (throwaway corpus, 7 voices; window 0.5→4s, profiles fixed):** longer detection windows make *detection worse*, not better (correcting an earlier hypothesis). Consensus same-gender detect 0.45→0.16→0.03→0.00→0.00 and cross-gender 0.79→0.67→0.55→0.48→0.32 as the window grows 0.5→4s; brief-1s-intrusion detect 0.53→0.00. Two mechanisms: (1) a long window over *co-present* (target+intruder) audio averages toward the dominant target → masks the intruder; (2) with a fixed calibration band the window length acts as an **operating-point knob** — short windows give noisier embeddings → larger distances → more flags (sensitivity↑ *and* false-OV↑), long windows the reverse (0 dB false-OV: consensus 0.91→0.05 over the same range). So window length trades sensitivity↔specificity, not discriminability. **Crucially, WavLM same-gender detect is ~0 at every window length (0.07/0.01/0.00/0.00/0.00)** → its same-gender failure is an information/capacity floor, **not** a length artifact, so retraining WavLM on short crops would not fix it (the deficit doesn't move with length). Implication: keep short (~1 s) detection windows for the recall-biased goal (they catch brief + same-gender intrusions); manage the resulting noise false-positives via SQUIM-gated trust / WavLM rather than by lengthening windows (which would mask the very intrusions we hunt). Same-gender recall needs a stronger discriminator (ECAPA carries it best; WavLM-Large SV), not longer windows.

**Build-window sweep (throwaway 7 voices; build W ∈ {1,2,3,5}s, detection grid fixed at 1.0/0.5).** Complements the detection sweep above by varying the *profile-build* window instead. Tests whether a longer-window centroid (the "5.0/2.5 enrollment, short detection" idea) is better. **It is not — building longer is mildly worse, and shorter-build ≈ detection-grid is best.**

| build W | centroid quality (mean cos target-1s→centroid) | clean false-OV (cons) | wham-0dB false-OV (cons) | same-gender detect | cross-gender detect |
|---:|---|---:|---:|---:|---:|
| 1.0 | ecapa 0.706 / resnet 0.750 / wavlm 0.884 | 0.06 | **0.15** | 0.14 | 0.66 |
| 2.0 | 0.701 / 0.746 / 0.881 | 0.06 | 0.17 | 0.16 | 0.67 |
| 3.0 | 0.693 / 0.740 / 0.877 | 0.06 | 0.18 | 0.17 | 0.67 |
| 5.0 | 0.688 / 0.736 / 0.874 | 0.06 | **0.19** | 0.17 | 0.67 |

- *Centroid representational quality DECREASES monotonically with build window* (all three models) — a centroid aggregated from 2–5 s windows matches the 1 s detection windows *less* well than one built from 1 s windows. The build/detect grid-mismatch I hypothesized is real, but it lives in the **centroid↔window geometry**, not the calibration band: longer-context build embeddings sit slightly farther from short detection windows.
- *Noise false-OV RISES with build window* (consensus 0.15→0.19; ecapa 0.36→0.41) — the worse centroid match means more clean-but-noisy target windows cross the cutoff. A real cost.
- *Clean false-OV is flat* (~0.06) and *same-gender detection barely moves* (0.14→0.17, doesn't fix the blind spot); cross-gender flat.
- **Conclusion: do NOT build longer.** A longer enrollment window gives no centroid gain (it loses a little), adds noise false-OV, and "more data" is a misconception — the centroid already pools *all* enrollment speech; window length only sets per-embedding granularity. If anything, **matching the build grid to the detection grid (build at ~1 s) is marginally optimal**; the 2.0/1.0 default is a fine, mild compromise. Keep detection short (1.0/0.5) per the detection sweep.
- *Caveat — band-recalibration arm void on clean enrollment:* the test also compared scoring under the build-grid band vs a detect-grid-recalibrated band, but on clean single-speaker enrollment both resolve to the literature fallback `(0.30, 0.70)` (no minority cluster → no empirical band). Resolved in the contaminated follow-up below.

**Contaminated-enrollment band-recalibration test (throwaway 7 voices; each target's enrollment contaminated with a cross-gender intruder clip to force an empirical band; build W ∈ {1,2,5}s, detect grid fixed 1.0/0.5).** With ground-truth-labelled bands this confirms the mismatch mechanism *and* shows recalibration is a pure operating-point shift, not a fix.

- *The empirical band's same-speaker floor drifts with build window* (ECAPA cos-distance `same_floor`: W=1 **0.487** → W=2 0.315 → W=5 **0.139**; detect-grid = 0.487). Longer build windows give tighter target clusters → smaller within-distance → lower `same_floor` → a **stricter** operating point applied to the noisier 1 s detection windows. This is the build/detect mismatch, now visible (the clean sweep couldn't show it).
- *Recalibrating the band to the detection grid slashes noise false-OV* — wham-0dB consensus: W=2 build-band 0.20 → detect-band **0.10**; W=5 0.29 → **0.12** (ECAPA 0.60 → 0.17). Clean false-OV improves modestly (W=5 ECAPA 0.09 → 0.06).
- *…but it equally slashes detection* — same-gender consensus: W=2 build-band 0.19 → detect-band **0.05**; W=5 **0.36 → 0.06**. Cross-gender 0.67 → 0.60, 0.72 → 0.63.
- **So recalibration is a pure sensitivity↔specificity move along the same ROC, not a discriminability gain.** The empirical band's `same_floor` *is* the operating point, and **build-window length sets it implicitly**: a longer-build (strict) band buys higher recall — including the best same-gender detection seen anywhere (0.36 at W=5) — at the cost of more false-OV; the detect-grid (loose) band does the reverse. Neither separates target from intruder better; they pick different points on the curve.
- *Two implications:* (1) **Do not naively "recalibrate to the detect grid"** — for the recall-biased downstream goal it moves the wrong way (same-gender detection collapses 0.19→0.05 at the default build). (2) The real design smell is that **the operating point is entangled with build-window length** via the empirical band; it should be an *explicit, deliberate* knob (cutoff / band anchor) decoupled from the build window — which is exactly the calibration/operating-point work deferred to the triage spec.
- *Practical caveat:* the **real `build_speaker_profile` band stayed at the `(0.30, 0.70)` fallback even with the intruder clip** (harvard-00 is too small a minority to trigger an empirical band in the build's clustering). So on lightly-contaminated real enrollment the operating point is often just the fallback; these empirical-band dynamics need substantial contamination to engage.

**Speech-enhancement-before-scoring (throwaway 7 voices; SepFormer `sepformer-wham16k-enhancement` as a denoise pre-stage; profiles built from clean enrollment as the live pipeline does).** Tests whether running noisy audio through enhancement *recovers* the clean threshold behaviour or instead injects embedding distortions. **It does not recover — and it distorts.**

- *No false-OV recovery (additive Gaussian).* Consensus false-OV is unchanged at 0 dB (0.62→0.62) and slightly *worse* at 5 dB (0.28→0.32); enhanced 5/0 dB never approaches the clean 0.06 baseline.
- *Embedding-distortion diagnostic — cosine(mean window embedding, own enrolled centroid) — reveals an architecture split:*

  | cond | ecapa | resnet | wavlm |
  |---|---:|---:|---:|
  | clean / raw | 0.938 | 0.955 | 0.990 |
  | clean / enh | 0.928 | 0.933 | 0.986 |
  | 5 dB / raw | 0.660 | 0.678 | 0.849 |
  | 5 dB / enh | **0.577** | 0.650 | **0.896** |
  | 0 dB / raw | 0.544 | 0.577 | 0.772 |
  | 0 dB / enh | **0.498** | 0.571 | **0.833** |

  Enhancement nudges even **already-clean** audio off-identity (clean/enh < clean/raw for every model; ResNet worst, −0.022) — it is never free. On noisy audio it **actively hurts the Fbank SV models** (ECAPA 5 dB 0.660→0.577: the *enhanced* signal is further from identity than the raw noisy one) while it is the **only** model WavLM helps (5 dB 0.849→0.896). The SepFormer artifacts damage the discriminative VoxCeleb-trained nets more than the original noise did; the SSL transformer prefers the denoised input. But WavLM is exactly the model that barely fires, so its recovery never reaches the equal-weight consensus (Fbank-dominated) — which is why Block 1 doesn't improve.
- *Detection collateral.* The single-output enhancer treats a co-present intruder as "noise" and partly suppresses it: cross-gender WavLM intruder-detect falls 0.51→0.38 and same-gender consensus 0.16→0.11. The same-gender blind spot is untouched.

**WHAM-style re-run (same job, pink/colored + non-stationary bursty noise — in-distribution for the WHAM-trained enhancer).** The harsh "enhancement *hurts* the Fbank models" result above is **largely an AWGN (out-of-distribution) artifact**. On realistic colored, non-stationary noise the picture softens and partly flips:

- *Colored/non-stationary noise is far gentler at equal nominal SNR* — raw consensus false-OV is only 0.10 @ 5 dB and 0.17 @ 0 dB (vs AWGN 0.28 / 0.62), because pink noise concentrates energy low and the bursty envelope leaves much of the signal clean. There is simply less to fix.
- *Enhancement now gives a small recovery* rather than a regression: consensus false-OV 0.17→**0.11** @ 0 dB (ResNet 0.32→0.20, WavLM 0.04→0.01); ECAPA roughly flat.
- *Embedding fidelity improves for ResNet/WavLM, ECAPA neutral* — cosine-to-own-centroid @ 0 dB: ResNet 0.788→**0.814**, WavLM 0.933→**0.952**, ECAPA 0.795→0.789. The opposite of the AWGN case, where enhancement pushed ECAPA/ResNet *further* off-identity.

**Two findings survive both noise types and bound the upside:** (1) enhancement still distorts **already-clean** audio identically (clean cosine drops ECAPA −0.010 / ResNet −0.022 / WavLM −0.004), so it cannot be applied blanket — it would have to be SNR-gated to low-SNR input only; (2) it still erodes co-present-intruder detection (Block 3, noise-independent: cross-gender WavLM 0.51→0.38, same-gender consensus 0.16→0.11) and does nothing for the same-gender blind spot, and ECAPA — the discrimination workhorse — never benefits.

**Implication:** "denoise everything first" is the wrong lever — on out-of-distribution noise it injects artifacts the consensus-driving models dislike more than the noise, and even on in-distribution noise it taxes clean audio and erodes intruder detection. A **narrowly SNR-gated** enhancement (apply only below ~5 dB) is at best a *small* net-positive for false-OV on realistic noise (consensus 0.17→0.11 @ 0 dB) and would need to be paired with not enhancing clean/overlap material — a marginal, conditional refinement, not a fix for the structural limitations (same-gender discrimination, SQUIM-gated flag trust), which it leaves untouched.

**Enhancement-as-PROBE — derive a signal from the raw→enhanced embedding delta instead of keeping the audio (job 15434943, both AWGN and WHAM-style).** Two candidate per-window signals tested:

- *Signal 1 — displacement `δ_self = 1 − cos(raw_win, enh_win)` — WORKS as a corruption meter.* Per-window mean is monotone in degradation and well above the clean floor: clean ≈ 0.04–0.06 (all models) → wham-0dB 0.16–0.29 → awgn-0dB 0.32–0.54. So how far enhancement moves a window's embedding is a usable, model-internal "this window is corrupted / its embedding is unstable" indicator. **But it does not say *which* corruption** — intruder-overlap windows displace 0.09–0.18, comparable to wham-5dB target windows. So δ_self is a *trust/abstention* gate (down-weight high-δ_self windows' OV decisions), not a noise-vs-other-voice classifier.
- *Signal 2 — signed recovery `Δ = cos(enh,centroid) − cos(raw,centroid)` — FAILS.* The hope (noisy-target recovers toward identity Δ>0; a different speaker does not) does not hold: all means sit within ±0.05 with std 5–15× larger, ECAPA barely recovers even noisy target (−0.02…+0.01), and **intruder windows recover about as much as noisy-target windows** (diff-gender +0.017/+0.024 for resnet/wavlm, same sign as noisy target). Separability is near chance: per-window AUC `P(Δ_target>Δ_intruder)` = 0.42–0.68 across all model/condition pairs (best resnet wham-0dB-vs-same-gender 0.68; several <0.5). Enhancement's recovery is **non-specific** — it cleans up whatever content dominates a window, target or intruder alike — so the delta's *direction* carries almost no target-vs-other-voice information.

**Verdict on the probe:** there *is* a derivable signal (δ_self ≈ a corruption/instability meter), but (a) the more valuable de-conflation signal isn't there — enhancement recovery is non-specific, not the clean-degradation that kills it but the lack of *specificity*; and (b) δ_self is **redundant with cheaper alternatives already in the stack** — SQUIM (T028 showed it degrades cleanly/monotonically) gives the same per-window trust signal directly, without an extra SepFormer forward pass. So enhancement-as-probe is not worth wiring in: for per-window flag-trust gating use SQUIM, not the enhancement delta.

**Contamination (build level):** centroid drift ≈ 0.000–0.002 and target-sim flat at 0.977 (vs intruder 0.28) across 0–30% single-file signal-mix contamination — dominant-cluster aggregation is strongly contamination-tolerant (SC-002 holds with margin).

**Other-voice cutoff:** on clean overlay audio, detection 0.89 / false-positive 0.00 are **flat across cutoff ∈ [0.4, 0.7]** (target unc≈0, intruder unc≈1 are cleanly band-separated). The cutoff is an insensitive knob; **noise, not the threshold, is the dominant sensitivity** → keep the cutoff at the neutral midpoint (`OTHER_VOICE_CALIBRATED_CUTOFF = 0.5`), do not tune an operating point.

## PR #523 review — reuse/altitude refactor design (2026-06-05)

Maintainer review (Satra) raised two architectural points: (1) *"reuse the sliding-window embedding extraction and clustering already in the analyze script"* and (2) *"move things behind the identity signal."* A focused multi-agent review **verified both against the code** — the leaf primitives (`extract_per_window_embeddings`, `cluster_pass_speakers`, `_empirical_calibration_band`, `calibrate_cosine_uncertainty`, `speech_window_mask_for_file`) *are* genuinely reused, but the **orchestration and aggregation are parallel duplicates**, and the identity-axis integration is decorative. Verified findings:

- **Inert identity votes (confirmed).** `analyze_audio` injects `model_votes["speaker_profile/<model>"]` + `/consensus` into the identity axis, but `aggregate_identity` (`aggregate.py`) only reads `same_label_uncertainty` / `change_inconsistency_uncertainty` / `__cross_diar_label_disagreement__` — there is **no `speaker_profile` reference in `identity.py`/`aggregate.py`**. The votes ride the identity *parquet/disagreements ranking* but do **not** feed the identity uncertainty; the profile signal reaches the headline only via a separate `summarize_other_voice` → `global_summary` `single_speaker` `max()` fold. So "rides the identity axis" was true only for display, not for the decision.
- **Parallel aggregation.** Two speaker-disagreement signals (identity `identity_mean` and profile `p95_other_voice`) are computed by two separate per-window→recording pipelines and merged only at the end by `max()` in `global_summary.py`. The profile carries its own rollup dataclasses (`RecordingOtherVoiceSummary`) + `speaker_profile.json` sidecar.
- **Orchestration duplication.** `build.extract_speech_windows_for_file` hand-reproduces the `extract → reference-grid → speech_window_mask_for_file → None-fallback` sequence that `compute.py` already runs; only the leaf calls are shared, not the glue.
- **Duplicated embedding compute.** `extract_per_window_embeddings` is **not cache-wrapped** (`embeddings.py` calls `extract_speaker_embeddings_from_audios` directly). So the same audio is re-embedded across build (2.0/1.0) and analyze (1.0/0.5), and leave-one-file-out **re-extracts every sibling** because the artifact stores only centroids, not per-window vectors. This is the same gap as the deferred **Phase 6 / FR-015**.
- **Minor reuse.** Three cosine helpers now exist (`compare._cosine_distance`, `clustering._cos_sim`, `identity._cos_dist`) with inconsistent clipping; `DEFAULT_SPEECH_PRESENCE_LABELS` is hand-copied from `analyze_audio` (sync-by-comment, drift risk vs the FR-002 "same signal" guarantee); `build_speaker_profile --cache-dir` is parsed but inert.

**What is genuinely new (keep, not duplication):** cross-file dominant-cluster selection, session-weighted persisted per-model centroids, leave-one-file-out, SQUIM target-quality. No sklearn clustering is reimplemented in `speaker_profile/`.

### The decision to settle with the maintainer FIRST (do not pre-bake)

"Move behind the identity signal" hides a real semantic mismatch: the **identity axis is reference-free** (how many speakers / where do they change, within one recording) while the **profile is reference-based** (does this window match an externally-enrolled target). Folding the profile in is therefore not a free "move" — `aggregate_identity` would need a **new reference-based voter type**. Options:

- **(A) True identity voter.** Add a `distance-to-enrolled-centroid` voter so the profile uncertainty flows through `aggregate_identity` like the other identity voters; drop the separate `global_summary` fold + sidecar. Cleanest per Satra, but extends the identity aggregator's semantics (reference-based voter alongside reference-free ones) and risks conflating "second speaker present" with "not the enrolled target."
- **(B) Keep a distinct signal, stop the pretense.** Leave the profile as its own recording-level claim (it answers a different question), but **remove the inert vote injection** (or make it explicitly display-only) so nothing reads as feeding identity when it doesn't. Smaller, honest, preserves the semantic distinction; doesn't fully satisfy "move behind identity."
- **(C) Hybrid.** Profile votes become real identity voters for the *other-voice* sub-signal (which genuinely is an identity question), while *target-quality* stays a separate claim.

**DECIDED 2026-06-05 → (C) hybrid, recall-primary.** The recall gate is the primary downstream consumer (a recording with any non-subject voice must surface), so: the profile's *other-voice / subject-identity* signal feeds the identity axis as a corroborating reference-based voter **and** retains an **independent presence-gated per-window flag** (so it can fire where the identity axis isn't measuring identity — e.g. a background voice in an otherwise non-speech region); *target-quality* stays its own claim. Still worth running A/B/C past Satra, but C is the working decision. Independent of it, the reuse fixes (shared extraction helper, shared cosine, label import, embedding cache) are unconditional.

### Signal design (decided 2026-06-05) — continuous composable atoms, not verdicts

This feature is a **signal producer for downstream mixed metrics** (the rank-orderable attention metric lives in the separate triage/metric-ranking spec). So it must emit **named, continuous atoms** and *not* pre-blend them into one tuned score.

- **"Wrong subject" is a continuous certainty, not a binary flag.** Every voiced window already runs through `calibrate_cosine_uncertainty` against the profile band → a continuous `other_voice_uncertainty ∈ [0,1]`. The discrete `flag` is just a threshold on top and is **demoted to a derived/optional layer**, not the product. File-level "wrong subject" = aggregate of the per-window certainties (`subject_dominance` = certainty-weighted fraction of voiced time matching the subject; its complement is the "recording may not be the subject" uncertainty).
- **Profile comparison does double duty per voiced window:** (primary, strong) *who* — subject vs not, reference-based identity attribution; (secondary, weak) *is a real speaker present* — a window strongly matching a known-speaker template is evidence of genuine voiced speech, usable as presence corroboration.
- **Atoms to emit:**
  - per window — `subject_similarity` / `other_voice_uncertainty` (continuous, within-profile calibrated), `voice_present` (scene-derived: speech/babble/conversation, foreground **or** background), `flag` (derived, optional).
  - per file — `subject_dominance` (continuous) and its complement `wrong_subject_uncertainty`, `nonsubject_voice_fraction / peak / p95` (max-like recall atoms), `profile_confidence`. Kept as **distinct exposed fields, not collapsed** into one `single_speaker` `max()` — the downstream mixer needs the parts.

**Three discipline constraints:**

1. **Acyclic dependency.** Presence *gates* the profile (low voice-presence → not scored). Do **not** feed the profile back into that same presence signal (circular boost). Feed it forward to the identity axis + a separate "matches-known-speaker" corroboration atom the mixer can combine.
2. **Certainty paired with confidence-of-certainty.** Always emit the per-window/​file certainty *alongside* `profile_confidence` — a flimsy (`low`/thin) profile producing "high other-voice certainty" must be down-weighted, not believed.
3. **Within-profile calibration only.** Certainties are calibrated against *this* profile's band — meaningful within the recording, **not** comparable across recordings/subjects. Cross-file rank-ordering is explicitly the downstream metric spec's job.

**Scoring gate = voice presence, not clean-speech presence.** Build-time selects *clean subject speech* for the centroid; scoring-time should gate on *any voice present* (scene-classification speech/babble/conversation labels — already computed and feeding `speech_window_mask_for_file`) so **background/secondary voice is caught** while cough/breath/silence are excluded from the *voice* flag. These are two deliberately different uses of the presence signal.

**Overlap → lean on diarization.** Simultaneous subject+other speech yields a *mixed* embedding between centroids; profile-distance is moderate and unreliable there (same physics as the same-gender blind spot). Reliable overlap detection should come from **diarization overlap**, corroborating the profile rather than relying on profile-distance.

**Deferred hooks (emit the raw ingredients now; solve later, likely the triage spec):**
- *Non-subject cough/breath attribution* — speaker-embedding models are trained on speech; cough/breath is out-of-distribution and not reliably speaker-attributable. Do **not** force a subject/non-subject call on coughs with a speech model. Emit the raw ingredients (scene label `cough`/`breath`, the segment's diarization speaker) so a downstream combiner can attempt attribution later.
- *Task-aware contextualization* — "this is a respiration task so any voiced speech is anomalous regardless of who" combines task metadata (BIDS task label) + scene classification; the profile only adds "…and it's not the subject." Out of scope; emitting `voice_present` + scene labels keeps it composable downstream.

### Scope / sequencing

- The embedding-recompute fixes **are** Phase 6 (FR-015); fold this refactor in with it rather than treating them separately.
- Likely a **follow-up PR**, not #523 (which stays the signal-producer). Tracked here as **Phase 8** below.
- Contract impact to watch: profile artifact schema may gain optional per-window vectors (to kill LOO re-extraction → bump `schema_version`); the `analyze-audio-profile` integration contract changes if option A/C is chosen (votes become aggregated, not decorative).

## Open items carried to tasks (not blocking)

- Exact margins/defaults (ambiguity margin, adaptive-threshold percentiles, consensus weighting) will be set with **small empirical sweeps** during implementation, seeded with the literature/existing defaults documented above.
- Whether to expose the short-window detection pass as cached task or compute-on-demand is an implementation detail decided in tasks (R1 keying applies either way).
