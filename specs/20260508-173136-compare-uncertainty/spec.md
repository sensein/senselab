# Feature Specification: Comparison & Uncertainty Stage for analyze_audio.py

**Feature Branch**: `20260508-173136-compare-uncertainty`
**Created**: 2026-05-08
**Status**: Draft
**Input**: User description: "add a comparison & uncertainty stage to scripts/analyze_audio.py that produces (1) raw vs enhanced mismatches, (2) within-stream model disagreements, (3) cross-stream mismatches (ASR↔diarization, AST/YAMNet↔diarization, ASR phonemes↔PPG), (4) per-region uncertainty estimates harvested from underlying models. Output parquet + JSON + LS bundle. Hooks into the existing cache."

## Clarifications

### Session 2026-05-09

- **Output shape** — three per-bucket uncertainty time series (`presence`, `identity`,
  `utterance`), each in `[0, 1]`. No per-pair sidecars; pairwise WER / label-flip /
  boundary-shift exists only as in-memory intermediate state during cross-model aggregation.
- **Contributing models per axis** — maximally inclusive: every model whose output naturally
  encodes the axis contributes. presence = pyannote + Sortformer + 4 ASR (token-overlap) +
  AST + YAMNet (Speech allowlist); identity = pyannote + Sortformer (cross-model labels) +
  ECAPA + ResNet (across-time cosine) + same-model raw-vs-enhanced; utterance = 4 ASR
  transcripts (pairwise mean WER) + Whisper `avg_logprob` + PPG phoneme-error-rate.
  Models with a null signal on a bucket are dropped (no zero-imputation).
- **Per-axis aggregation** — presence uses Shannon entropy normalized by `log(n_models)`;
  identity uses `1 − (fraction of agreeing label pairs)` + mean cosine distance + raw/enh
  binary; utterance uses mean pairwise normalized edit distance + `1 − exp(avg_logprob)` +
  PPG PER. Within each axis, sub-signals collapse via `--uncertainty-aggregator` (default
  `min` over confidences ≡ `max` over uncertainties).
- **Bucket grid** — `--cross-stream-win-length 0.5` / `--cross-stream-hop-length 0.5` (i.e.
  non-overlapping) by default. AST / YAMNet windows project onto this grid via
  `floor(start / native_hop)`.
- **Text-only ASR** — Granite Speech 3.3 and Canary-Qwen emit a `ScriptLine` without
  per-token chunks; the comparator consults the post-MMS alignment block when the raw ASR
  result lacks timestamps. Text without any time anchor is treated as
  `asr_says_speech = false` and contributes no transcript on the bucket.
- **Speech-presence label parsing** — `--speech-presence-labels` takes `nargs="+"` since
  AudioSet labels themselves contain commas (e.g. `"Narration, monologue"`).
- **Plot layout** — 5-row figure: presence, identity, utterance (each with raw solid +
  enhanced dashed overlay on a [0, 1] y-axis), raw-vs-enhanced delta strip with one band per
  axis, reference row (raw diar speakers + raw ASR token spans).
- **Disagreements ranking** — `disagreements.json` ranks rows across the 9 parquets by
  `aggregated_uncertainty desc`, ties broken by axis priority (utterance > identity >
  presence) then start time. Default top-N = 100; opt-out via `--disagreements-top-n 0`.
- **Phoneme-disagreement threshold** — two-tier: continuous `phoneme_per` on every ASR↔PPG
  contribution, plus a boolean `phoneme_disagreement` set when `phoneme_per ≥
  --phoneme-disagreement-threshold` (default `0.50`). The continuous PER feeds the utterance
  aggregator; the boolean controls LS region emission for the PPG sub-signal.
- **LS bin thresholds** — `aggregated_uncertainty < 0.33` → `low`, `[0.33, 0.66)` → `medium`,
  `≥ 0.66` → `high`. Plus `incomparable` and `unavailable` for degraded buckets.
- **Cache schema** — `_CACHE_SCHEMA_VERSION = 1`. Cache key embeds `(audio_signature, axis,
  pass_set, model_set, params, wrapper_hash, senselab_version, schema_version)`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — "Did enhancement help here?" at a glance (Priority: P1)

A reviewer runs `analyze_audio.py` against a recording and gets back, alongside the existing per-task outputs, three raw-vs-enhanced uncertainty time series (presence / identity / utterance). Wherever the raw and enhanced passes diverge — pyannote shifted speech boundaries, Whisper edited a word, the speaker label flipped — the corresponding axis's `aggregated_uncertainty` rises in that bucket. The reviewer scrubs the LS `pass_pair__uncertainty__*` tracks to audit the divergence in seconds rather than diff-ing two JSON files by hand.

**Why this priority**: This answers the original ask — "did enhancement help, hurt, or no-op for this clip?" — directly. It is the cheapest cut to deliver because it reuses the same per-axis aggregators as US2; only the input changes (votes from raw pass vs enhanced pass for the same model are compared on each row).

**Independent Test**: Run the script in default two-pass mode. Verify three parquets exist at `<run_dir>/uncertainty/raw_vs_enhanced/{presence,identity,utterance}.parquet`, each with non-empty rows for buckets where the two passes disagreed and the silent-on-both rule (FR-012) excluded buckets where neither pass had any signal. Verify three matching `pass_pair__uncertainty__*` Labels tracks land in `labelstudio_tasks.json` with bins drawn from `{low, medium, high, incomparable, unavailable}`.

**Acceptance Scenarios**:

1. **Given** both passes successfully ran with the default model set, **When** the comparator runs, **Then** three parquets emit at `<run_dir>/uncertainty/raw_vs_enhanced/{presence,identity,utterance}.parquet`. Each row's `aggregated_uncertainty ∈ [0, 1]` reflects how much the two passes diverged on that axis at that bucket, and `model_votes` carries each contributing model's raw raw-pass and enhanced-pass signals so the reviewer can audit the source of the divergence.
2. **Given** raw and enhanced agreed on a bucket for an axis, **When** the comparator runs, **Then** the row is emitted with `aggregated_uncertainty` near `0.0`. Buckets where no model contributed any signal on either pass emit no row at all (FR-012).
3. **Given** one pass failed for a particular model, **When** the comparator runs, **Then** that model contributes a `one_sided` vote to the bucket and the row's `comparison_status="one_sided"` flag is set, so reviewers can distinguish "enhancement changed the answer" from "enhancement broke the model".

---

### User Story 2 — Where do my models disagree on this clip? (Priority: P2)

The same reviewer runs the script with multiple models per task (default: pyannote + Sortformer for diar, four ASR models, AST + YAMNet for scene). They want to see, per axis, where those models disagree among themselves on the *same* input — does pyannote think the bucket is silence while Whisper has a token there? Did pyannote and Sortformer assign different speaker labels to the same bucket? Are the four ASR models' transcripts mutually inconsistent?

**Why this priority**: Raw-vs-enhanced is one mismatch axis (US1); per-pass cross-model agreement is the second. The same workflow function (`compute_uncertainty_axes`) returns both — only the output slice differs.

**Independent Test**: Run with the default model set. Verify six parquets land at `<run_dir>/<pass>/uncertainty/{presence,identity,utterance}.parquet` (3 axes × 2 passes). Verify each row's `contributing_models` lists the actual models that voted on that bucket; verify the LS bundle contains six matching `<pass>__uncertainty__*` Labels tracks plus the utterance TextArea sibling.

**Acceptance Scenarios**:

1. **Given** at least two models contributed signal to an axis on a bucket, **When** the comparator runs, **Then** the per-pass parquet for that axis emits a row with `aggregated_uncertainty` derived from the per-axis rule (entropy for presence; cross-model label disagreement + cosine for identity; pairwise mean WER + native confidences for utterance).
2. **Given** only one model contributed to an axis on a bucket, **When** the comparator runs, **Then** the row is still emitted but `aggregated_uncertainty` falls back to `1 − native_confidence` (when the model exposes one) or `0.0` (when no native signal exists). The fallback is documented in `disagreements.json`.
3. **Given** two ASR models produce native timestamps at very different granularities (Whisper word-level vs Qwen3-ASR sentence-level vs Granite text-only via MMS alignment), **When** the comparator runs, **Then** all four are projected onto the bucket grid (FR-010) before voting, so the per-bucket transcript text on each row reflects only the tokens whose timestamps overlap the bucket.

---

### User Story 3 — Cross-stream sanity contributions to the three axes (Priority: P2)

Cross-stream "should-agree" signals (ASR text in a region pyannote calls silence; AST/YAMNet top-1 = `Speech` while diar says non-speech; ASR-implied phoneme sequence vs PPG phoneme posteriors) feed *into* the three uncertainty axes — they do not produce separate output files. ASR token-overlap and AST/YAMNet Speech-allowlist contribute votes to the **presence** axis. ASR-vs-PPG phoneme-error-rate contributes to the **utterance** axis as one of three sub-signals.

**Why this priority**: Treating cross-stream as a separate output channel was the v1 design; the v2 design folds it into the maximally-inclusive contribution policy of FR-002. The disagreement still surfaces — just as a divergence between models on the same axis row — and the row's `model_votes` exposes which model is the outlier.

**Independent Test**: Run on a clip where pyannote says silence in `[12.0, 12.3]` but Whisper returns "hello" with word-level timestamps in that range. Verify the presence parquet's row for the `[12.0, 12.5]` bucket has `model_votes["pyannote/..."].speaks=False` and `model_votes["openai/whisper-large-v3-turbo"].speaks=True`, and that `aggregated_uncertainty` is high (entropy of mixed votes). Run a second clip with PPG provisioned and verify the utterance row's `model_votes["openai/whisper-large-v3-turbo"].phoneme_per_to_ppg` is populated.

**Acceptance Scenarios**:

1. **Given** ASR and diarization both ran, **When** the comparator runs, **Then** the presence parquet's row for each disagreeing bucket carries `speaks` votes from every contributing diar model and every contributing ASR model. `speaks` from an ASR model is `True` iff at least one transcript token whose timestamp range overlaps the bucket exists (per FR-011); pure-punctuation tokens and text without a time anchor count as `False`.
2. **Given** AST or YAMNet ran, **When** the comparator runs, **Then** their top-1 label is mapped through the `--speech-presence-labels` allowlist (default: AudioSet "Speech" subtree) to a `speaks` boolean and contributes that vote to the presence parquet row for every bucket their native window covers.
3. **Given** an ASR run plus a successful PPG run, **When** the comparator runs, **Then** each ASR model's `model_votes[asr].phoneme_per_to_ppg ∈ [0, 1]` is computed via grapheme-to-phoneme on the bucket's transcript and contributes to the utterance row's `aggregated_uncertainty`.
4. **Given** PPG is disabled or unavailable, **When** the comparator runs, **Then** `phoneme_per_to_ppg` is null for every ASR vote, the sub-signal drops out of the utterance aggregator (no zero-imputation), and the omission is recorded in `disagreements.json` under `incomparable_reasons`.

---

### User Story 4 — Ranked discovery via disagreements.json + timeline plot (Priority: P3)

Beyond the per-axis parquets, reviewers want a single ranked view that surfaces the top-N most-uncertain buckets across the entire run, plus a 5-row timeline plot that overlays raw vs enhanced for each axis at a glance. Native model confidences (Whisper `avg_logprob`, AST top-1 score, ECAPA cosine) ride inside `model_votes` so reviewers can drill from "this bucket is uncertain" down to "and here's which model said what".

**Why this priority**: Per-axis parquets answer "is this bucket uncertain on axis X?"; the index and plot answer "where should I look first?". Ranked discovery is a force multiplier on US1–US3 but not the primary deliverable.

**Independent Test**: Run on a clip with at least one high-uncertainty bucket per axis. Verify `<run_dir>/disagreements.json` ranks rows across the 9 parquets by `aggregated_uncertainty` desc, with axis-priority tiebreak (utterance > identity > presence) and start-time secondary tiebreak. Verify `<run_dir>/timeline.png` is a 5-row figure (presence raw+enhanced overlay, identity raw+enhanced overlay, utterance raw+enhanced overlay, raw-vs-enhanced delta strip with one band per axis, reference row).

**Acceptance Scenarios**:

1. **Given** Whisper produced chunks with `avg_logprob`, **When** the comparator runs, **Then** the utterance row's `model_votes["openai/whisper-large-v3-turbo"].avg_logprob` is populated and the bucket-level `1 − exp(avg_logprob)` contributes to that row's `aggregated_uncertainty` per FR-002.
2. **Given** a model exposes no native confidence (e.g. Sortformer post-processed labels), **When** the comparator runs, **Then** that model's `native_confidence` field is null in the model_votes struct and the model's vote is dropped from the aggregator's input rather than treated as zero (FR-004). The list of such models is documented in `disagreements.json`'s `models_without_native_signal`.
3. **Given** the comparator emits N>0 rows total and `--disagreements-top-n > 0`, **When** the index is built, **Then** `disagreements.json` lists the top-N rows with `aggregated_uncertainty desc`, axis-priority tiebreak (utterance > identity > presence), and start-time secondary tiebreak; each entry carries the source parquet path, row index, and matching `ls_region_id` so the reviewer can drill in.
4. **Given** the run completed with at least one axis_result, **When** the timeline plot is built, **Then** `<run_dir>/timeline.png` is a 5-row figure: presence (raw solid + enhanced dashed), identity (same overlay), utterance (same overlay), raw-vs-enhanced delta strip per axis, reference (raw diar speaker bars + raw ASR token spans).

---

### Edge Cases

- A pass failed entirely (e.g., enhancement crashed) — the comparator emits no rows for that pass (per FR-012) and records the failure reason in `disagreements.json`'s `incomparable_reasons` block (per FR-013).
- A model only produced output on one pass and not the other — the affected rows on the raw-vs-enhanced parquet carry `comparison_status: "one_sided"` and that model's vote is excluded from the aggregator's input for the missing pass.
- Multiple ASR models produce wildly different time granularities (sentence-level vs word-level vs no-timestamps-then-MMS-aligned) — every contributing ASR is projected onto the FR-010 bucket grid before voting; text-only ASR results consult the post-MMS alignment block per FR-011.
- The PPG backend is not installed or the audio is too short for the backend to run — `phoneme_per_to_ppg` is null on every ASR vote in the utterance parquet, the sub-signal drops out of the aggregator (no zero-imputation), and the omission is logged under `incomparable_reasons` in `disagreements.json`.
- Confidence signals exist at different time resolutions than the bucket grid (e.g., Whisper per-word `avg_logprob` vs AST per-10.24 s window) — the bucket-level scalar is the arithmetic mean of overlapping per-event values; the original per-event values remain reachable via the `model_votes` column on the same parquet row.
- LS Labels tracks require an enumerated label set — values come from the fixed set ``{"low", "medium", "high", "incomparable", "unavailable"}`` (mapped from ``aggregated_uncertainty`` bins per FR-005).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The script MUST add a new pipeline stage that runs after every existing per-task stage in both passes and produces three per-bucket uncertainty time series — ``presence_uncertainty`` (was there a speaker?), ``identity_uncertainty`` (was it the same speaker?), and ``utterance_uncertainty`` (what was said?) — each scaled to ``[0, 1]``.

- **FR-002**: The contributing model set per axis is maximally inclusive (every model whose output naturally encodes that axis votes):
    - **presence** — diarization (pyannote, Sortformer) + ASR token-overlap (whisper-turbo, granite, canary-qwen, qwen3-asr) + scene Speech-allowlist (AST, YAMNet). Up to 8 binary votes per bucket; computed via Shannon entropy normalized by ``log(n_contributing_models)``.
    - **identity** — three sub-signals: (i) cross-model speaker-label disagreement = ``1 − (n_agreeing_pairs / n_pairs)`` over the speaker labels assigned by every diar model with a segment overlapping the bucket; (ii) same-model raw-vs-enhanced speaker-label disagreement (only emitted on the raw-vs-enhanced parquet); (iii) across-time speaker-change distance = ``1 − cos_similarity(emb_this_bucket, emb_prev_bucket_same_track)`` using per-segment ECAPA / ResNet embeddings projected onto the bucket grid via diarization timestamps. The three sub-signals collapse via ``--uncertainty-aggregator``.
    - **utterance** — three sub-signals: (i) ASR pairwise mean WER (4 transcripts → 6 pairs averaged, clipped to ``[0, 1]``); (ii) Whisper native ``1 − exp(avg_logprob)`` averaged across overlapping chunks; (iii) ASR-vs-PPG phoneme-error-rate when PPG is available. Combined via ``--uncertainty-aggregator``.

    A model that emits a null signal in a bucket is dropped from that bucket's vote (no zero-imputation). Pairwise per-pair comparison artefacts are NOT an output — they exist only as in-memory intermediate state during the cross-model aggregation step.

- **FR-003**: The comparator MUST emit exactly the following per-run parquet layout: ``<run_dir>/<pass>/uncertainty/{presence,identity,utterance}.parquet`` (one parquet per pass per axis, six total for the default two-pass run), plus ``<run_dir>/uncertainty/raw_vs_enhanced/{presence,identity,utterance}.parquet`` (three deltas across passes). Each parquet row has columns ``start``, ``end``, ``axis``, ``aggregated_uncertainty`` ([0, 1]), ``contributing_models`` (list), ``model_votes`` (dict[model_id → raw signal: bool for presence, label for identity, transcript or float for utterance]), ``comparison_status`` ({"ok", "incomparable", "unavailable", "one_sided"}).

- **FR-004**: A top-level ``<run_dir>/disagreements.json`` index ranks rows across all nine parquets by ``aggregated_uncertainty`` desc, with ties broken by axis priority (utterance > identity > presence) and then by start time. N defaults to 100, configurable via ``--disagreements-top-n``. The aggregator combining sub-signals into ``aggregated_uncertainty`` is configurable via ``--uncertainty-aggregator``; allowed values are ``min`` (default), ``mean``, ``harmonic_mean``, and ``disagreement_weighted`` (``(1 − mean_confidence) × mismatch_severity``). Models lacking a native confidence signal MUST be dropped from the aggregator's input rather than treated as zero.

- **FR-005**: The Label Studio bundle MUST expose three Labels tracks per pass (``<pass>__uncertainty__{presence,identity,utterance}``) plus three raw-vs-enhanced tracks (``pass_pair__uncertainty__{presence,identity,utterance}``). Label values are drawn from the fixed enumerated set ``{"low", "medium", "high", "incomparable", "unavailable"}`` mapping ``aggregated_uncertainty`` to bins (``< 0.33`` → low, ``[0.33, 0.66)`` → medium, ``≥ 0.66`` → high). The utterance track additionally declares a sibling ``<TextArea>`` track carrying the per-bucket transcript consensus + dissenting-model transcripts so reviewers can audit which words drove the uncertainty.

- **FR-006**: The summary timeline plot MUST be a five-row figure: (1) presence uncertainty, raw pass solid + enhanced pass dashed on a shared y-axis in [0, 1]; (2) identity uncertainty, same overlay; (3) utterance uncertainty, same overlay; (4) raw-vs-enhanced delta strip with one band per axis; (5) reference context — raw diarization speakers + raw ASR token spans.

- **FR-007**: The comparator MUST hook into the existing content-addressable cache. The cache key MUST include ``(audio_signature, axis, pass_set, model_set, params, wrapper_hash, senselab_version, schema_version)``. Re-running with the same inputs MUST return ``cache="hit"`` for every axis without re-reading the upstream task results.

- **FR-008**: The comparator MUST be skippable via ``--skip comparisons``; the disagreements index is opt-out via ``--disagreements-top-n 0``. The per-region confidence columns inherited from upstream tasks are always emitted when the underlying signal is available.

- **FR-009**: ASR-vs-PPG phoneme comparison MUST run only when both an ASR result and a PPG result are present for the same audio variant, MUST project both to a shared phoneme inventory before differencing, and MUST report ``phoneme_per`` (continuous normalized phoneme error rate) plus a boolean ``phoneme_disagreement`` set only when ``phoneme_per`` meets or exceeds ``--phoneme-disagreement-threshold`` (default ``0.50``). The continuous PER feeds the utterance axis aggregator; the boolean controls LS region emission.

- **FR-010**: The bucket grid is controlled by ``--cross-stream-win-length`` (default ``0.5`` s) and ``--cross-stream-hop-length`` (default ``0.5`` s, non-overlapping). AST / YAMNet windows are projected onto this grid using floor-based index lookup (``win_idx = floor(start / native_hop)``). The grid parameters MUST be recorded in every parquet's provenance.

- **FR-011**: For text-only ASR backends that emit a ``ScriptLine`` without per-token chunks (Granite Speech 3.3, Canary-Qwen), the comparator MUST consult the alignment block produced by the post-MMS auto-aligner when the raw ASR result lacks timestamps. Text without any time anchor (raw ScriptLine and no alignment block) is treated as ``asr_says_speech = false`` — it produces no token overlap and no contribution to utterance text on that bucket.

- **FR-012**: Buckets where no contributing model produced a usable signal on an axis MUST emit no row (silent-bucket rule extended uniformly to all axes).

- **FR-013**: The comparator MUST gracefully degrade — a single signal failing (e.g., one model output is empty, the PPG backend is unavailable) MUST NOT abort the run; the affected rows MUST instead carry a ``comparison_status`` of ``"incomparable"`` or ``"unavailable"`` with a one-line reason in ``disagreements.json``.

- **FR-014**: All uncertainty outputs MUST be reproducible — the wrapper version hash and senselab version MUST be recorded in every parquet's provenance, and reviewers MUST be able to diff a parquet from one run against another to track stability.

- **FR-015**: The smoke-test suite MUST exercise the three-axis aggregation logic on synthetic / cached upstream results so the comparator can be tested without re-running heavy models. Existing skipif gates for venv- or weight-bound backends MUST continue to apply to the integration test that exercises the full stage.

### Key Entities

- **UncertaintyRow**: One row per (pass, axis, start, end) bucket in an uncertainty parquet. Carries ``axis`` ({"presence", "identity", "utterance"}), ``aggregated_uncertainty`` ([0, 1]), ``contributing_models`` (list of model_ids), ``model_votes`` (dict[model_id → raw signal]), ``comparison_status``, plus axis-specific extras (e.g., per-sub-signal scalars for identity / utterance).
- **DisagreementsIndex**: The top-level JSON file enumerating the top-N flagged buckets across the nine parquets, with each entry pointing to the source parquet path, row index, time span, axis, and LS region id.
- **PerSegmentSpeakerEmbedding**: A `(audio_signature, diarization_segment, embedding_model)` keyed cache entry holding the ECAPA / ResNet vector for one diarization segment; reused at bucket-aggregation time to compute the across-time identity sub-signal.
- **BucketGrid**: The shared time grid used to project all model outputs onto a common axis before aggregation. Parameters: ``--cross-stream-win-length`` (default 0.5 s), ``--cross-stream-hop-length`` (default 0.5 s).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a typical 1-minute conversational clip, the reviewer can open the Label Studio project, identify the top-5 most uncertain buckets via the ``disagreements.json`` index, and locate each one on the audio timeline in under 30 seconds total.
- **SC-002**: For a default two-pass run with default models, the comparator emits exactly 9 uncertainty parquets (3 axes × 2 passes + 3 raw-vs-enhanced deltas) — no per-pair sidecars. Buckets where no model contributed any signal on an axis emit no row.
- **SC-003**: Cache replay on identical inputs hits the comparator cache for ≥95 % of buckets; the remaining ≤5 % are documented in ``disagreements.json`` as deliberately uncached (e.g., one-sided rows that depend on which pass succeeded).
- **SC-004**: The comparator stage adds no more than 30 % wall-clock overhead on top of the existing analyze_audio run on a 1-minute clip when running on already-cached upstream task results.
- **SC-005**: Existing analyze_audio outputs (per-task JSON, features parquets, existing LS tracks) are unchanged in shape — adding the comparator stage MUST NOT break any consumer of the existing artefacts. A reviewer who skips the new stage with ``--skip comparisons`` gets bit-for-bit the same outputs the script produces today.
- **SC-006**: Smoke tests cover at least one happy-path scenario per axis (presence / identity / utterance) plus the documented degradation modes (failed pass, one_sided, unavailable PPG, missing confidence signal, single-model bucket); the tests run in under 30 seconds total without venv provisioning or model downloads.
- **SC-007**: The reviewer can answer "did enhancement help here?" for a clip in under one minute by visiting the LS project and reading the three ``pass_pair__uncertainty__*`` tracks.

## Assumptions

- The PPG-based phoneme sequence is obtained via senselab's existing PPG backend (``ppgs`` extra). The user's earlier work already produced ``artifacts/sample1_ppg_segments.json`` and a phoneme timeline visualization, so the dependency is available.
- The "shared phoneme inventory" for ASR↔PPG comparison defaults to the inventory used by the PPG backend; the ASR side is mapped onto it via grapheme-to-phoneme on the transcript (English-only on first pass; ja/zh covered by the existing uroman pipeline added in the prior PR).
- "Speech-presence" mapping for AST/YAMNet uses a fixed AudioSet label allowlist (e.g., ``Speech``, ``Conversation``, ``Narration, monologue``, ``Female speech, woman speaking``, ``Male speech, man speaking``, ``Child speech, kid speaking``); the allowlist is documented and the user can override it via a CLI flag if needed.
- The comparison stage runs on top of *cached* upstream results whenever possible; the wrapper version hash invalidation rules from the prior PR continue to apply.
- The Label Studio bundle is opened in a workspace that already accepts the script's existing config XML; no LS-side config migration is required beyond appending the new tracks declared in the per-run XML.
- Uncertainty signals are read from the *first available* native source per backend; a follow-up may add ensemble / Monte-Carlo Dropout style uncertainty, but those are explicitly out of scope for v1.
- The disagreements index ranking uses a single combined uncertainty score per region. The default aggregator is ``min`` of the available per-model confidences (most-doubtful-model wins); the user can switch to ``mean``, ``harmonic_mean``, or ``disagreement_weighted`` via ``--uncertainty-aggregator``. Models lacking a native confidence signal are dropped from the input rather than treated as zero.
- Backwards compatibility: the spec MUST NOT change any existing per-task output shape or filename. New outputs are strictly additive under ``<run_dir>/<pass>/comparisons/`` and ``<run_dir>/disagreements.json``.
