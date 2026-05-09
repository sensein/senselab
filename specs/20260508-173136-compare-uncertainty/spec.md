# Feature Specification: Comparison & Uncertainty Stage for analyze_audio.py

**Feature Branch**: `20260508-173136-compare-uncertainty`
**Created**: 2026-05-08
**Status**: Draft
**Input**: User description: "add a comparison & uncertainty stage to scripts/analyze_audio.py that produces (1) raw vs enhanced mismatches, (2) within-stream model disagreements, (3) cross-stream mismatches (ASR↔diarization, AST/YAMNet↔diarization, ASR phonemes↔PPG), (4) per-region uncertainty estimates harvested from underlying models. Output parquet + JSON + LS bundle. Hooks into the existing cache."

## Clarifications

### Session 2026-05-08

- Q: When two ASR models cover the same time window with different transcripts, what does the comparator record as the disagreement signal? → A: per-window WER (word error rate) with one model designated the soft reference, plus both transcripts retained on the row
- Q: How should the comparator combine multiple per-model confidences into a single sortable score per region for the disagreements index? → A: ``min`` of available per-model confidences by default; aggregator selectable via a ``--uncertainty-aggregator {min,mean,harmonic_mean,disagreement_weighted}`` flag
- Q: What time grid should cross-stream comparisons use by default? → A: dedicated ``--cross-stream-win-length`` / ``--cross-stream-hop-length`` flags; defaults 0.2 s / 0.1 s (decoupled from the features grid so each can be tuned independently)
- Q: What rule does the comparator use to classify a window as "ASR says speech"? → A: at least one transcript token whose timestamp range overlaps the window (works uniformly across native-timestamp models and MMS-auto-aligned text-only models; pure punctuation / hallucinated text without time anchor → not speech)
- Q: What threshold on the per-segment normalized phoneme edit distance triggers a ``phoneme_disagreement = true`` flag? → A: two-tier — emit a continuous ``phoneme_per`` (phoneme error rate) on every row, and set a boolean ``phoneme_disagreement = true`` only when ``phoneme_per >= 0.50``; threshold tunable via ``--phoneme-disagreement-threshold``

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Spot raw-vs-enhanced disagreements at a glance (Priority: P1)

A reviewer runs `analyze_audio.py` against a recording and gets back, alongside the existing per-task outputs, a single timeline that flags every region where the same task / model produced a different result on the raw input vs the speech-enhanced input. Wherever pyannote shifted speech boundaries, Whisper edited a word, AST flipped its top-1 label, etc., the reviewer sees a labelled region in the Label Studio bundle so they can scrub the timeline and audit the discrepancy in seconds rather than diff-ing two JSON files by hand.

**Why this priority**: This is the original ask — "temporally-coded summary of mismatches between raw and enhanced versions". It is the cheapest one to deliver because it operates entirely on outputs that already exist in the summary, requires no new model calls, and unlocks the first concrete reviewer workflow (does enhancement help, hurt, or no-op for this clip?).

**Independent Test**: Run the script twice on the same audio (one with `--no-enhancement`, one with the default two-pass mode); the first run produces no comparisons (raw-only), the second run emits a `raw_vs_enhanced.parquet` with one row per (task, model, time-bucket) where the two passes disagree, plus matching LS regions in `labelstudio_tasks.json`. Reviewer can open the LS project, scrub the timeline, and see the disagreements highlighted.

**Acceptance Scenarios**:

1. **Given** an audio with both passes successfully run, **When** the comparison stage runs, **Then** it produces a per-pass-pair parquet listing every (task, model, start, end, raw_value, enhanced_value, mismatch_type) where the two passes disagree above a per-task threshold, plus matching LS Labels-track regions on the audio timeline.
2. **Given** raw and enhanced agreed on a region, **When** the comparison stage runs, **Then** that region produces NO row in the parquet and NO LS region (silence is the signal that nothing flipped).
3. **Given** one pass failed for a particular task / model, **When** the comparison stage runs, **Then** the affected rows are emitted with a `comparison_status: "incomparable"` flag rather than being silently dropped, so reviewers can see *why* a comparison was skipped.

---

### User Story 2 — Surface within-stream model disagreements (Priority: P2)

The same reviewer also runs the script with multiple models per task (the current default already configures pyannote + Sortformer for diarization, four ASR models, AST + YAMNet for scene classification). They want to see, on the same timeline, where those models disagree among themselves on the same input — pyannote and Sortformer disagreeing on whether anyone is speaking, Whisper and Granite disagreeing on the transcript text in a region, AST and YAMNet disagreeing on the top-1 scene label.

**Why this priority**: Raw vs enhanced is one mismatch axis; within-stream is the second. Doing it after raw-vs-enhanced lets us reuse the same comparator framework and the same LS bundle shape. This generalizes the existing `scene_agreement.json` output (AST vs YAMNet only, present-or-absent) into a uniform per-task parquet with explicit start/end and per-pair disagreement type.

**Independent Test**: Run with the default model set; the output bundle includes one parquet per `(task, model_a, model_b)` pair under `<pass>/comparisons/<task>__<a>_vs_<b>.parquet` plus matching LS Labels tracks named `<pass>__compare__<task>__<a>_vs_<b>` with values `agree` / `disagree-X-Y`. Reviewer can confirm the per-pair pyannote/Sortformer track shows disagreement in regions where the two models partition the audio differently.

**Acceptance Scenarios**:

1. **Given** at least two models successfully ran for a task in the same pass, **When** the comparison stage runs, **Then** it emits one parquet per ordered pair plus matching LS tracks; the existing `scene_agreement.json` semantics are preserved (subset of this output).
2. **Given** the user passed only one model for some task, **When** the comparison stage runs, **Then** that task produces no within-stream parquet (graceful no-op) without erroring.
3. **Given** the two ASR models produce native timestamps at very different granularities (Whisper 30 s chunks vs Qwen3-ASR word-level), **When** the comparison stage runs, **Then** both are projected onto a shared time grid (same grid used by the script's existing windowed features) so disagreements are aligned to comparable buckets.

---

### User Story 3 — Cross-stream sanity checks (Priority: P2)

The reviewer wants to catch the cases where two *different* tasks should agree but don't: an ASR model returned text in a region pyannote labelled as silence, AST flagged "speech" in a region the diarization model said was non-speech, the ASR-derived phoneme sequence diverges from the PPG-based phoneme sequence over the same span. Each of these is a strong "look here" signal even when no individual model says it's uncertain.

**Why this priority**: Cross-stream is the most analytically valuable but also the most opinionated — it requires a reviewer-defined notion of what counts as "should agree." Doing it after the within-stream MVP lets us validate the comparator framework first and adopt the same LS bundle shape.

**Independent Test**: Run with the default model set on a clip that contains brief silence + brief speech segments. The `cross_stream.parquet` flags (a) regions where Whisper produced text but pyannote said silence (or vice versa); (b) regions where AST/YAMNet top-1 is in the {"Speech", "Conversation"} synset but diarization says non-speech (or vice versa); (c) regions where the ASR-implied phoneme sequence (via grapheme-to-phoneme on the transcript) and the PPG-based phoneme posterior sequence have edit distance above a threshold. Reviewer scrubs to the flagged regions and confirms the mismatch is real.

**Acceptance Scenarios**:

1. **Given** ASR and diarization both ran, **When** the cross-stream stage runs, **Then** for every grid-aligned window it emits `(asr_says_speech, diar_says_speech, agree)` and only the disagreements get an LS region. ``asr_says_speech`` is true iff at least one transcript token (word) whose timestamp range overlaps the window is present, so pure-punctuation outputs and hallucinated text without a time anchor do not count.
2. **Given** AST/YAMNet and diarization both ran, **When** the cross-stream stage runs, **Then** the AudioSet "Speech" / "Conversation" / "Narration" labels are mapped to a binary speech-presence flag and compared to diarization the same way.
3. **Given** a successful ASR run and a successful PPG run, **When** the cross-stream stage runs, **Then** the parquet contains a continuous `phoneme_per` per segment plus a `phoneme_disagreement` boolean set when `phoneme_per >= --phoneme-disagreement-threshold` (default 0.50); LS regions are produced only for segments where `phoneme_disagreement` is true.
4. **Given** the PPG path is disabled or fails, **When** the cross-stream stage runs, **Then** the ASR↔phoneme comparison is skipped with a clear note in `disagreements.json` rather than the whole stage failing.

---

### User Story 4 — Per-region uncertainty harvested from the models themselves (Priority: P3)

Beyond comparing models against each other, the reviewer wants to see each model's own confidence about every region it produced — Whisper's `avg_logprob` and `no_speech_prob`, pyannote's per-frame score, AST/YAMNet's top-1 score plus the entropy of the score distribution, Granite / Canary-Qwen generation confidence (token-level mean log-prob), and the MMS forced-aligner's per-segment score. Those numbers ride alongside every existing region in the LS bundle as a numeric annotation, and they ride in the parquet so reviewers can rank regions by uncertainty.

**Why this priority**: Uncertainty is a force multiplier for the previous three stories — high-uncertainty disagreements are far more interesting than low-uncertainty ones. But it touches every backend differently and is partially a metadata-plumbing exercise; doing it last lets us validate the disagreement story first and only then layer uncertainty on top of it.

**Independent Test**: Run the script. Each existing per-task parquet (or its successor) gains a `confidence` and `uncertainty` column populated from the underlying model's native signal; the `disagreements.json` index now ranks flagged regions by combined uncertainty so the reviewer can open the top-k worst regions first.

**Acceptance Scenarios**:

1. **Given** Whisper produced segments with `avg_logprob` and `no_speech_prob`, **When** uncertainty extraction runs, **Then** those values appear as `confidence` and `no_speech_prob` columns on every Whisper region in both the parquet and the LS payload.
2. **Given** AST emitted a full top-K distribution per window, **When** uncertainty extraction runs, **Then** every AST window gets a `top1_score`, `entropy`, and `margin_to_top2` column.
3. **Given** a model that does not natively expose any confidence signal (e.g., Sortformer post-processed labels), **When** uncertainty extraction runs, **Then** the `confidence` column is null and a single line in `disagreements.json` documents which models lacked a native signal.
4. **Given** the disagreements index is built, **When** the reviewer sorts by combined uncertainty, **Then** the highest-uncertainty disagreements appear first and link to the matching LS region IDs.

---

### Edge Cases

- A pass failed entirely (e.g., enhancement crashed) — the comparison stage records `pass_status: "failed"` and emits an empty parquet rather than crashing the whole run.
- A model only produced output on one pass and not the other — comparison rows are emitted with `comparison_status: "one_sided"` and counted distinctly from genuine disagreements.
- Multiple ASR models produce wildly different time granularities (sentence-level vs word-level vs no-timestamps-then-MMS-aligned) — the comparator projects everything to a configurable shared time grid before differencing, with the grid documented in the parquet provenance.
- The PPG backend is not installed or the audio is too short for the backend to run — cross-stream phoneme comparison records `phoneme_status: "unavailable"` rather than failing.
- Confidence signals exist but at different time resolutions than the comparison grid (e.g., Whisper per-word `avg_logprob` vs AST per-10.24 s window) — confidence is averaged within each grid bucket and the original per-event values are kept in a sidecar parquet for callers who need full resolution.
- An LS Labels track requires an enumerated label set, but `disagree-X-Y` is open-ended over many model pairs — labels are pre-declared in the LS XML config as the cartesian product of (task, model_pair) actually used in the run.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The script MUST add a new pipeline stage that runs after every existing per-task stage in both passes and produces a *comparison* result. Rows are keyed by ``(comparison_kind, task, model_a, model_b, pass_a, pass_b, start, end)`` for within-stream and raw-vs-enhanced (note that for raw-vs-enhanced, ``model_a == model_b`` and ``pass_a != pass_b``; for within-stream the inverse holds), and by ``(comparison_kind, stream_a, stream_b, pass, start, end)`` for cross-stream pairs.
- **FR-002**: The comparison stage MUST emit one parquet per (pass, comparison_kind, task, model_pair) combination — i.e., one file per ordered model pair — at ``<run_dir>/<pass>/comparisons/<task>/<a>_vs_<b>.parquet`` for within-stream and ``<run_dir>/<pass>/comparisons/cross_stream/<stream_a>__vs__<stream_b>.parquet`` for cross-stream, with at minimum the columns ``start``, ``end``, ``a_value``, ``b_value``, ``agree``, ``mismatch_type``, ``comparison_status``. Raw-vs-enhanced files mirror the same per-model layout under ``<run_dir>/comparisons/raw_vs_enhanced/<task>/<model>.parquet`` (one file per model carrying both passes' values on each row). For ASR-vs-ASR comparisons, the parquet additionally carries ``wer`` (word error rate per window with one model designated the soft reference), ``a_text``, and ``b_text``; both transcripts are retained on the row so reviewers can audit the WER without re-deriving.
- **FR-003**: The comparison stage MUST also emit a top-level ``<run_dir>/disagreements.json`` index that lists the top-N "interesting" regions across all comparison parquets, where N defaults to 100, sorted by combined uncertainty when uncertainty is available and otherwise by mismatch density. The aggregator that combines per-model confidences into a single sortable score MUST be configurable via ``--uncertainty-aggregator``; allowed values are ``min`` (default), ``mean``, ``harmonic_mean``, and ``disagreement_weighted`` (``(1 - mean_confidence) * mismatch_severity``). Models lacking a native confidence signal MUST be dropped from the aggregator's input rather than treated as zero.
- **FR-004**: The comparison stage MUST extend ``labelstudio_tasks.json`` and ``labelstudio_config.xml`` with one Labels track per (comparison_kind, task, model_pair) actually used in the run. Track names follow ``<pass>__compare__<task>__<a>_vs_<b>`` so the model pair is encoded in the track *name*, not the label values. Label values themselves are drawn from a fixed enumerated set ``{"agree", "disagree", "incomparable", "one_sided"}``, identical across every track. ASR-vs-ASR pairs additionally declare a sibling ``<TextArea>`` track named ``<pass>__compare__<task>__<a>_vs_<b>__text`` carrying the WER plus both transcripts per disagreement region.
- **FR-005**: The comparison stage MUST hook into the existing content-addressable cache. The cache key MUST include ``(audio_signature, comparison_kind, task, model_set, params, wrapper_hash, senselab_version, schema_version)``. Re-running with the same inputs MUST return ``cache="hit"`` for every comparison without re-reading the upstream task results.
- **FR-006**: The comparison stage MUST be skippable via the existing ``--skip`` mechanism (``--skip comparisons`` skips everything new) and MUST be controllable per-axis via ``--skip-comparisons raw_vs_enhanced``, ``--skip-comparisons within_stream``, and ``--skip-comparisons cross_stream``. Per-region uncertainty plumbing (FR-007) and the ranked ``disagreements.json`` index (FR-003) are not separate axes: the index is opt-out via ``--disagreements-top-n 0`` and the per-region confidence columns are always emitted when the underlying signal is available (skipping the entire comparator stage with ``--skip comparisons`` is the single switch that turns everything off).
- **FR-007**: For every region in every existing per-task output (diarization, AST, YAMNet, ASR, alignment), the script MUST emit, where the underlying model exposes one, a ``confidence`` and an ``uncertainty`` column (``confidence`` = the model's native scalar in [0, 1] when available; ``uncertainty`` = entropy or 1−confidence when no entropy is defined). When multiple native confidence events fall inside one comparison-grid bucket (e.g., several Whisper word-level ``avg_logprob`` values within a 0.2 s cross-stream bucket), the bucket-level confidence MUST be the arithmetic mean of those events. For models with no native signal, the columns MUST be null with the omission documented in ``disagreements.json``.
- **FR-008**: ASR-vs-PPG phoneme comparison MUST run only when both an ASR result and a PPG result are present for the same audio variant, MUST project both to a shared phoneme inventory before differencing, and MUST report ``phoneme_per`` (continuous normalized phoneme error rate) on every row plus a boolean ``phoneme_disagreement`` set only when ``phoneme_per`` meets or exceeds the threshold ``--phoneme-disagreement-threshold`` (default ``0.50``). LS regions are produced only for rows where ``phoneme_disagreement`` is true.
- **FR-009**: Cross-stream comparisons MUST be projected onto a dedicated shared time grid before differencing, so models that emit at incompatible granularities are still comparable bucket-by-bucket. The grid is controlled by ``--cross-stream-win-length`` (default ``0.2`` seconds) and ``--cross-stream-hop-length`` (default ``0.1`` seconds), decoupled from the features grid so each can be tuned independently. The grid parameters MUST be recorded in the comparison parquet provenance.
- **FR-010**: The comparison stage MUST gracefully degrade — a single comparison failing (e.g., one model output is empty, the PPG backend is unavailable) MUST NOT abort the run; the affected rows MUST instead carry a ``comparison_status`` of ``"incomparable"`` or ``"unavailable"`` with a one-line reason in ``disagreements.json``.
- **FR-011**: All comparison and uncertainty outputs MUST be reproducible — the wrapper version hash and senselab version MUST be recorded in every parquet's provenance, and reviewers MUST be able to diff a comparison parquet from one run against another to track stability across runs.
- **FR-012**: The smoke-test suite MUST exercise the comparator framework on synthetic / cached upstream results so the comparison stage's logic can be tested without re-running heavy models. The existing skipif gates for venv- or weight-bound backends MUST continue to apply to the integration test that exercises the full stage.

### Key Entities

- **ComparisonRow**: One row per (start, end) bucket in a comparison parquet. Carries ``comparison_kind`` (raw_vs_enhanced / within_stream / cross_stream), ``task`` or ``stream_pair``, ``a_value``, ``b_value``, ``agree``, ``mismatch_type``, ``comparison_status``, optional ``confidence_a``, ``confidence_b``, ``combined_uncertainty``.
- **DisagreementsIndex**: The top-level JSON file enumerating the top-N flagged regions across all parquets, with each entry pointing to the source parquet path, row index, time span, and LS region id.
- **UncertaintyAnnotation**: Per-region scalars (``confidence``, ``uncertainty``, plus task-specific extras like ``no_speech_prob`` for Whisper, ``entropy`` and ``margin_to_top2`` for AST/YAMNet, ``avg_logprob`` for Whisper / Granite / Canary-Qwen) attached to every existing per-task region.
- **ComparisonGrid**: The shared time grid used to project incompatible-granularity outputs onto a common axis before differencing. Parameters mirror the existing ``--features-*-length`` flags.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a typical 1-minute conversational clip, the reviewer can open the Label Studio project, identify the top-5 most uncertain disagreements via the ``disagreements.json`` index, and locate each one on the audio timeline in under 30 seconds total.
- **SC-002**: 100 % of (task, model_pair) combinations *eligible for comparison* (i.e., a task with two or more successful models in the same pass, or any model present in both passes for raw-vs-enhanced) produce a parquet sidecar. Single-model tasks gracefully produce no within-stream parquet (no pair to compare) — that is correct behavior, not a silent drop.
- **SC-003**: Cache replay on identical inputs hits the comparator cache for ≥95 % of comparison rows; the remaining ≤5 % are documented in ``disagreements.json`` as deliberately uncached (e.g., one-sided rows that depend on which pass succeeded).
- **SC-004**: The new stage adds no more than 30 % wall-clock overhead on top of the existing analyze_audio run on a 1-minute clip when running on already-cached upstream task results.
- **SC-005**: Existing analyze_audio outputs (per-task JSON, features parquets, existing LS tracks) are unchanged in shape — adding the comparison stage MUST NOT break any consumer of the existing artefacts. A reviewer who skips the new stage with ``--skip comparisons`` gets bit-for-bit the same outputs the script produces today.
- **SC-006**: Smoke tests cover at least one happy-path scenario for each comparison kind (raw_vs_enhanced, within_stream, cross_stream) plus the four documented degradation modes (failed pass, one_sided, unavailable PPG, missing confidence signal); the tests run in under 30 seconds total without venv provisioning or model downloads.
- **SC-007**: The reviewer can answer the question "did enhancement help here?" for a clip in under one minute by visiting the LS project and reading the ``raw_vs_enhanced`` track.

## Assumptions

- The PPG-based phoneme sequence is obtained via senselab's existing PPG backend (``ppgs`` extra). The user's earlier work already produced ``artifacts/sample1_ppg_segments.json`` and a phoneme timeline visualization, so the dependency is available.
- The "shared phoneme inventory" for ASR↔PPG comparison defaults to the inventory used by the PPG backend; the ASR side is mapped onto it via grapheme-to-phoneme on the transcript (English-only on first pass; ja/zh covered by the existing uroman pipeline added in the prior PR).
- "Speech-presence" mapping for AST/YAMNet uses a fixed AudioSet label allowlist (e.g., ``Speech``, ``Conversation``, ``Narration, monologue``, ``Female speech, woman speaking``, ``Male speech, man speaking``, ``Child speech, kid speaking``); the allowlist is documented and the user can override it via a CLI flag if needed.
- The comparison stage runs on top of *cached* upstream results whenever possible; the wrapper version hash invalidation rules from the prior PR continue to apply.
- The Label Studio bundle is opened in a workspace that already accepts the script's existing config XML; no LS-side config migration is required beyond appending the new tracks declared in the per-run XML.
- Uncertainty signals are read from the *first available* native source per backend; a follow-up may add ensemble / Monte-Carlo Dropout style uncertainty, but those are explicitly out of scope for v1.
- The disagreements index ranking uses a single combined uncertainty score per region. The default aggregator is ``min`` of the available per-model confidences (most-doubtful-model wins); the user can switch to ``mean``, ``harmonic_mean``, or ``disagreement_weighted`` via ``--uncertainty-aggregator``. Models lacking a native confidence signal are dropped from the input rather than treated as zero.
- Backwards compatibility: the spec MUST NOT change any existing per-task output shape or filename. New outputs are strictly additive under ``<run_dir>/<pass>/comparisons/`` and ``<run_dir>/disagreements.json``.
