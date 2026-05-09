# Phase 0 Research — Comparison & Uncertainty Stage

## 1. Token-timestamp inventory across ASR backends

**Decision**: Use the post-auto-align ScriptLine tree as the single source of truth. After the auto-align stage from PR #510, every successful ASR result carries either (a) native chunks with `start`/`end` per chunk, or (b) MMS-aligned segments with `start`/`end` per segment.

**Per-backend status as merged in PR #510**:

| Backend | Native granularity | After auto-align | Token-overlap-with-window check |
|---|---|---|---|
| openai/whisper-large-v3-turbo | word-level chunks (`return_timestamps="word"`) | unchanged (already has timestamps) | Iterate chunks; any `chunk.start < win_end and chunk.end > win_start` → speech |
| ibm-granite/granite-speech-3.3-8b | text-only ScriptLine, no chunks | MMS-aligned to per-segment chunks | Iterate aligned chunks |
| nvidia/canary-qwen-2.5b | text-only ScriptLine, no chunks | MMS-aligned to per-segment chunks | Iterate aligned chunks |
| Qwen/Qwen3-ASR-1.7B | word-level chunks via Qwen3-ForcedAligner-0.6B companion | unchanged | Iterate chunks |

**Rationale**: All four backends converge to the same shape post-auto-align. The comparator does not need backend-specific code — just iterate `result.chunks or []` and test interval overlap. When alignment failed (recorded as `comparison_status="incomparable"` per FR-010), the row reports that status rather than a false negative.

**Alternatives considered**: (a) Use the top-level ScriptLine `start`/`end` only — rejected because Granite/Canary-Qwen would only contribute one giant span. (b) Re-run a per-backend tokenizer — rejected because it duplicates auto-align's work.

## 2. Confidence signals per backend

**Decision**:

| Backend | Native scalar in [0, 1] | Wired in v1? | Source |
|---|---|---|---|
| Whisper (HF pipeline) | `1 - exp(-avg_logprob)` per chunk; `1 - no_speech_prob` per chunk | Yes | `pipe(audio, return_timestamps="word")` returns per-chunk `avg_logprob` and `no_speech_prob` when `return_token_timestamps` is enabled — confirm via probe and add the kwarg if not already on |
| pyannote diarization | None per-segment (post-processed) | No (null) | Documented as missing in `disagreements.json`; future work could pull `segmentation` raw scores |
| Sortformer (NeMo subprocess) | None per-segment exposed | No (null) | Same as above |
| AST (HF pipeline classify_audios) | top-1 score in [0, 1]; full distribution available | Yes — emit `top1_score`, `entropy`, `margin_to_top2` | Already in the per-window dict from `classify_audios` |
| YAMNet (subprocess venv) | top-1 score in [0, 1] | Yes — same shape as AST | Already in the per-window dict |
| Granite | None native per-segment | No (null in v1) | Could compute per-token mean log-prob from a future `output_scores=True`; deferred to keep the in-process backend simple |
| Canary-Qwen | None native | No (null in v1) | Same — would require subprocess worker change |
| Qwen3-ASR | None natively exposed by the wrapper | No (null in v1) | The wrapper hides token logits |
| MMS auto-aligner | Per-segment trellis log-probability | Yes | Already produced internally; needs surfacing on the alignment ScriptLine output |

**Rationale**: Wire what's cheap and already in the existing dicts (Whisper avg_logprob, AST/YAMNet top1+entropy, MMS trellis score) for v1; document Granite/Canary-Qwen/Qwen3-ASR/pyannote as null. Constitution VII (Simplicity First) and the spec's FR-007 ("for models with no native signal, the columns MUST be null with the omission documented") explicitly support partial coverage.

**Alternatives considered**: Force confidence on every backend via Monte-Carlo dropout / ensemble — rejected as out-of-scope by the spec's last assumption.

## 3. G2P library choice

**Decision**: `g2p_en` (English-only, ~1 MB, pure-Python with bundled CMUdict) for v1. Add to `[nlp]` extra alongside `uroman`.

**Rationale**:

- `g2p_en` produces ARPAbet phonemes that match the senselab PPG backend's inventory (the existing PPG output uses ARPAbet labels — visible in `artifacts/sample1_ppg_segments.json`).
- No system dependency (espeak required by `phonemizer` is a non-starter on macOS-arm64 + `uv`-managed environments without homebrew).
- ~1 MB install footprint. Trivially fits in the existing `[nlp]` extra.
- English-only matches the dominant use case for the senselab pipeline as it stands.

**Multi-language handling**: For non-English transcripts, the comparator records `phoneme_status="g2p_unsupported_language"` and skips ASR↔PPG comparison for that row. PPG itself is English-only at present, so this matches the upstream constraint.

**Alternatives considered**: `phonemizer` (multi-language but needs espeak system dep — rejected); `epitran` (multi-language, no system dep, but uses IPA — would need IPA→ARPAbet mapping table — rejected for v1, kept as a follow-up if non-English ASR↔PPG becomes common).

## 4. WER computation per window

**Decision**: Use `jiwer.wer(reference, hypothesis)` from the `[nlp]` extra. Per-window inputs are the concatenated tokens whose timestamp ranges overlap each cross-stream-grid bucket. The "reference" model defaults to Whisper (most established) and is overridable via `--asr-reference-model`. Empty bucket on both sides → WER = 0; empty on one side → WER = 1.

**Rationale**: `jiwer` is mature, well-tested, already in the `[nlp]` extra (>=3.0), and produces the standard WER metric reviewers expect. Picking Whisper as default reference matches industry practice.

**Edge cases documented in the contract**:

- Bucket entirely silent on both sides: skip — no row emitted (silence is the signal).
- Bucket has reference text but hypothesis is empty: WER = 1.0, mismatch_type = "deletion".
- Bucket has hypothesis but reference is empty: WER = 1.0, mismatch_type = "insertion".
- Both sides have text and they differ: WER ∈ (0, ∞] (jiwer can exceed 1 for many insertions); cap at 1.0 in the parquet for sortability.

## 5. AST/YAMNet speech-presence label allowlist

**Decision**: Default allowlist is the AudioSet "Human voice → Speech" subtree:

```text
{
  "Speech",
  "Conversation",
  "Narration, monologue",
  "Female speech, woman speaking",
  "Male speech, man speaking",
  "Child speech, kid speaking",
  "Speech synthesizer",
}
```

User can override via `--speech-presence-labels label1,label2,...`. The list is recorded in the comparison parquet provenance so historical runs are audit-able even if defaults change.

**Rationale**: These seven labels are the direct children of "Speech" in the AudioSet ontology and match the user expectation of "anything resembling a person speaking". Singing and laughter are deliberately excluded from the default — they are speech-adjacent but a human reviewer would normally not call them "speech in this region".

**Alternatives considered**: A broader set including "Singing", "Laughter", "Crying" — rejected because it would inflate cross-stream agreement with diarization (which only finds speech turns, not vocal events) and produce more false-positive disagreements.

## 6. Cache-key composition for the comparator

**Decision**: Per FR-005, the comparator cache key is

```text
sha256(
    audio_signature,
    comparison_kind,        # raw_vs_enhanced | within_stream | cross_stream
    task_or_pair,           # e.g. "asr" or "asr/whisper-large-v3-turbo:Qwen3-ASR-1.7B"
    upstream_cache_keys,    # sorted tuple of the upstream task cache_keys feeding this comparison
    params,                 # comparator-specific params (grid, threshold, aggregator, allowlist)
    wrapper_hash,           # script SHA — already used by upstream tasks
    senselab_version,
    schema_version=1,       # bump when we change the parquet shape
)
```

**Rationale**: Including `upstream_cache_keys` in the key means re-running an upstream task automatically invalidates downstream comparisons that depended on it. This satisfies SC-003 (cache replay ≥95 %) without manual cache-busting.

**Alternatives considered**: (a) Time-based invalidation — rejected, doesn't compose with the existing content-addressable design. (b) Compute cache key on the comparison output rather than input — rejected, requires running the comparison to know if it would have hit.

## 7. LS Labels enumeration strategy

**Decision**: One `<Labels>` block per (pass, comparison_kind, task) actually used in the run, with the *fixed* enumerated value set `{"agree", "disagree", "incomparable", "one_sided"}`. The model pair is encoded in the track *name* (e.g. `raw_16k__compare__asr__whisper_vs_qwen`), not in the label values. For ASR-vs-ASR specifically, an additional `<TextArea>` track per pair carries the WER score and both transcripts so reviewers can read the actual disagreement text.

**Rationale**: A fixed label set keeps the LS XML stable and small regardless of how many models the user adds. Per-pair information lives in the track name, which is already how the existing diarization / ASR tracks are namespaced (`<pass>__<task>__<model>`).

**Alternatives considered**: Per-pair label values like `disagree-whisper-qwen` — rejected because it would require regenerating the XML config every time the model set changes and combinatorially explode for runs with many models.

---

**All NEEDS CLARIFICATION items resolved**: yes (none in the plan; the five spec-level clarifications were captured in `spec.md ## Clarifications` and translated into FR-002/FR-003/FR-008/FR-009 rewrites + the new CLI flags above).
