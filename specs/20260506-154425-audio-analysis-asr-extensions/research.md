# Research: Audio Analysis Script + ASR Backend Extensions + Forced Alignment

**Branch**: `20260506-154425-audio-analysis-asr-extensions` | **Date**: 2026-05-06

## Decision: Script structure — single Python file under `scripts/`

Same as before. The script is a developer/research tool, not a packaged senselab API surface. ~1,000-LOC self-contained file.

## Decision: Cache key composition (refined for separable ASR + alignment)

ASR-step cache key:
```
sha256(audio_signature || "asr" || asr_model_id || asr_params || wrapper_hash || senselab_ver || schema_ver)
```

Alignment-step cache key (depends on the ASR's output text but not on ASR's params):
```
sha256(audio_signature || "alignment" || aligner_model_id || aligner_params || transcript_sha || language || wrapper_hash || senselab_ver || schema_ver)
```

`transcript_sha = sha256(asr_text)` is the binding between the two steps. The alignment cache entry records the parent ASR's cache_key in its provenance so a reader can trace which ASR run an alignment came from.

**Rationale**: with two separate cache keys, the user can re-run alignment after fixing the aligner (or after a senselab upgrade) without re-running the slow ASR. They can also force a fresh alignment on a stable ASR text by passing `--no-cache` — but the ASR cache stays warm.

**Alternatives considered**: nesting alignment as a sub-field of the ASR cache entry. Rejected because re-running just alignment would require re-running ASR or surgically editing cache entries; the two-entry model is simpler.

## Decision: Multilingual forced-aligner backend — Meta MMS via HuggingFace (`facebook/mms-1b-all` with per-language adapters)

**Rationale**: subagent research confirmed:
- Single model object covers 1100+ languages (full superset of Granite's 8 languages: en/fr/de/es/pt/ja/zh, plus optionally it).
- Loads through senselab's existing `transformers` dependency — no new top-level package needed in the main venv.
- Output shape compatible: reuses senselab's existing CTC trellis/backtrack code (`_get_trellis`, `_backtrack`, `_merge_repeats`, `_assign_timestamps` in `forced_alignment.py`); the only senselab-side addition is a new `model_type = "mms"` branch.
- Per-language adapter selection: `processor.tokenizer.set_target_lang(iso3)` + `model.load_adapter(iso3)` after `from_pretrained`. Cache the model once per (model_id, iso3) pair.

**License caveat**: MMS weights are CC-BY-NC 4.0. For research and senselab default use this is acceptable. Expose the aligner model id as configurable so commercial users can swap to a permissively-licensed alternative.

**Optional dep**: `uroman` (MIT, pure-Python) is needed to romanize ja/zh transcripts before MMS's character-set lookup, since MMS uses Roman characters internally. Add to the existing `nlp` extra (it's a small NLP-adjacent dep).

**Alternatives considered**:
- **torchaudio MMS_FA bundle**: deprecated in 2.8 / removed in 2.9; senselab pins `torchaudio>=2.8`, so this is brittle for the long term.
- **WhisperX-style per-language wav2vec2**: existing senselab path; covers ~30 languages; doesn't cover ja/zh that Granite produces. Keep as the English-default for backwards compat.
- **CTC segmentation (Kürzinger)**: model-agnostic algorithm, but requires per-language CTC posteriors; senselab would need to provide a separate per-language model anyway, so MMS already wins on coverage.
- **Aeneas**: AGPL-3.0 (license incompatibility), unmaintained since 2017. Reject.
- **Qwen3-ForcedAligner-0.6B**: keep as an internal pairing for the `qwen.py` ASR backend (Alibaba ships them together); not promoted to the unified senselab aligner because its language coverage is narrower than MMS.

## Decision: Aligner dispatch — language-keyed registry, no per-call backend selection

The `align_transcriptions(audio, transcript, language=...)` API stays unchanged. Internally a registry maps language ISO code → aligner backend (model_type + model_id). MMS becomes the default for any language not already in the WhisperX-style English-only dictionary. Users can override per-language by patching the registry, but the common case requires zero configuration.

**Rationale**: the spec's FR-020 requires "the module MUST internally pick a compatible aligner backend based on the language" — registry lookup is the simplest mechanism; no new dispatch concept beyond the existing `model_type` field.

## Decision: Granite Speech 3.3 path — reuse existing HF pipeline with `return_timestamps=False`, then post-align

senselab's current ASR HF path forces `return_timestamps="word"` (or similar), which Granite refuses. Instead of writing a new Granite-specific backend, we expose a `return_timestamps` parameter on the senselab transcribe call. When false (or when the model is a known timestamp-less type), the HF pipeline runs without timestamps and the analysis script's auto-align step adds timestamps via MMS.

**Rationale**: Granite is fully supported by the existing HF `automatic-speech-recognition` pipeline as long as we don't request native timestamps. Avoiding a Granite-specific subprocess venv keeps scope tight.

**Concrete patch**: `senselab.audio.tasks.speech_to_text.huggingface.py` accepts `return_timestamps: bool = True`; when False the call to `pipeline(...)` omits the timestamp keyword. The senselab dispatcher detects timestamp-less models (a small known-list, expandable) and defaults to False for them.

**Alternatives considered**:
- New Granite-specific backend module: gratuitous; the HF pipeline handles it once we relax the timestamp requirement.
- Force users to pass an explicit flag: bad UX; library should know which models can't produce timestamps.

## Decision: NVIDIA Canary-Qwen 2.5B — new subprocess venv `nemo-canary-qwen`

Same as previous plan. Loaded via `SALM.from_pretrained(...)` from `nemo.collections.speechlm2.models`, requires NeMo trunk and `[asr,tts]` extras, isolated from the existing `nemo-diarization` venv. Returns text-only ScriptLines; the script's auto-align stage adds timestamps via MMS.

## Decision: Qwen3-ASR 1.7B — new subprocess venv `qwen-asr` using Alibaba's `qwen-asr` package

Same as previous plan. The `qwen-asr` package internally bundles its companion `Qwen3-ForcedAligner-0.6B` model for word-level timestamps when invoked with `return_time_stamps=True, forced_aligner=...`. Treat this as the ASR backend's INTERNAL alignment (returns timestamped ScriptLines directly); the script's outer auto-align stage SKIPS Qwen3-ASR output because it already has timestamps. This is the right behavior because Qwen-specific alignment is more accurate for Qwen-specific text segmentation than a generic aligner.

**Concrete behavior**: the auto-align stage checks each ASR ScriptLine for the presence of `start`/`end` (or non-empty `chunks`); if present, it skips alignment for that one. This makes the auto-align stage a no-op for any ASR that natively produces timestamps (Whisper, NeMo Conformer CTC, NeMo Sortformer-related ASRs, Qwen3-ASR with the companion aligner enabled).

## Decision: Cache schema version bump — `_CACHE_SCHEMA_VERSION = 2`

The previous plan introduced `_CACHE_SCHEMA_VERSION = 1`. Adding the alignment-step cache entries and the `cache: hit/miss` field on a per-step basis is a cache-shape change, so bump to 2. Old cache entries become inert and won't be served (intentional). Document the bump in the script's CHANGELOG-style comment.

## Decision: macOS ARM64 + dependency footprint

- MMS via HF: zero new top-level deps. `uroman` becomes part of the `nlp` extra.
- Canary-Qwen: NeMo trunk in its own subprocess venv. Macos compatibility: NeMo's macOS support is uneven; expect CPU-only fallback.
- Qwen3-ASR: `qwen-asr` package in its own subprocess venv. Best on CUDA; CPU fallback should work.
- Granite: senselab main venv (already supports HF pipeline). MPS support depends on transformers version; typical Apple-Silicon fallback is CPU.

No conflict with existing pyannote/speechbrain pins.

## Decision: Test strategy — additive only

- New tests for the script and the new senselab backends use `@pytest.mark.skipif(...)` guards so default CI doesn't pull the new venvs.
- Existing alignment tests stay untouched; new MMS-specific tests skipif when MMS isn't yet downloaded.
- Smoke test for the script's new `--no-align-asr` toggle and the separable cache (verifies ASR cache hits while alignment cache misses, and vice versa).

## Open questions deferred to implementation

- **Exact NeMo trunk pin** for `nemo-canary-qwen` venv (depends on NeMo's release cadence).
- **Whether `mms-1b-all` adapter loading is picklable across senselab's process boundary** — affects whether to cache the loaded model in-memory or reload per call. Likely fine but unverified.
- **Cache eviction policy** when the cache directory grows large. Not a blocker for the first iteration; can add `--cache-max-size` or LRU later.
- **Romanization fidelity for ja/zh** — uroman is good for kanji-heavy text but romaji-vs-pinyin choices may vary. Confirm with a small set of Granite Speech 3.3 ja/zh samples at integration time.
