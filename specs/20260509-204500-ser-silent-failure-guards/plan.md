# Plan: Eliminate silent-failure modes in `classify_emotions_from_speech`

**Date**: 2026-05-09
**Branch**: `claude/ser-loading-guards` → PR target `fix/ser-wav2vec2-regression-head`

## Problem

PR #511 fixes one specific silent-failure mode (random-initialized head producing
~uniform softmax) for two known checkpoints (audeering MSP-Dim, ehcalabres
RAVDESS). The dispatcher relies on hand-rolled heuristics — encoder family,
architecture string, head topology, final-layer attribute name, label vocabulary —
each of which can fail silently on any model the maintainers haven't manually
catalogued. A user passing a new SER checkpoint can today get plausible-looking
but meaningless scores back without any warning.

This plan turns those silent failures into either loud errors or correct loads.

## Failure-mode audit

| # | Assumption | Failure today | Severity |
|---|------------|----------------|----------|
| 1 | Encoder family is Wav2Vec2 | Routes HuBERT/WavLM/Whisper SER models to standard pipeline; if those have the same broken-head pathology, output is silently random | High |
| 2 | Custom head is exactly `dense → tanh → dropout → final` | A 3-layer head, GELU instead of tanh, or extra LayerNorm: missing-key on partial loaded layers, silent random for the rest | Medium |
| 3 | Final-layer attribute is `out_proj` or `output` | A new checkpoint using `head`, `fc`, `linear`, etc. is not detected by the peek; routes to standard pipeline; silent random | Medium |
| 4 | Single-bin checkpoints are catalogued in `_KNOWN_WAV2VEC2_EMOTION_HEADS` | Anything not pre-registered is invisible to the peek (we refuse to download a multi-hundred-MB blob to inspect); silent random | Medium |
| 5 | Discrete-vs-continuous decided from English label keywords | A continuous model with non-AVD axes labelled `["energy","pleasantness"]` is misclassified; softmax applied to raw logits; silently wrong scores | Medium |
| 6 | Standard `HuggingFaceAudioClassifier.classify_audios_with_transformers` resamples internally | It does not — passes user's sampling rate through. Off-rate audio = OOD inference, silently wrong | Medium |
| 7 | `auto_map` checkpoints are deferred to their own `trust_remote_code` | If their custom code itself has a head bug, we can't help; correct to defer | Low (not our problem) |
| 8 | AutoConfig parse failure is the only signal of a broken config | A config that AutoConfig accepts but transformers later chokes on bypasses the fallback | Low |
| 9 | Single classifier head per checkpoint | Multi-task heads (e.g. arousal regression + categorical emotion in one checkpoint) — `num_labels`/`id2label` look correct but only one path gets exercised; silent partial-result | Medium |
| 10 | One head is shared across languages | Conditional / language-keyed heads in multilingual SER checkpoints route through one fixed head; silent wrong-language head | Low (rare) |
| 11 | Default Wav2Vec2 feature extraction matches the checkpoint | A `preprocessor_config.json` that overrides `do_normalize`, `return_attention_mask`, or applies pre-emphasis / CMVN is silently ignored — we instantiate `Wav2Vec2FeatureExtractor.from_pretrained` and trust its defaults | Medium |
| 12 | Loaded attention-mask plumbing matches the head | We currently pass only `input_values` to the forward; HuBERT/WavLM heads (Phase 3) often want `attention_mask` for variable-length batches. Single-clip inference today is fine; batching after Phase 3 may not be | Low (Phase-3-only) |
| 13 | Number of head outputs equals declared num_labels | A checkpoint whose head ships an out_features ≠ num_labels (e.g. config patched after upload) loads "successfully" but produces wrong-length scores; today only the defensive `result_labels` reconciliation papers over it without warning | Medium (now caught by Phase 1) |

## Phased approach

Each phase is independently shippable, tested, and reverts cleanly.
Order is roughly: cheapest+highest-coverage first, refactors last.

### Phase 1 — Loud-fail post-load assertion *(this PR)*

**Goal**: Convert assumption #2/#3/#4/#13 silent-random failures inside
`_classify_wav2vec2_speech_cls_ser` into a `RuntimeError` that names the
missing or mismatched checkpoint keys.

**Tasks**

- [x] T101 Pass `output_loading_info=True` to `EmotionModel.from_pretrained`
  and raise on any `classifier.*` key in either `missing_keys` (random-init)
  **or** `mismatched_keys` (wrong-shape weights). Both are silent-random
  failure modes; the architect review pointed out that `missing_keys` alone
  misses the second class.
- [x] T102 Add a post-load shape-sanity check: assert
  `loaded.classifier.<final_layer>.out_features == num_labels` (or
  `len(id2label)` when the field is absent). Catches assumption #13: a head
  whose declared and actual output sizes disagree.
- [x] T103 Existing `test_speech_emotion_recognition_continuous` passes
  unchanged (regression guard for the audeering happy path).
- [ ] T104 Add a parametrised unit test that mocks `EmotionModel.from_pretrained`
  to return loading_info with (a) `missing_keys={"classifier.dense.weight"}`,
  (b) `mismatched_keys={("classifier.out_proj.weight", ...)}`, and (c) a final
  layer whose `out_features` disagrees with `config.num_labels`. All three
  should raise `RuntimeError` with a useful message.

**Cost**: ~40 lines (the missing/mismatched/shape triad) + 1 parametrised
unit test. No new dependencies. No effect on the happy path.

**Coverage**: Catches **most** of the silent-random failure modes in
#2/#3/#4/#13 because the symptom — missing weights, mismatched shapes, or
mis-sized final layer — is the same regardless of why the layout was wrong.

**Doesn't catch**: Assumptions #1 (wrong encoder family), #5/#6 (wrong
softmax / wrong sampling rate), #9 (multi-task heads), and #11 (custom
preprocessor config). Those produce wrong-looking-but-not-loud-missing
outputs, so `loading_info` doesn't help.

**Explicitly rejected:** A "sentinel-input forward, assert output isn't
~uniform" check. Architect review flagged this as brittle — regression heads
legitimately produce near-uniform outputs in their bounded range, and a
single sample is not a reliable signal. Missing-keys + mismatched-keys +
shape sanity covers the actual failure mode that motivated this PR.

### Phase 2 — Standard-pipeline missing-key warning

**Goal**: Detect random-init in the *standard* HF pipeline path, not just our
custom one. The original PR-#511 bug (audeering before this work) lived here.

**Tasks**

- [ ] T201 In `HuggingFaceAudioClassifier.classify_audios_with_transformers`,
  preload the model with `output_loading_info=True` once per (model_id,
  revision, device), inspect `missing_keys` for `classifier.*` / `head.*` /
  `score.*` entries, and `logger.warning` (don't raise — backward-compat).
- [ ] T202 Add a `senselab` config flag `SENSELAB_STRICT_HEAD_LOAD=1` that
  promotes that warning to a hard error. Default off.
- [ ] T203 Test: mock loading_info; assert the warning fires for missing head
  keys and not for unrelated misses.

**Cost**: ~40 lines. Requires switching `pipeline()` to load the model
explicitly first; pipeline can then accept the pre-loaded model. Existing call
sites in this codebase: 1 (`huggingface.py`). Manageable.

**Risk**: Loading the model twice (once for inspection, once via pipeline)
unless we cache and pass the loaded instance into `pipeline(..., model=...)`.
Need to confirm transformers' `pipeline` accepts a `PreTrainedModel` instance
in v5.x — it does (verified earlier in PR #513 work).

### Phase 3 — Encoder-family generalization

**Goal**: Handle assumption #1. Today only Wav2Vec2 architectures get the
custom-head treatment; HuBERT/WavLM/wav2vec2-bert SER checkpoints with the
same pattern are silently random-initialized in the standard pipeline.

**Approach**: Introspect the loaded base-model class from the config rather
than hardcoding architecture strings. The custom head class (
`_Wav2Vec2EmotionModel`) is an inner attribute `self.<base>` plus
`self.classifier`; this generalizes to any `XxxPreTrainedModel`.

**Load-bearing detail (architect-review escalation):**
`Wav2Vec2/Hubert/WavLM ForSequenceClassification` each store the encoder
under a *different* attribute name (`self.wav2vec2`, `self.hubert`,
`self.wavlm`) — see `transformers/models/{wav2vec2,hubert,wavlm}/modeling_*.py`.
The checkpoint keys mirror that name (`wav2vec2.encoder.layers...` vs
`hubert.encoder.layers...`). The class factory must therefore know the
canonical attribute name *for each base*, not just the base class itself.
Map `config.model_type` → `(base_cls, attr_name)` in a small registry.

**Tasks**

- [ ] T301 Build a `_BASE_REGISTRY: dict[str, tuple[type, str]]` mapping
  `model_type` → `(base_pretrained_cls, encoder_attr_name)`. Seed with
  `wav2vec2`, `hubert`, `wavlm`, `wav2vec2-bert`. Test that constructing a
  class with the right attr name binds the checkpoint keys correctly.
- [ ] T302 Refactor `_make_wav2vec2_emotion_model_class` into
  `_make_emotion_model_class(model_type)` that pulls `(base_cls, attr_name)`
  from the registry and uses `setattr(self, attr_name, base_cls(config))`.
  Keep the cache keyed by `(model_type, final_layer)`.
- [ ] T303 Extend `_wav2vec2_emotion_head_kind` (rename to
  `_emotion_head_kind`) to consult the registry for any
  `For{SpeechClassification,SequenceClassification}` architecture string.
- [ ] T304 Manually verify against `superb/hubert-large-superb-er` (HuBERT,
  flat head — should NOT be routed) and any known WavLM/wav2vec2-bert
  emotion checkpoint. The Phase-1 guard will catch regressions: if the
  attr-name registry is wrong for a given model_type, every checkpoint key
  will appear missing and the `RuntimeError` from Phase 1 fires.
- [ ] T305 Forward-pass attention-mask shim: HuBERT/WavLM forwards may want
  `attention_mask` for variable-length batches (assumption #12). Today's
  single-clip inference is fine; batching past one item must thread the
  attention mask through. Defer until batching is exercised.

**Cost**: ~120 lines once you account for T301 and T305. Test infra: at
least one HuBERT-based test marked GPU-only (these models are larger).

**Risk**: Each new base class has slightly different forward signatures
(some return `BaseModelOutput`, some return tuples). The custom forward
method may need per-base-class shims; encode that into `_BASE_REGISTRY` if
needed (third tuple slot for a forward adapter callable).

### Phase 4 — Richer head registry *(scaled back from architect review)*

**Goal**: Address assumption #2 *without* over-engineering. Architect review
flagged the original `HeadSpec` dynamic-builder as premature: with two
known head shapes today and the unavoidable need for a per-checkpoint
activation registry anyway, dynamic ModuleDict construction adds risk
without reducing the registry footprint.

**Revised approach**: Promote `_KNOWN_WAV2VEC2_EMOTION_HEADS` from a
`dict[str, str]` (final-layer name only) to a `dict[str, HeadEntry]` with
fields `{final_layer, activation, dropout_field, num_dense_layers}`. The
existing class factory stays; it just consults a richer registry. New
checkpoint families need an entry, not a code change.

**Tasks**

- [ ] T401 Define `HeadEntry` dataclass and migrate the existing two
  registry entries (audeering implicit + ehcalabres) into it with their
  observed shapes.
- [ ] T402 Generalize `_make_wav2vec2_regression_head_class` to accept a
  `HeadEntry` and build the matching small head. Keep `tanh` as default
  activation (the observed common case); warn loudly if activation is
  inferred from defaults rather than a registry entry.
- [ ] T403 Tests: numerical regression on both known checkpoints (audeering,
  ehcalabres); the recorded scores must not move.

**Cost**: ~50 lines. Substantially smaller than the original Phase 4.

**Risk**: Future weird heads (3 layers, layer-norm in the middle) still
need a code change. Acceptable: when that third example appears, revisit
whether dynamic construction earns its complexity.

**Explicitly cut from the original Phase 4**: dynamic `nn.ModuleDict`
builder, `_introspect_head_topology` shard-key reader (the Phase-1 guard
already turns "wrong topology" into a loud failure, so introspection earns
us nothing on the failure side; on the success side it would just
re-implement the `from_pretrained` loader less reliably).

### Phase 5 — Discrete-vs-continuous from output shape, not labels

**Goal**: Address assumption #5. Today softmax is applied to anything
`_get_ser_type` doesn't classify as CONTINUOUS, including continuous models
with non-AVD axes.

**Approach**: Decide `apply_softmax` from `config.problem_type` if set
(`single_label_classification` / `multi_label_classification` / `regression`),
falling back to label-name heuristics only if absent. HF docs require this
field on properly-published models.

**Tasks**

- [ ] T501 Read `config.problem_type` first; only consult `_get_ser_type` for
  legacy checkpoints with no problem_type set.
- [ ] T502 Test: a checkpoint declaring `problem_type="regression"` whose
  labels happen to include "happy" must NOT get softmax applied.

**Cost**: ~10 lines + 1 test.

**Risk**: Older checkpoints (including the two we know) lack `problem_type`.
Backward-compat fallback to `_get_ser_type` covers them.

### Phase 6 — Resampling in the standard pipeline path

**Goal**: Address assumption #6. The original gemini-bot review on the
`Wav2Vec2Processor` line raised this; the standard pipeline silently passes
the caller's sampling rate through.

**Tasks**

- [ ] T601 In `HuggingFaceAudioClassifier.classify_audios_with_transformers`,
  read the model's `feature_extractor.sampling_rate` and resample mismatched
  audio before calling the pipeline. Mirror the convention already used in
  `_classify_wav2vec2_speech_cls_ser`.
- [ ] T602 Test: mock pipeline; pass 44.1 kHz audio; assert the audio is
  resampled to 16 kHz before reaching the pipeline.

**Cost**: ~15 lines + 1 test.

**Risk**: Touches code outside the SER module — affects all audio
classification, not just SER. Will need a separate code-review pass.

### Phase 7 — Documentation + diagnostics

**Goal**: Make the assumptions visible to users who add new models.

**Tasks**

- [ ] T701 Update
  `src/senselab/audio/tasks/classification/speech_emotion_recognition/doc.md`
  to enumerate the supported checkpoint families and the known assumptions.
- [ ] T702 Add a one-shot CLI helper
  `python -m senselab.audio.tasks.classification.speech_emotion_recognition
  --probe <model_id>` that prints the dispatcher's classification of the
  model and warns about anything risky. Useful for users adding new models.

## Out-of-scope (this PR)

- Refactoring the SpeechBrain SER backend — separate concern.
- Adding new SER model families to the registry — should be a follow-up
  per-model PR, not bundled here.
- Replacing the hand-rolled label vocab with an embedding-based classifier —
  way out of scope; flagged for future discussion.

## Acceptance criteria for this PR

1. `_classify_wav2vec2_speech_cls_ser` raises `RuntimeError` with a
   diagnostic message when `loading_info["missing_keys"]` contains any
   `classifier.*` entries.
2. Existing `test_speech_emotion_recognition_continuous` passes (audeering
   happy path).
3. Existing `test_speech_emotion_recognition_discrete` passes (ehcalabres
   happy path; gpu-only).
4. New unit test exercises the failure path with a mocked
   `from_pretrained` return.
5. The plan doc above is committed alongside, capturing future phases for
   discoverability.

## Open questions for review

- Should Phase 2 (standard-pipeline warning) be in this PR or split off?
  It touches code outside SER and may require coordination.
- Phase 4's dynamic-topology head is the largest and riskiest. Worth
  delaying until we have a third or fourth example of a non-conforming
  head, to avoid over-fitting to current cases.
- Should `_KNOWN_WAV2VEC2_EMOTION_HEADS` move to a YAML registry users can
  extend without a code change? It's growing in scope as a fixture.
