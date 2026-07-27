# Architecture review — where should the PR's code live?

> **Implementation status (2026-07-24)**: F1→T046, F2→T047, F3→T048, F4→T049 (minus the DSP-SNR
> move, blocked on splitting `quality_control/metrics.py`), F5→T050 are implemented in this PR and
> verified (15/15 unit + hermetic determinism e2e green; run15 reproduced all ground-truth metrics
> exactly). F6→T051 and F7→T052 remain sequenced follow-up PRs. Two policy knobs were added during
> implementation so degraded paths are explicit rather than raced: `u1_backend` and
> `audio_io_backend` (loader/backend recorded in provenance — fallbacks are never silent).

**Date**: 2026-07-24 · **Scope**: everything this branch adds/touches (`votes.py`, `compute.py`
split, `adaptive/` subpackage, `scripts/analyze_audio.py` triage + stages, `scripts/adaptive_loop.py`,
tests) reviewed against senselab's documented architecture: `tasks/<capability>/{__init__, api.py,
<backend>.py}` with model-type dispatch; Pydantic for cross-boundary `data_structures/`, dataclasses
for task/workflow internals (documented in `workflows/audio_analysis/types.py:1-4`); heavy imports
behind guarded/lazy patterns (`frame_posteriors.py:33-38` precedent).

**Verdict in one paragraph.** The workflow-shaped code (loop, policy engine, belief store, regions,
convergence, triage decisions, LS export, plotting) is in the right place —
`workflows/audio_analysis/` is exactly senselab's home for composite, opinionated pipelines. What
does NOT belong where it currently sits is (a) **model capabilities re-implemented inside
`adaptive/backends.py` and `adaptive/audio_io.py`** that duplicate existing `tasks/` capabilities —
these exist only to survive the heavy-import problem, whose root cause is two eager parent
`__init__`s; (b) **generic utilities grown inside the workflow** (ScriptLine leaf-walk, word-level
WER, DSP SNR, transcript fusion) that have named homes in `tasks/` and `utils/`; and (c) the
**cache/provenance machinery still living in a 2,400-line script** (pre-existing debt this PR
deepened). None of these moves should happen inside this PR — the current state is verified and
byte-reproducible; each move below is a small, testable follow-up (tracked as T046–T052).

## F1 — Import hygiene is the root cause; fix it before moving anything (T046)

`import senselab.audio.workflows.audio_analysis.aggregate` pulls torch+speechbrain+transformers
because of exactly two eager `__init__`s: `audio/workflows/__init__.py:3` (imports
`explore_conversation`, which imports four model task stacks) and `audio_analysis/__init__.py:18-42`
(imports `compute` → `embeddings` → speechbrain). The repo already contains the fix pattern:
`adaptive/__init__.py` is PEP-562 lazy by design. **Recommendation**: lazify both parents the same
way (public API unchanged; `pdoc` handles `__getattr__` re-exports). This single change deletes the
`_ensure_light_importable` sys.modules stub in `scripts/adaptive_loop.py` — a hack that exists purely
to route around those two files — and is a precondition for F2/F3 (workflow code calling task APIs in
degraded environments). Also aligns with the standing import-time optimization effort
(`specs/20260501-154228-optimize-import-times`).

## F2 — `adaptive/backends.py` re-implements task capabilities; dissolve it into `tasks/` (T047)

| backends.py function | Existing home | What's missing there |
|---|---|---|
| `transcribe_crop` (HF whisper pipeline) | `tasks/speech_to_text` — `transcribe_audios([...], model=HFModel("openai/whisper-base"))` already works, model-agnostic (`huggingface.py:242-258`) | Nothing functional. U1 should build an `Audio` from the crop and call the task API; the pipeline-object cache (`_ASR_CACHE`) mirrors the backend's own `_pipelines` cache |
| `consensus_align` (torchaudio MMS_FA) | `tasks/forced_alignment` — has a **dead torchaudio slot**: `DEFAULT_ALIGN_MODELS_TORCH` (`constants.py:59-64`) and a `model_type == "torchaudio"` branch (`forced_alignment.py:173`) that nothing ever reaches | Wire MMS_FA as the real torchaudio backend of `align_transcriptions`; U3 then passes the consensus text through the standard API (and inherits its caching/`levels_to_keep` semantics) |
| `overlap_posteriors` (per-class powerset + pyannote-4.x multilabel handling + chunk stitching) | `tasks/voice_activity_detection/frame_posteriors.py` — `chunked_frame_inference` already returns the full `(frames, classes)` array and then discards classes 4-6 (`:86-88`); `FramePosterior` has no per-class field | This is FR-016's specified location. Add `per_class`/`overlap` to `FramePosterior` (or an `include_per_class=True` flag); backends.py's 3.x/4.x format handling and stitching should merge into `stitch_frames`/`_output_to_array`, not live in a workflow |
| `embed_windows` (fine-hop ECAPA) | `tasks/speaker_embeddings` + the workflow's own `embeddings.extract_per_window_embeddings` (same computation at coarse hop) | Parameterize the existing extractor (window/hop already args) and call it on crops; delete the duplicate |

After T046+T047, `backends.py` reduces to thin guard wrappers (`(result, reason)` envelopes around
task APIs) or disappears entirely — guards can live in the intervention `guard` functions.

## F3 — `adaptive/audio_io.py` duplicates preprocessing *and* creates a waveform-parity hazard (T048)

`load_wav_16k_mono` (soundfile + `resample_poly`) and `crop` re-implement
`read_audios`/`resample_audios`/`extract_segments`. Beyond duplication there is a **correctness
nuance**: senselab's resampler and `resample_poly` produce different sample values, so an `Audio`
built from `audio_io` has a different `audio_signature` than the pipeline's own crop of the same
region — live U1/I4 results can never share cache entries with pipeline-produced crops
(contracts/region-reprocessing.md assumes they do). **Recommendation**: in full environments route
through `senselab.audio.tasks.preprocessing` (guaranteeing signature parity, requires F1); keep the
current numpy path only as the explicitly-labeled degraded-environment fallback. SepFormer
`_enhance` should likewise call `tasks/speech_enhancement.enhance_audios` rather than driving
speechbrain directly.

## F4 — Generic utilities grown inside the workflow → named homes (T049)

1. **ScriptLine leaf-walk**: ≥5 independent implementations (`fusion.iter_word_leaves`, two `_walk`
   closures in `harvesters.py:170,701`, `forced_alignment.flatten_script_lines:550`,
   `plotting.py:46`). Consolidate as `ScriptLine.iter_leaves()` (+ a dict-tolerant module helper for
   serialized JSON) in `utils/data_structures/script_line.py`; migrate call sites opportunistically.
2. **WER**: two code paths — `tasks/speech_to_text_evaluation` (jiwer) vs the workflow's hand-rolled
   `_levenshtein`/`_wer`/`_normalize_transcript_for_wer`. Keep `harvesters._levenshtein` (it works on
   phoneme *sequences*, jiwer doesn't), but `adaptive/evaluate.py` should call `calculate_wer` with
   the hand-rolled version as the no-jiwer fallback; move `_normalize_transcript_for_wer` to
   `speech_to_text_evaluation/utils.py` where both consumers reach it.
3. **DSP SNR**: `triage.dsp_snr_series` (incl. the posterior-masked noise floor) belongs with the
   three existing SNR estimators in `quality_control/metrics.py` — as `frame_snr_series_metric` /
   posterior-masked variant. Blocker: `quality_control/metrics.py:11-14` imports VAD+diarization APIs
   at top level (a heavy import for a "metrics" module — pre-existing debt worth splitting into pure-
   DSP vs model-based halves while moving this in).
4. **LS ground-truth parsing**: `adaptive/evaluate.load_ls_ground_truth` (import side) should sit
   next to the export side in `audio_analysis/labelstudio.py`.

## F5 — Transcript fusion is a capability, not workflow plumbing (T050)

`fuse_words` (ROVER-lite), `load_calibrator`, and `collect_word_streams` are model-independent,
ScriptLine-shaped, and pure — and there is no transcript-ensemble utility anywhere in `tasks/`.
**Recommendation**: promote to a new `tasks/speech_to_text_ensemble/` (naming parallels
`speech_to_text_evaluation/`), with the policy coupling removed at the boundary: accept an explicit
`weights: dict[model_id, float]` instead of the adaptive policy dict (the adaptive loop computes
weights via `policy.family_weights` and passes them in). The workflow keeps speaker/p_voice lookups
and `build_final_outputs` (those are workflow semantics). Defer until a second consumer exists if you
prefer rule-of-three — but the module is already dependency-free, so the move is cheap.

## F6 — Cache/provenance machinery in the script (pre-existing, deepened by T012) (T051)

`audio_signature`, `cache_key`, `align_cache_key`, `run_task_cached`, `run_alignment_cached`,
`_sync_cache_with_schema_version`, `wrapper_version_hash` are library-grade infrastructure inside
`scripts/analyze_audio.py`. T040 (in-process adaptive integration) cannot be built cleanly without
importing them, and the `_stage_*` functions (T012) belong in the workflow package once the cache
layer moves. **Recommendation**: `utils/tasks/cached_inference.py` (content-addressed outcome cache +
provenance envelope), then `workflows/audio_analysis/stages.py` for the six stage functions, leaving
the script a thin CLI. Note the `wrapper_version_hash` semantics change this implies: hash the
*stage/workflow modules* rather than the CLI file, so editing CLI plumbing stops invalidating the
model cache (make this an explicit, documented decision — it changes invalidation behavior).

## F7 — Typed internals instead of dict soup (T052, opportunistic)

House style for workflow internals is `@dataclass(slots=True)` (`types.py`, `BucketGrid`,
`WindowEmbedding` — Pydantic is deliberately reserved for cross-boundary types). The adaptive core
follows this for `Vote`/`PassHarvest` but passes `Region`, planner candidates/`InterventionRecord`,
election records, and the loop `ctx` (a 15-key god-dict shared with every intervention) as plain
dicts. data-model.md already specifies these entities. **Recommendation**: introduce
`adaptive/types.py` with `Region`, `PlannedIntervention`, `LoopContext` dataclasses; convert
incrementally (planner + regions first — they're the most-tested pure code). Also promote the
`RULES` list-of-dicts to a small `InterventionRule` dataclass or Protocol so rule authors get a typed
contract instead of a dict shape convention.

## Correctly placed — leave alone

`votes.py` (pure aggregate half) next to `aggregate.py`/`types.py`; `compute.harvest_pass` in the
workflow; `adaptive/{loop, policy, belief, regions, convergence, triage-decision, ls_final, plot,
evaluate-orchestration}.py`; `policy/default.yaml` as packaged data; the speckit docs; the e2e/env-
gated tests mirroring `src/tests/`; `scripts/make_degradation_suite.py` and `scripts/adaptive_loop.py`
as CLIs. `identity_repair.py` stays workflow-level (its calibration and consensus semantics are
uncertainty-workflow-specific), but its generic primitives (`_agglomerative_cosine`,
`detect_change_points`, `cross_source_disagreement`) are `utils/tasks/` candidates when a second
consumer appears — flagged, not moved.

## Sequencing

Nothing moves in this PR (the tree is verified, byte-reproducible, and every move above invalidates
that evidence). Follow-up order, each its own small PR with tests:

1. **T046** lazy `__init__`s (unlocks everything; deletes the import stub hack).
2. **T047** dissolve `backends.py` into task APIs (incl. wiring the dead torchaudio aligner slot;
   subsumes Phase-8 T041's posterior work via the FramePosterior extension).
3. **T048** `audio_io` → preprocessing routing (fixes the crop-signature parity hazard).
4. **T051** cache layer → `utils/tasks/cached_inference.py` + stages → workflow (enables T040).
5. **T049/T050/T052** utility consolidation, fusion promotion, typed internals — opportunistic.
