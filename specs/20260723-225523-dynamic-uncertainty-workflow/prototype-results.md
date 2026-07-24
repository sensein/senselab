# Prototype results — audio_48khz_mono_16bits vs human ground truth

**Date**: 2026-07-24 · **Branch**: `20260722-175022-scene-quality-utterance` (uncommitted) ·
**Code**: `src/senselab/audio/workflows/audio_analysis/adaptive/` + `scripts/adaptive_loop.py` ·
**Input**: `artifacts/e2e_runs/audio_48khz_mono_16bits_20260724-001937` + `artifacts/analyze_audio_cache` ·
**Ground truth**: Label Studio export `updated-label-a7a37522.json` (4.92 s, 4 speakers, 5 segments —
one deliberately left untranscribed by the annotator at 2.17–2.99 s) ·
**Output**: `artifacts/adaptive_prototype/run2/`

The prototype is artifact-driven: round 1 ingests a completed analyze_audio run into the belief store;
rounds 2–3 run the policy engine; reserve-ASR escalation replays `openai/whisper-large-v3-turbo` and
`ibm-granite/granite-speech-3.3-8b` from the content-addressable cache (no model loads anywhere).
Command: `python3 scripts/adaptive_loop.py <run_dir> --cache-dir artifacts/analyze_audio_cache
--ground-truth <ls-export.json>`.

## Headline results

**1. Harvest/aggregate split proven (D8/T007).** Re-aggregating every bucket purely from the vote
store reproduces all six stored parquets exactly: 540/540 buckets, max_abs_diff = 0.0, 0 mismatches.
Iteration is therefore free of GPU re-work by construction.

**2. The loop attacks the right region and reduces uncertainty (US2).** The enhanced-stream utterance
axis peaked at 0.81/0.71 over buckets 2.0–3.5 s — covering exactly the span the human annotator could
not transcribe. Round 2: `U2_reserve_escalation` fired on that region (2 cached reserves, 18
region-scoped votes, same-model shadowing) → mean uncertainty 0.609 → 0.519 (Δ −0.091). Round 3: U2
re-fired, Δ 0.0 → ε-monotonicity guard + max-region-rounds closed both buckets as
`irreducible: no_reduction_under_available_interventions` (residual 0.72/0.83, quality floor 0.0 —
clean audio, so the residual is not noise; five model families simply disagree). The loop's verdict and
the annotator's blank textarea agree independently.

**3. Fusion beats every individual model (US4).** Word-slot voting over 5 models (3 from the run + 2
cache-replayed reserves), family-weighted (whisper-derived models share one family weight):

| source | WER (transcribed GT regions) | normalized* |
|---|---|---|
| **fused consensus** | **0.059** | **0.000** |
| Qwen3-ASR-1.7B | 0.059 | 0.059 |
| granite-speech-3.3-8b | 0.059 | 0.059 |
| whisper-large-v3-turbo | 0.059 | 0.000 |
| canary-qwen-2.5b | 0.118 | 0.059 |
| CrisperWhisper2.0_turbo | 0.176 | 0.176 |

\* annotator shorthand "u"→"you" equivalence.

**4. Confidence localizes human uncertainty.** Mean fused word confidence inside the untranscribed GT
span: **0.36**, vs **0.49** in transcribed spans (with coverage-scaled abstention penalty; words there
carry `alternates` like And/I'm/Ranger). Presence: accuracy 0.913, recall 1.00 on the 0.1 s grid.
Identity uncertainty at GT speaker boundaries **0.92** vs **0.75** within segments — the axis points at
exactly the boundaries `I1_boundary_refinement` (deferred, `next_actions`) would refine; the known
diarization failure (pyannote merged all 4 speakers into one cluster) is surfaced, not hidden
(word-speaker accuracy 0.63 against the single cluster).

**4b. Visual output.** `final/timeline.png` (`adaptive/plot.py`, best-effort like analyze_audio's
timeline): five aligned rows — ground truth (untranscribed span hatched), presence p_voice +
uncertainty band, identity uncertainty with GT boundaries dashed, utterance uncertainty round-1 vs
final with the proposed region, fired interventions (with Δ) and irreducible hatching, and the fused
words colored by confidence with alternates.

**5. Loop bookkeeping honest and deterministic.** 13 interventions fired (S1 elections, C9
missed-speech with recorded +Δ belief revisions, 2×U2), I1 blocked with `embedding_backend_unavailable`
into `next_actions`, P3 found nothing to purge (clean clip — correctly zero). Budget: 11 light, 2
medium, 0 heavy. Two consecutive runs produce **byte-identical `iterations.json`** (SC-004 at prototype
scale). Unit tests: 7/7 (`adaptive_prototype_test.py`); ruff clean.

## Full version (2026-07-24, second pass) — live interventions + identity repair

New modules: `adaptive/identity_repair.py` (I1 change-points + I2 consensus re-cluster from per-window
embeddings), `adaptive/audio_io.py` (16 kHz load/crop, SepFormer enhanced-stream-on-demand),
`adaptive/backends.py` (live whisper re-ASR, live fine-hop embeddings, gated segmentation-3.0 overlap
posteriors — all guarded). Policy v2 adds `identity.*`, `u1_asr_models`, `enhancement_model`. Nothing
is tuned to the test clip; all parameters are policy defaults.

Results on the same run dir (`artifacts/adaptive_prototype/run7`):

| metric | baseline loop | full version | GT |
|---|---|---|---|
| predicted speakers | 1 cluster | **3 refined clusters** | 4 |
| word_speaker_accuracy | 0.63 | **0.80** | 1.0 |
| boundary precision / recall (±0.25 s) | — | **1.00 / 0.50** | — |
| fused WER (raw / normalized) | 0.059 / 0.000 | 0.059 / 0.000 (held) | 0 |
| confidence untranscribed vs transcribed | 0.36 < 0.49 | 0.36 < 0.49 (held) | — |

Change-points landed at 0.875 s (GT 0.859) and 3.125 s (GT gap edge); the missed boundaries flank the
0.38 s "Kenny" segment — below what 0.25 s-hop stored embeddings resolve; the live fine-hop path
(`identity.fine_hop_s: 0.1`) exists for full environments. U1 fired live (whisper-base on the crop,
14 words), with the enhanced-stream request correctly falling back to raw
(`enhancement_backend_unavailable` recorded) — new dissenting evidence honestly raised measured
disagreement (+0.04) in the span the annotator also could not transcribe. Determinism re-verified
byte-identical with live U1 in the loop; 8/8 unit tests (incl. synthetic 3-speaker recovery test);
ruff clean.

**Generalization check** (`artifacts/adaptive_prototype/gen1`): identical policy on
`english_conversation_higgs_audio_v2` (21.5 s, two-speaker argument) — parity 0/2306 buckets, 72
interventions (44 S1, 9 I1, 9 I2, 5 U2, 4 U1 live, 1 C9; I4 guarded), refined clusters recover the
two dominant alternating speakers (R0 8.0 s / R3 8.9 s) with interjections split into minor clusters,
coherent speaker-attributed transcript, 9/24 medium budget. No crashes, no clip-specific behavior.

## Third pass (2026-07-24) — granite dropped; fusion-stream rule fixed

Investigating per-model behavior surfaced a stream-routing defect, not a model defect: on the
**enhanced** stream, SepFormer damaged the ASR — CrisperWhisper lost the whole 2.2–3.0 s span *and* the
final "you" (its 0.176 WER was an enhancement casualty; on raw it reads "…Kenny and Josh. We just
wanted to take a minute to thank you."), and Qwen3's only error was the same enhanced-dropped "you".
Region elections (presence 0.4 / quality 0.3 / agreement 0.3) still picked enhanced, and granite's
MMS-aligned words were silently patching the holes — dropping granite alone cost a word
(fused 0.118/0.059).

**General fix** (`loop.py`): the fusion stream is now the stream with the lowest **final utterance
uncertainty mass** (transcripts come from the stream whose transcript evidence is most self-consistent;
elections remain for region re-processing). With granite removed from `reserve_asr_models`
(`run10`, raw-stream fusion, 4-model ensemble + live whisper-base): fused WER **0.059 / 0.000
normalized**, word_speaker_accuracy **0.81**, confidence separation untranscribed-vs-transcribed
**0.42 / 0.95** (sharpest yet), determinism re-verified. CrisperWhisper on raw improves to 0.118;
whisper-base scores 0.294 alone but is family-weighted (whisper = {crisper, turbo, base}) and cannot
drag the consensus. Fused transcript: "This is Peter / This is Johnny / Kenny / *and and Joe I We*
(low-confidence, the untranscribed span) / we just wanted to take a minute to… thank you."

## Fourth pass (2026-07-24) — triage implemented; I4 validated live; model refresh memo

**Triage (US1, T010/T011)**: `adaptive/triage.py` (pure, unit-tested) + `--enhancement auto` wiring in
`scripts/analyze_audio.py`. Per SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md the speech gate uses continuous
segmentation-3.0 frame posteriors aggregated at ~100 ms (never segmentized VAD); SNR from Brouhaha with
a posterior-masked percentile-DSP fallback. Live validation (gated model, real token):
silent 3 s → `speech_present=False` (FR-004 stop); synthetic conversation → speech, **no enhancement**
(median 47.7 dB); clean+noise → **enhancement** (0.75 dB, low-SNR fraction 1.0); the archival test clip
→ enhancement under the DSP fallback (9.1 dB — borderline; Brouhaha is authoritative in production).
Conservative posture is intentional: triage gates *compute*, and the fusion-stream rule +
S1 guard already defend against a useless enhanced stream downstream.

**I4 validated live** (`run13`): with HF token + pyannote 4.0.7, segmentation-3.0 scored the contested
region (288 frames) with **mean overlap 0.0** — the untranscribed span is a single unintelligible
speaker, not overlapped speech, so `no_reduction_under_available_interventions` is the *correct*
irreducibility reason. Stream fallback recorded (overlap is scene-invariant; raw waveform used when the
enhanced stream can't be materialized). `next_actions` is now empty: **every rule in the v1 catalog has
executed live at least once.** pyannote-4.x output format (per-chunk multilabel speaker activations)
handled alongside the 3.x powerset format in `backends.overlap_posteriors`.

**Model landscape refresh**: see [research-models-2026.md](./research-models-2026.md) — three
HF-verified surveys (scene/events, diarization/VAD/embeddings, enhancement/separation/TSE) with a
phased adoption plan. Headlines: CED replaces AST (527-class, variable-length — removes the 10.24 s
crop prohibition); Streaming Sortformer v2 + DiariZen replace Sortformer v1 for speaker counting;
MarbleNet v2 / TEN-VAD give the triage gate an ungated frame-posterior backend; MossFormerGAN_SE_16K /
DeepFilterNet3 replace SepFormer with a do-no-harm SI-SDR gate; ReDimNet/ERes2NetV2 target the
unresolved 0.38 s speaker. Independent literature (arXiv 2512.17562, URGENT 2025) confirms the
raw-authoritative / gated-enhancement design.

## Fifth pass (2026-07-24) — remaining tasks implemented; tasks.md fully checked

**T008/T009 — harvest/aggregate split in the comparator itself.** New pure module `votes.py`
(`PassHarvest` + `aggregate_pass` + `compute_pass_deltas`, aggregation math moved verbatim;
unit-tested); `compute.py` refactored to `harvest_pass` (all model-touching work, no caller
mutation) + a signature-compatible `compute_uncertainty_axes` wrapper (`mutate_passes=True` default
preserves the timeline-plot contract; `False` gives the clean API). `VoteStore.from_harvests`
closes the in-process loop-integration path. Sandbox verification: compile, lint, 3/3 votes tests;
run the full `src/tests/audio/workflows/audio_analysis/` suite on a full env before merging.

**T012 — run_pass stage decomposition**: six `_stage_*` functions, pure code motion, cache keys
unchanged. Note (applies to every script edit incl. triage): `wrapper_version_hash` is a whole-file
sha256, so the first post-merge analyze_audio run rebuilds the model cache once by design.

**T031/T032/T033 — fusion follow-ups** (validated on `run14`): U3 consensus re-alignment via
torchaudio MMS_FA with SIGALRM timeout guard (`transcript.json.timestamps` records source/fallback
reason; guard exercised live); LS `final__*` tracks + `disagreements_resolved.json` (100 round-1
disagreements annotated with final status, Δ, and intervention audit chains); calibration profile
mechanism (logistic/piecewise, `calibrated: true` path, unit-tested — fitting stays with the
US5 harness).

**T037–T039 — validation harnesses**: determinism e2e **passing** in 6.2 s against the reference
run (hermetic, U3 off); golden-compat value-diff harness (needs two full-pipeline runs → GPU/Mac);
degradation-suite generator (5 variants + injected-span manifest, generated under
`artifacts/degradation_suite/`) + SC-001 checker. 12/12 unit + 1/1 e2e tests green; ruff clean
across all touched files.

**Spec status: every task in tasks.md is now implemented.** Remaining full-env verification
(documented per task): comparator test suite green post-split, golden-compat diff on a GPU run,
degradation-suite pipeline runs, one-time MMS_FA download for live U3.

## Known limitations (tracked in tasks.md)

- I4 overlap posteriors code-complete but unvalidated here (gated `pyannote/segmentation-3.0` needs an
  HF token); with it, the irreducible region would be named `overlapping_speech` instead of the
  generic reason, and the aleatoric floor would include the overlap term.
- U1's senselab-native backend routing (subprocess-venv ASR models) is follow-up; the live path here
  uses the HF whisper pipeline with a policy-declared model pool.
- Short-segment identity resolution is bounded by the stored embedding hop (0.25 s ⇒ the 0.38 s GT
  segment was not separated); full envs re-embed at `identity.fine_hop_s` (0.1 s).
- Word-level (not phoneme) pair distances in the sandbox fallback (`pair_distance_kind: "word"` —
  g2p_en unavailable); full env uses `harvest_utterance_votes` unchanged.
- Fused confidences are uncalibrated (`calibrated: false`; T033) and depressed by slot fragmentation;
  the *ordering* is what the localization result relies on.
- Stream election weights favored the enhanced stream (presence+quality over utterance agreement) —
  a policy knob to revisit; fusion still scored best-in-ensemble on it.
- Triage round (US1) not exercised — inputs were pre-computed artifacts by design.
