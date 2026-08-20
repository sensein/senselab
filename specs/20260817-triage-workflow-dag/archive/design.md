# The triage workflow, designed from the questions backwards

> **Superseded, 2026-08-20.** This document indexes a graph that no longer exists: `flowchart.md` and
> `workflow.nf` have been deleted and will be regenerated from the node documents, and its framing of
> the work as phases against a findings register is not the current goal. The design is now
> `admit.md`, `taxonomy.md`, `branch-speech.md`, `branch-airway.md`, `branch-voice.md`, with
> `decisions.md` and `ports.md` alongside. Kept only for the measurements and arguments in its later
> sections, which nothing else records.

**Status:** design, 2026-08-17. Three files:

- **[`flowchart.md`](flowchart.md)** — the graph, as ten mermaid diagrams. Self-contained; start there.
- **[`ports.md`](ports.md)** — the normative port tables. Every task, every input and output port,
  every config key. Where this document and that one disagree, that one wins.
- **[`workflow.nf`](workflow.nf)** — the same graph as Nextflow DSL2 pseudocode, for reading the
  dependency structure as code.

This document is the argument. It starts from what a caller needs to know about a recording, works
back to the evidence that answers it, and only then asks which of today's tools can bear on it. It
does **not** start from the existing decomposition, and it treats no part of the current module
layout as a constraint. Where today's code fits, the mapping in §7 says so with a file and a line;
where it does not, the same table says "does not exist".

It is written to be read by someone who has not read the audit register. Findings are cited as
`F-nnn` where the register has one, but every claim is restated in full.

---

## 1. Goals — the decisions the workflow exists to make

Ranked by what a consumer loses if the answer is wrong, not by dependency order. The dependency
order is §2.

| # | Decision | Consumer | What a wrong answer costs |
| --- | --- | --- | --- |
| **D1** | **Is a voice other than the intended target present?** | Study team: consent scope, exclusion, privacy review | An intruder voice in a single-target corpus contaminates every per-speaker measure derived from the file, and may be a consent violation. This is the standing project goal. |
| **D2** | Does a human need to review this file, and why? | Annotation queue | A false pass ships bad data as good. A false flag spends the scarcest resource in the project. |
| **D3** | What kinds of sound are in here — lexical speech, non-lexical vocalization, non-vocal sound, silence? | Everything downstream, plus task verification | Every other answer is conditioned on this one. A cough classified as background noise gets trimmed out of a cough recording. |
| **D4** | How many distinct voices? | Protocol compliance | A two-voice recording scored as compliant single-speaker read speech invalidates the analysis it feeds. |
| **D5** | Did the participant do the task they were asked to do? | Re-collection decision | An unusable session is discovered months later instead of the same day. |
| **D6** | What was said, with per-word confidence? | Linguistic features, and D5 and D7 | Dropped words are the measured failure mode, and a dropped short quiet word is also a dropped intruder utterance. |
| **D7** | Is there personal information, and where? | Release gate | An unredacted identifier in a released corpus is not recoverable. |
| **D8** | Is the recording good enough to measure? | Feature validity, re-collection | Acoustic features computed on a clipped or reverberant file are precise and wrong. |
| **D9** | Which regions should be cut, and for what reason each? | Preprocessing, release | A trim that removes the target's own vocalization destroys the measurement. A trim that leaves an intruder in place defeats D1. |
| **D10** | Where exactly is voice, in time? | D4, D6, D9, and per-segment features | Boundary error propagates into every windowed measure. Currently measured at ~2.6 s onset/offset MAE on a hard clip. |

**D1 is first-class, and it constrains the graph in three ways** that no other goal does. Taken from
`SPEECH_DETECTION_SOTA_REVIEW_2026.md:12-15,43-51`:

1. **There is no enrolled anchor.** Recordings arrive without a clean sample of the intended
   subject, so the check must be unsupervised: estimate the file's own dominant identity and score
   deviation from it. The honest published claim is therefore "a voice other than the majority voice
   is present", not "this person is not the participant". Where a caller *does* supply a target
   sample, that upgrades novelty detection to verification — and in this design it does so by being
   a wired input port, not by being a caveat in prose.
2. **It must run on the raw audio.** Enhancement, denoising and target-speaker extraction all exist
   to suppress background voices, which is precisely the evidence D1 needs. In this design the
   enhanced variant has **no wire** into the identity sub-workflow, so it cannot be used by
   accident.
3. **Silence and no-speech thresholds must not be raised.** The same knob that curbs ASR
   hallucination deletes short, quiet, off-target utterances. The graph favours a max-recall union
   across recognizers and treats a voiced span with no words as a *measured disagreement* rather
   than as silence.

Today D1 has no implementation at all. `off_target` appears once in `src/`, in a docstring at
`source_separation/unasdiff.py:9`; there is no field, column, function or artifact. What the pipeline
publishes instead is an untargeted speaker count plus cluster structure — which matches the review's
own fallback claim, but as a byproduct of diarization rather than as a measured axis.

---

## 2. The question tree

For each question: what would settle it, what partial evidence exists, what makes it uncertain, what
the answer is used for, and what it is a prerequisite for. The prerequisite column is the skeleton of
the DAG.

### Q0 — Is there any usable acoustic signal? *(prerequisite for everything)*

- **Decisive:** integrated loudness above a floor, and a per-band noise floor that resolves.
- **Available:** `level.py:138-238` computes LUFS, true peak and clipped fraction — but only
  reachable through `measure_variant` (`level.py:292`), which has no caller.
  `noise_floor.estimate_noise_floor` (`noise_floor.py:257`) is wired and good.
- **Uncertain when:** the recording is very quiet but not silent; the floor estimator cannot resolve
  bands below ~140 Hz at short frames.
- **Used for:** ending the run honestly. A dead file gets one flagged answer and no fabricated
  others.

### Q1 — What kinds of sound are in here? *(prerequisite for Q2–Q9)*

Four classes, not two: **lexical speech**, **non-lexical vocalization** (cry, cough, laugh, breath,
groan), **non-vocal sound**, **silence**.

- **Decisive:** frame-level posteriors for voice and for sound-event classes, plus periodicity, fused
  on the reporting grid.
- **Available, partially:** frame posteriors exist as a task —
  `voice_activity_detection/frame_posteriors.py:241` (`chunked_frame_inference`), `FramePosterior` at
  `:108` — and reach `speech_presence.harvest_speech_presence_evidence` (`speech_presence.py:148`)
  through an optional argument. Scene classification exists at window level: AST and YAMNet at 0.96 s
  windows / 0.48 s hop (`default.yaml:166-173`). Periodicity exists (openSMILE 10 ms LLDs,
  `features_extraction/opensmile.py:74`; PPG voice fraction, `ppg.py:136`).
- **What makes it uncertain, and why this is the question the current pipeline answers worst:**
  - The existing root question is binary — "was there a speaker?" (`doc.md:6`). There is no class for
    a non-lexical vocalization, so one gets forced into either speech or background.
  - It gets forced into background. `data/audioset_source_map.json` maps `"Baby cry, infant cry"` and
    `"Crying, sobbing"` to `people`, and `people` is a *background source category*. The `speech`
    task's target vocabulary is `["speech", "breath", "mouth_noise"]`, so an infant's cry is
    classified like a passerby (F-168, and `phase2-notes.md:131-148`).
  - Two independent measurements make it worse. **First:** 292 of the 527 labels in that map go to
    `environment`, which is also the map's `default` for an unmapped label, and where `Silence`
    itself lands (measured: per-category counts are environment 292, machine 167, people 60, speech
    8). **Second:** the YAMNet top-1 label is used as an authoritative veto on whether a window
    counts as speech before clustering (`compute.py:893`), and that function's own docstring names
    the child-voice-as-Music failure mode (F-170).
  - So a cry is simultaneously (a) not speech, (b) a background source, and (c) vetoed out of the
    speaker analysis. Three mechanisms, one population.
- **Used for:** conditioning every later question, and answering "did the participant cough" for a
  cough task directly.

### Q2 — Where exactly is voice, in time? *(prerequisite for Q3, Q5, Q6, Q8)*

- **Decisive:** frame posteriors with a boundary-localization objective, scored with collar-based
  event F1 and PSDS1 rather than boundary MAE.
- **Available:** frame posteriors, yes. A boundary objective, no.
- **Uncertain when:** short inter-turn gaps. Measured on a hard validation clip: pyannote
  `segmentation-3.0` plus Brouhaha fire continuously through a ~0.4 s gap and never dip; onset/offset
  MAE ≈ 2.6 s with high recall only (`SPEECH_DETECTION_SOTA_REVIEW_2026.md:57-61`). This is the known
  failure mode of frame-posterior VADs with hysteresis and minimum-duration smoothing.
- **Used for:** trimming, alignment, drawing embedding windows.

### Q3 — How many distinct voices? *(prerequisite for Q4)*

- **Decisive:** agreement between independent diarizers and independent embedding clusterings, with
  the count published as a distribution whose width narrows as sources agree.
- **Available:** two diarizers (`default.yaml:141-143`), two embedders (`:148-150`), harmonization
  across them (`harmonize.py:307`), and a count posterior (`speaker_identity.py:117`).
- **Uncertain when:** short files (both neural diarizers collapsed to one speaker on a 5 s clip);
  when the speech mask feeding clustering is noisy; when turns are shorter than the embedding window.
- **Two defects that matter here.** A vote share is published and consumed as a posterior, and its
  width does not narrow with more unanimous sources (F-179). And `multimodal_threshold=0.15`
  (`speaker_identity.py:121`) flips the "is the count multimodal" verdict between five and six
  agreeing sources with no change in the audio (F-144).
- **Used for:** protocol compliance, and as context for Q4.

### Q4 — Is any voice not the intended target? *(the goal)*

- **Decisive with an anchor:** verification against an enrolled embedding. Not available.
- **Best available without one:** five enrollment-free voters, all on raw audio —
  1. embedding novelty from the file's own dominant centroid;
  2. overlap posterior (two concurrent voices);
  3. frame-level background-voice scores (babble, crowd, conversation, chatter);
  4. speaker-change points from the embedding trajectory;
  5. for read speech only, transcript deviation from the reference passage.
- **Uncertain when:** the intruder speaks a lot, so the "dominant" cluster may not be the intended
  subject. The mitigation is to report *presence of more than one voice* as the primary claim and
  dominant-deviation as secondary, and to publish the cluster structure so a human can adjudicate.
- **Prerequisites:** Q1 (voiced spans to embed), Q3-adjacent evidence (cluster structure, overlap),
  Q6 for the read-speech voter.
- **Used for:** exclusion, consent review, and as the highest-priority arm of the review flag.

### Q5 — Did the participant do the asked task?

- **Decisive:** the expected task, which only the caller has.
- **Available:** `AudioHints` / `ExpectedSpeech` are declared at `audio_hints.py:31,129`. **Nothing
  reads them** — `grep` for `.hints` over `src/senselab/audio/workflows/` and
  `scripts/analyze_audio.py` returns zero hits, and `audio_hints.py:5` says so outright.
- **Uncertain when:** free speech, where there is no reference text — the check reduces to
  "is this the right kind of sound", i.e. Q1.
- **Prerequisites:** Q1, and Q6 for a read passage.
- **Design consequence:** with no expected task supplied, this answer is **absent**, not "unknown".
  A consumer cannot distinguish a fabricated midpoint from a measured one, which is the whole content
  of F-156.

### Q6 — What was said?

- **Decisive:** a max-recall union across recognizer families, aligned once, fused per word with a
  confidence per word and per edge.
- **Available and in reasonable shape:** three recognizer families
  (`default.yaml:144-147`), one aligner (`:157-161`), phoneme-graded fusion
  (`asr.fuse_consensus_words`, `asr.py:194`).
- **Uncertain when:** short quiet words; non-lexical vocalization, where Whisper's `no_speech_prob`
  head is trusted to flag hallucination and is a known failure point for exactly that input (F-166).
- **Prerequisites:** Q1 for the spans that *should* have produced words.
- **Used for:** Q5, Q7, linguistic features.

### Q7 — Is there personal information?

- **Decisive:** multiple independent detectors agreeing on a span, placed in time.
- **Available:** the whole `text/tasks/pii_detection` task, with a clean scan/decide split
  (`api.py:369`, `:515`) and a subprocess-venv backend running Presidio and GLiNER.
- **Uncertain when:** a single detector fires. The current code answers this with
  `count >= 2` (`pii.py:241`) — which is an evidence count wearing a boolean.
- **The reason this reads as "the one clean output" and is not:** no register finding touches
  `pii.py`. But three thresholds gate the verdict directly — `presidio_score_threshold=0.4`
  (`pii.py:82`), `gliner_threshold=0.5` (`pii.py:85`), and that `count >= 2` — and **there is no
  `pii:` section in the config at all**, so none can be changed without editing Python.
- **Prerequisites:** Q6 for the text, Q2/Q6 for the timing.

### Q8 — Is the recording good enough to measure?

- **Decisive:** per-axis degradation against anchors chosen for the content that is actually there.
- **Available:** Brouhaha frames (`scene_quality/brouhaha.py:220`),
  `quality.harvest_quality_measurements` (`quality.py:243`) — the cleanest existing L1 task, emitting
  dB / Hz / proportion with per-signal provenance and no scores — and
  `degradation.scene_degradation` (`degradation.py:129`).
- **Uncertain when:** the content is not fluent conversational speech. `scene_degradation` takes no
  task-type or content input at all, and its anchors are fixed at 25 dB SNR and 30 dB C50
  (`degradation.py:33-44`) — F-169. What counts as clean for read speech is not what counts as clean
  for a breathing task.
- **Prerequisites:** Q0, and Q1 for anchor selection.

### Q9 — Which regions should be cut?

- **Decisive:** a union of typed span sets, each carrying the reason it was proposed.
- **Available:** almost nothing. `grep` for `trim_regions` / `propose_trim` over `src/` returns
  nothing; the only `trim` identifier in the package is `background_mask.guard_trimmed_s`, unrelated.
  The prior design sourced trim from `background_mask`'s `target_free` spans only — see §7.
- **Prerequisites:** Q1, Q4, Q7, Q8.

### Q10 — Does a human need to look?

- **Decisive:** nothing, because there is no labeled corpus of "recordings a human should have looked
  at". So the flag is a **stated rule over the answers, not a fitted classifier** — inventing one
  here would add exactly the kind of unmeasured decision this whole design is reacting to.
- **Three arms**, each of which names itself when it fires: an answer crosses its review band with
  enough evidence to say so; an answer has too little evidence to adjudicate; the recording
  contradicts a supplied expected task. A flag with no reasons is a bug, not a pass.
- **Prerequisites:** all of the above.

---

## 3. Tools, mapped to questions — and the gaps

### 3.1 What exists

`src/senselab/audio/tasks/` (public entry points, subprocess-venv and cost noted):

| task | entry point | bears on | measures | cost | known failure mode |
| --- | --- | --- | --- | --- | --- |
| `voice_activity_detection` | `api.py:37`; `frame_posteriors.py:241` | Q1, Q2 | speech segments; **and per-frame probability + hop** | small; Sortformer path uses the `nemo-diarization` venv | segment path applies hysteresis and min-duration that erase sub-100 ms onsets — the reason the frame path matters |
| `scene_quality` | `brouhaha.py:220` | Q1, Q8 | frame SNR, C50 reverb, VAD | `brouhaha` venv, py3.11 | required-when-enabled: an unreachable Brouhaha fails the run loudly rather than shipping nulls |
| `classification` | `api.py:32`; `yamnet.py:152` | Q1, Q4 | AudioSet posteriors, clip or windowed | AST in-process; YAMNet in the `yamnet` venv | window-level only; YAMNet top-1 used as a veto (F-170) |
| `speech_to_text` | `api.py:55` | Q6 | transcript + word timings + token confidence | 5 of 6 backends need a venv; Canary-Qwen 2.5B, Qwen3-ASR 1.7B | shortfall in returned segments would silently drop text (`canary_qwen.py:156`) |
| `speech_to_text_ensemble` | `api.py:197` | Q6 | per-word consensus, alternates, corroboration | pure | keying provenance on `timestamp_source` alone does not work; refuses silent grapheme fallback |
| `forced_alignment` | `forced_alignment.py:685`; `mms_fa.py:24` | Q2, Q6 | word/char timings | MMS_FA bundle ~1.2 GB | dictionary-case bug: wrong case makes every alphabet character miss the lookup so only punctuation survives (`forced_alignment.py:64-76`) |
| `speaker_diarization` | `api.py:65`; `capabilities.py:240` | Q3 | speaker-labelled segments | 4 of 6 backends need a venv; VibeVoice ~14 GB | six backends disagree on everything but the return type; structural ceilings Sortformer 4, child-adult 2; probe was one corpus, one seed, TTS-composed, no room acoustics |
| `speaker_embeddings` | `api.py:32`; `windowing.py:51` | Q3, Q4 | ECAPA/ResNet vectors, windowed | in-process | many models mono/16 kHz only; the per-file provenance block is uninterpretable because `resample_audios` drops `filepath` |
| `speaker_verification` | `speaker_verification.py:19` | Q4, with an anchor | (score, same-speaker) | in-process | needs the anchor Q4 does not have |
| `speech_enhancement` | `api.py:19` | Q6, Q8 only | denoised waveform | SepFormer in-process; DriftSE venv | suppresses the background voices Q4 needs — must never gate Q4 |
| `quality_control` | `quality_control.py:44`; `metrics.py` (24 metrics) | Q8 | per-audio metrics + boolean checks | pure, plus transitive VAD | — |
| `features_extraction` | `api.py:34`; `opensmile.py:74`, `ppg.py:136`, `praat_parselmouth.py` | Q1, Q8 | acoustic scalars, 10 ms LLDs, phoneme posteriors | `sparc` and `ppgs` venvs, py3.11 | — |
| `preprocessing`, `input_output` | `preprocessing.py:30-398` | Q0 | resample, downmix, segment | pure | — |
| `source_separation` | `api.py:34`; `unasdiff.py:577` | not used by this design | separated sources | `unasdiff` venv, py3.10, diffusion model | `speech_speech` ships with a caveat from upstream |
| `pii_detection` (audio) | `api.py:28` | Q7 | PII over an ASR transcript | delegates to the text task | default ASR is `whisper-tiny` — a second, worse transcript |

`src/senselab/text/tasks/`: `pii_detection` (`api.py:369` scan, `:515` decide, `:593` compose;
`rules.py` cascade of regex, gazetteers, NER, demographics, age-over-90, combinatorial re-ID;
`subprocess_backend.py:544` runs Presidio + GLiNER in the `pii-detection` venv) → Q7.
`embeddings_extraction` (`api.py:14`) → not used by this design.

`src/senselab/utils/tasks/`: `cached_inference.py` (the cache, §4.4); `cosine_similarity.py:6`,
`pooling.py:8`, `dimensionality_reduction.py:72`, `eer.py:23`, `cca_cka.py:78` →
Q3/Q4 primitives; `embedding_distribution.py:381,936` — `select_dominant_vectors` and
`describe_embedding_distribution`, which is **exactly the dominant-cluster machinery Q4 needs** and
is documented as describing rather than deciding. `batching.py`, `cross_correlation.py`,
`plotting.py`, `input_output.py` have no production caller.

Models configured, from `default.yaml`: diarization `pyannote/speaker-diarization-community-1`,
`nvidia/diar_sortformer_4spk-v1` (`:141-143`); ASR `nyralabs/CrisperWhisper2.0_turbo`,
`nvidia/canary-qwen-2.5b`, `Qwen/Qwen3-ASR-1.7B` (`:144-147`); embeddings
`speechbrain/spkrec-ecapa-voxceleb`, `speechbrain/spkrec-resnet-voxceleb` (`:148-150`); scene
`MIT/ast-finetuned-audioset-10-10-0.4593`, `google/yamnet` (`:151-152`); enhancement
`speechbrain/sepformer-wham16k-enhancement` (`:155`); alignment `Qwen/Qwen3-ForcedAligner-0.6B`,
`facebook/mms-1b-all` (`:159-160`); reserve ASR `openai/whisper-large-v3-turbo` (`:478`), live re-ASR
`openai/whisper-base` (`:480`). No `revision:` key exists anywhere in the file; commits are resolved
at load time by `utils/model_revision.resolve_revision:254`.

### 3.2 The gap list — questions with no adequate tool

This is the more useful half of the mapping.

| # | Question with no adequate tool | Why the nearest tool is inadequate | What would close it |
| --- | --- | --- | --- |
| **G1** | **Q1: is this a non-lexical vocalization?** | No tool answers it. VAD says voice/not-voice. AST and YAMNet answer at window level and route a cry to `people`, a *background* category. There is no class for "the participant vocalized without words". | A frame-level model that separates speech from cough / cry / laugh / breath. The named candidates are a small trained head on WavLM frames, or an AudioSet-Strong frame model summed over ontology sub-trees. Needs labeled data this repo does not have. |
| **G2** | **Q2: precise onsets and offsets** | Frame-posterior VADs with hysteresis bridge short gaps; measured MAE ≈ 2.6 s with high recall. No boundary-localization objective anywhere in the stack, and the reported metric is boundary MAE, which hides it. | Boundary-aware inference over the posteriors already computed, plus collar-based event F1 and PSDS1 as the reported metrics. No new model strictly required. |
| **G3** | **Q1/Q4: frame-level sound events** | Both scene classifiers are window-level (0.96 s / 0.48 s after tuning; AST's native window is 10.24 s). A window vote spans many reporting buckets and inflates apparent agreement. | A frame-level AudioSet-Strong model, which also supplies the `Speech`-vs-`Babble/Crowd/Hubbub` split that Q4 voter 3 needs and a cleaner speech mask for clustering. |
| **G4** | **Q4: the off-target axis itself** | Does not exist. Every ingredient exists separately; nothing composes them, and there is no `off_target` product. | The `novelty_track` / `off_target_fold` / `off_target_gate` tasks in `ports.md` §4. `utils/tasks/embedding_distribution.py` already provides dominant-vector selection. |
| **G5** | **Q4/Q5/Q6: the caller's hints** | `AudioHints`, `ExpectedSpeech` and `TargetSpeakerEmbedding` are declared and consumed by nothing. `Audio.hints` (`audio.py:60`) is a field no workflow reads. The one producer of `TargetSpeakerEmbedding` (`speaker_embeddings/api.py:142`) itself has no production caller. | Wire them as input ports. Three answers change behaviour when a hint is present, and none can today. |
| **G6** | **Q6: contextual biasing** | No biasing path. `transcribe_audios` takes no per-file vocabulary, so a known name that a recognizer drops stays dropped. | A biasing vocabulary input port on `transcribe`, filled from `hints.expected_speech`. Cheapest item on the review's priority list. |
| **G7** | **Q9: trim regions** | No implementation. The prior source of them — `background_mask`'s `target_free` spans — reaches its intended consumer through a port with no producer, and its sibling state fires on `Silence` (§7). | `trim_proposal` / `trim_gate`, over four typed span sets, each carrying its reason. |
| **G8** | **Every answer: evidence-carrying confidence** | The `Estimate` type exists — `utils/data_structures/estimate.py:28`, with `value`, `raw`, `n_evidence`, `prior` and an `Estimate.no_evidence()` constructor at `:137` — and **nothing in `audio_analysis` constructs one**. The workflow's own `estimates.py` is a parquet column schema (`estimates.py:101`), not a type. | Make every gate return one. This is the single highest-leverage change, and unlike the rest of this list it needs no new model. |
| **G9** | **Q10: a fitted flag** | No labeled review decisions exist. | Collect them. Until then the flag is a stated rule and says so. |
| **G10** | **Q3: population validity** | Every cosine floor, every diarizer, every anchor is adult-derived, and nothing in the output says so (F-164, F-167, F-169, F-170, F-171, F-173). | `Estimate.population`, per answer. Surfacing is not fixing, and the design says so out loud. |

---

## 4. The DAG

### 4.1 Nodes are tasks; edges are named products

The whole content of the node model is in [`ports.md`](ports.md). Three properties are worth
restating here because they are what the design buys:

**No untyped bag.** The current pipeline passes a `pass_summary: dict[str, Any]` between stages;
measured, nine modules read eight top-level keys from it at 33 sites, and no signature declares any
of them. Every one of those becomes a named product with one producer. The immediate benefit is that
F-187 — a consumer reading a key the producer never writes — becomes a build error instead of an
empty list.

**Absence propagates; there are no skip flags.** A gate that cannot decide emits nothing on its
downstream port, so its consumers have no input and do not run. The seven dead `run_*` booleans in
`RunConfig` (`run_config_liveness_test.py:78-87`) are what the alternative looks like: a stage-enable
mechanism that was superseded by `skipped_stages` and left in place, advertising control it does not
have.

**One producer per product name.** Two tasks emitting the same name is an error worth catching,
because the current tree has a live instance of the failure it prevents: `backends.py` is a second,
uncached, unprovenanced invocation surface for ASR, embeddings, Brouhaha frames, diarizer spans and
alignment (§7.3).

### 4.2 Acyclicity, and where the round loop sits

The data graph is acyclic. The refinement rounds are a **conditional re-entry of one sub-workflow**,
`REFINE`, not an unrolled copy of the graph.

**What crosses an iteration boundary:** the ledger (every answer's `Estimate` plus every span set),
the remaining budget, and the list of action sets already executed. Nothing else. No task takes a
round index; only `rank_undecided` and `stop_or_continue` see round state at all, and they see it as
ordinary input ports.

**Why that is not a cycle.** Products are versioned by round. `ledger@k` is an input to round *k+1*'s
tasks, and round *k+1* writes `ledger@k+1` — a different value at a different name. No task ever
writes a product one of its own ancestors read. The cycle exists only in the graph over *task names*,
and task names are not products.

**Exit criteria, stated once and in one place** (`stop_or_continue`):

| reason | condition |
| --- | --- |
| `DECIDED` | every answer's `Estimate` is outside its ambiguity band with evidence at or above its floor |
| `IRREDUCIBLE` | an answer is still ambiguous, and no unused action is predicted to add an *independent* source to it |
| `OSCILLATING` | the planned action set repeats one already in `action_history` |
| `EXHAUSTED` | budget spent, or the round index reaches `cfg.rounds.max_rounds` |

`stop_reason` is a published output. `IRREDUCIBLE` together with an ambiguous answer is the honest
terminal state: the available tools cannot decide this file, so a human must. That is the same edge
as Arm 2 of the review flag, and it is deliberate — "we could not tell" is one state, not two.

**Today there are two loops with different exit criteria in the same process**, which is the concrete
reason to state this once. `fuse.py:1065` loops rounds and calls `rounds.assess_convergence`, whose
criteria C1–C4 are at `rounds.py:322-328`. `adaptive/loop.py:224` loops rounds and exits at
`loop.py:315-317`, where `run_state="converged"` means only that no intervention fired and none was
even proposed — `assess_convergence` is never called from `adaptive/` at all. `rounds.py` contains no
`irreducible`; that word is a per-bucket status decided separately at `adaptive/convergence.py:79-87`.

### 4.3 Fan-out dimensions

Three, all explicit as channel dimensions rather than as loops inside a function:

- **variant** — `raw` and `enhanced`. `transcribe` and `scene_quality_frames` see both; **VOICE
  IDENTITY has no `audio_enhanced` port at all**.
- **model** — one call per model id in the relevant config list, for `transcribe`, `diarize`,
  `window_embeddings`, `pii_scan`.
- **region** — round ≥ 1 only, from `narrow_input`.

### 4.4 Where caching attaches

At every model-inference task and nowhere else. The key is already right:
`cached_inference.cache_key` (`cached_inference.py:353`) hashes
`{schema, audio_signature, task, model, params, code_version, senselab_version, commit_sha}` at
`:383-394`, with `CACHE_SCHEMA_VERSION = 23` at `:64` and `commit_sha` keyword-only with no default
on purpose (`:364-368`). Alignment has its own key (`align_cache_key:398`) including
`transcript_sha`, which is what makes ASR and alignment separately cacheable.

Pure computations and gates are **not** cached. They are cheap, deterministic in their declared
inputs, and a second invalidation surface is a liability — the whole point of `CACHE_SCHEMA_VERSION`
being a single coarse lever.

One consequence to state rather than discover: a round that narrows to a region hands the inference
tasks a **different waveform**, so it gets a different `audio_signature` and its own cache entry with
no new key field. But the slice must be materialised as audio and its offsets restored on the way
out. `adaptive/backends.py` already crops this way — and bypasses `run_task_cached` entirely, so
today those crops are recomputed on every round and recorded in no provenance.

---

## 5. The pseudocode, and why Nextflow

[`workflow.nf`](workflow.nf). Nextflow DSL2 rather than CWL, for one reason stated plainly: CWL
expresses a static DAG well and data-dependent iteration badly. Its `when` clause skips a step on a
per-invocation predicate, and iteration to a fixed point has no standard construct — the usual CWL
answer is to unroll the loop to a fixed depth, which makes the round count a structural property of
the workflow file instead of a stopping decision with stated criteria. That is exactly the thing this
design is trying to express. DSL2 gives first-class workflow composition, so a sub-workflow is a task
with ports, and a recursion construct whose exit predicate is a readable boolean over the emitted
state.

Two caveats, stated rather than hidden: Nextflow's workflow recursion is a preview feature, and the
implementation will be Python. Neither affects the review, which is of the port graph and the exit
conditions.

---

## 6. The diagrams

[`flowchart.md`](flowchart.md), ten figures: the overview (Figure 1), one per sub-workflow (2, 3, 4,
6, 7), `VOICE_IDENTITY` drawn both as a single node with its ports and expanded (5a, 5b), the round
loop with its exit criteria on the edges (8), where caching attaches (9), and — for contrast — the
current code's speaker-attribution path with its port that has no producer (10).

---

## 7. Mapping to today's code, and what falls away

### 7.1 Node by node

| proposed task | implemented today by | state |
| --- | --- | --- |
| `decode_audio` | `preprocessing.py:30,100`; `cached_inference.audio_signature:319` | exists |
| `level_and_floor` | `level.py:138-238` + `noise_floor.estimate_noise_floor:257` | half-dead: the loudness half is reachable only via `measure_variant` (`level.py:292`), which has no caller |
| `signal_gate` | `adaptive/triage.py:21-155`, called from `scripts/analyze_audio.py:559` | exists, but decides "was there speech", not "is there signal" |
| `enhance_audio` | `speech_enhancement/api.py:19` via `perturbations.speech_enhancement:174` | exists |
| `speech_frame_posterior` | `voice_activity_detection/frame_posteriors.py:241` | exists; optional argument to `speech_presence.py:148` |
| `sound_event_posterior` | `classification/api.py:32`, `yamnet.py:152` via `stages.stage_scene:170` | exists at window level only — **G3** |
| `voicing_track` | `features_extraction/opensmile.py:74`, `ppg.py:136` via `stages.stage_features:270` | exists |
| `taxonomy_fold` | nearest: `speech_presence.harvest_speech_presence_evidence:148` + `speech_presence_link.link_speech_presence:497` (binary) and `sound_sources.harvest_source_categories:175` (four *source* categories) | **does not exist** as a four-way content fold — **G1** |
| `content_gate` | — | **does not exist** |
| `transcribe` | `speech_to_text/api.py:55` via `stages.stage_asr:323` | exists |
| `align_words` | `stages.stage_alignment:363` | exists |
| `fuse_words` | `asr.fuse_consensus_words:194` over `speech_to_text_ensemble/api.py:197` | exists; `policy=` reached the only production call site only after the F-162 fix |
| `transcript_gate` | — | **does not exist** |
| `task_match_gate` | — | **does not exist**; its input has no producer — **G5** |
| `pii_scan` | `text/tasks/pii_detection/api.py:369`, `subprocess_backend.py:544` | exists |
| `pii_gate` | `pii.py:241`, `text/.../api.py:515` | exists, but publishes a boolean where it has a count |
| `window_embeddings` | `speaker_embeddings/windowing.py:51`, via `compute.py:134` | exists; drawn on a YAMNet veto (`compute.py:893`) rather than on measured vocal activity |
| `cluster_windows` | `embeddings.cluster_pass_speakers:97` | exists and is **dead in production** — see 7.2 |
| `diarize` | `speaker_diarization/api.py:65` via `stages.stage_diarization:136` | exists |
| `harmonize_labels` | `harmonize.harmonize_from_diarization:307`; overlap from `occupancy.spans_from_diarization:87` + `adaptive/backends.overlap_track_from_spans:237` | exists |
| `speaker_count_gate` | `speaker_identity.speaker_count_posterior:117`, `build_speaker_identity:495` | exists; publishes a vote share as a posterior (F-179) |
| `novelty_track` | nearest: `adaptive/identity_repair.change_point_trajectory:53`, `_agglomerative_cosine:125`; `embeddings.calibrate_cosine_uncertainty:611`; `utils/tasks/embedding_distribution.select_dominant_vectors:381` | **does not exist** as a product |
| `off_target_fold`, `off_target_gate` | — | **do not exist** — **G4** |
| `scene_quality_frames` | `scene_quality/brouhaha.py:220` via `stages.py` | exists |
| `quality_measures` | `quality.harvest_quality_measurements:243` | exists, and is the model for the rest |
| `degradation_gate` | `degradation.scene_degradation:129` | exists; no content or task input — F-169 |
| `defect_spans` | `level.clipped_fraction:209` | exists but unreachable |
| `trim_proposal`, `trim_gate` | — | **do not exist** — **G7** |
| `evidence_ledger` | `Estimate` at `utils/data_structures/estimate.py:28` | type exists, **zero consumers** — **G8** |
| `review_flag_gate` | nearest: `disagreements.build_disagreements_index:62`, `global_summary.compute_run_global_summary:226` | **does not exist** as a three-arm rule |
| `rank_undecided` | `adaptive/regions.propose_regions:11` + `adaptive/interventions.RULES:1142` | exists, partially — four of nine rules have no uncertainty term in their trigger |
| `stop_or_continue` | `rounds.assess_convergence:250` *and* `adaptive/loop.py:315-317` | exists **twice**, with different criteria |
| `narrow_input` | `adaptive/backends.py` crop paths | exists, uncached |

### 7.2 F-187: does the goal-driven graph need that edge?

**The finding, restated.** `speaker.py:549` reads
`pass_summary["background_mask"]["result"]["regions"]`. The producer, `BackgroundMask.to_json()`
(`background_mask.py:152-168`), emits **13 aggregate counter keys and no `regions`** — the per-region
table exists only in `to_rows()` → `L2/background_mask.parquet`. So `mask_regions` is `[]` on every
run, `attribution.target_activity_doubt` returns `(None, None)` for every bucket, and the region
`state` is always `None`. Three decisions therefore have never fired: the `target_free` clear
(`speaker.py:557`), the `_VOCAL_ACTIVITY` word-gate exemption (`speaker.py:562`, which is the entire
F-165 fix), and the `target_activity` voter (`speaker.py:606`) — the second of the two scored voters
the axis's own docstring declares. Corroborated in artifacts: `target_activity.parquet` is absent
from all three completed runs while the sibling `speaker_assignment.parquet` is present in all three,
and every `L2/background_mask.json` carries exactly 13 keys.

New, and not in the register: the only contract in the repo that declares a `regions` key is for a
`mask_introspection.json` artifact (`specs/20260728-221507-per-speaker-identity-scene/contracts/background-mask.md:67-80`),
and `grep -rn mask_introspection src/` returns nothing. **The consumer was written against a producer
that was never built.**

**The answer: neither. The graph needs something different, and the edge should be deleted rather
than wired.** Four reasons, in order of force.

**(a) The mask answers a different question than attribution needs.** `background_mask` reports
regions free of *target* activity, where what counts as target comes from `task.type` — in a
breathing task, speech detection is silent during the target event, so the polarity flips
(`doc.md:245-249`). Attribution needs "was any voice audible here", which is task-independent. Using
one for the other means a config key that describes the *task* silently changes who gets attributed.

**(b) `nontarget_active` cannot carry the meaning it is being asked to carry, and this is now
measured rather than open.** The register and `phase2-notes.md:111-125` record this as an open
question: `_classify_bucket` (`background_mask.py:256-264`) reaches the
`nontarget_confidence >= 0.5` test only after the bucket has passed `uncertainty <=
max_free_uncertainty` **and** `confidence <= free_at`, so the state means "the target is confidently
*absent*, and some non-speech source scored ≥ 0.5". The second conjunct is a `max` over every
non-speech category (`nontarget_confidence_by_bucket`, `background_mask.py:691-729`, excluding
`speech` alone), and 292 of the 527 labels in `data/audioset_source_map.json` map to `environment` —
also the map's `default`, and where `Silence` lands.

That was reasoned. It has now been **measured**, on the shipped artifacts of a completed run
(`artifacts/analyze_audio/streaming-audio-…_20260807-191739/`):

- three `nontarget_active` regions; `contains_nontarget_speech` is `False` for all three;
- normalised category mass in them: `src_environment` 0.736 / 0.734 / 0.877 against `src_people`
  0.048 / 0.003 / 0.004; `src_dominant` is `environment` in 25 of the 33 buckets and `speech` in the
  other 8, never `people`;
- the labels actually driving them are `Music 0.634` and `Television 0.422` (AST), `Music 0.860` and
  `Violin, fiddle 0.287` (YAMNet), and — in region `m7`, 10.56–11.26 s — **`Silence` at posterior
  0.674**.

So on real data the state fires on a television jingle and on YAMNet's `Silence` label. That is the
exact case `MASK_STATES`' own docstring (`background_mask.py:37-40`) says must read `target_free`.
Membership of `nontarget_active` in `speaker._VOCAL_ACTIVITY` is not merely unproven; it is
contradicted by every measured instance in this repo. Wiring F-187 through as-is would spare
precisely those buckets — 33 of the 34 in the cost table at `phase2-notes.md:104-109`.

**(c) The threshold is a code literal.** `nontarget_active_confidence` is read with a `0.5` fallback
at `background_mask.py:297`, and the shipped profile
`data/detection_margin/2026-07-29.json` does not contain the key — nor does the Python default
profile at `calibration.py:217-239`. So the operative value lives in Python, against CLAUDE.md's rule
that thresholds live in `data/` with a written derivation.

**(d) It is one L2 decision consumed as evidence by another.** The mask state is a thresholded
verdict. Attribution consuming it means a decision feeds a decision, and when the wire breaks the
failure is invisible, because a missing region is indistinguishable from a bucket with no vocal
activity. That indistinguishability is why this survived the life of the feature.

**What replaces it.** `taxonomy_fold` emits `vocal_spans` and `taxonomy_track`: a word-independent,
task-independent, four-way statement built from frame posteriors, sound events and periodicity.
Attribution gates on vocal activity, not on words. Consequences:

- The **word gate disappears entirely**, so F-165 stops being a defect to fix and becomes a defect
  that cannot be expressed: there is no rule anywhere that reads word absence as speech absence.
  What survives is the *finding* — word absence is not speech absence — now enforced structurally.
- The `target_free` clear, the `_VOCAL_ACTIVITY` exemption and the `target_activity` voter are all
  **deleted**, along with the `mask.regions` read. Measured cost of deleting them: zero, because none
  has ever fired.
- **Trim regions stop being sourced from `target_free`.** The prior design named those spans as the
  only source of trim (`specs/20260816-143540-triage-graph/design.md:221-223`); given (b), a trim
  built on them would propose cutting a region because YAMNet scored `Silence` at 0.674, and — given
  F-168, where a cry maps to `people` — would propose cutting an infant's vocalization as background.
  `trim_proposal` takes four typed span sets instead, each carrying its reason.
- **The background mask survives, with a narrower consumer set.** It remains the right product for
  task-match and background-scene characterisation, which is what it was designed for. It stops being
  an input to identity.

**One live consumer must be re-examined, and it is not in the register.** `nontarget_active` *is*
read in production, by a path that goes through the parquet rather than the pass summary:
`rounds.py:53` defines `_CONTRADICTING_STATES = ("target_free", "nontarget_active")`, and
`rounds.regional_weights` (`rounds.py:90-94`) cuts the weight of any signal that claimed a speaker
overlapping such a region, wired from `scripts/analyze_audio.py:945-966` →
`fuse.py:938,953-955`. Combined with the measurement in (b), that means **a bucket where YAMNet
reported `Silence` at 0.674 withdraws trust from the signals that claimed a speaker there.** F-187
says the region table never reaches the code that reads it; that is true of the pass-summary path and
false of this one. This should be filed.

> **The contents are now measured, and only the measurement carries forward.** PR #574 measured all 33
> `nontarget_active` buckets in that run against the run's own YAMNet windows, per-bucket so a 0.96 s
> window straddling a region edge cannot import content from outside it: **32 of 33 carry a non-vocal
> dominant label** — 26 `Music` at a median 0.86-0.90, 6 `Silence` — and 16 have vocal mass below 0.01.
>
> What that says about `rounds.regional_weights`, `_VOCAL_ACTIVITY` and the pass-summary path is
> **historical**. Those are the round-based pipeline this design replaces; a fix to how they read the
> mask is not work this branch wants, and #574's recommendation to gate `_VOCAL_ACTIVITY` on
> `contains_nontarget_speech` should be read as a note on superseded code rather than a task.
>
> Two things do carry forward, because they are properties of the mask and the audio rather than of any
> consumer:
>
> 1. **`nontarget_active` is not a vocal-activity signal.** It is reachable with the target confidently
>    absent and nothing but environmental mass present — 292 of 527 labels in the source map go to
>    `environment`, which is also its default and where `Silence` lands. Any node in *this* design that
>    wants "a vocalization happened here" must not read that state, and `contains_nontarget_speech`
>    already exists to say the narrower thing.
> 2. **The mask's region boundaries land inside events.** Region `0.0-2.0` ends on a speech onset — its
>    last buckets read `Speech` 0.466 then 0.878. That is a span-boundary defect, and span boundaries are
>    exactly what branch 1 asks DSP to fix. It belongs to `span_reconfirm`'s problem statement, not to a
>    voting-weight rule.

### 7.3 The refiner-only machinery: what survives as a node, what was scaffolding

**`VoteStore` round accumulation — scaffolding, and less of it than advertised.** `VoteStore`
(`adaptive/belief.py:253`) is append-only and cumulative, not partitioned by round. The only
round-keyed structure is `_round_added` (`belief.py:273`, written `:491`, read `:716`), and its only
caller is the per-round artifact writer `adaptive/loop.py:853`, which writes
`rounds/<k>/derivatives/votes_added.parquet`. The round index slices an output file; it drives no
decision. Attenuation records a round inside vote provenance (`belief.py:562-575`) and the
idempotence guard at `:696-712` matches on `reason` while ignoring the round. The genuinely
cross-round state is `BeliefState.history` (`belief.py:1040-1042,1074-1075`), and its single consumer
reads only the last two entries (`adaptive/convergence.py:63-77`). **Survives as a node:** nothing.
The ledger in this design carries the same information as an explicit product with one producer.

**The intervention rules — partially real, partially scaffolding.** Nine rules at
`adaptive/interventions.py:1142-1226`, dispatched by `policy.plan_round` (`policy.py:165-232`).
Findings that bear on the design:

- **Four of nine triggers contain no uncertainty term.** S1 keys on how many passes exist
  (`interventions.py:229-234`); I1 and I2 on "not done yet" (`:766-772`, `:823-829`); I4 on region
  co-location (`:1059-1072`). Only P3, C9, U1, U2 and P2 read a measured quantity. A loop whose
  planner is uncertainty-driven cannot admit those four as they stand.
- **I4 is effectively unreachable.** Its guard (`:1075-1084`) demands `pyannote.audio` and an HF
  token and `input_audio`, but its execute path only calls `overlap_track_from_spans` over
  already-persisted diarization spans (`:1102`) — no model, no token, no audio. `:1088` imports
  `get_stream_wav` and never calls it.
- **I1/I2's guard checks a path the loader does not use.** The guard inspects
  `run_dir/<stream>/embeddings` (`:776`); the loader resolves `layout.perturbation_dir` →
  `run_dir/L1/raw/embeddings` (`:732`, `layout.py:92-103`). So the "stored embeddings exist" escape
  is always false.
- The module docstring at `:22-23` says P2 is "still deferred"; P2 has a full trigger/guard/execute
  and is enabled at `default.yaml:507`. Stale prose.

**Survives as nodes:** the *underlying measurements* five rules wrap — re-ASR of a region, finer
Brouhaha frames, corroboration pooling, identity repair — each of which is already a pure function of
one evidence snapshot. **Scaffolding:** the proposal/admission machinery, the priority ranking (whose
`gain` is not one quantity across rules — F-158), and the eight ad-hoc `ctx` dedup caches
(`interventions.py:232,294,604,725,770,827`), each of which becomes an explicit input port or ceases
to exist.

**`adaptive/backends.py` — scaffolding, and a liability.** It declares itself a pure failure-envelope
gateway (`backends.py:1-20`), and the math is indeed delegated. What is duplicated is the
*invocation*: five call sequences that also exist in `stages.py`/`compute.py`, but **without
`run_task_cached`, without a cache key, and without provenance**.

| capability | refiner path | cached path |
| --- | --- | --- |
| ASR | `backends.py:93` `transcribe_audios(...)`, plus a process-global `_ASR_PIPELINE_CACHE` at `:28` | `stages.py:347-357` behind `run_task_cached` with a cache key and provenance |
| word-leaf flattening | `backends.py:357-363` | `stages.py:815-826`, whose comment at `:821` acknowledges the duplication; `adaptive/fusion.iter_word_leaves` is a third |
| embeddings | `backends.py:174-176` | `compute.py:134-140` |
| Brouhaha frames | `backends.py:225` | `compute.py:270,307,316` |
| diarizer spans → overlap | `backends.py:263-265` | `fuse.py:1262`, `speaker.py:296` |
| alignment | `backends.py:322-343` | `stages.py:405-437` behind `run_alignment_cached` |

In this design the round loop re-enters the *same tasks* over a narrowed input, so all six collapse
into the existing cached tasks. What is genuinely refiner-specific is the `(result, reason)` envelope
and the crop/offset bookkeeping — one task, `narrow_input`.

**`adaptive/provenance.py` — dead, confirmed.** Zero importers in `src/senselab/` or `scripts/`; zero
references to `RevisionRecord`, `classify_resolution`, `revision_log_entry` or `RESOLUTION_KINDS`
outside the module. Its *concept* is hand-rolled where it is needed: the self-confirmation guard at
`rounds.py:318-320`, and the undifferentiated delta it warns against at `convergence.py:75-77`.
`build_convergence_report`'s `per_quantity` seam (`convergence.py:148`) is never filled, so
`report["per_quantity"]` is `None` on every run. **Falls away.**

### 7.4 What else falls away

- **The four-chain decomposition, as a plan.** Chain 1 (window-embed-and-cluster) does not run in
  production: `derive_window_clusters` (`speech_presence_link.py:377`) has no caller, and
  `cluster_pass_speakers` is reachable only from inside it. Lifting a chain that is dead is not a
  lift.
- **Three whole modules with no importer:** `sources.py`, `foreground.py`, `measurements.py`. Plus
  `quality_series` (`quality.py:326`), `measure_variant` (`level.py:292`) and most of `level.py`,
  `signal.measurement` (`signal.py:115`), `declared_resolution_s` (`resolution.py:55`),
  `fuse_rounds` (`fuse.py:1185`), `resolve_influence` (`influence.py:121`), `axes.axis()`
  (`axes.py:405`), `occupancy.occupancy` (`occupancy.py:133`), `per_speaker_presence`
  (`identity_binding.py:98`), `target_spans_from_evidence` (`background_mask.py:732`).
- **The twelve dead config fields** (`run_config_liveness_test.py:65-88`). Under this design a
  parameter port either has a producer key or the task does not compile.
- **`_speech_window_mask`'s YAMNet veto** (`compute.py:893`), replaced by `vocal_spans`.
- **The `mask.regions` read and its three decisions** (§7.2).

---

## 8. What must be measured before building this

Each item: the question, why the design rests on it, and how to measure it. Nothing here is a
formality — items M1, M2 and M4 can each change the shape of the graph.

**M1 — `nontarget_active` semantics. Largely answered; finish it.** §7.2(b) reports the measurement
on one run: three regions, driven by `Music`, `Television` and `Silence`, with `src_people` at most
0.048. That is enough to reject the state as vocal evidence, but it is one run. **How:** repeat over
every completed run and every `nontarget_active` region; hand-label each for whether an audible human
voice is present; report the fraction. **Decision rule, stated before the measurement:** below
two-thirds, the state is not vocal evidence and `_VOCAL_ACTIVITY` must not contain it — which is the
design's assumption. Separately, recompute `nontarget_confidence` with `environment` excluded from
the `max` and report how many regions survive; and audit the live
`rounds._CONTRADICTING_STATES` path (§7.2, last paragraph) for how much signal weight it has been
withdrawing on `Silence`.

**M2 — does a frame posterior actually dip in a short gap?** The design's entire timing story assumes
raw frame scores carry information that segmentation smooths away. **How:** on clips with annotated
inter-turn gaps, extract raw `segmentation-3.0` scores through the frame-posterior path
(`frame_posteriors.py:241`) rather than the segment pipeline, and report collar-based event F1
(onset collar 200 ms, offset collar max(200 ms, 20% of duration)) and PSDS1 — not boundary MAE, which
is what hid the problem. If the posterior does not dip, G2 needs a different model, not a different
post-processor.

**M3 — can a frame-level sound-event model separate speech from babble well enough to be an
off-target voter?** **How:** clips with and without background talkers, AUROC for the
`Babble`/`Crowd`/`Conversation`/`Chatter` sub-tree against a hand-labelled per-window "another human
is audible" target. Compare against the current window-level AST/YAMNet baseline on the same clips,
since G3's cost is a new model and its benefit must exceed that baseline.

**M4 — the non-lexical class has no labeled data in this repo.** Without it, `taxonomy_fold`'s
four-way fold has no fitted voter weights and `content_gate` has no fitted bands. **How:** assemble
clips with spans for cry, cough, laugh, breath and groan, from the study's own recordings. Until they
exist, every `content` estimate must carry `population: unvalidated` and derive `n_evidence` from
voter count alone — and the config keys must say `derivation: unfitted`.

**M5 — does the dominant-cluster anchor survive a talkative intruder?** The mis-anchor risk is the
main threat to D1. **How:** synthetic mixes at controlled intruder-duration shares (5 / 10 / 25 /
50%), reporting recall of "at least one non-dominant voice present" and, separately, whether the
dominant cluster is still the intended target. The second number is what decides whether the
published claim can ever be stronger than "more than one voice".

**M6 — the embedding window, at both values.** The config sets 0.5 s / 0.25 s
(`default.yaml:179-181`), wired through `run_config.py:478-479` and
`scripts/analyze_audio.py:747-748`. The function defaults are 2.0 s / 1.0 s (`compute.py:101-102`,
`:535-536`). Any caller that reaches `compute` without the config — a lifted chain, a notebook, a
test — silently gets a 4× wider window. **How:** run the off-target voters at both settings on the
same clips and report brief-intrusion recall at each. Then delete whichever default is wrong rather
than keeping two.

**M7 — the shrinkage pseudo-count, per answer.** `Estimate.value` interpolates `raw` toward `prior`
with a per-quantity pseudo-count, and nothing has fitted one. **How:** it needs outcomes. Until they
exist, `pseudo_count` ships as `derivation: unfitted` and is written into every run's output, so it
appears in each artifact rather than only in a file nobody rereads.

**M8 — the review bands, and the corpus that would make a classifier possible.** **How:** have a
human review a stratified sample and record the decision *and its reason*. That record is
simultaneously the calibration set for Arm 1's bands and the labeled corpus a fitted flag would need.

**M9 — enhancement's effect on word boundaries.** No primary source quantifies it, so it is a
measurement to make rather than a number to cite. **How:** align the same transcript against raw and
enhanced audio and report the per-word boundary shift distribution. The practical rule until then:
align on the same audio the timings are reported from, and keep raw-vs-enhanced boundary
disagreement as an uncertainty signal.

**M10 — that a narrowed region actually misses the whole-file cache entry.** The design claims this
needs no new key field. **How:** slice an `Audio`, take `audio_signature` of both, confirm they
differ, and confirm the offsets are restored on the way out. Cheap, and it is load-bearing for the
loop's cost model.

---

## 9. Where the existing specs are wrong

Recorded here because several of these are cited elsewhere as settled.

1. **"Chains are pure over plain inputs"** (`remediation-decomposition.md:99`, `:110`, `:143`).
   Chain 3 is; chain 1 is not. `cluster_pass_speakers` (`embeddings.py:97`) takes a caller-supplied
   `failures` dict and **writes to it** at `:162,190,244,254,299` — a mutable out-parameter used as an
   error channel. And `_speech_window_mask` (`compute.py:893`) takes `pass_summary: dict[str, Any]`,
   the untyped bag. Also, chain 1 has no production caller at all (item 3 below), so "entangled only
   by being called inside `harvest_pass`" is false in both halves: it is not entangled and it is not
   called.

2. **`speech_presence_link.py:444`'s in-file comment** — "`derive_window_clusters` below stays — it
   is what `compute.harvest_pass` calls" — is false. `grep derive_window_clusters compute.py` returns
   nothing.

3. **The four-chain plan treats chain 1 as live.** It is dead: `derive_window_clusters`
   (`speech_presence_link.py:377`) has no caller, `cluster_pass_speakers` is reachable only from
   inside it, and the `embedding_silhouette` signal that `link_speech_presence`'s docstring at
   `:513` says is "derived here" is never produced.

4. **`design.md:47` — "In the current codebase nothing does this"** (carry an evidence count).
   Accurate about behaviour, misleading about the tree: the `Estimate` type now exists at
   `utils/data_structures/estimate.py:28`, with `value`/`raw`/`n_evidence`/`prior`, a
   `_raw_matches_evidence` validator at `:99` and `no_evidence()` at `:137`. It has zero consumers.
   A type with no producer is a different problem from a missing type, and the difference matters
   when scoping Phase 3.

5. **F-187's scope is narrower than the register states.** "The mask's regions never reach the code
   that reads them" is true of the pass-summary path and false of the parquet path:
   `rounds._CONTRADICTING_STATES` (`rounds.py:53`) reads the region states in production via
   `scripts/analyze_audio.py:945-966` → `fuse.py:938,953-955`. The region table has a live consumer;
   it is the *speaker* path that is broken.

6. **The `nontarget_active` question is no longer open.** Both `phase2-notes.md:124` and the register
   record "nobody has measured what those 33 contain". §7.2(b) measures it: `Music`, `Television`,
   and `Silence` at 0.674, with `src_people` at most 0.048. The state is not vocal evidence.

7. **`SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md` §3 recommends per-axis grids** ("Better still: per-axis
   grids", `:119-130`). That recommendation was implemented and then measured to be a defect: the
   four axes landed on four spacings sharing **zero** bucket keys, so every cross-axis coupling ran
   and did nothing (`doc.md:19-32`). The same file's context anchors are stale in two more ways — it
   cites `presence.py`, which is now `speech_presence.py`, and a `BucketGrid` default of 0.5 s / 0.5 s,
   which is now 0.1 s / 0.1 s (`default.yaml:109-111`).

8. **`SPEECH_DETECTION_SOTA_REVIEW_2026.md:273-274,303` says the 0.5 s embedding window is "already
   the default after this branch's tuning".** It is the *configured* value
   (`default.yaml:179-181`); the *function* default is 2.0 s / 1.0 s (`compute.py:101-102`,
   `:535-536`). F-171 cites the function default as if it were what runs, and the review cites the
   configured value as if it were the default. Both are half right, and a lifted chain gets the
   function default — M6.

9. **`interventions.py:22-23` says P2 is "still deferred".** P2 has a full trigger, guard and execute
   and is enabled at `default.yaml:507`.

10. **`calibration.py`'s "Stdlib-only; safe to import anywhere"** was falsified by its own
    module-level `axes` import (`calibration.py:46`) and has been corrected — noted here only because
    it is the same class of error as items 1, 2 and 9: a comment asserting a property the code
    contradicts, in a file nobody re-reads while editing.

---

## 10. Scope

**In:** the port graph, the question ordering, the loop and its exit criteria, the mapping to
existing code, the gap list, and the measurements above.

**Out:** any implementation. No code under `src/` changes on the strength of this document. Model
pluggability, lifespan validation, a fitted review-flag classifier, and suppression depth are all
out — the gap list names each rather than closing it.

**Not a new configuration surface.** One versioned YAML remains the only knob store: no node
registry, no per-task override file, no CLI flags beyond the existing two arguments. The five config
sections this design needs and does not have — `taxonomy:`, `pii:`, `evidence:`, `off_target:`, and a
`nontarget_active_confidence` entry in the detection-margin profile — are keys in that same file,
each shipping with a `derivation` block, and `derivation: unfitted` is an acceptable value. The
seventy deleted CLI flags are the reason the rule is stated as a rule: a node registry is a place for
them to reappear wearing a different name.
