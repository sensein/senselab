# Layered uncertainty architecture: signals, harmonization, mapping, rounds, categories

Authoritative design for the audio-analysis uncertainty workflow. Supersedes the ad-hoc
arrangement in which one layer both measured and interpreted. Companion documents:
`l1-signal-contract.md` (the principle and its measured symptoms) and
`l1-post-processing-register.md` (the 24 interpretation sites to be relocated).

Decisions recorded here were made 2026-07-31 and are marked **D-n**. They are decisions, not
inferences: where a decision closed off an alternative, the alternative is named so a later reader
can see what was chosen against.

## The root principle

**L1 reports what a tool measured. L2 decides what it means.**

Every defect found in this feature traced to a single violation: a reduction applied at L1 that
saturated independently of its input. Six instances, all measured, are catalogued in
`l1-signal-contract.md`. The clearest is scene quality — `clip((25 − snr_db)/20, 0, 1)` returned
`0.0` in every bucket of every recording, because clean speech measures 60–70 dB SNR against a
25 dB anchor. Probing brouhaha directly showed the heads span −5 to 70 dB and discriminate speech
from silence by +0.98. **The measurement was never the problem; the clamp on top of it destroyed a
working signal.**

The layering exists so that a saturating choice is visible as a fusion decision in L2 rather than
baked irretrievably into recorded data.

---

## Layer 1 — signals

Raw measurements, native units, native resolution. No thresholds, no ranks, no clamps, no
reductions across a dimension the tool reported separately. Every signal emits through
`signal.measurement(...)` so units, model, revision, resolution, window and any reduction travel
with the value.

| signal | model | reports | units | window / hop |
|---|---|---|---|---|
| `segmentation_activations` | pyannote/segmentation-3.0 | per-speaker activation matrix `(frames × speakers)` | probability | 61.9 ms / 16.9 ms |
| `brouhaha_vad` | pyannote/brouhaha ch0 | speech probability | probability | 61.9 ms / 16.9 ms |
| `brouhaha_snr_db` | ch1 | SNR | dB | 61.9 ms / 16.9 ms |
| `brouhaha_c50_db` | ch2 | C50 clarity | dB | 61.9 ms / 16.9 ms |
| `diar_*` (N models) | pyannote community-1, sortformer, + PR #537 | `(start, end, speaker_label)` | seconds | segment |
| `emb_ecapa`, `emb_resnet` | speechbrain ECAPA / ResNet | speaker embedding vector | vector | **2.0 s / 50 ms** |
| `asr_crisperwhisper` | CrisperWhisper2.0_turbo | text, word spans, `avg_logprob`, `no_speech_prob`, token entropy | seconds / log-prob / nats | word / ~30 s segment |
| `asr_qwen3` | Qwen/Qwen3-ASR-1.7B | text, word spans | seconds | word |
| `asr_canary_qwen` | nvidia/canary-qwen-2.5b | text → word spans | seconds | word |
| `ast_posteriors` | MIT/ast-finetuned-audioset | 527 label scores | probability | 10.24 s / 10.24 s |
| `yamnet_posteriors` | google/yamnet | 521 label scores | probability | 0.96 s / 0.48 s |
| `ppg_posteriors` | ppgs | per-frame phoneme posterior incl. `<silent>` | probability | native |
| `hnr_db` | openSMILE `HNRdBACF_sma3nz` | harmonics-to-noise | dB | 60 ms / 10 ms |
| `lufs` | pyloudnorm BS.1770 | gated loudness | LUFS | 400 ms / 100 ms |
| `flux_above_floor_db` | spectral flux vs measured band floor | spectral change above noise | dB | 20 ms / 10 ms |
| `noise_floor_bands` | `noise_floor.py` | per-band floor + ECMA-74 prominence | dB | 100 ms |
| `snr_brouhaha_db`, `c50_brouhaha_db`, `snr_spectral_gating_db`, `snr_peak_db`, `rolloff_95_hz`, `proportion_clipped`, `rms` | brouhaha / senselab DSP / torch.stft | scene quality measurements | dB / hertz / proportion / arbitrary | 0.5 s / 0.25 s |

### Decisions

**D-1. One aligner for the two text-only ASR paths.** `asr_qwen3` and `asr_canary_qwen` both use
Qwen3-ForcedAligner-0.6B; canary moves off MMS. *Why:* word-boundary differences between two
models must reflect the models, not two different aligners. `asr_crisperwhisper` keeps its native
timings — it has them, and forcing an aligner over a model that already reports boundaries would
discard the more direct measurement.

**D-2. Embeddings at 2.0 s window / 50 ms hop** (was 1.0 s / 0.5 s). *Why:* change-point
localisation was floored at ±0.5 s while `segmentation_activations` runs at 16.9 ms. *Consequence
that must not be forgotten:* adjacent windows then overlap 97.5%, so localisation improves ~10×
while the number of *independent* observations does not. Any estimator treating windows as
independent samples would overstate its evidence by ~40×.

**D-3. Drop the uncalibrated openSMILE signals.** `Loudness_sma3` and `spectralFlux_sma3` are
replaced by `lufs` (BS.1770, absolute) and `flux_above_floor_db` (referenced to the measured
per-band noise floor). *Why:* both had `arbitrary` units — no absolute reference — which is
precisely what forced the per-pass percentile ranking that broke them (~10% of frames pinned at 0
and ~25% at 1.0 *by construction*, independent of the audio). `hnr_db` stays: a ratio in dB is
already absolute and its 2–10 dB anchors are defensible.

**D-4. Keep all three ASR models**, and design harmonization to be model-count agnostic, since
PR #537 adds diarizers.

**D-5. Emit `segmentation_activations` with channels intact.** *Why:* it is the only signal that
can distinguish "two speakers talking simultaneously" from "uncertain which of two speakers", and
the collapse discarded the per-speaker structure the speaker category needs.

*Measured mechanism, which corrects an earlier account in this document.* The saturation was not
caused by the choice of reduction (`max` vs noisy-or) but by **misidentifying the output format**.
Probing the model directly:

```
problem: MONO_LABEL_CLASSIFICATION   powerset: True
classes: ['speaker#1', 'speaker#2', 'speaker#3']     output: (471, 3)
```

pyannote 4.x converts powerset → per-speaker activations *before returning*, so the columns are
speakers. With one speaker fully active those rows sum to exactly 1.0, so the row-sum heuristic
concluded "powerset" and computed `1 − data[:, 0]` — treating **speaker#1** as the no-speaker
class. On 4 s of speech followed by 4 s of digital silence that read `1.0000` in **100% of
frames**. The fix is to read the declaration against the output width (powerset over 3 speakers is
7 columns, per-speaker is 3), never row sums. After it, on the same input:

```
pooled P(speech): speech half 1.0000   silence half 0.0636   discrimination +0.9364
exactly-1.0000 frames: 53% (i.e. the speech half)   per-channel: [0, 1, 0]
```

*Consequence for the layering:* the pooled value is no longer a stored field. Keeping a collapse
next to the matrix it came from lets the two disagree, and consumers read the stored one — which
is how a reduction returning 1.0000 everywhere survived unnoticed.

Deferred at L1: a dedicated non-target measurement (for speech tasks, non-target reduces to
non-speech, making the mask a layer-2 derived quantity), and sampling/dropout uncertainty (tracked
separately; today only models that happen to expose internal doubt provide any).

---

## Layer 2 — post-processing

Two kinds of thing, kept separate because they fail differently. Harmonization can be *wrong*
(a mis-matched label); joint estimation can be *unsupported* (a claim no signal corroborates).

### 2a. Harmonization — same information, common frame

| # | harmonized signal | from | method | own uncertainty |
|---|---|---|---|---|
| H1 | common temporal lattice | all signals | `resample_series`: mean for rates, hold for coarse tags | interpolation error where source is coarser than target |
| H2 | **common speaker space** | diar labels ×N, embeddings | see D-6 | **assignment uncertainty** |
| H3 | aligned transcripts | ASR word spans ×3 | align transcripts to each other, not only to audio | alignment gap / insertion rate |
| H4 | common scene ontology | AST 527, YAMNet 521 | AudioSet ontology map (checked-in category JSON) | label-set divergence |
| H5 | absolute acoustic scale | `lufs`, `flux_above_floor_db`, `hnr_db` | reference to measured floor | floor-estimate uncertainty |

**Harmonization is an estimation step and carries its own uncertainty.** Each diarizer's labels are
arbitrary, so any cross-model identity comparison first *guesses* that `spk0` and `SPEAKER_01`
denote the same person. A guess propagated as fact makes two models that were never correctly
compared read as disagreeing — which is exactly how speaker uncertainty stayed high in regions
where per-speaker presence was unambiguous, the observation that started this work.

**D-5 addendum — keeping the channels was not enough; the stitch was averaging strangers.**
`segmentation-3.0` is defined on 10 s chunks, so L1 slides a bounded window and reconstructs one
timeline (stitching sliding windows back together is explicitly not post-processing). But the
overlap-average matched chunks **by column index**, and a speaker-segmentation model assigns its
channels arbitrarily *per inference* — the same permutation-arbitrariness that makes J4 a joint
space and J1 answerable. A speaker who is column 0 in one chunk and column 1 in the next was
therefore split into two half-strength channels across the whole 2 s overlap, which reads
downstream as two speakers each half-present: overlapped speech that never occurred, on 25% of
every recording longer than one chunk. pyannote's own pipeline resolves this permutation before
aggregating; `stitch_frames` now does the same, matching each chunk to the already-stitched
timeline (not to its predecessor, which would let a permutation drift chunk by chunk). It is
opt-in, because Brouhaha's `[vad, snr, c50]` columns carry fixed meaning and permuting them would
swap unrelated physical quantities.

*Evidence:* deterministic and model-free — a two-chunk stitch with a known relabelling produces two
0.5 channels without alignment and one full channel with it. That shows the code had no defence; it
does not measure how often a flip occurs on real audio, which needs a GPU run not yet done.

**D-6. Match speaker labels by both methods, and treat their disagreement as the uncertainty.**
Temporal-overlap assignment (Hungarian on co-occurrence) and embedding-centroid similarity per
label are computed independently; where they agree the assignment is confident, and where they
disagree that disagreement *is* the assignment uncertainty. *Why:* either method alone yields a
point assignment with no way to express doubt about itself.

### 2b. Joint estimation — signals that exist only by combining

| # | new signal | inputs | serves |
|---|---|---|---|
| J1 | overlap count | `segmentation_activations`, channels intact | speaker |
| J2 | speaker change points | embeddings @ 50 ms hop, adjacent-window distance | speaker |
| J3 | speaker-count posterior | diar counts, embedding clusters, invariance probes | speaker |
| J4 | per-speaker presence | H2 speaker space ⋈ `segmentation_activations` | speech-presence, speaker |
| J5 | target-free mask | speech-presence + H4 scene composition + noise floor | background-mask |
| J6 | stationary background sources | per-band floor + ECMA-74 prominence ≥ 9 dB | background-mask |
| J7 | phoneme-vs-transcript agreement | `ppg_posteriors` vs aligned words (H3) | ASR |
| J8 | cross-pass stability | same signal, raw vs enhanced | weights |
| J9 | invariance under gain / shift / DC offset | re-run under output-preserving perturbation | weights |

J8 and J9 serve no category: they set per-signal **weights** (stability × support). Keeping "how
much do we trust this signal" separate from "what does this signal say" is what allows a locally
wrong diarizer to be discounted regionally rather than silenced globally.

**D-7. J4 is a joint space, not a one-directional inheritance.** `segmentation_activations`
channels are permutation-arbitrary within a window, so channel *k* is not a stable speaker across
the recording; and the harmonized speaker space needs frame-level evidence to place its speakers.
The two constrain each other — channels inform who `S_k` is, `S_k` informs which channel is whom —
and the mapping is resolved by L2 rounds. *Consequence:* consistency between them is something
rounds **measure**, and its instability *is* part of the speaker uncertainty, rather than J4
inheriting H2's uncertainty as an external input.

---

## Layer 3 — mapping functions to uncertainty estimates

`statistics.py` supplies the estimators. What each signal needs is a **link function** carrying its
native measurement into a distribution over its category's outcome space, because entropy requires
a distribution.

### Outcome spaces

| category | outcome space | k |
|---|---|---|
| speech-presence | {speech, no-speech} | 2 |
| speaker | per-speaker {active, inactive}; count {0…K}; identity {`S_0`…`S_n`} given one active | 2 / K+1 / n |
| ASR | distinct normalised token sequences for the span | #distinct |
| background-mask | {target_active, target_free, nontarget_active, indeterminate} | 4 |

Speaker is deliberately three coupled quantities rather than one scalar, so that number of
speakers and their presence are both represented.

### Link functions

| signal type | examples | link | fitted |
|---|---|---|---|
| probability | `brouhaha_vad`, scene posteriors, `ppg_posteriors` | identity | no — calibrated by training |
| per-speaker activation | `segmentation_activations` | pooled `1 − Π(1 − p_k)`; per-speaker `p_k` | no |
| dB | `lufs`, `hnr_db`, `snr_brouhaha_db`, `flux_above_floor_db` | logistic `σ((x − x₀)/s)` | **yes** |
| cosine distance | `emb_ecapa`, `emb_resnet` | logistic on distance | **yes** (sequential band exists) |
| categorical posterior | AST / YAMNet over an ontology subset | sum of mass over subset | no |
| hard segment / word | diar segments, ASR word overlap | see D-9 | — |
| log-probability | `avg_logprob` | `exp(avg_logprob)` | no |
| nats | `token_entropy` | `÷ log|vocab|` | no |

**Noisy-or is correct at L2 and was wrong at L1.** `1 − Π(1 − p_k)` is the right probability
calculus for "any speaker active", stated as a fusion model over channels that still exist. At L1
it *replaced* the channels, which is what pinned the value at 1.0000 and destroyed the per-speaker
information.

### The decomposition

- `confidence` — `P(proposition)`, weighted vote share.
- `uncertainty` — normalised Shannon entropy over the outcome space (÷ `log k`).
- `epistemic_uncertainty` — `H(mean) − mean(H)`: disagreement *between* signals, the reducible part
  rounds can act on. `0 ≤ epistemic ≤ total` by Jensen.
- `variability` — sample std in native units (dB, cosine, seconds), never rescaled.

For ASR the decomposition is unusually clean: epistemic is cross-model transcript disagreement,
aleatoric is each model's own token entropy. A round can resolve the former and cannot touch the
latter.

**D-8. Fitted links calibrate against synthetic reference with known parameters.** TTS-generated
material, extended with SPARC reconstruction where appropriate. *Why not cross-model consensus as
pseudo-reference:* it would bake the ensemble's shared blind spots into the calibration and then
report the result as measured. *Availability:* SPARC is not yet in the environment — it needs
subprocess-venv isolation (can share with `ppgs`). Fitting therefore starts TTS-only, and the
reference set must be extensible rather than assuming both sources from the outset.

**D-9. Hard voters are softened by their measured reliability**: a signal with measured stability
`w` contributes `(w, 1 − w)` rather than `(1, 0)`. *Why:* a degenerate distribution has zero
entropy, so a hard voter could only ever inflate epistemic uncertainty and never share aleatoric
doubt — it would be incapable of being uncertain. *Acknowledged tension:* this converts a
reliability weight into a probability, and those are different objects; accepted for now, to be
revisited once perturbation-based reliability has been measured across more signals.

---

## Layer 4 — convergence rules and processes

### Round structure

**Round 0** — fuse all L1 signals with global weights (stability × support); emit all categories.

**Round n > 0** —

1. **Select regions by *epistemic* uncertainty, not total.** A region whose doubt is
   aleatoric-dominated cannot be improved by more of the same measurement; sending work there is
   the exact failure the decomposition exists to prevent.
2. **Choose an action, which must be one of two kinds.** Rounds exist for **disambiguation and
   confirmation**; a region with neither trigger gets no action even at high uncertainty. This is
   what stops the loop performing open-ended refinement.

   | kind | trigger | action |
   |---|---|---|
   | disambiguation | competing hypotheses coexist — 2 vs 3 speakers, two transcript variants, which `S_k` owns a channel | re-run a signal at tighter window/hop, or add a model, targeted at the discriminating question |
   | confirmation | high confidence on thin support — few signals, or all correlated | run an *independent* signal that could refute it |

3. **Re-fuse** with regional weights (mask contradiction discounts a signal locally, scaled by mask
   confidence).
4. **Check convergence.**

**D-10. A round's action may re-run L1 at a tighter window or hop.** An estimate that improves a
signal is permitted to re-measure it, not merely re-weight it.

**D-11. A flagged region is re-examined for all categories, not only the one that flagged it.**
*Why:* the categories are coupled — a speaker ambiguity is frequently a presence ambiguity — and
D-7 makes speaker and presence explicitly joint. Costs more per region; the alternative would
re-measure a region for one category while leaving a known-doubtful neighbour category stale.

### Convergence criteria

All four, not stability alone:

| # | criterion | why not sufficient alone |
|---|---|---|
| C1 | epistemic uncertainty stops falling | total uncertainty can plateau while reducible doubt remains |
| C2 | the `S_k` ↔ activation-channel assignment is stable | D-7's joint space converging; numbers can settle while the assignment still flips |
| C3 | no bucket went unmeasured → measured | new coverage is progress; counting it as stability would stop the loop exactly when it began working |
| C4 | no region has an untried available action | otherwise "converged" means "silently ran out of ideas" |

**D-12. Stopping: diminishing change, or cycle detection, capped at 10 rounds (default).** Cycle
detection is required separately from slow convergence: with D-7's mutual influence between `S_k`
and activation channels, an oscillation A→B→A→B is plausible and would otherwise consume the full
budget while reporting movement every round.

### Two guards

**Self-confirmation.** `adaptive/influence.py` and `adaptive/provenance.py` hold the principle:
uncertainty falling *because a value was overwritten* is not a confidence gain. Extended to rounds
— if an action replaced a signal's value and uncertainty then fell, that drop does not count
toward C1. Only a fall caused by an independent measurement agreeing counts.

**Divergence is a legitimate outcome.** A confirmation action that refutes a claim *should* raise
uncertainty. Convergence therefore cannot be defined as monotone decrease: that would bias the loop
toward ratifying whatever round 0 said, and would leave the confirmation half of the design unable
to change anything.

---

## Layer 5 — the uncertainty categories

Renamed to name the question rather than the abstraction.

| category | was | proposition | resolution | outputs |
|---|---|---|---|---|
| **speech-presence** | `presence` | speech was present here | 20 ms | `confidence`, `uncertainty`, `epistemic_uncertainty`, `variability`, per-signal contributions |
| **speaker** | `identity` | three coupled quantities | 50 ms | `per_speaker`, `count`, `assignment` |
| **ASR** | `utterance` | this span's token sequence | word spans | consensus text, dissenting models, `confidence`, `uncertainty`, `epistemic`, `aleatoric`, `variability` |
| **background-mask** | *new* | this region is free of the target | 100 ms | `state`, `confidence`, `uncertainty`, `variability`, detected sources |
| **task** | *new* | this region is unrelated to the task | — | **deferred** — focus on speech tasks first |

**D-13. Resolutions are per category, not one shared grid.** Each is the finest its inputs justify:
20 ms (segmentation, brouhaha at 16.9 ms), 50 ms (embedding hop), word spans, 100 ms (noise floor).
A single shared grid would force the coarsest resolution onto every output.

**D-14. ASR `variability` is word-boundary standard deviation in seconds.** WER is reported
separately as a distance. *Why:* `statistics.py` defines variability as dispersion of repeated
measurements of *one quantity*; word boundaries in seconds qualify, while a spread of WER ratios is
a different kind of object and would invite reading a ratio-spread as a dispersion.

### Output layout

```
<run_dir>/
  L1/<pass>/signals/<signal>.parquet        + provenance.json per signal
  L2/round<N>/{speech_presence,speaker,asr,background_mask}/
  final/
    speech_presence.parquet
    speaker/{per_speaker,count,assignment}.parquet
    asr.parquet
    background_mask.parquet
    speakers.json  summary.md  timeline.png
    decisions.json          <- trajectory, reversals, stopping reason
```

**D-15. `final/` holds only the converged state**; the round trajectory, including reversals and
the stopping reason, is summarised in `decisions.json`. Per-round outputs remain under
`L2/round<N>/` for audit.

`L2/rounds.json` carries the per-axis round log `fuse_rounds` produces, one entry per round:

| field | meaning |
|---|---|
| `numbers_settled` | the fused values stopped moving — **one** of four criteria, not convergence |
| `converged` | all four criteria held |
| `criteria_evaluated` | `false` on the round-0 shortcut, where the loop could not iterate at all |
| `blocking` | which criteria failed |
| `credited_epistemic_change` | C1's change *after* the self-confirmation guard |
| `diverged` | uncertainty rose — legitimate, and does not stop the loop |
| `stop_reason` | `converged` \| `oscillation` \| `no_improvement` \| `max_rounds` \| `null` |
| `repeating_states` | which states traded places, when oscillating |
| `action_scope` | which action inventory C4 was answered against |
| `coupled_from` | which other axes voted into this axis this round (D-11) |
| `derivatives_refreshed` | whether the round re-derived the mask and claims, or reused round 0's |
| `remeasured` | whether the round took a finer look rather than only re-weighting (D-10) |

`numbers_settled` and `converged` are deliberately separate fields: a consumer that reads the
first as the second is making exactly the claim the four-criteria design exists to prevent.

---

## Implementation order

Dependency-ordered; each step verifiable before the next.

1. **L1 raw emission. Done.** All 24 register items closed. The speech-presence
   harvester is dissolved into `harvest_speech_presence_evidence` (measurements) +
   `speech_presence_link` (beliefs under a named policy). Register item 24 — quality resampling —
   closed with it: `resolution.resample_series` is now wired into the quality harvest.
2. **Harmonization** H1–H5. **H2 done** (`harmonize.py`, dual matcher, assignment uncertainty as
   disagreement, D-6). **H4 done** (checked-in AudioSet category map). **H5 done** (absolute `lufs`
   / `excess_db` / `hnr_db`, D-3). **H1 partial** — `resolution.resample_series` is wired into the quality
   harvest (item 24), so scene quality now integrates or holds onto the reporting grid according to
   direction. The remaining piece is applying the same conversion to the *other* signal families,
   which still report on the reporting grid rather than on a lattice built from each declared
   native resolution. **H3 done** (`harmonize.harmonize_transcripts`) — the transcripts are
   aligned to each other, not only to audio. A model that inserts or drops one word shifts every
   timestamp after it, so a time-based comparison turns a single miss into a whole tail of apparent
   substitutions; aligning the token sequences keeps the rest lined up. Star-shaped against the
   *median-length* transcript, so an outlier (a hallucinated run, a truncated decode) does not
   become the frame everything else is measured against — an approximation against full
   multiple-sequence alignment, with the privileged model reported so it is visible. A slot with no
   strict majority publishes no consensus: a two-way disagreement has no winner, and naming one
   would manufacture agreement never observed. Normalisation (case, punctuation) decides agreement
   only; each model's surface form is kept.
3. **Joint estimation** J1–J9, with J4 as a joint space (D-7). **J3 done**
   (`speaker_identity.speaker_count_posterior`). **J5–J6 done** (`background_mask.py`,
   `noise_floor.py`, `sources.py`). **J8 done** (`reliability.py`, cross-pass stability as measured
   weight). **J9 done** (`invariance.py`, `--invariance-probe`). **J1 done** (`joint.overlap_count_posterior`, wired as the
   `<signal>::overlap_count` speaker sub-signal). **J2 done**
   (`joint.speaker_change_series`, wired as `<embedding_model>::change_point`). **J7 done**
   (`joint.phoneme_transcript_agreement`, over H3's slots). **J4 done** (`joint.per_speaker_presence`) — the `S_k` ↔ channel
   binding, decided by temporal agreement and Hungarian-matched, reporting how firmly it is
   determined rather than only what it decided. Nothing is thresholded: a speaker with no
   overlapping channel activity is left unbound rather than given the least-bad channel, and a
   channel no speaker claimed is reported — that is the shape a missed speaker takes. The
   `assignment_margin` is what C2 needs, and the fact that it is *evidence* rather than a
   preprocessing result is D-7's point.

   J7 is worth having over the per-window PER the ASR axis already computes because it asks a
   different question. That measure asks whether *a* model's transcript matches the audio; J7 asks
   which of the readings on the table does. PPG posteriors reach the audio without passing through
   a language model, so they can adjudicate between two ASR readings without echoing a third
   transcriber's opinion. The candidate distribution is `max(0, 1 − PER)` normalised — a linear
   link with no free temperature, so the adjudication introduces no tuned parameter. A slot where
   every reading is contradicted yields a uniform distribution and full doubt rather than being
   dropped: "all candidates are unsupported" is a finding, not a missing measurement.

   J1 was answerable before the rest because a *count* of active channels is invariant to the
   channels' arbitrary ordering, while anything naming *which* channel is whom waits on the joint
   space rounds resolve. It is also the signal the old noisy-or collapse destroyed: `1 − Π(1 − p_k)`
   answers "is anyone speaking" and discards how many. The posterior is built per frame and then
   pooled — two speakers taking turns within a bucket average to 0.5 on each channel, which as a
   per-bucket calculation would report a 25% chance of an overlap that never occurred.

   J2 compares each embedding window against the one a whole window-width later, not the adjacent
   one. The reason is **contrast, not a difference in what is measured** — an earlier description of
   this in the code said the adjacent distance "measures phonetic drift rather than speaker
   identity", which is wrong. Adjacent 2 s windows at a 50 ms hop share 97.5% of their audio, so
   swapping 50 ms moves the embedding only slightly: the speaker change is still present but
   low-amplitude and spread across the window width as one voice is gradually exchanged for the
   other, rather than appearing as a step. It is hard to place and easy to lose in noise, not
   absent. Lagging by the window width puts the full between-speaker difference into a single
   score; the fine hop still earns its keep by localising the boundary, which is what D-2 said it
   buys, and it does not buy independent samples, so neighbouring scores are near-duplicates and
   must not be counted as separate evidence.

   The distance is read through the speaker axis's existing calibration band rather than a new
   anchor, because a raw cosine of 0.2 is not evidence of anything — same-speaker embeddings sit in
   a 0.1-0.3 noise floor from phonetic variation alone. The band's anchors are **required
   arguments**, not defaulted: a pass measured to have no usable band must not get library anchors
   by omission, which is the FR-007 rule the other embedding sub-signals already follow.
4. **Link functions** and TTS-fitted calibration (D-8, D-9).
5. **Rounds and convergence** (D-10 – D-12, both guards). **Convergence done**
   (`rounds.assess_convergence`, wired into `fuse.fuse_rounds`): C1-C4 judged together, cycle
   detection separate from slow convergence, the self-confirmation guard on C1, and divergence
   treated as a legitimate outcome that continues the loop rather than aborting it. Each round's
   log records `numbers_settled` and `converged` separately plus which criteria blocked, so
   "cycled" and "ran out of rounds" stay distinguishable from "agreed".

   `fuse_rounds` accepts `speaker_assignment` (J4's binding, for C2) and `untried_actions` (for C4,
   from a caller whose action inventory is wider than the loop's own). Omitting `speaker_assignment`
   leaves C2 **unmeasured, which blocks convergence** rather than passing it; omitting
   `untried_actions` falls back to the loop's own countable inventory, with `action_scope` recording
   which was used (see C4 below).

   That last point was wrong in the first implementation and is worth recording. `assignment=None`
   compared equal between rounds and a defaulted `untried_actions=0` read as an exhausted
   inventory, so both criteria reported *passed* while nothing had been measured — the loop could
   reach `converged` on two criteria while claiming four, which is exactly the failure the
   four-criteria design exists to prevent. `None` now means unmeasured for both, and a test pins
   that an unmeasured criterion blocks while a measured zero passes: having checked and found no
   remaining action is a different statement from never having looked.

   **C2 is now measured.** `write_final_uncertainty` derives J4's binding from the reference pass
   (`speaker_spans_from_votes` over the harmonised cluster ids × the pass's frame posteriors, both
   carried on `PassHarvest`) and passes it to the speaker axis's rounds. It returns `None` rather
   than an empty mapping when the inputs are missing, because an empty binding and an unmeasured one
   mean different things to C2 — two empty mappings compare equal and would read as a stable
   assignment nobody checked.

   **Duplication resolved.** `adaptive/convergence.py` already had `detect_non_convergence`, which
   detects oscillation *and* stagnation over a sliding window of round states, and reports them
   separately because the remedies differ — a flip-flop means two signals disagree irreconcilably,
   standing still means no signal has anything left to contribute. `rounds.assess_convergence` was
   written without checking for it and carried its own cycle detection, which compared against all
   earlier signatures but collapsed both failures into `"cycle"`. The two were not redundant by
   accident: they serve different loops (the adaptive loop in `adaptive/loop.py`, the L2 fusion
   rounds in `fuse.fuse_rounds`). But the *detection* is one question about a round history, and two
   implementations of it could reach opposite verdicts on identical states.

   `detect_non_convergence` now lives in `rounds.py` — the dependency runs adaptive → workflow, so
   the shared piece belongs at the lower level — and `assess_convergence` calls it in place of its
   own check, so the fusion rounds report `"oscillation"` and `"no_improvement"` instead of one
   flattened `"cycle"`. Two things changed with the move beyond deduplication:

   - **The window now bounds recency in both loops.** The old check asked "does the current
     signature appear anywhere earlier", which never expires: a state that recurred once and was
     left behind would keep stopping the loop after it had resumed making progress. A cycle of
     period *p* becomes visible once the window holds a repeat, which takes `p + 1` rounds, so
     `DEFAULT_CYCLE_WINDOW = 4` catches periods one through three.
   - **Convergence suppresses the verdict, not the reverse.** When all four criteria hold, a
     repeated signature is agreement, not a cycle — the same state twice is what settling looks
     like. The detector runs only when something still blocks. The adaptive report needs the
     asymmetric form of the same rule: a settled loop *holds still*, so `no_improvement` cannot
     count against a convergence the loop reached on its own grounds, while `oscillation` still
     overrides it — those values are unsettled whatever the loop concluded. Without that
     distinction every clean convergence reported as a failure to converge.

   **The detector was never wired to anything.** Finding it unused was the reason to look: T082
   built it, `ConvergenceReport` was specified to carry `termination_reason`, `oscillation_states`
   and `unresolved_quantities`, and none of the three were emitted, so a run that oscillated
   reported `run_state: "converged"` when the loop stopped because nothing more would fire. That is
   precisely "an unsettled value presented as settled". `adaptive/loop.py` now snapshots each
   round's uncertainty mass and bucket-status census, and `build_convergence_report` runs the
   detector over them; a detector verdict *overrides* `run_state`, since a loop with nothing left to
   fire has still not settled if its state was trading places.

   `unresolved_quantities` is `None`, not `[]`, when no resolution inventory was supplied. An empty
   list would read as "we checked and everything settled" — the same default-stands-in-for-absent
   shape that C2/C4 hit above, and that the quality nearest-window fallback and J2's calibration
   floors hit before that. Unrecognised run states pass through rather than folding into `"budget"`,
   for the same reason: mapping the unknown onto a known outcome is how a state nobody has thought
   about starts reading as one that was.

   **C4 is now measured, against a named inventory.** The blocker was framed as "counting untried
   actions means reaching the intervention catalogue in `adaptive/`", and that framing was the
   mistake: it assumed one global action inventory that only the adaptive subsystem holds. There
   are two, because the loops can do different things.

   `fuse_rounds` works from a *fixed harvest*. The only thing it can do without new measurement is
   withdraw regional trust where the mask contradicts a claim, and the tightened weights apply
   every such region at once. So its inventory is countable by inspection: `available_actions` at
   round 0, zero from round 1 — a **measured** zero, which is what C4 asks for, rather than the
   defaulted zero that previously made C4 read as passed while nothing had been checked.

   That does not make the adaptive catalogue redundant; it makes the scope explicit. A measured
   zero here says *this loop ran out of moves*, not *no further measurement would help* — the
   adaptive loop can still re-run models over the same region. Each round's log records
   `action_scope` (`regional_trust` or `caller_supplied`) so the narrow claim cannot be read as the
   wide one, and a caller that does hold the wider inventory passes `untried_actions` to override
   the local count. The rule generalises: a criterion answered against a narrower inventory than
   the question implies is not the same failure as one never answered, but it misleads the same
   way unless the scope travels with the answer.

   Fixed alongside it: the round-0 shortcut (no mask regions or speaker claims) marked itself
   `converged` without evaluating any criterion, so a reader could take it for the four-criteria
   verdict the same field carries everywhere else. It now also records `criteria_evaluated: false`
   — the loop stopped because it *cannot iterate*, which is a different statement from agreeing.

   **D-11 done: the axes are interleaved.** The old shape was one `fuse_rounds` call per axis, each
   loop run to completion before the next began. A region doubtful on `speaker` therefore could not
   reach `speech_presence` — the coupling was structurally unable to act however it was configured,
   which is not a tuning problem. `fuse_axes` replaces it: every axis folds round 0, then each later
   round folds every axis against the *previous* round's outputs. `fuse_rounds` is now a wrapper
   around it with one axis and coupling off, not a second implementation — two round loops over the
   same criteria could reach opposite verdicts on identical history, which already had to be undone
   once for non-convergence detection.

   **The stage boundaries, stated as they actually are.** L1 emits signals. L2 round 0 takes the
   signals and emits *derivatives and axes*; round *N* takes round *N-1*'s outputs plus the signals
   and emits axes plus targeted refinements; `final/` takes round *N*'s outputs plus the signals.
   Two consequences the first implementation had wrong:

   - **Derivatives are round outputs, not fixed inputs.** The mask and the speaker claims are
     estimates. Computed once and held constant, every later round withdraws trust on the strength
     of a judgement the loop had already improved on. `fuse_axes` takes a `derive` hook called
     before each round after the first, and each round's log records `derivatives_refreshed` — a
     stale judgement must not look like a current one.
   - **Cross-axis input is estimation, not attention.** An earlier draft had another axis's doubt
     only raise `triage_score`, leaving the measured quantities untouched. That is a weaker claim
     than the design makes: D-7 has speaker and presence *jointly* estimated, so another axis's
     value is a genuine input to this axis's fold. It enters as `axis::<other>`.

   So each round takes **all three** things the loop holds — the signals, the derivatives, and the
   previous round's axes — and re-estimates every axis from them. Convergence is no axis changing,
   or the loop entering a periodic one: C1 and C3 cover "nothing moved" (values holding still *and*
   no bucket going unmeasured → measured), and a repeating state is caught by the shared
   non-convergence detector and reported as `oscillation` rather than as agreement.

   **No assigned discount on the cross-axis input.** A draft in between multiplied it by a fixed
   `0.4` — a `CouplingPolicy` "derivation gate" — to stop the extra input dominating the fold. That
   contradicts the premise the whole module rests on: weights here are *measured* (perturbation
   stability, physical support), with the explicit rule that a factor never measured must not act as
   a discount. A hand-set constant is exactly such a factor, so it was removed rather than tuned. A
   cross-axis input now carries full weight like any other signal absent from the weights mapping.

   The concern the constant was standing in for is real but is a different kind of thing: an axis's
   value is built from signals, so where two axes read the same signal its evidence appears twice.
   That is a correlation to **measure** — the same sort of quantity perturbation stability already
   measures — not a number to assume. Until it is measured, the derivatives carry the coupling that
   can be justified structurally and the axis path carries the rest at face value. Assuming a
   discount would have looked like a fix while leaving the bias in place, and convergence cannot
   catch it either: a biased fixed point is still a fixed point.

   Rounds that used cross-axis input or re-derived the shared structure set `overwrote_values`, so
   C1 refuses to credit any drop the loop produced itself. An axis is never an input to itself — its
   previous value is the thing being updated, not evidence about it — and an axis that measured
   nothing contributes nowhere: `None` is the absence of a claim, not a low uncertainty.

   **Four axes, not three.** `speech_presence`, `speaker`, `asr`, `background_mask`, with `task`
   punted. `background_mask` had modules but was never an emitted axis; it has no vote harvest
   because it is one derived judgement per region rather than a model ensemble, but it does report
   how sure it is, and `1 - confidence` is that judgement's uncertainty in the units the other axes
   already use. An `indeterminate` region is skipped rather than voted at maximum uncertainty —
   otherwise an unresolved region outvotes the resolved ones. `fuse_axes` takes any number of axes;
   a count baked into the loop is a fact that needs re-finding every time the set changes, and it
   had already been baked in at three.

   **Isolation stayed reachable, after briefly not being.** Removing `CouplingPolicy` also removed
   the only way to run several axes uncoupled, since `derive=None` disables re-derivation but leaves
   the axis-to-axis input on. `couple_axes=False` restores it. This matters more than a convenience
   flag: the coupling is only evaluable against the *same* axis set with coupling as the single
   difference — comparing against a one-axis run changes two things at once and says nothing.

   **The successor gate, measured.** `measure_axis_overlap` computes the fraction of a contributing
   axis's evidence the receiving axis already holds — the overlap of their `contributing_signals` —
   and that drives the weight through the existing `effective_weight`. An axis telling us only what
   we already have earns little; one with independent evidence earns full weight; an axis whose
   evidence cannot be measured earns **no discount**, because a factor never measured must not act
   as one. The measured value is recorded in `weight_basis` as `evidence_overlap`, so a reader can
   see which factor discounted a cross-axis input and by how much. Signals contributed by a previous
   round's coupling are excluded from the source's evidence — counting them would let the measure
   feed on its own output and drift every round.

   The uncertainty gate is deliberately left open on this path. The quantity a cross-axis input
   carries *is* the other axis's uncertainty, so discounting it for being uncertain would suppress
   precisely the informative case — the opposite of the reasoning that applies when a signal's own
   unreliability should limit its influence.

   **`influence.py` moved down** from `adaptive/` to the workflow level. It is pure and
   dependency-free, and three consumers at the workflow level now need it: `speaker_identity` for
   its source weights, `fuse` for the overlap gate, and the adaptive loop itself. A shared piece
   imported *upward* out of a subsystem inverts the dependency and makes the subsystem look like a
   library for the level above it — the same reasoning that moved non-convergence detection into
   `rounds.py`. The inversion predated this work (`speaker_identity` already reached up for it);
   adding a second consumer is what made it worth fixing rather than noting.

   **D-10 done: a round may re-measure, not only re-weight.** Re-weighting can only redistribute
   the evidence already gathered, so a region no re-weighting resolves is exactly the one that needs
   a finer look. `fuse_axes` takes a `remeasure` hook, called once per axis per round with that
   axis's regions still above `unsettled_above`; whatever inputs it returns join that axis's fold
   from then on. The dependency stays pointed the right way — the loop never learns what a model is,
   only that a region was offered and an answer came back.

   Three properties it has to have, each for a reason the loop would otherwise get wrong:

   - **A region is offered once.** The same finer look repeated is not new evidence, and C4 would
     never reach zero if a spent action kept being counted as pending.
   - **Pending re-measurements block C4** and make `action_scope` name a real inventory rather than
     a single-item one. This is the case the earlier `regional_trust`-only scope was honest about
     being narrow.
   - **A re-measured round sets `overwrote_values`**, because a re-measurement *replaces* a value.
     C1 must not read the resulting fall as independent evidence agreeing.

   `unsettled_above` is named rather than inlined: it decides what gets looked at again, and a
   silent default would make the loop's spending invisible.
6. **Rename** `presence`/`identity`/`utterance` → `speech_presence`/`speaker`/`asr`. **Done.**
   Renamed outright with no aliases; `CACHE_SCHEMA_VERSION` bumped 5 → 6 so cached entries carrying
   the old axis names are discarded rather than mixed with new ones.

## Open items

- **SPARC** needs subprocess-venv isolation before D-8's reference set can be extended beyond TTS.
- **`scene_quality_coupling`** is null in all 1070 rows — a separate defect, not a layering issue.
- **Degraded fixtures** (`/tmp/bprobe/noisy_*.wav`, `reverb_*.wav`) should become checked-in test
  fixtures: clean TTS genuinely sits at 70 dB / 59.8 dB, so the useful range of the SNR and C50
  links is only observable on degraded material.
- **Sampling/dropout uncertainty** is absent for every signal that does not expose internal doubt
  (`segmentation_activations`, embeddings, scene classifiers).
- **Non-target measurement** for the task category, deferred with it.

## Notes from executing the rename

**A blind sweep would have been wrong.** These words appear as ordinary nouns as well as axis
names: `words_in_utterance` is a spoken utterance, `speaker_identity` is a person's identity,
`utterance_transcript` is the text of an utterance. Those were protected by placeholder before
substitution, and substitution then ran at *identifier-component* level (underscore-delimited), so
`presence_confidence` became `speech_presence_confidence` while `speech_presence_labels` — already
correct — was left alone.

**Hyphen-delimited names needed separate handling.** CLI flags are hyphenated, so the
underscore-based protection did not see them and two classes of mangle appeared:
`--speech-presence-labels` (already correct) became `--speech-speech_presence-labels`, and
`--no-per-speaker-identity` became `--no-per-speaker-speaker`. Any hyphenated name that acquired an
underscore is the diagnostic for this, and it is worth recording because it is invisible to a test
suite that never parses those flags — one test did catch it, by luck rather than by design.

**Two flags were renamed rather than mechanically substituted.**
`--identity-same-speaker-floor` would have become `--speaker-same-speaker-floor`, which repeats
the axis name; they are now `--speaker-same-floor` / `--speaker-diff-floor`.

**A pre-existing failure surfaced.** `test_schema_version_is_pinned` asserted
`CACHE_SCHEMA_VERSION == 4` while the constant was already 5, so it had been failing unnoticed in a
suite area that was not being exercised. A pin that can drift silently cannot enforce what it
exists to enforce; it now reads 6 and records why.

---

## ASR word timestamps: source belongs at L0 (open)

Found by the first real cold run: `final/transcript.json` had **zero words** on a clip all three
backends transcribed correctly. Not an alignment failure — a collection one.

| model | word timestamps from |
|---|---|
| CrisperWhisper | native (word-level Whisper) — 62 chunks present |
| Qwen3-ASR | bundled `Qwen3-ForcedAligner-0.6B` companion — 62 chunks present |
| canary-qwen 2.5B | none (text-only) — currently sent to MMS, which returned 0 words |

`collect_word_streams` sourced words from the alignment outcomes only, so the two models carrying
usable chunks contributed nothing and the one that needed alignment contributed zero. Net: no
transcript, and an empty word row in `final/timeline.png`.

**Decisions.**

1. **The timestamp source is declared at L0, not sniffed downstream.** `speech_to_text` returns
   word chunks carrying `native` / `bundled_aligner` / `none`, set by each backend. Today
   `collect_word_streams` infers it by probing for `chunks`, which has a fusion helper
   reverse-engineering a fact the backend knows for certain. With it declared, the alignment stage
   runs only where the source is `none` (and stamps `mms_alignment` / `qwen_aligner` on what it
   produces), and every consumer reads one shape instead of three ad-hoc probes. Same
   L1-measures/L2-decides split, one level down: L0 reports what it produced and where the timings
   came from; deciding which to trust when models disagree stays at L2.

2. **canary-qwen aligns with the Qwen3 aligner, not MMS — which is already D-1, unimplemented.**
   D-1 decided this at design time: both text-only paths use Qwen3-ForcedAligner-0.6B so that
   word-boundary differences between two models reflect *the models, not two different aligners*,
   with CrisperWhisper keeping its native timings because forcing an aligner over a model that
   already reports boundaries discards the more direct measurement. The run confirms the shape D-1
   describes and confirms the code never followed it: canary still goes to MMS and gets 0 words.
   The MMS case bug is then incidental — canary should not be reaching MMS at all.

   **The cost D-1 accepts, which must be carried in provenance:** canary and Qwen3-ASR share an
   aligner, so their word times agree partly for reasons that have nothing to do with the audio.
   D-1 wants that — a common aligner is what makes a boundary difference attributable to the
   models — but the asr axis also *compares* timestamps across models, and there two of three
   agreeing because one aligner produced both is not corroboration. Same shape as the cross-axis
   double-counting the evidence-overlap gate measures. Agreement between models sharing a
   `timestamp_source` must not be read as independent confirmation.

**Also found by the same run:** the L2 rounds were inert. `analyze_audio.py` called
`write_final_uncertainty` with no `max_rounds`, `mask_regions` or `speaker_claims`, so `range(1, 1)`
was empty and every run folded once and stopped — no regional trust, no fourth axis, `coupled_from`
empty in all 85 rows. Fixed by wiring all three, with `mask_regions_from_rows` converting the mask's
`uncertainty` to the `confidence` regional trust reads (unconverted, `.get("confidence", 1.0)` would
have defaulted an unsure mask to *fully confident* and withdrawn the maximum trust available).
Not yet re-run, so the rounds iterating end-to-end remains unverified.
