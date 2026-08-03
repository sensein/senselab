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

---

## Output store and cache location (decided; cache move not yet implemented)

**`artifacts/analyze_audio/` is the store.** There were two conventions for the same thing —
`analyze_audio.py` defaulted here while `e2e_runs/` was the documented target for the adaptive e2e
test, `adaptive_loop.py`, quickstart and T110. One of them was also tracked in git (618 files,
142 MB) while the other was not. All references now point at `analyze_audio/`, and both paths are
gitignored: run outputs are not source.

**The CLI must accept any analysis directory.** `--out-dir` already does. The default is a local
convenience, not an assumption the code may rely on — which is why
`adaptive/loop.py:_resolve_input_audio` is fragile: it re-roots a recorded audio path by walking a
fixed number of parents from the run directory, inferring the repo root from the path *shape*. That
works for the default layout and silently fails for an arbitrary `--out-dir`. It should look for a
marker rather than count directory levels.

**The cache should live inside the target output directory.** Today it is a separate sibling
(`artifacts/analyze_audio_cache/`, `--cache-dir`), so pointing `--out-dir` at another disk or
another machine's export leaves the cache behind and the run silently recomputes — or worse, reuses
entries keyed to inputs that are no longer the ones being analysed. Making the cache a child of the
output directory keeps a run self-contained and movable.

The alternative worth naming: a **global** cache (`~/.cache/senselab/analyze_audio/`), shared across
every output directory. That trades self-containment for reuse across runs, which is the thing the
content-addressable keys were designed to give. The two are not exclusive — a per-output cache with
a global read-through fallback would give both — but that is a third design, not a compromise.

Not implemented. Recorded so the choice is deliberate when it is made, rather than settled by
whichever call site is edited first. `CACHE_SCHEMA_VERSION` already makes invalidation free, so
relocating costs recomputation and nothing else.

---

## The belief store's place in L1 → L2 → final (open, and the root of items 25–27)

**There are two L2 implementations.** `fuse.fuse_axes` is pure, per-round, returns rows. The
adaptive belief store (`adaptive/belief.py`) is mutable, vote-based, carried across rounds. Both
fuse per-signal evidence into per-axis values, both iterate rounds, both decide convergence, both
emit a speaker axis — on different grids, from different provenance.

That is the root the register's items 25, 26 and 27 are symptoms of, and it explains the first
defect found in this work: two non-convergence detectors, merged early as an accident. It was not
an accident. Two L2s each need to decide when to stop.

**Correct placement.** L1 emits per-signal measurements. **L2 holds one belief state**, fuses it,
and iterates. `final/` holds the converged answer and is read by no stage. The belief store should
*be* L2's state — seeded from L1's per-signal votes rather than L1's pre-folded axes, with
`fuse_axes` as its aggregation step rather than a rival implementation. Its vote schema is already
right for this: one entry per `(axis, bucket, source, stream, scope)` is exactly L1's proper
granularity. Only the *seed* is wrong.

**"Vote" is the wrong word for what L2 does.** Voting implies exchangeable, independent ballots
answered by tally. The design specifies weighted inference over measurements carrying their own
uncertainty, weighted by measured reliability. Those are not ballots — not independent, not equally
informative, not combinable by counting. The metaphor has cost: it is what made a hand-set `0.4`
"derivation gate" feel natural, when in a statistical combination that is an unmeasured prior the
module explicitly forbids. `measure_axis_overlap` replaced it precisely because votes are *not*
independent, which is a fact no voting model can state.

**`purged_hallucination` claims more than the evidence supports.** The rule
(`interventions.py:253`) fires when ASR produced words where `p_voice` is low — non-corroboration
between two sources. It cannot distinguish a fabricating recognizer from a correct one contradicted
by a wrong presence estimate, which is what a quiet or overlapped speaker produces. It is also
asymmetric: presence indicts ASR, never the reverse, though word boundaries are the more precise
measurement. `unsupported` would name the observation; `hallucination` names a cause.

**The open decision — settled 2026-08-02.** The quantity should not exist. See
"An uncertainty axis is an aggregator" below.

---

## The sequence, written out

What should happen, in order, with the invariant that makes each step checkable. Written because
every defect found in this work was a step reading from the wrong place, and none of them failed
loudly.

### L1 — measure

**In:** audio, once per *pass*. A pass is the same recording under a transform — `raw_16k` as
recorded, `enhanced_16k` after speech enhancement. Both see identical content, so disagreement
between them is attributable to the transform. That is what makes them a perturbation sample rather
than two recordings.

**Do:** run each signal. Report what the tool produced, in the tool's units, at the tool's
resolution. A signal may report its own uncertainty — that is the signal's final measurement.

**Out:** `L1/<pass>/signals/<signal>.parquet`, each value carrying units, model and revision,
window and hop, and any reduction the tool itself performed.

**Must not:** threshold, rescale to `[0,1]` against an anchor, reduce across a dimension the tool
reported separately, select among estimators, invert, resample off native resolution — or emit an
**axis**. An axis is a fold across signals, and a fold is an answer.

*Checkable:* nothing under `L1/` is named for an axis. If it is, L2 has leaked downward.

### L2 — fuse and iterate

One belief state, per `(axis, bucket, source, pass, scope)`. This is the belief store's proper
place: it is L2's state, not a subsystem's private copy.

**Round 0 — in:** L1 measurements only.

1. **Link** measurements to beliefs under a *named policy* (`SpeechPresencePolicy` and siblings).
   Every threshold lives here, is recorded in the row's provenance, and can be changed without
   re-running a model. This is the only place a threshold belongs.
2. **Weight** each signal by what was *measured* about it — perturbation stability across passes,
   physical support. Never by an assigned constant: a factor never measured must not act as a
   discount.
3. **Fuse** per-axis, keeping the four quantities distinct — `uncertainty` (normalised entropy),
   `epistemic_uncertainty` (its reducible part), `confidence` (a probability), `variability` (a
   dispersion in native units) — plus `triage_score`, the policy fold for ranking where to spend
   budget.
4. **Derive** the derivatives: mask, speaker allocation, ASR consensus, scene components. These are
   round *outputs*, not fixed inputs.

**Round N — in:** L1 measurements (unchanged), round N−1 axes, round N−1 derivatives.

1. **Re-derive** the derivatives from the previous round's axes. Frozen derivatives make every later
   round withdraw trust on a judgement the loop already improved on.
2. **Re-estimate** each axis from its own signals, conditioned on the current derivatives and
   informed by the other axes projected onto *its* lattice. Coupling informs a grid; it never
   extends one. Exact-key matching is not projection — on real audio the axes share no keys.
3. **Optionally re-measure** (D-10). This is the *only* arrow back to L1, and it must be explicit:
   the round asks for a finer window or hop over a flagged region and receives new L1 measurements.
   Re-measurement replaces a value, so the round records `overwrote_values` and C1 declines to
   credit the resulting fall.

**Stop when:** no axis changes, or the loop enters a periodic one. Judged on all four criteria —
epistemic uncertainty stopped falling, the speaker↔channel assignment is stable, no bucket went
unmeasured→measured, no region has an untried action — with unmeasured criteria *blocking* rather
than passing.

**Out:** `L2/round<N>/` — axes, derivatives, and the decision log naming which criteria blocked and
why the loop stopped.

*Checkable:* round N reads only round N−1 and L1. Two rounds that differ produce two directories.

### final — the converged answer

**In:** round N outputs, and L1 measurements where a deliverable needs raw evidence.

**Out:** transcript, diarization, `speakers.json`, timeline, summary — the converged state, and the
trajectory summarised in `decisions.json`.

**Must not:** be read by any stage. A deliverable that something reads is an intermediate wearing
the wrong name, and `final/summary.json` carrying 4.8 MB of L1 evidence that the pipeline reads back
is the case in hand.

*Checkable:* nothing in the pipeline opens a path under `final/`.

### The invariant behind all three

**Each layer reads only the layer below it, plus its own previous round.** No layer stores another
layer's data. Where that was violated, nothing broke — two copies of the same bytes are
indistinguishable to a reader, a glob that matches nothing looks like a stage that produced nothing,
and coupling that matches zero keys looks like coupling that had nothing to say. Every one of those
was found by looking at a real run, and none by a test.

### What is present at the output of each stage

Concretely, for a run with two perturbations. Marked **[is]** where this is what a run produces
today and **[should]** where the target differs from current behaviour.

#### After L1 — evidence

```
L1/
  perturbations.json             the open register: name, transform, parameters, measured   [is]
  raw/                           the identity perturbation                                  [is]
  perturbation/<k>/              every other transform of the recording                     [is]
    diarization/<model>.json     per-model speaker spans
    asr/<model>.json             ScriptLine tree: text, word chunks, avg_logprob,
                                 no_speech_prob, timestamp_source
    alignment/<model>.json       word boundaries for models that report none natively
    embeddings/                  per-window speaker vectors (2.0 s / 50 ms)
    ast.json, yamnet.json        [{label: score}, ...] per window
    features.json, features/     praat / opensmile / torchaudio-squim measurements
    noise_floor.parquet          per-band floor + ECMA-74 prominence
    background_sources.parquet   per-band source candidates
    background_mask.{parquet,json}  regions + state + uncertainty
  signals/<signal>.parquet       per-signal measurement across raw AND every perturbation,
                                 one row per (perturbation, bucket), native units          [is]
  signals.png, timeline.png      evidence views, signals in native units
```

`L1/signals/` is **L2's only input from L1**, and it accumulates: the perturbation is a column,
not a directory, so a consumer that wants one perturbation's evidence has to say so on the data.
Present per value: the number, its units, the model and revision that produced it, the window and
hop it was measured over, and any reduction the *tool* performed. Absent: any axis, any threshold,
anything in `[0, 1]` that was not natively a probability.

*`L1/<pass>/uncertainty/<axis>.parquet` and `L1/stability/raw_vs_enhanced/<axis>.parquet` are gone
(register item 25, closed 2026-08-02). `L1/stability/` and `L1/passes.json` are gone with the D-17
restructure: the first is a fold over perturbations and belongs to a round; the second was
rewritten by the run's last stage, a back-edge from the deliverable to the file defining L1's
inputs, and its content now rides in `perturbations.json` beside the declaration it measures.*

#### After each L2 round — belief

```
L2/
  round/<n>/
    estimates/<axis>.parquet     one row per bucket per axis, EVERY active axis,
                                 EVERY round, one schema                              [is]
    derivatives/
      votes/<axis>.parquet       linked evidence, (axis, bucket, source, pass, scope) [is, round 0]
      stability/<signal>.parquet per-signal cross-perturbation |Δ| per bucket         [is, round 0]
      regions.json               the spans this round proposed                        [is, intervening rounds]
      votes_added.parquet        what this round added to the store                   [is, intervening rounds]
      mask, speaker allocation, ASR consensus, scene components                       [should]
    timeline.png                 the same figure the final timeline draws             [is]
    summary.json                 what this round did, per producer                    [is]
```

**What a round owes, and what it merely may leave.** Three artifacts are owed by *every* round —
`estimates/` for every active axis, `summary.json`, `timeline.png` — because every round has a
belief, an account and a view; a round missing one is a round whose place in the trajectory cannot
be read. The four derivative families are not owed: `votes/` and `stability/` are the ingest
round's, computed once from L1, and `regions.json` / `votes_added.parquet` belong to the rounds
that ran interventions. Writing an empty `regions.json` from a round that does no region proposal
would be the claim "we looked and found none", which is not what happened — **absent is not zero**.
`Artifact.enumerated` in `contracts.py` is where that distinction is declared, and it is checked
per member: a wildcard used to be satisfied by one match, so `L2/round/*/summary.json` read as
produced on a run where three of five rounds wrote none.

`L2/rounds.json` is **gone**. It was fusion's per-round fold log flattened to the belief root,
where it had no round to belong to; each round's entry is now the `fusion` block of that round's
`summary.json`, beside the loop's `adaptive` block for the round it adopts as a baseline.

**One tree, 0-based.** There were two — `L2/round<N>/` from fusion and `L2/rounds/<N>/` from the
adaptive loop — so the fusion loop's round 0 and the adaptive loop's round 1 were the same
iteration under two names. The loop now *adopts* fusion's index rather than assigning its own.

`estimates/` rather than `uncertainty/`: a row carries uncertainty, epistemic uncertainty,
confidence and variability, so the old name named one column rather than the thing itself.

**Closed, and how.** Round 0's estimates were written by `fuse` carrying
`signal_weights`/`weight_basis`; later rounds were written by the belief store carrying
`p_voice`/`aleatoric_floor` instead — one artifact name, two schemas, and no guard could see it
because both are genuinely keyed `(axis, bucket)` and what differed was below the key. They are
one artifact because a round's estimate of an axis is one quantity and rounds 2 and 3 are
consecutive iterations of one loop; declaring them separately would assert a seam the path does
not mention. So the schema is the **union**, declared once in `estimates.ESTIMATE_COLUMNS`, and a
producer with nothing to say for a column writes null — "this producer does not compute a
convergence status" is exactly what a null in `status` means. `estimate_frame` raises on a column
no declaration names, so a producer that grows one has to grow the schema where both see it.

The guard that now catches the class: `Artifact.slices_of_one_table` derives, from a pattern's own
wildcards, whether its files differ only in a key dimension. `L2/round/*/estimates/*.parquet`
varies in round and axis and in nothing else, so its files are slices of one table and must have
one shape; `L1/signals/**` varies in *which tool measured*, so no such rule applies.

Present per axis row: `start`, `end`, `axis`, `round`, `uncertainty`, `epistemic_uncertainty`,
`confidence`, `variability`, `triage_score`, `contributing_signals`, `contributing_passes`,
`signal_weights`, `weight_basis`, `coupled_from`, `scene_quality_coupling`,
`triage_score_pre_coupling`, `status`, `irreducible_reason`, `speech_presence_confidence`,
`overlap_posterior`, `p_voice`, `aleatoric_floor`, `aleatoric_floor_terms`, `n_sources`,
`n_attenuated_sources`, `attenuated_sources`, `attenuation`.

**The fourth axis is a participant.** `background_mask` had estimates in the fusion rounds and none
in the loop's, and the convergence report answered *0 buckets, residual mass 0.0* — which reads as
settled and meant never asked. Two causes, both closed: `VoteStore.from_harvests` enumerated three
axes in a literal tuple, and the mask's votes were written under a fabricated perturbation called
`"mask"` that is in no run's perturbation set, so the artifact ingest path (which skips rows naming
a perturbation the run did not take) dropped every one of them. The mask is measured on the
unmodified recording and now says so; an active axis with no vote harvest must be handed in
explicitly, and omitting one raises.

Four quantities kept distinct because they answer different questions and have different estimators:
entropy, its reducible part, a probability, a dispersion in native units. `triage_score` is the
policy fold and is *not* one of them — it ranks where to spend budget.

Present per round in the log: `numbers_settled` and `converged` separately, `criteria_evaluated`,
`blocking`, `credited_epistemic_change`, `diverged`, `stop_reason`, `repeating_states`,
`action_scope`, `derivatives_refreshed`, `remeasured`, `coupled_from`.

The belief state itself lives here — one per `(axis, bucket, source, pass, scope)` — rather than in
a subsystem's private store. **[should]** — the *vote level* is done (`round0/votes/<axis>.parquet`
is what the store ingests on both paths); the store still keeps a private per-`(stream, axis,
bucket)` fold, blocked on `fuse_axis` accepting per-`(bucket, signal)` weights. See the register's
"What closed items 25–27".

#### After final — the answer

```
final/
  transcript.json                fused words with confidence and alternates       [is]
  diarization.json, .rttm        fused speaker turns                              [is]
  estimates/<axis>.parquet       the last round's estimates, copied verbatim,
                                 one file per active axis                         [is]
  speakers.json                  count posterior + per-speaker hypotheses         [is]
  per_speaker_presence.parquet   one track per hypothesised speaker               [is]
  decisions.json                 trajectory, reversals, stopping reason,
                                 every intervention entry                         [is]
  disagreements_resolved.json    which flagged regions the rounds resolved        [is]
  timeline.png, summary.md       the human-facing views                           [is]
  summary.json                   run provenance: policy hash, versions, budget    [is: ~6 KB]
  run_summary.json               the headline numbers of the last round           [is]
  labelstudio_{tasks.json,config.xml}                                             [is]
  eval.json                      the score, when there is ground truth to score against
```

**Seventeen declarations, and what happened to each.** Ten were produced and stay: `transcript`,
`diarization` ×2, `disagreements_resolved`, `timeline`, `summary.md`, `summary.json`,
`run_summary`, and the two Label Studio files. Seven were not:

- `speakers.json`, `per_speaker_presence.parquet` — written to the `L2` root by the fusion stage,
  which is where its own docstring never said they went. Moved.
- `speech_presence.parquet` — worse than misplaced: *rebuilt* at the `L2` root from the belief
  state with two columns (`speech_presence_confidence`, `overlap_posterior`) that no round
  carried, so the number a consumer acted on was not a number any round believed. Those columns
  are on the estimate row now and the file is an extraction.
- `asr.parquet`, `background_mask.parquet`, `speaker/*.parquet` — three per-axis declarations for
  the deliverable axes, and the *speaker* axis was missing from them entirely. That declaration
  was itself a list of three axes with the fourth absent, which is the failure `axes.AXES` exists
  to make impossible; all four are replaced by one `final/estimates/*.parquet` enumerated over
  `AXIS_NAMES`.
- `decisions.json` — existed as `L2/convergence.json` plus `L2/iterations.json`, two per-run
  documents at the belief root, so `final/` had no account of how the run reached its answer and
  the evaluator read the belief tree to reconstruct one. Replaced outright.
- `eval.json` — absent because the run had no ground truth. The one declared output whose absence
  is a property of the *input*, and the register says so rather than tolerating it silently.

The evaluator now reads `final/` and nothing else, which is what makes it a consumer of the answer
rather than a stage that builds it.

Present: the converged state and how it was reached. Absent: anything another stage reads.
`summary.json` no longer inlines the per-perturbation evidence (register item 26, closed
2026-08-02); the index later stages need is `L1/perturbations.json`.

**Nothing reads `final/`.** The last surviving read was an `.exists()` probe of
`final/labelstudio_tasks.json` that decided what the driver printed — a stage branching on a
deliverable is treating it as state, and a probe is a read. The loop now returns what it produced.

**[should], what is still open — stated, not closed.** Three things remain, and they are the
remaining `artifact` entries in `contracts.KNOWN_DEVIATIONS`:

- `L2/background_mask.{parquet,json}` — written by `stage_background_mask`, which runs inside
  `run_pass`, an **L1** stage, and produces an L2 artifact. That is the cycle edge D-17 forbids,
  and it is untouched here: the mask's *votes* now enter the belief store correctly, but the mask
  document itself is still an L1 stage's output in the belief tree. It is a round derivative.
- `L2/disagreements.json` and `L2/labelstudio_{tasks.json,config.xml}` — per-run indices at the
  belief root, written by the driver and read back by the loop and the LS final stage.
- `triage.json` at the run root — an L2-shaped decision (speech_present, needs_enhancement) taken
  before L1 has run, stored where no stage claims it.

And on the *static* side, the largest open item is unchanged: `scripts/analyze_audio.py` runs the
stages inline instead of invoking them, so the driver opens run artifacts directly and the DAG
cannot see inside it. The adaptive loop does the same to FINAL — it writes `final/decisions.json`
and drives the extraction, which are FINAL's artifacts written from an L2 node. The *content* of
`final/` is right; the caller is not, and both close with the same restructure.

#### The one-line test per stage

- **L1** — could a different lab reproduce this number from the audio and the provenance alone?
- **L2** — is every threshold that shaped this value named in a policy recorded on the row?
- **final** — does anything in the pipeline open this directory?

### What is present at the input of each stage

The input side is where the violations actually happened: every defect in this work was a stage
reading something it should not have had, or failing to read something it should.

#### Into L1

```
audio file                       one recording
task_type                        what counts as the *target* — speech, breathing, cough …
pass definitions                 raw_16k, enhanced_16k: the transform applied
model ids + revisions            what to run, pinned
device, cache location
```

Present: the recording and the configuration. **Absent: any prior result.** L1 never reads L2 or
`final/`, and never reads its own earlier output — a re-measurement (D-10) arrives as a *request*
for a window and hop over a span, not as a belief to refine. That is what makes an L1 value
reproducible from provenance alone.

`task_type` is load-bearing at the input, not a label: without it the mask has no definition of
target activity and marks the whole clip active. A run given no `task_type` produced
`regions_total: 0` on audio containing a 6 s speech-free gap.

#### Into L2 round 0

```
L1 measurements                  per signal, per pass, native units + resolution
link policy                      SpeechPresencePolicy and siblings — every threshold, named
reliability evidence             perturbation stability across passes, physical support
aggregator choice                min / mean — the policy fold for triage_score
calibration profile              versioned anchors, or absence recorded as absence
```

Present: measurements and the named policies that interpret them. **Absent: any axis.** Round 0
constructs the axes; receiving one would mean L1 had already decided.

A calibration profile that is missing must arrive as *missing*, not as a default — a sub-signal that
cannot be calibrated drops out rather than voting, because a confident derived signal outvotes
unanimous agreement between diarizers.

#### Into L2 round N

```
L1 measurements                  unchanged, the same ones round 0 saw
round N-1 axes                   every axis, all of them, at their own grids
round N-1 derivatives            mask, speaker allocation, ASR consensus, scene components
round N-1 belief state           votes with provenance and status
convergence history              prior rounds' records, for C1-C4 and cycle detection
```

Present: everything the loop knows. **Absent: round N's own partial results** — an axis reads the
*previous* round, never a sibling updated earlier in this one, or the fixed point depends on visit
order.

The axes arrive on their own grids and must be *projected* onto the receiving axis's lattice.
Matching raw bucket keys is not projection: on real audio the four axes carried 85 / 41 / 1070 / 1
buckets and shared zero keys, so coupling silently did nothing while every test passed.

#### Into final

```
round N axes + derivatives       the converged belief
L1 measurements                  where a deliverable needs raw evidence — word text, spans
run provenance                   policy hash, model revisions, versions, budget
```

Present: the last round, and evidence for the things a deliverable quotes verbatim. **Absent:
anything from an earlier round** — trajectory belongs in `decisions.json` as a summary, not as
inputs to re-fuse. `final/` composes; it does not decide.

#### The failure mode this side has

Every input violation in this work was **silent**: an absent directory returned `{}`, a glob matched
nothing, a projection matched zero keys, a default stood in for a missing measurement. None raised.
An input a stage should not have is invisible because it works; an input a stage needs and lacks is
invisible because empty and absent look identical. That is why each was found by inspecting a real
run's outputs, and none by a test.

---

## Correction: derive first, link inside estimate, and the store holds decisions not estimates

Three corrections to the sequence above, from review. They collapse the round-0 special case and
resolve the two-L2 problem.

### 1. Derive comes first, in every round including round 0

The sequence above put `derive` at the *end* of round 0, so round 0's axes were estimated with no
derivative conditioning and only round 1 ever saw a mask. That is not a design decision — it is an
artifact of the mask being computed in `stages.py` and handed to fusion as a fixed argument.

Every derivative is computable from L1 alone: the mask from target confidence, speaker allocation
from embeddings and diar labels, ASR consensus from word streams, scene components from AST/YAMNet.
None requires an axis. Round 0 has no reason to skip the step, and skipping it makes round 0's axes
different in kind from every later round's — which is what made "is coupling working?" hard to
answer at all.

### 2. Link belongs inside estimate, not before the loop

Linking measurements to beliefs under a named policy is the first half of estimation, not a
preamble. It was hoisted to round 0 because the policy is constant today, so the result repeats and
caching looked free. That is an assumption, not a property: the design's "named, replaceable,
recorded in provenance" framing anticipates a round proposing different thresholds, and hoisting the
link out of the loop silently forbids it.

**Uniform, no special case:**

```
round N:  derive(L1, axes[N-1], derivatives[N-1])              -> derivatives[N]
          estimate(L1, policy[N], derivatives[N], axes[N-1])   -> axes[N]

round 0:  identical, with axes[-1] and derivatives[-1] empty
```

### 3. The belief store's irreducible state is the decision log, not the beliefs

Four kinds of thing live in it, and only one is an estimate:

| held | kind | belongs to |
|---|---|---|
| the value | estimate | L2's product |
| `scope`, provenance | fact about the measurement | L1 |
| `status` — active / shadowed | decision | L2's process |
| `evidence_weight` + its factors | measurement about a measurement | L2's process (see "Purging is no longer a decision") |
| history | trajectory | L2's process |

The estimates are **re-derivable** — the store's own contract says "aggregation is a pure function
of the active votes". Given L1, the policy, the record of which votes were shadowed and the record
of what weight was withdrawn from which, every value can be recomputed. So the store should not persist estimates at all.

**This dissolves the two-L2 problem rather than fixing it.** `fuse_axes` is the pure aggregation;
the belief store is the decisions selecting which measurements are active. They are not rivals —
they are the two halves of one L2. What made them look like rivals is that the store also keeps a
materialised copy of the estimates, seeded from L1's pre-folded axes (item 25), and uses
`within_pass_uncertainty` as a parity oracle against its own recomputation.

Removing the materialised copy removes the second lineage, the parity oracle's dependence on an L1
fold, and the need for item 27's plot overlay, in one change.

---

## Purging is no longer a decision

*Recorded after removing both erasure sites. The paragraph at line 804 named the defect; this
records what replaced it, and the general finding underneath it.*

### What was there

Two places deleted evidence rather than weighing it.

**The belief store.** `VoteStore.purge_source_in_bucket` set `status = "purged_hallucination"`,
and `active_votes()` returned only `status == "active"`, so `reaggregate_bucket` never saw the
payload again. The source left `contributing_sources`; its `speaks`, its `native_confidence`, its
`avg_logprob` and its `token_entropy` left the fold. The vote survived only as a row in
`rounds/<k>/votes_added.parquet`, where nothing downstream could weigh it.

**The word streams.** `collect_word_streams(purged_spans=...)` removed every word of the indicted
model overlapping the span, before the ensemble ever saw it. `transcript.json` had no record;
`final/` had no record.

### Why it was wrong on this codebase's own terms

Every other attenuation mechanism here is floored, and all four floors cite the same sentence:
*the dissenter may be the only source that noticed something.* `MIN_REGIONAL_TRUST`,
`MIN_RELIABILITY`, `SUPPORT_FLOOR`, `effective_weight`'s `min_gate` — four literals of `0.05`,
each docstring pointing at the others' reasoning. Purging was the one mechanism that went to zero,
and it did so on the weakest signal in the system:

- **Non-corroboration is not fabrication.** The chain observes that an ASR produced words where
  presence evidence is low. A quiet talker, a distant talker, an overlapped talker and a
  fabricating recognizer all produce that. The name asserted the one cause the measurement cannot
  reach.
- **It was asymmetric.** Presence indicted ASR; ASR never indicted presence, though word
  boundaries are the finer measurement and presence buckets are 0.5 s.
- **It was self-confirming.** The trigger read `p_voice`, a weighted mean over *all* presence
  voters including the indicted ASR — whose `hallucinated: True` payload `_weighted_p_voice` maps
  to `p = 0.1`. The source partly protected itself, and acting on it moved the number that had
  indicted it. This is the failure `adaptive/provenance.classify_resolution` exists to catch,
  running inside the rule that was supposed to improve the belief.
- **It was simultaneously total and inert.** On the asr axis it removed `avg_logprob` and
  `token_entropy` but never touched `__pairwise_phoneme_distances__` — the dominant sub-signal,
  keyed on `"<src_a>|<src_b>"` inside a `__`-prefixed synthetic vote. The "purged" transcript kept
  driving the axis it had been purged from.
- **It made survival depend on budget.** Word dropping was gated on the intervention having fired
  *and* having been admitted within budget, so `deferred_budget` silently changed the transcript.

### What replaced it

One measured quantity, applied as a floored weight, in both places.

| | before | after |
|---|---|---|
| belief store | `status = purged_hallucination`, vote leaves `active_votes` | `Vote.evidence_weight` ∈ (0, 1], vote stays active and keeps aggregating |
| word streams | word deleted from the stream | `word["corroboration"]`, entering the vote weight **and** `coverage` |
| trigger | `p_voice` (contains the claimant) | `bucket_corroboration` over an evidence pool that structurally excludes claimants |
| threshold | `p_voice_hallucination / 2`, unnamed | `adjudication.corroboration_very_low`, named and in provenance |
| floor | none | `floors.MIN_EVIDENCE_WEIGHT` / `MIN_CORROBORATION`, validated `> 0` at policy load |
| name | `hallucination` — a cause | `uncorroborated` — the observation |

Four properties are load-bearing and each was reachable only by giving up the deletion:

1. **The weight *is* the measurement.** `max(floor, corroboration)` — the identity above the
   floor. Any other shape (a multiplier, an exponent, a sigmoid) inserts a constant nobody
   measured. The claim is "this source asserts speech here"; the independent evidence for that same
   event is already a probability in `[0, 1]`; that probability is how far the assertion carries.
2. **The measurement does not move when it acts.** The evidence pool contains only signals that
   observe presence directly, so the claimant can never be in it. The fixed point is reached in one
   step; re-measuring in a later round returns the same number. Under `p_voice` it could not have
   been.
3. **Unmeasured is not zero.** `corroboration is None` produces no factor and no discount, and a
   source absent from the weight map aggregates unweighted. A run with no informative presence
   voter is *inert*, and says so in `transcript.json`'s `evidence_pool` / `evidence_pool_rejected`
   — rather than condemning every word at once.
4. **Every withdrawal is re-derivable.** `provenance.evidence_weight_factors` records the
   measurement, the pool, the pooling rule, the map, the floor and the resulting weight, appended
   rather than overwritten so two rules acting on one vote both stay visible.

One decision survives, and it is deliberately at the rendering layer:
`fusion.corroboration.segment_min_corroboration` keeps a word out of `segments[].text` while
leaving it in `words[]` with its measurement. Keeping it in the readable transcript would let it
*win* — the deliverable would assert it and the text consumers would ingest it. Dropping it from
`words[]` would be the erasure. `withheld_word_indices` makes the rollup a pure function of
`words[]` plus one number, so the exclusion is re-decidable by re-reading one file. That number is
now the pressure point: raised far enough it reproduces purging in effect, which is why the
invariant a test pins is that `words[]` is untouched at *any* setting.

### The general finding: "vote" made exclusion feel natural

Line 795 already recorded that *"vote" is the wrong word for what L2 does* — that the metaphor made
a hand-set `0.4` derivation gate feel natural. This is the same defect, one step further along, and
it is the more expensive instance.

Statistical aggregation has exactly one lever: **weight**. There is no operation in a weighted fold
that removes a term; setting a weight to zero is a limit, not a separate act, and it is the one
value of the lever that is unrecoverable. Voting has a second lever — **eligibility**. Ballots are
counted or they are not; disqualifying one is an ordinary, reversible-sounding administrative act,
and it leaves the tally *correct*, because a disqualified ballot was never evidence about anything.

Once the data structure was named `Vote` and carried a `status` field, "exclude this vote" read as
routine. Nobody had to argue that a source's log-probability should stop informing the ASR axis;
they only had to argue that a ballot was invalid. The reasoning that would have been demanded of
`weight = 0` was never demanded, because the operation never appeared as a weight.

Three specific costs traceable to the metaphor, all of them found in this change:

- **`status` is a filter, so filters proliferate.** `active_votes()` tests `status == "active"`.
  Every consumer inherits that test, and each new status silently changes what five call sites
  measure. `_p2_trigger` counted `active_votes` to compute a coarse-voter share — a purge shrank
  its denominator, changing a rule that has nothing to do with adjudication.
- **A tally makes partial exclusion unthinkable.** The store already had three weight channels it
  never used: `payload["weight"]` (honoured by `_weighted_p_voice`),
  `per_source_confidence` (honoured by `aggregate_asr`), and `reliability` (honoured by
  `aggregate_speaker`). `contracts/belief-store.md` rule 4 *specified* weighted aggregation. The
  machinery to attenuate was present and wired; what was missing was the thought, because
  eligibility is binary and votes have eligibility.
- **Exclusion hides its own incompleteness.** A weight that fails to reach a sub-signal is visibly
  a bug. An *exclusion* that fails to reach a sub-signal looks like success — the vote is gone from
  `active_votes`, the invariant holds — which is how the pairwise phoneme family went on consuming
  purged transcripts without anyone noticing.

The rule that follows, and the one to apply to the rest of the store: **if a mechanism cannot be
written as a weight, it does not belong in an aggregation layer.** Where a decision genuinely has
to be binary — the segment rollup — it belongs at a rendering boundary, applied to a copy, with the
number that caused it recorded next to the thing it excluded.

`shadowing` survives this test and stays: a region-scoped vote *supersedes* a file-scoped one from
the same source about the same bucket. That is not two pieces of evidence with one removed; it is
one source's later, finer answer replacing its earlier, coarser one about the same question. The
superseded row is still on disk and the substitution is recorded. Worth stating explicitly, because
the next reader will reasonably ask why one status survived the argument that removed the other.


---

## An uncertainty axis is an aggregator

**D-16, decided 2026-08-02.** Recorded because it is a *definition*, and because the code
contained the contrary definition in enough places that it had started to read as natural.

An uncertainty axis aggregates across signals **and** across passes. There is therefore no such
thing as a per-pass axis: it is a category error, and the column name `within_pass_uncertainty`
was a contradiction in terms.

A **pass** is the same recording under a transform — `raw_16k` as recorded, `enhanced_16k` after
speech enhancement. Passes are a **perturbation sample**: a signal whose answer flips between them
has not earned its weight. So a pass is an input dimension to the aggregation, exactly like a
signal is — never an index on its output.

Consequences, definitional rather than negotiable:

- `L1/<pass>/uncertainty/<axis>.parquet` was wrong twice over: an axis at L1 (a fold L1 must not
  perform) *and* indexed by pass (an index an axis cannot have).
- A **vote** may be per pass — a signal measured on a pass is a legitimate per-pass measurement.
  An **axis** may not be. That is why `L2/round0/votes/<axis>.parquet` is keyed
  `(axis, bucket, source, pass, scope)` and `L2/round<N>/uncertainty/<axis>.parquet` is not.
- Perturbation stability is computed **from** per-pass per-signal measurements. That is the passes'
  entire purpose, and it needs no per-pass axis. `reliability.signal_stability` had been doing it
  correctly the whole time; `votes.compute_pass_deltas` was a second, wrong computation of the same
  idea, and its output had no reader anywhere.
- The pass dimension appears on a fused row only as the `contributing_passes` **column**.

*What this closed off.* The alternative was "L2 exposes a per-pass view alongside the fused one",
which line 811 above had left open. It is rejected: a per-pass view of an aggregator over passes is
not a view, it is a different and less-informed quantity, and having both is what produced two
numbers for one `(axis, bucket)` — register item 27. Anything a consumer wanted from the per-pass
view is available more directly: per-pass evidence from `L1/<pass>/signals/`, and what the second
pass *bought* from `L1/stability/`, in the form the fusion weights actually consume it.

*Why the wrong shape felt natural.* This is the same failure line 795 already recorded about the
word "vote": a data structure's name licensed an operation nobody would have argued for on its
merits. `AxisResult(pass_label, axis, rows)` made `(pass, axis)` the primary key of the workflow,
so `axis_results: dict[tuple[str, UncertaintyAxis], AxisResult]` threaded through eight modules and
every new consumer inherited the product type without having to justify it. The clearest symptom
was `scripts/analyze_audio.py` picking the *lower-uncertainty pass* as the run's bottom line —
treating a perturbation sample as two competing answers, and discarding the disagreement that was
the evidence. That reads as reasonable only if a pass can have an axis.

---

## `background_mask` is an output axis, not a derivative (correction)

**Recorded because the code still says three.** `types.UncertaintyAxis`, `adaptive.types.AxisName`
and `adaptive.belief.AXES` are all three-valued, and ten further sites carry the literal tuple
`("speech_presence", "speaker", "asr")`. Every one of them is wrong by one.

`background_mask` is a **fourth output axis** and carries the same estimates as the others:
uncertainty, probability, and confidence, per bucket. It is not a derivative, not a sidecar, and
not a mask-shaped special case. `fuse_axes` already emits it — `L2/round<N>/uncertainty/background_mask.parquet`
appears on every real run — so the *fusion* path has four axes while the *type* system, the belief
store and the adaptive loop have three.

Consequences of the miscount, each observed rather than predicted:

- The L1 guard's `AXIS_NAMES` omitted `background_mask`, so `L1/<pass>/background_mask.parquet`
  sat on disk while the guard reported the invariant held. A list of names cannot be complete for
  a set that grows; the guard has to key on **shape** — a fold across signals — which is what makes
  it correct for a fifth axis nobody has written yet.
- The adaptive loop iterates `AXES` for region proposal, convergence marking and the report, so
  `background_mask` is never proposed for intervention, never marked converged or irreducible, and
  never appears in `convergence.json`. It is fused and then dropped from the loop that acts on axes.
- `ATTENUATED_AXES` is a second, differently-sized subset (`speech_presence`, `asr`) with no stated
  relation to `AXES`. Two hand-maintained lists of axes will drift; which one is authoritative is
  not written down anywhere.

The design has said four throughout — `L2/round<N>/{speech_presence,speaker,asr,background_mask}/`
appears in the output layout — with `task` a punted fifth. The three-valued types are the residue
of the axis set before the mask became an axis, and they are the reason "four axes" keeps being
re-discovered by measurement instead of being a fact the code states once.

**What this requires.** One authoritative axis set, derived rather than restated, with the type
aliases generated from it; `background_mask` participating in region proposal, convergence and the
convergence report on the same terms as the other three; and `ATTENUATED_AXES` either justified
against it in writing or removed. Adding the fifth axis later must be one edit, not eleven.

**Closed 2026-08-02.** `audio_analysis/axes.py` is the declaration. Each axis carries the
properties consumers branch on — `harvested`, `attenuable`, `overlap_informed`, `calibrated`,
`rank`, `active` — and every subset is computed from them, so the justification for an exclusion
sits on the axis excluded. `AxisName` is `str` for the reason a perturbation name is: the set is
open. `task` is declared with `active=False`, which makes "the fifth axis is missing" a checkable
statement. The driver writes `background_mask` votes into the round's derivatives, so the belief
store ingests four axes and the loop's existing `for axis in AXES` covers proposal, marking and
reporting with no fourth branch anywhere. A source scan (`axes_test.py`) forbids writing the set
out by hand; the two survivors it found were genuine subsets and became declared properties.

---

## `L1/stability/` violates the contract too — no cross-pass evaluation at L1

*(Closed 2026-08-02. `L1/stability/<signal>.parquet` moved to the round's derivatives; the
run-level `signals.json` was **deleted** rather than moved, because the number it held is already
on every fused row as `weight_basis[signal]["stability"]` and one quantity in two places is one
quantity that can disagree with itself. What follows is the argument that put it there.)*

The per-pass axis was removed and stability was re-keyed by *signal*, which fixed the axis half of
the error and left the other half in place. `L1/stability/<signal>.parquet` and
`L1/stability/signals.json` (`{signal → instability}`) are **comparisons between passes**, and a
pass is an input dimension to the fold. Evaluating across it is a fold, by exactly the argument
that makes an axis L2's. Re-keying changed what the fold is indexed by, not that it is one.

**The rule, stated so it cannot be satisfied by renaming:** L1 emits one measurement per
`(pass, signal, bucket)` and never relates two passes. Anything whose value depends on more than
one pass — a delta, a flip rate, an instability, a weight derived from any of them — is L2's,
because "how much did this signal move under the transform" is a judgement about the signal, not an
observation of the recording.

That is also what makes the passes worth having. They are a perturbation *sample*; the sample is
L1's, the statistic over it is not.

**What moves.** `reliability.signal_stability` and everything downstream of it —
`measured_weights`, the fusion weights it feeds — are already L2 computations reading L1
measurements, which is correct. Only the *artifact* is misfiled: the stability parquets and
`signals.json` belong beside the round that used them (`L2/round<N>/stability/`), or nowhere at
all if the weights they produce are already recorded on the fused rows' `weight_basis`. Writing
both is the two-copies-of-one-quantity problem that item 26 was about.

**Why this kept surviving.** Three artifacts have now been found under `L1/` that L1 cannot
produce — the per-pass axis parquets, `background_mask.parquet`, and the stability files — and each
was found by a person reading the tree, never by a check. The guard that keys on *shape* would
catch the first two. It would not catch this one: a per-signal instability parquet has no `axis`
column and no fold-across-signals column, and looks like a measurement. The shape that gives it
away is **its keyspace**: it is keyed by signal alone while every legitimate L1 artifact is keyed
by `(pass, ...)`. A file under `L1/` that does not carry a pass, or that carries two, is a
cross-pass evaluation whatever its columns are named.

---

## D-17. The pipeline is a DAG of workflows, each declaring its inputs and outputs

Three rounds of guards were written against the violation last found and missed the next instance
of the same class — a name list that omitted the fourth axis, a regex an alias slipped past, a glob
that saw the workflow package but not `adaptive/`, three artifact rules that all pass on a genuine
per-pass axis table. Enumerating what is forbidden cannot terminate. **Declaring what is permitted
does.**

### Each stage declares a contract, and the guard checks conformance

L1, each L2 round, and final are each a **workflow**: a node with a declared set of inputs and a
declared set of outputs. The guard reads the declaration and the code, and fails when a stage
*reads* something outside its declared inputs or *writes* something outside its declared outputs.
That is complete by construction — it needs no list of axis names, no list of forbidden columns, and
it covers the fifth axis and the next perturbation before either is written.

The stages compose as a **DAG**: L1 → L2 round 0 → L2 round 1 → … → final. No cycles, so no stage
reads an artifact produced downstream of it. The two remaining contract violations are both cycle
edges — `final/` read by the pipeline, and a round reading a sibling updated within the same round —
and a DAG check names them as what they are rather than as separate defects.

### L1 — perturbations, and the signals that accumulate across them

A **perturbation** is a transform of the recording. `raw` is the identity perturbation and
`enhanced` is one more; the set is open, and **a future L2 round may propose a new one** — which is
why L1 must be re-enterable rather than a single up-front pass.

```
L1/
  raw/                      the identity perturbation
  perturbation/<k>/         each further transform, its parameters recorded
  signals/                  cumulative across raw + every perturbation — the L2 input
```

`signals/` is the accumulating artifact and the only thing L2 reads from L1. That is what makes the
"two passes" vocabulary a special case rather than the design: a pass was always a perturbation,
and hard-coding two of them is why `raw_vs_enhanced` could masquerade as a third pass.

Cross-perturbation evaluation stays out of L1 entirely — comparing two perturbations is a fold over
an input dimension, which is L2's, exactly as an axis is.

### L2 — one round tree, and what a round contains

*(Closed 2026-08-02 — this section records what was wrong and how it reads now.)* There were
**two** round trees, `L2/round<N>/` and `L2/rounds/<N>/`, written by two
producers with different numbering bases. One tree:

```
L2/round/<n>/
  derivatives/     mask, speaker allocation, ASR consensus, scene components
  estimates/       the axes — currently named "uncertainty", which names one of the four
                   quantities on the row rather than the thing itself
  timeline.png     the same figure the final timeline draws, so a round is comparable to the answer
  summary.json     what this round did (actions, interventions) and what it now estimates
```

`derive` runs before `estimate` in every round including round 0, and both read only round *n−1*
and L1's `signals/`.

### final — an extraction, not a computation

`final/` is the last round's estimates extracted, plus the summaries a human reads. It computes
nothing and is read by nothing. If a number in `final/` is not present in the last round, it was
computed at the wrong stage.

### What this buys that a rule list does not

- The guard question becomes "does this stage touch anything it did not declare", answerable from
  signatures and the artifact tree, with no list to keep current.
- A new axis, a new perturbation, or a new derivative needs no guard edit.
- The two round trees, `L1/stability/`, the per-pass axis, and `final/` being read are one finding —
  edges the DAG forbids — rather than four to be found separately.

### Where it lives, and what each half can see

`src/senselab/audio/workflows/audio_analysis/contracts.py` is the declaration and the only place
it exists. `src/tests/audio/workflows/audio_analysis/stage_contract_test.py` applies it.

Nothing restates the declaration. `dag_edges` builds the graph by asking which of a consumer's
declared reads could name one of a producer's declared writes, so a contract change moves the
graph with it; `unrolled_contracts(n)` turns the round into *n* nodes, which is what makes
ordering checkable at all — as one node, `L2_ROUND` reads and writes the same directory and is
trivially its own predecessor.

**The key is the mechanism that terminates.** Each declared artifact carries the tuple its rows
are indexed by, and three rules fall out of it: a key dimension the path does not supply must
appear as a column; a dimension outside the key must not appear at all; a non-interval dimension
may be spelled once. That yields *keyed by no perturbation* and *keyed by two* as consequences
rather than as entries, which is exactly the pair the shape-based guard could not express, and it
holds for the fifth axis and the third perturbation before either is written.

**Two guards, because neither subsumes the other.** The static one resolves aliases through the
AST to a fixpoint and cannot see a path handed to a helper as a parameter. The artifact one sees
concrete paths and cannot run without a run. Conformance is *subsumption*, not intersection: an
access the guard cannot prove conformant is not a permitted one. Intersection was tried, and it
silently unenforced the rule that matters most — `pass_dir(run_dir, stream) / "asr"` resolves to
`L1/*/asr`, whose `*` intersects the `signals` in `L1/signals/**`, so every `adaptive/` read of a
per-perturbation directory read as permitted.

**`KNOWN_DEVIATIONS` is the worklist, not an exemption list.** 82 entries, each naming the clause
it breaks and what closes it, and `dead_static_deviations` fails when an entry stops matching —
so closing a violation requires deleting its entry. Every rule has been observed failing against
a constructed violation, which the previous three rounds of guards had not been.

**What neither guard can see**, recorded so the silence is not read as absence: an L1 figure
rendered from an L2 belief conforms on paths and is still wrong, and a round reading a sibling
mutated earlier in the same round is in-memory state rather than an artifact.

`layer_boundary_test.py` keeps its rules 4 and 5 — a threshold-derived value naming its policy,
and a field read by a name the rows have. Those are not path rules and D-17 does not replace
them. Its rules 1–3 are superseded and go with the restructure, not with the mechanism.

---

## D-18 (plan). Replace guard-by-inspection with capability-passing I/O

Not implemented. Written first, because the current approach cannot be finished and the reason is
structural rather than a matter of effort.

### What exists today

`contracts.py` — 1,883 lines, 62 functions — plus `stage_contract_test.py` at 1,143. Three parts:

1. **Declarations.** Per stage (`L1`, `L2_ROUND`, `FINAL`), a set of `Artifact` patterns, each with
   permitted `suffixes`, a `key` naming the dimensions its rows are indexed by, and a `folded`
   licence. Breadth is refused at construction: a `**` with no content rule raises on the line that
   wrote it.
2. **A static guard.** An AST walk over every pipeline module. `_PathResolver` tries to evaluate
   path *expressions* to a declared pattern, tracking bindings across seven node types, and flags
   reads or writes outside the declaring stage's contract.
3. **A dynamic guard.** Walk a completed run tree; every file must match some stage's declared
   output, and every declaration must be satisfied by some file.

Plus a `Deviation` register — currently ~30 entries — waiving known violations by
`(module, op, pattern, reason)`.

### Why it cannot be finished

**The static half is attempting an undecidable problem.** "Which path does this expression evaluate
to?" is not answerable without running the program. Every bypass found is an instance of that, not
an oversight:

```
str(final_dir(run_dir) / "x.parquet")     os.path.join(run_dir, "final", "x.json")
str(run_dir) + "/final/x.json"            Path(f"{run_dir}/final/x.json")
PATHS["final"] / "x.json"                 (lambda d: d / "final")(run_dir)
```

Closing these means enumerating, and keeping current, four independent open sets: **binding forms**
(seven so far), **write verbs** (`to_parquet`, `savefig`, `write_table`, `to_feather`,
`os.makedirs`, …), **path constructors**, and **file formats**. Each list is complete only for the
bypasses already demonstrated. Four generations of this guard have shipped and each was defeated;
the fifth closed four holes and the attacker immediately found ten, including a plain logic bug
(`_check_call` returns after the first resolvable argument, so a call carrying two paths hides the
second).

**The dynamic half is never reliably green.** It needs a *complete* run. With none, it skips — so
the artifact rules are checked only against a fixture written in the same commit. With one, it
fails. Green means "no complete run exists", which is the least informative state.

**And the two halves can disagree**, with no rule for which wins.

### The reframing: this is a data-structure problem

The run directory is a **typed namespace**. The declarations already state the type. What is missing
is that nothing *holds* the type at the moment of use — a stage has ambient authority over the whole
tree (it is handed `run_dir`), and the guard tries afterwards to reconstruct what it did with it.

Invert it. Make the permitted set something a stage **holds**, and I/O something it can only do
**through** what it holds. Then the check is a runtime predicate, exactly as proposed:

```python
check(path, state, direction) -> None | raise
#   state     ∈ {L1, L2_ROUND(n), FINAL}
#   direction ∈ {input, output}
```

No AST analysis. No verb vocabulary. No path-expression resolution. Not because those problems were
solved but because they stop being asked: the only way to reach the run directory is through a
handle that knows the stage.

### The design

**One I/O module owns the run directory.** Every read and write of a run-relative path goes through
it. It already nearly exists — `io.py`, `layout.py`, `ctx.write_sidecar` — so this is consolidation,
not invention.

**A stage receives a capability, not a path.**

```python
class StageIO:
    """Scoped run-directory access. Holds the stage identity; resolves names, not paths."""
    def read(self, artifact: str, **dims) -> Any: ...
    def write(self, artifact: str, payload: Any, **dims) -> Path: ...
    def exists(self, artifact: str, **dims) -> bool: ...
```

`artifact` is a **declared name** (`"signals"`, `"estimates"`, `"round_summary"`), not a path
fragment; `dims` are the declaration's own dimensions (`perturbation=…`, `axis=…`, `signal=…`). The
handle composes the path from the declaration — so a stage cannot *name* an artifact that is not
declared, and the check is a dict lookup rather than an inference.

**The strong version: remove ambient authority.** A stage never receives `run_dir`. `StageIO` holds
it privately; `read`/`write` return payloads, not paths. Then bypassing requires reconstructing the
run root from somewhere else, which is conspicuous in review and detectable by one narrow static
rule instead of four open ones.

That single remaining static rule is closed and small: **no pipeline module may import `open`,
`pathlib.Path`, `pandas.read_*`/`to_*`, `pyarrow`, or `matplotlib.savefig`** — only `StageIO` may.
An import list is decidable; a path expression is not. This is the whole gain: the undecidable
question is replaced by one that a grep can answer.

### What each current mechanism becomes

| today | after |
|---|---|
| AST path resolution across 4 open sets | deleted — nothing to resolve |
| static read/write verb vocabulary | one import rule over a closed list |
| dynamic tree walk for undeclared files | kept: it answers a different question (what got written) |
| unproduced-declaration check | kept, and strengthened: it is now the only completeness check |
| `Deviation` register (~30 entries) | kept, but each entry becomes a *call site*, not a pattern — a waiver names the line that needs it |
| the two halves disagreeing | cannot: one mechanism, checked at the point of use |

### Migration, incrementally

1. **Declare artifacts by name.** Add the name → path-template mapping to `contracts.py`. Pure
   addition; the existing pattern list is derived from it so nothing forks.
2. **Build `StageIO` over the existing writers.** `io.write_*` and `ctx.write_sidecar` become its
   implementation. No caller changes yet.
3. **Convert one stage** — `L1` is the widest and least entangled — and delete the static rules that
   stage needed. Each conversion *removes* guard code; that is the signal it is working.
4. **Convert `L2_ROUND`, then `FINAL`.** `FINAL` should end up with `read` only.
5. **Drop `run_dir` from stage signatures** once no stage uses it. This is the step that makes the
   guarantee real, and it is checkable by signature inspection.
6. **Delete `_PathResolver`** and the verb/binding/constructor tables. Expect `contracts.py` to lose
   most of its 1,883 lines; the declarations are the part worth keeping.

Each step leaves the tree green and reduces guard surface. There is no big-bang commit.

### What this does not solve, stated plainly

- **A wrong value written to a permitted path.** The capability checks *where*, never *what*. The
  key/fold content rules stay, and they run inside `write` — which is strictly better than today,
  because they run before the bytes land rather than after a run completes.
- **Reads of things outside the run directory** — HF cache, model checkpoints, `~/.cache/senselab`.
  Out of scope, and correctly so.
- **A stage that declares an output and never writes it.** Only a completed run can show that, so
  the dynamic check survives for exactly this.
- **Order.** The DAG's acyclicity is a separate property: `StageIO` prevents `FINAL` being *written*
  by an earlier stage, but that L2 round *n* reads only round *n−1* is a claim about arguments, and
  stays a signature-level check.

### The argument for doing this

Four guard generations were defeated, each by a mechanism its author had not enumerated. The pattern
is not carelessness; it is that inspection-after-the-fact of an undecidable property cannot
terminate. Every hour spent extending the vocabulary buys one bypass and leaves the class intact.
Capability-passing changes what must be true from *"we thought of every way to write a path"* to
*"the run directory has one door"* — and the second is a property that can actually be held.

### D-18 revised. The capability is rooted, and the schema travels with the artifact

The plan above still had the guard's mindset: it built a checker and then listed what the checker
could not catch. Two of those "limits" were artefacts of under-designing, not properties of the
problem. **These are our own functions.** The goal is not to detect a bad write; it is to leave no
way to express one.

#### 1. A stage's handle is rooted at its own directory

`StageIO` is not "run-directory access that knows which stage you are". It is constructed **at the
stage's own directory** and holds no other root for writing:

```python
io = StageIO.for_stage(L2_ROUND, round=3)     # write root: L2/round/3/
io.write("estimates", payload, axis="speaker")  # -> L2/round/3/estimates/speaker.parquet
```

Writing outside is not checked, it is **unreachable**: the handle exposes no way to name a parent,
`..` and absolute paths are rejected at construction, and the root is private. There is no
`run_dir` in the stage's scope to build one from. The earlier plan's import rule — "no pipeline
module may import `Path` or `open`" — becomes a hygiene check rather than the load-bearing
guarantee, because even a module that imports `pathlib` has no run root to point it at.

#### 2. Reading another stage is a read-root, and the read-roots *are* the DAG

A stage needs upstream evidence: L2 reads `L1/signals/`. So a handle carries one **write root** (its
own directory) and zero or more **read roots** (its declared upstream stages), read-only:

```python
StageIO.for_stage(L2_ROUND, round=3, reads=[L1_SIGNALS, L2_ROUND(2)])
```

This is the DAG, made concrete rather than asserted. A stage's in-edges are literally the roots it
was handed. Acyclicity holds by construction — a handle can only be given roots for stages already
constructed — so `FINAL` being read by an earlier stage, and round *n* reading round *n* instead of
*n−1*, both become impossible rather than checked. The separate acyclicity test disappears.

#### 3. The schema travels with the artifact, so a wrong value cannot be written

`write` does not take an untyped payload. Each declared artifact names the **record type** its rows
are, and the writer's signature is that type:

```python
@dataclass(frozen=True)
class EstimateRow:            # the four quantities, kept distinct, plus attribution
    start: float; end: float
    uncertainty: float | None
    epistemic_uncertainty: float | None
    confidence: float | None
    variability: float | None
    triage_score: float | None
    contributing_signals: tuple[str, ...]
    contributing_passes: tuple[str, ...]

ARTIFACTS["estimates"] = Artifact(row=EstimateRow, key=("axis", "bucket"), ...)
```

`io.write("estimates", rows: Sequence[EstimateRow], axis=…)` then cannot carry a `perturbation`
column, because `EstimateRow` has no field for one. The per-perturbation axis — the category error
this whole thread has been chasing — stops being a thing a guard notices in a parquet and becomes a
field that does not exist on the type. Same for a fold column under L1: `SignalRow` has no
`uncertainty` field.

So "a wrong value on a permitted path" is not a residual risk. It was a residual risk *of a design
that accepted `Any` at the boundary*.

#### 4. Keys are derived, not maintained

The earlier plan had a hand-written `key` per artifact — another list to keep current, and the fifth
axis would have been missed again. Instead the key space is **generated from the structure that
already exists**:

- **L1 keys come from L1's own shape.** A signal measurement is indexed by
  `(perturbation, signal, bucket)` because that is what L1 produces — one measurement per signal,
  per bucket, per perturbation. The perturbation set is enumerated from the perturbation registry;
  the signal set from the harvesters that exist.
- **L2 keys come from the derivative functions.** A derivative's key is the dimensions of *its own
  function signature* — `mask` is keyed by bucket, `speaker allocation` by `(speaker, bucket)`,
  `estimates` by `(axis, bucket)` where the axis set is the one authoritative axis enumeration.
  Adding the punted `task` axis extends the key space with no declaration edit, because the
  enumeration is the source.

A key nobody wrote cannot drift from a structure nobody updated.

#### 5. What the registry is

A small table — the "lightweight database" — with one row per declared artifact:

| field | source |
|---|---|
| name | the call site's vocabulary (`"signals"`, `"estimates"`) |
| stage | which stage owns the write root it lands in |
| row type | a frozen dataclass; the writer's signature |
| key | derived from L1 structure or the L2 derivative's signature |
| path template | derived from stage + key, not written by hand |

Nothing in it is a pattern to match after the fact. It is the thing paths and schemas are *built
from*, so conformance is not a property to test — it is the only shape the code can produce.

#### 6. What genuinely remains

Two things, and both are about absence rather than transgression:

- **A declared artifact that no run produces.** Only a completed run shows it. This is the one place
  the dynamic tree walk survives, and it is now its whole job.
- **A stage that writes nothing at all.** Same class: the registry says what should exist; only a run
  says what did.

Everything else in the earlier plan's "does not solve" list was a consequence of accepting untyped
payloads and an ambient run root. Removing those two removes the list.

#### 7. Revised migration

The order changes, because the row types are what everything else derives from:

1. **Row types first.** `SignalRow`, `EstimateRow`, `DerivativeRow`, `VoteRow`. These already
   half-exist in `types.py`; make them the writer signatures.
2. **Derive the key space** from the perturbation registry, the harvester set, and the axis
   enumeration. Delete the hand-written `key=` tuples.
3. **`StageIO.for_stage`** with a private write root and explicit read roots.
4. **Convert L1**, then `L2_ROUND`, then `FINAL` (read-only). Each conversion deletes static rules.
5. **Remove `run_dir` from stage signatures.** After this the guarantee is structural.
6. **Delete `_PathResolver`**, the verb/binding/constructor tables, and the acyclicity test — the
   read-roots supersede it. `contracts.py` should end up a registry, not a checker.

---

## D-18. L1 workflow — pseudocode

Grounded in the 25 signals a real run emits today. Not implemented; this is the target shape.

### The registry L1 reads from

```python
# contracts.py — a registry, not a checker. Nothing here is a pattern to match after the fact;
# these are the things paths and payloads are BUILT from.

@dataclass(frozen=True)
class SignalRow:
    """One signal's measurement in one bucket of one perturbation. No axis. No fold."""
    start: float
    end: float
    measurement: Mapping[str, float]     # the tool's own quantities, in the tool's own units
    status: Literal["ok", "incomparable", "unavailable"] = "ok"
    # NOTE there is no `uncertainty`, no `axis`, no `confidence`. A fold has no field to land in,
    # so a fold under L1 is not a violation to detect — it is unrepresentable.

@dataclass(frozen=True)
class SignalProvenance:
    """What a different lab needs to reproduce the number from the audio alone."""
    units: Mapping[str, str]             # per measurement key
    model_id: str | None
    revision: str | None
    native_window_s: float
    native_hop_s: float
    tool_side_reduction: str | None      # a reduction the TOOL performed, named

ARTIFACTS["signals"] = Artifact(
    stage=L1, row=SignalRow, sidecar=SignalProvenance,
    key=derive_key(L1),                  # (perturbation, signal, bucket) — from L1's own shape
    template="signals/{signal}.parquet",
)
ARTIFACTS["perturbations"] = Artifact(stage=L1, row=PerturbationRow, key=("perturbation",),
                                      template="perturbations.json")
```

`key=derive_key(L1)` is computed from the perturbation registry and the harvester set — not written
out. Adding a harvester extends the key space; nothing is edited.

### Perturbations are declared transforms, and the set is open

```python
@dataclass(frozen=True)
class Perturbation:
    name: str                            # "raw", "enhanced", …
    transform: Callable[[Audio], Audio]  # identity for "raw"
    params: Mapping[str, Any]            # recorded, so the transform is reproducible
    requested_by: str                    # "run_start" | "L2/round/{n}"

RAW = Perturbation("raw", identity, {}, requested_by="run_start")
```

`requested_by` is the field that makes L1 re-enterable: a later L2 round may propose a perturbation,
and the artifact records who asked for it. Nothing about "two passes" is encoded anywhere.

### The node

```python
def run_L1(audio_path, *, perturbations, harvesters, grids, io: StageIO) -> None:
    """L1 measures. It does not decide, fold, threshold, or relate two perturbations.

    io is rooted at L1/ and holds NO other write root. There is no run_dir in this scope, so
    writing to L2/ or final/ is not forbidden — it is unreachable.
    """
    source = load(audio_path)

    # ── one directory per perturbation, and its parameters ────────────────────
    for p in perturbations:                       # RAW first; the rest in registry order
        variant = p.transform(source)
        io.write("perturbation_audio", variant, perturbation=p.name)
        io.write("perturbation_params", PerturbationRow.of(p), perturbation=p.name)

    io.write("perturbations", [PerturbationRow.of(p) for p in perturbations])

    # ── measure each signal on each perturbation, independently ───────────────
    # Nested this way round, not the other, for a reason: the inner loop never sees a second
    # perturbation, so it CANNOT compute a delta. Cross-perturbation evaluation is L2's, and here
    # it has no operand to reach for.
    for p in perturbations:
        variant = io.read("perturbation_audio", perturbation=p.name)

        for h in harvesters:                      # the 25 below
            if not h.applicable(variant, grids):
                io.write("signals", [], perturbation=p.name, signal=h.name,
                         sidecar=h.provenance(reason="unavailable"))
                continue                          # absent, recorded as absent — not zero-filled

            raw = cached(h.measure, variant, key=h.cache_key(variant, grids))

            rows = [
                SignalRow(
                    start=b.start, end=b.end,
                    measurement=raw.at(b),        # native units, verbatim
                    status="ok" if raw.covers(b) else "unavailable",
                )
                for b in h.native_buckets(variant, grids)   # the TOOL's grid, not a shared one
                if raw.has(b)                     # a bucket the tool said nothing about has no row
            ]

            io.write("signals", rows, perturbation=p.name, signal=h.name,
                     sidecar=h.provenance())
```

Three properties are structural rather than checked:

- **No fold.** `SignalRow` has no field an axis value could occupy.
- **No cross-perturbation evaluation.** The measuring loop holds one perturbation at a time.
- **No write outside L1.** `io` has one write root and it is `L1/`.

### The 25 harvesters, at their own resolutions

```python
HARVESTERS = [
  # frame-rate posteriors — 61.9 ms / 16.9 ms
  Harvester("frame_segmentation",               model="pyannote/segmentation-3.0",
            measurement={"activations": "probability"},   # per-speaker CHANNELS INTACT
            note="the collapse to one number is a fold; it is L2's to make"),
  Harvester("frame_segmentation_overlap_count", model="pyannote/segmentation-3.0",
            measurement={"n_active": "count"}),           # permutation-invariant, so well-defined
  Harvester("frame_brouhaha_vad",  model="pyannote/brouhaha", measurement={"p_speech": "probability"}),
  Harvester("frame_dispersion",    derived_from="frame_*", measurement={"sd": "probability"}),

  # scene classifiers — their own windows, NOT resampled here
  Harvester("ast",     model="MIT/ast-finetuned-audioset", window=10.24, hop=10.24,
            measurement={"label_scores": "probability"}),  # [{label: score}, …]
  Harvester("yamnet",  model="google/yamnet",              window=0.96,  hop=0.48,
            measurement={"label_scores": "probability"}),
  Harvester("sound_sources", derived_from=("ast", "yamnet"),
            measurement={"per_category_mass": "probability"},
            note="per source, per classifier — NOT averaged across them; that reduction is L2's"),

  # scene quality — dB / hertz / proportion, never a [0,1] score
  Harvester("scene_quality", measurement={
      "snr_brouhaha_db": "dB", "c50_brouhaha_db": "dB", "snr_spectral_gating_db": "dB",
      "snr_peak_db": "dB", "rolloff_95_hz": "hertz", "proportion_clipped": "proportion"}),
  Harvester("acoustic_hnr",   measurement={"hnr_db": "dB"}),
  Harvester("acoustic_lufs",  measurement={"lufs": "LUFS"}),
  Harvester("acoustic_level_above_floor", measurement={"excess_db": "dB"}),

  # diarization — one per model, spans in seconds
  *[Harvester(f"{safe(m)}", model=m, measurement={"speaker_label": "label", "span": "seconds"})
    for m in DIAR_MODELS],                        # pyannote community-1, sortformer, …

  # embeddings — 2.0 s / 50 ms (D-2). One per (diar model × embedding model).
  *[Harvester(f"{safe(d)}_{safe(e)}", measurement={"cosine": "distance"})
    for d in DIAR_MODELS for e in EMB_MODELS],
  *[Harvester(f"{safe(e)}_change_point", measurement={"cosine": "distance"})
    for e in EMB_MODELS],
  *[Harvester(f"embedding_silhouette_{safe(e)}", measurement={"silhouette": "score"})
    for e in EMB_MODELS],

  # ASR — one per model. Words at their own boundaries.
  *[Harvester(f"{safe(m)}", model=m, measurement={
      "text": "tokens", "word_spans": "seconds", "avg_logprob": "log-probability",
      "no_speech_prob": "probability", "token_entropy": "nats",
      "timestamp_source": "category"})            # native | bundled_aligner | external
    for m in ASR_MODELS],
]
```

Every entry names its units and its native window/hop. That is the whole L1 contract: a number, what
it is measured in, and at what resolution.

### What L1 does not contain, and where each thing went

| removed from L1 | now |
|---|---|
| `L1/<pass>/uncertainty/<axis>.parquet` | `L2/round/{n}/estimates/{axis}` — an axis is a fold |
| `within_pass_uncertainty` | gone; the name is a contradiction |
| `L1/stability/` | `L2/round/{n}/derivatives/stability/` — relating two perturbations is a fold |
| `background_mask` | `L2/round/{n}/derivatives/` — a verdict from six thresholds |
| `quality_snr`/`clip`/`reverb` scores | `L2` — the dB→[0,1] anchoring is calibration |
| `sound_sources` argmax + normalise | `L2` — a reduction across what the tools reported separately |
| `noise_floor` estimator selection | `L2` — choosing perceptual over recorder floor is a decision |

### Re-entry, when an L2 round asks for a perturbation

```python
def extend_L1(audio_path, *, new_perturbation, harvesters, grids, io: StageIO) -> None:
    """Add one perturbation and measure every signal on it. Additive; nothing is recomputed."""
    assert new_perturbation.requested_by.startswith("L2/round/")
    run_L1(audio_path, perturbations=[new_perturbation], harvesters=harvesters, grids=grids, io=io)
    io.append("perturbations", PerturbationRow.of(new_perturbation))
```

L1 stays a pure function of `(audio, perturbation, harvester, grid)`. A round may ask for more
evidence; it cannot ask L1 to conclude anything.

### The one check that survives

Everything above is structural. The single remaining question is **absence** — a declared artifact
no run produced:

```python
def unproduced(io: StageIO) -> list[str]:
    """Declared and never written. Only a completed run can answer this."""
    return [a.name for a in ARTIFACTS.for_stage(L1) if not io.any_exists(a.name)]
```

---

## D-18. Signal keys

Every L1 signal is identified by a tuple, not a filename. Grounded in the 25 files a real run emits.

### Why the flat name has to go

Today a signal is a mangled string:

```
nvidia_diar_sortformer_4spk_v1_speechbrain_spkrec_ecapa_voxceleb.parquet
embedding_silhouette_speechbrain_spkrec_ecapa_voxceleb_speechbrain_spkrec_ecapa_voxceleb.parquet
```

Model ids contain `_`, `/`, `.` and digits, and the separator is also `_`, so **`a_b` cannot be
parsed back into `(a, b)`**. The encoding is lossy in the direction that matters: a reader cannot
recover which model is the diarizer and which the embedder. The second name carries the same model
twice — that is not a typo, it is what a lossy encoding looks like when it is generated correctly
from a key nobody kept.

And the perturbation is **absent from the name entirely**, because it used to be the directory
(`L1/<pass>/…`). Once `L1/signals/` is cumulative across perturbations, the perturbation must become
a key *column*. That is not bookkeeping: it is the difference between a signal and a signal-on-a-
perturbation, and only the second is a measurement.

### The key

```python
SignalKey = tuple[Family, *Instance, Perturbation]
```

`Family` says what kind of measurement it is; `Instance` names the tool(s) that produced it — a
variable-arity slot, because a distance between two embedding models needs two names and a VAD needs
none beyond its own; `Perturbation` says which transform the audio had.

### The 25, keyed

| key | current flat name | native window / hop |
|---|---|---|
| `(frame, pyannote/segmentation-3.0, raw)` | `frame_segmentation` | 61.9 ms / 16.9 ms |
| `(frame_overlap_count, pyannote/segmentation-3.0, raw)` | `frame_segmentation_overlap_count` | 61.9 / 16.9 |
| `(frame, pyannote/brouhaha, raw)` | `frame_brouhaha_vad` | 61.9 / 16.9 |
| `(frame_dispersion, ·, raw)` | `frame_dispersion` | 61.9 / 16.9 |
| `(scene, MIT/ast-finetuned-audioset, raw)` | `ast` | 10.24 s / 10.24 s |
| `(scene, google/yamnet, raw)` | `yamnet` | 0.96 s / 0.48 s |
| `(source_mass, MIT/ast-finetuned-audioset, raw)` | — folded into `sound_sources` | 10.24 / 10.24 |
| `(source_mass, google/yamnet, raw)` | — folded into `sound_sources` | 0.96 / 0.48 |
| `(quality_snr, pyannote/brouhaha, raw)` | — bundled in `scene_quality` | 61.9 / 16.9 |
| `(quality_c50, pyannote/brouhaha, raw)` | — bundled in `scene_quality` | 61.9 / 16.9 |
| `(quality_snr, spectral_gating, raw)` | — bundled in `scene_quality` | 0.5 s / 0.25 s |
| `(quality_snr, peak, raw)` | — bundled in `scene_quality` | 0.5 / 0.25 |
| `(quality_rolloff, stft, raw)` | — bundled in `scene_quality` | 0.5 / 0.25 |
| `(quality_clipping, pcm, raw)` | — bundled in `scene_quality` | 0.5 / 0.25 |
| `(acoustic_hnr, opensmile, raw)` | `acoustic_hnr` | 60 ms / 10 ms |
| `(acoustic_lufs, pyloudnorm, raw)` | `acoustic_lufs` | 400 ms / 100 ms |
| `(acoustic_level_above_floor, band_floor, raw)` | `acoustic_level_above_floor` | 20 ms / 10 ms |
| `(diarization, pyannote/speaker-diarization-community-1, raw)` | `pyannote_speaker_diarization_community_1` | segment |
| `(diarization, nvidia/diar_sortformer_4spk-v1, raw)` | `nvidia_diar_sortformer_4spk_v1` | segment |
| `(speaker_distance, pyannote/…community-1, speechbrain/spkrec-ecapa-voxceleb, raw)` | `pyannote_..._speechbrain_spkrec_ecapa_voxceleb` | 2.0 s / 50 ms |
| `(speaker_distance, pyannote/…community-1, speechbrain/spkrec-resnet-voxceleb, raw)` | `pyannote_..._speechbrain_spkrec_resnet_voxceleb` | 2.0 / 0.05 |
| `(speaker_distance, nvidia/diar_sortformer_4spk-v1, speechbrain/spkrec-ecapa-voxceleb, raw)` | `nvidia_..._ecapa_...` | 2.0 / 0.05 |
| `(speaker_distance, nvidia/diar_sortformer_4spk-v1, speechbrain/spkrec-resnet-voxceleb, raw)` | `nvidia_..._resnet_...` | 2.0 / 0.05 |
| `(speaker_change, speechbrain/spkrec-ecapa-voxceleb, raw)` | `speechbrain_spkrec_ecapa_voxceleb_change_point` | 2.0 / 0.05 |
| `(speaker_change, speechbrain/spkrec-resnet-voxceleb, raw)` | `speechbrain_spkrec_resnet_voxceleb_change_point` | 2.0 / 0.05 |
| `(embedding_silhouette, speechbrain/spkrec-ecapa-voxceleb, ·, raw)` | `embedding_silhouette_..._ecapa` | 2.0 / 0.05 |
| `(embedding_silhouette, speechbrain/spkrec-ecapa-voxceleb, speechbrain/spkrec-ecapa-voxceleb, raw)` | `..._ecapa_ecapa` | 2.0 / 0.05 |
| `(embedding_silhouette, speechbrain/spkrec-ecapa-voxceleb, speechbrain/spkrec-resnet-voxceleb, raw)` | `..._ecapa_resnet` | 2.0 / 0.05 |
| `(asr, nyralabs/CrisperWhisper2.0_turbo, raw)` | `nyralabs_CrisperWhisper2_0_turbo` | word / ~30 s |
| `(asr, Qwen/Qwen3-ASR-1.7B, raw)` | `Qwen_Qwen3_ASR_1_7B` | word |
| `(asr, nvidia/canary-qwen-2.5b, raw)` | `nvidia_canary_qwen_2_5b` | word (externally aligned) |

`·` marks an unused instance slot. Each row above exists once **per perturbation**, so a two-
perturbation run has twice these keys, and a third perturbation adds a third set with no code edit.

### Three things the keying exposes

**`scene_quality` is six signals in one file.** It bundles measurements from four different
estimators at two different resolutions — brouhaha SNR and C50 at 61.9 ms, spectral-gating and peak
SNR, roll-off and clipping at 0.5 s. One row per bucket forces them onto one grid, which is a
resample L1 must not perform. Keyed properly they are six artifacts at three resolutions.

**`sound_sources` folds AST into YAMNet.** The current file averages a 10.24 s classifier against a
0.96 s one and normalises the result — a reduction across a dimension the tools reported separately,
at incompatible resolutions. Keyed by classifier, the fold has nowhere to happen at L1.

**`embedding_silhouette_ecapa_ecapa` is a real question, not a naming bug.** A silhouette of ECAPA
scored against ECAPA is a self-comparison; scored against ResNet it is cross-model. Both may be
wanted, but the flat name made them indistinguishable from a duplicate, and one of the three
silhouette files has an empty second slot — three files, three different arities, one name shape.

### How the key becomes a path

```python
def signal_path(key: SignalKey) -> str:
    family, *instances, perturbation = key
    return "signals/" + "/".join([family, *map(slug, instances), perturbation]) + ".parquet"

# (asr, nyralabs/CrisperWhisper2.0_turbo, raw)
#   -> signals/asr/nyralabs__CrisperWhisper2.0_turbo/raw.parquet
# (speaker_distance, pyannote/…community-1, speechbrain/spkrec-ecapa-voxceleb, enhanced)
#   -> signals/speaker_distance/pyannote__…community-1/speechbrain__spkrec-ecapa-voxceleb/enhanced.parquet
```

A directory per dimension rather than one mangled name, so the structure survives on disk and
`slug` only has to be injective per segment — not parseable, because nothing parses it back. The key
is the identity; the path is derived from it.

### What the key is for

- **The registry is generated.** `derive_key(L1)` enumerates families × instances × perturbations
  from the harvester set and the perturbation registry, so a new model or a new perturbation extends
  the key space with no declaration edit — and the fifth axis problem cannot recur at L1.
- **Provenance attaches to the key**, so units and native window/hop are per signal rather than per
  file-name convention.
- **L2 addresses evidence by key.** `votes_for(family="asr")` selects across models and
  perturbations without string matching, which is what today's `"::" in name` selectors are standing
  in for.

---

## D-18. Output type per signal key — and the gap between what tools generate and what L1 writes

Asked of a real run's `L1/signals/*.parquet`. The answer is that **L1 does not currently store any
tool's output**. It stores a reduction of one, resampled onto a common 0.1 s grid, with the native
window and hop recorded beside a value that is not at them.

### What the parquet actually contains

| key | tool generates | L1 writes | grid |
|---|---|---|---|
| `(frame, pyannote/segmentation-3.0, ·)` | `(n_frames × n_speakers)` probability **matrix**, channels permutation-arbitrary per inference | `{frame_mean, frame_std}` | 0.1 s |
| `(scene, MIT/ast-…, ·)` | **527-label score vector** per 10.24 s window, multi-label sigmoid | `{speech_label_mass: 0.623}` | 0.1 s |
| `(diarization, pyannote/…community-1, ·)` | **set of `(start, end, label)` spans**, variable length, labels arbitrary per model | `{covered_fraction, speaker_label}` | 0.1 s |
| `(asr, nyralabs/CrisperWhisper…, ·)` | **`ScriptLine` tree** — text, nested word chunks with times, per-segment `avg_logprob`/`no_speech_prob`/`token_entropy` | `{word_overlap_s, n_words, avg_logprobs[], …}` | 0.1 s |
| `(quality_*, four estimators, ·)` | six quantities at **three different resolutions** | one row, `units: "mixed"`, mostly `null` | 0.1 s |
| `(acoustic_hnr, opensmile, ·)` | series at 60 ms / 10 ms | `{hnr_db}` | 0.1 s |

`frame_segmentation` records `native_window_s: 0.0619`, `resolution_s: 0.0169` on a row spanning
`0.0 → 0.1`. The provenance describes a measurement the file does not contain.

### Each of those reductions is an L2 decision

- **`frame_mean`** collapses the per-speaker channels. This is the exact collapse that returned
  `1.0000` in 100% of frames on a clip that was half digital silence, and D-5 says the channels must
  stay intact because they are the only thing that distinguishes "two speakers at once" from
  "uncertain which of two". Storing the mean discards that, irrecoverably.
- **`speech_label_mass`** is a *selection* — which of 527 AudioSet labels count as speech — plus a
  sum. Both are choices, and the label set is a policy the row does not carry.
- **`covered_fraction`** reduces spans to a proportion, and `speaker_label` picks one label for a
  bucket that may contain two.
- **`avg_logprobs: []`** is a list because the bucket may span several ASR segments — a bucket grid
  imposed on an object that is not per-bucket.
- **`units: "mixed"`** is the honest admission that six quantities in three unit systems have been
  put in one row.

### The structural problem: one row type cannot hold L1

`SignalRow(measurement: Mapping[str, float])` fits only the scalar-per-bucket case. L1's outputs
are **six different kinds of object**, and four of them have no per-bucket scalar form:

```python
Series      # (n_frames,) at a fixed hop, one named quantity
            #   brouhaha vad/snr/c50, hnr, lufs, level_above_floor
Matrix      # (n_frames × n_channels) where channels are NAMED or ARBITRARY — the distinction
            # matters: brouhaha's 3 channels have fixed meaning, segmentation's speakers do not
            #   segmentation activations
Categorical # (n_windows × |vocabulary|) over a FIXED label set, plus the vocabulary itself
            #   AST (527), YAMNet (521)
Embedding   # (n_windows × n_dims) — 192 for ECAPA, 256 for ResNet
Spans       # variable-length [(start, end, label)] — not on any grid
            #   diarization, ASR word spans
Tree        # ScriptLine: text with nested chunks, per-node scores
            #   ASR transcripts
```

A bucket grid is meaningful for `Series` and `Matrix`. It is a **projection** for `Categorical` (a
0.96 s or 10.24 s window is not a 0.1 s bucket), and it is a **reduction** for `Spans` and `Tree`
(there is no natural per-bucket value of a transcript). Forcing all six into one tabular row is what
produced every reduction in the table above.

### What L1 should store instead

Native shape, in the file format that shape belongs in:

| kind | stored as | key |
|---|---|---|
| `Series`, `Matrix` | parquet, one row per **native frame**, one column per channel + the channel names in metadata | `(family, *instances, perturbation)` |
| `Categorical` | parquet, one row per **native window**, `label_scores` as `[{label: score}, …]`, vocabulary + version in metadata | same |
| `Embedding` | parquet or npy, one row per **native window**, vector as a fixed-width list | same |
| `Spans` | parquet, one row per **span** — `start`, `end`, `label` — no grid at all | same |
| `Tree` | JSON, the `ScriptLine` verbatim with `timestamp_source` | same |

Then `SignalRow` is not one type but a small union, and each carries its own resolution because it
*is* at its own resolution. `native_window_s` stops being a claim beside a resampled value and
becomes a description of the rows present.

### Where the reductions go

Every one of them is a named L2 derivative, taking native L1 and producing a bucketed estimate:

```python
# L2/round/{n}/derivatives/
project_categorical(ast_native, grid, labels=SPEECH_LABELS)   -> speech_label_mass   # selection + sum
pool_channels(segmentation_native, how="noisy_or")            -> p_speech            # the collapse, named
count_active(segmentation_native, threshold=θ)                -> overlap_count       # permutation-invariant
cover(diar_spans, grid)                                       -> covered_fraction    # span -> proportion
words_in(asr_tree, grid)                                      -> word_overlap_s      # tree -> bucket
resample(series_native, grid, how="mean")                      -> per-bucket series
```

Each names the choice it makes, records it on the row, and can be re-made without re-running a
model — which is the whole point of the split, and is impossible while the reduction is what got
stored.

### Answering the question directly

**`pyannote/segmentation-3.0` generates a `(n_frames × n_speakers)` matrix of per-speaker speech
probabilities at 61.9 ms / 16.9 ms, whose speaker channels are permutation-arbitrary within each
inference.** Two facts follow and neither survives the current storage: a *count* of active channels
is permutation-invariant and therefore well-defined (which is why J1 is answerable and J4 needs
rounds), and any pooling to a single p(speech) is a choice among several — `mean`, `max`, `noisy-or`
— that changes the answer. L1 stores `frame_mean`, which has silently made that choice, at a
resolution the model never reported.

---

## D-19. Speaker capacity is a per-tool ceiling, and it must travel to L2

Every speaker-aware tool has a **maximum number of speakers it can represent**, fixed by
architecture, not by the audio. It is not a tuning parameter and a tool does not report when it runs
out — it simply assigns what columns it has.

| tool | capacity | note |
|---|---|---|
| `pyannote/segmentation-3.0` | **3 per 10 s chunk** | powerset over 3 (7 classes) → 3 per-speaker columns; channels permutation-arbitrary per chunk |
| `nvidia/diar_sortformer_4spk-v1` | **4** | |
| `mago-ai/ultra_diar_streaming_sortformer_8spk_v1` | **8** | not yet integrated; the option when samples exceed 4 |
| `pyannote/speaker-diarization-community-1` | **unbounded** | clustering pipeline, not a fixed-width head |

**L1 records the capacity on every speaker-aware signal**, beside the measurement. It is provenance
of the same kind as units and window length: without it a reader cannot distinguish *"3 speakers
active"* from *"3 active and the model had no fourth column"*.

**L2 must use it when combining.** A capacity-bounded tool asked about a recording that exceeds its
bound does not fail, it produces a confident wrong answer — so a count posterior fused across tools
of different capacity is biased toward the smallest. Concretely: `segmentation-3.0` cannot report a
4th concurrent speaker, so its overlap-count contribution must be treated as *censored at 3* rather
than as evidence against a 4th. Censoring is not the same as absence and not the same as a bound
being met.

Current corpus is ≤3 concurrent speakers, so this is latent rather than active. It is recorded now
because it is invisible in the output when it does bite.

### `pyannote/segmentation-3.0` is to be removed

pyannote 4.x has moved to the community diarization model; `segmentation-3.0` is the previous
generation and should not remain a signal.

**What depends on it today**, so the removal is not a silent capability loss. Ten modules reference
it, and two capabilities are keyed specifically on its *per-speaker frame channels*:

- `joint.overlap_count_posterior` (J1) — returns `None` unless `channel_format != "single"`. This is
  the Poisson-binomial count-of-active-speakers posterior.
- `joint.per_speaker_presence` (J4) — same guard. This is what convergence criterion **C2** (the
  `S_k` ↔ channel assignment) is measured from; with no per-speaker posterior, C2 is unmeasured and
  therefore blocks convergence.

Brouhaha's VAD head is `channel_format == "single"` and cannot supply either. Diarization spans give
speaker identity but not a per-frame per-speaker probability, so a count posterior built from them
would be a count of *decisions*, not a distribution.

**Open question before removal:** does `community-1` expose frame-level per-speaker activations
(its internal segmentation head, or a `speaker_probabilities` output), or must J1/J4/C2 be rebuilt
on spans and accept that they then measure decisions rather than posteriors?

### Resolved: diarization emits spans, and L2 derives occupancy from them

No frame-level per-speaker posterior is needed. Every diarization tool emits what sortformer and
`community-1` already emit — `(start, end, speaker_label)` spans at their own boundaries, no grid —
and L2 derives frame or bucket occupancy by projecting them. `segmentation-3.0` is removed with no
replacement source, because there is nothing to replace: the object L1 owes is a span set.

**This corrects J1 rather than degrading it.** `overlap_count_posterior` built a Poisson-binomial
over `segmentation-3.0`'s per-speaker channel probabilities, treating them as independent Bernoullis.
They are not independent — they are a powerset conversion, where the classes are mutually exclusive
by construction and the per-speaker columns are derived from them — so the independence the
Poisson-binomial assumes was never there. It was one model's internal confidence, dressed as a
distribution over speaker count.

The honest uncertainty about "how many speakers are active here" is the same as for every other axis
in this design: **disagreement across models.** Each diarizer's spans give a count at time *t* — how
many of its spans cover *t* — and the spread across diarizers of differing capacity is the
uncertainty. That is a measured disagreement rather than an assumed independence, and it composes
with D-19's censoring: a tool at its capacity contributes a *lower bound*, not a point.

So J1, J4 and C2 are rebuilt on spans:

```python
# L2/round/{n}/derivatives/
occupancy(spans, grid)        -> per-bucket [(speaker_label, covered_fraction), ...]  per tool
count_at(spans, t)            -> int, per tool; censored at that tool's capacity
count_posterior(per_tool_counts, capacities) -> distribution from cross-tool spread
assignment(spans, clusters)   -> S_k <-> tool-label binding (C2), from spans not channels
```

L1 stores, per diarization tool: the span set, the tool's speaker capacity (D-19), and nothing
reduced. A span set has no grid, so there is no resolution to record and no projection to get wrong.

---

## D-20. The key's first element is the TARGET, not the mechanism

Corrects D-18's signal keys. They were written as `(family, *instances, perturbation)` where
`family` conflated three different things: a **resolution** (`frame`), a **domain** (`scene`), and
occasionally an actual target (`asr`, `diarization`). That is why `scene_quality` could bundle six
quantities and `frame_*` could group a probability with two dB measures — the first element was not
naming what was measured, so it could not tell them apart.

```python
SignalKey = tuple[Target, Tool, Perturbation]
#   Target       what the tool is measuring — the quantity, not how it was obtained
#   Tool         which tool measured it, at its own resolution
#   Perturbation which transform the audio had
```

**Brouhaha is three signals, because it targets three things** in one forward pass:

```
(speech,   pyannote/brouhaha/VAD, raw)    probability   — it is a speech detector
(snr,      pyannote/brouhaha,     raw)    dB            — a scene-quality measure
(c50,      pyannote/brouhaha,     raw)    dB            — a different scene-quality measure
```

Sharing a forward pass is an implementation fact and not a reason to share an artifact. What makes
them one call is caching; what makes them three signals is that they answer three questions.

### Why target-first is the load-bearing choice

**All voters on one target share a first element.** That is what makes cross-tool disagreement
computable without string matching: `signals_for(target="speech")` returns every speech detector —
brouhaha's VAD, a diarizer's occupancy projected to a grid, an ASR's word coverage — regardless of
mechanism or resolution. Under the old keying those lived under `frame_*`, `diarization` and an ASR
model name, and only a `"::"`-style selector could gather them, which is what the code does today.

**It makes the axes derived rather than declared.** An axis is the fusion of every signal sharing a
target (and of targets a policy groups). The "one authoritative axis set" problem, and the fifth
axis needing eleven edits, both dissolve: the axis set is the set of targets, enumerated from the
signals that exist.

**A bundle cannot form.** `scene_quality` was possible because `scene` named a domain that six
quantities could all claim. With the target first, `snr`, `c50`, `rolloff` and `clipping` are four
targets and cannot share a file — and each keeps its own native resolution, which is what the bundle
destroyed by forcing one grid.

**Mechanism moves to provenance, where it belongs.** `frame`, `window`, `span`, `tree` describe the
*shape* of the output and belong beside `units` and `native_window_s` — they are how the measurement
is represented, not what it is of.

### Keys settled so far

| key | output shape | capacity |
|---|---|---|
| `(speech, pyannote/brouhaha/VAD, p)` | series, 61.9 ms / 16.9 ms, probability | — |
| `(snr, pyannote/brouhaha, p)` | series, same grid, dB | — |
| `(c50, pyannote/brouhaha, p)` | series, same grid, dB | — |
| `(speaker_spans, pyannote/speaker-diarization-community-1, p)` | span set, no grid | unbounded |
| `(speaker_spans, nvidia/diar_sortformer_4spk-v1, p)` | span set, no grid | 4 |
| ~~`(·, pyannote/segmentation-3.0, ·)`~~ | removed — pyannote 4.x moved to community diarization | ~~3~~ |

`speaker_spans` is the target both diarizers share, which is exactly what lets their counts be
compared and their capacities censored against each other (D-19).

### `(scene_labels, MIT/ast-finetuned-audioset, p)` — one key, top-k configurable

One signal, not several. AST's target is the **label distribution itself**; which labels count as
speech, music, machine or environment is an L2 mapping over that distribution, not something L1
decides. That keeps the AudioSet→category map (`data/audioset_source_map.json`) where it can be
changed without re-running the model, and it means a new category needs no new signal.

```
(scene_labels, MIT/ast-finetuned-audioset, p)
  shape       window, 10.24 s / 10.24 s (non-overlapping)
  measurement label_scores: [{label: score}, ...] — top-k, descending
  units       probability   (sigmoid, multi-label — NOT softmaxed; softmax across 527 classes
                             structurally suppressed secondary background categories)
  provenance  k, vocabulary id + size (527), model revision
```

**`k` is configurable, default 7.** The stored tail is truncated, so this bounds what L2 can ask:
label mass over a set whose members fall outside the top *k* is not recoverable without re-running.
`k` therefore travels **on the row**, so a consumer can tell a zero mass ("this label scored below
the 7th") from a real absence ("this label scored nothing") — the same absent-vs-zero distinction
this design turns on everywhere else. A category map whose labels routinely fall outside the top *k*
is a reason to raise *k*, and that is now a visible decision rather than a silent truncation.

The `speech_label_mass` reduction and the 0.1 s projection both leave L1: the first is a selection
plus a sum over a policy label set, the second asserts 102 independent values inside one 10.24 s
window.
