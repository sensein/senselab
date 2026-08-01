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
   native resolution. **H3 open** — transcripts are aligned to audio, not to each other.
3. **Joint estimation** J1–J9, with J4 as a joint space (D-7). **J3 done**
   (`speaker_identity.speaker_count_posterior`). **J5–J6 done** (`background_mask.py`,
   `noise_floor.py`, `sources.py`). **J8 done** (`reliability.py`, cross-pass stability as measured
   weight). **J9 done** (`invariance.py`, `--invariance-probe`). **J1 done** (`joint.overlap_count_posterior`, wired as the
   `<signal>::overlap_count` speaker sub-signal). **J2 done**
   (`joint.speaker_change_series`, wired as `<embedding_model>::change_point`). **J4, J7 open** —
   per-speaker presence as the `S_k` ↔ channel joint space, and phoneme-vs-transcript agreement.

   J1 was answerable before the rest because a *count* of active channels is invariant to the
   channels' arbitrary ordering, while anything naming *which* channel is whom waits on the joint
   space rounds resolve. It is also the signal the old noisy-or collapse destroyed: `1 − Π(1 − p_k)`
   answers "is anyone speaking" and discards how many. The posterior is built per frame and then
   pooled — two speakers taking turns within a bucket average to 0.5 on each channel, which as a
   per-bucket calculation would report a 25% chance of an overlap that never occurred.

   J2 compares each embedding window against the one a whole window-width later, not the adjacent
   one. This is D-2's warning made operational: at the 50 ms hop two adjacent 2 s windows share
   97.5% of their audio, so their distance measures phonetic drift rather than speaker identity.
   Lagging by the window width makes the two sides disjoint spans meeting at a boundary, and the
   fine hop then buys localisation of that boundary — which is what D-2 said it buys, and it does
   not buy independent samples, so neighbouring boundary scores must not be counted as separate
   evidence. Each bucket takes the *strongest* boundary it contains rather than the mean: a sharp
   change surrounded by continuation is a change, and averaging would dilute it away. The distance
   is read through the speaker axis's existing calibration band rather than a new anchor, because a
   raw cosine of 0.2 is not evidence of anything — same-speaker embeddings sit in a 0.1-0.3 noise
   floor from phonetic variation alone.
4. **Link functions** and TTS-fitted calibration (D-8, D-9).
5. **Rounds and convergence** (D-10 – D-12, both guards).
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
