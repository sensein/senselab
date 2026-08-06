# L1 post-processing register: every interpretation to be moved to L2

Companion to `l1-signal-contract.md` (the principle) and `layered-architecture.md` (the full
five-layer design and the decisions D-1 – D-15 governing it). This one is the complete list of
places the current code violates the principle, so each can be reviewed and moved deliberately
rather than in bulk.

Decisions from `layered-architecture.md` that dispose of entries below: **D-3** drops items 8–9's
signals outright in favour of absolute replacements; **D-5** requires item 16's channels intact;
**D-9** determines what items 1–2 become once softened by measured reliability rather than voting
`(1, 0)`.

**Why one-by-one rather than a sweep.** Every entry below is a decision someone made for a reason,
and the reason is usually still valid — it is the *layer* that is wrong, not the arithmetic. Moving
a threshold to L2 unexamined would carry its unstated assumptions along with it. Several entries
turn out not to survive review at all (see `quality_uncertainty`, item 21), and a bulk move would
have preserved them.

## What counts as post-processing

A step is post-processing, and therefore L2's, if it does any of:

- **Thresholds** a continuous measurement into a decision (`>= 0.5`, `speaks: bool`).
- **Rescales** to `[0, 1]` against an anchor — fixed (`(25 - snr)/20`) or data-derived (percentile
  rank). The anchor is a claim about what counts as normal, which is calibration.
- **Reduces** across a dimension the tool reported separately (max/noisy-or over speaker channels,
  mean over frames, argmax over 527 labels).
- **Selects** among estimators (`brouhaha else mean of DSP`), which is fusion.
- **Inverts** a measurement into a badness score (`1 - rolloff/nyquist`, `1 - no_speech_prob`).
- **Resamples** onto a reporting grid that is not the tool's native resolution.

A step is *not* post-processing if it recovers the measurement the model actually produced:
applying the model's own trained output activation, converting sample indices to seconds,
stitching sliding-window chunks back into one timeline, or naming units.

## Evidence that the layering is the defect

Brouhaha's three heads, measured directly on synthetic inputs whose correct answers differ
maximally (`/tmp/bprobe`, CPU, `BrouhahaInference`):

| input | VAD median | SNR median | C50 median |
|---|---|---|---|
| digital silence, 8 s | 0.0001 | 43.5 dB | 31.1 dB |
| white noise, −40 dBFS, 8 s | 0.0028 | −5.0 dB | 22.9 dB |
| clean TTS speech, 21.5 s | 0.9980 | 70.1 dB | 59.8 dB |
| 4 s speech + 4 s silence | speech half **0.9981** / silence half **0.0167** | — | — |

Within-file speech-vs-silence discrimination is **+0.9815**. The heads are sound and have full
dynamic range — 75 dB of SNR span across inputs. What reached the parquet was `quality_snr = 0.0`
in every bucket, because `clip((25 − 70.1)/20, 0, 1)` is zero. **The measurement was never the
problem; the clamp on top of it destroyed a working signal.**

A corrected earlier finding, recorded because it shaped several decisions: an intermediate
analysis reported brouhaha's VAD as having −0.003 discrimination and being "unusable". That test
compared loud against quiet *frames within continuous speech* and used dBFS as ground truth for
speech presence. Short inter-word gaps are not non-speech, and a frame VAD with 6 s of LSTM
context is correct to read through them. The signal was fine; the test was wrong.

## The register

Status: `open` = still in L1. Each entry names the L1 replacement — what the tool should emit
instead — and the L2 question the moved step becomes.

### Presence voters — dissolved into L1 evidence + an L2 link stage

`harvest_presence_votes` returned `{"speaks": bool, "native_confidence": float}` per model. Both
fields are conclusions. Per the governing instruction *"there is no presence at L1 just good
signals"*, the module had no L1 role. It is now `speech_presence.harvest_speech_presence_evidence`
(measurements in native units) plus `speech_presence_link.link_speech_presence` (an L2 stage under
a named `SpeechPresencePolicy`). `PassHarvest.speech_presence_votes` became
`speech_presence_evidence`; every consumer that needs beliefs calls
`speech_presence_link.votes_for_harvest`.

| # | site | current post-processing | L1 should emit | L2 question | status |
|---|---|---|---|---|---|
| 1 | diarization voters | `diar_speaks_in_window` → `speaks` bool | `covered_fraction` + `speaker_label`, no verdict | does any segment cover this bucket, and whose? | **closed** |
| 2 | ASR token overlap | `token_overlaps_window` → `speaks` bool | `word_overlap_s` (union, clipped) + `n_words` | does enough of a word land here? | **closed** |
| 3 | ASR hallucination gate | `speaks and not (nsp >= 0.5)` — threshold + override | per-chunk `no_speech_probs`, unpooled | is a transcript over probable silence trustworthy? | **closed** |
| 4 | Whisper confidence | `mean(exp(avg_logprob))` → bucket `native_confidence` | per-chunk `avg_logprobs`, unpooled | which pooling, and how does logprob map to belief? | **closed** |
| 5 | `::no_speech_prob` voter | `speaks = nsp < 0.5`, `nc = 1 − nsp` — threshold + inversion | `no_speech_prob` + measured `segment_span_s` | same as #3 | **closed** |
| 6 | AST | top-1 argmax over 527 labels, then `label in speech_labels` | full label→score map per native window | which categories are present, and is any of them speech? | **closed** |
| 7 | YAMNet | top-1 argmax over 521 labels, then `label in speech_labels` | full label→score map per 0.48 s hop | same as #6 | **closed** |
| 8 | `acoustic_loudness` | per-pass **percentile band** p10→p75 → `[0,1]` → direction flip | replaced by absolute `lufs` (D-3) | what loudness counts as audible here? | **closed** |
| 9 | `acoustic_spectral_activity` | per-pass percentile band on `spectralFlux_sma3` | replaced by `level_above_floor_db` (D-3) | what excess above the measured floor counts as activity? | **closed** |
| 10 | `acoustic_hnr` | fixed 2→10 dB ramp; low maps to `p = 0.5` (abstain) | `hnr_db`, units dB | what HNR indicates voicing, and when is it uninformative? | **closed** |
| 11 | `ppg_voice_fraction` | per-frame argmax, count `!= "<silent>"`, ÷ n, then `>= 0.5` | `mean_silence_posterior` + dispersion + frame count | what silence posterior means speech? | **closed** |
| 12 | `embedding_silhouette` | cluster all windows, silhouette coefficient, `>= 0.5` | *withdrawn from this axis* — see below | wrong question: geometry, not voicing | **closed by removal** |
| 13 | frame posteriors | bucket-mean over frames, then `>= 0.5` | `frame_mean`, `frame_std`, `channel_means`, `resolution_s` | how do frames aggregate to a bucket, and where is the cut? | **closed** |
| 14 | `frame_dispersion` | `clip(2 × mean(std), 0, 1)` — ×2 rescale + clip | dispersion in probability units, unrescaled | how does dispersion enter a belief? | **closed** |
| 15 | coarse-voter demotion | hand-set `coarse: True`, `weight = 0.25` when grid < 0.5 s | measured `native_window_s` / `resolution_s` per signal | how should resolution mismatch be weighted? | **closed** |
| 16 | segmentation-3.0 reduction | noisy-or over per-speaker channels (was `max`) | per-speaker activation matrix, channels intact | how do per-speaker activations combine? | **closed** |

Item 16 is the sharpest case, and its diagnosis changed once measured. The model reports one
activation per speaker, and the collapse returned **exactly 1.0000 in all 1070 buckets** on the
Higgs conversation — but the cause was the output format being *misidentified*, not the choice of
reduction. `segmentation-3.0` declares `powerset=True` while pyannote 4.x returns per-speaker
columns; a single active speaker makes those rows sum to 1.0, so the row-sum heuristic took the
powerset branch and computed `1 − data[:, 0]`, treating speaker#1 as the no-speaker class. Reading
the declaration against the output width instead took discrimination from 0.0000 to **+0.9364** on
a speech-then-silence probe. Closed; see D-5 in `layered-architecture.md` for the measurements.

Items 1 and 12 together are why diarization rows render one colour: L1 never emits the labels, so
nothing downstream can distinguish speakers.

Items 6–7 turned out to need less than expected: the full `labels`/`scores` lists already reach
`sound_sources.py`, which maps them through the checked-in `audioset_source_map.json`. Only
`presence.py` was reducing to top-1. It now uses `window_label_mass`, the subset's share of total
score mass — a window topped by `Music` at 0.40 with `Speech` second at 0.38 previously voted a
confident *no speech*, discarding 0.38 of speech evidence.

### Scene quality (`quality.py`)

| # | site | current post-processing | L1 should emit | L2 question | status |
|---|---|---|---|---|---|
| 17 | `quality_snr` | `clip((25 − snr_db)/20, 0, 1)` → **0.0 in every bucket** | `snr_brouhaha_db`, units dB | what SNR counts as clean for this task? | **closed** |
| 18 | `quality_reverb` | `clip((30 − c50_db)/35, 0, 1)` → **0.0 in every bucket** | `c50_brouhaha_db`, units dB | what C50 counts as dry? | **closed** |
| 19 | `quality_bandwidth` | `clip(1 − rolloff/nyquist, 0, 1)` — inversion | `rolloff_95_hz`, units hertz | is this band-limited for the sample rate? | **closed** |
| 20 | `quality_clip` | `clip(proportion_clipped, 0, 1)`, renamed as degradation | `proportion_clipped`, units proportion | how much clipping matters? | **closed** |
| 21 | `quality_uncertainty` | `clip(std(snr_estimates)/15, 0, 1)` | **deleted, not moved** — see below | n/a | **closed** |
| 22 | `primary_snr_db` | brouhaha, else mean of DSP estimators | all three estimators, unreduced | which estimator to trust where? | **closed** |
| 23 | silence gate | `rms < 1e-4` → nulls all quality columns | `rms`, reported as its own signal | where is quality undefined? | **closed** |
| 24 | grid broadcast | nearest-analysis-window value copied to each reporting bucket | values resampled via `resolution.resample_series` | how to resample to the reporting grid? | **closed** |

**Item 21 fails on its own terms, not just on layering.** It takes the standard deviation of
brouhaha's SNR head, `spectral_gating_snr_metric`, and `peak_snr_from_spectral_metric` — three
quantities that are not the same measurement. Brouhaha reads ~70 dB on clean TTS where the DSP
metrics use different noise-floor definitions entirely, so their spread reflects *definitional
disagreement*, not measurement uncertainty. Divided by a 15 dB reference it pins at 1.0
structurally, and it would do so on perfect audio. Per `statistics.py`, `variability` is the
dispersion of *repeated measurements of one quantity*; these are three different quantities. This
is not a variability estimate and should not be re-derived at L2 under a new name. What is worth
keeping is the underlying observation — that the estimators disagree — reported as the estimators
themselves (item 22), letting L2 decide whether disagreement is informative.

Item 24 is the one still open in this group, and it interacts with the resolution work:
`resolution.py` exists and is not yet wired in, so each reporting bucket still copies its nearest
analysis window's value rather than being resampled from the native 0.5 s / 0.25 s analysis grid.

Items 17-23 closed together: `quality.harvest_quality_measurements` emits the seven signals in
native units (dB, hertz, proportion, plus uncalibrated `rms`) with per-signal provenance and
status, and `degradation.scene_degradation` applies the anchors at L2, where a fitted calibration
profile can replace them. Item 22's selection among estimators survives as an explicit
`SNR_PREFERENCE` order that also records `snr_source` — choosing an estimator is legitimately L2's
job; doing it silently at L1 was the defect.

## Cross-cutting

- **Nothing emits through `signal.measurement(...)` yet.** The provenance envelope exists
  (`signal.py`: units vocabulary, model, revision, resolution, window, reduction, backend, status)
  and is unused. Until L1 emits through it, "what units is this?" has no answer in the data, only
  in the code that produced it.
- **`resolution.py` is not wired in.** Declared native resolutions (17 ms frame signals, 10 ms
  acoustic, 480 ms YAMNet, 10.24 s AST) and `resample_series` are implemented but unused; L1 still
  reports on the reporting grid.
- **`acoustic.py` LUFS is not wired into `presence.py`**, and its `loudness_confidence` dB ramp is
  itself an L1 interpretation (item 8's replacement) that belongs at L2.
- **`scene_quality_coupling` is null in all 1070 rows** — separate defect, not a layering issue;
  tracked in `l1-signal-contract.md` open items.
- **Fixtures.** Clean TTS cannot validate SNR/C50 because it genuinely sits at 70 dB / 59.8 dB.
  Degraded fixtures exist at `/tmp/bprobe/` (`noisy_{0,10,20,40}.wav`, `reverb_{0.3,0.8}.wav`) and
  should become checked-in test fixtures, since the useful range of items 17–18 is only observable
  on them.

## Review order

Grouped so each group can be validated by one measurement rather than a full e2e run.

1. **Items 17–24, scene quality.** Self-contained, and the degraded fixtures make the fix
   verifiable immediately. Item 21 is a deletion, not a move.
2. **Item 16, then 13–14, frame signals.** Unblocks per-speaker presence and identity; the
   saturation is measurable on the existing run.
3. **Items 6–7, scene classifiers.** Keeping full posteriors also serves background
   characterization, which currently re-derives what argmax threw away.
4. **Items 8–10, acoustic.** Requires deciding the absolute anchor: LUFS for loudness, measured
   noise floor from `noise_floor.py` for flux. HNR already has a defensible absolute anchor.
5. **Items 1–5, 11–12, 15.** Largest surface: dissolves `presence.py` into an L2 stage. Do last,
   once the signals it consumes are emitting raw.

## Findings from closing items 8-9

**The two questions the percentile rank was conflating.** Gain scaling changes no signal-to-noise
ratio — it lifts the source and the floor together — so a measure of "is something happening beyond
the room's floor" must be gain-invariant, while a measure of "how loud is this recording" must not
be. One within-file rank cannot answer both, and answered neither. The replacements are
`lufs` (absolute, gain-sensitive) and `level_above_floor_db` (relative, gain-invariant), verified
by a −12 dB probe: the excess measure moves < 1 dB, LUFS moves 12.0 dB.

**The excess measure must abstain, not deny.** A low excess has two indistinguishable causes:
nothing is happening, or a source runs through the whole recording and has been absorbed into its
own floor estimate, since the floor is a percentile of this file's own frames. Voting `False` there
would make the signal contradict correct models on any recording without pauses — measured on a
wall-to-wall AM tone, where it read `speaks=False` against ground truth. It now maps low excess to
`0.5` (uninformative) and only asserts on the upper half of the range, the same asymmetry `hnr_db`
already uses. LUFS retains the ability to claim absence, because −90 LUFS is unambiguous.

**A test fixture was self-contradictory and only these voters could reveal it.**
`compute_uncertainty_axes_test.py` built its happy path from `torch.zeros(...)` — digital silence —
paired with mocked diarization and ASR reporting speech. Every voter read the mocks, none read the
audio, so the contradiction was invisible and the suite asserted low presence uncertainty. The
absolute voters read the waveform, correctly dissented, and presence uncertainty rose to 0.62. The
fixture now carries audible content, with a separate `_silence_audio` helper for tests where
absence of signal is the point.

## Findings from the partial dissolve

**`speaks: bool` was not the only loss — the measurement behind it was.** Rather than restructure
the vote contract in one step, every vote now carries the evidence it was derived from, so L2 can
re-decide without re-running a model. `covered_fraction` is the clearest case: a segment overlapping
5% of a bucket and one covering all of it both set `speaks=True`, and the difference matters most at
segment boundaries — exactly where speaker uncertainty is highest. Coverage is a *union* over
segments, not a sum, so two speakers talking at once cannot report more than a bucket's worth.

**Item 14 closed outright.** `frame_instability = clip(2 * mean(std), 0, 1)` doubled a dispersion
because the std of a value bounded in `[0, 1]` is at most 0.5. That turns a dispersion into
something that reads like a probability, and the clip then hides where the rescale was wrong. L1 now
reports dispersion in probability units and `votes.py` maps it once, explicitly, via
`MAX_PROBABILITY_STD` — the rescale is a modelling claim about how temporal instability should
contribute to doubt, which is L2's to make and to change.

**Items 1-5, 13, 15 are `partial`, not closed.** The measurements now travel, but the thresholds
still fire in L1 and `speaks` is still what the aggregator reads. Fully closing them means moving
`harvest_speech_presence_votes` into an L2 stage, which changes the contract consumed by
`aggregate.py`, `votes.py`, `fuse.py` and the adaptive loop. Recorded as partial rather than closed
so the remaining half is not mistaken for done.

Item 11 (`ppg_voice_fraction`) remains fully open: it is a *derived* signal rather than a tool
output, and per D-7 it should be recomputed at L2 from L1 posteriors.

**Item 12 (`embedding_silhouette`) closed by removal, 2026-08-06.** It is no longer a speech-presence
voter at all, so there is no threshold left to dissolve. The row above said `closed` while this
paragraph said `fully open`; both were describing a signal that should not have been on this axis.

A silhouette coefficient measures cluster *geometry* — how well-separated the clusters are — computed
over every embedding window including silent ones. Silence embeds consistently too, so well-separated
silence scores well; the measure cannot distinguish "a coherent speaker is here" from "this window is
coherently not speech". Measured on `english_conversation_higgs_audio_v2`:

| | value |
|---|---|
| its doubt across 214 buckets | 0.4022 – 0.4996, **stdev 0.0227** |
| its fusion weight | **1.0** — the highest of all fifteen presence signals |
| every informative voter's weight | 0.78 – 0.91 |
| the four diarizers / three recognizers / brouhaha VAD | **0.0000** doubt |
| published presence doubt | 0.0682 |
| without this voter | 0.0385, and 47 of 214 buckets can reach zero |
| without it and `acoustic_hnr`'s abstain-at-0.5 | 0.0204, 93 of 214 |

Two findings worth keeping separate from the fix:

1. **`_directed(score)` with no ramp.** Every neighbouring linker anchors its measurement —
   `_link_hnr` ramps between `policy.hnr_low_db` and `hnr_high_db`, `_link_level_above_floor`
   likewise. This one passed a `[-1, 1]` coefficient straight through as a `[0, 1]` confidence, so an
   ordinary good silhouette of 0.58 became 0.42 of permanent doubt.
2. **Stability-based weighting rewards a constant.** `reliability.signal_stability` measures
   cross-pass `|Δ|`, and a near-constant is perfectly stable, so the least informative voter earned
   the most weight. That is not specific to this signal and is *not* fixed by removing it — any
   future uninformative-but-stable signal will be weighted the same way. Open.

Nothing was lost. The clustering already reaches the **speaker** axis as a first-class diarization
source (D-20): `compute.harvest_pass` injects a synthetic `embedding_silhouette/<model>` diarizer
built from `derive_window_clusters`, whose spans and cluster ids feed
`attribution.speaker_assignment_doubt`. Asking one clustering to also vote on presence counted a
single body of evidence twice, on the axis where it was least apt. The vote's `cluster_id` payload had
no consumer — label reassignment reads the synthetic diarizer's spans.


## Findings from the full dissolve (items 1-5, 13, 15)

**Two thresholds turned out to be pooling choices in disguise.** Whisper's bucket confidence was
`mean(exp(avg_logprob))` over the contributing chunks. By Jensen's inequality that strictly exceeds
`exp(mean(avg_logprob))` whenever the chunks disagree, so these are two different statistics and one
had been chosen silently, inside a getter named as though it merely read a value. L1 now emits the
per-chunk list and the choice is `SpeechPresencePolicy.asr_confidence_pooling` — the default
reproduces the old numbers, but a reader can now see that a choice exists.

**"Coarse" is not a property of a voter.** The harvester hand-marked AST, YAMNet, the ASR token
voter and the Whisper `no_speech_prob` sibling as `coarse: True`, then applied a fixed `0.25` weight
whenever the reporting grid was finer than 0.5 s. Both halves were wrong in the same way: a voter is
only coarse *relative to the grid it is reported on*, so the comparison needs two numbers that are
never both known at L1. AST's 10.24 s window is stretched across 20 buckets at 0.5 s and across none
at 10.24 s. The demotion now reads a measured `native_window_s` against the reporting width.

This changes behaviour at the historical 0.5 s grid, deliberately. The old rule (`grid < 0.5 s`) fired
on no default run at all, so AST at 10.24 s and the Whisper segment voter at ~30 s were counted at
full weight against 0.5 s buckets they could not resolve. Under the ratio rule they are demoted and
YAMNet (0.96 s) is not. The old cutoff was arbitrary; this one is measured.

**The ASR voter had no declared window at all.** It was marked coarse with `native_window_s: None`,
which under a resolution-driven rule would mean "not coarse". The fix was to *measure* it:
`claim_span_s` is the unclipped union of the transcript spans reaching a bucket, and
`segment_span_s` the same for the segments whose scalars were pooled. Both are facts about the
transcript, so the demotion is now derived from evidence rather than from a hand-set flag.

**Three consumers were reading beliefs from what is now a measurement.** `support.py` (does the
audio corroborate this claim), `fuse.py` (round fusion) and `adaptive/belief.py` (the round-1 vote
store) all need verdicts, and all three read the harvest field directly. They now call
`votes_for_harvest`, which links at the grid recorded on the harvest that produced the
measurements — passing the grid separately is how a coarse voter ends up demoted against the wrong
bucket width.

**A test fixture spelled beliefs where the pipeline now carries measurements.** `votes_test.py` and
the adaptive tests built `PassHarvest(speech_presence_votes=[{"votes": {"m1": {"speaks": True}}}])`.
Translating them to `{"evidence": {"m1": {"covered_fraction": 1.0}}}` is not a mechanical rename:
it forces each fixture to say which *measurement* produces the belief it was asserting, which is
the property the layering exists to make explicit.

Item 11 (`ppg_voice_fraction`) remains open; item 12 is closed by removal (above). It was reduced
inside the harvester rather than recomputed at L2 from posteriors and embedding vectors. Per D-7
they are derived signals, and they move once `PassHarvest` carries the underlying measurements.


## Findings from closing items 10-11

**The PPG signal had the scene classifiers' defect, one model over.** `ppg_voice_fraction` took the
argmax phoneme per frame and counted the ones that were not `<silent>`. That is the same reduction
as the AST / YAMNet top-1 (items 6-7): each frame's whole distribution collapses to a hard 0 or 1,
so a frame the model called silent at 0.6 confidence votes exactly as strongly as one it called
silent at 1.0. L1 now emits `mean_silence_posterior` — the model's own posterior mass on `<silent>`
averaged over the bucket — with its dispersion and frame count; L2 takes the complement.

`ppg_argmax_per_frame` stays, and is still correct where it is used: the ASR axis compares phoneme
*sequences* (`ppg_sequence_per_in_window`), and there the argmax label is the quantity being
compared rather than a reduction of one.

**Item 10 closed as a side effect of the speech-presence split**, not as separate work: the 2→10 dB
ramp and the abstain-at-low asymmetry now live in `SpeechPresencePolicy.hnr_low_db` / `hnr_high_db`
and `_link_hnr`, and L1 emits `hnr_db` alone. The asymmetry is preserved verbatim — whispered and
distorted voice both read low, so a low HNR cannot be distinguished from silence and must not be
voted as absence.

**All 24 register items are closed.**

## Findings from closing item 24

**Nearest-copy is neither of the two correct rules.** `resolution.py` had stated them and gone
unused: finer-than-the-bucket is an *integral*, coarser is a *hold*. Copying the nearest analysis
window is neither. Going coarser it kept one window and discarded the rest, and which one survived
was an artefact of where the bucket centre happened to fall — at the default 0.5 s grid against a
0.25 s analysis hop that is half the measurements thrown away in every bucket.

**A failed estimator in one window used to fail the whole bucket.** Windows where a signal produced
nothing are now dropped from that signal's series rather than carried as `NaN`, so a bucket averages
over the windows that did report. Previously, if the nearest window happened to be the one where
the estimator failed, the bucket reported `None` even though neighbouring windows had measured it.
That was found by the test written for the resampling, not by the resampling itself.

**Provenance had to survive the resample.** On a grid finer than the analysis hop the same
measurement now repeats across several buckets. The declared `resolution_s` is what stops a
consumer counting those repeats as independent evidence, so it stays the *analysis* resolution and
the reduction applied is recorded next to it as `grid_reduction`.

## Findings from closing item 12

**"Pure" meant model-free, not dependency-free.** Clustering at L2 brings numpy and scikit-learn
into the aggregation path, which was the reason to hesitate. But the property `aggregate_pass`
actually needs is that re-aggregating is deterministic and touches no model, waveform or file — and
a computation over vectors already in hand satisfies that. `votes.py` now says so explicitly, since
the old wording ("no imports beyond stdlib") would have made this look like a violation rather than
a clarification.

**The derivation answers two questions, and only one was surviving.** `cluster_pass_speakers`
returns both a per-window voice score *and* a per-window cluster assignment; the harvester used the
`silhouette_voice_score` wrapper, which discards the labels, and then thresholded what was left. So
the layer violation was also a lost signal: the cluster id is what a later stage needs to assign and
re-assign speaker labels, and re-deriving it there would risk two stages disagreeing about the
clustering they are each reasoning over. `derive_window_clusters` now returns the whole result, and
the vote carries `cluster_id` alongside `silhouette`.

**Nearest-centre matching, with the window width carried.** An embedding window (2 s by default) is
usually wider than a reporting bucket, so several buckets legitimately share one window's answer.
The width rides along as `native_window_s`, which means the coarse-voter demotion sees it — under
the old hand-set `coarse: True` the same fact was asserted rather than measured.

---

## Item 25 — L1 emits per-axis estimates (**closed** 2026-08-02)

**The rule.** A signal may report its own uncertainty: that is the signal's final measurement, in
its own terms. L1 must not report an **axis** estimate, or an axis uncertainty/confidence. Folding
signals into one per-axis number *is* the answer, and the answer is L2's.

**The violation.** L1 writes `L1/<pass>/uncertainty/{speech_presence,speaker,asr}.parquet`, each
row carrying `within_pass_uncertainty` — a per-axis fold across that pass's signals. The L1
timeline then plots those three axes as rows. Once a fold is persisted under `L1/` and drawn on a
timeline labelled `speech_presence_uncertainty`, it has acquired the authority of a measurement,
which is exactly how the folds this register exists to remove survived in the first place.
`fuse.py`'s own module docstring already names this as the defect: *"Level 1 computes signals and
each signal's own uncertainty. It must not decide the answer. The per-pass fold it used to perform
is a within-pass diagnostic … folding early is precisely how one saturated sub-signal came to pin
an axis at 1.0 while two independent diarizers, both embedding models and the per-speaker presence
track all agreed nothing had changed."*

**Why this is not a cleanup.** `within_pass_uncertainty` has **76 references across 17 modules**:

| module | refs | reads it as |
|---|---|---|
| `votes.py` | 18 | the per-pass axis value |
| `adaptive/belief.py` | 17 | the belief store's per-bucket state |
| `disagreements.py` | 6 | ranking across axes |
| `types.py`, `io.py` | 10 | the row schema itself |
| `adaptive/{ls_final,loop,convergence,evaluate,regions,interventions,fusion,plot}.py` | 15 | region proposal, convergence marks, LS tracks, evaluation |
| `global_summary.py`, `labelstudio.py`, `fuse.py`, `plot.py`, `analyze_audio.py` | 10 | summary + rendering |

The adaptive loop's entire belief store is keyed on an L1-computed axis value. So the violation did
not stay at L1 — every consumer that reads a per-pass axis number is reading a fold L1 should never
have made, and deleting the producer without re-pointing them at L2's axes would remove the loop's
state.

**Shape of the fix.** L1 keeps per-signal uncertainties and drops the axis parquets; the L1 timeline
plots signals in native units (`L1/signals.png` already does this). Consumers that need an axis
value read L2's `L2/round<N>/uncertainty/<axis>.parquet`. The adaptive belief store is the hard
part: it is per-pass and per-round, and L2's axes are fused across passes, so "the speaker axis on
the raw pass at round 2" has no L2 equivalent today. Whether that quantity should exist at all —
or whether the belief store should be keyed on L2 axes plus per-signal L1 evidence — is a design
question, not a mechanical substitution.

**Found by:** a real run, asking why the L1 timeline had axis rows at all.

---

## Item 26 — `final/` stores L1 evidence, and the pipeline reads it (**closed** 2026-08-02)

`final/summary.json` is 6.9 MB, of which 4.8 MB is a `passes` key holding per-pass model output.
Everything in it already exists on disk under `L1/<pass>/`:

| inlined key | bytes/pass | already on disk as |
|---|---|---|
| `features` | 1,849,387 | `features.json`, `features/` |
| `yamnet` | 76,165 | `yamnet.json` |
| `asr` | 15,632 | `asr/` |
| `alignment` | 8,071 | `alignment/` |
| `diarization` | 6,878 | `diarization/` |
| `ast` | 4,679 | `ast.json` |
| `background_mask`, `background_sources` | 1,583 | `.json` + `.parquet` |

So it is duplication, not information — and the duplicate is the copy the pipeline reads:
`speaker_identity` takes `summary["passes"]`, and the adaptive loop's artifact-driven path
reconstructs a finished run from it.

**This is item 25 in the other direction.** Item 25 has L1 computing a value that belongs to L2;
this has `final/` storing evidence that belongs to L1 *and being consumed as an input*. D-15 says
`final/` holds only the converged state, and `layout.py` calls it "the summary, the timeline, and
the consensus deliverables" — a deliverable is not something a later stage reads to rebuild state.

It also explains why these boundaries have been easy to cross without noticing: with two copies of
the same bytes, nothing enforces the boundary. A consumer reaching into `final/` gets exactly what
one reading `L1/` would, so nothing breaks and the violation leaves no trace.

**Fix.** Drop `passes` from `final/summary.json`; both readers read `L1/<pass>/` like every other
evidence consumer. That takes the file from 6.9 MB to roughly 6 KB.

**Check first:** whether the inlined copies are byte-identical to the files or transformed on the
way in. If they differ, the readers depend on the transformation, and that has to move too — a
path substitution would then silently change what they see.

**Found by:** asking why `final/` contained anything that something else reads.

---

## Item 27 — two parallel speaker-axis pipelines, and the timeline draws the uncoupled one (**partially closed** 2026-08-02)

Measured on one run, after H1 made cross-axis coupling work:

| source | rows | round 0 → 1 (median) | coupled rows |
|---|---|---|---|
| `fuse_axes` → `L2/round<N>/uncertainty/speaker.parquet` | 72 | 0.646 → 0.739 | 72 / 72 |
| adaptive belief → `L2/rounds/<n>/belief/speaker.parquet` | 144 | 0.667 → 0.833 | none — no coupling path exists |

`adaptive/plot.py` draws the second. So per-speaker presence visibly fails to move speaker
uncertainty in `final/timeline.png` even though the fused axis now couples on every bucket: the
plot is showing a different lineage.

They are not two views of one quantity. Different grids (72 vs 144, the latter carrying both
streams), different provenance — the belief store ingests **L1's per-pass axis folds**, which is
item 25's quantity — and only one of them is reachable by D-11's coupling. Two different numbers
sharing a name.

This is the consequence item 25 predicted, made concrete: because L1 emits per-axis folds, a second
axis lineage exists at all, and the adaptive loop was built on it. Fixing item 25 and fixing this
are the same work — there should be one speaker axis, fused at L2, and the belief store should read
it rather than re-deriving from per-pass folds.

Until then, any statement about "the speaker axis" has to say *which one*, and the timeline is the
misleading case because it is the artifact a human actually looks at.

**Found by:** noticing that per-speaker presence still did not move speaker uncertainty in the
final timeline after coupling was verified working.

**Why two lineages exist at all.** They should not. The layers specify one: L1 emits per-signal
measurements, L2 fuses them into axes, `final/` holds the converged answer. There is no second
speaker axis in that design and nowhere for one to live. The duality is not two views of a
quantity — it is item 25's violation with a consumer attached. L1 emits a per-pass axis fold it
should not compute, and the belief store was built to ingest precisely that. Delete the fold and
the belief store has nothing to read but L2's axes, at which point item 27 dissolves rather than
being fixed.

The interim `_fused_axis` overlay in `adaptive/plot.py` is scaffolding for the defect: justified
only while the violation is real and invisible on the artifact people trust, and to be deleted with
it, not maintained.


---

## What closed items 25–27, and what remains

*Recorded 2026-08-02. The correction that drove the work is stated once, in
`layered-architecture.md` under "An uncertainty axis is an aggregator"; this records what
changed in the code.*

### The correction

**An uncertainty axis IS an aggregator.** It aggregates across signals *and* across passes.
There is therefore no such thing as a per-pass axis: a pass is an input dimension to the fold,
never an index on its output. `within_pass_uncertainty` was a contradiction in its own name.

A *pass* is the same recording under a transform. The two passes are a **perturbation sample**:
a signal whose answer flips between them has not earned its weight. Perturbation stability is
therefore computed *from* per-pass per-signal measurements and needs no per-pass axis — which
`reliability.signal_stability` had been doing correctly all along, in the live path, while
`votes.compute_pass_deltas` computed the same idea the wrong way into a file nothing read.

### Item 25 — closed

| was | is |
|---|---|
| `L1/<pass>/uncertainty/<axis>.parquet` | `L1/<pass>/signals/<signal>.parquet`, native units |
| `L1/stability/raw_vs_enhanced/<axis>.parquet` (no reader) | `L1/stability/<signal>.parquet` + `signals.json` |
| `UncertaintyRow` (axis + 8 fold columns) | `SignalRow` (no axis, no fold) |
| `AxisResult(pass_label, axis, …)` | `SignalResult(pass_label, signal, …)` + `FusedAxis(axis, …)` |
| `PassLabel` including `raw_vs_enhanced` | `Literal["raw_16k", "enhanced_16k"]` |
| `votes.aggregate_pass` — 3 folds × N passes | `votes.link_pass` — link under a named policy, fold nothing |
| `votes.compute_pass_deltas` — \|raw axis − enh axis\| | deleted; `reliability.stability_rows` per signal |
| 3 axis folds reachable from two lineages | one, `fuse.fuse_axis`, which already took `buckets_by_pass` |

`disagreements.json` ranks the fused axes on `triage_score` and has no `pass` field. The LS
bundle has three axis tracks attached once, plus per-pass per-signal evidence tracks — which is
where "what did each model say on each pass" is legitimately served. `final/timeline.png` draws
one line per axis with `epistemic_uncertainty` shaded beneath and a per-signal stability strip,
replacing the raw/enhanced overlay that rendered the category error on the artifact a human
actually looks at. `compute_pass_global_summary` became `compute_run_global_summary`, and
`best_pass` is gone: picking the lower-uncertainty pass treats a perturbation sample as two
competing answers and throws away the disagreement that is the evidence.

**Four live bugs surfaced and fixed while closing it**, each of which had been invisible because
it failed silently:

1. **Presence stability had never been measured.** `reliability._COMPARABLE_FIELDS` matched no key
   `harvest_speech_presence_evidence` emits, so `signal_stability(axis="speech_presence")`
   returned `{}` on every real run and every presence signal kept weight 1.0. It floored
   correctly — absent is not a discount — which is exactly why nobody noticed. Stability is now
   measured on `fuse.per_signal_uncertainty` of the linked belief, the same quantity the fold
   consumes, so weight and value can no longer be derived from different things. A voter that
   reports only a direction (`speaks: True`, no confidence) falls back to that direction, because
   its flip is what stability is asking about.
2. **`disagreements.json` pointed at paths that did not exist** — `<pass>/uncertainty/…` and
   `uncertainty/raw_vs_enhanced/…`, while the writer used `L1/…` and `L1/stability/…`.
3. **`frame_dispersion` never reached the artifact path**, so P2's frame-instability trigger read
   `None` for every bucket and one of its two documented triggers was structurally dead. It is now
   a persisted L1 signal.
4. **`write_final_uncertainty` linked presence under a different policy** than round 0 did
   (packaged `DEFAULT_POLICY` vs `policy_from_params`), so the two folds read the same
   measurements under different thresholds.

### Item 26 — closed

`final/summary.json` no longer carries `passes` (~5.7 MB → ~6 KB). The index later stages
actually need — duration, audio signature, input path — is `L1/passes.json`, where evidence
lives. `build_final_outputs` returns the diarization instead of the loop re-reading
`final/diarization.json` one statement after writing it; the adaptive timeline takes the
transcript from its caller.

Three readers were **already broken and silent** about it, all from the earlier layout split:

- `evaluate.py` read `final/speech_presence.parquet` and `<out_dir>/rounds/`; both had moved to
  `L2/`, so the evaluation harness crashed on every run that reached it.
- `ls_final.py` read `<run_dir>/labelstudio_*` and `<run_dir>/disagreements.json` (moved into
  `final/`) and `<out_dir>/rounds/` (moved to `L2/`). Absent read as "not found", which is
  indistinguishable from a run with no bundle — so the final LS tracks and
  `disagreements_resolved.json` had quietly stopped being produced.
- `_final_belief_index` keyed on `(stream, axis, start, end)` and matched entries by their `pass`
  field. Neither exists now; re-keyed `(axis, start, end)`.

### Item 27 — partially closed

The second lineage's *source* is gone: the belief store no longer ingests an L1 axis fold. It
ingests the linked evidence at the vote level (`L2/round0/votes/<axis>.parquet` on the artifact
path, `PassHarvest` in process), where `(axis, bucket, source, pass, scope)` is a legitimate key —
a signal measured on a pass is a per-pass measurement. `parity_check` became `replay_check`: the
old one compared two different implementations of the fold, so a mismatch could not distinguish
"the store missed an input" from "the two folds disagree", and the in-process path could not run
it at all. The replay rebuilds each bucket from what is persisted and proves the property the
architecture correction turns on — estimates are re-derivable, so the store need not persist them.

**What remains.** The store still computes its own per-`(stream, axis, bucket)` fold, so two
numbers for one `(axis, bucket)` are still producible. Collapsing that is blocked on one concrete
change named in the consumer map: **`fuse_axis` takes `weights: Mapping[str, float]` — one weight
per signal for the whole axis — while the store's withdrawals are per `(bucket, source)`.** Until
`fuse_axis` accepts per-`(bucket, signal)` weights, a per-bucket withdrawal cannot be expressed in
it and the store has to keep its own fold. Everything else follows mechanically after that:
drop the `stream` index from `BeliefState`, `regions.propose_regions`, `convergence`,
`loop.uncertainty_mass` and `_write_round_belief`; move `p_voice` onto the fused presence axis as
a directional confidence (it is *not* `fuse_axis`'s `confidence`, whose proposition is different);
use `fuse_axis`'s `epistemic_uncertainty` rather than the store's second definition of the word;
and delete `adaptive/plot._fused_axis`, the comparison overlay whose own docstring calls it
scaffolding for the defect.

Two silent fallbacks found while tracing and **still open**, both in `adaptive/fusion.py`:
`r.get("speech_presence_confidence", r.get("p_voice"))` always takes the fallback because the
key is only ever set inside `row["meta"]`, and `(r["meta"] or {}).get("elected_stream", stream)`
always emits the fusion stream because `elected_stream` is written to the *region*, never to
`row["meta"]`.

## Items 25 and 26 were not closed, and how that was found (2026-08-02)

The four guards written to hold items 25–27 shut all **passed while the rules were being
violated**, which is the same failure signature as every defect in this register: a check that
resolves to nothing is indistinguishable from a check that found nothing.

- The item-25 guard walked a tmp tree it built from two writers, so it never saw what a run
  puts on disk, and it matched a hard-coded `AXIS_NAMES` list — leaving any axis added later
  unguarded by construction. On a real run `L1/<pass>/background_mask.parquet` was sitting there,
  carrying a per-region `uncertainty` folded across every presence signal and thresholded by the
  detection-margin profile. It is now `L2/background_mask.parquet`, and the guard keys on
  **shape** — an `axis` column, or a column whose value is an aggregate across signals — so
  `L1/<pass>/asr/<model>.json`, which merely shares a name with an axis, stays legal for the
  right reason.
- The item-26 guard was a regex requiring the read to hang off the `final_dir(...)` call within
  one expression. Every real caller binds the directory to a name first, so it matched none of
  them: `adaptive/plot.py` read `final/transcript.json` and `adaptive/ls_final.py` read three
  files under `final/`, all live, while the guard was green. It now resolves aliases per scope
  through the AST.
- `build_final_ls_bundle` was still a no-op for a second reason the register did not reach:
  analyze_audio did not write the run bundle until *after* the adaptive loop had finished, so
  re-pointing the reader could not have helped. The bundle is now completed (scene tracks
  included) and written to `L2/` before the loop. Measured: 0 `final__consensus_transcript`
  regions before, 72 after, on the same clip.
- Two more reads that had never resolved: `_aggregator_from_run` read the pre-L1/L2 flat
  `<run>/disagreements.json`, so every standalone adaptive run silently fell back to `min`; and
  both background-mask plot readers were globs that had drifted, so each row reported "no
  background mask" on runs whose mask had found regions.

### Item 27 — artifact half closed

`L2/rounds/<N>/belief/<axis>.parquet` carried a `stream` column and twice the rows of the
cross-pass fold. It now holds one row per bucket, folded across passes by **most doubtful wins** —
the policy `_final_belief_index` was already applying at read time — with `stream_fold`,
`elected_stream` and `folded_from` recorded on the row. Three readers had each invented their own
collapse (`_final_belief_index` kept the more doubtful row, `adaptive.plot` filtered to the fusion
stream, `evaluate` filtered to the transcript's): three answers from one file, none written down.
The column is `uncertainty`, not `within_pass_uncertainty` — the row is no longer within a pass.

**What still remains** is unchanged from the section above: the store's in-memory
per-`(stream, axis, bucket)` fold, blocked on `fuse_axis` accepting per-`(bucket, signal)`
weights. That is correct for *votes* and only wrong when it reaches an artifact, which it no
longer does. A new guard — no parquet keyed by both a pass and an axis, paired with one requiring
a fold to have one row per bucket — now fails if it comes back, including via the paper-compliant
escape of renaming the column while still emitting two rows per bucket.

All three run-reading guards resolve their run from `artifacts/analyze_audio/` and skip, naming
the command that produces one, when there is none.

## Item 27 — closed (2026-08-02)

The remaining half was **not** blocked on `fuse_axis` accepting per-`(bucket, signal)` weights.
That blocker was stated from the wrong side of the call: `fuse_axis` is invoked *per bucket* by the
store, so the per-signal weights it takes are already this bucket's. The store composes them across
passes (mean over the passes a signal voted in, an unmeasured pass contributing `1.0`) and hands
them over; no signature change was needed. The claim had gone unchecked for two rounds and was
recorded as a blocker in both.

**Nothing is keyed by both a pass and an axis, in memory or on disk.**

| was | is |
|---|---|
| `VoteStore.reaggregate_bucket(stream, axis, bucket)` | `reaggregate_bucket(axis, bucket)` |
| `VoteStore.buckets(stream, axis)` | `buckets(axis)`; `vote_buckets(stream, axis)` for vote-level readers |
| `BeliefState.rows[(stream, axis)]` | `rows[axis]`; `axis_rows(axis)`, `uncertainty_mass(axis, θ)` |
| `propose_regions(rows, axis, stream, …)` | `propose_regions(rows, axis, …)`; `Region.stream` → `action_stream` |
| `touch_counts[(stream, axis, bucket)]` | `[(axis, bucket)]` |
| the store's own fold | `fuse.fuse_axis` — one definition, checkable |
| `elected_stream` on the belief row | `contributing_passes` |

**The fold is `fuse_axis`.** The store had a second implementation of the axis, which is what made
"the two L2s disagree" a real disagreement rather than a formatting difference. With one
implementation the comparison becomes possible, and `VoteStore.fused_parity` performs it against
`L2/round0/uncertainty/<axis>.parquet`. Round **0**, not the last round: later rounds condition
each axis on the others, evidence the store does not have, so comparing against them skipped every
bucket as coupled and reported a vacuous zero. On clip18s (18 s, two passes):
`speech_presence` 896 compared / 0 mismatches, `speaker` 72 / 0, `asr` 24 / 0, `max_abs_diff` 0.0.
`replay_check` keeps proving the other property — re-derivability from what is persisted.

**Stream election survives only where a pass must genuinely be chosen**, and only on per-signal
evidence. Two sites did it over a per-pass *axis*, which is the category error itself:

- the fusion stream was `min(passes, key=uncertainty_mass(pass, "asr"))` → now the mean per-signal
  ASR doubt on each pass, read from that pass's votes;
- `S1_stream_election` read three per-pass axis quantities → now `p_voice`, scene degradation and
  per-signal ASR doubt, each per pass from votes and measurements. Its result is
  `Region.action_stream`: which audio to hand a model, not an index on the belief.

### What this exposed

- **`aleatoric_floor` was `0.0` on every bucket of every run.** It read `quality_snr` and siblings —
  anchored *scores*, which neither ingest path has ever carried (the harvest holds dB; the fused
  presence parquet holds neither). Every lookup missed and the floor defaulted to `0.0`, which is
  the confident claim "this audio imposes no floor", so the `snr_floor` irreducibility verdict
  could not fire and a run could only ever report
  `no_reduction_under_available_interventions`. The floor is now derived from the dB under named
  anchors, carries `aleatoric_floor_policy` on the row, and is `None` — not `0.0` — where nothing
  was measured. Scene measurements ride the presence grid, so the other two axes fill in from the
  presence buckets they overlap; without that the verdict stayed unreachable on two axes out of
  three. Measured: 178/896 presence, 15/72 speaker, 8/35 asr buckets now carry a floor, and
  `snr_floor` fires on the same clip that previously reported only the other reason.
- **The artifact ingest dropped `__cross_diar_label_disagreement__`.** It was diverted into
  `row_meta` by a payload-shape test ("dunder name with a `value` key"), while the in-process path
  kept it as a vote — the two ingests differed by one speaker signal and nothing said so. Reserved
  names are now listed.
- **`_attach_scene_measurements` matched nothing.** It joined the bucket measurements from
  `L2/round0/uncertainty/speech_presence.parquet`, a file that carries none of them: empty column
  intersection, early return, no measurements attached on any run. They travel in the vote file
  under `__quality__` and are read from there.
- **The scene→asr coupling never reached disk.** `_apply_scene_coupling` ran on
  `compute_uncertainty_axes`'s in-memory rows; `write_final_uncertainty` then re-folded every axis
  across its rounds and analyze_audio copied `triage_score` and `coupled_from` back, so every
  persisted asr row had `coupled_from == []` while `scene_quality_coupling` and
  `triage_score_pre_coupling` sat on the row asserting an adjustment its number did not contain.
  The implementation moved to `votes.apply_scene_coupling`, applied per round, both columns
  written. Measured: 35/35 asr rows coupled, `triage_score != triage_score_pre_coupling` on all 35.
- **The parity check found a difference between two spellings of "nothing".** Parquet reads a
  missing value back as `NaN`, the store holds `None`; 11 asr buckets where *neither* had a value
  were reported as mismatches until both sides went through `_float_or_none`.

### What the guard now asserts

`test_no_belief_api_takes_a_pass_to_produce_an_axis` inspects the signatures of every
axis-producing call. The artifact guards can only see the last step: a writer that collapses two
per-pass rows at the moment it writes satisfies them while the loop still holds one axis value per
pass — which is exactly the state the previous round left, and why its "artifact half closed" was
the whole of what it had closed. Both guards were shown failing on the tree before the fix:

```
L2/rounds/1/belief/speech_presence.parquet: fold ['uncertainty'] keyed by pass ['elected_stream']
VoteStore.reaggregate_bucket(stream) / BeliefState.axis_rows(stream) / propose_regions(stream)
```

`elected_stream` joined `PASS_COLUMNS` for the artifact guard: naming the pass whose reading was
taken *as* the axis's is a per-pass axis with the index moved into the value.
`contributing_passes` is not, and stays — it says which passes fed the fold.

### Consequences worth knowing

The presence axis's `uncertainty` is now `fuse_axis`'s entropy over per-signal doubt, not
`1 − |2·p_voice − 1|`. On a bucket where the only voters report a direction and no confidence, the
fold is `None` where the old estimator returned a number: no signal expressed doubt, which is not
the same as agreement. On real runs this changes nothing measurable — 896/896 presence buckets
still fuse, because the frame-posterior and ASR signals carry confidences — but a synthetic
coverage-only fixture now reports `None`. `p_voice` itself is unchanged and still comes from
`speaks`, so every consumer that needed a probability about the world still has one.

### Direction-only voters: judged, and `aggregate`'s treatment promoted

The paragraph above ("this changes nothing measurable — 896/896 presence buckets still fuse,
because the frame-posterior and ASR signals carry confidences") was measured on a configuration
nobody ships, and it does not hold on the defaults. Only Whisper reports `avg_logprob`, so on the
shipped ASR set the presence axis fused **7 signals of 14**: all three ASR models and both
diarizers were dropped, because each asserts a direction and scores nothing. `fuse_axis` was the
only reader of a presence vote that did not understand that shape — `aggregate.per_source_voice`
and `support.presence_probability` both already map such a vote to `p = 1.0`/`0.0` — and
`reliability._bucket_beliefs` had *already* had to reintroduce these voters by hand to measure
their stability, so a weight was being computed for signals the fold could never use.

Judged in favour of `aggregate`'s treatment, and promoted: `fuse.is_direction_only_claim` names the
shape and `per_signal_uncertainty` reads it at full strength (doubt `0.0`), by `setdefault` so a
measured pairwise doubt still wins. Re-folding a completed run's own stored votes: 6.97 → 13.97
signals per bucket, `uncertainty` 0.5438 → 0.3415, `confidence` 0.8675 → 0.9340, and
`triage_score` **unchanged to four decimals** with 0 of 1070 buckets dropping — the default fold is
max-doubt, so a voter carrying no doubt cannot make a region look calmer than it is.
`reliability._bucket_beliefs` now keys its substitution on the shape rather than on the fold's
silence, so cross-pass flip detection is byte-identical to before. Guarded by
`asr_presence_signal_test.py`, parameterised over the ASR result shapes.

Found while doing it, and **still open**: Whisper's word coverage never reached the axis either.
`avg_logprob` sits on the line, and `asr_bucket_chunk_evidence` only falls back to line-level
scalars when no chunk of that line overlapped the bucket — so Whisper's score reaches exactly the
buckets where it placed *no* words, and a bucket full of its words was as direction-only as any
other backend's. Its presence signal was carried entirely by its silent buckets, which is the whole
of what made it look like the shape that worked. Whether a segment's score should also describe the
buckets its own words landed in is a separate decision about the scalar fallback (item 4's
neighbourhood), not about the fold.

### Still open

`aggregate.aggregate_speech_presence`, `aggregate_speaker` and `aggregate_asr` still have no caller
in `src/`. Leaving them in the tree leaves a second definition of the axis available to the next
reader who goes looking; the direction-only rule they encoded now lives in `fuse`, so they should be
deleted with their tests.

---

## `L1/timeline.png` — the axis figure that lived in the evidence layer (fixed)

Found by asking why L1 had a timeline showing axis uncertainty at all. It did:
`scripts/analyze_audio.py` called `build_aligned_timeline_plot(run_dir=…, fused_axes=…)` with no
`save_path`, and `plot.py` defaulted to `evidence_dir(run_dir) / "timeline.png"` — whose own
docstring reads *"everything measured, nothing concluded."* The figure's top three rows are each
axis's `uncertainty` with `epistemic_uncertainty` shaded beneath. Every run wrote it.

This is not the usual entry in this register. Nothing was thresholded, rescaled or reduced; the
arithmetic was L2's and correct. **The artifact was in the wrong layer**, which is the same defect
one level up: a conclusion presented where a reader expects a measurement.

### Why it passed every guard

Four mechanisms, each missing it for a different reason — worth keeping because three of them are
still the mechanisms protecting everything else:

1. **A default argument chose the layer.** The call site passed `run_dir`; the renderer picked the
   directory. No reviewer of the call site could see which layer was written. Same shape as
   `settled_below=0.35` (D-21 rule 4), in the write path rather than in a policy.
2. **`contracts.py` declared the path**, as `Artifact("L1/timeline*.png", "evidence view: the
   signals against the recording")`. The pattern was legal, so the artifact guard passed, and the
   only characterisation of the content was that prose — which was wrong.
3. **`MODULE_STAGE` declared `plot.py` to be `"L1"`,** on the line directly below `l1_plot.py`. So
   the static write-scope check *also* passed: an L1 module writing an L1 path. Two independent
   declarations agreed with each other, and both were wrong, because neither was derived from what
   the module consumes.
4. **`check_layering.py`'s "L1 has no axis-named artifact" tests file stems.** `timeline.png` is
   not named for an axis; it draws three. The rule is about content, the check is about the name.

And the enforcement designed to make writes checkable could not apply: `stage_io.ReportKey` states
that a rendering *"has no target and no producer"*. The one artifact class outside the capability
system is the one that crossed the layer.

**The collision that caused it.** The comment left in `plot.py` records the history: this figure was
writing `final/timeline.png`, the same path `adaptive/plot.py` writes, and the later write silently
replaced it. The fix chosen then was to move it to `L1/` and relabel it "the evidence timeline" —
resolving a name collision by moving one of the two conclusions into the evidence layer. The
figure's first parameter is `fused_axes`; it was never the evidence timeline.

### The fix

- Writes `final/uncertainty_detail.png` (chunks: `final/uncertainty_detail_NNN.png`). Two
  conclusions answering different questions get **two names**, which is what the collision needed.
- `MODULE_STAGE["…/plot.py"] = "FINAL"`. A module's stage follows from what it consumes: a module
  that reads an axis is downstream of every round that produced one.
- The `L1/timeline*.png` declaration is deleted, and the recorded run-tree fixture in
  `stage_contract_test.py` no longer contains it — the fixture is what a run *leaves*, and while
  the declaration existed every guard in that file reported the tree clean.
- `check_layering.py` gains an allowlist: L1 may contain `signals.png` and no other figure. Weaker
  than a content test, and the honest limit — a PNG's layer cannot be read off the bytes, and the
  stem test provably cannot catch this class.
- `plot_test.py` pins the output path directly (`test_axis_figure_never_lands_in_l1`), asserted on
  the path rather than against a declared contract, because the contract declared it.

**Generalisation, and the reason this entry is here rather than in a changelog:** a declaration is
not evidence. Three of the four guards above were *satisfied by declarations that were themselves
the bug*. The check that caught it was a person looking at the output and asking why an axis was in
the evidence layer — the same way every defect in `l1-signal-contract.md` was found.
