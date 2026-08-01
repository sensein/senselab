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
| 12 | `embedding_silhouette` | cluster all windows, silhouette coefficient, `>= 0.5` | embedding vectors per window on `PassHarvest` | does clustering support a coherent speaker here, and which cluster? | **closed** |
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

Items 11 (`ppg_voice_fraction`) and 12 (`embedding_silhouette`) remain fully open: both are
*derived* signals rather than tool outputs, and per D-7 they should be recomputed at L2 from L1
embeddings and posteriors.


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

Items 11-12 (`ppg_voice_fraction`, `embedding_silhouette`) remain open: both are still reduced
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

## Item 25 — L1 emits per-axis estimates (open, largest remaining)

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
