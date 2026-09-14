# DDK branch

What the branch answers: **is there a repeated syllable train here, how fast is it, how regular, and
does it hold across the effort?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md);
shared rules are in [`branch-conventions.md`](branch-conventions.md); owed ground truth is in
[`branch-listening-sample.md`](branch-listening-sample.md).

## The state of this branch

**There is no node.** No `nodes/ddk.py` exists, `DDK` is absent from `GRAPH_ORDER`
(`vocabulary.py:14-25`) and from `run.py`'s dispatch table (`run.py:297-301`). `DDK` *is* in
`BRANCHES` (`vocabulary.py:31`), so ROUTING writes it a `branch_decision` and `_drive_branches`
records it `SKIPPED` with `NO_NODE` (`run.py:302-305`).

**Since ruleset stage 2 it is routable for the first time.** No `kind` ever mapped to DDK, so under
the old fold it could not be selected at all. The consequence is live and large: a recording whose
DDK gates fire now flags the file, because VERDICT records "was asked to run and never ran" — and
that is **all 22,363 recordings DDK routes, of which 14,878 declare no DDK family**. An earlier
version of this document implied it fires only on recordings with DDK content, understating it by
roughly threefold against numbers the same document carries below. Since this flag is the standing
argument for building the node, the number matters.

Everything below is **not built** unless it says otherwise.

## The tasks this branch serves

Declared families, not ground truth. `SYLLABLE_REPETITION` (`families.py`) holds exactly these ten,
and they sum to 7,989 — matching the declared DDK count exactly, with no residual.

| family | n | what the task asks for |
| --- | --- | --- |
| `diadochokinesis-buttercup` | 896 | repeat a trisyllabic word |
| `diadochokinesis-ka` | 896 | repeat /ka/ |
| `diadochokinesis-pa` | 896 | repeat /pa/ |
| `diadochokinesis-pataka` | 896 | repeat /pa-ta-ka/ |
| `diadochokinesis-ta` | 896 | repeat /ta/ |
| `diadochokinesis-v2-buttercup` | 702 | as above |
| `diadochokinesis-v2-kuh` | 702 | repeat /kuh/ |
| `diadochokinesis-v2-puh` | 702 | repeat /puh/ |
| `diadochokinesis-v2-tuh` | 702 | repeat /tuh/ |
| `diadochokinesis-v2-puhtuhkuh` | 701 | repeat /puh-tuh-kuh/ |

The protocol's own instruction says what is wanted — a v2 DDK `_acoustictask-metadata.json` reads
*"imitate the speaker by repeating the syllables 'puhtuhkuh' as quickly and consistently as possible
until the timer runs out"*.

**"As quickly and consistently as possible" is the measurement.** Diadochokinetic rate is a standard
clinical measure of articulatory motor control with two components: **rate** and **regularity**. The
families split into *alternating motion rate* — a single repeated syllable, 4,794 declarations — and
*sequential motion rate*, a repeated sequence of different syllables, 3,195. Sequential motion rate
stresses the ability to switch place of articulation and is the more sensitive of the two.

### A recording routed here whose declared task is not DDK

DDK routed 22,363 recordings against 7,989 declaring a DDK family. Repetition occurs in ordinary
speech — a stutter, a false start, a repeated word. The branch measures what it finds and says what
it is; it does not assert that a Harvard sentence failed to be a DDK task.

**DDK's gate evidence is `unavailable` on 2,345 recordings** (`ruleset_score.json`,
`totals.unavailable`). Ascending, the four are SPEECH 0, VOICE 29, **DDK 2,345**, AIRWAY 56,505 — so
DDK is second-*highest*. An earlier version called it the highest, and the first correction called it
the second-lowest; both were wrong. The design conclusion is unchanged either way: an unavailable
posteriorgram is an absence, never a negative.

## How DDK is routed today

Two gates, either of which routes (`default.yaml:211`, `:257-264`):

| gate | feature | threshold |
| --- | --- | --- |
| `ddk.lexical_repetition` | `transcript_repeat` — largest repeat count of any normalised token | `>= 3`, marked **UNMEASURED** in the config |
| `ddk.ppg_segment_rate_per_s` | `ppg.segment_rate_per_s` — contiguous argmax-phoneme segments per second | `>= 10` |

The second reads no transcript, so it routes a syllable train an ASR declines to transcribe — which
is most of them, since `/pa-pa-pa/` is not lexical.

**Neither may be refit.** Both would be fitted against declared DDK families, encoding which
recordings the protocol labelled rather than which carry a syllable train.

## Capabilities

### D1 — Locate the train and measure its modulation (**not built; primary**)

**Question.** Where is the repeated production, how fast is it, and how regular?

**Propose the train from the envelope, then qualify it by repetition structure** — the same
inversion VOICE V1 applies. A train defined *by* detected repetition cannot be found when the
repetition is irregular, which is the presentation the task exists to detect. An envelope-proposed
train with weak or absent modulation structure is a finding about the production; a train that was
never proposed is an absence of data, and would concentrate `DDK: FAIL` on the most impaired
speakers.

**Reads.** The audio, from **`plain`** — not `enhanced`, and not PREPROCESS's stored
`energy_envelope`. Both exclusions matter: the stored envelope is wrong for the reasons below, and
the `enhanced` stream is FRCRN output, which is out of domain on a DDK train and already feeds the
PPG and therefore this branch's own routing gate
([`praat-instrument-audit.md`](praat-instrument-audit.md) finding 0).

**Computes.** The amplitude-envelope modulation spectrum over the proposed train. A syllable train is
an amplitude modulation at the repetition frequency; the spectral peak gives the **rate** directly.

**The stored envelope is the wrong input, for three reasons.** `energy_envelope` is
`hilbert_envelope_dbfs` over the **pre-emphasised** stream, Butterworth-lowpassed, stored in
**dBFS** (`preprocess.py:1269-1273`):

1. **A logarithmic envelope generates harmonics of the modulation fundamental through the
   nonlinearity alone.** The f-versus-3f peak ratio is this capability's entire unit-disambiguation
   mechanism, and on a dB envelope part of that structure is an artefact of the conversion rather
   than of production unevenness. **D1 needs a linear-amplitude or power envelope** and says so.
2. **Pre-emphasis (+6 dB/octave) reweights burst against vowel energy** — which is exactly the
   asymmetry that creates f/3f structure on `/pa-ta-ka/`. So the harmonic ratio would depend on
   `preemphasis.coefficient`, a key derived for cough and mouth-sound contrast rather than for this.
3. The lowpass is `envelope.lowpass_hz: 40.0`, derived at `config-derivations.md:43-47`. D1 listed
   "a lowpass cutoff" among the conventions it must declare; **one exists and is in force**, so D1
   inherits it by name rather than declaring a second.

**The unit is ambiguous by a factor of three on the sequential families, and this must be resolved
in the output.** For `/pa-ta-ka/`, `/puh-tuh-kuh/` and `/buttercup/` — **3,195 declarations** —
syllables within a cycle differ in amplitude, so the envelope carries energy at the **cycle** rate
(syllable rate ÷ 3) as well as at the syllable rate. Which is the global peak depends on how uneven
the production is. Reporting whichever peak is larger as "the rate" would put a normal speaker at 5
syllables/s into the severe-dysarthria range at 1.7, or the reverse.

**So report the harmonic peak structure and state which unit the number is in.** For the alternating
families the ambiguity does not arise, but the output is explicit there too rather than implicitly
different.

**f and 3f do not by themselves resolve the unit**, because an alternating train is also a
quasi-periodic pulse train carrying energy at 3×. The peak structure says *there is harmonic
structure*; it does not say which harmonic is the syllable rate.

**D2's nucleus count over the train extent supplies the unit** — syllables ÷ duration is unambiguous.
That does not reconcile the two rates into one number, which stays forbidden; it uses the
unambiguous one to label the precise one.

**For a sequential train the 2f component is more informative than 3f**, since the strongest
within-cycle asymmetry is usually between one syllable and the other two rather than evenly across
three.

**Not parameter-free.** An earlier version claimed it was. The method needs an envelope extraction
method, a lowpass cutoff, an analysis window, and — decisively — **a search band**, since the peak
must be sought somewhere. All four are declared as conventions per
[`branch-conventions.md`](branch-conventions.md); the search band in particular determines which
harmonic can be found at all.

**Peak sharpness is not a regularity measure.** Spectral resolution scales as 1/T, so a short train
gives a broad peak however regular the production was, and the measure would report brevity as
irregularity. Either normalise sharpness by analysis duration or do not report it as regularity —
**D3's interval coefficient of variation is duration-robust and carries regularity instead**.

**Emits.** A `propose` span, `family: "ddk"`, subject to the `propose`/`refine` rule; a per-span
measurement carrying the harmonic peak structure with its stated unit and the declared conventions.

**Owed.** A gap criterion for where the train breaks. Stating it relative to the train's own
inter-onset interval keeps it scale-free, but the factor is owed.

### D2 — Syllable-nucleus rate (**not built; cross-check**)

**Question.** What rate does nucleus counting give, and does it agree with D1?

`extract_speech_rate` (`praat_parselmouth.py:91`) returns `speaking_rate`, `articulation_rate`,
`phonation_ratio`, `pause_rate` and `mean_pause_dur`.

**Report rate over the train extent as primary, articulation rate as secondary.** Articulation rate
excludes intra-train pauses — and in DDK those pauses are part of the deficit the task probes.

**Four undeclared operating points, one of them data-dependent.** They are literals inside the
helper, in no config:

| what | where | value |
| --- | --- | --- |
| silence threshold | `praat_parselmouth.py:142` | `silence_db = -25` |
| minimum dip between peaks | `:149` | `min_dip = 4` |
| **…dropped to 2 when mean HNR < 60** | `:154-155` | data-dependent |
| minimum pause duration | `:159` | `min_pause = 0.3` |

**The HNR switch is the serious one.** It changes syllable-detection behaviour based on a
voice-quality measurement of the recording itself, on a corpus of dysphonic speakers — so the
detector's sensitivity is conditioned on the thing being measured. In practice mean HNR is far below
60 dB for any real recording, so the `min_dip = 2` branch is effectively always taken.

**And it fails in the wrong direction on the degenerate case.** If Praat returns undefined for the
mean, `NaN < 60` evaluates `False`, so `min_dip` stays at **4** — the stricter setting — on exactly
the recordings where pitch could not be measured at all. The one population where the branch flips is
the one least able to bear it, and it flips silently.

**Two further deviations from Praat** in the same helper: `min_pause` **0.3 s against Praat's 0.1 s**
and minimum sounding interval **0.1 s against 0.05 s** — both of which change what counts as a
syllable in a rapid train, which is this branch's entire measurement. And its `to_pitch_ac` differs
from Praat on six parameters, with the code's own comments recording that no reason was found.

**All of it is owed**, and the measurements are in
[`praat-instrument-audit.md`](praat-instrument-audit.md).

**Demote the PPG segment rate.** `ddk.ppg_segment_rate_per_s` is a fine *gate* — it routes without a
transcript — but as a measurement an argmax-change rate has no interpretable units: one syllable with
an onset consonant and a vowel yields two or more segments, and the count depends on the
posteriorgram's inventory rather than on articulation. Keep it for routing; do not report it as a
rate.

**Do not reconcile D1 and D2 into one number.**

### D3 — Inter-onset interval structure (**not built; carries regularity**)

**Question.** How does the interval between productions behave across the train?

**Onsets come from the time domain, never from D1's modulation phase.** Narrowband phase tracking
around a spectral peak **imposes near-constant intervals by construction**, so an interval CV derived
from it is biased toward zero — and biased *most* where the production is most irregular, which is
the presentation the task exists to detect. Since D3 now carries regularity (D1's peak sharpness
having been demoted), taking intervals from the phase would re-create, inside the replacement, the
exact bias structure the envelope inversion removed.

**Pick onsets in the time domain from the linear envelope.** D2's nuclei are an acceptable source;
D1's phase is not. **The time-domain peak-picking parameters are owed** and were not previously
listed.

**Computes.** From those onsets: the sequence of inter-onset intervals, then

- its **dispersion** — the coefficient of variation, dimensionless and so comparable across speakers
  at different rates, and **duration-robust** in a way D1's peak sharpness is not;
- its **trend across the train** — festination, slowing, or irregular-without-trend. Among the most
  discriminative DDK features and currently absent entirely;
- **amplitude regularity** across productions, the intensity analogue.

**But CV rests on a detector D2 declares compromised, and the bias runs the wrong way.**
`extract_speech_rate`'s **minimum sounding interval is 0.1 s against Praat's 0.05 s** — and at
6–7 syll/s the syllable period is 140–170 ms with only 80–110 ms voiced, so **the minimum sits on the
measurand**. Merged or dropped syllables produce doubled intervals, which **inflates CV at high
rates — the fastest, healthiest speakers.** That is the opposite direction from the phase-tracking
bias just removed, and not smaller.

**And CV over a sequential train is not comparable to CV over an alternating one.** In `/pa-ta-ka/`
the envelope onset lands differently relative to the release for /p/, /t/ and /k/, so the three
within-cycle intervals are **unequal by measurement convention** and a pooled CV has a floor set by
syllable identity rather than by motor control. **Compute CV within syllable position** on sequential
trains, or report both — 4,794 alternating declarations against 3,195 sequential.

**Parameter-free as measurements.** Dispersion and trend of a sequence need no threshold.

**Emits.** A per-span measurement, with its support count per
[`branch-conventions.md`](branch-conventions.md). **No verdict.**

### D4 — Segment inventory over the train (**not built**)

**Question.** What phonetic segments make up the train?

**Reads.** `extract_ppg_segments`
(`tasks/features_extraction/ppg.py:349`), which returns contiguous argmax-phoneme segments carrying
`phoneme_index`, `phoneme`, `start_frame`, `end_frame`, `frame_count`, `start_seconds`,
`end_seconds` and `duration_seconds` — eight keys.

**Emits.** A per-span measurement over the D1 train. The train span itself is D1's; this adds the
segment inventory to it.

**Owed.** Whether the posteriorgram is usable here at all — see D6.

### D5 — Train duration as a fraction of recording duration (**not built**)

The conformance check the task actually admits. The instruction is *"until the timer runs out"*, so
there is no declared syllable count — but a train occupying a small fraction of the recording is a
departure from *"until the timer runs out"* that is measurable without any norm.

**Emits.** A `counts` entry carrying `found` (train duration, and the fraction) with no `declared`
half.

### D6 — Sequence conformance (**not built**)

**Question.** Were the syllables produced in the order the task asked for?

For `/pa-ta-ka/` the expected cycle is three places of articulation in order. A speaker producing
`/pa-ta-pa/` has substituted; one producing `/pa-pa-pa/` has collapsed a sequential task into an
alternating one — a clinically meaningful finding, and the reason the sequential families exist.

**Emits.** One `deviate` assertion per departure with its extent, typed
**`syllable_sequence_mismatch`** — *not* `stimulus_mismatch`. The contract defines
`stimulus_mismatch` as a lexical word differing from the stimulus text; DDK tasks carry no stimulus
text and expect no lexical content, and the provenance here is PPG argmax over out-of-domain rapid
nonsense syllables. Sharing the name would pool two different objects.

**Owed.** The mapping from PPG phoneme labels to expected syllables. Whether `/puh/` reliably
surfaces as a particular argmax phoneme is **unmeasured**, and the posteriorgram is out of domain on
rapid nonsense repetition.

## Deviations

| type | evidence |
| --- | --- |
| `syllable_sequence_mismatch` | a produced syllable that is not the one the sequence expected (D6) |
| `truncation` | the train runs to the recording boundary |

**A deviation is not evidence of a bad recording.** See
[`branch-conventions.md`](branch-conventions.md): `syllable_sequence_mismatch` is produced by apraxia
of speech and by dysarthria — it is the finding, not a fault of the recording.

**`off_task_extent` is withdrawn from this branch.** It previously covered *"lexical speech, or
silence, where the task expected sustained repetition"* — which makes every breath pause in a DDK
train a deviation and needs an undeclared minimum duration to avoid firing constantly. **Pause
structure is a measurement**: D3's interval and trend analysis reports it, with locations, and
reports it better.

`filler` does not apply — DDK expects no lexical content. Train duration and fraction are `counts`
entries (D5).

## Quality covariates

D1's modulation spectrum and D2's nucleus detection are both amplitude-envelope measures and are
therefore directly sensitive to AGC, which is the default in consumer capture. Every measurement
carries the quality covariates of its extent, per
[`branch-conventions.md`](branch-conventions.md).

## A branch `FAIL` is an absence of detected content

`DDK: FAIL` means **this branch's detector found no train**, never that the speaker produced none.
A detector keyed on regular repetition fails most often on irregular trains — the presentation the
task exists to detect — so without care `FAIL` would concentrate on the most impaired speakers.
**D1's envelope-first proposal removes most of that cause**: a train with weak modulation structure
is still a train, reported with weak structure. See [`branch-conventions.md`](branch-conventions.md)
and Unresolved on `Outcome.FAIL`'s wording.

## What exists today

| capability | status |
| --- | --- |
| D1 propose and measure the train | **not built**; no module computes an envelope modulation spectrum; unit ambiguity must be resolved in the output |
| D2 nucleus rate | **not built**; `extract_speech_rate` exists, no branch consumes it; four undeclared operating points |
| D3 interval structure | **not built**; nothing computes an inter-onset sequence, its dispersion or its trend |
| D4 segment inventory | **not built**; `extract_ppg_segments` exists but no *branch* calls it — `features.py:421` does, to compute the `ddk.ppg_segment_rate_per_s` gate feature |
| D5 train fraction | **not built** |
| D6 sequence conformance | **not built**; PPG-to-syllable mapping owed |

**The routing side is built and works.** What is missing is everything downstream of the decision.

**What senselab provides**: `extract_ppg_segments` (`ppg.py:349`),
`extract_mean_phoneme_durations` (`ppg.py:429`), `to_frame_major_posteriorgram` (`ppg.py:287`),
`load_ppg_posteriorgram` (`ppg.py:327`), `extract_speech_rate` (`praat_parselmouth.py:91`).

**What is missing**: an envelope modulation-spectrum computation, an inter-onset interval sequence
with dispersion and trend, and a PPG-phoneme-to-syllable mapping.

## What the branch would emit

```
spans        family: "ddk", one per syllable train (D1)
assertions   refine (corrected_extent) where a PREPROCESS span's extent is wrong;
             deviate (syllable_sequence_mismatch, truncation)
measurements harmonic peak structure f and 3f with its stated unit (D1),
             nucleus rates with their four declared operating points (D2),
             interval dispersion and trend, amplitude regularity (D3),
             segment inventory (D4) — each with its support count
counts       train duration and fraction of recording {found}
verdict      { trains_n, train_s, train_fraction, modulation_peak_hz,
               modulation_unit, articulation_rate, interval_cv,
               interval_trend, flags }
```

**The verdict's basis, proposed**: `FAIL` when no syllable train was found; `FLAG` when one
measurement contradicts another — a modulation peak that routed the branch over a region where
nucleus detection finds nothing; `PASS` otherwise. Rates and regularity are carried as measurements
and **no normative judgement enters the verdict**.

**There is no `contest` capability.** An earlier version of this document listed one; no capability
here produces an object for it, and contesting the branch's own proposal has no object. If DDK is to
contest, the object must be a PREPROCESS span.

## Out of scope

Normative interpretation of rate or regularity. Reconciling D1 and D2 into one number. Any refit of
`ddk.lexical_repetition` or `ddk.ppg_segment_rate_per_s` against declared families.

## Unresolved

- **`Outcome.FAIL`'s wording is a hazard** — `no_content_found` would carry the meaning — but
  `Outcome` is a closed vocabulary with readers, so this is recorded rather than changed.
- **D6 reads an expected syllable sequence the contract's declaration does not define.** The
  contract says `instructions` is prose and must not be parsed, and it froze the entry keys.
  `data/task_expectations/` does not exist yet, so the extension is free — but it must be asked for
  explicitly.
- Whether the PPG posteriorgram is usable at all on rapid nonsense syllables.
