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
the old fold it could not be selected at all. The consequence is live: a recording whose DDK gates
fire now flags the file, because VERDICT records "was asked to run and never ran". That flag fires
only on recordings with DDK content, and it is the standing argument for building the node.

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
`totals.unavailable`). An earlier version of this document called that "the highest of the four
branches"; it is the second-lowest — AIRWAY's is 56,505, twenty-four times larger. The design
conclusion stands unchanged: an unavailable posteriorgram is an absence, never a negative.

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

### D1 — Rate and regularity from the amplitude-envelope modulation spectrum (**not built; primary**)

**Question.** How fast is the repetition and how regular?

**Reads.** The energy envelope PREPROCESS already computes.

**Computes.** The modulation spectrum of the amplitude envelope. A syllable train is an amplitude
modulation at the repetition frequency; the spectral peak gives the **rate** directly in Hz, and the
peak's sharpness gives the **regularity** directly. Both fall out of one transform.

**This is primary because it dissolves two problems at once.** It does not require locating
individual syllable nuclei, so it does not depend on nucleus-detection operating points; and it does
not require first segmenting the train, so it removes the circularity of needing a train extent in
order to measure the rate that defines the train.

**Emits.** A per-span measurement carrying peak modulation frequency and peak sharpness.

**Serves.** All ten families.

### D2 — Syllable-nucleus rate (**not built; cross-check**)

**Question.** What rate does nucleus counting give, and does it agree with D1?

`extract_speech_rate` (`praat_parselmouth.py:91`) returns `speaking_rate` (syllables ÷ duration),
`articulation_rate` (syllables ÷ phonation time), `phonation_ratio`, `pause_rate` and
`mean_pause_dur`.

**Report rate over the train extent as primary, articulation rate as secondary.** Articulation rate
excludes intra-train pauses — and in DDK those pauses are part of the deficit the task probes, so
excluding them removes the signal.

**Its operating points are inherited and must be named.** Praat's syllable-nuclei method carries a
silence threshold, a minimum dip between peaks, and a minimum pause duration. All three move
`articulation_rate` directly and none is declared here. SPEECH discharges the analogous question for
its aligner explicitly; this does the same. **Owed.**

**Demote the PPG segment rate.** `ddk.ppg_segment_rate_per_s` is a fine *gate* — it routes without a
transcript — but as a measurement an argmax-change rate has no interpretable units: a single syllable
with an onset consonant and a vowel yields two or more segments, and the count depends on the
posteriorgram's inventory rather than on articulation. Keep it for routing; do not report it as a
rate.

**Do not reconcile D1 and D2 into one number.** They measure different things and reconciling them
needs a calibration nobody has.

### D3 — Inter-onset interval structure (**not built**)

**Question.** How does the interval between productions behave across the train?

**Computes.** From D1's modulation phase or D2's nuclei: the sequence of inter-onset intervals, and
then

- its **dispersion** — standard deviation and the coefficient of variation, which is dimensionless
  and so comparable across speakers at different rates;
- its **trend across the train** — festination (accelerating), slowing, or irregular-without-trend.
  These are among the most discriminative DDK features and are currently absent entirely;
- **amplitude regularity** across productions, the intensity analogue of interval regularity.

**Parameter-free as measurements.** Dispersion and trend of a sequence need no threshold; only
interpreting them would.

**Emits.** A per-span measurement. **No verdict** — mapping a coefficient of variation to normal or
disordered needs norms this project does not have.

### D4 — Locate the train (**not built**)

**Question.** Where is the repeated production?

**Reads.** The D1 modulation analysis and `extract_ppg_segments`
(`tasks/features_extraction/ppg.py:349`), which returns contiguous argmax-phoneme segments carrying
`phoneme_index`, `phoneme`, `start_frame`, `end_frame`, `frame_count`, `start_seconds`,
`end_seconds` and `duration_seconds` — eight keys.

**Emits.** A `propose` span, `family: "ddk"`, subject to the `propose`/`refine` rule in
[`branch-conventions.md`](branch-conventions.md).

**Owed.** A gap criterion for where the train breaks. Stating it relative to the train's own
inter-onset interval rather than absolutely keeps it scale-free across speakers, but the factor is
still owed.

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

## What exists today

| capability | status |
| --- | --- |
| D1 modulation spectrum | **not built**; no module computes it |
| D2 nucleus rate | **not built**; `extract_speech_rate` exists, no branch consumes it; operating points owed |
| D3 interval structure | **not built**; nothing computes an inter-onset sequence or its trend |
| D4 locate the train | **not built**; `extract_ppg_segments` exists and no branch calls it |
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
spans        family: "ddk", one per syllable train (D4)
assertions   refine (corrected_extent) where a PREPROCESS span's extent is wrong;
             deviate (syllable_sequence_mismatch, truncation)
measurements modulation rate and sharpness (D1), nucleus rates (D2),
             interval dispersion and trend, amplitude regularity (D3)
counts       train duration and fraction of recording {found}
verdict      { trains_n, train_s, train_fraction, modulation_rate_hz,
               modulation_sharpness, articulation_rate, interval_cv,
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

- **D6 reads an expected syllable sequence the contract's declaration does not define.** The
  contract says `instructions` is prose and must not be parsed, and it froze the entry keys.
  `data/task_expectations/` does not exist yet, so the extension is free — but it must be asked for
  explicitly.
- Whether the PPG posteriorgram is usable at all on rapid nonsense syllables.
