# DDK — the posteriorgram CV instrument

What it is, why it is a measurement instrument and not a detector, what it reads on known material,
and what it is deliberately not allowed to conclude. Landed 2026-09-16.

The operating points are in `src/senselab/audio/workflows/triage/data/config/default.yaml` under
`branch:` and their derivations in [`config-derivations.md`](config-derivations.md). The code is
`src/senselab/audio/workflows/triage/nodes/ddk.py`.

## Why it exists, and why it is not a classifier

The branch had one rate instrument: the modulation spectrum of `energy_envelope` over an amplitude
carrier (D1). Amplitude says a syllable happened; it cannot say which syllable, and it cannot
separate a stop release from any other energy rise. The phonetic posteriorgram PREPROCESS already
writes (`preprocess.py:149 PPG_MEASUREMENT`) carries phoneme identity per frame, which is exactly
the axis the envelope lacks.

An earlier draft of this instrument was a DDK-vs-not detector with a fitted operating point. It is
not one, on the owner's ruling of 2026-09-16:

> no participant would have done a DDK under a different instruction. it's a very specific task. so
> all tasks labeled ddk will go to the ddk branch, and tasks that don't have ddk will not go to that
> branch.

**The declared task label is ground truth for "is this DDK", in both directions.** There is no
classification problem for the instrument to solve. What remains — and what the branch is actually
for — is characterising *how* the declared task was performed, tolerating the three axes of
variation the owner named: phonemic, timing, articulatory.

## The method

1. Read `ppg_posteriorgram`: `(frames × 40)` float, the ARPAbet-40 phoneme axis including
   `<silent>`, and the frame period. Absent derivative, absent phoneme axis or a non-positive
   period all read as an absent instrument.
2. `argmax` per frame; collapse consecutive identical labels into runs. Adjacent runs carry
   different labels by construction.
3. Class per run: **C** if the phoneme is a key of `branch.ddk_stop_places`, **V** if it is in the
   union of the `branch.ddk_nucleus_classes` entries the declared syllable template names, **O**
   otherwise. See [`ddk-syllable-template.md`](ddk-syllable-template.md).
4. A **CV unit** is a C run followed by a V run. Scan forward from each C run: the first V closes a
   unit whose onset is the C run's own start; the first further C ends the scan with nothing
   emitted. Because runs alternate, this resolves without a window — **there is no lookahead
   parameter and none may be added.**
5. A **train** is a maximal contiguous stretch of CV onsets whose consecutive inter-onset intervals
   stay within `branch.ddk_interval_tolerance` of the stretch's running median, in both directions,
   and which holds at least `branch.ddk_min_repetitions` onsets. Stretches do not overlap: the
   interval that ended one is where the next is tried from.
6. The train with the most repetitions is reported, with its `repetitions`, `period_s` (the median
   interval), `rate_hz` (its reciprocal) and jitter (the population deviation of the intervals over
   their median).

Steps 5 and 6 are segmentation and reporting. **Nothing here accepts or rejects a train on its
rate.**

## What it answers, on the three axes

**Timing.** The unit onsets feed the functions the envelope path already uses — `intervals_of`,
`dispersion`, `trend` and `dispersion_by_position` — rather than a parallel timing path, with
`len(expectation.sequence)` as the cycle. The posteriorgram is the better onset source for all four:
an envelope onset lands at an energy rise, a CV onset lands at a stop release.

**Phonemic.** `ddk_expected_place_fraction` is the fraction of units whose place is the one its
cycle position expected, with the same fraction per position beside it. A substituted or missing
consonant lowers that fraction; it is a finding, never a rejection. Vowel variation costs nothing by
construction — `/pa/`, `/paw/` and `/puh/` all satisfy V.

**Articulatory.** `unit_places` reads a place per onset off the posteriorgram. Where
`spectrogram_wideband` is present, `ddk_place_agreement_ppg_vs_burst` reports how often that reading
and `ddk_places`' burst-spectrum reading agree, over the onsets both resolved.

### The PPG place is a reported reading, not the authority

`expected-patterns.md` ruled the posteriorgram out as the place instrument and D6 of
[`branch-ddk.md`](branch-ddk.md) listed the PPG-label-to-syllable mapping as **owed and unmeasured**
— *"whether `/puh/` reliably surfaces as a particular argmax phoneme is unmeasured, and the
posteriorgram is out of domain on rapid nonsense repetition."* Those are the same evidence; one
leaves the question open and the other closes it. The ground for reporting the PPG place beside the
burst spectrum rather than instead of it is simpler than either: **neither spec claims PPG place
works on this material, and nobody has measured it.** The agreement fraction is the measurement that
would settle it, and producing it is worth more than either reading alone.

Until that agreement is measured on labelled material, `ddk_places` stays the place decision and
nothing downstream may treat `unit_places` as one. Both measurements carry that sentence as their
`reading` covariate.

## What the instrument reads on known material

Measured on the b2ai corpus at `/orcd/scratch/bcs/002/satra/clipfix_20260913`, 2026-09-16, over 597
declared-DDK recordings across the five DDK families (`-pa -ta -ka -pataka -buttercup`) against
connected-speech and non-speech controls. **These are descriptive context for reading a measurement,
not thresholds, and nothing in the code compares against them.**

- Rate separates the populations: DDK families 2.94–4.54 Hz, connected speech 0.92–1.37 Hz.
- Jitter does not: DDK 0.16–0.17, connected speech 0.15–0.18. This is why
  `ddk_interval_tolerance` is a segmentation guard and is said to separate nothing.
- Sustained phonation, glides, loudness, breathing and cough produce **structurally zero** CV units,
  because they contain no stops. That is a property of the walk, not a fitted result.
- At a stricter segmentation point over 52,260 non-DDK recordings, 67 stretches were reported as
  trains (0.13%), every contributing family connected speech — story-recall-v2 1.97%,
  caterpillar-passage 1.18%, cinderella-story 1.16%, picture-description, free-speech. Under the
  routing this instrument ships with, none of those recordings reaches DDK.
- A known sensitivity ceiling, diagnosed on 105 declared-DDK recordings on which the walk found no
  train: 74 had CV units present whose timing failed the regularity test, 25 had fewer than four CV
  units detected, 5 ran under a second, 1 was over 90% silent raster. Their ASR transcripts confirm
  the task WAS performed (*"Pap- papapapapapapapa…"* yielding no train). **The ceiling is in the
  argmax raster's stop detection, not in absent content and not in unperformed tasks** — which is
  also the reason the instrument may not be the sole conformance evidence.

## How conformance uses it

`align_ddk` still concludes conformance on the expected pattern being realised. The posteriorgram
folds in through `_with_ppg` and does three things and no more:

- An absent or unreadable instrument changes nothing. The envelope's answer stands.
- A train found where the instruction asked for one is conformance, whichever instrument found it.
  Either suffices, because the expensive error is reporting "task not performed" on a recording
  where it was — see the ceiling above.
- A readable instrument that found no train, on a recording where the other instrument found none
  either, is a **task non-conformance** (`False`) rather than an unanswered question. No train at
  all where the instruction asked for one is an answer.

A collapsed sequence is never a non-conformance. `/pa-pa-pa/` for `/pa-ta-ka/` is the clinically
meaningful finding the sequential families exist to surface, and it travels as a low
`ddk_expected_place_fraction` and a `syllable_sequence_mismatch` deviation. A substituted nucleus is
the same kind of finding, on `ddk_expected_nucleus_fraction`.

**Absent posteriorgram is UNDETERMINED, never a refusal.** The absence is recorded as a
`ddk_syllable_rate_from_ppg_cv_onsets_hz` measurement with no value naming the missing derivative,
and as the `NO_PPG` note on the branch report.

## The expectation table

Every DDK row names a **syllable template**: one `branches.Syllable` per position, carrying both the
onset place and the nucleus class its instruction asks for. `(("labial","low"),)` for `-pa` and
`-v2-puh`, `(("alveolar","low"),)` for `-ta`/`-v2-tuh`, `(("velar","low"),)` for `-ka`/`-v2-kuh`, the
three-place all-low cycle for `-pataka`/`-v2-puhtuhkuh`, and
`(("labial","low"), ("alveolar","rhotic"), ("velar","low"))` for both `buttercup` rows — which are
therefore `SYLLABLE_SEQUENCE` like `pataka`, not a lexical special case.
[`ddk-syllable-template.md`](ddk-syllable-template.md) is the whole of that change.

This removes a limitation `expected-patterns.md` named as a defect in its own design — *"three
identical rows, one per target syllable, because nothing in the matcher reads which syllable it
is."* `/pa/` IS labial: the place is the task definition, with the same epistemic status as
pataka's triple, and no fit. `len(sequence)` is then the cycle for a one-syllable train as it
already was for a sequential one, which is what `dispersion_by_position` wants.

## `detect_ddk`

Under the shipped routing this mode is **unreachable**: a recording reaches DDK only when its
declared family is a DDK family, which is exactly the condition `align_ddk` runs under. It is kept
because the two-mode contract requires both arms and `dispatch` enforces it, it reads the same
instrument the in-family mode does, and nothing more is invested in it. It cannot conclude anything
in any case — `dispatch` raises if an out-of-family mode returns other than `UNDETERMINED`.

## What is deliberately not here

- **No rate acceptance band, in `branch:` or `verdict:`.** A plausibility judgement is a decision and
  a branch does not decide; and no derivation for a clinical cutoff exists, DDK rate norms being
  age-, sex- and disorder-dependent. See the NOT SHIPPED entry in
  [`config-derivations.md`](config-derivations.md).
- **No lookahead parameter** on the CV scan. Runs alternate; a window would be an unmeasured decision
  with a public interface.
- **No sensitivity/specificity target.** The instrument classifies nothing, so there is nothing to
  tune against.
- **No second stop-set key.** `branch.ddk_stop_places` is the one declaration; the stop set is its
  union.
