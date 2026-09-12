# Praat scalars and the posteriorgram as routing evidence, and the detectors they make possible

## What this is

Two fields on `RecordingFeatures` — `praat` and `ppg` — and twenty candidate detectors reading
them. Nothing here changes `branch_gates`. Every threshold below is a point in a detector's own
sweep grid; which one, if any, the ruleset adopts is the sweep's answer, not this document's.

The two derivatives already exist in every recording's store, added by
`specs/20260911-ppg-praat-batch/`: `praat_features` carries forty scalars as attributes,
`ppg_posteriorgram` names a float16 npz by path and digest. 60,202 of the 62,547 readable stores
carry both. Until now `RecordingFeatures` surfaced neither, so no detector could read either.

## The three routing failures this is aimed at

Measured on the 62,547-recording run of 2026-09-10 under the current ruleset.

### 1. Glides reach no branch, and the duration floor cannot be lowered to reach them

117 recordings in the glide families reach no branch at all. Both voice gates miss them
independently:

| gate | reads | on the lost glides |
| --- | --- | --- |
| `voice.sustained` | `span_longest.amplitude >= 3.0 s` | p5 0.81 s, median 2.16 s, **p95 2.92 s** — the whole distribution is under the floor, so it catches **0 of 117** |
| `voice.glide` | YAMNet singing union `>= 0.05` | **0.009** on the lost glides against **0.540** on the routed ones |

Lowering the duration floor is not the repair: 2.0 s recovers 64 of the 117 and takes non-voice
firing from 21.0% to 45.3%. The floor is not mis-set — duration is the wrong quantity. A glide is a
monotonic F0 sweep, and the store now carries `std_f0_hertz`, which is what separates a sweep from
a held vowel without asking either to last three seconds. There is no F0 minimum or maximum among
the forty scalars, so no direct pitch-range ratio is available; `std_f0_hertz` is the whole of the
spread evidence.

### 2. DDK never routes to DDK, because the only gate asks whether the syllable is a word

859 declared-DDK recordings never reach the DDK branch. `ddk.lexical_repetition` reads the
transcript for a repeated normalised token, so what it actually detects is whether the elicited
unit is in the recognizer's dictionary:

| family | instruction | falls through |
| --- | --- | --- |
| `diadochokinesis-buttercup` | repeat the **word** /buttercup/ | 1.6% |
| `diadochokinesis-v2-puh` | repeat the **syllable** /PA/ | 24.6% |

The two families differ in exactly that one word of their instructions. Diadochokinesis is fast
syllable repetition; `articulation_rate` and `speaking_rate` measure the rate directly, and the
posteriorgram's argmax phoneme sequence carries the repetition itself, in a representation that
never had to be a word.

### 3. Short-breath tasks are unexplained, because the gate needs energy to accumulate

~210 recordings are unexplained in the breath families, worst in the shortest ones:

| family | falls through |
| --- | --- |
| `respiration-and-cough-threequickbreaths` | 6.0% |
| `respiration-and-cough-fivebreaths` | 2.0% |

`airway.breath` reads `residual.energy_fraction >= 0.10`. That is a fraction of the whole
recording's energy, so it rises with how long the manoeuvre lasts; three quick breaths never
accumulate it. Breaths are unvoiced, and `phonation_ratio`, `pause_rate` and `mean_pause_duration`
measure that property without integrating over duration.

## What `RecordingFeatures` now carries

### `praat`: the scalars, keyed as the store keys them

The whole `features` attribute mapping, filtered to finite real numbers and otherwise unchanged —
no prefix, no rename, no whitelist, so a scalar added to the Praat set later arrives without a
code change.

**Absent stays distinct from zero.** PREPROCESS writes a scalar Praat could not place as `null`
(JSON has no NaN, and `specs/20260911-ppg-praat-batch/` fixed the type there). `_finite_scalars`
drops a null, a bool and a non-finite number rather than coercing any of them, so the key is simply
not present; `_optional` then returns `None` and `score_detector` excludes that recording from that
detector rather than scoring it as a zero. A jitter Praat measured as 0.0 is keyed as 0.0 and is
scored. This is the same rule the existing `residual`, `level` and `disruptions` mappings follow,
and the distinction matters most exactly where it is easiest to lose: `speaking_rate` and
`articulation_rate` are null on every recording whose syllable nuclei Praat could not find, which
is a population a DDK detector must not be credited with rejecting.

### `ppg`: the posteriorgram reduced, never the posteriorgram

The array is not carried. It is 8.31 GB corpus-wide — the 3.3 GB in
`specs/20260911-ppg-praat-batch/design.md` was estimated at a 5.4 s mean duration, and the corpus's
tail is longer than that — against a features shard that is already 1.3 GB and that `load_features`
reads whole. The sidecar is opened during extraction, reduced, and closed.

Nineteen keys, `PPG_SUMMARY_KEYS`. Three come off the measurement entity and survive an unreadable
sidecar — `frames`, `n_phonemes`, `seconds_per_frame` — because a posteriorgram having been taken
is itself evidence, separate from what it says. The rest need the array:

| key | what it is |
| --- | --- |
| `segment_count`, `segment_rate_per_s` | contiguous argmax-phoneme segments, and the same per second of analysed audio |
| `segment_duration_{n,min,median,mean,max,iqr,p75,p90}` | the segment duration distribution, through the module's own `_stats` |
| `distinct_phonemes` | how many of the forty appear as a segment's argmax |
| `silent_fraction` | frames whose argmax is the inventory's `<silent>` label |
| `repetition_{peak,lag_segments,mean,prominence}` | see below |

**The reduction reuses `ppg.py`.** `to_frame_major_posteriorgram` normalises the layout (a no-op on
the stored frame-major array, and idempotent) and `extract_ppg_segments` finds the segments. That
function takes an `Audio` because it derives real time from `waveform.shape[1] / sampling_rate`;
the npz carries `duration_s` and `sampling_rate`, and `round(duration_s * sampling_rate)` recovers
exactly the sample count the block divided by, so a zero-filled waveform of that length is an exact
clock and not an approximation. Reimplementing the segmentation instead would have put a second
definition of "segment" in the tree, and the two would have drifted the first time either changed.

**Cost of the reuse, stated because the corpus extraction runs in a process pool.** Importing
`senselab.audio.tasks.features_extraction.ppg` executes the package `__init__`, which pulls the
whole feature-extraction stack: the module's resident set goes from **314 MB to 635 MB** and its
import from **1.5 s to 10.2 s**, measured cold on an otherwise idle laptop. Torch alone accounts for
202 MB of that; the package `__init__` is the other 380 MB. The import is therefore made inside
`_ppg_segments` rather than at module scope, so a run that reads no posteriorgram never pays it —
but the corpus run does pay it, in every worker, and `--workers 32` on the re-extract should be
sized against 635 MB per worker rather than against the 314 MB the last extraction used.

Reduction time, measured on synthetic posteriorgrams at three points of the corpus duration
distribution. These are the **worst case** for the repetition measure, which is quadratic in the
segment count: a random posteriorgram changes argmax almost every frame, so segments ≈ frames,
where a real one holds a phoneme for several.

| corpus point | frames | segments | reduction |
| --- | --- | --- | --- |
| median 7.2 s | 835 | 805 | 3 ms |
| p95 60 s | 6,960 | 6,800 | 15 ms |
| longest 307 s | 35,670 | 34,773 | 160 ms |

Timings taken on a laptop that was not idle; treat them as upper bounds.

### The repetition measure

Nothing in the store carried repetition, and a DDK detector is nothing without one.

Take the segment sequence — the argmax phoneme of each contiguous run, which is the argmax frame
sequence with its runs collapsed — and for every lag from 1 to half its length, take the fraction
of positions that agree with the sequence shifted by that lag. `repetition_peak` is the largest
such fraction, `repetition_lag_segments` the lag it occurs at, `repetition_mean` the mean over
every lag, and `repetition_prominence` the peak over that mean.

Four things about the definition:

- **Half the length is the lag bound**, so every lag is scored over at least half the sequence. An
  unbounded lag would let a three-position overlap at the tail read 1.000 and rank above a real
  period. This is the ordinary autocorrelation convention and it is not a fitted threshold.
- **`repetition_mean` is the chance level of this sequence**, not of the inventory. A recording
  whose posteriorgram holds two phonemes agrees with itself half the time at every lag; the peak
  alone would call it periodic and the prominence would not. Both are offered to the sweep.
- **Silence segments are kept.** `/pa pa pa/` reads as `p aa <silent> p aa <silent>` and the
  `<silent>` is part of the period, not noise in it. Dropping it would shorten the period and, in a
  recording with irregular pausing, break the periodicity that is there.
- **A sequence admitting fewer than two lags gets no repetition keys at all** — one lag is a period
  with nothing to compare it against. Absent, not zero.

### What the two fields cost the shard

Upper bound, taking every scalar to print at full double precision, which is what a derived Praat
value does:

| field | keys | bytes per recording |
| --- | --- | --- |
| `praat` | 40 | 1,552 |
| `ppg` | 19 | 812 |
| both | 59 | **2,364** |

At 60,202 recordings carrying both, **~142 MB** added to a 1.3 GB shard: about 11%. Values that
print short — an integer count, a rounded rate — come in under the bound, so 142 MB is a ceiling
rather than an estimate.

## The detectors

Twenty, in three groups, none of them wired into `branch_gates`.

| detector | reads | unit | grid | polarity |
| --- | --- | --- | --- | --- |
| `glide.praat_std_f0_hertz` | `std_f0_hertz` | Hz | `F0_SPREAD_HZ_GRID` | above |
| `glide.praat_f0_relative_spread` | `std_f0_hertz / mean_f0_hertz` | ratio | `RELATIVE_SPREAD_GRID` | above |
| `glide.praat_std_f0_hertz+no_agreed_word` | the same, on recordings with no agreed word | Hz | `F0_SPREAD_HZ_GRID` | above |
| `glide.praat_phonation_ratio` | `phonation_ratio` | fraction | `PROPORTION_GRID` | above |
| `voice.praat_phonation_ratio` | `phonation_ratio` | fraction | `PROPORTION_GRID` | above |
| `voice.praat_mean_hnr_db` | `mean_hnr_db` | dB | `VOICE_QUALITY_DB_GRID` | above |
| `voice.praat_cepstral_peak_prominence_mean` | `cepstral_peak_prominence_mean` | dB | `VOICE_QUALITY_DB_GRID` | above |
| `ddk.praat_articulation_rate` | `articulation_rate` | syllables/s | `SYLLABLE_RATE_GRID` | above |
| `ddk.praat_speaking_rate` | `speaking_rate` | syllables/s | `SYLLABLE_RATE_GRID` | above |
| `ddk.ppg_segment_rate_per_s` | `segment_rate_per_s` | segments/s | `SEGMENT_RATE_GRID` | above |
| `ddk.ppg_repetition_peak` | `repetition_peak` | fraction | `PROPORTION_GRID` | above |
| `ddk.ppg_repetition_prominence` | `repetition_prominence` | fraction | `PROPORTION_GRID` | above |
| `ddk.ppg_repetition_lag_segments` | `repetition_lag_segments` | segments | `COUNT_GRID` | below |
| `ddk.ppg_segment_duration_median` | `segment_duration_median` | seconds | `SEGMENT_DURATION_GRID` | below |
| `ddk.ppg_distinct_phonemes` | `distinct_phonemes` | phonemes | `COUNT_GRID` | below |
| `airway.praat_phonation_ratio` | `phonation_ratio` | fraction | `PROPORTION_GRID` | **below** |
| `airway.praat_pause_rate` | `pause_rate` | pauses/s | `PAUSE_RATE_GRID` | above |
| `airway.praat_mean_pause_duration` | `mean_pause_duration` | seconds | `DURATION_GRID` | above |
| `airway.praat_mean_hnr_db` | `mean_hnr_db` | dB | `VOICE_QUALITY_DB_GRID` | **below** |
| `airway.ppg_silent_fraction` | `silent_fraction` | fraction | `PROPORTION_GRID` | above |

Three groups, one per failure, plus the neighbours that make each an alternative rather than a
proposal:

- **The glide group** offers absolute F0 spread and spread over the mean, because absolute spread
  scales with register and a 40 Hz sweep is a different thing at a 90 Hz fundamental and at a
  220 Hz one. The `+no_agreed_word` variant is there because a glide carries no words; its gate is
  a structural zero (`words.agreement < 1`), not a fitted cut.
- **The three `voice.*` detectors** are the same failure from the other side. `voice.sustained`
  misses a 2.16 s glide because it asks for duration; phonation ratio, harmonics-to-noise and
  cepstral peak prominence all say "this is voiced" without asking how long it lasted.
- **The DDK group** offers the rate two ways (Praat's syllable rate, the posteriorgram's segment
  rate) and the repetition two ways (peak, and peak over its own chance level), plus the three
  things a syllable train also is: a short period, short segments, few distinct phonemes.
- **The airway group** reads unvoicedness three ways and pausing two ways. Two of the five fire
  **below** their threshold; `polarity` carries that and `_rule_fires` and `score_detector` both
  already read it.

### Grids

Six new grids, each spanning the real range of one quantity: `PROPORTION_GRID` (the unit interval,
for anything that is a proportion of a whole), `F0_SPREAD_HZ_GRID` (1–120 Hz), `RELATIVE_SPREAD_GRID`
(0.01–1.0), `VOICE_QUALITY_DB_GRID` (0–30 dB, which brackets both the 8.12 dB median HNR of
ordinary voiced speech and the CPP range), `SYLLABLE_RATE_GRID` (1–10 syllables/s, so a normal
speaking rate of 3–5 and a DDK rate of 5–8 both fall inside it rather than at an edge),
`PAUSE_RATE_GRID` (0.1–4 pauses/s), `SEGMENT_RATE_GRID` (2–40 segments/s) and
`SEGMENT_DURATION_GRID` (10–500 ms, since `DURATION_GRID` starts at 250 ms and one argmax segment
is an order of magnitude shorter than that).

### One reader is new: `ratio`

`("ratio", left, right)`, alongside the existing `("difference", left, right)`. It returns `None`
when either side is absent **or when the denominator is zero**, so a recording Praat put no mean F0
on is excluded from the normalised-spread detector rather than dividing by zero.

## What this does not do

- **No `branch_gates` entry, and no threshold anywhere outside a sweep grid.** The recall-first
  criterion that will choose among these is being built separately.
- **The `ddk` kind has no reference standard yet.** `write_report` scores a detector against every
  `ReferenceStandard` whose `kind` matches, and `REFERENCE_STANDARDS` has no `ddk` entry, so the
  eight DDK detectors currently enter `bucket_augmentation` — which reads every detector regardless
  of kind, and is the table that answers "what would adding this rescue" — but not `sweeps.json`.
  Making them scorable is one entry in `report.py`, with `predicate=lambda features: features.family
  in SYLLABLE_REPETITION`; the family set already exists in `families.py` and needs no new name.
  `DECLARED_KIND` deliberately gains no `ddk` key: it would be a second name for
  `syllable_repetition` and would change what `declared_kinds` returns for ten families.
- **`breath` and `sustained` are likewise standard-less kinds**, which is why the breath detectors
  carry `kind="airway"`: their failure is a recording that reaches no branch, and AIRWAY is the
  branch it should reach, so `declared_airway` is the standard that answers the question asked.

## Tests

`src/tests/audio/workflows/triage/routing_analysis_test.py`, over synthetic stores with a
synthetic one-hot posteriorgram sidecar written into the run's own `derivatives/`:

- the Praat mapping surfaced under the store's own keys; a null scalar absent from the mapping and
  from `detector_value` while a measured zero reads as `0.0`; the mapping empty, and every Praat
  detector `None`, when the measurement is absent;
- the posteriorgram reduced to segment count, rate and duration distribution, with every key inside
  `PPG_SUMMARY_KEYS` and no array in the record;
- `silent_fraction` over a sequence half of whose frames are the `<silent>` label;
- the repetition measure 1.000 at lag 3 on a three-phoneme cycle and 0.000 on twenty distinct
  phonemes; absent on a three-segment sequence, which admits one lag;
- an unreadable sidecar keeping the three entity keys and yielding no summary key and no detector
  value;
- each of the twenty new detectors reading a value on a record carrying both derivatives and `None`
  on one carrying neither, one parametrised case each;
- the normalised spread equal to the quotient of the two scalars;
- the two below-polarity airway detectors carrying `polarity == "below"`.

Nothing here runs ppgs or Praat: the sidecar is written by the test and the scalars are store
attributes.
