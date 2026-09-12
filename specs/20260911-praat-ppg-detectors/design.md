# Praat scalars and the posteriorgram as routing evidence, and the detectors they make possible

## What this is

Two fields on `RecordingFeatures` — `praat` and `ppg` — and twenty candidate detectors reading
them. The catalogue was written without a `branch_gates` entry; the corpus sweep of 2026-09-12 gave
two of them one, and "Two of them are gates now" below is that result. Every other threshold here is
a point in a detector's own sweep grid; which one, if any, the ruleset adopts is the sweep's answer,
not this document's.

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
| `glide.praat_phonation_ratio` | `phonation_ratio` | fraction | `SATURATING_PROPORTION_GRID` | above |
| `voice.praat_phonation_ratio` | `phonation_ratio` | fraction | `SATURATING_PROPORTION_GRID` | above |
| `voice.praat_mean_hnr_db` | `mean_hnr_db` | dB | `VOICE_QUALITY_DB_GRID` | above |
| `voice.praat_cepstral_peak_prominence_mean` | `cepstral_peak_prominence_mean` | dB | `VOICE_QUALITY_DB_GRID` | above |
| `ddk.praat_articulation_rate` | `articulation_rate` | syllables/s | `SYLLABLE_RATE_GRID` | above |
| `ddk.praat_speaking_rate` | `speaking_rate` | syllables/s | `SYLLABLE_RATE_GRID` | above |
| `ddk.ppg_segment_rate_per_s` | `segment_rate_per_s` | segments/s | `SEGMENT_RATE_GRID` | above |
| `ddk.ppg_repetition_peak` | `repetition_peak` | fraction | `PROPORTION_GRID` | above |
| `ddk.ppg_repetition_prominence` | `repetition_prominence` | fraction | `PROPORTION_GRID` | above |
| `ddk.ppg_repetition_lag_segments` | `repetition_lag_segments` | segments | `SEGMENT_LAG_GRID` | above |
| `ddk.ppg_segment_duration_median` | `segment_duration_median` | seconds | `SEGMENT_DURATION_GRID` | below |
| `ddk.ppg_distinct_phonemes` | `distinct_phonemes` | phonemes | `COUNT_GRID` | below |
| `airway.praat_phonation_ratio` | `phonation_ratio` | fraction | `SATURATING_PROPORTION_GRID` | **below** |
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
  rate) and the repetition two ways (peak, and peak over its own chance level), plus three more
  things a syllable train is: a segment lag, short segments, few distinct phonemes. The lag was
  written as a short period and fires the other way; "Two definitions that could not reach their
  own data" below is the correction and the measurement behind it.
- **The airway group** reads unvoicedness three ways and pausing two ways. Two of the five fire
  **below** their threshold; `polarity` carries that and `_rule_fires` and `score_detector` both
  already read it.

### Grids

Each spans the real range of one quantity: `PROPORTION_GRID` (the unit interval, for anything that
is a proportion of a whole), `F0_SPREAD_HZ_GRID` (1–120 Hz), `RELATIVE_SPREAD_GRID` (0.01–1.0),
`VOICE_QUALITY_DB_GRID` (0–30 dB, which brackets both the 8.12 dB median HNR of ordinary voiced
speech and the CPP range), `SYLLABLE_RATE_GRID` (1–10 syllables/s, so a normal speaking rate of 3–5
and a DDK rate of 5–8 both fall inside it rather than at an edge), `PAUSE_RATE_GRID`
(0.1–4 pauses/s), `SEGMENT_RATE_GRID` (2–40 segments/s) and `SEGMENT_DURATION_GRID` (10–500 ms,
since `DURATION_GRID` starts at 250 ms and one argmax segment is an order of magnitude shorter than
that).

Two more were added on 2026-09-12, each because the grid it replaces stopped short of the measured
distribution: `SEGMENT_LAG_GRID` (1–1,200 segments) and `SATURATING_PROPORTION_GRID` (the unit
interval again, resolved to 0.999 near the top). The distributions that required them are below.

### One reader is new: `ratio`

`("ratio", left, right)`, alongside the existing `("difference", left, right)`. It returns `None`
when either side is absent **or when the denominator is zero**, so a recording Praat put no mean F0
on is excluded from the normalised-spread detector rather than dividing by zero.

## Two of them are gates now

Scored over the same 62,547-recording corpus on 2026-09-12, each branch's declared families as
positives against the whole rest of the corpus, recall at the over-routing budgets
`recall_at_budgets` reports at:

| branch | detector | absent | R@2% | R@5% | R@10% | R@20% |
| --- | --- | --- | --- | --- | --- | --- |
| AIRWAY | `airway.ppg_silent_fraction` | 2,345 | — | 0.570 | 0.814 | 0.965 |
| AIRWAY | `airway.residual_energy_fraction`, as gated | 31 | 0.169 | 0.582 | 0.739 | 0.815 |
| DDK | `ddk.ppg_segment_rate_per_s` | 2,345 | 0.260 | 0.480 | 0.480 | 0.714 |

### `ddk.ppg_segment_rate_per_s` joins `branch_gates.DDK`, at 10 segments per second

Targeted at the failure DDK actually has: of the **855** declared-DDK recordings that route
somewhere and never to DDK, a cut at 10 recovers **351 (41.1%)** while firing on **4.7%** of
non-DDK recordings. The neighbouring cuts price the same recall three ways:

| cut | of the 855 recovered | fires on non-DDK |
| --- | --- | --- |
| 8 | 579 (67.7%) | 18.6% |
| **10** | **351 (41.1%)** | **4.7%** |
| 12 | 168 (19.6%) | 1.1% |

10 is also what the recall-first criterion picks at the 5% budget: 4.7% is inside it and 8's 18.6%
is not, so the loosest cut the budget allows is 10 to the resolution of this table. The two
neighbours are the other two budgets — 20% would buy 8 and 2% would buy 12 — and DDK is gated at the
5% budget for the same reason every other routing gate is.

It does not replace `ddk.lexical_repetition`; it is a second entry alongside it, and the difference
between the two is the point. The lexical gate reads the transcript for a repeated normalised token,
so what it detects is whether the elicited unit is a dictionary word: `diadochokinesis-buttercup`
falls through at 1.6% and `diadochokinesis-v2-puh` at 24.6%, and those two instructions differ in
exactly "repeat the **word** /buttercup/" against "repeat the **syllable** /PA/". A segment rate is
indifferent to that, because the posteriorgram never had to spell anything.

### `airway.ppg_silent_fraction` joins `branch_gates.AIRWAY`, beside `airway.breath`

**OR'd alongside the residual gate, not in place of it.** The silent fraction wins at the 10% and
20% budgets and loses at 5%, so on the budget alone neither dominates. What settles it is
availability: the silent fraction is absent on **2,345** recordings — those with no consensus
transcript, which carry no posteriorgram — and `residual.energy_fraction` is absent on **31**. Each
reads a population the other cannot. `evaluate_routes` already routes a branch on any one of its
gates firing while naming the unread ones in `unavailable`, so the two compose without either
withholding the branch.

**The cut is 0.90 of the recording's frames, and it is provisional.** The recall-first criterion
reads the values the corpus carries rather than a written grid, so the operating point at the 10%
budget comes off the corpus and not off this document: it is
`gate_recalls["airway.ppg_silent_fraction"].curve` at budget 0.10 in the `ruleset_score.json` that
`scripts/score_taxonomy_ruleset.py` writes, which scores this gate now that it is in `branch_gates`.
0.90 is what the configuration carries until that run replaces it, and correcting it is one number
in `taxonomy.ruleset.gates`.

## Two definitions that could not reach their own data

Both scored flat at every budget, which reads as a measured null and was not one. Neither is a
broken measurement: each is a definition that never touches the region its own values occupy.

### `ddk.ppg_repetition_lag_segments` had its polarity inverted

Measured over the corpus, absent on 6,301:

```
min=1  p25=2  median=6  p75=13  max=1199     COUNT_GRID spans [1, 30], 7,703 values above it
median in DDK = 8, median outside = 5
```

The detector was declared `polarity="below"`, so it fired on the short lags — and DDK recordings
carry the **longer** lag of the two, which means the definition selected against the thing it was
for. The polarity is now `above`, which is the default and is no longer written out.

Its grid was `COUNT_GRID`, which stops at 30 with 7,703 values above it, so no threshold past 30
could be scored at all. `SEGMENT_LAG_GRID` replaces it: 1 to 1,200 in twenty-one points, dense
across the quartiles and log-spaced above them.

**This does not earn it a gate, and the fix is not expected to.** A median of 8 against 5 is weak
separation between two overlapping distributions, and repairing a definition manufactures no
separation the quantity does not carry. It stays in the catalogue to be scored by the re-score, with
`declared_ddk` as the standard that makes scoring it possible at all; if it comes back flat it is a
measured null, which is a different statement from the flat line it produced before.

### `airway.praat_phonation_ratio` had a grid that stopped below its own mass

Measured over the corpus, absent on 2,499:

```
min=0.0223  p25=0.7524  median=1.0  p75=1.0  max=1.0     PROPORTION_GRID spans [0.02, 0.95]
31,464 of 60,048 values sit ABOVE the grid's top
median in airway = 0.9693, median outside = 1.0
```

More than half the corpus sits above the grid's last point, so every threshold on the grid saw the
same undivided block at the top and no cut could separate anything inside it. The quantity is a
ratio bounded at 1.0 whose mass piles against that bound; `SATURATING_PROPORTION_GRID` is the same
unit interval resolved where the mass is — 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999 and 1.0 above
the old top, with the low tail kept, because this detector fires below its threshold and the
minimum is 0.0223.

The grid belongs to the feature rather than to the detector, so `glide.praat_phonation_ratio` and
`voice.praat_phonation_ratio` move onto it too: they read the same scalar with the opposite
polarity, and the same 31,464 values sat past the end of their grid as well.

**Expect a null.** The separation this detector has to work with is 0.9693 in the airway families
against 1.0 outside them — a third of a percent of the range, between a median and a saturated
bound. A grid that can express cuts in that interval is the difference between measuring the
separation and not measuring it; it is not evidence that the separation is usable, and nothing here
predicts that it is. Recording that as a null with its numbers is what this fix is for.

## Grids that may not cover their feature's range

A grid that stops short of its feature's range makes a detector look measured-and-weak when it was
never measured at all — the defect the two above turned out to be, twice, which makes it a class
rather than two instances. The rows below are read off the grid constants against each feature's own
bounds and are **not confirmed against the corpus**.

| detectors | grid | what looks wrong |
| --- | --- | --- |
| `airway.ppg_silent_fraction`, `ddk.ppg_repetition_peak`, `ddk.ppg_repetition_prominence` | `PROPORTION_GRID` 0.02–0.95 | all three are bounded at 1.0 and saturate there — a silent fraction is 1.0 on a recording with no phoneme in it, and the repetition peak is exactly 1.000 on a clean syllable train. The same shape as the phonation ratio, left alone only because that one has the measured distribution to move it on |
| `ddk.ppg_distinct_phonemes` (below) | `COUNT_GRID` 1–30 | the inventory holds forty phonemes, so 31–40 exist and no grid point reaches them; the loosest cut cannot fire on the many-phoneme end this detector exists to exclude |
| `speech.words_total`, `speech.words_lexical`, `speech.words_agreement*` | `COUNT_GRID` 1–30 | a Rainbow Passage read is ~100 words, and no cut above 30 is scored |
| `airway.praat_mean_pause_duration` (above) | `DURATION_GRID` 0.25–10 s | a pause is tenths of a second, so the grid's floor is above most of the distribution and its top is twenty times past any pause: a span grid being read as a pause grid |
| `voice.longest_amplitude_span`, `voice.total_amplitude_span` | `DURATION_GRID` 0.25–10 s | maximum-phonation-time tasks run past 10 s, and a total over a 307 s recording is far past it |
| `airway.zero_crossing_rate`, `cough.zero_crossing_rate` | `ZCR_GRID` 100–3000 /s | unvoiced airway noise at 16 kHz routinely exceeds 3,000 crossings per second, which is the end these detectors fire toward |
| `airway.praat_mean_hnr_db` (below) | `VOICE_QUALITY_DB_GRID` 0–30 dB | HNR is negative on unvoiced material and a breath is unvoiced; the grid floors at 0.0 dB, so everything below it is one undivided block — the below-polarity mirror of the phonation-ratio defect |
| `cough.level_peak_dbfs` | `DBFS_GRID` −60…−3 | a clipped recording peaks at exactly 0.0 dBFS and four campaign files do; nothing above −3 is scored |
| the SQUIM detectors | `STOI_GRID` 0.3–0.9, `PESQ_GRID` 1.05–3.0, `SI_SDR_GRID` −20…15 | each head's range is wider than its grid at both ends — STOI is 0–1 and PESQ is 1.0–4.5, which `PESQ_GRID`'s own docstring says |
| `cough.squim_stoi_iqr` | `SPREAD_GRID` 0.01–5 | an IQR of a 0–1 quantity cannot exceed 1.0, so the top two points are unreachable: dead grid rather than a missed range |

## What this does not do

- **No threshold anywhere outside a sweep grid or `taxonomy.ruleset.gates`.** The two gates below
  carry their cuts in the configuration beside every other gate's; nothing in `detectors.py` holds
  an operating point.
- **The `ddk` kind now has a reference standard, `declared_ddk`**, whose predicate is
  `features.family in SYLLABLE_REPETITION` — the one `report.py` entry this document anticipated.
  Without it `write_report` matched no standard to a `kind="ddk"` detector and the eight DDK
  detectors reached `bucket_augmentation` but never `sweeps.json`, so a DDK detector could not be
  scored and could not be called a null either. `DECLARED_KIND` still gains no `ddk` key: it would
  be a second name for `syllable_repetition` and would change what `declared_kinds` returns for ten
  families.
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
- the two below-polarity airway detectors carrying `polarity == "below"`;
- the repetition lag reading 3.0 on a three-phoneme cycle, firing at a cut of 3 and not at 4, over a
  grid whose ends bracket that value — the polarity assertion is the fix, the old definition having
  fired the other way;
- the phonation ratio at 0.97 firing at 0.98 and silent at 0.95, with all three detectors reading
  that scalar on `SATURATING_PROPORTION_GRID`;
- `declared_ddk` the one `ddk`-kind standard, positive on a diadochokinesis family and negative on
  free speech.

`src/tests/audio/workflows/triage/routing_ruleset_test.py`, over synthetic feature records:

- both new gates in `branch_gates`, their thresholds read out of `taxonomy.ruleset.gates` rather
  than asserted as literals, each firing at its configured cut and silent just under it;
- a recording routing to DDK on the segment rate alone, its transcript carrying no repeated token;
- a recording routing to AIRWAY on the silent fraction with `residual.energy_fraction` absent, and
  the converse with the posteriorgram absent, each naming the unread gate in `unavailable`.

Nothing here runs ppgs or Praat: the sidecar is written by the test and the scalars are store
attributes.
