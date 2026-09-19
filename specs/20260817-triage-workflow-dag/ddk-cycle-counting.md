# DDK: counting repeats instead of scoring positions

`ddk_expected_place_fraction` is gone. What replaces it counts how many complete repeats of the
declared syllable cycle the recording holds, how fast they came, and how much of what was detected
they account for. This document is why, what the replacement computes, and what it measured.

## The defect

`expected_fraction` compared realised unit *i* against `expected[i % len(expected)]`. Two
assumptions are buried in that single expression and neither is established anywhere:

1. **The first detected CV unit is cycle position 0.** Nothing aligns phase. The CV walk's first
   unit is whichever stop the argmax raster resolved first, which on a `/pa-ta-ka/` recording is
   frequently the `/ta/` or the `/ka/` of the first repeat, or a lead-in the subject produced before
   settling.
2. **The detected series is complete.** One syllable the walk missed shifts every subsequent
   comparison by one position, so a single miss near the start drives the score of an otherwise
   perfect production toward chance.

For a one-position template `i % 1 == 0` always, so neither assumption can bite and the metric is
correct by degeneracy — which is why the defect was invisible on `-pa`, `-ta`, `-ka` and the three
`v2` single-syllable rows, six of the ten families. On the four sequence families it dominates.

The failure is not subtle once probed: a clean ten-cycle `/pa-ta-ka/` train missing only its first
unit scores **0.0** under the positional metric, because every one of the remaining 29 comparisons
is off by one position. That is the regression test
`test_a_train_missing_its_first_unit_still_reports_the_cycles_it_contains` pins.

## What replaces it

A greedy scan over the realised place series against the template's place sequence. The scan holds
one cursor into the cycle:

- A unit whose place is the one the cursor awaits advances the cursor.
- A unit whose place is anything else, while the cursor is past position 0, is **skipped as an
  insertion** — it does not reset the cycle.
- A unit before any position of the cycle has matched is lead-in, and is neither.
- A cycle closes when the cursor reaches the end of the template; the cursor then returns to 0.

A trailing partially matched cycle is discarded: only complete repeats are counted.

`cycle_scan` returns which units matched which position, where each complete cycle opened and
closed, and which units were skipped while a cycle was open. Four numbers are reported from it:

| name | what it is |
|---|---|
| `cycles` | complete repeats of the template found |
| `cycle_rate_hz` | the reciprocal of the **median gap between cycle starts**; for a sequence family this is the sequential motion rate |
| `gap_cv` | the population deviation of those gaps over their median |
| `consumed` | `cycles * len(template) / units` — the fraction of detected units the complete cycles account for |

`insertions_n` travels beside them: how many detected units the scan had to step over.

### Why it scans every unit, not the train's

The scan runs over `PpgReading.units` — every CV unit the walk found — and not over
`SyllableTrain.units`. The train finder admits only a contiguous stretch whose inter-onset intervals
stayed inside `ddk_interval_tolerance` of their running median; that is exactly the irregularity the
repeat count is built to tolerate. Restricting the scan to the regular stretch would throw away the
repeats the metric exists to recover. This is also why the cycle measurements are emitted on
recordings where the train finder found no train at all, which the old place fraction was not: the
`branch-ddk-ppg-instrument.md` ceiling diagnosis found 74 of 105 no-train declared-DDK recordings
had CV units present whose timing failed the regularity test, and those recordings now report their
cycles.

### `cycle_rate_hz` against `ppg_rate_hz`

They are different quantities on a sequence family and the same quantity on a single-position one.
`ppg_rate_hz` is the reciprocal of the median **inter-onset** interval — syllables per second, over
the regular train. `cycle_rate_hz` is the reciprocal of the median gap between **cycle starts** —
cycles per second, over every unit. Where the template holds one position, a cycle *is* a syllable
and the two measure one thing; they may still differ numerically because one is scoped to the train
and the other to the whole unit series. Both are reported. Neither is special-cased.

## Units: cycles against syllables

`expected_event_count` is 30 for `-pataka` and `-buttercup` — syllables, since `-buttercup` moved
from 10 words to 30 syllables. The repeat count counts cycles. The declaration is **not** changed
again; instead the cycle measurement carries both sides:

- `syllables_n` — CV units detected
- `declared_syllables` — the row's `expected_event_count`
- `declared_cycles` — that count divided by the template length

so a reader comparing 6 found cycles against 10 declared, or 20 found syllables against 30 declared,
has both without inferring either.

## The nucleus fraction: kept, phase-aligned, renamed

`ddk_expected_nucleus_fraction` read **1.00 for eight of ten families by construction**. The cause is
`admitted_nuclei`: CV extraction admits the union of the nucleus classes the declared template
names, so on a template whose every position says `low`, a unit can only *have* a `low` nucleus and
the fraction cannot be anything but 1. Only the two `buttercup` rows — whose union is
`low ∪ rhotic` — could carry a value, and there the positional phase error corrupted it the same
way it corrupted the place fraction.

It is **kept, not deleted**, as `ddk_cycle_nucleus_fraction`, scored against **the position's own
class** over the units the cycle scan matched. That is informative wherever the template's union
holds more than one class, which is exactly the `buttercup` case, and it is now phase-correct
because the scan established which position each unit took rather than assuming it from its index.

What it does **not** do is remove the by-construction ceiling on the eight single-class families.
Removing that would mean widening extraction to admit every nucleus class regardless of template —
which is a different decision with its own evidence (`ddk-syllable-template.md`: the declared
template is deliberately what admits the rhotic nucleus, and widening it to a global vowel set is
the defect that document fixed). It is not overturned here. For a template whose positions all name
one class the fraction reads 1.00 and says only that extraction did its job.

## Measured

Over all 7,994 declared DDK recordings, shipped unit extraction
(`phoneme_runs` → `cv_units` → `unit_places`), scoring only changed. Medians per family:

| family | cycles | declared | cycle rate | gap CV | consumed | old positional |
|---|---:|---:|---:|---:|---:|---:|
| `-pa` | 10 | 10 | 3.12 | 0.45 | 1.00 | 0.95 |
| `-ta` | 10 | 10 | 3.22 | 0.40 | 1.00 | 1.00 |
| `-ka` | 8 | 10 | 2.93 | 0.48 | 0.80 | 0.80 |
| `-v2-puh` | 10 | — | 3.77 | 0.59 | 0.89 | 0.88 |
| `-v2-tuh` | 10 | — | 3.70 | 0.53 | 1.00 | 1.00 |
| `-v2-kuh` | 8 | — | 3.44 | 0.65 | 0.82 | 0.82 |
| `-pataka` | 6 | 30 (=10 cyc) | 1.39 | 0.37 | 0.69 | 0.42 |
| `-buttercup` | 6 | 30 (=10 cyc) | 1.51 | 0.36 | 0.72 | 0.40 |
| `-v2-puhtuhkuh` | 3 | — | 1.20 | 0.24 | 0.62 | 0.38 |
| `-v2-buttercup` | 4 | — | 1.66 | 0.16 | 0.75 | 0.43 |

`/pa/` and `/ta/` recover a median of exactly 10 cycles against a declared 10 with every detected
unit consumed. The six single-position families agree with the old metric to within rounding — which
is the control, since phase cannot be wrong there. The divergence is confined to the four sequence
families, where the old metric read 0.38–0.43 on productions that hold 3–6 clean repeats.

**These are descriptive. Nothing in the code compares against them and no threshold is derived from
them.**

## The envelope and burst path

`align_ddk`'s own sequence evidence — `realised_cycles`, `sequence_collapse_fraction` and the
`syllable_sequence_mismatch` deviations, all read off `ddk_places`' burst spectrum over the envelope
onsets — carried the same positional assumption in two places: the deviation's `expected` covariate
was `places_expected[index % len(places_expected)]`, and `realised_cycles` counted exact contiguous
windows, which no missed or inserted syllable survives. Both now go through the same `cycle_scan`:

- `realised_cycles` is `scan.cycles`.
- A `syllable_sequence_mismatch` is raised for each unit the scan skipped while a cycle was open,
  with the position the scan was awaiting as its `expected`. A unit the burst spectrum left
  `UNRESOLVED` raises none, as before. A lead-in unit before any match raises none, which the
  positional version did raise and could not justify.

The deviation's meaning in `branch-conventions.md` — *a produced syllable that is not the one the
sequence expected* — is unchanged; what changed is that "the one the sequence expected" is now
established by the scan rather than assumed from the index.

## What is not changed

- `ppg_rate_hz`, `ppg_repetitions`, `ppg_period_s`, `ppg_jitter_over_median`, `ppg_cv_units_n`,
  `ppg_interval_trend_s_per_step` and `ddk_ppg_interval_dispersion` — the syllable-level
  measurements — all stay. The cycle measurements are additional.
- `ddk_place_agreement_ppg_vs_burst` stays scoped to the train's units, against `ddk_places` over
  the same extents. It compares two instruments on one set of onsets and has nothing to do with the
  cycle.
- The `PPG_PLACE_NOT_AUTHORITY` reading travels on the cycle measurement, as it did on the place
  fraction: the posteriorgram's place is reported, never the place decision.
- `admitted_nuclei` and the declared-template-admits-the-nucleus rule, per above.
- Conformance. `_with_ppg` and `done` are untouched by the metric change, except that
  `realised_cycles` feeding `done` is now the tolerant count rather than the contiguous one, so a
  production with one insertion is no longer read as zero cycles.
