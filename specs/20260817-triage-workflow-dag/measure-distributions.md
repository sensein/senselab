# What the measurements actually look like, over 62,273 recordings

Read from the finished stores of the 2026-09-21 corpus run by `scripts/triage_measure_stats.py`.
Raw quantiles in `measure-stats-20260921.json`. 29 distinct measurements; numeric readings carry
quantiles, flags and vocabulary terms carry counts.

This exists because every gate that moved into `verdict.gates` arrived at its current value by
design rather than derivation, and a bound is only arguable against the distribution of the reading
it cuts.

## `story-recall`: the gate sits above the 95th percentile

`source_content_coverage` is the fraction of the source's distinct tokens a recall reproduces. Its
gate is `coverage_min: 0.5`.

| family | n | p5 | p25 | **p50** | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `story-recall` | 886 | 0.01 | 0.19 | **0.28** | 0.37 | 0.52 |
| `story-recall-v2` | 658 | 0.14 | 0.37 | **0.47** | 0.55 | 0.66 |

**The median recall reaches 0.28 and the 95th percentile reaches 0.52.** A cut at 0.5 therefore
fails almost the entire population by construction — which is exactly the 93.8% `False` the corpus
reports for `story-recall`, and the 60.2% for its v2, whose distribution sits higher and so fails
somewhat less.

Nothing about those participants is unusual. Recalling a story in your own words does not reproduce
half of the source's distinct vocabulary; that is what recalling *is*. The threshold was placed
without the distribution in front of it, and this is what it looks like when that happens.

## The DDK counts: the number was close, the spread is the point

`ddk_repetition_count_from_ppg_decode`, against a configured `expected_event_count` of 10 for the
single-syllable families and 30 for the multi-syllable ones.

| family | n | p5 | **p50** | p95 | configured |
| --- | ---: | ---: | ---: | ---: | ---: |
| `diadochokinesis-pa` | 895 | 4 | **11** | 24 | 10 |
| `diadochokinesis-ta` | 893 | 6 | **11** | 24 | 10 |
| `diadochokinesis-ka` | 891 | 3 | **10** | 20 | 10 |
| `diadochokinesis-pataka` | 892 | 6 | **10** | 15 | 30 |
| `diadochokinesis-buttercup` | 890 | 6 | **10** | 14 | 30 |
| `diadochokinesis-v2-tuh` | 702 | 7 | **14** | 19 | — |

Two things:

**The guess was good and the units are not.** A median of 10–11 repetitions makes the configured 10
a reasonable central tendency for the single-syllable families. For `pataka` and `buttercup` the
configured 30 counts *syllables* where this measure counts *repetitions* — 10 repetitions of a
three-syllable carrier. The same quantity in two units, in one field, with nothing saying which.

**The spread is why it can never be a gate.** `/pa/` runs from 4 at the 5th percentile to 24 at the
95th, a factor of six across the population. Owner-directed, and the data agrees: an expected count
is a heuristic and individuals vary. A measured median is a useful covariate to report beside a
rate. It is not a bound anything may be judged against.

## What else is here

29 measurements, of which `carrier_rejected` at 383,212 readings is by far the largest — the
visibility fix added on 2026-09-20, now the most-written measurement in the graph, which is what a
gate that discards evidence looks like once it is made to say so.

The distributions for the remaining gate inputs — `voiced_duration_s`, `glide_extent_semitones`,
`train_fraction_of_recording`, `verbatim_overlap_fraction`, `breath_coverage_fraction`,
`ddk_syllable_rate_from_ppg_decode_hz` — are in the JSON, and are the input to any refit of the
sixteen gates now in `verdict.gates`.

**Denominator note.** The scan read 62,273 of 62,548 unique recordings; 275 were still re-running.
The manifest carries 62,578 lines but only 62,548 distinct stems — 29 appear twice and one three
times — so every earlier figure quoted against 62,578 is 0.05% low.
