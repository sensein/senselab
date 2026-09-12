# Sweep grids derived from the corpus, not written down — 2026-09-12

Every threshold grid in `routing_analysis/detectors.py` was a hand-written constant. Twenty-four of
them, shared across 175 detectors by unit. A grid that stops short of the range its feature occupies
collapses everything past its end into one block no threshold divides, and the sweep reports the
detector as measured and weak when its firing side was never measured at all.

The constants are gone. Each grid is now derived at catalogue construction from that detector's own
distribution over the corpus, shipped as `data/detector_profile/2026-09-12.parquet`.

## The defect

Measured on all 62,547 recordings, polarity-aware — for `at_least`, the mass *above* the top
threshold; for `below`, the mass *under* the bottom one. The non-firing side is correctly ignored: a
threshold below every value of an `above`-polarity detector routes everything, which is a legitimate
sweep endpoint, not a blind spot.

```
detectors scored              175
  grid covers the firing side 128
  firing side undivided (>5%)  44
  feature is constant           3
```

The worst:

| detector | polarity | grid | data | undivided |
| --- | --- | --- | --- | --- |
| `voice.squim_stoi_max` | above | 0.3 … 0.9 | 0.14 … 0.9999 | 72.2% |
| `airway.residual_enhanced_energy_fraction` | above | 1e-06 … 0.9 | 4.7e-07 … 1.014 | 69.2% |
| `speech.yamnet_peak.plain` | above | 0.001 … 0.99 | 1.1e-12 … 1 | 64.4% |
| `glide.squim_si_sdr_max` | above | -20 … 15 | -18.71 … 33.8 | 61.7% |
| `voice.silence_fraction` | below | 1e-06 … 0.9 | 0 … 1 | 48.7% |

`voice.squim_stoi_max` is the clearest case. STOI's head emits 0–1, so the hand grid stopped at 0.9
on the reasoning that 0.9 is already a high intelligibility. The corpus median is 0.983 and the 80th
percentile is 0.997: nearly three recordings in four sit above the grid's last cut, and every one of
them reads identically to the sweep. Whatever separates them was never asked about.

### It has produced wrong conclusions three times

| detector | reported | grid stopped at | actual |
| --- | --- | --- | --- |
| `amplitude_peak_over_floor_db_max` | J 0.534 | 50 dB | optimum 55 dB, J 0.594 |
| `ppg_repetition_lag_segments` | 0.000 everywhere | 30 segments | values reach 1,199 |
| `praat_phonation_ratio` | scored nothing | 0.95 | median is exactly 1.0 |

Each was patched by hand afterwards — `DB_OVER_FLOOR_GRID` was extended to 80, `SEGMENT_LAG_GRID` to
1,200, `SATURATING_PROPORTION_GRID` to 1.0 — which is why three of the numbers above no longer
reproduce against the tree as it stood. That is the argument for deriving rather than patching: each
repair fixed one grid, left the other twenty-three alone, and left nothing behind that would catch
the twenty-fourth. The 44 undivided detectors are what three rounds of hand-patching left.

## What replaced it

A grid is now the detector's own **quantile ladder**: its minimum, the fifteen quantiles
0.001 … 0.999, and its maximum, deduplicated and sorted. Three properties follow by construction.

- It spans the data. The firing side always reaches the profiled extreme, so the block this document
  is about cannot exist. `build_catalogue` asserts it per detector and raises if it does not hold.
- Its resolution sits where the mass is. A hand grid spends thresholds on a uniform-looking spacing;
  a quantile ladder spends them where recordings actually are. `cough.amplitude_peak_over_floor_db_max`
  now carries 46.5, 49.0, 51.7, **55.0**, 59.8, 64.1 where the hand grid stepped 45, 50, 55, 60. The
  55 dB optimum the first sweep could not see had to be put there by hand afterwards; here it is
  q0.80, and it is on the grid because four recordings in five fall below it.
- Each threshold is a known fraction of the corpus. "The loosest cut inside a 5% over-routing
  budget" is then answerable by reading the ladder position, not by interpolating between cuts whose
  prevalence nobody measured.

### The quantiles, and the extremes

The ladder is `0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999` —
decile resolution through the body, finer at both tails because a routing decision made inside an
over-routing budget lives in a tail. Fifteen points is the same order as the constants it replaces
(10 to 21, median 14), so this is not a resolution change dressed as a coverage fix.

**`min` and `max` are in, for both polarities.** The argument against `max` is that an
`above`-polarity threshold there fires only on the recordings that tie the maximum, which is usually
one. The argument for it won:

- The sweep uses `>=`, so a threshold at `max` is a real row with a real confusion table, not an
  empty one.
- It makes "the grid reaches the profiled extreme on the firing side" an exact, checkable invariant
  rather than one qualified by "to within the 0.999 quantile". An invariant with an exception is how
  this defect survived.
- The 0.999-to-`max` interval holds about 63 recordings. Leaving it undivided would be a 0.1% version of
  the same defect, and the whole point is that the size of the undivided block is not the thing that
  makes it wrong.

`min` for an `above`-polarity detector, and `max` for a `below` one, are the route-everything
endpoint. They cost one row each and anchor the sweep's curve at the trivial end, which a
budget-constrained search wants as its starting point.

### Counts get their integers

`words`, `spans`, `tokens`, `phonemes` and `segments` are counts. Quantiles of a small count
collide: `airway.bracketed_throatclearing` takes five values (0–4), and fourteen of its fifteen
quantiles are 0. Deduplicating leaves a three-point grid, which is worse than the ten-point
`COUNT_GRID` it replaced even though five of those ten fired on nothing.

A count-unit grid is therefore **every integer from 0 to 32, unioned with the rounded ladder**.
Thirty-two is a resolution choice, not a fitted threshold: below it a count is individually
interpretable — three coughs, ten words — and the corpus carries enough mass at each integer to
estimate a cut; above it the ladder takes over and the tail is described by its quantiles.
`speech.words_lexical` gets 0–32 plus 57, 80, 198, 347 and 1069, against a hand grid that stopped at
30 for a feature reaching 1,069. The cap cannot reintroduce the defect: coverage comes from the
ladder's `max`, which is present whatever the cap is.

`0` is in every count grid. For `below` polarity it is the meaningful "no tokens at all" cut;
for `above` it is the route-everything endpoint. `1` — "any agreed word", "any bracketed cough" — is
the presence test and falls out of the enumeration rather than needing to be named.

### A gate costs resolution only where it costs evidence

A gated detector reads `GATE_CLOSED` (-1.0) when its corroborator did not fire, and that sentinel is
a large share of the corpus: eleven of the fifteen gated detectors read it at or past the median.
Its ladder is then mostly -1.0 and dedups to as few as five points, four of which sit in the top
decile. `cough.level_crest_db+hear_cough>=0.5` came out with 21.2, 24.7, 29.3, 33.7, 41.8 — honest
about what the gate leaves, but too coarse to find an optimum.

The sentinel is dropped, and the grid takes the ladder of the **ungated detector reading the same
primary feature**. A gate selects which recordings are scored; it does not change what is read.
`cough.level_crest_db+hear_cough>=0.5` and `cough.level_crest_db` read the same crest, so the
crest's own distribution is the right grid for both, and the gated one's values are a subset of it —
which keeps the coverage invariant true for free. Every gated detector in the catalogue has such a
sibling; the merge takes them from the 5–10 points of their own ladder to 20–38.

Nothing is interpolated anywhere. Every threshold in the catalogue is a measured quantile, a
measured extreme, an integer of a count, or one of the four declared cut points below. A test
asserts exactly that.

### Cut points a corpus cannot move

Two kinds of threshold are properties of the quantity rather than of this corpus, and a quantile grid
would destroy them.

**Sign tests.** `cough.yamnet_cough_minus_breath.plain` is the cough-set score minus the breath-set
score on the same classifier and the same stream. Zero is where the comparison flips, and it is not
on the ladder: q0.80 is -3e-06 and q0.90 is +0.0067, so the derived grid steps straight over the one
cut the detector was built to make. `cough.hear_cough_minus_breath.plain` is the same construction
on HeAR, stepping from -0.0108 to +0.0218. `airway.residual_correlation_residual` is a correlation,
where the sign says whether the residual tracks the original or opposes it; the ladder brackets zero
closely (q0.01 is +0.00089) but does not contain it. All three pin 0.0.

**The configured floor.** `taxonomy.consolidation_floor` is 0.2 in the shipped configuration, and
`report.py` marks the row where it falls so a sweep can say what the value in force is costing. That
marking needs 0.2 to be a row. It is pinned into all 60 score-unit grids. A configured decision the
sweep cannot express is a decision the sweep cannot report on.

Rejected, and why:

| candidate | why not |
| --- | --- |
| SI-SDR at 0 dB | parity between signal and distortion is a reference point, but no verdict flips there; it is a continuous quality whose operating point is empirical. q0.20 is -4.97 and q0.30 is +9.18, so the ladder brackets it. |
| `cough.level_crest_db` at 0 | also a `difference` reader, but of peak minus RMS, which is non-negative by construction (observed minimum 2.14 dB). A pin there would fire on everything. This is why the pins are a named map and not a rule over `difference` readers. |
| `praat_phonation_ratio` at 1.0 | genuinely meaningful — fully phonated — but it is the profiled maximum, so it is already in every grid that reads it. |
| residual energy fraction at 1.0 | the residual carrying as much energy as the original is a threshold worth having, but `airway.residual_enhanced_energy_fraction` exceeds it (max 1.014) and the others do not reach it; in neither case is 1.0 a flip. |

A pin naming a detector the catalogue does not declare raises at import. A stale pin that silently
applies to nothing is the same failure mode as a silent grid fallback.

### A grid that cannot be derived raises

`build_catalogue` raises `ValueError` when a declared detector is absent from the profile, when its
profiled feature is constant, or when the derived grid fails to reach the profiled extreme on the
firing side. There is no default grid to fall back to. A silent fallback is precisely how a grid that
never covered its feature survived three sweeps and three hand patches.

## The three constant detectors

```
speech.ast_peak.consensus   always 0 on all 62,547
airway.ast_peak.consensus   always 0 on all 62,547
voice.ast_peak.consensus    always 0 on all 62,547
```

Removed. Pre-alpha, so deleted rather than deprecated.

**This is not an extraction bug.** The consensus taxonomy is consolidated from the per-span
classifiers, and `nodes/taxonomy.py` names them: `PER_SPAN_CLASSIFIERS = {"yamnet": "span_yamnet",
"hear": "span_hear"}`. AST is a whole-file classifier in this pipeline — it runs over the plain,
enhanced and residual streams and never per span — so `peak_by_classifier` never carries an `ast`
key, and `peak_key("consensus", "ast", label)` is never written. The three detectors asked for a
stream-by-classifier combination the pipeline does not produce and never will while AST stays
whole-file. Running AST per span is a design question with a real cost; it is not a repair to
something broken.

What *is* worth recording is the mechanism that turned the miss into a zero. `detector_value`
guards stream availability only for `plain`, `enhanced` and `residual`:

```python
if stream in ("plain", "enhanced", "residual") and f"{stream}|{classifier}" not in features.classifier_streams:
    return None
```

The `consensus` and `span` streams fall through to `features.peaks.get(key, 0.0)`, so an absent
measurement reads as a confident 0.0 rather than as `None`. `None` excludes the recording from the
detector's scoring; 0.0 scores it as a negative. That is why three detectors could read a field that
does not exist, on every recording in the corpus, and report a plausible-looking null result.

The removal is structural rather than a deletion of three names: `_peak_detectors` now builds a
consensus detector only for the classifiers in `CONSENSUS_CLASSIFIERS`, which is
`frozenset(SPAN_CLASSIFIERS.values())` — the same set the consensus taxonomy is consolidated from.
Adding a per-span AST pass would bring the three detectors back on its own; nothing else can.

## What changed, measured

| | before | after |
| --- | --- | --- |
| detectors in the catalogue | 175 | 172 |
| firing side undivided (>5%) | 44 | 0 |
| grids reaching the profiled extreme | 36 | 172 |
| total sweep points | 2,472 | 3,066 |
| grid size (min / median / max) | 10 / 14 / 21 | 5 / 17 / 38 |

All 172 grids changed. 136 gained an operating region no threshold could previously reach; for most
that region is a tail, for the 44 above it is a large fraction of the firing side. 118 grids grew,
38 shrank. The five named at the top:

| detector | grid before | grid after | undivided |
| --- | --- | --- | --- |
| `voice.squim_stoi_max` | 0.3 … 0.9 | 0.140 … 0.99993 | 70% → 0% |
| `airway.residual_enhanced_energy_fraction` | 1e-06 … 0.9 | 4.7e-07 … 1.014 | 60% → 0% |
| `speech.yamnet_peak.plain` | 0.001 … 0.99 | 1.1e-12 … 1.0 | 60% → 0% |
| `glide.squim_si_sdr_max` | -20 … 15 | -18.71 … 33.80 | 60% → 0% |
| `voice.silence_fraction` | 1e-06 … 0.9 | 0 … 1.0 | 40% → 0% |

The 38 that shrank are the honest cost. `airway.praat_phonation_ratio` goes from 21 thresholds to
9, because over half the corpus reads exactly 1.0 and no grid can resolve inside a point mass; the
old grid's 0.93, 0.95, 0.97, 0.98, 0.99, 0.995 and 0.999 were eight rows describing the 10% of the
corpus between q0.40 and q0.50. `airway.bracketed_throatclearing` goes from 10 to 5, which is every
value the feature takes. Losing thresholds that fired on nothing is not a loss of measurement.

The sweep costs 24% more rows. Nothing else in the pipeline changes: a grid is read only by
`sweep_points`, and only `report.py` calls it.

## Re-deriving

The profile is a recorded measurement, not a hand-edited file. It carries `profile_version`, the
generation date, the corpus description, the recording count and the quantile ladder it was taken
at; `load_detector_profile` validates all of it and refuses a ladder that decreases, a non-finite
quantile, an unknown state or a missing recording count.

A new sweep ships as a new dated file beside the old one; the loader takes the last by name. A
detector added to the catalogue without being in the newest profile raises at import, which is the
intended way to find out that the corpus needs re-sweeping — rather than discovering it from a
detector that scored 0.000 everywhere.
