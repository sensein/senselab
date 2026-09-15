# The forced re-derivation, rehearsed on one real recording — 2026-09-14

The first measurement of what the per-recording F0 narrowing
([`praat-instrument-audit.md`](../20260817-triage-workflow-dag/praat-instrument-audit.md) step 2,
landed by this plan's Task 1) does to real scalars, and the reason the corpus pass Task 9 Step 6
describes **was cancelled by the owner on 2026-09-14** rather than submitted.

Nothing here says either reading is better. There is no ground truth on this recording. What is
established is that on it the flag claims a narrowing that did not occur, and that the scalars move
materially.

## What was run

One real store, a Story-recall recording: **88.07 s**, **115 consensus words**. The extend pass was
rehearsed locally on it — seeded under `dea0a622^`, which still carries the sex-typed bin, and
re-derived at `db7c3070` with `scripts/extend_ppg_praat.py --force`. Both readings are on the
unchanged `enhanced` stream; the only thing that differs between them is the range rule.

## The derived range is the unnarrowed search range, and the flag says otherwise

**Old behaviour.** The wide 50–600 Hz pass gave **5,691 voiced frames of 17,603**; the z-filtered
mean was **195.14 Hz**, at or above the 170 Hz threshold, so the bin selected **[100.0, 500.0]**
(*'female'/'child'*). That reproduces the stored `mean_f0_hertz` of **218.68** exactly, which is what
makes the pairing below attributable to the rule change and to nothing else.

**New behaviour.** p5 = **62.34**, q3 = **238.62**, p95 = **265.81**.

- floor `max(50, 62.34 / 1.5 = 41.56)` = **50.00**
- ceiling `min(600, max(238.62 × 2.5 = 596.55, 265.81 × 1.5 = 398.71))` = **596.55**

The derived range is **`[50.00, 596.55]`** against a search range of `[50, 600]`: the floor sits
*exactly* on the search floor and the ceiling **3.45 Hz** below the search ceiling. And
`pitch_range_fell_back = 0.0`, asserting that the narrowing branch was taken and applied.

Both clamps are in the same expression,
`src/senselab/audio/tasks/features_extraction/praat_parselmouth.py:475-489` — the percentiles at
`:475-477`, the fallback predicate at `:478`, the floor's `max` at `:481` and the ceiling's
`min(..., max(...))` at `:482-488`. Two independent mechanisms reach the same place.

### Floor pinning — new, and recorded nowhere before this

**16.1%** of the wide pass's voiced frames sit below 100 Hz — **12.0%** below 75, **2.0%** below 60 —
a second mode near **52–70 Hz**. Creak or period doubling that the 50 Hz search admits. That drags p5
down far enough that `p5 / 1.5` falls under the search floor and the `max()` at `:481` pins the floor
there.

Generalised: **any recording with roughly ≥5% creaky frames pins its floor at the search floor.** The
rule's own fallback does not catch this — the fallback tests p95, which on this recording is 265.81,
far above `2 × 50`, so the narrowing branch is taken and reports itself as taken.

This is a distinct hazard from the one step 2 already records. The recorded one is the 60 Hz-hum case
where a one-pass narrowing returns a range **excluding** the speaker; the pinned-contour fallback
closes that. This one returns a range that **contains** the speaker and contains everything else too,
while the flag says it was narrowed.

### Ceiling pinning — already owed, now realised on a real recording

`q3 × 2.5 ≥ 600` whenever q3 ≥ 240 Hz. This voice's q3 is **238.62**, **1.4 Hz** under. For ordinary
adult-female and child voices the ceiling term is effectively a constant at the search ceiling.

This is the fact already recorded as owed under the audit's step 4, beside the
`voice.f0_search_range_hz` / CPPS-band tension — *"`2.5 × q3` reaches 600 for any q3 ≥ 240 Hz, so
above roughly that F0 the ceiling every recording gets is the search bound itself and not a
narrowing."* It is cross-referenced here, not restated: what is new is that it has now been met on a
real recording rather than reasoned about.

### What is owed

**A code change, at minimum to the flag.** `pitch_range_fell_back = 0.0` is read as "the narrowing
branch was taken and applied". On this recording the branch was taken and applied nothing. A
consumer cannot distinguish a narrowed range from a pinned one without recomputing the predicate from
`pitch_floor`, `pitch_ceiling` and the search bounds it does not have. Whether the *rule* should also
change — whether a floor pinned by a creak mode is the right instrument — is a separate question this
measurement does not settle, because there is no ground truth here.

**The corpus pass is cancelled pending that decision.** Not deferred for capacity: the pass would
have re-derived the Praat block across the stores the `ppg_20260911` manifest names, stamping
`pitch_range_fell_back = 0.0` onto every one of them.

## The contour moves more than the summary does

Contour p99 goes **279.6 Hz** (old bin) → **546.9 Hz** (derived). The ceiling is what changed, and
the top of the contour follows it.

## Analysis windows moved the opposite way the audit predicted

The audit reasons from a **60 → 100 Hz** floor move: intensity `3.2 / f₀` **53.3 → 32.0 ms**,
harmonicity `4.5 / f₀` with `periods_per_window=4.5` **75.0 → 45.0 ms**. Those are the retired bin's
own step and are correct as a description of it.

Here the floor moved **100 → 50 Hz**, so the same two formulas give **intensity 32.0 → 64.0 ms** and
**harmonicity 45.0 → 90.0 ms** — **both doubled**. Whatever the narrowing does to coverage on this
recording, it is not the shrinkage the bin's figures suggest to a reader skimming them. The audit's
finding 1 is annotated accordingly.

## Paired scalars, old bin `[100, 500]` → new derived `[50, 596.55]`

| scalar | old | new | change |
|---|---|---|---|
| mean_f0_hertz | 218.680 | 201.528 | −7.84% |
| **std_f0_hertz** | 34.359 | 77.881 | **+126.67%** |
| mean_hnr_db | 6.7691 | 6.1240 | −9.53% |
| std_hnr_db | 6.2661 | 5.0065 | −20.10% |
| local_jitter | 0.041057 | 0.034040 | −17.09% |
| localabsolute_jitter | 1.8800e-4 | 1.6799e-4 | −10.65% |
| rap_jitter / ddp_jitter | 0.020520 / 0.061561 | 0.016923 / 0.050768 | −17.53% |
| ppq5_jitter | 0.022950 | 0.018540 | −19.21% |
| local_shimmer | 0.127460 | 0.122509 | −3.88% |
| localDB_shimmer | 1.19727 | 1.17281 | −2.04% |
| apq3 / apq5 / apq11 / dda | 0.050802 / 0.061666 / 0.079467 / 0.152407 | 0.049370 / 0.062193 / 0.079750 / 0.148110 | −2.82 / +0.85 / +0.36 / −2.82% |
| CPPS mean / std | 7.43028 / 2.40814 | 7.38979 / 2.30714 | −0.55 / −4.19% |
| mean_f1 / mean_f2 | 503.84 / 1665.23 | 529.20 / 1683.22 | +5.03 / +1.08% |
| std_f1 / std_f2 | 222.07 / 435.46 | 260.75 / 429.02 | +17.42 / −1.48% |
| mean_b1 / mean_b2 | 281.51 / 381.48 | 305.49 / 390.13 | +8.52 / +2.27% |
| std_b1 / std_b2 | 255.45 / 385.94 | 327.26 / 403.92 | +28.11 / +4.66% |
| mean_intensity_db | 69.3886 | 69.4855 | +0.14% |
| std_intensity_db | 13.9699 | 12.8542 | −7.99% |
| range_ratio_intensity_db | 3.64297 | 3.22489 | −11.48% |
| spectral gravity/stddev/skew/kurt/slope/tilt | 345.01 / 339.08 / 7.7475 / 142.03 / −18.245 / −4.6387e-3 | 340.31 / 346.45 / 7.7845 / 143.93 / −18.784 / −4.7276e-3 | −1.36 / +2.17 / +0.48 / +1.34 / −2.95 / −1.92% |

**Six of the forty-five are unmoved** — `duration`, `speaking_rate`, `articulation_rate`,
`pause_rate`, `mean_pause_duration`, `phonation_ratio`. The speech-rate block takes no F0 range,
which is the arithmetic Task 7's premise already states (34 of the 40 descriptors conditioned on the
derived range, 6 independent of it) confirmed on a recording.

`std_f0_hertz` at **+126.67%** is the entry to read first: a ceiling raised from 500 to 596.55 admits
the octave-doubled frames the 500 Hz cap used to exclude, and the spread of the contour is where that
shows.

## The mechanics worked

Recorded because the pass was cancelled on the finding above and not on the driver, and the next
person to run it should not re-establish this:

- `--force` appended a second `praat_features` and retired the first, via an activity whose step is
  `praat_features_superseded` (`extend.py:82`) carrying `_RETIRED_F0_RANGE_REASON`
  (`scripts/extend_ppg_praat.py:114-115`, called at `:354-355`).
- `n_features` went **40 → 45** on the new measurement — the forty descriptors plus the five range
  keys Task 8 added.
- The PPG measurement id was **unchanged** and **ppgs was never invoked**. The one-sided `--force`
  behaves as Task 7 Step 5 describes.
- A **second unforced run wrote nothing**.
- **13 new store lines.**
