# The Praat instrument audit

What the Praat-derived measurements in triage actually measure, and where the wrapper departs from
Praat's own guidance. **This is instrument provenance, not branch design** — the branch documents
cross-reference it rather than restating it.

Findings below were **measured with probes**, not read off the source. Where a number appears it is a
probe result.

## The clean contrast

Two modules measure the same quantities to opposite standards, and triage reads both.

| | `tasks/phonation/api.py` | `tasks/features_extraction/praat_parselmouth.py` |
| --- | --- | --- |
| undeclared literals | **zero** | roughly sixty |
| config-reachable parameters | every Praat parameter, as a required keyword wired to a config key with a written derivation | **three** |
| support counts exposed | **five of five functions** | **zero of thirteen** |

## The route into triage

- `preprocess.py:883-885` calls `extract_praat_parselmouth_features_from_audios` on the **`enhanced`**
  stream. **All 40 Praat scalars come from that one call**, and only `time_step` and `window_length`
  are config-reachable.
- `preprocess.py:941` derives the F0 range on **`plain`**.
- `voice.py:71-73` derives it again on **`plain`**.

So **three streams are analysed, and the same F0 range is derived twice independently** — and, given
finding 1, the two derivations can land in different bins for the same recording.

## Findings, ranked by distortion

### 1. The 170 Hz binary sex bin is the root cause

`praat_parselmouth.py:429-436` selects one of two hardcoded floor/ceiling pairs from the recording's
trimmed mean pitch: below 170 Hz → (60, 250); otherwise → (100, 500). That range then sets intensity,
harmonicity, formant pulse sampling, jitter, shimmer, LTAS, spectral-moment voicing and pitch
descriptors.

**Measured downstream step at the 40 Hz floor change**: intensity window **53.3 → 32.0 ms**,
harmonicity window **75.0 → 45.0 ms**, minimum analysable segment **91.7 → 55.0 ms** — a **1.67×**
discontinuity in a continuous measurement, which can flip between two streams of the same recording.

**The project already documented this as an error not to repeat.** `capability-map.md:117` and
`:314` name this bin as the exact mistake to avoid, and the code does it anyway.

Consumers: [`branch-voice.md`](branch-voice.md) V3, V4, V7, V8.

### 2. The CPPS `> 4` cut deletes the dysphonic range

`:800` appends a per-interval CPPS value only when `CPP_Value > 4`.

**Measured**: severe dysphonia **2.78 dB**, dropped. Severe dysphonia at F0 410 Hz, **2.17 dB**,
dropped.

Two consequences:

- When every interval is dysphonic the list is empty and `:809-810` returns NaN — **identical to the
  crash return at `:821`**. A severely dysphonic recording and a crashed one are indistinguishable in
  the corpus.
- For partially dysphonic recordings the reported mean is over **only the intervals that exceeded
  4** — selection on the outcome variable.

### 3. The vuv mean period is 10× Praat's default and violates Praat's own constraint

`:752` calls `To TextGrid (vuv)` with maximum period **0.02** and mean period **0.1**. Praat's
default mean period is 0.01, and Praat documents that mean period must be **less than** maximum
period. Here it is five times larger.

**Measured** on five 120 ms voiced runs in 1.2 s: true voicing **0.599 s** at Praat's default,
**1.021 s** at the code's value — a **70% inflation**, each segment carrying roughly 100 ms of
silence into the cepstrogram.

**Severity scales inversely with voiced-run length**, so it is worst on fragmented, dysarthric and
apraxic productions and negligible on the sustained-vowel control task. And it **depresses CPPS**,
pushing values toward finding 2's cut — the two multiply.

### 4. The CPPS peak search is 60–330 Hz regardless of the caller's ceiling

`:778-779` hardcodes the peak search band, silently replacing the 500 Hz ceiling finding 1 derived.

**Measured penalty at F0 420 Hz: 3.8 dB** (10.33 vs 14.14) — roughly the entire normal-to-dysphonic
span. Children and high-F0 females read as dysphonic; combined with noise they cross the `> 4` cut
and disappear entirely.

### 5. Support counts: zero of thirteen

Not one function in `praat_parselmouth.py` returns a frame, cycle or interval count. `:759`, `:804`,
`:896` and `:1036-1039` each **compute one and discard it**.

**So [`branch-conventions.md`](branch-conventions.md)'s mandatory support count is satisfiable on the
`phonation/api.py` path and impossible on the Praat path** — and findings 1–4 are unauditable after
the fact, because you cannot tell whether a `mean_cpp` rests on 2 intervals or 60.

**The convention is currently unsatisfiable for the Praat scalars.** Closing it needs those four
functions to return the count they already compute.

### 6. `range_db_ratio` is dimensionally invalid

`:566` computes `range_db_ratio = max_dB / min_dB` — a ratio of two logarithmic quantities.

**Measured**: a buzz with no silence gives **1.000**; add 0.5 s of silence at each end and
`min_dB = −344.05`, so the ratio is **−0.244**. A "range ratio" that goes negative whenever the
recording contains a quiet passage, exported to the corpus as `range_ratio_intensity_db`.

**It tracks silent fraction, not dynamic range.** The meaningful quantity is `min_dB − max_dB`.

**This bears directly on [`branch-voice.md`](branch-voice.md) V6** — the loudness measure must not be
built on it.

### 7. Spectral moments are band-limited to 5 kHz by an unnamed default

`:999` builds the spectrogram with "default settings other than window length and frame shift", so
`maximum_frequency` is Praat's 5000 Hz default and **never appears in the source** — it reads as
absent rather than chosen.

High-frequency energy is the acoustic correlate of breathiness and turbulent noise, which is the
dysphonia signal. Gravity, skewness and kurtosis are computed on a band that excludes it.

### 8. Jitter and shimmer return NaN for period doubling

**Measured**: `To PointProcess (periodic, cc)` places **0 pulses** on an alternating-period signal
and **1** on a period-doubled one — so the 1.3 maximum-period factor never gets the chance to act.

And with floor 60 from the bin, a 45 Hz or 55 Hz source yields **zero pulses**: vocal fry, low male
voices and Parkinsonian creak are excluded **upstream by the bin**, not by the 0.02 s period ceiling.

**`default.yaml:145` names `phonation.period_doubling_factor: 2.0` as a phenomenon of interest, and
the measurement is structurally incapable of representing it.** A diplophonic voice reads as
*unmeasurable*, not as *severely disordered*.

**The jitter/shimmer literals `0.0001 / 0.02 / 1.3 / 1.6` are Praat's own form defaults**, not
senselab inventions. What makes them a problem is different: Praat's *Voice 2. Jitter* manual says
the period floor should be `0.8/ceiling` and the ceiling `1.25/floor` — for the (100, 500) bin,
**0.0016 and 0.0125**, not 0.0001 and 0.02. The code passes form defaults while passing a *derived*
range to the point process, which is the mismatch Praat's documentation tells you to avoid.

### 9. The `hnr < 60` branch is measurably dead

`:154-155` drops `min_dip` from 4 to 2 when the recording's mean HNR is below 60.

**Measured mean HNR**: synthetic buzz **50.75 dB**; with noise **16.90 dB**; pure sine **105.60 dB**.
Real speech never approaches 60, so **`min_dip` is always 2 on all 62,547 recordings** and the
documented "clean signal" setting of 4 is dead code.

It costs one whole-file harmonicity computation per recording, and it remains a live data-dependent
branch on the aggressively denoised `enhanced` stream triage actually feeds it.

### 10. `voice_tracks.npz` carries unmasked −200 dB sentinels

`voice.py:305` and `:374` write `hnr_db` with Praat's undefined-frame sentinels unmasked — **measured
389 sentinel frames** in a padded 2 s signal — while `phonation.hnr_floor_interval_db` is null so
nothing masks them downstream. Any mean or percentile over that array is destroyed, and will silently
disagree with `extract_harmonicity_descriptors` on the same audio.

**A useful negative result**, contrary to a widely-held assumption: `extract_harmonicity_descriptors`'
own `Get mean` **excludes** the sentinels, so it is *not* confounded with pause fraction.

### 11. A derivation is factually wrong

`config-derivations.md:571-578` documents `phonation.periods_per_window: 4.5` as *"Praat's own
documented defaults for the cc method"*. **Parselmouth 0.4.7 and the Praat form both say 1.0.**

The value may still be the right choice; the justification for it is not correct. **This is the first
derivation found to be factually wrong rather than merely thin**, which matters for how much weight
`config-derivations.md` can carry unchecked.

### 12. joblib caches a closure

`:1476-1478` — `time_step`, `window_length` and eleven toggles are free variables of `_extract_one`
rather than arguments, so two runs with different settings and one `cache_dir` collide.

**Dormant in triage** (no `cache_dir` is passed at `preprocess.py:883`) but a live hazard for any
batch over 62,547 recordings.

## Also recorded

**`extract_speech_rate` deviates from Praat on two thresholds.** `min_pause` **0.3 s against Praat's
0.1 s** — 3×, so hesitation pauses at 0.3–0.4 s sit on the edge and `pause_rate` under-reads for
halting speech; and minimum sounding interval **0.1 s against 0.05 s**, 2× stricter, dropping short
voiced fragments.

**Its `to_pitch_ac` call at `:290` differs from Praat on six parameters** — floor 30 vs 75,
candidates 4 vs 15, voicing_threshold 0.25 vs 0.45, voiced_unvoiced_cost 0.25 vs 0.14, ceiling 450 vs
600 — and **the code says so itself**, across five consecutive comment lines at `:294-304`:

> `# Key Hyperparamter are different to praat recommended - can't find a reason for this`
> `# max_number_of_candidates: Positive[int] = 15 (can't find a reason for this value being lower)`
> `# voicing_threshold: float = 0.45, (can't find a reason for this value being different)`

The code declares its own literals undeclared.

**`measure_f1f2_formants_bandwidths`' parameters are unreachable.** The wrapper forwards only `snd`,
`floor`, `ceiling` and `frame_shift` (`:1342-1347`), so `max_formants`, `maximum_formant_hz`,
`window_length` and `pre_emphasis_from_hz` cannot be set by any caller — while the identical four
values are config-driven on the `phonation/api.py` path. `maximum_formant 5000` is the adult-male
convention where child is 8000.

## Where this lands

| finding | affects |
| --- | --- |
| 1, 8 | [`branch-voice.md`](branch-voice.md) V3, V4, V7 |
| 2, 3, 4, 5 | `branch-voice.md` V4 — CPPS and its support |
| 6 | `branch-voice.md` V6 |
| 7 | `branch-voice.md` V4; [`branch-quality.md`](branch-quality.md) Q2 |
| 9, and the `extract_speech_rate` deviations | [`branch-speech.md`](branch-speech.md) S4; [`branch-ddk.md`](branch-ddk.md) D2 |
| 10 | `branch-voice.md` V1, V4 |
| 5, 11 | [`branch-listening-sample.md`](branch-listening-sample.md) |

## What this changes about "owed"

[`branch-listening-sample.md`](branch-listening-sample.md) divides owed items into values with a
derivation, values marked unmeasured, null keys, and literals in no config. **This audit adds a
fifth kind**: parameters that are neither configurable nor reachable, whose values deviate from the
instrument's own documented guidance, and whose effect has now been measured.

Those are not owed a listening sample. They are owed a **code change**, and finding 11 shows the
derivations file is not by itself sufficient evidence that one has been thought through.
