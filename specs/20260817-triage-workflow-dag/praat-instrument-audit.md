# The Praat instrument audit

What the Praat-derived measurements in triage actually measure, and where the wrapper departs from
Praat's own guidance. **This is instrument provenance, not branch design** — the branch documents
cross-reference it rather than restating it.

Findings below were **measured with probes**, not read off the source. Where a number appears it is a
probe result.

---

## Finding 0 — every Praat scalar is computed on FRCRN-enhanced audio

**This reorders everything else in this document.**

`preprocess.py:880` resolves the **`enhanced`** stream and hands it to
`extract_praat_parselmouth_features_from_audios`. `preprocess.py:2338` shows what `enhanced` is:
`enhance_audios([plain], model=model)` — ClearVoice/**FRCRN**, a deep speech-enhancement network.

So all 40 scalars — CPPS, HNR, jitter, shimmer, LTAS, slope and tilt, spectral moments, formants,
speech rate — feeding [`branch-voice.md`](branch-voice.md) V4 and V6,
[`branch-speech.md`](branch-speech.md) S4, [`branch-ddk.md`](branch-ddk.md) D2 and
[`branch-airway.md`](branch-airway.md) A6 are measured on the output of a generative denoiser.

**Three things make this worse than anything below.**

**The project already warns about exactly this, about a weaker version of it.**
[`branch-conventions.md`](branch-conventions.md) states that *"spectral gating manufactures HNR and
CPP values outright"* — written about **consumer** noise suppression arriving in the recording. The
pipeline then applies a more aggressive one deliberately, and until now no document connected the two
sentences.

**The distortion is correlated with the variable of interest.** FRCRN is trained to reconstruct
typical speech from noise. Aperiodic energy is what it removes — and **breathiness is aperiodic
energy**. It therefore normalises dysphonic voices more than typical ones. That does not add noise to
V4; it **inverts its sensitivity**: the measurement is most altered exactly where the finding would
be.

**It is out of domain on most of this corpus.** Sustained vowels, coughs, DDK trains and maximal
shouts are none of them speech-in-noise. And the same `enhanced` stream feeds the PPG, hence
`ddk.ppg_segment_rate_per_s`, hence **DDK routing**.

### The bound — what finding 0 does *not* reach

There are exactly **two** `resolve_stream(..., "enhanced")` sites in PREPROCESS: `:758`
(`ppg_input`) and `:880` (the Praat call). Everything else reads `plain` or `preemphasised` —
**HeAR (`:1856`), YAMNet (`:1922`) and SQUIM all run on `plain`.**

So finding 0's scope is **the Praat scalars and the PPG, and nothing else.** AIRWAY's per-span
evidence, the taxonomy labels and the quality measures are not implicated. This is worth stating
plainly, because the finding otherwise reads as "the whole corpus is compromised", which it is not.

### The suppressions attach to the function, not to a branch

**Findings 2–5 are properties of `extract_cpp_descriptors`, so every caller inherits them.** An
earlier version of this document attributed them to [`branch-voice.md`](branch-voice.md) V4 alone,
and V4 duly suppressed CPPS while [`branch-speech.md`](branch-speech.md) S4 went on listing
"connected-speech CPP" unqualified.

**S4's population is the larger one — roughly 25,000 recordings against V4's 5,113 — and finding 3's
70% vuv inflation scales *inversely* with voiced-run length.** So the distortion is **worst on
connected speech** and mildest on the sustained vowel where it was first suppressed. The same holds
for the `> 4` cut, the 60–330 Hz search and `voicing_threshold=0.3`.

The rule generalises: **a finding about a function is stated once, here, and every document naming
that function inherits it.**

**Consequence for the rest of this document.** Findings 1–12 are real, measured, and worth fixing —
but they are repairs to an instrument pointed at the wrong signal. Read them as *what remains once the
stream is correct*, not as the primary problem.

---

## The remediation path

In this order. Steps 0 and 1 are the ones that matter; the rest are wasted before them.

### Step 0 — largest single gain, and it is *not* free: stop publishing the compromised scalars

**62,547 stores already carry 40 Praat scalars** computed on FRCRN output through a sex-binned range.
Four — `mean_cpp`, `std_dev_cpp`, jitter, shimmer — carry names that will be read against published
norms, and `range_ratio_intensity_db` is dimensionally invalid (finding 6) and already exported.

**The marker already exists and nothing reads it.** Every Praat measurement is written with
`signal="enhanced"` (`preprocess.py:892`), and the PPG likewise (`:809`). So "flag them" is already
done and it changed nothing.

**The effective Step 0 is therefore a code change: make every reader of the Praat scalars require
`signal == "plain"`.** Stronger than a marker, far smaller than re-deriving anything, and it makes
the compromised values unreadable rather than merely labelled. CLAUDE.md settles the
withdraw-versus-flag question this document previously left open — cache invalidation is free, and
pre-alpha replaces outright.

**An earlier version labelled this step "no code", which was wrong and would get it skipped as
trivial.** Withdrawing or flagging 40 scalars across 62,547 finished stores is an extend driver —
**more code than step 1's stream switch**. The ordering here is by *risk*, not by effort: step 0
comes first because leaving norm-bearing values addressable is the most damaging state, not because
it is the cheapest.

**And the verdicts are the larger hazard, which Step 0 did not mention.** VOICE `FAIL`s on **every**
recording today, and every branch document insists a `FAIL` means "no phonation found" — so 62,547
stores carry a **conclusion that is 100% instrument artefact**. A verdict reads as a conclusion in a
way a scalar does not.

**The derived artifacts carry the same values and were also unmentioned**: reports, figures, and any
recompute by `scripts/analyze_routing_evidence.py`.

### Step 1 — switch the Praat stream to `plain`

**Nothing downstream is defensible until it happens**, and every wrapper repair below is wasted work
before it.

**It is not "one argument" — that framing was wrong.** The same function writes `signal="enhanced"`
on the measurement (`preprocess.py:892`) and `derived_from=(enhanced_id,)`, and its docstring and
`Raises:` clause both name the enhanced stream. Beyond the function it **invalidates 40 scalars ×
62,547 stores**, so it needs a `CACHE_SCHEMA_VERSION` bump and a corpus recompute.

**And say what carries noise robustness once FRCRN leaves the path.** The governing contract requires
a raw-versus-enhanced **pilot** for the structurally identical diarization decision. The domain
argument here is stronger — FRCRN removes the aperiodic energy that *is* the measurement — but the
asymmetry should be stated rather than left as an inconsistency between two decisions of the same
shape.

### Step 1b — switch the PPG stream, which is a different site and a different event

The PPG resolves `enhanced` at **`preprocess.py:758`** (`ppg_input`), writing `signal="enhanced"` at
`:809` — **a separate call site from the Praat one.** An earlier version of this path scoped step 1
to `:880` alone, which would have repaired the scalars and left the routing gate reading denoiser
output.

**This one is not a scalar re-derivation.** `ppg.segment_rate_per_s` is the `ddk.ppg_segment_rate_per_s`
gate feature, so switching the stream **changes DDK routing on every recording in the corpus** —
which branch runs, not merely what a number reads. It needs its own before-and-after count, in the
same way [`branch-quality.md`](branch-quality.md)'s rule (a) gate count does.

[`branch-ddk.md`](branch-ddk.md) D1 already specifies reading `plain`; the ordered path did not.

### Step 2 — replace `derive_f0_range`, not the wrapper

Highest leverage in the codebase: it is the shared root of the sex bin (finding 1) and the pulse
exclusions (finding 8), it contaminates **both** modules (see the corrected contrast below), and it is
roughly fifteen lines.

Narrow per recording from the wide search — robust percentiles of the wide-search contour **in
log-Hz**, with declared margins. This is **Hirst's two-pass method**, which is the precedent and
should be named as such. Keep the typed absence. And **distinguish the crash return from genuine
absence**, which it currently conflates (below).

**But narrowing is not right for every task, and step 2 and [`branch-voice.md`](branch-voice.md) V7
currently prescribe opposite things.** Narrowing buys octave-error robustness on **stationary**
material and is **actively wrong on a glide**: the derived ceiling is then set by how high the
speaker went, which makes V3's "did F0 reach the derived limit" flag partly circular. **3,150 glide
recordings sit on that difference**, so the narrowing is task-conditioned, not universal.

### Step 3 — build V4 on `phonation/api.py`

**Compute jitter and shimmer directly from the `PeriodMark` sequence.** The mechanics work:
`PeriodMark` carries `time_s`, `period_s` **and** `amplitude`, so successive period differences and
successive peak-amplitude differences give both measures.

That yields the variant name, the support count, and — decisively — **the same point process for the
value and for its validity qualifier**, which V4 requires and which the current split cannot deliver
(finding 8).

**But the admission rule is the measurement, and this step omitted it.** Jitter and shimmer are
*defined* by which consecutive cycle pairs are allowed to contribute. Praat excludes pairs whose
period ratio exceeds 1.3; `period_marks` applies only a range test (`phonation/api.py:147`) and **no
successive-ratio constraint at all**.

Computing over that sequence unfiltered lets a single octave error or one voice break dominate the
value. **That may be exactly what is wanted** — it is how a diplophonic voice comes to read as
*disordered* rather than *unmeasurable*, which finding 8 identifies as the current failure. But it is
then **not the published quantity**, and the entire difference between the two is the admission rule.

**So it is a design decision, stated as one, and forced into the name** per
[`branch-conventions.md`](branch-conventions.md)'s convention-in-the-name rule.

### Step 4 — implement CPPS directly, roughly thirty lines

Log-power spectrum → cepstrum → robust regression over the quefrency range → peak prominence,
frame-wise, returning the frame count.

This removes the interval gating, so it works on the aperiodic voices that motivated promoting CPPS
in the first place; has **no value cut** (finding 2); is **duration-weighted** rather than unweighted;
and carries a support count (finding 5).

**The quefrency band is a fixed wide one, declared — not derived per recording.** An earlier version
of this step said "from the recording's own derived F0", which defeats the step's own purpose three
ways:

- on a type-3 voice `derive_f0_range` **raises** (`phonation/api.py:60-70`), so a band derived from it
  is **unavailable on exactly the population the reimplementation exists to serve**;
- CPPS is a peak prominence measured against a regression over a quefrency range, so **changing the
  range changes the value** — a per-recording band destroys cross-recording comparability and
  contradicts the common-band rule in this same document set;
- it is not what the method does: the published convention uses a fixed wide search.

**Declare 60–500 Hz**, which covers children and the top of an upward glide while fixing the 330 Hz
truncation (finding 4).

### Step 2b — track F0 on the signal the range was derived on

`phonation_tracks` derives the F0 range on **`plain`** and then tracks F0 on **`preemphasised`**
(`preprocess.py:939-950`). [`branch-voice.md`](branch-voice.md) V1 identifies this as **structurally
the same mismatch this audit condemns for the jitter form defaults** (finding 8) — a range derived
under one condition applied under another — and the ordered path omitted it.

It matters for the same reason: +6 dB/octave attenuates the fundamental relative to the upper
harmonics, **raising octave-error risk upward**, worst on low-F0 and creaky voices. Octave errors
then propagate into the steadiness qualifier V1 requires and into the bin selection of finding 1.
Praat's guidance is to track pitch on the unmodified signal.

### Step 5 — report **instrument coverage** as a first-class measurement

Jitter and shimmer return NaN at the disordered end, so a corpus distribution of either reads
**conspicuously healthy** — because **the disordered cases are missing, not extreme.**

**"Proportion of recordings on which the point process placed no pulses", per group**, is
parameter-free and needs no norm and no listening sample.

**Call it instrument coverage, not a dysphonia indicator.** An earlier version framed it as "arguably
a better dysphonia indicator than the values that survive", which is the normative reading every
other capability here declines. The no-pulses outcome is produced by the **analysis floor** — a 45 Hz
source yields zero pulses because the floor is 60 (finding 8) — plus SNR, level and stream. It is
confounded with F0 range, hence with sex and age, and with device and duration.

**And CPPS drops out of this list once step 4 lands.** CPPS is missing at the disordered end *only*
because of the `> 4` cut, which step 4 removes. After the path completes, **only jitter and shimmer
are missing-not-at-random.**

---

## The contrast, and its limit

Two modules measure the same quantities to different standards, and triage reads both.

| | `tasks/phonation/api.py` | `tasks/features_extraction/praat_parselmouth.py` |
| --- | --- | --- |
| undeclared literals **in the body** | **zero** | roughly sixty |
| config-reachable parameters | every Praat parameter, as a required keyword wired to a config key with a written derivation | **three** |
| support counts exposed | **five of five functions** | **zero of thirteen** |

**But `phonation/api.py` is not clean at its input, and an earlier version of this table implied it
was.** `phonation/api.py:63` calls `extract_pitch_values` and returns its two hardcoded pairs — so
**`derive_f0_range` *is* the 170 Hz bin**, and `f0_track` (`phonation/api.py:170-171`) and
`hnr_track` and `period_marks` all document their `f0_min_hz` as *"read it from `derive_f0_range`"*.
(`formant_track` does not — it takes `max_formants`, `formant_max_hz`, `window_s` and
`preemphasis_hz` and no F0 floor at all, `api.py:205-213`. An earlier version listed it here; the
argument survives without it.)

The module therefore **imports the bin wholesale**, along with everything that follows from it: the
1.67× window discontinuity in `hnr_track`, the sub-60 Hz pulse exclusion in `period_marks`, and the
ceiling truncation in `f0_track`. The contrast is true of the module's **body** and false of its
**input**.

**Two more defects at that entry point**, both in the module credited with typed absence:

- `extract_pitch_values` returns `{nan, nan}` from a **bare `except Exception`**, so
  `F0RangeUnavailable` fires on a parselmouth *crash* as readily as on genuinely unvoiced audio —
  **the identical crash/absence conflation this audit calls out for CPPS at finding 2.**
- The ±2 SD trim is computed **in linear Hz over the wide 50–600 search**, so octave errors
  contaminate the very mean that selects the bin.

**This is why step 2 replaces `derive_f0_range` rather than the wrapper**: it is the shared root, and
fixing only `praat_parselmouth.py` would leave the clean path carrying the same bin.

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
  4** — selection on the dependent variable. **No covariate recovers an estimate from that**, and with
  support counts unavailable (finding 5) a reader cannot even see how much was deleted. Note that
  4 dB sits *inside* the published normal/dysphonic decision region, not below it.

**Three further defects in the same function:**

- **CPPS is averaged unweighted across intervals**, so a 3 s interval and a 60 ms interval count
  equally — and finding 3's vuv inflation manufactures many short padded intervals on fragmented
  speech, so a dysarthric recording's mean is dominated by short, badly-estimated segments.
- **`std_dev_cpp` is a between-interval standard deviation**, not the within-recording CPPS SD that
  published work reports — carrying a name that will be read as the published quantity.
- **`voicing_threshold=0.3` against Praat's 0.45** is *lower*, labelling more marginal frames voiced.
  It **compounds** finding 3's inflation rather than being independent of it.

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

Not one function in `praat_parselmouth.py` returns a frame, cycle or interval count. Three
**compute one and discard it**: `:759` (`n_intervals = ... "Get number of rows"`), `:896`
(`n = ... "Get number of points"`) and `:1005` (`num_steps = spectrogram.nx`).

An earlier version of this finding also cited `:804` and `:1036-1039`; neither computes a count —
`:804` is `if cpp_list:`, a truthiness test, and `len(cpp_list)` is **never computed anywhere**, while
`:1036-1039` are four `np.mean` calls.

**And there is a fourth, which matters more than the three above.** `extract_speech_rate` computes
`numpeaks` (`:245`) and `number_syllables` (`:310`) and **returns only rates**. That syllable count
**is** the support for [`branch-speech.md`](branch-speech.md) S4's and [`branch-ddk.md`](branch-ddk.md)
D2's speaking and articulation rates — so it is what makes the mandatory support count specifically
unsatisfiable for **the two largest rate populations in the corpus**, roughly 25,000 recordings and
7,989.

The finding stands and is sharper: **no function returns a support count, and the one that would
matter most computes it and throws it away.**

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

**It tracks silent fraction, not dynamic range.** The meaningful quantity is `max_dB − min_dB`.

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

**Measured on three synthetic probes** — buzz **50.75 dB**, buzz with noise **16.90 dB**, pure sine
**105.60 dB**. Real speech does not approach 60 dB, so `min_dip` is **almost certainly 2 on
essentially every recording** and the documented "clean signal" setting of 4 is effectively dead.

**This has not been measured on the corpus**, and an earlier version of this finding said "always 2
on all 62,547 recordings" and then, four lines later, that it "remains a live data-dependent branch".
Both cannot hold. The accurate statement is the conditional one: the branch is live, its condition is
data-dependent, and on three probes it always took the same side.

**And there is a case where it takes the other side.** If Praat returns undefined for the mean,
`NaN < 60` evaluates `False` and `min_dip` stays at the **stricter 4** — on exactly the recordings
where pitch could not be measured. [`branch-ddk.md`](branch-ddk.md) D2 states this; the audit omitted
it.

It costs one whole-file harmonicity computation per recording, on the denoised `enhanced` stream.

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

The value may still be the right choice; the justification for it is not correct.

**That derivation also routes readers through a wrong line**: it cites `praat_parselmouth.py:569`
for `extract_harmonicity_descriptors`, which is at **`:580`**, its `periods_per_window=4.5` call at
`:621`. Two documents send readers through it.

**And it is not the only stale one.** `config-derivations.md:74` says *"both `spans.k_db` values"* —
**plural**, stale phrasing from when two `k_db` keys existed, which now contradicts *"one shared
value"* in the live derivation at `:106-120`. That is a second instance of the same class.

**So the derivations file carries factually wrong and stale entries as well as thin ones**, which
bounds how much weight it can take unchecked — including from
[`branch-listening-sample.md`](branch-listening-sample.md), whose whole function is "check what is
documented before calling something owed".

### 12. joblib caches a closure

`:1476-1478` — `time_step`, `window_length` and eleven toggles are free variables of `_extract_one`
rather than arguments, so two runs with different settings and one `cache_dir` collide.

**Dormant in triage** (no `cache_dir` is passed at `preprocess.py:883`) but a live hazard for any
batch over 62,547 recordings.

### 13. The 16 kHz resample is an undeclared analysis-band decision

`resample.target_hz: 16000`, and its config comment derives it from **model input requirements** —
*"YAMNet, HeAR, AST and the recognizers are all native here"* — not from any measurement requirement.

**It caps every acoustic measurement at 8 kHz**, for recordings captured at 44.1 or 48 kHz. That is a
peer of finding 7: an analysis band chosen for one reason and binding on measurements chosen for
another, nowhere declared as such.

It also means **every stream any of these measurements sees is 16 kHz mono** — which changes what the
sample-rate covariate in [`branch-voice.md`](branch-voice.md) V4 can be (see below).

## Also recorded

**`extract_speech_rate` deviates from Praat on two thresholds.** `min_pause` **0.3 s against Praat's
0.1 s** — 3×, so hesitation pauses at 0.3–0.4 s sit on the edge and `pause_rate` under-reads for
halting speech; and minimum sounding interval **0.1 s against 0.05 s**, 2× stricter, dropping short
voiced fragments.

**Its `to_pitch_ac` call at `:290` differs from Praat on six parameters** — floor 30 vs 75,
candidates 4 vs 15, voicing_threshold 0.25 vs 0.45, voiced_unvoiced_cost 0.25 vs 0.14, ceiling 450 vs
600 — and **the code says so itself**, in five comment admissions at `:294`, `:297`, `:300`, `:303` and `:304`:

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
| **0 — the enhanced stream** | **every Praat-derived measurement in the graph**: `branch-voice.md` V4 and V6, `branch-speech.md` S4, `branch-ddk.md` D2 and its routing gate, `branch-airway.md` A6 |
| 1, 8 | [`branch-voice.md`](branch-voice.md) V3, V4, V7 |
| 2, 3, 4, 5 | **every caller of `extract_cpp_descriptors`** — `branch-voice.md` V4 (5,113 recordings) *and* [`branch-speech.md`](branch-speech.md) S4 (~25,000, where finding 3 bites hardest) |
| 6 | `branch-voice.md` V6 |
| 7 | `branch-voice.md` V4; [`branch-quality.md`](branch-quality.md) Q2; **[`branch-airway.md`](branch-airway.md) A6**, which reads `extract_spectral_moments` for cough descriptors — and coughs are the most broadband events in the corpus |
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
