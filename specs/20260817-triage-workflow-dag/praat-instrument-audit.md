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
`Raises:` clause both name the enhanced stream.

**And the invalidation lever is not `CACHE_SCHEMA_VERSION`.** That constant lives in
`utils/tasks/cached_inference.py` and belongs to audio_analysis; grepping `cached_inference`,
`cached_call` and `cache_dir` under `workflows/triage/` returns **zero hits**, so bumping it
invalidates nothing in these stores. An earlier version of this step named it.

**The mechanism is the extend driver step 0 already names — specifically
`scripts/extend_ppg_praat.py`, which wrote both the PPG and the Praat blocks.** And it **skips any
recording whose store already holds both measurements** (`extend_ppg_praat.py:26`), so **re-running
it after a stream switch changes nothing on all 62,547 stores.** It needs either a withdrawal pass
that retires the existing measurements first, or a force flag. Cheap to fix here; expensive to
discover mid-implementation.

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
same way the contract's rule (a) requires a gate count
(`../20260913-branch-contract-and-hints/design.md:121`, with the requirement at `:178` and `:551`).

[`branch-ddk.md`](branch-ddk.md) D1 already specifies reading `plain`; the ordered path did not.

### Step 2 — replace `derive_f0_range`, not the wrapper

Highest leverage in the codebase: it is the shared root of the sex bin (finding 1) and the pulse
exclusions (finding 8), it contaminates **both** modules (see the corrected contrast below), and it is
roughly fifteen lines.

Narrow per recording from the wide search — robust percentiles of the wide-search contour, with
declared margins. The percentiles are taken **on linear Hz**: an earlier draft of this line said
log-Hz, but the ratio margins (`p5 / 1.5`, `2.5 × q3`, `1.5 × p95`) are what make the rule
scale-free, and taking a log before a percentile changes nothing, since percentiles commute with
any monotone transform. This is **Hirst's two-pass method**, which is the precedent and should be
named as such. Keep the typed absence. And **distinguish the crash return from genuine
absence**, which it currently conflates (below).

**Landed 2026-09-14.** `extract_pitch_values` narrows per recording and reports a failed analysis as
data. What follows is the record of the rule, of the citations that may and may not be attached to
it, and of what it does not reach. Every coefficient's own derivation is in
[`config-derivations.md`](config-derivations.md) under `praat_features`, because a coefficient that
decides a measurement's range is a parameter of the run and belongs in `data/` rather than as a
module literal.

**The rule is asymmetric and has three parts.** Floor `max(search_floor, p5 / 1.5)`; ceiling
`min(search_ceiling, max(2.5 × q3, 1.5 × p95))`; and if p95 falls below twice the search floor the
whole contour sits within an octave of the bottom of the search range, so the unnarrowed search
range is used instead. Measured over fourteen adversarial cases the full rule places the true F0
inside its own derived range on all thirteen that track at all (the fourteenth, a 45 Hz fry, is an
absence at a 50 Hz floor); `2.5 × q3` with the p5 floor misses the register break and `1.5 × p95`
misses the emphatic peak. The probe that regenerates those rows is
`src/tests/audio/tasks/f0_range_probe.py` — committed because neither table here states the
contamination levels its hum cases were synthesised at, so the cells could not otherwise be
reproduced from this document.

**A one-pass narrowing without the fallback would have been a regression on a common clinical
recording.** On a synthesised 120 Hz voice with a 60 Hz hum at −22 dB the wide search locks to the
subharmonic across the whole contour and a one-pass narrowing returns `[50.0, 90.0]`, a range the
speaker's F0 never enters. The retired bin was accidentally robust there — a 60 Hz trimmed mean
selected `(60, 250)`, which still contains 120 Hz — and the problem gets *more* frequent under
step 1, because FRCRN suppresses stationary low-frequency noise and `plain` does not.

**The fallback captures a larger population than the hum case it was designed for, and that must be
stated rather than discovered.** A clean 90 Hz buzz has p95 = 90, under 2 × 50, so an ordinary low
male voice takes the wide range. It is safe — a wide range never excludes the voice — but those
recordings lose the octave-error robustness narrowing buys, and which voices fall back is
floor-dependent by construction: at the shipped 50 Hz floor it is every voice whose p95 is below
100 Hz. Raising the floor to 60 also makes the 220 Hz-plus-120 Hz-hum case fall back, to
`[60, 600]`. Both are among the things a raised search floor would change, and neither is decided
here.

**A live misattribution in the document that justified this approach.**
`specs/20260911-ppg-praat-batch/design.md:322` reads, verbatim: *"That is the pitch-range
standardization method it cites (doi:10.3758/BRM.41.2.318). No fixed corpus-wide range is needed,
because the range is derivable per recording."* That DOI is **Vogel, Maruff, Snyder & Mundt (2009),
"Standardization of pitch range settings in voice acoustic analysis", *Behavior Research Methods*
41(2):318–324**, and **Vogel supports none of it**: he recommends sex-specific *fixed* settings
(male 70/250, female 100/250–300) and explicitly rejects the per-recording approach — *"managing
speaker specific analysis settings individuality requires extensive expertise and time and is
impractical for large volumes of data."* So the retired bin's *kind* of rule is what its cited
source recommends; what it misattributed are the **values** — 60 Hz appears nowhere in Vogel, and
100–500 is one of the candidates he tested and found significantly worse than gold standard, with
`d = 2.14` as a value that *saturates* across most of his wide ranges rather than as a per-condition
effect size his tables support. The approach is still right on the merits — Vogel's own caveat that
*"caution should be exercised when applying suggested settings to pathological voice populations"*,
on 20 speakers over an office-telephone channel, is an argument *for* per-recording derivation — it
is just not Vogel's argument, and his authority must not be borrowed for it. `:320-321` of the same
document describes the z-trim and the two bins as live behaviour, which this step deletes; rewrite
`:320-322` together. The DOI is gone from `extract_pitch_values`' docstring, where it annotated code
that now contradicts it.

**Cite Hirst 2011 for two things and nothing else: the two-pass structure, and the ceiling's
quartile coefficient.** His rule is a first pass at 50–700 Hz, then `floor = 0.75 × q1` and
`ceiling = <coefficient> × q3` — quartiles, with a deliberate asymmetry that is itself an empirical finding
worth quoting: *"if the Pitch Floor is too low then we are likely to get octave errors […] Setting
the Pitch Ceiling too high does not, however, seem to lead to any systematic errors."* That is why
the ceiling coefficient is adopted and the floor is not. **Which coefficient, and why 2.5 rather
than 1.5, is not settled here** — Hirst offers both values in the same lineage and splits them by
material; the full argument and senselab's choice of 2.5 are in
[`config-derivations.md`](config-derivations.md) under
`praat_features.pitch_ceiling_quartile_multiplier`, and citing "Hirst's 2.5" as an unconditional
constant is the specific error that reading has to avoid. His floor is not adopted: `0.75 × q1`
misses a 100→400 Hz glide's low end at 107.2, and `p5 / 1.5` is −7.02 semitones off p5 and lands below
`0.75 × q1` on every source measured, which is the right direction for a corpus enriched for
pathological voices. **The `p5 / 1.5` floor and the pinned-contour fallback are senselab's own; cite
nobody for them.** Checked across De Looze & Hirst 2008, De Looze & Rauzy 2009, De Looze & Hirst
2010, De Looze 2010 (thesis), De Looze & Hirst 2014/2014b, Hirst 2007, Hirst 2011, Hirst & De Looze
2021, and four shipped implementations including Hirst's own Momel-INTSINT plugin: no rule of the
form "compare the second-pass median against the first and widen back" exists in any of them. The
authors' own term is **"two-pass"**, not "iterative", and none of the four implementations has a
loop in the estimation path; octave errors appear in their papers only as *motivation* for
narrowing, never as a test applied to the result. The only "revert to wide" in the lineage is a
degenerate-input guard — Praat Vocal Toolkit's `minmaxf0.praat` reverts to 40/600 `if voicedframes =
0`, which `_no_pitch_range()` already does on the same trigger.

**Record the search-floor question rather than deciding it silently.** senselab uses 50 Hz.
Published first passes: Hirst 2007 → 75; De Looze & Hirst 2008 and Hirst's shipped code → 60; De
Looze 2010 (thesis) → 60, not stated in DL&H 2010, so cite the thesis; Hirst 2011 → 50; Hirst & De
Looze 2021 → "e.g. 60"; Prosogram → 65 (verified in `prosomain.praat` 3.05, 2024; the two-pass is
that version's, not something the 2004 paper describes), which excludes both mains fundamentals
incidentally, with a second pass of median −12/+18 semitones — a 30-semitone window wide enough to
survive a one-octave-low median, i.e. structural tolerance rather than correction. Raising the floor
would trade away the 45–60 Hz creak cases this document bounds by the declared search range, and per
the residual below it is only partial anyway, since the second harmonic survives any floor.

**One consequence for another key.** The range ratio widens: the bin always gave 4.17 or 5.0, and
the measured ranges here reach 12 (`[50, 600]`).
`voice.f0_range_ratio_max` is null so nothing fires, but `voice.py:77-80` **raises** rather than
flags once it is set.

**Residual capture is owed, and it is the state of the art rather than a gap in this work.** The
fallback catches capture landing within an octave of the search floor. It does not catch
second-harmonic capture against a higher voice: measured, a 220 Hz voice under a 120 Hz-dominant hum
gives p95 ≈ 110 against 2 × 50 = 100, so the fallback does not fire — Hirst's ceiling coefficient
rescues that case here, but a deeper version would be caught by neither part. What no published
method does is **test whether its own first-pass distribution was octave-halved and act on the
answer**; that is the citable claim, and not the wider "no published estimator corrects this", since
Mertens' *Polytonia* (2014) §5.5 is a published range estimator with designed octave handling
(discarding syllables ≥ 18 ST from the median) and Liberman's 2018 Language Log treatment
mode-anchors and re-tracks — a described procedure with no published code. Mark the residual owed a
measurement on real recordings.

**Why `pitch_failed` exists, and why step 2 carries it rather than `derive_f0_range`.** The
`except Exception` returned a NaN pair byte-identical to the legitimate no-pitch return, so a caller
could not tell a parselmouth crash from a silent recording. The distinction can only be made inside
`extract_pitch_values`, because that frame is the only one that sees the exception — a `try/except`
in `derive_f0_range` would be unreachable in production. The swallow stays, because other callers
depend on the batch extractor not aborting; the signal is added as data, on all three return paths.

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

Log-power spectrum → cepstrum → robust regression over the trend range → peak prominence, frame-wise,
returning the frame count — **and then the two smoothing steps, which are the *S* in CPPS.**

**An earlier version of this step gave four operations and would have produced CPP, not CPPS.** The
parameters it omitted are the ones that set the value:

| parameter | value | why it is not optional |
| --- | --- | --- |
| time-averaging window | **0.01 s** — the incumbent's literal (`:776`), **not** Praat's form default of 0.02 | the first smoothing; without it the measure is CPP |
| quefrency-averaging window | **0.001 s** — the incumbent's literal (`:777`), **not** Praat's 0.0005 | the second smoothing; likewise |
| cepstrogram band | **5000 Hz** — the incumbent (`:767`), and this *is* finding 7's defect | declared rather than left open; changing it is a separate decision |
| subtract tilt before smoothing | **`"no"`** (`:775`) | value-setting and previously unmentioned |
| tilt line type | **`"Straight"`** (`:784`) — Praat's CPPS convention is exponential decay | value-setting and previously unmentioned |
| tolerance | **0.05** (`:780`) | value-setting and previously unmentioned |
| trend-line fit range | **distinct from the peak search** | Praat fits from 1 ms to the end of the quefrency axis |
| peak search band | **60–700 Hz** | see below |

**Two of these are inherited, not derived, and the table previously presented them as though they
were.** The 0.01 and 0.001 windows are this wrapper's own literals and depart from both Praat's form
defaults and Hillenbrand's. **Carrying them forward is defensible continuity** — it keeps the new
implementation comparable to the old — but it must be stated, because most published CPPS was
collected under different ones. That is the trap finding 11 names: a derivation is evidence a
decision was recorded, not that it was correct.

**The trend range and the peak-search band are different things**, and an earlier version conflated
them: "declare 60–500 Hz" read as the regression range. Fitting the trend over 2–16.7 ms instead of
Praat's 1 ms-to-end **changes every value**.

**And 500 Hz is too low for the peak search.** Untrained falsetto routinely exceeds 700 Hz, so an
upward glide's endpoint sits above it — the same truncation finding 4 identifies at 330 Hz, moved
rather than removed. **Declare 60–700 Hz.**

**Which raises a tension nothing in this set reconciles: `voice.f0_search_range_hz` is `[50, 600]`.**
That is the derived wide search every pitch-based measure inherits. **If falsetto above 700 Hz is
real enough to move the CPPS band, then finding 4's truncation logic applies at 600 Hz too** — and
step 2, which replaces `derive_f0_range`, narrows *within* that ceiling and cannot exceed it. So
either the F0 search ceiling is owed the same widening, or the two bands differ for a stated reason.
**Leaving it silent invites an implementer to pick one arbitrarily.** Recorded as owed. One fact the
landed narrowing adds to this entry: `2.5 × q3` reaches 600 for any q3 ≥ 240 Hz, so above roughly
that F0 the ceiling every recording gets is the search bound itself and not a narrowing.

**And a CPPS at F0 700 is not comparable to one at F0 120.** At 700 Hz the peak quefrency is 1.43 ms,
only ~0.4 ms above the trend-fit origin at 1 ms, where source and filter quefrencies are not
separable. Widening is still right — falsetto truncation is worse — but by this set's own
per-measure-band logic the non-comparability belongs in the name or beside the value.

This is a replacement for a *suppressed primary descriptor*, in a document set whose own rule is that
a measurement with no stated window is comparable to nothing. It cannot ship under-determined in
exactly the parameters that determine it.

This removes the interval gating, so it works on the aperiodic voices that motivated promoting CPPS
in the first place; has **no value cut** (finding 2); is **duration-weighted** rather than unweighted;
and carries a support count (finding 5).

**The quefrency band is a fixed wide one, declared — not derived per recording.** An earlier version
of this step said "from the recording's own derived F0", which defeats the step's own purpose three
ways:

- on a type-3 voice `derive_f0_range` **raises** (`phonation/api.py:60-70`), so a band derived from it
  is **unavailable on exactly the population the reimplementation exists to serve**;
- CPPS is a peak prominence measured against a regression over a quefrency range, so **changing the
  range changes the value** — a per-recording band destroys cross-recording comparability, which
  is what [`branch-conventions.md`](branch-conventions.md) requires of a per-measure band — fixed
  *across recordings*, whatever it is;
- it is not what the method does: the published convention uses a fixed wide search.

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
because of the `> 4` cut, which step 4 removes.

**But "only jitter and shimmer" is too narrow — coverage is per instrument, not per scalar.**
`extract_slope_tilt` builds a **pitch-corrected LTAS** with the same `0.0001 / 0.02 / 1.3` admission
and the same binned range (`praat_parselmouth.py:678`), so the zero-pulse failure takes **slope and
tilt down with jitter and shimmer**. And `hnr_db_mean` uses `Get mean`, which excludes undefined
frames — so it is conditioned on the **pitch tracker** finding pitch, a different failure from the
point process finding pulses.

**Report two coverage figures**: one for the **point process** (jitter, shimmer, slope, tilt) and one
for the **pitch tracker** (HNR, and anything reading the F0 contour).

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
argument survives without it. Note **`derive_f0_range`'s own `Returns:` at `api.py:55` names
`formant_track` as a recipient of the derived range** — that docstring is what misled two revisions
of this document, and it is wrong about its own consumer.)

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

### 9. The `hnr < 60` branch takes one side on every probe, and the other where pitch fails

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
| 10 | `branch-voice.md` V1 and V4 — unmasked −200 dB HNR sentinels in `voice_tracks.npz` |
| 5, 11 | [`branch-listening-sample.md`](branch-listening-sample.md) |
| 12 | nobody today (dormant: triage passes no `cache_dir`); any future batch over the corpus |
| 13 | [`branch-conventions.md`](branch-conventions.md)'s per-measure bands, and `branch-voice.md` V4's sample-rate item |

## A process finding: a revision deleted seven rules and no check caught it

Worth recording here because it is about the documents rather than the instruments.

A revision replacing one section of [`branch-conventions.md`](branch-conventions.md) **silently
deleted 86 lines and seven rules** — the octave-jump interpretation, deviation-is-not-a-bad-recording,
the whole quality-covariates section with its support counts, capture chain and distance items, the
analysis-window conventions, and the proposing-branches precondition. **Nine passages across five
documents still cited them.** This document's own finding 0 had its second supporting argument
quoting verbatim a sentence that no longer existed, and [`branch-quality.md`](branch-quality.md) Q8's
entire premise was a requirement that had been removed.

**Six rounds of citation checking could not have caught it.** Every check verifies that a cited
`file:line` exists and is in bounds; **dropped prose has no citations to fail**, and a cross-reference
to a *document* stays valid when the *rule* inside it disappears. The failure mode is invisible to
the tooling that had been catching everything else.

What would catch it: comparing section inventories across revisions, or treating a large net deletion
in a document others cite as something to justify rather than to review line by line.

**A second process rule, from a failure that recurred four times across three rounds.** A retraction
is not complete when the new text is written. It has to:

1. **delete the retracted text, not merely quote it as superseded** — the sharpest instance had a
   retracted sentence surviving verbatim thirty lines below its own retraction, inside the same
   numbered item, so an implementer reading top to bottom ended on the retracted conclusion, and one
   grepping for the parameter found it either way;
2. **propagate to every consumer in other documents** — the CPPS suppression landed in V4 and not in
   S4, whose population is five times larger;
3. **propagate to every summary of the changed section in the same document** — status tables, emit
   blocks, descriptor rows. Twice the prose was rewritten and the table above it was not, **and the
   table is what an implementer reads first.**

All three failed at least once in a set that had already added a note about retraction propagation.

## What this changes about "owed"

[`branch-listening-sample.md`](branch-listening-sample.md) divides owed items into values with a
derivation, values marked unmeasured, null keys, and literals in no config. **This audit adds a
fifth kind**: parameters that are neither configurable nor reachable, whose values deviate from the
instrument's own documented guidance, and whose effect has now been measured.

Those are not owed a listening sample. They are owed a **code change**, and finding 11 shows the
derivations file is not by itself sufficient evidence that one has been thought through.

**And there is a sixth kind, with two members: owed a bench measurement.** Neither is owed a
listening sample, a code change or a config literal; both are answered by **synthesising signals of
known value and measuring what comes back** — the **jitter floor** at 16 kHz (≈ 0.56 Δ/T, about
0.42% at F0 120 Hz and 0.88% at 250 Hz, inside the normal range) and **shimmer's own floor**, which
arises through a different mechanism and is not bounded by the jitter result.
[`branch-listening-sample.md`](branch-listening-sample.md) carries the full spec, including that the
synthesis must use **non-integer, dithered periods** — an integer-period signal produces correlated
error and would measure a falsely clean floor.
