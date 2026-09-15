# The listening sample

Many items marked *owed* across the branch documents are owed to the same missing thing: **nothing in
this corpus has been heard.** This document states that once so the others cross-reference it.

## First, what is already documented

**Before writing that a value is unmeasured or undocumented, check
[`config-derivations.md`](config-derivations.md), [`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md)
and the config's own comments.** An earlier version of this document asserted that four shipped
values were "in force with no marking". All four have written derivations, and saying otherwise
misrepresented work that had been done.

Nor are the null keys undocumented: `config-derivations.md` carries an **"UNSET, and why"** section
(`:915` onward) giving each one a stated reason. A null in this config is a recorded decision, not an
oversight.

## Seven kinds of owed, and they are not equivalent

### Derived and in force — listening would *validate*, not supply

| key | value | derivation |
| --- | --- | --- |
| `spans.k_db` | `6.0` | `config-derivations.md:106-120` — **but derived for the pre-emphasised envelope**, and [`branch-voice.md`](branch-voice.md) V1 reads a linear one on `plain`, so it is owed a **re-derivation**, which listening cannot supply |
| `airway.labels_of_interest` | `[Cough, Breathe]` | `:555-557` — recorded as *"Vocabulary, not thresholds."* |
| `spans.min_duration_ms` | `50` | `:234` — *"conventional and not fitted"* |
| `voice.f0_search_range_hz` | `[50.0, 600.0]` | `:748-760`, with the five narrowing coefficients at `:598-730` — the derivation describes per-recording narrowing and, since 2026-09-14, so does the code. Until then it did not, and that mismatch is what this document exists to catch. See below. |
| `ddk.ppg_segment_rate_per_s` | `10 /s` | `family-taxonomy-ruleset.md:102` — *"recall-first, not J"* |

These carry reasoning. Listening would tell you whether the reasoning holds on real audio; it is not
needed to explain what the number is doing.

`airway.labels_of_interest` is the case to be careful with: its derivation states it is a vocabulary
decision rather than a threshold, which is exactly the no-refits exemption
[`branch-airway.md`](branch-airway.md) A1 claims. That exemption is correct.

**And `voice.f0_search_range_hz` was the case that bounded this whole document.** Its derivation
described a wide search bound narrowed per recording; until 2026-09-14 the code selected one of two
hardcoded pairs at a 170 Hz mean-pitch boundary, and an earlier version of this table repeated the
derivation as fact. **The code now matches the derivation** — `extract_pitch_values`
(`praat_parselmouth.py:465-497`) narrows off percentiles of the wide pass, with the five
coefficients as `praat_features.pitch_*` keys, each with its own derivation.

**The lesson outlives the instance, which is why this entry stays.** A derivation is evidence that a
decision was recorded, not evidence that the code does what it says — see
[`praat-instrument-audit.md`](praat-instrument-audit.md) findings 1 and 11, and the stale plural at
`config-derivations.md:74`. The way that was caught was reading the implementation, not the
derivation, and nothing about the fix makes the next such case detectable any earlier.

### Marked unmeasured and in force — exactly one

`ddk.lexical_repetition: 3`, marked in **both** the config (`default.yaml:260`) and the derivations
table (`family-taxonomy-ruleset.md:101`, *"not measured"*). It routes DDK today.

This is the only shipped value in the triage config whose own documentation says no measurement
stands behind it.

### Null by documented decision — listening could supply these

`airway.contest_labels` (`config-derivations.md:566`), `speech.target_match_cosine` (`:921`),
`speech.speech_test_stoi_floor` and `speech.speech_test_si_sdr_floor` (`:923`),
`speech.second_diarizer` (`:919`), `phonation.hnr_floor_interval_db` and
`phonation.rms_floor_interval` (`:927`), `speech.nontarget.*` (`:675`),
`voice.f0_range_by_population` (`:650`), `voice.task_duration_ranges` (`:657`),
`redaction.padding_ms` (`:916`).

Each has a stated reason for being null, and the capability behind it does not run. The failure is
visible by design.

Two are worth separating from the rest. `phonation.hnr_floor_interval_db` and
`phonation.rms_floor_interval` **cannot be resolved by derivation at all** — the measurement that
exists is in normalised-autocorrelation units that do not transfer to the Praat-dB implementation,
and Praat self-calibrates neither. And `voice.f0_range_by_population` is not a gap to close:
[`branch-voice.md`](branch-voice.md) V7 argues the derived per-recording range is the better path and
a population prior would clip the voices most likely to be studied.

### Not reachable at all — owed a code change, not a listening sample

**[`praat-instrument-audit.md`](praat-instrument-audit.md) adds a fifth kind of owed**: parameters
that are neither configurable nor reachable from any caller, whose values deviate from Praat's own
documented guidance, and whose effect has now been measured. The CPPS `> 4` cut, the vuv mean
period, the 330 Hz peak-search cap, `range_db_ratio`, the 5 kHz moments band, the formant parameters
the wrapper does not forward — and, until 2026-09-14, the 170 Hz sex bin.

**These are not owed a listening sample. They are owed a code change**, and no amount of annotation
would validate them. **The bin is the worked example of that**: it was closed by a code change, and
its five replacement coefficients moved into `data/` with derivations, which is the shape the rest of
this list wants.

**The support-count gap belongs here too, and it has a specific worst case.**
`extract_speech_rate` computes `numpeaks` (`praat_parselmouth.py:257`) and `number_syllables`
(`:330`) and **returns only rates** — so the support count for S4's and D2's speaking and
articulation rates, over roughly 25,000 and 7,989 recordings, is computed and discarded. That is what
makes [`branch-conventions.md`](branch-conventions.md)'s mandatory support count specifically
unsatisfiable for the two largest rate populations in the corpus.

The audit also found that **`config-derivations.md:575-580` is factually wrong** about
`phonation.periods_per_window: 4.5` — it cites "Praat's own documented defaults for the cc method"
where Parselmouth and the Praat form both say 1.0. The value may still be right; the justification is
not. **That is the first derivation found to be incorrect rather than merely thin**, and it bounds how
much weight this document's "check the derivations first" rule can carry unchecked.

And **the mandatory support count is still unsatisfiable for every Praat scalar a branch reads** —
one of thirteen functions exposes one (`extract_pitch_values`' `pitch_frames`, added 2026-09-14, and
no branch reads it yet), four of the rest compute a count and discard it, while `phonation/api.py`
exposes support in five of five.

### Not in any config — the genuinely undocumented literals

**This is the sharpest category, and it is small.** `praat_parselmouth.py`'s syllable-nuclei
operating points: `silence_db = -25` (`:154`), `min_dip = 4` (`:161`) dropped to `2` when the
recording's own mean HNR is below 60 (`:166-167`), `min_pause = 0.3` (`:171`).

They are literals inside a helper, in no config and in no derivations file, and they reach every rate
measure [`branch-speech.md`](branch-speech.md) S4 and [`branch-ddk.md`](branch-ddk.md) D2 would
produce. The HNR switch conditions detection sensitivity on a voice-quality measurement of the
recording being measured.

Alongside these sit the operating points for capabilities that do not exist yet and so have no config
key: AIRWAY A5's breath-event detection parameters, DDK D1's modulation search band and gap
criterion, SPEECH S3's omission score cut, VOICE V1's envelope threshold and minimum attempt
duration. Each is named in its own document.

### Owed a bench measurement — two members

Not a listening sample, not a code change, not a config literal: answered by **synthesising signals
of known value and measuring what comes back.**

- **The jitter floor at 16 kHz.** With uniform pulse-placement error Δ = 62.5 µs the induced
  local-jitter floor is ≈ 0.56 Δ/T (0.564, from ε ~ U(±Δ/2)) — about **0.42% at F0 120 Hz and 0.88% at 250 Hz**, inside the
  0.2–1% normal range. Whether Praat's sub-sample interpolation recovers it is unmeasured, and
  [`branch-voice.md`](branch-voice.md) V4 withholds jitter until it is.
- **Shimmer's own floor**, which the timing argument does not give. Shimmer is an amplitude measure;
  its floor comes through uninterpolated peak-amplitude picking at roughly four samples per cycle of
  a 4 kHz component. It needs its own synthesised bound.

**Three things the bench must do, or its answer is worthless:**

- **Synthesise non-integer, dithered periods and sweep F0.** The 0.564 Δ/T floor assumes
  *independent* placement error. A signal whose period is an integer number of samples produces
  deterministic, correlated error and would measure a **near-zero floor** — the bench would come
  back **falsely clean**. This is the single most important line in the spec.
- **State shimmer's mechanism as an assumption.** "Four samples per cycle of a 4 kHz component" is a
  worst case, not the mechanism: per-period peak amplitude is dominated by F1-region energy at
  20–30 samples per cycle at 16 kHz, with the 4 kHz content setting curvature near the peak. Written
  as a stated assumption, the measured bound is interpretable; written as the mechanism, it is not.
- **Synthesise known shimmer as well as known jitter.** The two floors arise differently and neither
  bounds the other.

Cheap, decisive, and nobody has done either.
[`praat-instrument-audit.md`](praat-instrument-audit.md) states the same kind at its tail.

### Owed a purpose-collected study — one member, and it is future research

**The impact of speech enhancement on disordered voices.** Added 2026-09-14, when the owner withdrew
[`praat-instrument-audit.md`](praat-instrument-audit.md)'s **step 1** — the proposal to move the
Praat scalars off the FRCRN-`enhanced` stream onto `plain`.

**What is known.** `praat_features` measures all forty-five scalars on `enhanced`
(`../../src/senselab/audio/workflows/triage/nodes/preprocess.py:884`, recorded as `signal="enhanced"`
at `:905`). The argument that was raised against that — *FRCRN removes the aperiodic energy that is
the measurement* — is false: sibilants are broadband aperiodic energy and FRCRN preserves them, and
the narrower fallback, that the low-level noise component inside voiced phonation is stripped even
though sibilants survive, is false too — FRCRN preserves vocal texture very well. So the mechanism
that would have made `enhanced` the wrong stream does not exist.

**What is not known, and this is the item.** Whether `enhanced` or `plain` is the better stream for
these scalars **on disordered voices** has never been measured. The prior now runs the other way:
background noise depresses HNR and CPPS and perturbs period detection, so on `plain` a healthy voice
can read as dysphonic, and many recordings in this corpus carry background noise. That makes
`enhanced` the defensible default — it does **not** make it measured.

**Why it is a study and not a pass.** Settling it needs **selected voices with disorders, with hand
labels**, measured both ways. Nothing cheaper reaches it:

- **A corpus pass cannot answer it.** This corpus's only labels are declared task names, so a
  paired `enhanced`-vs-`plain` re-derivation over the corpus yields two distributions with nothing to
  say which is *closer to the voice*. A difference is not a direction. (This said "all 62,547 stores"
  and would reach fewer: the scalars exist in 60,202, and 2,376 stores carry none —
  [`../20260911-ppg-praat-batch/design.md`](../20260911-ppg-praat-batch/design.md).)
- **A quick paired measurement cannot answer it either.** Showing that CPPS shifts by *n* dB between
  the two streams measures the enhancer, not the instrument's validity. The question is which reading
  agrees with a perceptual judgement of the disorder, which requires the judgement to exist.
- **A bench measurement cannot answer it.** Synthesised signals carry no pathology, so the sixth kind
  does not reach here.

**Scope, so nobody plans it as part of something else.** This is a **future research direction**, not
a blocker: no repair in the audit waits on it, and the withdrawal of step 1 removed the only thing
that did. The goal of the triage work is much simpler than answering it.

## Why the corpus cannot supply any of it

A declared family is what the protocol *asked for*, not what the participant *did*. Fitting against
it produces a detector for the declaration. That is the no-refits rule in
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).

## What would supply it: three samples, not one

### Sample A — event and content annotation

*Is there a cough here? Was the sentence read as printed? Is there a second voice? Where does the
phonation start and stop?*

Any trained annotator, high throughput, good agreement — so a second rater on a subset establishes
reliability. This serves most of the owed items: the span gate's validation, the airway label
vocabulary, the contest definition, the breath-event parameters, the DDK gates, the omission cut.

### Sample B — perceptual voice rating

*How does this voice sound?*

CAPE-V or GRBAS, rated by speech-language pathologists, **multiple raters**, with repeated items for
intra-rater consistency. Perceptual voice ratings have notoriously moderate inter-rater agreement, so
a single rater produces a number with no known reliability.

### Sample C — signal typing

**Titze type 1 / 2 / 3 is in neither sample above, and it is the ground truth
[`branch-voice.md`](branch-voice.md) V4's whole qualification scheme rests on.** Sample A is
event and content annotation; Sample B is CAPE-V or GRBAS severity. Neither produces a signal type,
which is a **visual** judgement from narrowband spectrograms — a third annotation kind, with a third
kind of annotator.

**And there is a constructive substitute available today, better than what V4 currently proposes.**
Run `f0_track` **twice with floors an octave apart and compare the returned `strength` arrays**: a
period-doubled voice locks with higher strength at the lower floor. That detects type 2 directly,
where V4's period-length modality test provably fails in the case it exists to catch (the tracker
locks to the subharmonic and the distribution is unimodal at 2T).

**It is not parameter-free, and owes its cut.** For a periodic signal the autocorrelation peak at 2T
is nearly equal to the one at T, so the discriminator is "strength at the lower floor exceeds the
higher **by some margin**" — and the margin is an operating point.

**And it is confounded, because Praat's analysis window is set *by* the floor.** Halving the floor
doubles the window — the same coupling [`praat-instrument-audit.md`](praat-instrument-audit.md)
finding 1 measures as the 1.67× step. Over a longer window a non-stationary voice yields a lower
normalised autocorrelation **for any signal, doubled or not**, so the two `strength` arrays differ by
window length before any subharmonic exists. The bias is conservative, but it **scales with how
steady the voice is** — which is the thing being measured.

**It cannot be controlled away.** `f0_track` (`phonation/api.py:185-191`) exposes only `f0_min_hz`,
`f0_max_hz` and `hop_s`, and `to_pitch_cc` derives the window from the floor. The clean form —
comparing autocorrelation functions directly at a **fixed** window — needs **a pitch-tracker window
control independent of the floor, which is not in the inventory.**

It remains the best available proxy for the absent subharmonic-to-harmonic ratio. It owes the margin,
and it carries the confound.

**Merging A, B and C gets the staffing and the power wrong for all three.**

### Two constraints on Sample B

**Its unit is the session, not the recording.** CAPE-V requires sustained vowels, sentences *and*
running speech, which in this corpus are separate recordings of one sitting — the same grouping
[`corpus-level-node.md`](corpus-level-node.md) needs for C2 and C3.

**Raters must hear the stream the measurement is computed on** — 16 kHz mono `plain` — not the
original file. A rating made on 48 kHz audio does not validate a measurement made on a resampled,
band-limited one.

## Sizing: the positives set the power

"A few hundred, stratified across device × task × quality" gives single-digit cells against roughly
thirteen operating points, several of which detect **rare events** — a cough in a non-airway
recording, a second speaker, a contradicted clip. A sample balanced on *recordings* is sparse on
*positives*.

**Size by the rarest positive class each operating point must characterise.**

## Stratify along the decision variable

Sampling uniformly spends most of the budget where the answer is obvious. **Sample densely near each
candidate operating point** — recordings whose measured value sits close to the threshold being
validated — and sparsely far from it. That is how a small budget pins an operating characteristic,
and it means the sample is drawn **per operating point**, with recordings shared between draws.

**Record the inclusion weight of every sampled recording.** Enriched sampling makes the sample
unrepresentative of the corpus by design — that is the point — so a sensitivity measured on it cannot
be converted into a corpus-level error rate without the probability each recording had of being
drawn. Without the weights the sample validates a threshold and says nothing about how often it
fires.

## What it does not supply

**Norms.** Jitter, shimmer, CPP, HNR, DDK rate and maximum phonation time have published normative
ranges a few hundred annotated recordings cannot reproduce — and no published perturbation norm was
collected on AGC'd, band-limited, noise-suppressed phone audio, so the published ranges do not
transfer cleanly here either. See [`branch-conventions.md`](branch-conventions.md).

Nor discourse-content scoring keys for story recall and picture description.

## Status

**Not commissioned.** Recorded here as the prerequisite behind the owed items, so a reader
encountering "owed" in any branch document finds one explanation — and finds, first, the check that
the item is genuinely owed rather than already derived.

**The seventh kind is not one of the three samples and must not be folded into them.** Samples A, B
and C annotate *this* corpus. The enhancement-on-disordered-voices question needs selected disordered
voices that are not in it, so commissioning A–C would not advance it by a step.
