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

## Four kinds of owed, and they are not equivalent

### Derived and in force — listening would *validate*, not supply

| key | value | derivation |
| --- | --- | --- |
| `spans.k_db` | `6.0` | `config-derivations.md:106-120` — *"provisional and expected to be refit… a deliberately permissive placeholder"* |
| `airway.labels_of_interest` | `[Cough, Breathe]` | `:555-557` — recorded as *"Vocabulary, not thresholds."* |
| `spans.min_duration_ms` | `50` | `:234` — *"conventional and not fitted"* |
| `voice.f0_search_range_hz` | `[50.0, 600.0]` | `:641-648` — **but the derivation is wrong**: it describes per-recording narrowing, and `derive_f0_range` is a binary sex bin. See below. |
| `ddk.ppg_segment_rate_per_s` | `10 /s` | `family-taxonomy-ruleset.md:102` — *"recall-first, not J"* |

These carry reasoning. Listening would tell you whether the reasoning holds on real audio; it is not
needed to explain what the number is doing.

`airway.labels_of_interest` is the case to be careful with: its derivation states it is a vocabulary
decision rather than a threshold, which is exactly the no-refits exemption
[`branch-airway.md`](branch-airway.md) A1 claims. That exemption is correct.

**And `voice.f0_search_range_hz` is the case that bounds this whole document.** Its derivation
describes a wide search bound narrowed per recording; the code
(`praat_parselmouth.py:429-436`) selects one of two hardcoded pairs at a 170 Hz mean-pitch boundary.
An earlier version of this table repeated the derivation as fact. **A derivation is evidence that a
decision was recorded, not evidence that it is correct** — see
[`praat-instrument-audit.md`](praat-instrument-audit.md) findings 1 and 11, and the stale plural at
`config-derivations.md:74`.

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
documented guidance, and whose effect has now been measured. The 170 Hz sex bin, the CPPS `> 4` cut,
the vuv mean period, the 330 Hz peak-search cap, `range_db_ratio`, the 5 kHz moments band, the
formant parameters the wrapper does not forward.

**These are not owed a listening sample. They are owed a code change**, and no amount of annotation
would validate them.

The audit also found that **`config-derivations.md:571-578` is factually wrong** about
`phonation.periods_per_window: 4.5` — it cites "Praat's own documented defaults for the cc method"
where Parselmouth and the Praat form both say 1.0. The value may still be right; the justification is
not. **That is the first derivation found to be incorrect rather than merely thin**, and it bounds how
much weight this document's "check the derivations first" rule can carry unchecked.

And **the mandatory support count is currently unsatisfiable on the Praat path** — zero of thirteen
functions expose one, four of them computing a count and discarding it, while `phonation/api.py`
exposes support in five of five.

### Not in any config — the genuinely undocumented literals

**This is the sharpest category, and it is small.** `praat_parselmouth.py`'s syllable-nuclei
operating points: `silence_db = -25` (`:142`), `min_dip = 4` (`:149`) dropped to `2` when the
recording's own mean HNR is below 60 (`:154-155`), `min_pause = 0.3` (`:159`).

They are literals inside a helper, in no config and in no derivations file, and they reach every rate
measure [`branch-speech.md`](branch-speech.md) S4 and [`branch-ddk.md`](branch-ddk.md) D2 would
produce. The HNR switch conditions detection sensitivity on a voice-quality measurement of the
recording being measured.

Alongside these sit the operating points for capabilities that do not exist yet and so have no config
key: AIRWAY A5's breath-event detection parameters, DDK D1's modulation search band and gap
criterion, SPEECH S3's omission score cut, VOICE V1's envelope threshold and minimum attempt
duration. Each is named in its own document.

## Why the corpus cannot supply any of it

A declared family is what the protocol *asked for*, not what the participant *did*. Fitting against
it produces a detector for the declaration. That is the no-refits rule in
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).

## What would supply it: two samples, not one

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

**Merging A and B gets the staffing and the power wrong for both.**

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
