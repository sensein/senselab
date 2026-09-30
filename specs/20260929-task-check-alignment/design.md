# Task checks aligned with what each task asks

Owner, 2026-09-29, after the audit in `/orcd/scratch/bcs/002/satra/tmp_task_audit/`
(`family_gate_audit.csv`, `undetermined_paths.csv`, `harvard_sample.jsonl`). Four fix groups were
approved: what an undecided gate means (A), the read-aloud alignment (B), the voice tasks (C), and
items, counts and speakers (D). The reviewer's result cache (E) and the viewer's theme and keyboard
navigation (F, G) and the reviewer's speaker judgment (H) ride in the same batch. Measurements are
from the r8 corpus (`triage_r8_20260929`, 62,550 recordings) unless a section says otherwise.

## A. What an undecided gate means

### What the audit found

`apply_gates` answered `UNDETERMINED` whenever a reading was absent, whatever the absence was, and
any undetermined gate made the whole conformance undetermined. Neither undetermined conformance nor
an undetermined flag gate flags (`verdict.undetermined_flags: false`; a flag gate flags only on
`passed is False`), so every absence was silent. On r8:

| gate | undetermined | cause |
|---|---|---|
| `dominant_speaker_share_min` | 2,217 | 1,423 nothing spoken (no lexical word, no diarization); 646 words present and no share written; 125 share written null (no diarized segment meets the task extent) |
| `production_min_s` / `voiced_fraction_min` / `continuity_min` / `f0_spread_max_semitones` | 666 / 666 / 617 / 622 | no qualifying carrier: rejected `lexical_separator` (467), `production_min_s` (597), `no_voicing` (183) |
| `events_min` | 144 | AIRWAY's classifier windows never reached the store (`event_instrument` written, no count) |
| `items_min` | 472 | random-item-generation: the category was never read, so no item reading was written |

And one undetermined gate hid a failed sibling in 5 recordings (`gates.py`, the any-undetermined rule
was checked before any-false).

### The rule

The absence is explained by the fold (`verdict.reading_absences`), which alone can see the store, and
the gate answers by the explanation:

| reason | when | answer |
|---|---|---|
| `no_carrier` | a VOICE carrier reading is absent, no carrier reading at all was written, and the tracks arrived | **False** for a conformance gate: the branch looked for a production and found none, so nothing was produced. **Not applicable** for a flag gate: a production that does not exist has no quality to ask about |
| `instrument_absent` | VOICE wrote its tracks-absent sentinel, or AIRWAY its `event_instrument` measurement | undetermined |
| `no_speech` | the speaker share is absent and the store carries no lexical word | **not applicable** |
| `null_no_overlap` | the speaker share was written null | undetermined |
| `null_value` | another reading was written null | undetermined |
| `absent_not_computed` | anything else: the branch had its inputs and wrote no reading | undetermined |
| `bound_unmeasured` | the reading is present and the winning layer's bound is null | undetermined |

A carrier reading absent while another carrier reading was written (`carrier_duration_s` present) is
`absent_not_computed`, not `no_carrier`: a carrier did qualify.

Conformance is then: not-applicable gates take no part; **any False decides False** (the fix to the
hidden-failure case); otherwise any undetermined leaves it undetermined, as does a group with no
applicable gate. The gate record carries `reason`; the gate outcome record carries `reason:
no_owner_report` where the owning branch left no in-family report.

**An undecided conformance gate whose reason is `absent_not_computed` or `instrument_absent` is a flag
ground of its own** (`verdict.uncomputed_reading_flags: true`, "a reading this task is judged on was
not computed", with each gate and reason appended). A reading the task is judged on that should exist
and does not is a defect in the pipeline or its inputs; leaving it silent is how 472
random-item-generation recordings sat unjudged. It is not raised for flag gates: a speaker share that
was not computed is recorded (`gate_dominant_speaker_share_min_reason`) but is not itself a statement
about the recording.

### Where it is visible

Parquet schema 12: `gate_<name>_reason` for each of the applied gates, `gate_reason`,
`gate_exempt`, `gate_not_applicable_n`, and `not_applicable` as a `gate_<name>_passed` value. The
data dictionary describes each; the viewer offers the reasons as categorical axes; the review page
shows "not applicable" and the reason beside each gate.

## B. The read-aloud alignment (ORDERED_TOKENS)

### What the audit found

1,800 of 13,700 Harvard recordings flagged; Caterpillar 49%, Stroop 31%, Rainbow 23%. The bound was
`omissions_max: 0`, whose only derivation was "the value the code used". A 40-recording stratified
sample of flagged Harvard (`harvard_sample.jsonl`) split into: nothing read at all (1,012 corpus-wide:
empty, `[breath]`, one or two words); exactly one omission (540), of which about 230 were alignment
defects (compounds `desk top`/`desktop`, `half way`, `thumb tacks`, `pure bred`; a swap `not to
spend`/`to not spend`), about 230 a single dropped function word, about 80 real misreadings; several
omissions (62), mostly real.

### What changed

1. **Compounds and swaps** (`stimulus.align_stimulus`). An adjacent pair on either side whose
   concatenation is one token of the other side, and which are not both tokens of it on their own, is
   aligned as that one token; an expected pair so joined is realised by the single word. After the
   alignment, an absent expected token is paired with an unpaired word of its own key at most one
   position away (`TRANSPOSITION_WINDOW = 1`): a swap of neighbours. One position, not two, because
   Stroop's list is ordered and a rotation of three answers is a real departure. A plural is not a
   compound (`canoes` stays a substitution of `canoe`). PREPROCESS's stored alignment is not rebuilt
   by a replay, so `stimulus_alignment_rebuild_agrees` reads False wherever the new pairing moved a
   count; it is a record, not a ground.
2. **Nothing read** is its own conformance ground ("read none of the words the stimulus asked for")
   where `expected_tokens_matched` is 0, instead of the generic non-conformance.
3. **The omission gate reads content, as a fraction.** `omissions_max` is replaced outright by
   `content_omission_fraction_max`, reading `expected_content_omitted_fraction`: of the stimulus's
   content tokens (`residue.is_content_word`), the fraction nothing realised; over every token where
   the stimulus has no content token (`hey hey hey`). Function-word drops are left out of both counts:
   the ~230 single dropped "the"/"a"/"to" were the sentence read, not unread. The count
   `expected_tokens_omitted` is still written.
   - Sentences (group `ORDERED_TOKENS`): **0.0** — any content word unread fails. This is not a fit;
     it is the instruction ("read the following sentences"), now applied to the words that carry the
     sentence.
   - **UNFITTED**: `rainbow-passage` 0.1, `caterpillar-passage` 0.1, `word-color-stroop` 0.2. No
     labelled verdicts exist to fit against; the values are proposals from the r8 distribution below,
     for the owner to approve.
4. **Stroop's stimulus is the answers.** `stimulus_text` lists the displayed ink colours, which is
   what the instruction asks to be said: over 400 r8 Stroop recordings the realised fraction has
   median 0.73 and 49% at or above 0.8, which could not happen if the text were the printed words
   (about half of which differ from the ink). The expectation already matched; only the bound changed.

## C. The voice tasks

### Glides

`dominant_segment_min_fraction` failed 54% of both glide families (861/1,596 and 813/1,554), and
`monotone_tolerance_semitones` could never fail: the run was segmented with that gate's own bound as
its tolerance, so the reading could not exceed it (r8: 0 false). A probe over 391 r8 glides
(`glide_probe.py`, 200 per family, seed 7) measured the sweep five ways:

| variant | pitch | tolerance | denominator | median held | held ≥ 0.5 | direction as declared |
|---|---|---|---|---|---|---|
| r8 (v0) | raw | 1 st | amplitude span | 0.43 | 42.5% | 83.6% |
| v1 | raw | 1 st | voiced extent | 0.51 | 52.4% | — |
| v2 | 7-frame median | 1 st | voiced extent | 0.67 | 66.0% | 84.9% |
| **v3** | 7-frame median | **2 st** | voiced extent | **0.76** | **77.2%** | **90.5%** |
| v4 | 7-frame median | 3 st | voiced extent | 0.78 | 81.3% | — |

93% of the sample travels at least 6 semitones (p10 8.2 st, median 17.0 st), so these are glides and
the r8 reading was mismeasuring them: single mistracked frames and 1–2 st vibrato broke the run, and
the amplitude span's unvoiced edges sat in the denominator. v3 is shipped (`branch.sweep_smoothing_frames:
7`, `branch.sweep_reversal_tolerance_semitones: 2.0`, the voiced extent as denominator); it also picks
the declared direction more often, which is independent evidence the run it finds is the sweep.

The fraction held does not separate a glide from a held note (recordings travelling under 6 st passed
v3 at 96%). Pitch travel does, so `monotone_tolerance_semitones` is replaced outright by
**`glide_extent_min_semitones`** over `glide_extent_semitones` (the dominant run's travel):
**UNFITTED, proposed 6.0 st**, the value that the measured distribution puts below its p10. The tolerance
became an instrument setting, as it always was in effect.

### Sustained vowels

- **Quality is a flag, not the task.** `voiced_fraction_min`, `f0_spread_max_semitones` and
  `continuity_min` are now flag gates (their own grounds), not conformance terms. Neither instruction
  asks for a steady or fully voiced vowel: maximum phonation time asks for "as long as possible",
  the prolonged vowel for the vowel held until the timer runs out. On r8 these three failed 507 + 183
  (spread) and 405 + 118 (voicing) MPT recordings as "the task did not happen"; they are now visible as
  what they are. A missing carrier makes them not applicable (there is no production to judge); it
  fails `production_min_s`, the conformance term.
- **Sustained conformance** is `production_min_s` (a floor: 0.5 s) and, where the family configures
  it, **`declared_duration_min_fraction`** over `production_declared_fraction` (the carrier's duration
  over the row's `declared_duration_s`). Only `prolonged-vowel` declares a duration (12 s): **UNFITTED,
  proposed 0.5**. MPT declares none and reports its duration (`phonation_onset_to_offset_s`,
  `carrier_duration_s`) without a ceiling, as "as long as possible" asks.
- **The count-in no longer rejects the vowel.** `prolonged-vowel` ("1, 2, 3 aah") had 482 of 1,603
  undetermined: the vowel shares one amplitude span with "three", and a span overlapping any lexical
  word was rejected as `lexical_separator`. The carrier is now the longest part of the span no lexical
  word touches (`branches.longest_free_interval`), rejected only if that part is shorter than
  `production_min_s`.

## D. Items, counts and speakers

### Item categories

`random-item-generation` (both versions) had 472 recordings undetermined because the repetition rule
read `hint.metadata["category"]`, which nothing writes. The category is in each recording's own
instructions ("… The recording will automatically stop at the end. Category: Animals."). Over 300
sampled r8 recordings (`counts_probe.py`, seed 7) it parses on 297 (99%): City names 39, Numbers 38,
Letters 35, First names 33, English words starting with 't' 32, Jobs 29, Fruits 26, Country names
23, Drinks 22, Animals 20. `branches.item_category` reads it (`Category: <name>.`); `animal-fluency`
declares `item_category: Animals` on its row. The repetition rule then follows (Letters and Numbers
allow repeats), `items_min` is judged, and the list's `category_items` count (listed words that are
members) and `item_category` are recorded.

**The category's items are task content.** `stimulus.task_lexicons` now enables `animal-fluency` and
both `random-item-generation` families; `task_lexicon(config, family, hint)` adds the category's
packaged list (`data/task_lexicon/categories/<slug>.yaml`: animals, fruits, drinks, jobs,
country-names, city-names, first-names) or its rule (a single letter for Letters, a digit or spelled
number for Numbers, the named initial for "words starting with 't'"). The residue, REDACT's exemption,
the reviewer's context and the fold's mask plan read it, so a listed first name, city or digit run is
not handed to the PII detectors. The lists are starting lists, not exhaustive: a missing item is
scanned as before. This changes the residue for at most 668 recordings (r8: animal-fluency 195, RIG 265, RIG-v2 208), so their
PII scans are expected result-cache misses; no other family's residue changes.

### Instructed counts

The v1 syllable instructions say "as fast as possible 10 times" (`task_instructions_curated.json`, all
five of pa/ta/ka/pataka/buttercup). The earlier ruling that no number was spoken
(`specs/20260921-required-and-typical-counts/`) read the v2 demo template the harvest carried; the
five rows now carry `RequiredCount(10, repetitions)` and no typical count. `ppg_typical_repetitions`
is replaced by `ppg_required_repetitions`.

A new conformance gate, **`instructed_count_min_fraction`** over `instructed_count_fraction` (produced
over asked), applies to EVENT_SERIES, EVENT_ALTERNATION, SYLLABLE_TRAIN and SYLLABLE_SEQUENCE. AIRWAY
writes it beside `airway_events_found`; the syllable body beside `ddk_repetitions_found`, only where the
decode counted (a weak decode beside an envelope carrier is not a count of zero). A family whose
instruction speaks no count writes none, and the gate is not applicable (`no_instructed_count`).

**UNFITTED, proposed 0.5 for all four groups.** Measured over r8 (120 per family, seed 7):

| family | asked | found p5 / p10 / p25 / p50 / p75 |
|---|---|---|
| diadochokinesis-pa | 10 | 5 / 8 / 10 / 12 / 15 |
| diadochokinesis-ta | 10 | 5 / 7 / 10 / 11 / 14 |
| diadochokinesis-ka | 10 | 3 / 6 / 9 / 10 / 12 |
| diadochokinesis-pataka | 10 | 5 / 8 / 9 / 10 / 11 |
| diadochokinesis-buttercup | 10 | 7 / 8 / 9 / 10 / 10 |
| voluntary-cough | 3 | 1 / 2 / 5 / 8 / 11 |
| respiration-and-cough-v2-threebreaths | 3 | 2 / 3 / 4 / 6 / 7 |
| respiration-and-cough-v2-threebreathsnose | 3 | 0 / 1 / 3 / 5 / 6 |
| respiration-and-cough-v2-threebreathsmouth | 3 | 0 / 2 / 4 / 6 / 7 |
| breath-sounds | 3 | 0 / 1 / 4 / 6 / 7 |

AIRWAY's events are not the instructed unit: one breath is often an inhale and an exhale event, and a
cough a burst of several, so the found count runs about twice the asked one and 0.5 is lenient there
(it fails the 0-and-1-event recordings). For DDK 0.5 fails about 5% (fewer than five of ten).

### Speaker share for model-speaker families, quiet breathing

- `dominant_speaker_share_min` is exempt for `harvard-sentences-list`, `cape-v-sentences` and
  `cape-v-sentences-v2` (done in A): their session instructions allow "someone else say the sentence
  first and you just repeat it after them".
- `verdict.hint_mismatch_exempt_families: [respiration-and-cough-breath, respiration-and-cough-v2-breath]`:
  quiet breathing the classifier does not hear is not "declared and did not find it".
