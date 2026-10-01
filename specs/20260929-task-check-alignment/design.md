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

## H. The reviewer's speaker judgment weighs the instructions

r8's reviewer answered `speakers: more_than_one` on recordings whose second voice was the task:
sub-9488c55d… animal-fluency (the participant asking "Is that enough?"), story-recall examiners, and
the model speaker Harvard/CAPE-V instructions allow. The judgment was one word, so the fold could only
flag every one of them.

- **Prompt** (point 4): judged "weighed against the task's instructions"; the participant addressing
  the examiner is one speaker; an examiner giving instructions, or someone saying a sentence first
  where the instructions allow repeating after them, is a voice the task expects. A seventh part,
  `OTHER_SPEAKERS`, quotes each other person's words exactly with `expected` and `why`.
- **Parse**: `OtherSpeaker(text, expected, why)` on the result and the per-round payload. The
  `SPEAKERS:` label no longer matches inside `OTHER_SPEAKERS:`.
- **Iteration**: `more_than_one` with no quote is an `answer_problem`, fed back like a proposal-less
  judgment; a quote that does not occur in the ORIGINAL is too.
- **Fold**: `REVIEWER_HEARD_SECOND_SPEAKER` ("…an unexpected second speaker…") is raised only where
  some quoted voice is `expected: false`, or where none was quoted after the loop; the ground names the
  unexpected quotes. The annotation (the ledger) records `other_speakers` and `prompt_version`.
- **`PROMPT_VERSION = 2`** (`redaction_review.py`): the prompt and its parse as one number. It is the
  reviewer result cache's version term (E); every r8 reading is a miss for that reason alone.

## E. The reviewer's result cache

REVIEW's loop is the one GPU step no cache served: every run re-read every residue, although most
recordings' residue, context and prompt are unchanged between runs. It now goes through the triage
result cache (`SENSELAB_CACHE/results/schema-24/`) as process `redaction_review` (version 1; adding it
changes no other process's key).

- **Key** (`review.review_cache_key`): `transcript_signature` of the canonical JSON of exactly what the
  loop reads — the ORIGINAL, the RELEASED text (or null) and the task context — plus
  `prompt_version` (`redaction_review.PROMPT_VERSION`), the model id and the 40-hex commit its ref
  resolved to, `max_new_tokens` and `max_iterations`. `timeout_s` and `keep_worker_resident` shape no
  answer and are not in it.
- **Value**: the loop's final reading (the annotation's fields) and every round's payload, so a hit
  writes the same per-round records and annotation a miss would. Only an answered reading
  (`clean`/`flagged`) whose loaded commit is the resolved one is kept; an absent one is asked again.
- **Record**: the annotation's `result_cache` says `{key, hit, stored}`.
- **Backfill**: `scripts/triage_review_cache_backfill.py RUN_ROOT… [--manifest] [--config]` rebuilds the
  key from a finished store (its transcript texts, the annotation's task context, model and commit,
  the configured settings) and stores the reading. It seeds only readings made with the current
  prompt version: every r8 reading predates `PROMPT_VERSION` (H bumps it to 2), so r9 misses on every
  reviewed recording, for that reason alone, and r10 hits wherever the residue is unchanged. Run it
  over r9's tree only if r9's stores were written by a checkout without this cache.

## F. The viewer's theme

`theme.js`: a `theme: <mode>` button on the landing and in the top bar cycles system → light → dark.
The choice is kept in `localStorage` (every access in try/catch; unavailable storage reads as
system), and a script in `<head>` sets `data-theme` before first paint. `styles.css` defines the light
tokens on `:root`, the dark ones under `@media (prefers-color-scheme: dark)` for
`:root:not([data-theme="light"])` and again under `:root[data-theme="dark"]`. Every colour the two
canvases and the brush overlay draw (the 59 literals in `corpus.js`, `recording.js` and `app.js`, and
the 16 in the stylesheet) is a token; the canvases read `--c-<name>` through `Theme.color`, and a theme
change (or a system change while in system mode) rebuilds the colour map and repaints both views.
`theme.test.mjs` checks both themes define every canvas token, the two dark blocks agree, no JS
colour literal remains, and contrast: text (`ink`, `ink-2`) at least 4.5:1 on `ground`, `panel` and
`panel-2`, marks (accent, the status colours, the ten series) at least 3:1 on `ground`, in both themes.

## G. Stepping through the selection

`keys.js`: `j`/`↓` next, `k`/`↑` previous, `Home`/`End` first/last, over the rows the current brushes and
facets select, in row order (the list's order). Each step opens that recording (detail panel,
highlighted line, scroll) and the under-bar reads `n of N`; a step past an end stays and says "first/last
in the selection". Keys typed in an input, select, textarea or editable element, or with a modifier,
are not steps. From a recording outside the selection a step enters it at the nearest selected row.
The key help is in the under-bar. `keys.test.mjs` and two browser specs cover it.

The browser suite (`npx playwright test`, fixture regenerated for schema 12) passes 16 of 20; the four
failures (`gate_repetitions_min` axis bound, zero-on-band, outcome axis, facet gate count) fail
identically at the commit before F and G: the fixture script still generates the pre-schema-10 gate
set (`train_min_s`, `coverage_min`, `items_min`, `dominant_speaker_share_min`), and the facet spec
counts located gates that carry no `_passed` column since schema 10. Not fixed here.

## Offline replay (288 recordings, `tmp_taskfix/replay200b`, at 832935a5)

Three per family over all 67 families plus seven more in fourteen, and the owner's two cases; CPU, into
a copy, compared against r8's final summaries with every reviewer ground left out of both.

- Flagged (non-reviewer grounds): r8 61 → 45 of 288. Caterpillar 7 → 2, Rainbow 3 → 0, Stroop 4 → 1,
  Harvard 2 → 1 (the one left: nothing read), glides-low-to-high 6 → 3, quiet breathing 5 → 0.
  New flags: MPT/prolonged-vowel quality flags (voicing 8, spread 5), prolonged-vowel short of half its
  12 s (4), and single cases of an instructed count short of half (cough, five breaths, /pa/).
- Undetermined applied gates: only `instructed_count_min_fraction` not applicable (`no_instructed_count`,
  18). The first pass also showed 3 prolonged-vowel `declared_duration_min_fraction` absent_not_computed
  and 3 no-carrier rejections caused by an ASR "Ah." covering the vowel; both fixed in 832935a5.
- RIG: all 20 were undetermined in r8; all now carry `items_min` (6–147 items) and pass.
- The owner's cases: sub-f2876e24… harvard-sentences-list-13-1 now passes (6 matched, content omission
  0.0, the dominant-speaker gate exempt); sub-9488c55d… animal-fluency passes on the replay (its r8 flag
  was a reviewer ground, re-read in r9 under H).
- Measured distributions for the UNFITTED bounds: prolonged-vowel declared fraction
  [0.08, 0.28, 0.36, 0.43, 0.46, 0.62, 0.66, 0.70, 0.70, 0.72]; glide travel (st) median 18.2, 3 of 23
  under 6; content omission — Harvard all 0 but one nothing-read, Caterpillar 0.01/0.01/0.11/1.0 above
  zero, Rainbow 0.03/0.09, Stroop 0.07/0.07/0.13; instructed fraction — DDK 0.4–4.1, airway 0.0–4.3.

## I. Any other voice is flagged (owner, 2026-09-30) — supersedes H's "unexpected only" and D(3)'s exemption

The story-recall card sub-00053adb… opens with the examiner's instructions ("You were given the text …
up to five minutes"); pyannote diarized one speaker over 50.5 s, and under H an examiner the instructions
provide for was "expected" and did not flag. The owner's rule: any speech attributed to someone other than
the participant is flagged for review, whether or not the instructions expect it — an expected voice is
still another voice. The instructions only keep the reviewer from mistaking the participant addressing the
examiner ("Is that enough?") for a second voice.

- Prompt (`PROMPT_VERSION` 3, which changes the reviewer cache key): an examiner or a model speaker is
  `more_than_one`, quoted, with `expected` recorded as information.
- Fold: `REVIEWER_HEARD_SECOND_SPEAKER` ("the redaction reviewer read another speaker in the transcript")
  flags on any quoted other voice, the ground naming each quote as expected or unexpected; an unquoted
  `more_than_one` is still fed back through the loop, and flags with "no words quoted" if it survives.
- The `verdict.flag_gate_exemptions` table is now empty: Harvard and CAPE-V are again held to
  `dominant_speaker_share_min`. `verdict.model_speaker_families` names them, and the speaker gate's flag
  ground appends "the task's instructions permit a model speaker" — information, not exemption. The
  `gate_exempt` parquet column stays and reads `[]`.

## J. Diagnoses only, and instructions addressed to the participant (prompt v4)

Owner, 2026-09-30: "conditions should be specific medical diagnosis not just 'change in my voice' or
coughed." r9 (prompt v3) listed a condition in 2,256 readings and held 1,831 recordings for "other"
condition review; the top phrases were surgery 42, anxious 36, stress 26, allergies 24, covid 22,
tired 21, sadness 21, and only about 355 of the 1,831 named a diagnosis (a keyword heuristic over
`evaluations_r9_20260930/other_condition_phrases_r9.csv`). v4's CONDITIONS part lists only named
diseases, disorders and syndromes the speaker attributes to themselves. Symptoms and sensations,
feelings and moods, everyday events and procedures, and a medication or treatment on its own are not
listed; a treatment reaches the list only through the diagnosis it names.

The same pass exposed a missed second voice. r9's story-recall reading for sub-00053adb… read the
examiner's "You were given the test… I said you have up to five minutes" and reasoned that "the
participant [is] repeating the instructions", so it answered `speakers: one`; diarization had heard one
speaker too. The recall residue did not hide those words (99 residue words, 49 story words subtracted).
v4 tells the reviewer that words addressed to the participant which give, restate or enforce the
instructions are another person's voice unless the words themselves show the participant reading them,
and to answer `unclear` with those words quoted when it cannot tell. The fold now flags an `unclear`
reading that quotes words, under the same second-speaker ground, so a potential second person reaches a
person; `unclear` with nothing quoted flags nothing.

`PROMPT_VERSION = 4`. Owner's scope: re-review now only the r9 readings that listed a condition (plus
the story-recall card); a full re-review of every reading on v4 is scheduled (see `triage_r9_20260929/RUN.md`).

## K. The closed-class inventory (owner, 2026-09-30: "between should not be flagged")

`residue.FUNCTION_WORDS` is the one definition behind `is_content_word`: the residue's content test,
the mask trim, and which words of a reviewer proposal carry a mark. It lacked common prepositions, so
"between" in "synovial joint cyst between Cone one and Ctwo" was drawn red.

The list was audited against the closed-class inventory of the *Cambridge Grammar of the English
Language* (Huddleston & Pullum 2002, ch. 7 prepositions, ch. 5 determinatives, ch. 5 pronouns,
ch. 3 auxiliaries, ch. 15 subordinators) and 96 closed-class words were added: prepositions (across,
against, along, amid, among, amongst, around, behind, below, beneath, beside, besides, between, beyond,
despite, down, during, except, inside, near, outside, past, per, since, through, throughout, till,
toward, towards, underneath, unlike, until, via, within); subordinators (because, although, though,
unless, whereas, whether, while); determinatives (another, either, neither, few, many, much, more, most,
several, such, other); reflexive, possessive and compound pronouns (myself … themselves, yours, hers,
ours, theirs, someone, something, anyone, anything, everyone, everything, nobody, nothing, whatever,
whoever); and negated and cliticised auxiliaries (hasn't … shan't, we've … it'd). Numbers stay content
(an age, a date, a phone digit can identify), and no noun or lexical verb was added.

Measured on the r9 stores (62,550): 32 currently masked words and 43 red proposal words become
non-content — past 14, few 10, down 5, couldn't 4, more 4, something 4 — and 0 residues change their
content flag. The residue string itself does not change (function words stay in it; only the content
test reads the list), so the PII result cache is unaffected. Masks and proposal marks follow at the next
re-fold; the scan's content flag would change only on a replay, which here changes nothing.

The free-speech page's trim-released words (orange) are drawn with a dashed underline in both themes,
legend included, since 2a4340de; red (masked or proposed) and green (unmasked by the reviewer) stay
solid. A page test pins it. The recording-vectors viewer draws no word states.

## L. The task's instructions spoken in the recording

Owner, 2026-09-30/10-01: a recording whose transcript contains the task's own instructions is flagged
for review, whoever spoke them. The case: a story-recall recording that opens with the examiner's "You
were given the test, read the text, have you familiarized? I said you have up to five minutes to read
it as many times as you want". Diarization heard one speaker and the v4 reviewer read the words as the
participant repeating the instructions, so neither caught it.

**Rejected: a deterministic match.** A lexical rule — the longest in-order run of content words the
transcript shares with the sidecar instructions — was measured on r9 (all 62,550). The run was 0 for
52,085 recordings, 1 for 8,944, 2 for 285, 3 for 1,228 (1,207 the prolonged-vowel count-in "one two
three", which the instructions ask for) and ≥ 4 for only 8, every one instruction text. Examiners
paraphrase ("You were given the test, read the text" against "You are given a text. Read the text"),
so an exact run finds a handful and misses the rest; an embedding similarity would need a fitted
threshold and labels. Owner: ask the reviewer, which already reads the instructions in its context.

**Rule (prompt v5).** Point 6 asks whether the instructions are spoken, verbatim or paraphrased, by
anyone, judged apart from SPEAKERS; the stimulus is never instructions. A required
`INSTRUCTIONS_SPOKEN` part quotes each passage exactly from the ORIGINAL, or `[]`. A missing part is fed
back for another round, and a quote absent from the ORIGINAL is rejected (`answer_problem`). The fold
flags a non-empty part under `INSTRUCTIONS_SPOKEN`, naming the quotes; the release is unchanged
(`verdict.llm_instructions_spoken_flags: true`). Parquet schema 13 adds `llm_instructions_spoken_n`.
Readings before v5 carry no part and never flag; the scheduled full re-review on v5 applies it.
