# The PII span ledger, the word-level mask rule, and a condition held for a person

2026-09-27. Branch `design/triage-workflow-dag`.

## What the owner asked

On an r6 free-speech card (`sub-00053adb…_task-free-speech-1`) REDACT had masked one DATE_TIME ("this
morning"); the reviewer proposed releasing it and redacting two health conditions ("essential
tremors", "synovial joint cyst"); the fold withheld the recording; the page showed only the date.

> this depends on uniqueness of condition, especially when coupled with other information being
> released. so should be flagged for review, and the speech transcript should show the mask for this.
> instead what it is showing is the date time, which i think the reviewer unmasked. we need a way of
> indicating all PII, including condition, and which were left masked and unmasked by the reviewer.

And, on how much of a mask a reviewer quote resets:

> the reviewer should indicate specifically what words or masks to unmask or what words to mask. this
> could involve the entire mask or part of a mask. it should never keep not content words masked if
> that happens.

## 1. One word-level mask rule

`mask_plan` in `nodes/redact.py`, the one place the rule lives; VERDICT computes it, the fold reads
its counts, `settle_release` renders what it recorded, and the page and the parquet read the ledger
it produced.

1. A planned mask covers every consensus word whose timing hull its extent overlaps.
2. A word leaves the mask when a reviewer `release` entry names it — matched as a run of whole
   tokens, at every place the quote occurs — and the fold lets the reading unmask
   (`verdict.llm_reset_redactions`, and the reviewer read the text). An entry may cover a whole
   mask or part of one.
3. A word also leaves the mask when it is not a residue content word:
   `residue.is_content_word` — lexical, not a vocalisation, filler, fragment or bracketed marker, and
   not in `residue.FUNCTION_WORDS`, the list the residue's own `content` test reads — or when it is
   not in the residue at all (task content the padding reached). This applies with or without a
   reading, whether the padding or the finding itself caught the word.
3a. And a content word leaves the mask when no detector marked it (SPEECH's `label`/`pii`
   assertions, the words each finding overlaps) while another word of the same mask is marked: only
   the padding reached it. On the first replay the free-speech-1 card's DATE_TIME mask, planned over
   "this morning" with 250 ms of padding, also held "study", a content word no detector marked; under
   rules 1-3 alone it stayed masked. A mask none of whose words is marked keeps its content words,
   since nothing then says which of them the finding was.
4. The kept words of each mask are re-cut into one extent per run of stream-adjacent kept words.
5. A mask none of whose words stays masked disappears. A mask covering no word at all is kept as
   planned: nothing says what it hides.
6. The reviewer's entries apply whatever else the reading proposes. A reading that also proposes a
   `redact`, or reads the original as `carries_pii`, still unmasks exactly the words it named; its
   `redact` entries decide the release on their own (withheld, or held for human review), and only
   the mask state changes. Owner, 2026-09-27: the reviewer indicates exactly what to unmask. This
   removed a guard that withheld the entries where they would have unmasked every word of an
   original read as `carries_pii` (122 readings on r6; §6).

**The padding.** A kept run's extent is the words' timing hull widened by `redaction.padding_ms` on
each side, the widening stopping at the consensus extent of the nearest word left unmasked on that
side, and never shrinking below the run's own words' consensus extents. So the audio does not clip a
masked word, and the padding never silences a word the rule unmasked. Where two recognizers placed
neighbouring words so that their hulls overlap, the kept word's own extent wins: an unmasked
neighbour may lose a sliver of audio, a masked word never gains one.

**Text follows words, not geometry.** The released transcript masks exactly the words the ledger
lists under each final mask (`_render` with `owners`), so the text and the audio say the same thing
even where a padded extent grazes a neighbour's hull.

**What this replaced.** The rule of 2026-09-26 had two branches: over an original read as clean, a
quote naming any word under a padded mask reset the whole mask; otherwise every hidden word had to be
named. The first released words the reviewer never named (an r6 Spanish card restored
"mi estado de ánimo" under a quote of "de que"); the second kept whole masks over words nobody
thought identifying (5,259 masks on r5, all from padding). Both are gone.

**Names spelled as closed-class words.** `FUNCTION_WORDS` holds `may`, `will`, `can`, `us`, and the
Spanish articles, so rule 3 alone unmasked "May", "U.S.", "Los" where a detector had marked them. A
word a detector marked in one of `verdict.trim_protected_categories` (PERSON, NAME, LOCATION, LOC)
and written as a proper noun (`residue.is_proper_form`: capitalised, not a first-person pronoun, and
mid-sentence or one of `NAME_HOMOGRAPHS`) counts as content for the trim; only a reviewer entry naming
it unmasks it. §7 has the measurement that chose this over protecting every marked word.

## 2. The ledger

`pii_ledger`, one measurement per fold, generated by VERDICT's activity and derived from its verdict.
It carries the release and its ground, every planned mask with each word it covers (`id`, `text`,
`state` ∈ `masked` / `unmasked_by_reviewer` / `unmasked_by_trim`, `named` — whether a `release`
entry named it, applied or not — and `content`), the mask's `outcome` (`unchanged`, `trimmed`,
`partly_unmasked`, `unmasked`), the final masks with the word ids each hides, every reviewer `redact`
entry placed on words (`placed` = `words` for whole-token runs, `substring` where only a substring
matched, empty where nothing did; `masked_ids`; `human_review`), the `release` quotes that placed
nowhere and those that named no masked word, and per-state counts and categories.

A `release` entry the fold did not apply — the reading also proposed a redaction, as on the card
above — leaves its words `masked` with `named: true`: the page can show that the reviewer would have
unmasked the date while the recording is withheld for the conditions.

The released artefacts carry none of it: `consensus.json` folds each final mask into one placeholder
with its category, bounds and word count, as before, and no masked surface.

**A re-fold keeps it live.** The store is content-addressed, so a re-fold that concludes what already
stands mints the same ledger entity; `refold_verdict` now excludes the new ledger from the entities it
retires, as it already excluded the new verdict. Without that a second identical re-fold left no live
ledger (`test_an_unchanged_re_fold_keeps_one_live_ledger` fails on the old line with 0 live).

## 3. A condition is held for a person

`verdict.llm_human_review_categories: [CONDITION]`. A residue reading every `redact` entry of which is
in one of those categories withholds for human review (§10 names the two grounds), and the triage axis flags
on the same ground with the categories appended. A reading proposing a condition beside a name, a
place or anything else keeps `REVIEWER_PROPOSED_REDACTION`, and the ledger still lists the condition
spans. The redact proposals are not applied as masks; the release policy for them is unchanged.

## 4. The hoarse card was not a matching defect

`sub-01f52a78…_task-free-speech` carries two MISC masks, over "am very hoarse." (3.25–5.05 s, whose
padding reaches "My voice") and over "voice is very hoarse" (5.33–6.89 s). The reviewer's one release
entry is "am very hoarse". It occurs once; the second phrase is "is very hoarse", which the quote does
not match. The multi-occurrence rule was already right (`test_a_quote_resets_every_place_it_occurs`)
and is kept. Under the word-level rule the first mask goes (named words, plus "My" trimmed), and the
second keeps "voice" and "hoarse" — "is" and "very" are trimmed. The reviewer's reason ("a common
symptom like hoarseness does not identify the speaker") covers both, but its quote named one; the
rule applies what a quote names, not what its reason implies.

## 5. A reviewer fault is an error, and a resubmission reads again

r6 review slices 25 and 34 ran on node2119, whose GPU raised `CUDA error: an illegal memory access`
once and poisoned the resident worker: 715 recordings got an `absent` annotation carrying the failure,
and the driver logged every row `ok`. `extend_llm_review.standing` counted that annotation as a
standing reading, so a plain resubmission would have read nothing. Now an `absent` annotation that
carries a `failure` is not standing, and a read that produces one is logged `error` and not written,
so the store keeps whatever reading it had.

## 6. Measured on r6

An in-memory re-fold of all 15,210 reviewed r6 stores at `83d89d0e` (job 24106151; nothing written
into the tree), over the stored REVIEW readings, with `review_on.yaml`.

| release, ground | r6 as folded | word-level rule |
|---|---|---|
| without redaction, scan found nothing | 6,687 | 6,687 |
| without redaction, reviewer unmasked all | 6,118 (reset every) | 5,398 |
| without redaction, no content word masked | — | 177 |
| with redaction, REDACT's plan | 1,316 | 612 |
| with redaction, reviewer unmasked some | 174 (reset some) | 621 |
| with redaction, trimmed to content | — | 820 |
| with redaction, re-scan cleared | 23 | 3 |
| withheld, reviewer proposed redaction | 823 | 623 |
| withheld, human review (condition only) | — | 200 |
| withheld, REDACT | 69 | 69 |

Of 14,028 planned masks over 7,977 recordings: 9,621 unmasked, 362 partly unmasked, 3,741 trimmed,
304 unchanged. Words: 12,714 stay masked, 21,800 unmasked by the reviewer, 62,812 unmasked by the
trim, 1,743 proposed by the reviewer and not masked. The triage axis is unchanged in count (2,016
flags among these); 200 flags now carry the human-review ground.

Release quotes that place on no residue word: 46 of 11,204 (0.41%), most of them placeholders the
reviewer quoted back (`[PERSON+NAME]`, `the [DATE_TIME]`). 287 more name only words no mask covers.
Of 1,503 `redact` entries, 1,487 place as whole-token runs, 10 only as substrings, 6 nowhere. At this
rate a prompt change asking the reviewer to cite word or mask ids is not needed.

Capitalised closed-class words the trim unmasked: 335, of which "So" 186 and "Well" 125 are
discourse markers; "May" 13, "Will" 3 and "Can" 3 are the names-as-function-words risk of §1.
122 readings had their `release` entries withheld by the `carries_pii` guard.

## 7. The two changes of 2026-09-27, afternoon

**Named unmasks always apply.** The `carries_pii` guard is gone, and a reading that also proposes a
redaction unmasks what it named; its redact entries still decide the release. On r6, the 122 readings
the guard had held: 121 move from release with redaction to release without redaction under
`REVIEWER_UNMASKED_ALL`, 1 stays a partial copy; their masked words go from 386 to 1. The free-speech-1
card (withheld for two conditions) now shows "morning" as `unmasked_by_reviewer`; its release is
unchanged.

**The trim protects marked proper nouns.** Measured on the same replay with PERSON and NAME protected:

| rule | closed-class words kept masked | PERSON/NAME "May"/"Will"/"Can" trimmed |
|---|---|---|
| none | 0 | 19 |
| every marked word | 2,976 ("and" 162, "the" 154, "I" 144, …) | 0 |
| marked and written as a proper noun | 71 | 0 |

The literal rule contradicts rule 3, because a finding labels every word its span overlaps. With the
proper-form rule, the marked proper-form words the trim still released were, by category, PERSON 205
(the caterpillar passage's own words, which are task content, not residue), DATE_TIME 39 ("May" 11,
"The" 10), LOCATION/LOC 64 ("Los" 14, "U.S." 11, "Las" 8, "La" 4). LOCATION and LOC are protected;
DATE_TIME is not.

**Both changes, as shipped** (`f1d2e2f4`, protected categories PERSON, NAME, LOCATION, LOC; job
24111691, all 15,210 reviewed r6 stores): without redaction, scan found nothing 6,687; reviewer
unmasked all 5,522; no content masked 163. With redaction, REDACT's plan 604; reviewer unmasked some
630; trimmed to content 709; re-scan cleared 3. Withheld: reviewer proposed redaction 623; human
review 200; REDACT 69. Masks: 10,316 unmasked, 482 split, 2,952 trimmed, 278 unchanged. Words:
10,410 masked, 24,332 unmasked by the reviewer, 62,584 unmasked by the trim, 1,744 proposed and not
masked. Closed-class words the protection keeps masked: 97. No PERSON/NAME/LOCATION-marked
"May", "Will", "Can", "Los", "Las", "La" or "U.S." written as a proper noun is released by the trim;
the 12 "May" still trimmed are DATE_TIME months.

## 8. The released text follows the ledger where the extents did not move (2026-09-27, evening)

**Defect.** Across all 1,946 r6 releases with redaction, 13 released transcripts dropped a word the
ledger marks `unmasked_by_trim` (the page's sample of 408 had found 7). In every one the trim's re-cut
extent came out equal to REDACT's planned extent, so `settle_release` took its "REDACT's own plan"
branch, which rendered the text by geometry: every word whose *hull* overlaps an extent folds into the
placeholder. The dropped word's hull reached the mask because one recogniser timed it far wider than
its derived extent, e.g. "in" at 8.63–8.94 s with a hull to 9.04 s against a mask from 8.95 s, or
"sorry," with a hull of 28.7–69.7 s. The audio was right in all 13 (the extents are the same); only
the text was wrong, and it erred toward hiding.

**Fix.** The text is rendered from the ledger's owners on every path. Only the audio source depends
on whether the final extents are REDACT's own: they are, and REDACT's `redacted` stream is reused;
they are not, and the source is re-masked. The test is built from the r6 "back in [DATE_TIME]" case.

## 9. A mask covers only the residue: the task's own words are never masked (2026-09-27, evening)

**Owner:** "regarding redact for caterpillar passage or any read passage there should be very little
to redact that's related to the text itself."

**Measured on r6** (every read-aloud recording REDACT planned masks for; words counted by consensus
extent overlapping a planned extent):

| family | recordings | masks | seconds masked | words reached | of them task words | residue words |
|---|---|---|---|---|---|---|
| caterpillar | 124 | 164 | 4,357 | 10,213 | 9,831 | 382 |
| rainbow | 67 | 87 | 765 | 1,505 | 1,235 | 270 |
| harvard | 428 | 451 | 822 | 1,870 | 1,069 | 801 |
| stroop | 131 | 185 | 2,462 | 1,357 | 610 | 747 |
| cape-v | 129 | 139 | 221 | 478 | 211 | 267 |

Single caterpillar masks ran 157 s, 145 s and 125 s: the whole take.

**Mechanism, in SPEECH step 7 (`nodes/speech.py`, the pii placement loop).** The scan reads only the
residue, but a finding is placed back on the full consensus transcript, and two paths carry it onto
the task's words:

1. **Bridging.** A located finding runs from the position of its first residue token to that of its
   last, and every consensus word between them is covered and marked. Residue tokens are adjacent in
   the scanned text and far apart in the transcript, so a two-token finding over a misread word at
   10 s and another at 150 s covers the passage between.
2. **The unlocated fail-safe.** A finding whose text cannot be matched back to the scanned tokens
   covers `(0, len(words) - 1)`: every word of the transcript, the passage included. The caterpillar
   examples carry `haystack: consensus` and mark "Do you like amusement parks? Well, I sure do. To
   amuse myself…", the passage's own opening.

Neither is a finding on the task's words: the detectors never read them.

**Fix, at the fold (`mask_plan`), so no REDACT re-run and no re-review.** A mask covers only residue
words; a word outside the residue is never listed under a mask and never masked. A planned extent
that reaches a task word's audio (its consensus extent) is re-cut to runs of its kept residue words
even where no word changed state, with the padding stopping at the nearest unmasked word, which now
includes the task's words either side. A mask left covering task words only disappears. The unlocated
fail-safe keeps its intent, narrowed to what the scan read: every residue content word stays masked.
Each mask records `task_words_n`, how many task words its planned extent reached, so the ledger keeps
the evidence of what REDACT planned without listing those words as masked.

SPEECH's placement itself is not changed here: changing it re-runs SPEECH's PII step, which retires
every REVIEW reading and needs a GPU re-review. The fold-level rule makes the released copy right
without it; the placement defect is recorded for the next SPEECH change.

REVIEW's inputs are untouched: it read REDACT's rendering of the planned extents, and the stored
readings stand.

## 10. The study's own conditions are a separate section (2026-09-27, evening)

**Owner:** "put study's conditions in a separate section". The r6 human-review list was led by the
cohort's own recruitment diagnoses (Parkinson's in 50 recordings, idiopathic subglottic stenosis 13,
spasmodic dysphonia 11), which say little about who someone is inside a corpus recruited for them;
the rest were 222 phrases each in a single recording.

**Where the list comes from.** Not recall: the release's
`phenotype/diagnosis/` (`b2aivoice/4.0-release/adult/bids_adult_2026_09_04`) holds one file per
diagnosis the study recruits for, 21 files with `control`, so 20 cohort diagnoses. The profile
`data/cohort_conditions/bridge2ai_voice_adult_2026-09-04.yaml` has one entry per file, keyed by the
file's stem. Each entry's patterns are the diagnosis name and the spellings the r6 reviewer actually
quoted (e.g. "sublotic stenosis", "spasmatic dysphonia", "lavaedoba"). A treatment is included only
where that diagnosis's own phenotype file names it:
- levodopa, dopamine agonists and deep brain stimulation (DBS): `parkinsons_disease`, which also
  names DBS alongside `essential_tremor` and `laryngeal_dystonia`;
- botulinum toxin (Botox): `laryngeal_dystonia`;
- dilation: `airway_stenosis`;
- thyroplasty: `glottic_insufficiency` and `unilateral_vocal_fold_paralysis`.

"Spasmodic dysphonia" appears in no file. `laryngeal_dystonia` names its adductor subtype ADLD, the
current name for adductor spasmodic dysphonia, so the older name maps there.

**Checked against every distinct r6 CONDITION phrase:** 148 of 266 phrases are cohort (282 mentions),
118 are other (135 mentions).
- By diagnosis: airway stenosis 28, Parkinson's 24, vocal fold paralysis 22, laryngeal dystonia 16,
  benign lesions 10, and the rest 1–7 each.
- The other list is cancers, strokes, surgeries, sleep disorders, autoimmune disease and medication
  names, plus "a very rare voice disorder", which correctly stays other.
- Kept other on purpose: "vocal cord dysfunction" (a distinct disorder, not a cohort diagnosis);
  "damage to my voice box" and "vocal cord damage" (no diagnosis named); a growth on the thyroid
  (not a laryngeal lesion); a joint cyst and a brain cyst.

**Behaviour.** Policy is unchanged: both kinds stay withheld for human review. A reading whose every
human-review proposal is a cohort condition withholds under `REVIEWER_NEEDS_HUMAN_REVIEW_COHORT`; one
naming any other condition, under `REVIEWER_NEEDS_HUMAN_REVIEW_OTHER`, since the rarer condition is the
one a reviewer must weigh. The ledger records each proposal's `condition_kind` and `cohort_diagnosis`,
and the recording's `human_review_kind`, `cohort_diagnoses` and per-kind counts. Those feed the page
section and the parquet (schema 9).

## 11. The page shows the state, the popup says why (2026-09-27, evening)

**Owner:** "i don't need an explanation when reviewing the text. such a thing can be put into the popup
of what determined status if it's not already there. regarding the text itself, let's minimize
explanations embedded. simply show the category and use underline colors to dissociate between masked
(red), unmasked (green), and padding unmasked (orange). no need for a condition for padding." And:
"also for those without consensus words show a single stream if available".

**The text.** Each span carries its category and an underline colour, and nothing else.
- **Red:** masked in the released copy, or proposed by the reviewer for masking; either way it is
  meant to be hidden.
- **Green:** unmasked by the reviewer, or a real detector finding the release shows unmasked, such as
  one exempted as expected speech.
- **Orange:** a word the trim released because only the padding reached it or it is not a content
  word. No category label.
- **Nothing drawn:** a detector mark on a word outside the residue, which is the §9 placement artefact.

The reviewer's own per-finding verdict moved from the underline colour to a small dot after the span,
so the underline carries only the state. The legend has three swatches.

**The popup** ("what determined this status") gained an "on this card" section with everything that
used to sit on the card or inside the marks:
- the release ground and the deciding reason;
- the scan state;
- the residue method, its count and its words;
- the ledger's per-state counts, and the task words REDACT's plan reached;
- the condition-review kind and diagnoses;
- a table with one row per span: its words, category, state, why the trim released it, whether the
  reviewer named it, its condition kind and diagnosis, and whether it is held for human review.

**Condition review** is a rail filter: study cohort condition, other condition, or none. It sits on
the card as `data-hk`, not as text.

**A single stream.** Where the consensus carries no words, the card shows the first live
`asr_hypothesis` with words, in the order the store wrote them, which is the order PREPROCESS runs its
recognisers (`asr_crisperwhisper`, then `asr_qwen`). The card is tagged "<source> only", and the
popup says why. No PII mark is drawn on it, because the ledger's spans are placed on consensus words
and this stream has none.

**Theme** (owner: "also add a dark/light toggle to the html viewer"). The review page's colours are
custom properties with a light and a dark set. The dark set applies where the system prefers dark and
the reader has not chosen light, and wherever the reader chose dark. A rail button cycles through
system, light and dark, and the choice is kept in `localStorage` (`senselab.fsreview.theme`); every
access is wrapped, so a page with storage blocked still works. A head script applies a stored choice
before the page paints. The underline colours were chosen by contrast, measured against each theme's
card background:

| theme | red | green | orange | red vs orange |
|---|---|---|---|---|
| light | #b3123a, 6.9:1 | #1e7b34, 5.3:1 | #a86b00, 4.4:1 | 1.56:1 luminance, 53° hue apart |
| dark | #ff5c7a, 5.6:1 | #5fcf7a, 8.5:1 | #ffc145, 10.3:1 | 1.84:1 luminance, 51° hue apart |

The first choice put red and orange 27° apart; red was moved toward crimson and orange toward amber.
The recording-vectors viewer is not themed: it is dark-only, and its canvas plots draw with 59 fixed
dark-palette colours in `recording.js` and `corpus.js`, so a light theme there is a redraw of every
plot, not a stylesheet.

## 12. Measured: the final code replayed over the reviewed r6 corpus (2026-09-27, evening)

All 15,210 reviewed r6 stores were folded again in memory at `864de752` (the fold code; later commits
touch only the page). None failed, and nothing was written to the tree.

- **Release and triage totals are unchanged**: 12,372 without redaction, 1,946 with redaction, 892
  withheld, 2,016 flagged. No recording changes release.
- **Released text (§8):** 13 releases with redaction change text, exactly the 13 defect cases. For
  example, the stroop "[MISC]" placeholder moves from a distant "Blue." whose hull overlapped the mask
  to the word it actually masks, and "forest," and "dying" return to their released transcripts.
- **Read passages (§9):** the masked words did not change: harvard 86, stroop 83, rainbow 58,
  caterpillar 92, cape-v 54. The trim already refused to keep a task word and re-cut such extents, so
  the released audio and text were already confined. What changed is the ledger and the page: the
  13,291 task words they listed under masks as "unmasked by trim" are no longer listed (caterpillar
  9,895, rainbow 1,312, harvard 1,152, stroop 711, cape-v 221). They are counted as `task_words_n`
  instead. Three masks whose only non-kept words were task words reached by hull alone now read as
  REDACT's plan unchanged rather than trimmed.
- **Conditions (§10):** 314 recordings carry a human-review proposal, 206 cohort and 108 other. Of the
  200 withheld on a human-review ground, 124 are cohort only and 76 name another condition. The 420
  proposal spans are 286 cohort and 134 other. The most named diagnoses are Parkinson's (75), airway
  stenosis (55), vocal fold paralysis (30), laryngeal dystonia (24) and muscle tension dysphonia (13).
- **Single stream (§11):** no free-response card needs it. The two recordings with one recogniser are
  rainbow-passage and random-item-generation-v2, neither free-response, and on both the fallback
  picks `asr_qwen`. The 43 empty free-response cards have no recogniser words at all.
