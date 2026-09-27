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

**Known risk.** `FUNCTION_WORDS` holds `may` and `will`, so a name spelled as a closed-class word
("May", "Will") that a detector masked is unmasked by the trim. The owner's instruction covers words
the finding itself caught; the count on r6 is in §6.

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
in one of those categories withholds under `REVIEWER_NEEDS_HUMAN_REVIEW`, and the triage axis flags
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
