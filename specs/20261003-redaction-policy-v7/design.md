# Redaction policy v7

Owner, 2026-10-03. Replaces the HIPAA Safe Harbor reading of identifiers (A)-(C) the reviewer applied
since 2026-09-28 (prompt v3-v6), and the condition human-review route of 2026-09-27. Identifiers
(D)-(R) -- numbers, contacts, codes -- are unchanged and still listed to the reviewer from
`data/safe_harbor.yaml`.

## The owner's rules, verbatim

> names
> redact all names by default
> do not redact celebrity names like "Ray Bradbury". Only names for people related to the patient. We'd probably have to manually review name tags to do this, which is fine, I think it would be very quick
> do not redact family members / relative relationships
> dates: remove all absolute references to dates, *including* year. For the statement "I had COVID in 2021" -> it's better to just remove 2021. my opinion on releasing years is it's better to redact dates everywhere, because more than 1 mention allows triangulation of an exact date. whereas if you redact it everywhere, you can consciously release a date once in the structured data and be much safer, and much more usable for researchers.
> my rules would be:
> 2021 -> redact
> October -> redact
> summer -> redact
> Halloween -> redact
> Monday -> debatable, we can keep redacting it if that's what's currently done
> 2-3 weeks ago -> keep
> last year -> keep
> locations: I think removing states would be better, but it's not legally required.
> conditions: we should not withhold anything based on condition
> organization: any specific organization. so "voice center" -> keep, but "USF voice center" -> redact.
> employment: there are some employment mentions - In general I think removing specific organizations as above will handle any sensitive employment. For example some people are in specific military organizations, or attending specific schools which are identified as LOCATION currently, but only some are "red" locations (see sub-09f16959-f984-42c4-a304-4f6ddb4e189c for an example)
> ages: debatable; my default is to redact, but I see they are currently kept. legally we can do either (acknowledging >89 stuff is already handled).
>
> and the model way over tags the Spanish records.

## Decisions taken with the owner

| Question | Owner's answer |
|---|---|
| Countries | "countries won't help identify accents. usually it's people talking about where they or family member are. countries can be kept, unless there is the possibility of triangulation from other unredacted information" |
| Condition words | Released unmasked (recorded on the ledger), never withheld |
| Names review | Mask every name; the reviewer may propose a public figure's release; only a human's approval releases it; a recording with a masked name flags for review (not withheld) |
| Weekdays | Released |
| Ages | Owner's default: redact |

## The rule, as the fold applies it (`nodes/redact.py:mask_plan`)

Everything below takes effect on a plain re-fold (`scripts/extend_refold.py`); only the reviewer's
own judgements need the prompt-v7 re-review.

1. **Always masked, whatever the reviewer says** (`MaskWord.locked`):
   - a date element: a year (four digits 1800-2099, `'98`, a spelled year), a month (English
     capitalised, Spanish any case), a season ("spring"/"fall" only beside a season context word), a
     holiday, a day of the month beside a month -- `redaction_policy.date_positions`;
   - an age: a policy age pattern (`redaction_policy.age_positions`), or any content word of a finding
     a detector labelled AGE that is not released as a time or duration;
   - a state, province or equivalent (`redaction_policy.state_positions`), and every capitalised word
     of a LOCATION-family finding that is not a country standing alone;
   - a person's name: every capitalised word of a PERSON-family finding, until a human approves it.
   Words the policy always masks that no detector finding covered get a `policy` mask of their own,
   so a recording REDACT never ran on still releases a masked copy (`POLICY_MASKS_ONLY`).
2. **Released by kind, whatever the reviewer says** (`RELEASED_BY_KIND`, `MaskWord.kind`):
   - a time of day, a weekday, a relative day, a length of time, an unquantified unit after a
     relative modifier ("last year") -- `time_by_kind`, `data/time_release.yaml`;
   - a kinship word, English or Spanish -- `redaction_policy.kinship_positions`;
   - a country that is the whole of a place name; "United States Marine Corps" is not one.
3. **A listed health condition** is never masked and never withholds (`RELEASED_CONDITION`). The
   cohort/other kind is recorded on the ledger only. `verdict.condition_categories` replaces
   `verdict.llm_human_review_categories`; a prompt-v6 reading's CONDITION `redact` entries are read as
   conditions.
4. **Person names**: `redaction.name_approvals` maps a recording's file stem to the names a human
   approved; an approved name is `UNMASKED_BY_APPROVAL`. A reviewer `release` entry on a locked name is
   recorded (`name_release_proposed`) and moves nothing. A recording keeping a name masked flags under
   `PERSON_NAME_AWAITS_REVIEW` (`verdict.person_name_review_flags`); the release is unchanged.
5. **Countries the reviewer asks to hide**: a `redact` entry naming a country masks it
   (`COUNTRY_MASKED`) without proposing more, so it does not withhold.
6. **Organisations**: unchanged in the fold; the reviewer judges specific (mask) against generic
   (release). A capitalised ORGANIZATION word is protected from the trim where
   `verdict.trim_protected_categories` lists the category.

## Spanish

Diagnosed on the r10 free-speech extract (`free_speech_page_20261002_r10/extract.jsonl`):

| | recordings | with a mask | masked words | lower-case among them |
|---|---|---|---|---|
| English (`en`) | 11,524 | 907 (7.9%) | 1,829 | 435 |
| Spanish (`es`) | 177 | 88 (49.7%) | 382 | 351 |

Category of the Spanish masked words: PERSON 308, OTHER 41, LOCATION 33. The source is presidio's
English spaCy NER (and the `rules/ner` NAME pass) tagging ordinary Spanish words as PERSON ("pero" 12,
"muy" 10, "estado" 8, "ánimo" 8, "porque" 8, "está" 7), which the trim then kept because none is an
English function word. Two fixes at the fold, both language-aware and both on a plain re-fold:

- Spanish closed-class words join `residue.FUNCTION_WORDS` (`data/function_words_es.yaml`), so the trim
  releases them in any recording.
- In a recording whose sidecar declares a language other than English, a word of a name-family or
  OTHER mask that is not written as a proper noun is released (`MaskWord.kind_cut`). Proper nouns,
  dates and ages stay masked.

The reviewer is told the transcript's declared language and that the English detectors over-tag
Spanish words. The detectors themselves are unchanged: a Spanish NER model in the PII venv would need a
re-scan of every Spanish recording, which the fold-level fix makes unnecessary for the over-tagging.

## The reviewer, prompt v7

`redaction_review.PROMPT_VERSION = 7` (cache identity moves with it, so no v6 answer is reused). The
prompt states the rules above, lists Safe Harbor (D)-(R), asks for conditions for the record only (a
separate `conditions` part on the annotation, never a proposal), and keeps the six judgements, the
eight-part answer and the `answer_problem` loop. `answer_problem` now also feeds back a `release` of
something the policy always removes (`never_released`) and a name or country released without a reason.

## Measured before and after (in-memory re-fold, 2026-10-03)

The r9 stores re-folded in memory at the v7 code under the r11 config (`second_opinion_on.yaml`), with the
v6 reviewer readings they hold and the sidecar's language as the hint; nothing was written to the tree.
Sample: every Spanish free-response recording on the r10 page (177), the owner's sub-09f16959 (5), and 120
English free-response recordings carrying a mark (seeded 20261003).

| | Spanish | English sample |
|---|---|---|
| recordings with a standing mask, v6 -> v7 | 88 -> 43 | 13 -> 45 |
| masked words, v6 -> v7 | 465 -> 109 | 21 -> 136 |
| withheld -> released | 6 of 13 | 23 of 25 |
| flagged for names review | 19 | 18 |

Spanish: the over-tagged function and common words are gone ("pero", "muy", "estado", "porque"); what
stays masked is seasons (otoño 11, invierno 11, verano 10, primavera 3), holidays (navidad 3) and names
awaiting review (Sandra Bullock, Dios, Jesús). English: years ("nineteen ...", "two thousand ..."), seasons
(winter 7, summer 6), holidays, ages and names the v6 reviewer had released (Tom Friedman, Nancy Drew,
Holden Caulfield), which now wait for a human's approval. Withholdings that were only conditions release.

sub-09f16959: free-speech-3 keeps "United States Marine Corps" and "Tennessee Highway Patrol Cadet School"
both masked (v6 released the first: one red, one green); free-speech-2 masks "thirty-six years old" (age)
while "thirty-three years ago" stays shown; the picture description shows "mom" (kinship); free-speech-1
shows "today".

## Every withholding names its ground (2026-10-04)

The r12 parquet held 106 withheld recordings with a null `release_ground`. Every one was a REDACT
`fail` whose verification re-scan of the redacted transcript still read a finding (`unremediable`:
PERSON 90, DATE_TIME 9, NAME 8, LOCATION 7, MISC 4, DATE 1, LOC 1) that no reviewer reading cleared;
`_release_from` returned ground None wherever REDACT itself decided. The release stays withheld (a
name surviving the redacted copy is what v7 masks by default); the fold now records
`REDACT_VERIFY_FOUND` for a re-scan fail and `REDACT_UNRESOLVED` for any other REDACT fail or flag,
both in `RELEASE_WITHHELD_GROUNDS`. Only a REDACT pass with a standing mask still carries no ground.
