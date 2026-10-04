# Redaction policy v8

2026-10-04. Builds on policy v7 (`specs/20261003-redaction-policy-v7/design.md`), which stays in force
except where this note changes it.

## What the owner said

On the v7 review page (`free_speech_page_20261004_r12`):

> still seeing redactions like this "[OTHER] man,", "LOCATIONRoman Empire.", 'OTHERdoing [UH]
> OTHERbrisk" ... don't understand why such things are being marked for redaction.

> also why "Star Wars" is PERSON and redacted

> redaction should take task into account. in tasks like productive vocabulary, the target word
> should be considered.

On the second opinion:

> remove questions that don't need answers or need to raise flags

## Where the masks came from (measured, r12 page, 11,701 recordings, 4,305 final masks)

- **"man", "doing", "brisk", "gardener", "overusing".** A `MISC` finding from `rules/rare_role`, the
  rule that marks an uncommon occupation or activity, or a PERSON tag from gliner or presidio on a
  lower-case word. A lower-case word in a non-name family is no name, so v7's name trim never reached
  it, and the reviewer's v7 reading did not release it. Detector masks on words written with no capital
  letter: 941 masks in 691 recordings. Most are the v7 date and age locks (518 presidio DATE_TIME
  seasons and the like), which stay. The rest are released by rule 1 below: 155 presidio DATE_TIME
  ("year", "hot", "last", "day"), 107 PERSON ("said", "kills", "teacher", "ninety-four"), and 66
  OTHER.
- **"Roman Empire", "Colosseum", "Pillars of the Earth".** A LOCATION finding below a country is
  locked under v7 whatever the reviewer says. In a sample of 100 locked LOCATION masks, about 30 named
  no place connected to the speaker: historical ("Roman Empire", "Central Europe"), fictional or a work
  title ("New York City" in a book's plot, "Pillars of the Earth", "Boys in the Boat"), a landmark or
  general knowledge ("Colosseum", "a New York winter", "California poppies", "Broadway"), task content
  ("Lima, Peru" defined in a vocabulary task), or no place at all ("Botox", "AI", "Nice"). About 70
  were places the speaker lives in, travels to, or is treated in.
- **"Star Wars".** A PERSON finding, locked under v7 until a human approves it. In a sample of 100
  locked PERSON masks, about 50 were no person: work titles ("Harry Potter", "Forrest Gump", "Spider Man
  No Way Home", "The Nightingale"), ordinary or garbled words ("Butter", "Battle", "Loyal", "Denie"),
  and characters in the task's picture or story ("Little Tommy", "Mother Alice", "Cinderella"). About
  20 were public figures and about 25-30 people connected to the speaker.
- **Seasons in free-speech-v2-1.** That task asks "What is your favorite season (winter, fall, summer,
  or spring) and why?". The v7 season lock masked the participant's answer, the words of the question
  itself: 418 masks carried `in_stimulus` true.
- **Productive vocabulary.** The target word reaches the reviewer as `asked_to_say` from the sidecar
  (`stimulus_from: sidecar`, e.g. "gladiator", "clef"). Masks in productive-vocabulary recordings: 441
  words in 347 recordings under v7.

## The rules

1. **No mask on a word nobody writes as a proper noun.** A detector finding's word written with no
   capital letter is released (`released_not_proper`), in every language. Kept: a number, an ordinal or
   a decade word ("twenty-second", "fifties"), a state name in any case, a word a v7 date, age or state
   rule locks, a word a reviewer `redact` entry names, and the lower-case words of a name finding that
   holds a capitalised word -- the v7 name trim decides those, so "ben" in "ben Fletcher" stays.
   Applied at re-fold. The ASR consensus carries casing: 1.5% of mid-sentence words are capitalised,
   and the 431 page transcripts with no capital at all are one- to six-word answers.
2. **Places that identify nobody.** A reviewer `release` entry may let a place below a country through
   with `place_reason` one of `historical`, `fictional`, `public_landmark_or_general_knowledge`,
   `task_content`. A state goes only as `historical` ("the Civil War in Virginia"). A place the speaker
   connects to themselves or their people stays masked whatever the reason. Needs the v8 re-review.
3. **Names that are no person.** A reviewer `release` entry may relabel a removed name with `relabel`
   one of `work_title`, `brand_or_product`, `organization`, `place`, `other_non_person`. A work title,
   a brand and a non-person are released. An organization follows the specific-versus-generic rule. A
   place follows rule 2. Only names the reviewer leaves as a person wait for a human's approval. A
   relabel or place reason applies to every occurrence of the same term, as a release does. Needs the
   v8 re-review.
4. **Task content.** Every word of the task's own texts -- the stimulus or target word the sidecar
   declares, and the task's instructions -- is task content no mask covers, inflections included
   ("Gladiators" for "gladiator", "ninetythree-year-old" for "ninety-three years old"). Applied at
   re-fold. Content the reviewer must judge -- a definition's own words, a picture's figures, a story's
   characters -- is guided per family by `src/senselab/text/tasks/pii_detection/data/task_guidance.yaml`
   (productive vocabulary, picture description, story recall, the read passages, open response). The
   prompt carries the family's guidance, and its digest joins the reviewer's and the second opinion's
   cache keys.
5. **The answer check.** `answer_problem` rejects a relabel or a place reason outside its set, a place
   below a country released without a place reason, and a state released for any reason but
   `historical`.

Prompt version 8.

## Second opinion, question set 2

Three questions, each of which can raise a disagreement flag: `other_voice`, `instructions_spoken`,
and `policy_identifier_present`. The last is defined by this policy: a name of a person connected to
the participant (not a public figure, a fictional character or a work title), an absolute date element,
a place below a country connected to the participant, a specific named organization, an age, a contact
or ID number. Relationship words, weekdays, relative times, countries, conditions and task content do
not count. The reviewer's answer to it is any mask standing after its reading is folded, or a `redact`
entry proposing more. `named_diagnosis` and the Safe Harbor question are gone: conditions no longer
affect release, and the Safe Harbor question flagged nothing.

The defect this replaces: the `named_diagnosis` comparison read the reviewer as "no" unless a v6-style
CONDITION `redact` proposal existed, and since v7 lists conditions in their own part, every listed
condition read as a disagreement. In r12, 647 recordings carried a `named_diagnosis` disagreement, 643
of them with conditions listed, and for all 647 it was the only disagreement.

An opinion to an earlier question set, or from other weights, no longer counts as present, so the
second-opinion run after the v8 re-review asks again. Run order: v8 reviewer, then question set 2,
then the full re-fold.

## Effect at re-fold (rules 1 and 4, in memory over the r12 page, v7 readings)

An in-memory mask plan at the v8 code over all 11,701 page recordings, each against its own v7 ledger;
nothing was written, and there were 0 errors.

| | v7 | v8, rules 1 and 4 |
|---|---|---|
| masked words | 5,798 | 4,657 |
| recordings with a mask | 2,559 | 2,048 |

- **Rule 1 freed 266 words in 227 recordings:** "day" 34, "year" 13, "last" 6, "man" 5, "girl" 5, "misdiagnosed", "gardener", "restaurant", "said".
- **Rule 4 freed 879 words in 536 recordings:** the free-speech-v2-1 seasons ("summer" 342, "winter" 224, "fall" 139, "spring" 67, and the Spanish seasons), and "ninety-three-year-old" from the story-recall stimulus.
- **5 words newly masked:** number words a v7 trim had let through.
- **Productive vocabulary:** 441 masked words in 347 recordings became 395 in 317. What stays masked is mostly capitalised: misheard or garbled words written as names ("Kean", "Battle", "Denis", "Stratter", "Darly"), places ("Africa", "Middle East"), and names ("Trump", "Larry"). Those wait on the v8 reviewer's relabels and place reasons. A cut-off attempt at the target ("Gladie-" for "gladiator") is not matched as task text.
- **Rules 2 and 3** (the reviewer's place reasons and relabels) take effect only after the v8 re-review.
