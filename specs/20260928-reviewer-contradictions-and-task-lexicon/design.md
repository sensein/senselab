# Reviewer contradictions, the task lexicon, the name kind cut and Safe Harbor

Owner, 2026-09-28, on r7 cards. Six decisions, one batch:

1. **Task-declared names are task content.** A name the task declares is never masked.
2. **A contradiction is flagged.** A reading that judges the redaction wrong but names no words keeps
   REDACT's masks and goes to a person.
3. **The reviewer must name its words.** A judgment asking for the redaction to change lists every word
   concerned.
4. **The reviewer iterates.** An unusable answer is fed back and asked again, bounded.
5. **The Cinderella themes are task content** ("it's a picture book … we can include the common
   cinderella themes in a list").
6. **A name is its proper nouns** (LOC "Florida Like"), and **the reviewer applies HIPAA Safe Harbor**
   ("reviewer should use safe harbor"), places included.

Items 1, 2 and 6's kind cut are fold-only. Items 3, 4, 5's residue and REVIEW's context, and Safe Harbor
in the prompt change what REVIEW reads or is told, so they reach a corpus only through a SPEECH replay
and a GPU re-review.

## 1. What the r7 Cinderella card showed

`sub-17578482-…_ses-A8D26790-…_task-cinderella-story`: reviewer `flagged`, original `clean`, redaction
`incomplete`, **empty proposal**; 13 unchanged masks over Cinderella ×5, prince ×4, midnight ×2, "next
day", Cinderella's; released with redaction.

Two defects, one policy gap:

- **REDACT's stimulus exemption missed the cast.** The pii entities carry `in_stimulus: true` and exact
  `word_ids`, but `_expected_exemptions` still found a finding's words by extent hull, and the hull check
  failed. It now reads the finding's placed `word_ids` (the fold already refuses a store without them).
  On the card, 22 of 25 findings are exempt afterwards; "midnight" and "the next day" are not declared
  names.
- **The fold never excluded declared names.** Residue `free_response` subtracts nothing, so the cast
  reached the detectors and the masks. The fold now treats any word inside a task-lexicon phrase as task
  content: excluded from every mask, counted in `task_words_n`, listed in `task_lexicon_ids`.
- **A reading that names nothing moved nothing.** Correct under the word-level rule (the reviewer must
  name words), but silent. It now raises `REVIEWER_NAMED_NO_WORDS` (`verdict.llm_contradiction_flags`),
  keeps the masks, and the ledger records `reviewer_named_no_words`.

r7 parquet: 75 readings flagged, original clean, no proposal entry; 73 still masked; 41 cinderella-story,
7 free-speech-v2, 7 random-item-generation, 5 story-recall-v2, 4 story-recall, 5 picture-description*.

## 2. The task lexicon

`task_lexicon.py` is one definition read by the residue (SPEECH step 7), REDACT's exemption units, the
fold's mask plan and REVIEW's context (`task_words`). A family's lexicon is its expectation row's
`expected_names` plus, where `stimulus.task_lexicons` lists the family, `data/task_lexicon/<family>.yaml`.
A phrase matches as a whole run; each token after the residue's normalisation, a trailing possessive or
plural, and the stimulus near-match (five letters or more, so "Sinderella" is Cinderella).

### cinderella-story

The stimulus is the wordless 25-picture sequence of the DementiaBank/AphasiaBank discourse protocol,
from Walt Disney's *Cinderella* (ISBN 0-7364-1296-4):
<https://talkbank.org/dementia/protocol/participant/sequences/Cinderella/index.html>. The sidecar
declares no stimulus text; the instructions name "the Cinderella storybook".

Sources, in the file's own header:

- **Core lexicon.** Dalton & Richardson (2015), AJSLP 24(4) S923–S938, and Dalton, Kim, Richardson &
  Wright (2020), Semin Speech Lang 41(1) 45–60: the 95-lemma Cinderella checklist, as shipped in
  github.com/rbcavanaugh/coreLexicon `R/sysdata.rda` (commit 5f025c75fbf1). Only story-content lemmas are
  kept: ball, cinderella, clock, dance, daughter, dress, fairy, father, glass, godmother, horse, midnight,
  mother, mouse, prince, pumpkin, shoe, sister, slipper, marry. Function words and generic verbs and
  adjectives (a, the, go, get, beautiful, …) are dropped.
- **Cast.** The storybook's named characters, `CINDERELLA_CAST`.
- **Time phrases.** The DATE_TIME-family detector texts on the 258 r7 cinderella-story stores that are
  story events: midnight 116, one day 36, twelve o'clock 19, the next day 13, one night 7, before
  midnight 8, all night 5.
- **Owner.** The common themes approved as task content (stepmother, stepsisters, pumpkin,
  carriage/coach, mice, glass slipper, ball, castle/palace, invitation, gown, wand, "happily ever after",
  "once upon a time", …).

Excluded on purpose: names outside the story that the detectors also found in retellings (Joan Didion,
DeSantis, Walt Disney, Princess Diana, Snow White), and kinship words used generically outside the
story's roles. The list: 78 entries (with the 20 declared cast names):

cinderella, prince, prince charming, charming, princess, fairy godmother, godmother, fairy, king, queen,
duke, grand duke, stepmother, stepsister, sister, mother, father, daughter, lady tremaine, tremaine,
anastasia, drizella, lucifer, gus, jaq, bruno, mouse, mice, horse, footman, coachman, lizard, pumpkin,
carriage, coach, glass slipper, slipper, glass shoe, shoe, ball, gown, dress, ballgown, wand, magic,
spell, clock, invitation, castle, palace, kingdom, staircase, stairs, steps, dance, dancing, marry,
married, wedding, happily ever after, once upon a time, midnight, stroke of midnight, twelve o'clock,
twelve am, before midnight, almost midnight, one day, next day, the next day, one night, that night,
the night, all night, next morning, the next morning, one evening, that evening..

### Other families

- **picture-description** (Cookie Theft etc.) has a published picture and published main-concept and
  core-lexicon lists (AphasiaBank: Cat in the Tree, Refused Umbrella, Broken Window; Richardson & Dalton
  2016/2019). Whether B2AI's picture-description options are those pictures is not stated in the
  sidecars; not implemented.
- **random-item-generation** asks for "items from a given category e.g., cities, animals"; the category
  is shown at recording time and is in no sidecar, so which items are task content is unknown per
  recording. Not implemented. Under Safe Harbor a listed city is a sub-state place; only knowing the
  category would make it task content.

## 3. The name kind cut

r7 `sub-28a1a382-…_task-free-speech-v2-1`: LOCATION "Florida", LOC "Florida", LOC "Florida Like"
(words Florida + like). The reviewer released "in Florida"; "like" stayed masked, because nobody named
it and the residue classifier calls it a content word.

- **SPEECH.** `pii.name_words_max` 3 → 1: a name-family finding of two words or more keeps its proper
  nouns (`finding_placement.cut`). A span with no proper noun keeps its run, since the recognizer may
  have written a name lower-case.
- **Fold (for stores written before).** Under a mask whose findings are all of `name_families`
  (`data/pii_category_families.yaml`: PERSON, LOCATION, ORGANIZATION) and which holds a proper noun, a
  word not in proper form is released (`kind_cut` on the ledger word, state `unmasked_by_trim`), unless
  another mask's non-name finding also covers it.

The first fold rule released every non-proper word under such a mask. On the reviewed r7 stores that
cut 922 words in 442 recordings, among them real names written lower-case or sentence-initially ("ben",
"John,", "Fletcher", "Canada,", "Louis,"). The shipped rule is narrower: only at a mask's edge, only a
word written all lower-case, and only one the transcript also uses outside every finding. That cuts 105
words in 81 recordings (the 52, and 10, ten 5, …), none of them a name.

The owner's Florida card is not cut by the fold. "like" occurs once in its transcript, and a
single-occurrence lower-case word cannot be told from a lower-case name without a dictionary. SPEECH's
cut does fix it: at `name_words_max: 1`, "Florida like" is cut to "Florida", so any replay writes the
finding on "Florida" alone.

## 4. The reviewer names its words, and iterates

The prompt now says: a judgment that the redaction is incomplete, that the original is clean while the
released text still removes words, or that the original carries something identifying, must list every
word concerned, quoted exactly; a place released without a reason is not an answer.
`answer_problem()` (in `redaction_review.py`, beside the parser) returns the problem as one sentence:
a judgment with no entries, a place release with no reason, or quotes not in the original. REVIEW's loop
sends it back as feedback over the same text until `max_iterations`. Each round records `feedback` and
`problem`, and the annotation records `converged` and the unresolved `problem`. The "clean original over
masks" test reads REDACT's own masks, not the loop's masking of the reviewer's earlier redact entries.

## 5. Safe Harbor

The reviewer applies 45 CFR 164.514(b)(2): the 18 identifiers (A)–(R) and the actual-knowledge condition
(b)(2)(ii). They are rendered into the prompt from `text/tasks/pii_detection/data/safe_harbor.yaml`, with
worked contrasts:

- "I grew up in Florida" (a state, releasable);
- "St. Petersburg" (a city, masked);
- a unique role in Florida (masked by the residual clause);
- "I'm 93" (an age over 89, masked; a story character's "ninety-three" is task content, not the
  speaker's age).

Each proposal entry names its Safe Harbor letter. The ledger records the Safe Harbor codes of every mask
and proposal (mapped from category families) and every release entry with its reason and letter.

A health condition is not one of the 18. The fold's CONDITION → human-review routing is unchanged.

The owner's place rule: a place is masked by default, and the reviewer may release it only where, weighed
with the rest of the transcript, it could not single the speaker out. That is Safe Harbor's (B) plus the
residual clause. Measured r7 inconsistency behind it:

- Florida: 155 masks (60 unmasked by the reviewer, 70 reviewer-proposed redaction, 25 masked with the
  reviewer silent);
- New York: 104 (55 / 19 / 30).

Voice prints are Safe Harbor (P); the released audio is itself a voice sample. That is outside what
masking a transcript can address and is not claimed here.

## 6. What the r7 readings show under the fold-only changes

The 15,193 reviewed r7 stores were re-folded in memory, with no writes to the tree and 0 errors, at the
batch's final code (job 24252618):

| | before | after |
|---|---|---|
| released without redaction | 13,256 | 13,288 |
| released with redaction | 1,422 | 1,389 |
| withheld | 515 | 516 |
| triage pass → flag | – | 71 |
| words masked | 6,040 | 5,440 |

- 32 recordings move from released with redaction to released without redaction, and 1 to withheld.
- The 71 new flags are `REVIEWER_NAMED_NO_WORDS`, on exactly the 75 readings the parquet showed (4 were
  already flagged): cinderella-story 41, random-item-generation 7, story-recall-v2 5, story-recall 4, …
  - A first version flagged every flagged reading with an empty proposal, 578 readings. 502 of them
    read the original as carrying identifying content and the redaction as complete, which agrees with
    the masks. The ground now needs `redaction: incomplete`, and `answer_problem` likewise asks for words
    after "carries_pii" only where the redaction is not complete.
- **cinderella-story** (239 reviewed): masked words 715 → 123; 195 released without redaction.
- **The owner's card** is released without redaction and flagged for review, with nothing masked. Its
  release ground reads "every mask REDACT planned hid only non-content words", which is wrong in
  wording: the words were task words. The zero-mask ground does not distinguish the two.

## 7. Cluster steps

- **Fold-only (items 1, 2, 6's kind cut): a re-fold of r7.** No REDACT re-run: the fold reads the stored
  findings' `word_ids`.
- **Everything else: a SPEECH replay and a full GPU re-review**, since every reading gets the new prompt
  and REVIEW's context gains `task_words`. The pieces:
  - the prompt rule and iteration (3, 4) and Safe Harbor;
  - the cinderella-story residue (5);
  - REDACT's word-id exemption;
  - `pii.name_words_max: 1`.

GPU re-review: all ~15,200 reviewed recordings, about 42 A100 slices, as r7. The prompt changes every
reading, so there is no smaller correct set. The SPEECH replay is all 62,550 on CPU; for
cinderella-story it removes the 78 lexicon entries before the detectors, which r7's residue did not.

## Health conditions are their own required part (r8 regression)

**What happened.** r8's prompt made HIPAA Safe Harbor the reviewer's whole rule and listed CONDITION
among the PROPOSAL categories "when it could identify the speaker". The model read Safe Harbor as the
test for that too, and since a condition is none of the 18 identifiers, it stopped naming them: readings
flagging CONDITION fell from 889 (r7) to 31 (r8), and condition spans held for human review from 383 to
15. The owner's policy is the opposite: every condition a speaker attributes to themselves goes to a
person, who judges rarity against what else is released.

**Change.** The answer gains a sixth, required part, `CONDITIONS`, placed before `PROPOSAL`: a JSON
array of every diagnosis, disease, symptom-as-condition, treatment, medication or procedure the speaker
attributes to themselves, `[]` when none, independent of Safe Harbor. CONDITION is removed from the
PROPOSAL categories. The parser appends each listed condition to the proposal as a `redact` entry of
category `CONDITION` (one representation in the store; a condition named in both parts is kept once), so
the fold, ledger, parquet, page and evaluations read conditions exactly as before and route them to
cohort/other human review. An answer without the part is fed back ("your answer had no CONDITIONS
part…") and read again, bounded by `max_iterations`; `conditions_answered` is recorded per round.

**Task content.** A condition entry every word of which is the task's own content — task-lexicon words,
or stimulus words outside the residue — carries agreement `task_content` and is held for no one.

**Page.** Only a `redact` entry with agreement `new` draws red on a word no mask hides; an entry agreeing
with the masks (or naming task content) leaves such a word unmarked, as the release leaves it visible
(r8 card sub-004d42e9… open-response: "A … or so … that" were drawn red while released).
