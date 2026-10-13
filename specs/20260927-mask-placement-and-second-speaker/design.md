# Mask placement, reviewer agreement, a second speaker, and the task's own text

Owner, 2026-09-27, on three r6 cards:

- `sub-00053adb…` story-recall: "what is person+dat_time+misc. doesn't make sense. also doesn't make
  sense the reviewer would keep most of it. also in this particular case one could infer that perhaps
  two people are conversing."
- `sub-004d42e9…` open-response-questions: one mask over the whole answer, labelled
  `PERSON+DATE_TIME+LOCATION+LOC+DATE`, keeping "brother", "lives", "died" masked; withheld because
  the reviewer "proposed hiding more" -- Wisconsin, Australia and Alan, which the masks already hid.
- `sub-01401050…` cinderella-story: "prince" released in one place and masked in another, because
  the reviewer's quotes named some occurrences and not others.

And on the review page: "the +1/-1/flag annotation doesn't make a lot of sense. perhaps it should
just match the semantics: withhold, with redaction, without redaction, those three would be mutually
exclusive and a separate flag for review." On context: "make sure to add the instructions/task as
context to the reviewer. is it not in the task information metadata?"

Two commit sets. The **fold-only** set (1-4) changes VERDICT and the page and needs a re-fold, no GPU.
The **replay** set (5-7) changes SPEECH's residue and REVIEW's input, so it needs a replay and a GPU
re-review.

## 1. What `PERSON+DATE_TIME+MISC` was

SPEECH step 7 places each detector finding on the transcript by matching its text against the
scanned tokens (`nodes/speech.py`, `_locate`). A finding whose text matches no token run -- a name
inside a possessive, "Alan" against the token "Alan's"; a reading from the second recogniser's
haystack -- is noted `pii_unlocated (<category>)` and placed over the **whole transcript**, marking
every word. REDACT then padded every finding by 250 ms and merged overlapping extents into one span
whose category is every member's joined with `+`. On the story-recall card the unlocated PERSON
covered 0.48-61.7 s: every word, the examiner's instructions included. The trim released
non-content words and the reviewer's quote released the participant's retelling; what stayed masked
was the rest of the instructions, which nobody named.

The fold cannot re-place the finding: SPEECH stores its category and extent, not its text.

## 2. The rule (fold, `nodes/redact.py:mask_plan`)

- **One mask per located finding**, on the residue words that carry its category's mark inside its
  own extent. Findings covering exactly the same words are one mask. A mask is labelled with its
  category's **family** in the reviewer's own vocabulary (`data/pii_category_families.yaml`: PERSON =
  PERSON, NAME; LOCATION = LOCATION, LOC; DATE_TIME = DATE_TIME, DATE, AGE; ...), so a span shows one
  category; the ledger keeps every member category.
- A word's categories for the proper-noun protection are those of the located findings placed on it,
  not every mark on it: an unlocated finding marks every word, and those marks say nothing about any
  one word ("A" in "Okay A week ago" read as a capitalised PERSON).
- **An unplaced finding masks nothing.** It is recorded on the ledger with a state:
  - `placed` -- a reviewer `redact` entry of its family places it on the words the reviewer quoted
    (whether or not a mask already hides them); those words are masked, as a `reviewer`-source mask;
  - `cleared` -- the reviewer read the original as clean;
  - `open` -- a reading exists and settles neither: a triage flag for review
    (`UNPLACED_FINDING_OPEN`), release unchanged;
  - `unread` -- no reading at all: withheld and flagged (`UNPLACED_FINDING_UNREAD`).
  Masking the whole transcript was the fail-safe; it protected nothing the reviewer had not already
  judged and hid the words the reviewer never thought to name. Where no reviewer read the text the
  recording is withheld, which is the same protection without the false precision.
- A REDACT extent that hides exactly the kept words of the masks inside it and reaches no other word
  stands as REDACT planned it, relabelled with the mask's family, so REDACT's own copy is still the
  one released (`same_extents` compares the audio, not the label).
- **Agreement is not hiding more.** A reviewer `redact` entry whose content words a mask already
  hides, or that places an unplaced finding of its family, agrees (`AGREED_MASKED`, `AGREED_PLACED`)
  and is set aside before any residue or human-review test (`vocabulary.deciding_reading`). Only a
  `NEW` entry -- a content word no mask hides, or a quote that cannot be placed -- withholds.
- **A term, not an instance.** A reviewer `release` entry that names a finding word releases every
  occurrence of the same content-word term (surface and family) in the recording, unless any
  `redact` entry places on that surface. The ledger records `propagated` per word.
- A reading with `speakers: more_than_one` flags the recording for review
  (`REVIEWER_HEARD_SECOND_SPEAKER`, `verdict.llm_second_speaker_flags`), except where diarization's
  own `dominant_speaker_share_min` gate already failed. It does not move the release.

Why no gate fired on the story-recall card: the free-response gate set checks
`extent_dominant_speaker_share` against 0.9, and diarization read the examiner and the participant as
one speaker (share 1.0). The reviewer, reading the words, did not.

## 3. The page

Three mutually exclusive release choices -- withhold (`w`), with redaction (`d`), without redaction
(`o`), the graph's own `Release` values -- and an independent flag for review (`f`). The export is
version 4 and carries `release` and `flags`; the row-mark collection is gone. A mark shows one
category, its mask's own; every category that names the span is on `data-c` for the popup.

## 4. Measured on the 15,210 reviewed r6 stores (fold-only set, re-folded in memory)

Re-folded in memory at the fold-only set's tip (before the audio-only relabel, which moves the last
two joined labels), 0 errors:

| | before | after |
|---|---|---|
| withheld | 892 | 460 |
| -- reviewer proposed hiding more | 623 | 128 |
| -- human review, cohort / other condition | 124 / 76 | 179 / 89 |
| -- REDACT | 69 | 64 |
| released with redaction | 1,407 | 1,681 |
| released without redaction | 12,911 | 13,069 |
| triage flags | 2,016 | 1,972 |
| final masks with a joined label | 1,181 recordings | 2 |
| released transcript text changed | | 1,393 |

- Old masks hid 1,860 content words no finding covered, in 158 recordings -- the whole-transcript and
  chained masks.
- Unplaced findings: 473 recordings (PERSON 478, DATE_TIME 32, ID 11, LOCATION 10): 412 cleared, 42
  placed, 87 open (80 recordings flagged for review), 0 unread (every one was read).
- Reviewer `redact` entries: 951 agree with a mask, 14 place an unplaced finding, 538 are new. Of the
  623 "reviewer proposed hiding more" withholdings, 495 move: 229 to released-trimmed, 174 to
  released-reviewer-unmasked-some, 83 to human review (their remaining new entries are all
  conditions), 8 as REDACT planned, 1 without redaction.
- Term propagation: 226 occurrences in 127 recordings ("frog", "day", "boy", "Cinderella", "summer");
  none is a term any `redact` entry names.
- A second speaker: 379 flags from 395 `more_than_one` readings; diarization had already flagged the
  other 16. Families: free-speech-1 36, story-recall 30, random-item-generation 19, word-color-stroop
  18, free-speech-2 18, picture-description 17, story-recall-v2 15, cinderella-story 13.

The three cards: story-recall keeps one DATE_TIME mask ("five minutes", the examiner's) and no
whole-transcript mask; the open-response card is released with redaction, its masks on "a week",
"ago", Wisconsin, Australia and "that week" and the unplaced PERSON placed by the reviewer's "Alan";
the Cinderella card is released without redaction, every "prince" and "midnight" released.

Still open: a detector span that overruns its words stays a mask on all of them. On the
open-response card a DATE finding covers "Australia, my brother Alan's wife, had died that week", so
"brother", "wife" and "died" stay masked as DATE_TIME and "Alan's" reads DATE_TIME although the
reviewer placed PERSON on it. The reviewer can release such words; the fold does not second-guess a
located finding's extent.

## 5. The reviewer never saw the task

`scripts/extend_llm_review.py` built each recording's hint for the re-fold and called REVIEW without
it, so every r5 and r6 reading carried `task_context: {'task': ...}` and nothing else; `run.py`'s
full pipeline passes it. The driver now passes the hint to REVIEW. The context adds the task's
`speech_type` and `instructions` beside its stimulus, and the prompt says that words that are the
task's own stimulus, or that its instructions ask for, are not identifying for being said, and that
instructions read aloud by someone else are a sign of a second speaker.

## 6. The task's own text

`AudioHints` carries `instructions` and `speech_type` as typed fields (routing reads the typed field;
`metadata["speech_type"]` retired). `b2ai_hints.build_hint` reads both BIDS sidecars and resolves each
recording's text through b2aiprep's registry, vendored at 8c43256 under `data/b2ai_task_registry/`:

- instructions: the sidecar's, unless the registry's curated file corrects that registry task (per
  recording where it keys them so) and the recording is English;
- stimulus: the sidecar's `stimulus_text`; for an English recall task whose sidecar leaves it empty,
  the flat descriptions' prompts.

An alias two versions share is resolved by the recording's own name: `story-recall` is v1 (the
grandfather passage) though the registry's `alias_index` maps it to v2 (the boy and the frog).

Cross-check over all 62,550 r6 recordings' sidecars against the vendored registry:

- Every recording has both sidecars and non-empty instructions.
- The curated instructions equal the sidecar's for every English recording the curated file covers
  (24,643 recordings resolve to a curated task). They differ on 262, all Spanish sessions -- 5 per v1
  family, 24 per v2 family -- whose Spanish text an English correction would replace; hence the
  language gate.
- story-recall: 884 of 889 sidecars carry the grandfather passage and none carries the frog story as
  its stimulus; 5 Spanish sessions carry the v2 frog story's **instructions** ("una historia sobre un
  niño y una rana") on a v1 recording. story-recall-v2: 637 of 660 carry the frog story; the other 23
  are Spanish.
- respiration-and-cough-v2 (3,494 recordings): the flat descriptions carry prompt text the sidecars
  leave empty by design; nothing is borrowed for a non-recall task.
- breath-sounds (326): the sidecar already carries the curated text, not the flat file's unrelated
  "5-second auto-stop" protocol.

## 7. A recall's story is task content (`residue.py`, `RECALL`)

A free-response family declaring a stimulus under `speech_type: recall` is read with a new residue
method: a word the story holds -- its content words, in any order, within the residue rule's variant
tolerance -- is the task and never reaches the detectors. A name or place the speaker adds stays
residue. Cinderella declares no stimulus and is unchanged.

Of the 3,304 content-word findings on the 1,496 reviewed r6 story-recall recordings, **2,334 (71%)
are passage content** under this method (860 recordings): DATE_TIME 963, PERSON 679 ("boy", "frog"),
NAME 394, AGE 213 ("ninety-three"), LOCATION 50.

## 8. What each set needs on the cluster

- Fold-only (1-4): re-fold all 62,550, parquet, page. No GPU.
- Replay (5-7): replay all 62,550 (SPEECH's residue for recall tasks; the replay retires every REVIEW
  reading), the review manifest, a full GPU review of every recording with residue (about 15,200;
  the owner asked for every reading to come from one prompt), re-fold, parquet, page. Staged under
  `/orcd/scratch/bcs/002/satra/triage_r7_20260927/`, with `RUN.md`.

## 9. SPEECH places each finding on its own words (replay set)

Owner, 2026-09-27: fix the placement in SPEECH rather than compensate in the fold. `nodes/speech.py`
step 7 matched a finding's text to the scanned tokens by exact normalised token equality, placed a
finding that matched nowhere over the whole transcript, and covered every word between the first and
last matched positions, which bridged words the scan never read. It now (`finding_placement.py`):

- matches the finding's text to whole scanned tokens by their joined keys -- edge punctuation, a
  possessive and internal hyphens dropped -- at every occurrence (`Alan` on `Alan's`, `ninety-three`
  on `ninety three` or `ninetythree`); where no run matches, a token whose hyphen pieces hold the
  text as consecutive whole pieces (`year-old` in `93-year-old`); never a part of any other word;
- covers exactly the matched words, never the words between them the scan skipped;
- cuts a date, time or age finding to its temporal words (`data/temporal_words.yaml`: units, calendar
  and relative-time words, digits, number words and their compounds), each kept run starting and
  ending on one; a name, place or organisation finding longer than `pii.name_words_max` (3) to its
  proper nouns; a run with nothing of its kind stands;
- records each finding's `text` and `word_ids`; a finding that places nowhere is an
  `unplaced_findings` record on `pii_scan` with its text, and writes no `pii` entity.

The fold (`mask_plan`) reads `word_ids` and `unplaced_findings` directly; the extent-based
re-placement and the `pii_unlocated` note reading are gone, and a store without `word_ids` is refused
(r6 cannot be re-folded under this code; r7 replays it). A reviewer entry places an unplaced finding
only where REDACT ran to mask it; otherwise it is a new proposal. The page joins a mark to its
findings by word id.

Measured by re-scanning the stored residue of the 8,516 reviewed r6 recordings with findings
(detectors re-run on CPU, no ASR, no REVIEW; the reviewer column applies the stored release quotes
as an approximation of the fold -- the r7 readings will differ):

| | old placement | new placement |
|---|---|---|
| findings (all haystacks) | 61,301 | 61,301 |
| placed over the whole transcript | 1,263 (473 recordings) | 0 |
| bridged over unread words | 2,680 (799 recordings) | 0 |
| unplaced | -- | 9 (PERSON, texts like "you", "i", "haven", empty) |
| words covered | 67,972 | 33,341 |
| content words covered | 37,239 | 29,053 |
| content words left after the stored release quotes | 12,888 | 10,773 |

Cuts: 9,126 runs cut, 1,870 of them dropping a content word (3,156 words): DATE_TIME 1,529, PERSON
226, LOCATION 115. Examples: DATE "Australia, my brother Alan's wife, had died that week" -> "week";
DATE "last year that I also have PVCs AFib started probably ten years ago" -> "last year",
"ten years ago"; DATE_TIME "forty-six seconds Water." -> "forty-six seconds". The remaining 7,256
cuts drop only a leading or trailing function word ("the past two weeks" -> "past two weeks"),
which the mask trim released anyway. `name_words_max` 3: a name span of four or more words was cut
226 + 115 times, and no name of three words or fewer is touched.
