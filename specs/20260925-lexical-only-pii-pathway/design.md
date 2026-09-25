# The PII pathway reads lexical residue only

Owner, 2026-09-25: *"fix the residue string. in fact the check for what goes into PII detection
pathway within speech should be specific to lexical content. any non-lexical content should not
even go there."* A follow-up the same day: read-aloud tasks reach the pathway **only** where their
residue lies outside the target words, and an off-script disclosure inside or around a read passage
must still be scanned.

## What was wrong

- **Nothing subtracted the task.** SPEECH's scan read the whole lexical transcript, and REVIEW read
  the whole transcript (`transcript_texts`, then `nodes/redact.py:957`). The only gate was a word
  test: the scan was skipped only when *every* word sat inside `declared_content()`, then
  `nodes/speech.py:2352`. One novel token and everything was scanned: syllables, vowels, fillers.
- **DDK could not pass the word test.** All 7,993 DDK sidecars carry an empty `stimulus_text`, so
  the haystack was one token cut from the family name (`branches.py:1126`). The match was exact
  below 5 characters. Recognizers write `Ta-ta-ta` (one token `tatata`), `ta` for `tuh`, `paraca` for
  `pataka`, `parika` for `puhtuhkuh`, CJK syllables `咔 啪`, Devanagari `पतक`. The result: 7,306 of
  7,984 DDK recordings were scanned, almost all findings were PERSON/NAME on syllables, and the
  reviewer flagged 1,676 of the 2,233 it read in the finished slices while proposing a redaction on
  only 18.
- **Vocal tasks declared nothing.** `declared_content` never read an expectation row's `tokens`, so
  loudness (`hey`) and prolonged-vowel (`1, 2, 3 aah`, from the sidecar instructions) had an empty
  haystack. 1,584 of 1,601 loudness recordings and all 1,281 SPEECH-routed prolonged-vowel
  recordings were scanned.
- **Read-aloud matched per token, not per sequence.** Any recognition variance outside the edit
  tolerance opened the scan and then exposed the *whole* passage: `heart stump` for `hearts jump`,
  `consul` for `council`, `It's` for `It is`, `treetop top`, restarts. Stroop colour words were
  tagged NAME. Read-aloud recordings scanned: 4,992 of 19,258.
- **The fold read a release proposal as residue.** `_reviewer_found_residue` withheld on
  `redaction == "incomplete"`, which the reviewer also writes when it means *over*-redacted. Over
  the 37 finished slices, `llm_redaction_withholds` on would have newly withheld 1,845 recordings.
  1,630 of those proposed only `release` and 122 proposed nothing. Only 93 proposed a `redact`.

## The design

`senselab.audio.workflows.triage.residue` computes **one residue per recording**: the lexical words
that are neither non-lexical nor what the task asked for. SPEECH computes it once and records the
residue word ids on its `pii_scan` measurement (`residue_word_ids`, `residue_method`,
`residue_content`, `non_lexical_words_n`, `task_words_n`).

- **SPEECH** scans only the residue: each haystack, consensus and per recognizer, is built from
  residue positions alone. The scan runs only if the residue is non-empty and carries at least one
  word outside the closed-class list (`FUNCTION_WORDS`). A residue of `the` / `a` alone carries
  nothing a detector could find. Numbers are not closed-class.
- **REDACT** verifies, and re-plans, over the residue words only (`residue_words`). The released
  transcript is still the whole transcript, with findings masked.
- **REVIEW** reads `transcript_texts`, which renders the same residue words. Where the scan was
  declined, or SPEECH never ran, the residue is empty and REVIEW records `nothing_to_read` without
  contacting a model. This supersedes `specs/20260924-reviewer-over-every-transcript/`, whose
  premise (a reader for what the detectors never saw) the owner replaced with *the reviewer reads
  the string the detectors read*.

Non-lexical, for every family (`is_non_lexical`):

- bracketed markers, including ones with trailing punctuation;
- vocalisations and fillers (`uh`, `umm`, `hmm`, `mm-hmm`, `ahh`, `aaah`, `oh`, `ooh`);
- single-character Chinese interjections (`啊 嗯 呼 哈 …`);
- truncated fragments of at most two letters (`wa-`);
- in a vocal task only, a vocalisation run: one vowel letter repeated, or a run of `m`, with
  optional `h` (`aaah`, `ahh-ahh`, `hee`, `eee`, `uhhh`, `mmm`, `hmm`, `ha ha`). Two distinct vowel
  letters, or a `y` or `w`, make a word, so `Amy`, `Emma`, `Mia`, `May`, `Hawaii` and `my` are
  lexical. No word or name list is needed: the rule is on the letter shape. The first version took
  any token of vowels plus `h m w y`, which dropped exactly those names (owner, 2026-09-25).

Task content, by method (`residue_method`):

| method | families | task content |
|---|---|---|
| `syllable_train` | the ten `diadochokinesis-*` | A token spelled only from the template's consonants and their voicing/flap letters (`p→pb`, `t→tdr`, `k→kcgqx`, `er→r`) plus vowels (`aeiouyw`) and non-initial `h`. A non-Latin token of at most 4 characters. A token periodic in a unit of ≤4 characters. A token recurring `train_repetitions_min` times. |
| `vocal_task` | VOICE and AIRWAY families, and loudness / loudness-v2 | The expectation row's literal `tokens`, aligned (`hey`; `one two three`), and every vocalisation. |
| `stimulus_alignment` | read tasks with a stimulus text; also any undeclared family whose hint carries a prompt | The transcript is aligned to the prompt tokens with `align_pair`. A matched word is the task. A substitution or insertion is the task when it is a *variant* of a stimulus word in a window of `insertion_window` words either side: a near match, a similarity ≥ `variant_similarity_min`, a prefix fragment, the same Soundex-class consonant skeleton (≥2 classes, the homophone test: consul/council, game/gain, spilt/spilled), a clitic base (`it's` → `it`), a compound join or split, or a variant of the joined pair (`heart stump` ↔ `hearts jump`) for a word of at least `near_match.exact_below` letters. A shorter word
is not excused by pairing it with a neighbour: `heymy` against `heyhey` scores 0.67, and that is
how `My` beside `hey` was set aside. An inserted word equal to any stimulus word is the task (a re-read sentence). A word in the row's declared `vocabulary` (Stroop colours, new) or `expected_names` is the task in any position. |
| `free_response` | free response, item lists | Nothing: every lexical word is residue. |

The invariant: a disclosure over any take is residue. `pa pa pa my name is Alice Smith pa pa` gives
exactly `my name is Alice Smith`. The rainbow passage with `my name is Alice Smith` inserted keeps
`Alice`/`Smith`. `The game runs in Brooklyn.` over a different declared sentence is all residue.
`pii_interlock_test.py` and `residue_test.py` pin these.

The residual risk, stated rather than hidden: a name spelled only from a DDK task's own letters
(`Bob` in a /pa/ take, `Kate` in /ka/) cannot be told from a realisation of the task in text. The
acoustic decode's per-repetition spans, not the transcript, are what could separate them.

## Config, and how each value was chosen

`stimulus.residue` in `data/config/default.yaml`. The replay was the new gate over the stored r4
transcripts: 43,450 SPEECH-routed recordings, the words extracted from each store, and the hint
rebuilt with the corpus `hints.py`.

**`variant_similarity_min` = 0.66.** One edit per three characters. The sweep, with window 2 or 3
and the skeleton rule in place, gave the following recordings scanned:

| θ | total scanned | harvard | caterpillar | rainbow |
|---|---|---|---|---|
| 0.75 | 16,021 | 1,436 | 363 | 302 |
| 0.67 | 15,985 | 1,401 | 363 | 301 |
| 0.60 | 15,550 | 1,137 | 330 | 245 |

- **0.75 → 0.67** accepted only morphological and recognizer variants (`dreamed`, `waiting`,
  `terpiller`, `carapillar`).
- **0.67 → 0.60** began accepting unrelated function-word swaps (`there`, `this`, `which`, `men`).
- **0.66, not 0.67:** `looks`/`looked` sits at exactly 2/3.

**`insertion_window` = 3.** At θ 0.67, harvard scanned 1,442 with W=2, 1,401 with W=3, 1,380 with W=4
and 1,358 with W=6. The gain past 3 is under 2% per step, and every step widens what an
inserted name can be excused by.

**`train_repetitions_min` = 3.** R=2 read a word said twice as a train (a name repeated). R=4 left
more DDK scanned (/ka/ 70 against 52 at R=3).

## Replay: recordings reaching the detectors, old → new

`new` is the first version of the residue; `new'` is the current one, with the vocal-task rule
narrowed and the short-word pair rule removed.

| class | recordings | old | new | new' |
|---|---|---|---|---|
| DDK | 7,984 | 7,306 | 497 | 497 |
| sustained / airway | 21,314 | 2,246 | 645 | 663 |
| loudness | 1,601 | 1,584 | 47 | 48 |
| read-aloud | 19,258 | 4,992 | 1,512 | 1,944 |
| free response | 12,362 | 12,091 | 12,061 | 12,061 |
| **total** | 62,519 | 28,219 | 14,762 | 15,213 |

Read-aloud per family, old → new': harvard 3,054 → 991; caterpillar 588 → 262; rainbow 485 → 203;
word-color stroop 407 → 278; cape-v 322 → 159; cape-v-v2 136 → 51.

The narrowed vocal rule adds words to 203 sustained/airway recordings and 7 loudness recordings.
They are glide-spelled words it used to drop: `you`, `me`, `we`, `my`, `yeah`, `yay`, `meu`. The
pair rule's removal adds words to 836 read-aloud recordings, all short misreadings or asides:
`tea`, `Mac.`, `pier pier`, `Rows pays`, `What's mom on eggs,`.

What still reaches the pathway is off-target by construction:

- **Asides:** `Okay, when it says go, can hold it.`; `I don't know what that is`.
- **The wrong sentence read:** `Peter keeps keep the check`, whose declared sentence is
  `My mama makes lemon muffins`.
- **DDK residue:** examiner speech (`Perfect.`, `kind of stuff can talk`) and real-word misreadings
  (`call`, `come`, `particular`).

The Stroop residue is mostly the participant's commentary. Free response is unchanged apart from
fillers.

## Other fixes in the same change

- **The fold's residue rule** (`vocabulary._reviewer_found_residue`). A reading is residue only
  where the reviewer flagged it **and** proposed at least one `redact` entry. On the 37 finished
  slices that turns 1,845 new withholdings into 93.
- **`_derived_ran`** (`nodes/verdict.py`). REVIEW concludes with its live annotation. It writes no
  verdict and no report, so it read `errored` on every re-fold. An activity whose every output a
  later pass retired was superseded, not attempted: the buttercup row's REDACT read `errored` from
  the pre-replay pass's `plan`/`apply`/`verify`. Neither value was read by any decision. The fold
  takes REDACT's live verdict in `_release_from`, and `ran` is consulted only for PREPROCESS,
  routing, SPEECH and the routed branches. So the defect was in the record, not in a release.
- **`extend_llm_review.standing`.** A replay under the packaged config writes REVIEW's own
  `disabled` annotation into every store. Counting it as a standing reading would have made the
  review pass after a replay report `present` on every row.
