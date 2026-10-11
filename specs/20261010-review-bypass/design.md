# REVIEW and SECOND_OPINION record a bypass, with its reason

## What changed

`nothing_to_read` is gone from both nodes. Where the lexical residue (`transcript_texts(store)[0]`)
is empty and the node is enabled, REVIEW's `redaction_llm_annotation` and SECOND_OPINION's
`second_opinion_answers` now record `status: bypassed` and a `bypass_reason`:

| `bypass_reason` | the consensus transcript |
|---|---|
| `no_words` | carries no words |
| `non_lexical_only` | carries only bracketed, non-lexical or degenerate tokens |
| `task_content_only` | carries lexical words, every one task content (stimulus, instructions or family lexicon) |
| `non_lexical_task` | carries lexical words, no `pii_scan`, and the declared family is airway or voice |

A first version carried a fourth reason, `unscanned`, for lexical words with no `pii_scan`. It is gone; see
"Lexical words with no PII scan" below.

`disabled` is unchanged and still wins: the switch is read before the residue.

## Why

The owner (2026-10-10): "why nothing to read instead of bypass reviewer?" and "it has something to
read, just that it matches task content exactly". `nothing_to_read` described the r31 skip of
47,490 recordings as if they had no words; most have words, and the reviewer was bypassed because
every one of them is task content. The record now says the reviewer was bypassed and why.

## How the reason is derived

`review.bypass_reason` reads the word kinds `consensus_transcript()` already assigns (`WORD_SCANNED`,
`WORD_TASK`, `WORD_NON_LEXICAL`); it adds no classification of its own. No words gives `no_words`;
no `WORD_TASK` word gives `non_lexical_only`; otherwise `task_content_only`. The fourth reason,
`unscanned`, exists because `consensus_transcript()` labels every lexical word outside the residue
as task content, and with no scan there is no residue. With no scan, the declared family decides
(`task_speech.non_lexical_family`): an airway or voice family is `non_lexical_task`; any other family is not
bypassed at all (below).

## What did not change

- The fold. Every check in `vocabulary.py` reads `status in ("clean", "flagged")` or
  `status == "ok"`, so `bypassed` is not a reading, exactly as `nothing_to_read` was not.
  `vocabulary_test.py::TestABypassFoldsAsTheOldNothingToRead` folds both and compares every field
  of the record except the two carried statuses.
- No model is asked, no server started, no cache key touched; `PROMPT_VERSION` and
  `QUESTION_SET_VERSION` stand.

## What a reader sees

- A bypassed annotation carries `failure: null`. `nothing_to_read` carried the failure "the transcript
  is empty", so `recording_vectors.llm_failed` was true for it and is now false: a bypass is not a
  failure.
- `recording_vectors` gains `llm_bypass_reason` and `second_opinion_bypass_reason` (schema 26 → 27;
  the viewer's facets carry both). The fold's `second_opinion` record carries `bypass_reason`.
- The review page (extract 10 → 11) names the bypass and its reason; its filter offers `bypassed`.
- `triage.tsv` has no review-status column, so `triage_table.yaml` is unchanged.

## Backfill over r18

The r31 reviewer pass skipped the 47,490 rows of `r31/review/rows_nowords.jsonl`. They are written
by a CPU-only pass of `extend_llm_review.py` and then `extend_second_opinion.py` at this change's
commit (`r18d_20261010/common_r33.sh`). Both drivers already start their model lazily: REVIEW's
vLLM worker is only reached from `_rounds`, behind the residue check, and the second-opinion driver's
`LazyAsk` opens Ollama on its first ask, recording `server_started` in each slice summary.

## Lexical words with no PII scan (owner, 2026-10-10)

The owner: "a recording with lexical words and no PII scan is a missing measurement, not a valid reason to
skip the reviewer", refined the same day for families that ask for a sound: "for non-lexical task families we
don't need to know whether the speech contains PII ... every lexical word outside task text is masked ... the
recording is flagged for review".

### What the nodes record

| family | consensus | `pii_scan` | REVIEW / SECOND_OPINION |
|---|---|---|---|
| any | no words, or only non-lexical tokens | any | `bypassed`, `no_words` / `non_lexical_only` |
| any | lexical words, empty residue | present | `bypassed`, `task_content_only` |
| airway, voice (`non_lexical_family`) | lexical words | absent | `bypassed`, `non_lexical_task` |
| every other family, or none declared | lexical words | absent | `absent`, `absent_reason: pii_scan_missing` |

The `absent` record asks no model, associates no model agent, and carries a `failure` sentence. The fold
flags the operational ground `pii_scan_missing` (`vocabulary.KEY_PII_SCAN_MISSING`, in
`OPERATIONAL_GROUND_KEYS`) when REVIEW's annotation or SECOND_OPINION's measurement carries that
`absent_reason`; `decision_reasons.yaml` (v11) maps it to `not_measured`, so the verdict is `review` and
`run_status` is `incomplete`, the same route as `owning_branch_input_absent`.
`TestNotMeasuredIsIncomplete` folds both carriers and the `non_lexical_task` bypass.

### No PII scan is added for airway and voice families

The first brief asked for a scan of every family's lexical words. The owner's refinement replaced it: an
airway or voice task asks for no words, so any lexical word outside its task text is masked on the
non-task-speech path and the recording is reviewed, whatever the word says. The path, from 2026-10-10
(`specs/20261007-task-events-in-background/design.md`, "Speech outside the task"):

- The owning branch (AIRWAY in align mode, VOICE per phonation reading) writes `task_speech_reading`
  (`task_speech.task_speech_of`): every lexical consensus word that is not bracketed, degenerate,
  unconfirmed or a vocalisation, less the declared task text and the recognisers' misreadings of the task's
  own events.
- `data/task_speech.yaml` `words_min: 1`: one such word flags `speech_in_task` (or `other_speaker_in_task`
  where COHORT attributes the runs to another speaker), reason `off_task_speech` / `other_speaker`.
- `redact._non_task_speech` masks every timed such word (`words_min` 1), so the release is `redacted` on
  `non_task_speech_masked`, or `withheld` on `non_task_speech_untimed` / `non_task_speech_extensive`.

DDK families are owned by SPEECH, which scans them, so they are lexical here.

### Where lexical words go unscanned, measured over r18d

Read-only over all 62,550 r18d stores at 3e7097ca (`r31/unscanned_count/`, a 16-task CPU array with
`senselab.utils.fastio`): 1,324 stores carry lexical consensus words and no `pii_scan`. Every one is an
airway or voice family with SPEECH `declined`, and every one was written `bypassed` / `unscanned` by the r33
backfill. None is a lexical family, so the patch round owes no PII scan (`r31/unscanned_stems.txt` is empty).

| family | stores |
|---|---|
| glides-low-to-high | 573 |
| glides-high-to-low | 468 |
| respiration-and-cough-fivebreaths | 56 |
| respiration-and-cough-breath | 48 |
| maximum-phonation-time | 46 |
| respiration-and-cough-cough | 33 |
| respiration-and-cough-threequickbreaths | 25 |
| maximum-phonation-time-v2 | 12 |
| respiration-and-cough-v2-hardcough, voluntary-cough, respiration-and-cough-v2-threebreathsnose | 9 each |
| respiration-and-cough-v2-threebreathsmouth, high-to-low | 7 each |
| prolonged-vowel, respiration-and-cough-v2-breath, respiration-and-cough-v2-threebreaths | 6 each |
| breath-sounds | 4 |

Every airway or voice store with a lexical word holds a `task_speech_reading` (0 without one). Whether the
path covers those words, over the 1,325 airway or voice stores with SPEECH declined and a lexical word (the
1,324 above and one SPEECH-declined store a scan reached):

- 310 carry task-speech words. All 310 flag `speech_in_task` / `other_speaker_in_task`; none leaves a timed
  task-speech word outside the ledger's `non_task_speech_ids`; 231 are `review` with release `redacted` on
  `non_task_speech_masked`, 79 are discarded on other grounds.
- 1,015 carry none, each a single word: 740 vocalisations (`is_non_lexical(vocal_task=True)`), 235 task
  content (event misreadings or the family lexicon), 10 declared task text, 30 `unconfirmed`. They release
  `as_is` on `non_lexical_task`.

So no family and no route leaves a lexical word outside the task text unmasked and unflagged, except the
30 `unconfirmed` words below.

### Open: words the task-speech path does not count

`task_speech_of` leaves out an `unconfirmed` word (one recogniser read it, none located it), by the
2026-10-10 decision that it "never withholds on its own". Read literally, "every lexical word outside task
text is masked" would include it, and an unconfirmed word has no timing, so it could only withhold. The
recognisers' misreadings of the task's own events are also exempt, beside the declared task text. Neither is
changed here; over r18d the unconfirmed case is 30 stores, one word each.

### Surfaces

`recording_vectors` schema 27 → 28 (the `llm_bypass_reason` / `second_opinion_bypass_reason` vocabulary
changed); the fold's `second_opinion` record carries `absent_reason`; the review page (extract 11 → 12) names
`non_lexical_task` and renders a `pii_scan_missing` absence as a missing measurement, not a load failure.
