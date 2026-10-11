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
| `unscanned` | carries lexical words, but SPEECH wrote no `pii_scan`, so no residue was computed |

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
as task content, and with no scan there is no residue: calling those words task content would be
the same mislabelling this change removes. It is the existing `detector_state == "unscanned"` read.

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
