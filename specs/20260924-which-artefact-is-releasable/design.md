# The release axis names which artefact may be handed on

## The owner's direction, 2026-09-24

> "for the free speech reviewer, and for verdict add an option to release-w/o-redaction and
> release-w-redaction. do a replay to generate the verdict."

## 1. What was wrong with `releasable`

The axis shipped as `releasable | withheld | nothing_to_redact | not_assessed`. Its name answered a
yes/no question — *may this be handed on?* — but the corpus holds two different artefacts per
recording, the original and REDACT's redacted copy, and the two do not have the same answer.

`releasable` is reached on exactly one condition: **REDACT left a `pass`**. REDACT's own `why` for
that outcome is *"every finding redacted; the redacted transcript re-scans clean"*. So `releasable`
has always meant *the redacted copy is clean and the original is not* — it is the **restrictive**
state, and it was wearing the permissive name. A consumer asking "may I use the audio as recorded?"
read `releasable` and got the wrong answer on 12,046 recordings.

`nothing_to_redact`, meanwhile, was the state where there is no redacted copy at all, so the only
artefact is the original — the genuinely permissive case, wearing a name about redaction.

The axis was therefore not about confidence and never had been. It was about **which artefact**, and
it did not say so.

## 2. The vocabulary

```python
class Release(Enum):
    WITHOUT_REDACTION = "release_without_redaction"
    WITH_REDACTION = "release_with_redaction"
    WITHHELD = "withheld"
    NOT_ASSESSED = "not_assessed"
```

Four values, total and exclusive over the one question a consumer has: *which artefact of this
recording may I hand on?*

| value | what it permits | what it does not assert |
| --- | --- | --- |
| `release_without_redaction` | the recording **as recorded**, and anything derived from it. A reading of its lexical content ran and found nothing a redaction would remove | that the recording is non-identifying. A voice identifies its speaker by construction; this axis is an authority on lexical content only |
| `release_with_redaction` | **only** REDACT's redacted artefact. The original located findings and had them removed, so handing the original on would hand the findings on | nothing about the store, which holds the unredacted consensus transcript by design and is append-only |
| `withheld` | neither artefact | — |
| `not_assessed` | nothing | — |

`release_with_redaction` permits **exactly what `releasable` permitted** — every byte of the old
rule, renamed so that the name says which file it was always about. No recording gains permission
from this change.

## 3. `nothing_to_redact` does not survive, and it does not collapse whole

This is the decision the vocabulary turns on, and the argument runs both ways.

**For collapsing it whole into `release_without_redaction`:** redaction is a lexical operation. A
recording with nothing lexical to remove has no redacted copy, so the only artefact is the original
and the original is what is handed on. On that reading all 45,854 `nothing_to_redact` recordings are
`release_without_redaction`, and the change is a pure rename with zero movement — attractive, and
the safest-looking option because nothing moves.

**Against:** `nothing_to_redact` is not one population. Its four grounds split cleanly into *a
reading cleared this recording* and *nothing read this recording*:

| ground | did anything read the lexical content? | n (r3) |
| --- | --- | ---: |
| `SCAN_FOUND_NOTHING` | yes — the scan ran over the transcript | 11,526 |
| `NOTHING_BEYOND_STIMULUS` | yes — every word was matched against the task's own stimulus | 13,810 |
| `NO_LEXICAL_WORD` | yes — SPEECH produced a transcript and it carried no lexical word | 1,421 |
| `NO_TRANSCRIPT` | **no** — SPEECH never ran | 19,097 |

Under the old axis that split did not matter, because the axis described REDACT's artefacts and
there were none in any of the four rows. **The owner's change voids that premise.** The moment the
axis can say *release the original*, "did anything read the original?" becomes the load-bearing
question, and `NO_TRANSCRIPT` cannot answer yes. Those 19,097 are overwhelmingly recordings routing
sent to AIRWAY or VOICE — a breath, a cough, a held vowel — where the ruleset's gates concluded no
speech branch applies. That is a reading of the *task*, not of the audio's lexical content, and an
off-script aside at the head of a breath recording is exactly the case it does not cover.

**Decision: it splits.** The three grounds where a reading cleared the recording become
`release_without_redaction`. `NO_TRANSCRIPT` becomes a ground for `not_assessed`.

**This is the conservative direction on every row.** Nothing gains permission: `releasable` and
`withheld` are unchanged in substance, the three cleared grounds were already outside `withheld`,
and `NO_TRANSCRIPT` moves from a determination to an admission of ignorance. The axis cannot be used
to release anything it could not release yesterday.

### The 2026-09-22 change is preserved, not reversed

`verdict.md` § "The release fold" shrank `not_assessed` from 44,623 to 1 and argued specifically
that *"A SPEECH withheld by a critical failure reads `NO_TRANSCRIPT`, not `not_assessed`."* That
argument rested on one sentence: *"there is no artifact and no transcript either way, so the release
axis has nothing to withhold"* — true while the axis described REDACT's artefacts, false now that it
describes the original too.

What the 09-22 change was actually about is **separating "nothing was found" from "REDACT did not
run"**, and that separation survives intact: the four grounds are unchanged as grounds, still
recorded, still distinguishing the populations. What moves is which side of the axis one of them
sits on. `not_assessed` returning to 19,098 is the axis's subject having changed, not the rule
having regressed — and it is now an honest 19,098 rather than an unexamined 44,623, because the
41,000-odd recordings a reading did clear stay cleared and say so.

`NO_TRANSCRIPT`'s wording changes with its side of the axis. It read *"SPEECH did not run, so no
transcript exists for a redaction to read"* — a statement about the redaction. It now reads
*"SPEECH did not run, so nothing read the recording for content a redaction would remove"* — a
statement about what is unknown, which is what a `not_assessed` ground must be.

### Why not a fifth value

A fifth state for "nothing was examined" was considered and rejected. It would be a *reason* for
`not_assessed` promoted to a value, and `release_ground` is the field that carries reasons — the
same argument `verdict.md` already makes for keeping `discard_ground` off the triage axis. Four
values answer the consumer's question; the ground decomposes it.

## 4. The table

Read in order; the first row that matches wins. The last row is the fall-through, so the table is
total.

| condition | `release` | `release_ground` |
| --- | --- | --- |
| REDACT left a verdict, `pass` | `release_with_redaction` | — |
| REDACT left a verdict, `flag` or `fail` | `withheld` | — |
| live `pii` findings and no REDACT verdict | `not_assessed` | `REDACTION_OWED` |
| SPEECH errored | `not_assessed` | `SPEECH_UNREAD` |
| SPEECH left no lexical count and did not complete | `not_assessed` | `NO_TRANSCRIPT` |
| SPEECH completed and left no lexical count | `not_assessed` | `SPEECH_UNREAD` |
| SPEECH read no lexical word | `release_without_redaction` | `NO_LEXICAL_WORD` |
| the scan was declined — every word is in the stimulus | `release_without_redaction` | `NOTHING_BEYOND_STIMULUS` |
| the scan ran and found nothing | `release_without_redaction` | `SCAN_FOUND_NOTHING` |
| SPEECH read lexical words and recorded no scan | `not_assessed` | `SCAN_UNRECORDED` |

`release_ground` is `None` on exactly the two rows REDACT itself decided, as before.

`RELEASE_DETERMINED_GROUNDS` is renamed `RELEASE_WITHOUT_REDACTION_GROUNDS` and loses
`NO_TRANSCRIPT`; `RELEASE_UNKNOWN_GROUNDS` gains it. Three grounds stand behind
`release_without_redaction` and four behind `not_assessed`.

## 5. Nothing moves a recording toward release without a human

The graph reads **no reviewer record on this axis**, and this change adds no path by which one could
reach it.

- REDACT's optional LLM re-read still annotates the triage axis only, under
  `verdict.llm_redaction_flags`. Unchanged; see [`llm-check.md`](../20260817-triage-workflow-dag/llm-check.md).
- The free-speech reviewer — whether a person or the LLM pass over the review page's rows — writes
  into the page's own `localStorage` and its JSON export. **`fold_file_verdict` does not read that
  file and has no parameter for it.** A reviewer calling a withheld recording clean is a reading
  someone may act on; it is not the act, and the graph does not perform it.
- Every movement this change produces is `nothing_to_redact -> not_assessed` or a rename. No
  recording leaves `withheld`.

If a reviewer record is ever to reach the axis, it arrives as a declared input with provenance and
its own row in the table above, and that is a separate decision with its own derivation.

**That decision was taken on 2026-09-24, in one direction only.** Owner: *"verdict still decides,
and we can weigh the verdict towards the LLM reviewer."* REVIEW's `redaction_llm_annotation` is a
measurement with provenance, and `fold_file_verdict` reads it onto this axis through exactly one
declared parameter, `_release_from(..., reviewer_withholds=...)`, resolved from
`verdict.llm_redaction_withholds` and defaulting to False. It may only **tighten**: a reviewer that
read residue turns a REDACT `pass` into `withheld`, and no reading of any kind moves a recording out
of `withheld`. The section's title is therefore still exact — nothing moves a recording *toward*
release without a human — and `vocabulary_test.py` asserts both halves: the parameter set is closed,
and the movement is one-directional.

## 6. The reviewer action on the free-speech page

`scripts/free_speech_review_page.py`. The page already carries three reviewer collections — the
per-finding verdict, the per-row mark, the per-recording note. The release decision is a **fourth**,
not an overload of the third: a row mark says *how this row reads to me* and a release decision says
*which artefact I am willing to hand on*, and a reviewer can hold the second without the first.

| | |
| --- | --- |
| vocabulary | `release_without_redaction` (key `o`) and `release_with_redaction` (key `d`) — the axis's own value strings, so an export row joins to a verdict without a mapping |
| keys | `o`/`d`, card-scoped like the row marks. Disjoint from `1234` (finding verdicts), `+ - f` (row marks) and `j k J K` (movement); the test that holds those sets apart covers this one |
| state | `store.release`, keyed by card stem, `{v, t}` — the same shape as `store.triage` |
| storage | the same `senselab.fsreview.v1` namespace, as a fourth collection |
| export | its own key in the payload, beside `findings`, `recordings` and `triage`. `version` 2 → 3 |
| filter | `#dec`: `any / decided / undecided /` the two values |
| progress | `… · N of M rows given a release decision` |

The pipeline's own release state keeps its whole namespace on the page — `data-rel`, `.rel-f`,
`.r-*`, `RELEASE_ORDER`, the `release` fieldset. The reviewer's decision takes `data-dec`, `.dec-f`,
`.d-*` and `#dec`, so the page never renders a human's decision and the graph's in the same colour.
A reviewer's decision changes nothing the graph concluded, and the card goes on showing both.

## 7. Every reader that moves in this change

| file | change |
| --- | --- |
| `triage/vocabulary.py` | the enum, the two ground tuples, `NO_TRANSCRIPT`'s wording, `_release_from`'s table, the `FileVerdict` docstring |
| `triage/corpus_report.py` | the release tally's ordered vocabulary |
| `triage/recording_vectors.py` | the release categorical's level list; `SCHEMA_VERSION` 4 → 5 |
| `triage/replay_diff.py` | the axis's declared values |
| `triage/nodes/report.py` | the rendered release line |
| `triage/nodes/figure.py` | the release band's colours and order |
| `triage/viewer/` | the decoder's schema version and the release axis's levels |
| `scripts/triage_audio.py` | the printed verdict line |
| `scripts/triage_viewer_fixture.py` | the fixture's release values |
| `scripts/free_speech_review_page.py` | `RELEASE_ORDER`, the `.r-*` CSS, and the new reviewer action |
| `specs/20260817-triage-workflow-dag/verdict.md` | the two-axes table, the release fold, the product block |

## 8. The measurement

[`measurement.md`](measurement.md) — the r3 baseline, the prediction this design makes from it, and
the differential over the replay that tests the prediction.

## 7. A non-lexical task is cleared, not held

Added 2026-09-25, after the first corpus measurement of the split.

The split put **19,097 recordings** at `not_assessed` on one ground — `NO_TRANSCRIPT`, "SPEECH did
not run". Every one is a task the ruleset declined SPEECH for because it carries no lexical content:
respiration and cough (7,512), maximum phonation (2,880), glides (2,616), prolonged vowels. Under
the pre-split vocabulary they read `nothing_to_redact`; the rename moved them to "the graph cannot
say", which is 27% of the corpus parked behind a word that asks a person to look.

Owner, 2026-09-25: *"if it's non-lexical then they should not be held back by not_assessed, they
should be passed through as cleared as non-lexical, or flagged for not containing lexical items if
they were routed to the speech branch for a speech task."*

Two changes, and they are deliberately not the same change.

**Cleared.** `NON_LEXICAL_TASK` joins `RELEASE_WITHOUT_REDACTION_GROUNDS`. `_release_from` takes
`speech_declined`, read from `routes[SPEECH] == declined`, and a recording whose branch the ruleset
declined is releasable without redaction. The ruleset's decision *is* the reading: there is nothing
a redaction could remove from a task that never asked for a word. A branch that was **routed** and
still left no count keeps `NO_TRANSCRIPT` and `not_assessed` — that is a gap in the record, and the
distinction between "nothing was asked for" and "something was asked for and did not arrive" is the
whole of this section.

**Flagged.** A task the ruleset routed to SPEECH is a task that asks for words. SPEECH running over
it and reading none is the task not having happened, and the release axis calling it releasable is
true without being the whole of it. `NO_LEXICAL_ITEM_PRODUCED` is a flag ground on exactly that
shape: `lexical_words_n == 0`, SPEECH completed, SPEECH routed. It does not withhold — there is
nothing in the recording to redact — it makes the silence visible on the triage axis where a person
looks.

### What it moves

About 19,097 recordings from `not_assessed` to `release_without_redaction`, and about 1,421 gain a
triage flag they did not carry. Both are pure fold: no reading changes, no audio is touched, and the
corpus is brought to it by a re-fold pass rather than a replay.
