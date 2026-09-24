# Rebuilding the page on r3

The page currently on disk was built from `triage_rerun_20260923`. Three graph changes merged at
`7edcfcf2` are computed at run time and stored on entities, so only a fresh pass makes them visible.
This records what the page had to change before it could read such a pass honestly, and the baseline
the rebuild will be measured against.

## What the page had to change

### 1. `not_run` means something else now

`redact.py` used to test the detector outcome before it tested whether the reviewer was enabled, so
a withheld recording recorded `not_run` whether or not the reviewer was switched on. The ladder is
now `disabled` → `not_run` → run, and `not_run` means one thing: **the detectors marked nothing, so
there was nothing to read back**.

The sentence this page carried — that the reviewer *could not have run* because REDACT withholds
first — was correct for the old code and is a false statement about an r3 run. It is gone, and a
test asserts it stays gone.

Six readings are now kept apart, and every one that is not a reading says outright that nothing was
concluded, so no reader can infer a review happened:

| state | what the page says |
| --- | --- |
| `disabled` | switched off; no review attempted; not a verdict about the recording |
| `not_run` | enabled, but the detectors marked nothing, so there was nothing to read back |
| `absent` | it tried and could not load — not the same as finding nothing |
| `clean` | it ran and flagged nothing |
| `flagged` | it flagged these categories |
| unrecorded | REDACT left no annotation; nothing is known |

`detector_outcome` is new on the annotation and is now shown, because it is the only record of which
decision a reading was taken beside. A `clean` beside a detector `fail` is a **disagreement** — the
reviewer read the redacted text and saw nothing the detectors still saw — and the page says so
rather than letting it read as confirmation.

A reviewer facet joins the rail, with a `did it actually run` shortcut over `absent`/`clean`/
`flagged`. On r3 it will read `disabled` corpus-wide, because `redaction.llm_check.enabled` is still
false and the replay is CPU-only. The facet exists for the GPU pass, not for r3.

### 2. The declaration and the answer came apart

`expected_names` puts a task's own cast on its expectation row, and `declared_carrier` now feeds
`in_stimulus` and the exemption pass rather than only the scan gate. So a task can declare **no
prompt text** and still have its findings checked.

The page's stimulus section inferred "nothing could be checked" from `expected_speech_declared`
being false. That inference is about to be wrong for `cinderella-story`, which declares a cast of 20
and no prompt text, and right for `picture-description`, which declares neither. Only the answer
tells them apart, so the section now reports the tri-state tally the recording's findings actually
returned — matched, checked and unmatched, uncheckable — and names the sources it was checked
against.

### 3. An extract now says which graph it came from

Reading the old rows under the new wording would print that false `not_run` sentence over a whole
corpus. `extract` writes a header line carrying `EXTRACT_VERSION`, `load` reads it, an extract
without one reads as version 2, and the page shows a banner saying the reviewer section does not
describe the run that produced it. This is the one failure mode worth structural protection: a page
that is confidently wrong about provenance is worse than one that says nothing.

## The baseline

From `scripts/free_speech_review_page.py census` over the r2 extract. `compare` diffs two extracts
key by key, so the rebuild is measured rather than described.

| | r2 |
| --- | ---: |
| recordings | 11,701 |
| findings | 22,319 |
| marks | 15,595 |
| `in_stimulus: true` | 5,477 |
| `in_stimulus: false` | 12,305 |
| `in_stimulus: null` | 4,537 |
| withheld | 2,972 |
| releasable | 3,719 |
| nothing to redact | 5,009 |
| not assessed | 1 |

Reviewer: `disabled` 3,719, `not_run` 2,972, no annotation 5,010 — the mixture the old ladder
produced. On r3 this should collapse to `disabled` almost everywhere.

**A unit caution.** Bracketed findings have two counts and they are not interchangeable:

| | count | of which `PERSON` |
| --- | ---: | ---: |
| bracketed **marks** (a mark is a run of adjacent words of one category) | 542 | 340 |
| bracketed **words** | 1,007 | 723 |

The 1,007 / 723 figures are words. Comparing them against a mark count would show a fall that never
happened.

Per family, r2:

| family | recordings | findings | marks | unchecked | in stimulus | withheld |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cinderella-story` | 258 | 2,989 | 1,964 | **2,989** | 0 | 179 |
| `free-speech` | 3,074 | 4,581 | 3,509 | 0 | 298 | 1,090 |
| `free-speech-v2` | 2,120 | 5,275 | 3,563 | 0 | 555 | 864 |
| `open-response-questions` | 199 | 173 | 156 | 0 | 3 | 45 |
| `picture-description` | 889 | 748 | 549 | 748 | 0 | 99 |
| `picture-description-option1` | 373 | 308 | 218 | 308 | 0 | 42 |
| `picture-description-option2` | 329 | 436 | 283 | 436 | 0 | 42 |
| `productive-vocabulary` | 2,910 | 2,101 | 1,524 | 56 | 810 | 277 |
| `story-recall` | 889 | 2,561 | 1,767 | 0 | 1,493 | 262 |
| `story-recall-v2` | 660 | 3,147 | 2,062 | 0 | 2,318 | 72 |

## What to expect, and what would falsify it

- **`cinderella-story`'s 2,989 unchecked findings should largely become answerable.** The cast is
  declared and 80.5% of its findings sit within one edit of it. If `unchecked` does not fall sharply
  there, the declaration is not reaching `in_stimulus`.
- **`picture-description` ×3 should not move at all** — 1,492 unchecked findings across 1,591
  recordings, by design, because no text list can cover an image. Movement there would mean
  something is matching that should not.
- **Withheld should fall**, since a finding the task itself accounts for is exempted rather than
  redacted, and `withheld` is REDACT's verification outcome. `cinderella-story`'s 179 is the place
  to look first.
- **The near-match window should raise `in_stimulus: true` modestly across the families that already
  declare prompt text** — `story-recall-v2`'s 2,318 and `story-recall`'s 1,493 — at a measured cost
  of 1.3 false admits per 10,000 name findings.
- The DDK families are not in this page's scope: all ten free-response families are `SPEECH`, and
  `diadochokinesis-*` is `SYLLABLE_TRAIN`/`SYLLABLE_SEQUENCE`. `buttercup`'s 51,066 `null` findings
  will not appear in this comparison, and looking for them here would be a category error.

## Not built yet

The r3 tree was still being written when this was prepared — array `23604010`, 24 of 250 slices
outstanding, 1,480 of 1,527 participants present. Building from a partial tree would produce a
census that looks like a corpus and is a sample, so the build waits.
