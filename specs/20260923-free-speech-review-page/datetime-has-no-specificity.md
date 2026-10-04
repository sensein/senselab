# `DATE_TIME` fires on temporal expressions, not on dates

Measured 2026-09-23 over the 11,701 free-response recordings of
`triage_rerun_20260923/out`, from the page's own extract. Counts and shapes only; no detected
string appears in this document.

## The claim

`DATE_TIME` is the second most common finding in the corpus — 7,283 findings, 6,397 marks, on 3,844
of the 11,701 recordings. **Only 3.5% of those marks carry a calendar anchor.** The rest are
temporal expressions of one kind or another, and a temporal expression is not an identifier.

## What the marked spans are made of

A mark is a maximal run of adjacent words sharing one category set, which is the unit the page
shows and the reviewer judges. Classified by what tokens the run actually contains:

| shape | marks | share |
| --- | ---: | ---: |
| bare time noun, no quantity | 2,907 | 45.4% |
| quantified duration — a number and a time unit | 2,299 | 35.9% |
| no temporal token at all | 865 | 13.5% |
| **calendar-anchored — month, weekday, year, or clock time** | **222** | **3.5%** |
| number words only | 103 | 1.6% |
| digits with no calendar cue | 1 | 0.0% |

The classifier is lexical and deliberately generous to the detector: "calendar-anchored" fires on
any month name or abbreviation, any weekday, any `19xx`/`20xx`, any `d/d` pair, `o'clock`, `am` or
`pm`. A separate cue regex counting only those same anchors independently reaches 237 of 6,397,
3.7% — the two agree.

So **96.5% of `DATE_TIME` marks contain nothing that fixes a point on a calendar.** The 13.5% with
no temporal token at all is the sharper end of the same thing: those marks are not temporal by any
lexical test, and are the direct analogue of `[UH]` under `PERSON`.

## Shape and concentration

- Length: median 12 characters, p90 24, p99 41. A long tail of 65 marks over 40 characters and 31
  over 100, the longest 622 — a 622-character span is not a date under any reading.
- Tokens: 1,360 single-token (21.3%); the mode is two tokens (2,098), then three (1,166) and four
  (1,139). `PERSON` by contrast is 70.2% single-token. A two-to-four-token span is the shape of a
  phrase, not of a date.
- Concentration: 1,554 distinct normalised surfaces, but **25 of them cover half of all 6,397
  marks**, and the top 20 cover 46.4%. A handful of recurring English phrases account for nearly
  half the category.
- Families: `free-speech` 2,124, `free-speech-v2` 1,743, `story-recall` 1,219, `story-recall-v2`
  634, `cinderella-story` 338. It tracks how much connected narrative a family elicits, which is
  what you would expect of a phrase-level artefact and not of disclosure.

## Detector, and why the split is not what it looks like

`DATE_TIME` findings: 7,141 `presidio`, 142 `gliner/date`. That reads as "presidio produces
essentially all of them", and it is true but not for the reason it suggests.

`nodes/speech.py` keys surviving findings on `(category, first word, last word)` and keeps the
**first** detector to reach that key. The scan order is fixed — presidio, then gliner, then rules —
so `source` reads `presidio` whenever presidio found the span, whoever else also found it. The
detector field is therefore a *precedence* record, not an attribution, and the same caveat applies
to every category: `PERSON` reads 4,046 `gliner/name` against 3,753 `presidio` only because those
4,046 are spans presidio did **not** find.

Two consequences worth stating plainly:

1. The corpus cannot answer "how many findings did gliner corroborate", because the second finder
   is discarded along with its finding. The `detectors_used` attribute says who *ran*, not who hit.
2. The 98.1%-presidio figure for `DATE_TIME` is a floor on presidio's involvement, not a measure of
   gliner's absence.

## Bracket overlap, by category

The earlier `[UH]`-as-`PERSON` defect, resolved per category over marks:

| category | marks touching a bracketed token | marks | share |
| --- | ---: | ---: | ---: |
| `DATE` | 21 | 227 | 9.3% |
| `PERSON` | 340 | 7,034 | 4.8% |
| `MISC` | 39 | 969 | 4.0% |
| `NAME` | 140 | 3,817 | 3.7% |
| `LOC` | 9 | 378 | 2.4% |
| `LOCATION` | 16 | 1,134 | 1.4% |
| `DATE_TIME` | 74 | 6,397 | 1.2% |
| `AGE` | 6 | 505 | 1.2% |

The bracket artefact is a name-detector problem, concentrated in `PERSON`/`NAME`/`DATE`. It is
**not** what is wrong with `DATE_TIME`: at 1.2% it is the second-cleanest category on this measure.
The two defects are independent and want different fixes.

## What the fix would be, and where it belongs

Not shipped in this task. Three candidates, in increasing order of how much they claim:

1. **A specificity gate on `DATE_TIME`, at the detector boundary.** Keep the finding only when the
   span carries a calendar anchor. On this corpus that retires roughly 96.5% of the category's
   marks. It belongs beside the bracket exclusion in `_scan_tokens`/the PII backend, because it is
   the same kind of rule: a class of token the scanner should not treat as evidence. It needs a
   derivation file and a fitted anchor list, not a literal in code.
2. **Stop acting on `DATE_TIME` for release, keep recording it.** The release fold, not the
   detector, decides. Defensible if the judgment is that a temporal expression is never on its own
   identifying — but that judgment should be the owner's and written down, because a date of birth
   or a date of death *is* identifying and would be caught by (1) and lost by (2) only if the
   anchor list is wrong.
3. **Leave it and let the review layer measure it.** The page now records a per-finding verdict, and
   `not-identifying` is exactly the verdict this category needs counted. A few hundred judged
   `DATE_TIME` marks would turn the 96.5% above from a lexical proxy into a measured rate, and would
   also settle whether the 3.5% calendar-anchored remainder is worth keeping.

(3) is the cheapest and the only one that produces evidence rather than consuming it, which is why
the review layer ships first and no detector change ships with it.

## Method

`scripts/free_speech_review_page.py extract` produces the rows; the shape classification is a
throwaway lexical pass over the extract, not repository code, and is reproducible from the row
schema alone. The counts above are over **marks**, which merge adjacent words of one category — so
they are lower than the raw finding counts, which double-count a finding the consensus scan and a
per-recognizer scan both reached.
