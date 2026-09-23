# Building the page

Two commands. The first runs where the corpus is, the second where the page will be read.

## Which families the page covers

```bash
uv run python scripts/free_speech_review_page.py families
```

Prints the ten `Pattern.FREE_RESPONSE` families from `EXPECTATIONS`. Nothing else is in scope, and
nothing is hard-coded — the list moves when the graph's declaration moves.

## Extract, on the cluster

Never on a login node. `pi_satra` or `mit_normal`; the sweep is I/O bound, not CPU bound, so the
core count is about how many stores can be read at once.

```bash
#!/bin/bash
#SBATCH --job-name=fsreview
#SBATCH --partition=pi_satra
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
CHECKOUT=/orcd/scratch/bcs/002/satra/senselab-rerun
WORK=/orcd/scratch/bcs/002/satra/free_speech_review
"$CHECKOUT/.venv/bin/python" "$WORK/free_speech_review_page.py" extract \
  /orcd/scratch/bcs/002/satra/triage_rerun_20260923/out \
  --out "$WORK/free_speech_rows.jsonl" --workers 16 | tee "$WORK/extract_report.json"
```

The corpus root is read-only. Each run's `streams/` are symlinks into the design tree, and writing
through one is how an earlier driver overwrote 17,924 corpus files; this script opens only
`run/store.jsonl`, for reading.

The report is the scale check — recordings, participants, total transcript characters — and is
printed before any page exists.

## Render, locally

```bash
scp orcd:.../free_speech_rows.jsonl ~/Downloads/free-speech-review/
uv run python scripts/free_speech_review_page.py render \
  ~/Downloads/free-speech-review/free_speech_rows.jsonl \
  --out ~/Downloads/free-speech-review/free-speech-review.html \
  --title "Free-speech review"
```

Add `--shard-size 200` to split participants across files with an index; `--out` is then a
directory. Every file written is chmod `0600`.

## Measured, 2026-09-23

Corpus `triage_rerun_20260923/out`, generator run against the pinned checkout `71324c35`.

| | |
| --- | --- |
| candidates and recordings | 11,701 (no store skipped, no store unreadable) |
| participants | 1,514 of the corpus's 1,527 |
| transcript characters | 2,914,546 |
| extract wall clock | 77 s on 16 cores |
| extract output | 14.5 MB JSONL |
| rendered page | 12.2 MB, one file |

Per family, matching `expected-patterns.md` row for row: `free-speech` 3,074, `productive-vocabulary`
2,910, `free-speech-v2` 2,120, `picture-description` 889, `story-recall` 889, `story-recall-v2` 660,
`picture-description-option1` 373, `picture-description-option2` 329, `cinderella-story` 258,
`open-response-questions` 199.

Release: `nothing_to_redact` 5,009, `releasable` 3,719, `withheld` 2,972, `not_assessed` 1.

One page was viable, so no sharding was used. 2.9 MB of text became a 12.2 MB document because the
markup — one card, one header, one release chip per recording — outweighs the prose four to one.
Sharding becomes worth it somewhere above this, not at it.

## What the page already shows

6,691 of the 11,701 recordings carry at least one finding, 48,236 words are marked, and **1,007 of
those marked words are bracketed transcription tokens** — 723 of them as `PERSON`. That is the
`[UH]`-as-a-name defect, and it is countable from the extract and visible on sight in the page,
which is the whole reason the marks are inline and the brackets are styled apart.

## Verifying the page without a browser

```bash
npm install --no-save jsdom
node --max-old-space-size=8192 <check.mjs> ~/Downloads/free-speech-review/free-speech-review.html
```

Loading the document under jsdom with `runScripts: "dangerously"` exercises the real filter code:
the initial status line, the redaction three-way, the release checkboxes, the reset link and the
search box all report counts that must reconcile with the extract report — 6,691 with findings plus
5,010 without is 11,701, and `withheld` is 2,972 either way.

## Measured again, 2026-09-23, with findings as first-class objects

Job 23549767 on `pi_satra`, same corpus and pinned checkout. `pi_satra` takes `--qos=normal` for
this account; `--qos=pi_satra` is rejected, and **`mit_preemptible` does not exist — the partition
is `mit_preemptable`**.

| | |
| --- | --- |
| recordings, participants, characters | unchanged: 11,701 / 1,514 / 2,914,546 |
| reviewable marks | 15,595 |
| extract output | 19.4 MB JSONL |
| rendered page | 18.5 MB, one file |
| extract wall clock | ~100 s on 16 cores |

## Reviewing

Click a mark, or tab to it and press Enter. The panel takes `1`–`4` for the four verdicts, or a
click; pressing the same verdict again clears it. `Esc` closes. Notes are per finding in the panel
and per recording under each card. The rail shows how many of the 15,595 findings are judged and
filters to the unjudged.

`Export JSON` puts the record in the textarea and offers it as a download. `Import` merges an export
back in, from the textarea or from a file. Nothing is published; the export is a local file.

## Verified under jsdom

No browser extension was reachable, so the page was driven under jsdom with
`runScripts: "dangerously"` — real event dispatch against the real script, not a reimplementation.

```bash
npm install --no-save jsdom
node --max-old-space-size=12288 <harness.mjs> ~/Downloads/free-speech-review/free-speech-review.html
```

What the harness exercised, and what reconciles:

- facet counts against the independent measurement — `DATE_TIME` alone selects 6,397 marks and
  `PERSON` alone 7,034, matching the sidebar and the offline count exactly; bracket-overlap alone
  selects 542; `PERSON` + `gliner/name` + bracketed selects 201;
- combination and reset, span-length and findings-per-recording thresholds;
- the panel: verdict by click and by key, un-judging by re-pressing, per-finding note, per-recording
  note, progress line, filter to unjudged (15,593 of 15,595 after two judgments);
- export → fresh page → import → marks repaint with their verdicts;
- the storage-failure path, on an opaque origin where `localStorage` throws: all 11,701 cards still
  render, the status line says judgments are memory-only, and export still works.

Zero jsdom errors, and the document still contains no `http`, `<link>`, `<img>` or `src=`.

**Not verified:** the page has never been opened in a real browser in this session. jsdom does not
lay out or paint, so nothing here tests the CSS — the dark-mode palette, the dimming of non-matching
marks, the sticky rail at this document size, and scroll performance over 11,701 cards are all
unobserved. jsdom also does not implement download behaviour, so the `<a download>` path is
untested; the textarea fallback is what was exercised.

## Measured again, 2026-09-23, with the determination panel

Job 23556083 on `pi_satra` (`--qos=normal`). Same corpus, same pinned checkout `71324c35`.

| | |
| --- | --- |
| recordings, participants, marks | 11,701 / 1,514 / 15,595 |
| extract output | 40.3 MB JSONL |
| rendered page | 24.2 MB, one file, 11,701 cards and 11,701 panels |

The extract grew because each row now carries its determination; the page grew far less than that,
because the account is interned into one pooled payload and the panel is built on demand.

## Reading why

Each card has a *what determined this status* button. It opens over a scrim and closes on `Esc`.
It shows the decisive line (which of REDACT or the fold decided, and what it said), the LLM
reviewer's state in plain words, every gate with its reading and bound, every node with its run
state and outcome, the findings with their detectors and stimulus check, and what the stimulus
check had to work with.

## Verified under jsdom, this pass

Three recordings covering all three decision shapes — one withheld by REDACT, one released by
REDACT, one decided by the fold — plus the scrim, the close button, and the reset of all earlier
filters and review controls. Zero jsdom errors. The document still contains no `http`, `<link>`,
`<img>` or `src=`.

**Still not verified in a real browser.** jsdom neither lays out nor paints, so the panel's
appearance at this width, its scrolling inside a 24 MB document, the dark palette and the scrim's
z-order are unobserved.
