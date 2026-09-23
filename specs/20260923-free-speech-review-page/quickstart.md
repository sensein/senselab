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
