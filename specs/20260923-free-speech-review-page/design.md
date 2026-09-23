# Reading the free-response transcripts as prose

The summary PDF already draws each recording against its waveform, spectrogram and span lanes. That
figure answers *where in the signal*. It does not answer *what did this person say*, and it cannot:
a reader following a sentence across a time axis is reading a plot, not a paragraph.

This page is the opposite instrument. No waveform, no spectrogram, no lane, no time axis. One
paragraph of consensus text per recording, grouped by the participant who produced it, with the
redaction findings marked where they fired.

## Why the opposite is worth building

Two defects found on 2026-09-23 are invisible in the figure and obvious in prose:

- `[UH]` — a transcription-convention token, not a word anybody said — was flagged `PERSON`. In a
  lane that is one coloured tick. In a sentence it is absurd on sight.
- 469 recordings had redaction removing task content rather than identity. Whether a redaction ate
  the answer is a question about the sentence, and can only be asked of the sentence.

So the page's diagnostic value is *inline* marking. A separate findings list beside the text would
reproduce the lane's failure in a new typeface: it tells you a category fired without letting you
see what it fired on.

## Which recordings

The graph declares its own task groups. `scripts/free_speech_review_page.py families` reads
`EXPECTATIONS` and prints every family whose expectation carries `Pattern.FREE_RESPONSE`, which is
the only definition this page uses. Ten families:

`cinderella-story`, `free-speech`, `free-speech-v2`, `open-response-questions`,
`picture-description`, `picture-description-option1`, `picture-description-option2`,
`productive-vocabulary`, `story-recall`, `story-recall-v2`.

Two edges are worth naming because a name-matching reader would get them wrong in both directions:

- **`productive-vocabulary` is in.** The task asks for definitional speech about a cue word, and the
  expectation is `FREE_RESPONSE` with `unviable=(("defines_its_cue", …),)`. Its name suggests a word
  list; the declaration says connected response.
- **`animal-fluency` is out.** The name sits next to `productive-vocabulary` in every corpus table
  and the task is unscripted, but its declared pattern is `ITEM_LIST`. Reading it as free response
  would put a minute of bare nouns among paragraphs and dilute exactly the thing the page is for.

The filter is applied twice: once on the run directory's stem (cheap, so most stores are never
opened) and once on the store's own `declared_family` (authoritative). A recording is kept only when
both agree the family is in scope.

## Two phases, because the corpus and the reader are not on the same machine

`extract` runs on the cluster. It walks `<root>/sub-*/ses-*/<stem>_<timestamp>/`, reads each
`run/store.jsonl`, and writes one compact JSON line per recording.

`render` runs on the laptop and turns those lines into HTML. Splitting it this way means the page's
appearance can be iterated on without re-reading 40 GB of stores.

### The store read is deliberately partial

A finished store is a few megabytes, and nearly all of it is derivative arrays under
`prov_type: "measurement"` — gammatone frames, posteriorgrams, per-window classifier scores. The
page reads none of them. `read_store_light` skips any line carrying the measurement marker unless it
also carries `"name": "pii_scan"`, which is the one measurement that matters here. Over the whole
free-response set this is the difference between a job and an afternoon.

### Live generation only

The rerun stores carry two generations of decision. Every read goes through `StoreView.live`, which
drops anything a `wasInvalidatedBy` edge retired. That is the same rule `live_entities` applies in
the nodes themselves, and it is strictly weaker than `replay_diff.split_generations`: the `after`
set is a subset of the live set, and this page wants "what the finished run currently says", not
"what the replay changed".

## What one recording carries

| field | source |
| --- | --- |
| participant, session, task | the run directory's stem, via `identity` |
| family | the live `VERDICT` entity's `declared_family` |
| release, release ground | the live `VERDICT` entity's `release` / `release_ground` |
| words | live `word` entities in `index` order — `text` and `bracketed` |
| per-word categories | live `label`/`pii` assertions, followed through `wasDerivedFrom` to the word |
| findings | live `pii` entities — category, detector, haystack, `in_stimulus` |
| scan state | the live `pii_scan` measurement's `scanned`, or absent |

The per-word categories come from the **label assertions**, not from intersecting each finding's
extent with each word's hull. Both routes exist in `redact.py`; the assertion is the one SPEECH
actually wrote against a specific word, so it is the one that says where the detector looked. Extent
overlap would re-derive the same answer with a rounding error's worth of disagreement, and would
silently widen a finding whose extent fell back to the whole transcript.

## What the page does

- **Inline marks.** A run of adjacent words carrying the same category set becomes one `<mark>` with
  the category named once. A two-word name reads as one finding, which is what it is. Two different
  categories side by side stay two marks, because merging them would invent a finding neither
  detector made.
- **Bracketed tokens are visibly not speech.** `bracketed` is a word attribute, so the page does not
  have to guess from the square brackets. They render monospaced, smaller and muted — the `[UH]`
  case reads as a convention token wearing a `PERSON` label, which is the whole point.
- **Release state and ground are on every card.** All four states of the vocabulary, plus the
  `release_ground` sentence where one exists and REDACT's `why` where it does not, because
  `release_ground` is `None` exactly when REDACT itself decided and the reason is in the verdict.
- **Filters over the whole set.** Free-text search across transcript and header, family checkboxes,
  release checkboxes, and a redaction-fired three-way. Filtering hides recordings rather than
  participants, so a participant section disappears only when none of their recordings survives.
- **A jump rail.** Every participant, with their recording count, filtered in step with the main
  column.

## Scale and sharding

The `extract` report prints recordings, participants and total transcript characters before any
page is written. `render --shard-size N` splits participants into files of N with an index; the
default is one file. The decision is made from the measured character count, not from a guess: HTML
markup multiplies transcript bytes by roughly four, and a single document past the tens of megabytes
stops being a thing a browser opens comfortably.

## Data handling

The page carries real transcribed speech and marked PII, which is the point of it: the owner
authorised that for local viewing, and a page with the words removed could not show the two defects
above.

That authorisation has hard edges, and they are structural, not advisory:

- **The output is local.** `write_pages` chmods every file it writes to `0600`. The page goes to
  `~/Downloads`, never into the repo tree.
- **Never published.** Not as an Artifact, not to any host. Self-contained HTML, no network
  reference of any kind — a test asserts the rendered document contains no `http`, no `<link>`, no
  `<img>` and no `src=`.
- **Nothing downstream carries the words.** No transcript text and no detected string appears in a
  commit, a spec, a test fixture or a report. Every fixture in
  `src/tests/audio/workflows/triage/free_speech_review_page_test.py` is invented.
- **The corpus is read-only.** `streams/` under each run are symlinks into the design tree; writing
  through one is how an earlier driver overwrote 17,924 corpus files. This script opens
  `run/store.jsonl` for reading and writes only to its own output path.
