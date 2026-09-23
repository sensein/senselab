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

---

# The review layer, and facets over the findings

Added 2026-09-23 after the owner read the first page. Three requests, all additive.

## Findings became first-class

The first page marked words and computed the runs in the renderer. Nothing downstream could name a
finding, so nothing could carry a judgment about one. Marks are now computed in the extractor and
each carries:

| field | what it is |
| --- | --- |
| `k` | the review key — `sha1(stem, categories, first word index, one past last)`, truncated |
| `c` | the category set, in store order |
| `d` | the detectors of the findings that plausibly produced it, tightest first |
| `dn` | how many findings were candidates — the ambiguity, recorded rather than hidden |
| `i`, `nt`, `nc` | word range, token count, character count |
| `brk` | whether it touches a bracketed transcription token |
| `stim` | whether the declared stimulus accounts for it |
| `tx` | 1 inside the task extent, 0 outside, −1 when the branch minted none |

The key is content-addressed on the consensus word index, which is the position PREPROCESS emitted
the word at and the only order a reader may use. It therefore survives re-extraction, re-filtering
and re-ordering of the page — a judgment is not lost because a filter moved a card. It does *not*
survive a re-run that changes the transcript, and should not: a different transcript is a different
finding.

## Two joins that are not what they look like

**The task extent is a span, not a duration.** `response_duration_s` is written with `start=None`
and `end=None` — it is the length of the response hull and records no position, so it cannot say
whether anything sits inside it. The extent is the `span` entity the in-family branch mints with
`role="task_extent"`, read as `view.last("span", role="task_extent", family="speech")`. Absence is
the branch's record that it found no task, not a read failure, which is why `tx` has three states
and not two.

Measured, this facet discriminates almost nothing here: 15,588 of 15,595 marks are inside, 5
outside, 2 in recordings with no extent. That is structural rather than surprising — for
`FREE_RESPONSE` the branch mints the extent as the hull of the lexical consensus words, and every
marked word is a consensus word, so containment is nearly a tautology. The facet is kept because it
is cheap and because the 5 exceptions are exactly the kind of thing worth being able to find; it is
not kept because it is discriminating.

**Detector attribution is geometric and approximate.** The label assertion that marks a word carries
`{verb, label, category}` and no pointer to the finding, so the join must go through category and
time. But a `pii` entity's extent is `_timings_hull` over *every recognizer's* placement of its
words, while the words it marks carry the consensus interval — so the finding's extent is generally
wider than the mark it produced and can reach neighbouring unmarked words. The extractor prefers a
finding that contains the mark over one that merely meets it, and the narrowest within each group,
then records `dn`. Measured: 9,787 marks have exactly one candidate, 3,811 have two, 1,051 three,
937 four or more, and 9 have none. So attribution is unambiguous for 63% of marks and a ranked guess
for the rest — which is why `d` is a list and the page shows all of them.

One consequence reaches further than this page. `nodes/speech.py` keys surviving findings on
`(category, first word, last word)` and keeps the **first** detector to reach that key, in a fixed
scan order of presidio, gliner, rules. A second detector finding the same span is discarded along
with its finding. So `source` is a precedence record, not an attribution, and the corpus cannot
answer "which findings were corroborated". That is a property of the graph, not of this page, and it
is recorded here because the detector facet would otherwise read as more authoritative than it is.

## The review vocabulary: four terms, not three

| verdict | what it claims | what it implies for a fix |
| --- | --- | --- |
| `identifying` | a real disclosure; the redaction is right | nothing to change |
| `not-identifying` | the category fits, but nobody is identified | tighten the category, or stop acting on it |
| `not-the-category` | not an instance of the category at all | tighten the detector |
| `unsure` | needs a second look | nothing yet |

The obvious vocabulary is *real* / *false positive* / *unsure*. It was rejected because this corpus
contains the two middle cases in bulk and they want opposite fixes:

- 1,007 marked words are bracketed transcription tokens, 723 of them `PERSON`. `[UH]` is not a name
  by any reading — the detector should never have seen it. That is `not-the-category`, and the fix
  is a token-level exclusion beside the one already there.
- 96.5% of `DATE_TIME` marks carry no calendar anchor. A duration genuinely *is* a temporal
  expression; presidio is not wrong about what it saw. It just identifies nobody. That is
  `not-identifying`, and the fix is a specificity rule or a release-fold decision, not a detector
  patch.

Collapsing both into "false positive" would produce an export that cannot tell those two apart,
which is precisely the decision the measurement is heading toward. The cost is one extra term.

## Persistence: the export is the record

`localStorage` under `senselab.fsreview.v1`, every read and write in `try`/`catch`. It throws
outright on an opaque origin and comes back empty in a private window, so the page treats it as a
convenience: a failed write swaps the status line for a warning and the session continues in memory.
Verified — with storage throwing, all 11,701 cards render, judgments are still recorded and the
export still works.

The durable artefact is the JSON export: `{schema, version, exported, findings, recordings}`, keyed
by review key and by BIDS stem. It is written to a textarea *and* offered as a download, because a
sandbox may block the download and the textarea always works. Import merges rather than replaces, so
two sessions' judgments can be combined.

The export carries the verdict, the note, a timestamp, and the mark's category and detectors. It
carries **no transcript text**, so it is the one artefact here that could travel without carrying
speech — though nothing in this task sends it anywhere.

## Facets

Recording-level filters (family, release, search, redaction-fired, minimum findings) and
finding-level facets (category, detector, bracket overlap, task extent, span length in tokens,
review state) compose. When any finding facet is narrowed, a recording shows only if at least one of
its marks matches, and its non-matching marks dim rather than disappear — so the reader keeps the
sentence around the finding they selected for. The status line reports participants, recordings and
matching findings, so the size of a subset is known before it is read.

A compound mark carries its categories joined with `+` in one attribute, and the filter splits on
that. Matching the whole attribute instead dropped every multi-category mark from every category
facet — 4,770 of 15,595 marks, silently. There is a test pinned to the split.
