# A compact vector per recording, and the pages it drives

Owner-directed 2026-09-22: a parallel-coordinate view over the corpus where each line is a
recording and any measure can be swapped in; selecting a line opens an approximate view of that
recording — *without rasters, but able to generate an approximate vector representation* — the whole
thing driven by one parquet.

## What the summary PDF draws, and what survives compression

| panel | what it is | compact form |
| --- | --- | --- |
| wideband spectrogram | a raster | **dropped**. A time–frequency image cannot be made small and honest; the panels below carry what it was read for |
| conditioned waveform, envelope, floor, continuity | four traces over time | four decimated polylines |
| spans by source | rectangles, five rows | `(row, t0, t1)` triples |
| YAMNet / HeAR per-span scores | a small matrix | per span, the top label index and its score |
| SQUIM per span | three traces | three values per span |
| consensus ASR | words with extents and text | extents **and text**, with PII spans marked — see below |
| branch lanes | rectangles with roles | `(role, t0, t1)` triples |

Everything that remains is a polyline or a rectangle. Nothing is an image.

**The parquet carries transcript text and marks its PII.** Owner-directed 2026-09-22: *PII can be
stored in this and indicated. That's ok — this is all for my viewing.* So the ASR lane keeps its
words, and every `pii` entity in the store becomes a marked span carrying its category, so the
recording view can show what was detected and where.

That makes the parquet a **sensitive artefact**, and it inherits the handling the corpus PII census
already has:

- **mode 600, and never committed.** No path under the repository, and `.gitignore` carries the
  name so it cannot be added by accident.
- **Never published, never attached, never copied to a shared location.** It goes to the owner's
  own machine or to the mode-700 tree on ORCD scratch, and nowhere else.
- **The HTML that reads it inherits the same status.** A page that renders detected PII is not a
  page to host; it is opened from disk by the person the data belongs to.
- **The corpus-level artefacts stay clean.** `corpus_decisions.json`, the measurement distributions
  and everything committed to `specs/` remain counts and categories only. This exemption is for one
  file, for one reader.

Marking rather than omitting is also the more useful choice: a reader looking at a recording whose
release was withheld can see *why* — which of the twenty-one categories fired, over which words —
and that is exactly the judgement the census showed the detectors get wrong 91% of the time on DDK
carriers.

## The encoding

Time is normalised to the recording's duration and quantised to **uint16** — one part in 65,535 of a
30 s recording is 0.5 ms, far below anything the graph measures. Amplitudes, scores and fractions
are **uint8** over their known range. The envelope decimates to **256 points**, which at 30 s is one
sample per 117 ms and preserves the shape the eye reads.

Each recording is therefore a handful of short byte strings plus its scalars. Budget: **≈2 KB**, so
the whole corpus is **≈125 MB** — small enough for one parquet a browser can fetch in pieces.

## The parquet

One row per recording. Three identity columns first, as directed — `participant`, `task`,
`verdict` — then every numeric measurement the graph writes, then the compact vectors as binary
columns.

The measurements are already enumerated: 29 distinct names over the corpus, in
`specs/20260817-triage-workflow-dag/measure-distributions.md`. A measure absent for a recording is
null, never zero — the distinction the whole session turned on.

## The pages

**The corpus view.** Parallel coordinates, each line a recording, **up to ten axes**, any measure
assignable to any axis and axes addable and removable. Defaults start at participant, task and
verdict, and then the measures that discriminate: duration, conformance per branch, the PII finding,
the flag count.

**The recording view.** Selecting a line draws that recording from its compact vectors — envelope,
spans, branch lanes, scores — beside its decision record. An approximate view, arrived at without
fetching a single image.

## Speaker embeddings — held

An embedding per speaker, computed over **task extent**, is owner-directed and deliberately not
started: task extent is still moving. VOICE's families mint one on 38–76% of recordings today, and
the gates that decide whether a carrier survives moved into VERDICT hours ago. An embedding derived
from an extent that is about to change would have to be recomputed. It waits for the re-run.
