# The summary's first two pages: margins, and which page owns which block

Two defects the owner found by reading the rendered `summary.pdf`, not by reading the code.

## 1. The cover ignored the page margins

`summary_pages` built the cover with `cover.add_axes((0.06, 0.02, 0.92, 0.86))` — four hardcoded
figure fractions. Every evidence page below it is built with `plt.subplots(..., constrained_layout=True)`,
which places its axes inside a layout box and never lets an artist be positioned outside it.

Measured on the rendered page, at the default `FigureStyle` on the 11 x 8.5 in Letter landscape the
report uses:

| page | left edge of content | right edge of content | bottom stop |
| --- | --- | --- | --- |
| decision record (page 1) | 0.147 of width | 0.77 | subplot default |
| cover (page 2), before | 0.058 | 0.944 | 0.02 of height |
| evidence (page 3+) | constrained-layout box | constrained-layout box | constrained-layout box |

Two numbers matter. The bottom of the cover's axes sat at `0.02` of the page height — a 0.17 in
bottom margin on a page whose other margins are ten times that. And the axes' right edge sat at
`0.06 + 0.92 = 0.98`, so content was licensed to run to 0.22 in from the paper edge.

The widest line the cover can declare is `2 + 3 * _SUMMARY_COLUMN_WIDTH` = 158 monospaced
characters, which measures **9.875 in** at the default `text_fontsize` of 7.5 pt (DejaVu Sans Mono,
0.602 em advance). On an 11 in page that leaves 1.125 in for both side margins together, so the
largest symmetric margin the declared content fits inside is 0.5625 in.

`FigureStyle.cover_margin_in = 0.5` is that bound rounded down to a conventional print margin: the
widest declarable line clears the right margin by 0.125 in, and no value above 0.5625 could be
chosen without clipping a cover the summary is allowed to produce.

The fix is not a better set of fractions. The cover is now built with `layout="constrained"`, the
same engine the evidence pages use, and the margin is handed to that engine as a rect derived from
`cover_margin_in`. The suptitle is then laid out rather than dodged by a hand-tuned `0.86` height,
and the axes cannot be placed outside the margin box at any figure size.

## 2. The same blocks were printed on page 1 and page 2

Read off both rendered pages of one seeded run, the duplicated content was:

| fact | decision page (page 1) | cover (page 2), before |
| --- | --- | --- |
| recording route state | `recording: routed` | `recording: routed` |
| per-branch route state | `routes: AIRWAY=routed; SPEECH=routed; VOICE=routed` | `AIRWAY   routed       runs` |
| per-branch conformance | `AIRWAY: True; route_routed` | `ran - conformance True` |
| gates that fired | `fired: airway.cough, speech.words` | `gates fired: airway.cough, speech.words` |
| gates flagging a branch | `SPEECH flagged by: speech.short` | `speech.short    flagged` |
| branch measures | `AIRWAY: labelled_n=1` | `measures     labelled_n=1` |

Not duplicated, and wrongly suspected of it: the decision page's `SUPPORTING EVIDENCE` lists each
classifier's consolidated label counts, while the cover's `WHOLE-FILE CLASSIFICATION SUMMARY` lists
each label's peak and median score over the windows. Those are two different measurements of the
same classifier and both stay.

### Which page owns what

The decision record leads the document because it is fixed-length, it is what a reviewer reads
first, and on a file ADMIT refused it is the entire product. That is unchanged. It therefore owns
**every routing decision and the numbers the verdict rests on**: the route state per branch, the
conformance each branch reported, the gate outcomes, and the branch measures.

The cover owns **what the figure is of**: the source path, the alignment quality of the transcript
the evidence pages draw, the whole-file classifier score distributions, the enhanced and residual
stream summaries, and each branch's contribution to the span axis drawn below it — what its
conformance was about, its deviations, its unmeasured config points, and how many spans it proposed
and paired.

So, when a decision record precedes the cover:

- `ROUTE STATES AND GATE OUTCOMES` leaves the cover entirely. Every line of it is on page 1.
- The per-branch `measures` line leaves `BRANCH REPORTS`. Page 1's `MEASURED BRANCH FINDINGS` is
  the same numbers.
- Each branch's header line drops its route state and its conformance, keeping only whether it ran.
  Without the run state the block below it is uninterpretable — `0 spans proposed` would not
  distinguish a withheld branch from one that ran and proposed nothing — so that word stays.

### What a reader loses

One thing, and it is recovered rather than dropped. The cover's route block was the only place in
the PDF that printed **why** a gate could not be read; page 1's `ROUTING GATES` prints the
unavailable gates' names alone. That residue is now its own cover block, `GATES THAT COULD NOT BE
READ`, printed only when there is one. It is not a restatement of anything on page 1.

Nothing else is lost. Every removed line has a same-run equivalent one page earlier, and
`summary.json` carries the complete record either way, as `ANALYTIC RECORD` says.

## The two consumers

`summary_pages` has two callers and only one of them writes a decision record.

- `report()` writes the decision pages itself and then the figure's pages into one PDF.
- `preprocess_figure` regenerates the figure alone over a completed run directory. It has no
  decision page to lead with.

The dead `cover_prefix` parameter was the residue of a design where `report()` would have stacked
its decision record on top of the cover. Neither caller ever passed it, and stacking is what
produced the duplication in the first place. It is replaced outright by its inverse:
`decision_record: bool`, which says a decision record precedes these pages, so the cover must not
restate it. `report()` passes it; `preprocess_figure` does not, and its standalone cover keeps the
full route block, the full branch header and the measures — nothing it needs has vanished.
