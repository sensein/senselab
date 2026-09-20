# The second diarized stream had no reader

## What was configured, and what reads it

`diarization.streams` shipped as `[enhanced, residual]`, so PREPROCESS ran pyannote **twice per
recording** — once on the enhanced stream, once on the residual.

`nodes/speech.py:_read_diarization` loops over the same configured list and **returns on the first
stream that has a readable measurement**. With that list it always returns `enhanced`, and never
looks at `residual`.

A sweep of the whole triage package for any other reader finds none: `speech.py` is the only module
that reads a `<stream>_diarization` measurement at all. AIRWAY, VOICE, QUALITY, REDACT, VERDICT,
REPORT, FIGURE and the routing analysis read none.

So the residual pass was computed on all 62,578 recordings and read by nothing.

## Why it survived

The config comment recorded the intent — *"`enhanced` answers how many voices survive it,
`residual` answers whether one was taken out"* — and the intent is a good one. What was never built
is the reader that would answer the second question. The writer shipped; its consumer did not.

This is the session's recurring defect class inverted. Twelve times over we found a reader keyed to
something no writer produces. This is a writer producing something no reader consumes, and it costs
compute on every recording rather than producing a wrong answer, which is why nothing surfaced it.

## What changed

`diarization.streams: [enhanced]`.

PREPROCESS keeps its ability to diarize any number of configured streams — that is tested, and the
two tests covering it now configure their own two streams rather than inheriting the shipped list,
so the capability and the shipping decision are pinned separately.

A new test asserts the contract rather than the count: SPEECH consults exactly one stream, and the
shipped config names exactly one. Adding a stream back is a one-line change the day something reads
it.

## What this does not say

**It does not price the saving.** No activity in a triage store carries `started`/`ended` — a census
of a real corpus store found `activities with start+end: 0` — so the run cannot say what any step
cost, diarization included. PREPROCESS as a whole is 143.6 s of the 158.3 s per recording (90.7%),
measured over 6,221 corpus rows, but the split within it is unmeasured.

Recording per-activity timing is owed and is already named in
`specs/20260817-triage-workflow-dag/llm-check-first-run.md` for a different step. It should be
general: a graph whose store cannot answer where its time goes cannot be optimised from its own
provenance.

## Open

- If the residual reading is wanted, the reader is owed: what it would conclude, and which node
  concludes it.
- Per-activity `started`/`ended` in every store.

## Looking for the rest of the class, and why it could not be settled

The residual diarization was found by accident. Two attempts to find the rest of its class
systematically both over-report, in opposite directions, and neither is usable as proof.

**By name, statically.** Take every measurement a real store holds (81 distinct, over 40 corpus
stores) and grep the triage package for each. It names 24 as unreferenced outside `preprocess.py` —
and the list includes `asr_qwen`, which obviously feeds the consensus, and the AST measurements,
which TAXONOMY, REPORT and `routing_analysis/detectors.py` all read. The method fails because the
names are **constructed**: `f"{stream}_{classifier}_scores"` never appears as a literal, so a
grep for the literal finds nothing and concludes nothing reads it.

**By provenance, from the store's own `used` edges.** Over 120 stores, ask which measurements carry
a `used` edge from an activity of any node. It names 38 as never used — and the list includes
`stimulus_alignment`, which SPEECH reads through `find_measurement`, and `praat_features`, `squim`
and `spectrogram_narrowband`, which REPORT draws on every summary. The method fails because **a node
may read a measurement without recording that it did**. REPORT is the clearest case: it reads the
whole store and writes no activity at all, so every panel it draws looks unread.

So the honest position is that this class cannot presently be swept. The one instance proven here
was proven by reading the consumer — `_read_diarization` returns on the first configured stream, and
no other module reads a diarization measurement at all — not by either sweep.

**The gap underneath is a provenance gap, not an optimisation one.** A store that cannot answer
"what did this node read" cannot support the question at all, and that same absence breaks the
derivation chain a reviewer would follow backwards from a decision to its evidence. Closing it —
every node recording `used` for the measurements it reads — would make this class sweepable as a
by-product, and would make the graph's provenance answer a question it currently cannot.

Counted from the two sweeps, the candidates that survive both and are worth checking individually
are the ones written on every recording: the `residual_*` classifier summaries. Those are read —
REPORT draws them on the cover under "RESIDUAL — BACKGROUND AFTER SPEECH REMOVAL", confirmed on a
rendered page. `residual_diarization` is not on that page, and is the one this document removes.
