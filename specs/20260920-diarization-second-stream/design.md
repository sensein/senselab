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
