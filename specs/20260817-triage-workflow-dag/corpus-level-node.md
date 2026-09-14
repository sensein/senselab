# The corpus-level node

Some capabilities cannot be computed from one recording. They belong to **a node that runs last, over
the finished per-recording stores**, after the per-recording graph has completed for every recording
in a corpus.

This is a **placement decision, not a subsystem design.** What follows says where these capabilities
live and what the node reads; it does not specify the node's internals beyond that.

## Why the class exists

The per-recording graph — ADMIT → PREPROCESS → SCREEN → BRANCHES → QUALITY → REDACT → VERDICT —
takes one recording and concludes about it. Two kinds of question fall outside that:

- **Cross-recording comparison.** Is this audio the same as another recording's?
- **Cross-recording assembly.** An instrument whose protocol requires material that, in this corpus,
  is spread across several recordings of the same session.

Both were previously parked inside per-recording branches, where neither could run.

## What it reads

**Finished per-recording stores**, not audio — the same discipline as
[`branch-quality.md`](branch-quality.md), and for the same reason: it is what lets the node run over
a completed corpus without re-deriving anything. A capability that genuinely needs the waveform says
so explicitly and is the exception.

**Same-participant and same-session grouping is part of what it reads**, and is what makes the class
possible at all. The BIDS layout carries it: `sub-<id>/ses-<id>/` in the run path, and the
declaration carries the session identity.

## The capabilities

### C1 — Exact-duplicate detection

Moved from [`branch-quality.md`](branch-quality.md) Q4, which correctly identified it as
QUALITY-shaped and correctly noted it has no per-recording home.

ADMIT records `checksum_sha256` on the `recording` stream entity. Two recordings under different task
ids with the same checksum are the same audio. Re-submission of a previous recording under a new task
id is a known failure mode of app-based collection at this scale, and it corrupts exactly the
declared-family comparisons the corpus is scored against.

**A lower bound.** A checksum catches byte-identical duplicates only; a re-encode on upload — a
different container, bitrate or sample rate — defeats it. A zero count means "no *exact* duplicates",
not "no duplicates". Near-duplicate detection needs an audio fingerprint, which is not in the
inventory.

**Free**: the digest exists and the comparison is equality.

### C2 — Composite voice-quality indices

Moved from [`branch-voice.md`](branch-voice.md) V5.

The Acoustic Voice Quality Index (Maryn et al. 2010) combines CPPS, HNR, shimmer local, shimmer dB
and LTAS slope and tilt. **Its protocol requires a sustained vowel *and* continuous speech,
concatenated** — and in this corpus those are different recordings of the same session
(`prolonged-vowel` or `maximum-phonation-time`, and one of the passage or free-speech families). So
assembling its input is a same-session operation, which is why it has no per-recording home.

**Whether this project emits any composite severity index at all is undecided and is the owner's
call.** [`branch-voice.md`](branch-voice.md) V5 sets out the argument: published coefficients
discharge the no-refits rule, but a 0–10 severity scale with a published cut-off engages the
*non-diagnostic* constraint instead, and every other capability in these documents declines to map an
acoustic value to normal or disordered. It also ingests shimmer with the largest positive
coefficient and has no validity gate, while `branch-voice.md` V4 gates shimmer everywhere else.

**This document places the capability; it does not decide whether to build it.**

If it is built: the concatenation protocol it was validated on must be followed rather than
approximated, its published recording-chain sensitivity — including smartphone offsets — travels with
every value, and the shimmer component inherits V4's validity gate rather than bypassing it.

## Candidates, not yet specified

Flagged rather than designed, since each needs its own argument:

- **Within-participant consistency across sessions** — the same measure on the same person at two
  times. Longitudinal comparison is a different question from a single verdict, and this corpus has
  repeat sessions.
- **Device-class grouping.** [`branch-quality.md`](branch-quality.md) Q2's effective bandwidth is
  per-recording, but the *distribution* of bandwidth across a participant's recordings is what
  distinguishes a participant who changed device from a corpus with mixed devices — and that
  distinction matters to every covariate built on it.
- **The listening sample's own draw.** [`branch-listening-sample.md`](branch-listening-sample.md)
  specifies stratifying densely near each candidate operating point, which is a selection over
  measured values across the corpus — a corpus-level read by construction.

Anything else that cannot be computed from one recording belongs here. **Candidates are flagged, not
invented.**

## What it does not do

It does not re-run per-recording nodes, re-derive per-recording measurements, or change a
per-recording verdict. Those stores are finished when it reads them.

It writes no deviation: a deviation is a departure from what a *task* asked for, and this node holds
no task declaration — it holds a corpus.
