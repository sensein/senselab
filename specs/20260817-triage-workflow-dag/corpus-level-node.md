# The cross-recording node

Some capabilities cannot be computed from one recording. They belong to **a node that runs last, over
the finished per-recording stores**, after the per-recording graph has completed.

This is a **placement decision, not a subsystem design.**

## Name the grouping level per capability

"Corpus-level" was doing double duty in an earlier version. The capabilities here group at
**different levels**, and the level determines what may be pooled:

| level | what it groups | capability |
| --- | --- | --- |
| **session** | recordings from one participant in one sitting | C2 composite indices; [`branch-quality.md`](branch-quality.md) Q2's bandwidth for vowel-only recordings |
| **participant** | all sessions for one person | cross-session consistency (candidate) |
| **device** | recordings sharing a capture chain | device-class grouping (candidate) |
| **corpus** | everything | C1 duplicate detection; the listening sample's draw (candidate) |

A capability states its level. Two capabilities at different levels are not interchangeable and
their outputs are not poolable across each other.

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

**Report the grouping, not just the match.** A duplicate means very different things depending on
where it falls, and the node knows which:

| grouping | what it means |
| --- | --- |
| same session | a re-submission — the mildest case, and the expected one |
| same participant, different session | a re-submission across sittings |
| **different participants** | **an identity or upload failure** — far more serious, and it corrupts any participant-level analysis |

**A lower bound, with an available extension.** A checksum catches byte-identical duplicates only; a
re-encode on upload defeats it, so a zero count means "no *exact* duplicates", not "no duplicates".

But the node's no-audio discipline does **not** block a content fingerprint: stored duration, the F0
track, the energy envelope and the HeAR embeddings already sit in the finished stores. **Near-duplicate
detection that survives re-encoding is partly available from records alone.** Keep the byte digest as
the lower bound it is, and note the record-based fingerprint as the extension.

**Free**: the digest exists and the comparison is equality.

**Level: corpus.**

### C2 — Composite voice-quality indices — **session-level, and the protocol is unsatisfiable**

Moved from [`branch-voice.md`](branch-voice.md) V5, which carries the full argument.

**Level: session** — not corpus. AVQI needs a *specific pair* from one participant in one sitting,
not any two recordings.

**It requires explicit audio access, which is this node's stated exception.** Concatenation is both a
waveform read and a fresh derivation, not a read of stored records — so C2 does not fit the no-audio
discipline the rest of the node keeps. The cleanest resolution is that **C2 is a separately declared
session-level capability with declared audio access**, rather than an exception carved inside a
no-audio node. Note also that **REDACT runs before this node**, so the audio C2 would read may
already have been altered.

**The protocol cannot be followed here, so the real choice is between an AVQI-shaped number that is
not AVQI and emitting nothing.** `branch-voice.md` V5 sets out why: the continuous-speech material
must be the short standardized sentence set in a prescribed proportion, which none of Rainbow,
Caterpillar or Harvard is and free speech disqualifies; only `prolonged-vowel` can supply the vowel
and only when it is /a/, making V8 a precondition; independent AGC states, gain and mic distance put
a level and spectral discontinuity at the join, directly into the LTAS terms; bandwidth and sample
rate must match across the pair and do not; and `extract_slope_tilt`'s bands are not AVQI's, so
following the protocol means reimplementing the index rather than composing helpers.

**A value assembled this way has no known relationship to the published 0–10 scale or its ~2.95
cut-off**, and the published smartphone recording-chain sensitivity applies on top of that.

**This document places the capability and states the conclusion; the owner decides whether to build
something that is not AVQI.**

### C3 — Vocal effort correlates (**session-level**)

Moved from [`branch-voice.md`](branch-voice.md) V6.

F0 shift, spectral tilt change and CPP change at maximal effort are **interpretable only against the
same participant's comfortable phonation in the same sitting** — which is a different recording. The
`loudness` families supply the effort; `prolonged-vowel` or `maximum-phonation-time` supplies the
reference.

**This also resolves a gap V6 could not.** `loudness` v1 (897) has only **one** condition — three
maximal attempts, no comfortable baseline — so a between-condition correlate is not computable from
that file at all. At session level the second condition comes from the session.

**Level: session.** It shares the grouping mechanism with C2 and with
[`branch-quality.md`](branch-quality.md) Q2's bandwidth-from-another-recording.

**Three cautions specific to this pair.**

**The bin.** `derive_f0_range` selects from the recording's own trimmed mean, and shouting raises F0
by 3–8 semitones — so the effort recording and the reference can land in **different bins** for the
same speaker (`branch-voice.md` V6). Any cross-recording F0 comparison records which bin each side
used, or it compares two instruments.

**The vowel, which V8 does not solve here.** `loudness` is "hey" — /heɪ/, a **diphthong**;
`prolonged-vowel` is a sustained monophthong. Tilt, CPP and F0 are vowel-dependent by the same order
as the effort effect being measured, so the comparison confounds effort with vowel. V8 is named a
precondition for C2; it is one for C3 too — **and for C3 identifying the vowels does not make them
comparable.**

**Only `prolonged-vowel` is an admissible reference.** `maximum-phonation-time` is **not** a
comfortable-effort condition: "as long as possible" produces declining intensity and downward F0
drift across the attempt, which `branch-voice.md` V2 knows and requires covariates for. Using it as
the baseline would compare maximal effort against a decaying maximal effort.

**And distance moves with the condition.** Participants pull the phone away when asked to be loud,
which moves tilt and depresses CPP — the same correlates being compared. C3 previously carried the
bin caution and not this one.

**Duration and onset dynamics are a fourth confound.** A 300–500 ms shouted diphthong against a
multi-second steady monophthong differs in tilt and CPP by **segment duration and onset transient
alone**, at the same order as the effort effect being measured.

**C3 therefore serves `loudness` v1 only** (897 recordings). `loudness-v2` has both conditions in one
file, matched on vowel, duration, distance, gain state and F0 bin — that contrast belongs in
[`branch-voice.md`](branch-voice.md) V6 and is strictly better than anything this node can assemble.

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

## What it writes, and where

**Three things are owed here and this document does not decide them**, because each has consequences
beyond this node:

1. **What it writes.** The old Q4 emitted "a file-level assertion naming the other recording"; that
   was dropped in the move and nothing replaced it. A cross-recording finding names two or more
   recordings, so it does not fit an assertion scoped to one store — the shape is undecided.
2. **Where the finding lives.** Per-recording stores are finished when this node reads them. Writing
   back into them contradicts that; writing somewhere else means a new artifact with its own
   location and reader.
3. **Whether it takes a node name.** The contract makes node name a **join key** — for verdicts,
   `run.json`, figure and report. So taking a name in `GRAPH_ORDER`, or writing a verdict entity, or
   carrying a `node:` attribute, are three different decisions with three different reader
   consequences. `GRAPH_ORDER` has ten entries and every reader iterating it would see a new one.

## Everything here is deferred, not scheduled

Worth recording plainly: **nothing "moved to this node" has a date, an owner or a dependency in any
plan** — C1, C2, C3, and [`branch-quality.md`](branch-quality.md) Q2's session-level bandwidth. Moving
a capability here resolved *where it belongs*; it did not schedule it.

**And C1 — the one nearly-free capability — is blocked on the three open questions below**: what this
node writes, where the finding lives, and whether it takes a node name. A digest comparison needs
none of the domain work the others do, and it is gated on the same unresolved plumbing.

## What it does not do

It does not re-run per-recording nodes, re-derive per-recording measurements, or change a
per-recording verdict.

It writes no deviation: a deviation is a departure from what a *task* asked for, and this node holds
no task declaration.
