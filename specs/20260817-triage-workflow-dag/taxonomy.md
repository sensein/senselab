# TAXONOMY — what is in this recording

Decided 2026-08-19. This file governs; where `flowchart.md` or `workflow.nf` disagree they are stale.

## What it decides

**What kinds of sound this recording contains.** Nothing more. Not where they are, not how good they
are, not how well any of them can subsequently be measured.

`kinds` are the branches. There is one presence answer per branch, and a branch runs when its kind is
present. So TAXONOMY discriminates at exactly branch granularity — finer would produce a distinction
nothing consumes, coarser could not route.

## Signature

```
taxonomy(audio) -> fail(reason) | flag(reason, kinds) | pass(kinds)
```

| port | direction | type | meaning |
| --- | --- | --- | --- |
| `audio` | in | decoded audio | from ADMIT, the recording as supplied |
| `fail` | out | reason | no branch would run; nothing is present |
| `flag` | out | reason, kinds | a human decides; the partial answer travels with it |
| `pass` | out | kinds | one presence `Estimate` per branch |

Each `Estimate` carries its evidence count and its spread, so `kinds` is the whole product — there is
no separate evidence port for a consumer to have to join back.

## Outcomes

| outcome | condition |
| --- | --- |
| **pass** | every branch is decided, and at least one is present |
| **fail** | every branch is confidently absent, with the evidence families agreeing |
| **flag** | any branch is undecided |

**`flag` is per file, not per branch.** One undecided branch flags the whole recording. A partial run —
certain branches proceeding while an uncertain one waits — would publish results whose completeness
depends on something a human has not yet looked at, and nothing downstream could tell that apart from
a file where the uncertain branch was genuinely absent.

**`fail` and `pass` are not symmetric.** A low presence score means either "not there" or "there but
quiet or masked", and quiet-and-masked is the case this workflow exists to catch. So `fail` requires
positive agreement across evidence families that nothing is present; low-and-disagreeing is `flag`.
Discarding is the destructive action and carries the burden of proof.

## Two constraints on how presence is computed

**Aggregate over label families, not over the top label.** A single verified exhalation returns
`Breathing` 0.89, `Sigh` 0.77, `Gasp` 0.72 and `Sneeze` 0.70 from the same classifier on the same
window. Those margins are inside what a different speaker or noise level reorders, so an argmax over
class names is reading noise. The unit of evidence is family mass.

**Aggregation over time is a per-kind choice, not arithmetic.** A cough is ~0.3 s in a recording of
minutes: a mean over windows dilutes it to nothing and a max fires on a single spurious window. Each
kind names its own aggregator, and short kinds need a high quantile or top-k mean.

## What TAXONOMY does not do

- No localisation. No spans, no onsets, no offsets. Those are branch questions.
- No per-branch flagging, per the rule above.
- No judgement about how measurable a present kind will turn out to be. Whether a branch can do its
  job on this recording is that branch's finding, reported by that branch.
- No enhancement or separation. If a branch wants a channel in which one element survives, that is an
  operation inside the branch.

## Thresholds

The confident-present and confident-absent thresholds are unfitted, and there is no labelled corpus to
fit them on. Because this node can `flag`, the default under doubt is to flag: the derivation slots stay
in the config and stay empty rather than holding literals nobody measured.
