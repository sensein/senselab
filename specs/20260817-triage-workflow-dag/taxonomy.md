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

## The screening set: YAMNet + AST + CrisperWhisper + HeAR

Decided 2026-08-19. These four screen for which kinds are present. What each contributes, and the
limit that bounds what its vote means:

| detector | contributes | limit |
| --- | --- | --- |
| **YAMNet** | 521 AudioSet labels on 0.96 s windows; the broadest vocabulary, and the only one with an explicit `Silence` class. Separates events 1.1 s apart. | No human-vocalic roll-up node exists in the 521 (`Human sounds`, `Human voice`, `Respiratory sounds` are all absent), so only a union of specific labels is available. |
| **AST** | A second AudioSet opinion, and it does disagree usefully — it called the second verified cough `Throat clearing` 0.96 where YAMNet said `Cough` 1.000. | Same training corpus and label space as YAMNet, so the two can be wrong together. Their agreement is not two independent votes. |
| **CrisperWhisper** | The only member that reports **words**, so it is what separates lexical from non-lexical. Also emits `[breath]`, `[cough]`, `[UH]` inline with timings. | Its non-lexical tokens are evidence, not verdicts: on the verified pair it bounded cough 1 almost exactly (onset −26 ms, offset −14 ms, 580 ms against 568 ms) and split cough 2 into two mislabelled tokens, `[UH]` and `[breath]`, covering 440 of its 640 ms. |
| **HeAR** | Strongest on breath — 0.998 and 0.997 on the two verified breaths, where YAMNet reads 0.726 and 0.893 and the speech-detector pair reads exactly 0.0000. | **Must not vote on the speech kind.** On the verified speech span it reports `Snore` 0.88 and `Speech` 0.01 across six independent measurements: it is not weakly detecting speech, it is confidently assigning it elsewhere. Its input is fixed at 2 s and it needs ~160 ms of an event, so it is a presence gate and never a locator. |

**Four detectors are not four families.** YAMNet and AST share AudioSet, so the independence
available here is closer to three: one AudioSet pair, one lexical reader, one health-acoustic model.
Any rule of the form "the families agree" has to count them that way.

**Coverage gaps this set cannot screen for**, worth naming so their absence is not read as absence of
the thing:

- **Mouth non-speech sound.** YAMNet has no label for it — `Lip smacking`, `Mouth`, `Smack, lip smack`
  and `Tooth` are all absent from the 521, leaving only chewing, biting and gargling — and it is not
  among HeAR's eight. The verified 202 ms mouth click scored 0.000 on YAMNet.
- **Sustained vowel and pitch glide.** No label in any of the four. Nobody has yet measured what a
  prolonged vowel *does* fire, so whether this kind is screenable at all is an open measurement rather
  than a settled negative.

## Detection aggregates over the waveform

Presence is a whole-file question, so each detector's output is reduced over the entire recording to
one value per kind. No localisation happens here and none is needed: a branch asks where.

The reduction is not the same operation for all four, because they do not all produce a score series:

- **YAMNet, AST, HeAR** emit a per-window score series. The reduction is the named per-kind aggregator
  over that series, after folding labels into the kind's family.
- **CrisperWhisper** emits a token list with timings. Its reduction is a presence count over token
  types — did any `[cough]` or `[breath]` token appear, and were there words at all — not a quantile
  over anything.

**The consequence to design against: whole-waveform aggregation dilutes a short event by recording
length.** A 0.3 s cough is 5% of a 6 s recording and 0.05% of a 10-minute one. A mean over windows
takes it to nothing in the second case while a max fires on a single spurious window in either. So

- the aggregator is a named per-kind config value, not a default, and short kinds want a high quantile
  or a top-k mean rather than either extreme;
- and a presence threshold fitted on short recordings does not transfer to long ones. The threshold
  is duration-dependent, or the aggregator has to remove the dependence before the threshold sees it.

This is the same failure that made a four-axis grid report that it had run while its defaults disabled
every cross-axis coupling: a default aggregator that silently nulls what it claims to measure.
