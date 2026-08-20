# ADMIT — the node, and why it holds no thresholds

Decided 2026-08-19. This file governs; where `flowchart.md` or `workflow.nf` disagree they are
stale, and neither is a source of structure.

## What it decides

One question: **is this file measurable at all.** Not whether it is good, not whether it is loud
enough, not whether it contains speech.

**Outcome vocabulary: `pass` or `fail`. No `flag`.** Nodes in this graph report `pass`, `fail` or
`flag`, and ADMIT is the one node that cannot return the third. `flag` exists to carry a judgement
that could have gone either way, and ADMIT's conditions admit no doubt: a file either decodes or it
does not, its samples are either all zero or they are not. There is no borderline for a human to
adjudicate, so offering one would be inventing uncertainty the measurement does not have.

That is also what makes two outcomes safe here rather than lossy. A two-way gate is dangerous exactly
when its condition is a matter of degree — which is why the "too quiet" row below is refused.

## Signature

```
admit(audio_file) -> fail(reason) | pass(audio, level_track, band_floor, clip_track)
```

| port | direction | type | meaning |
| --- | --- | --- | --- |
| `audio_file` | in | path | the recording, as supplied |
| `fail` | out | reason | the file cannot be measured; nothing else is claimed about it |
| `audio` | out | decoded audio | samples, rate, channel count |
| `level_track` | out | series | broadband level over time |
| `band_floor` | out | per-band scalar | estimated noise floor per band |
| `clip_track` | out | series | samples at or beyond full scale |

The three tracks are **measurements, not verdicts**. Nothing here judges them. A consumer that wants
to act on level or clipping applies its own threshold and owns that decision.

## Decision rule — degenerate conditions only

| condition | determined by |
| --- | --- |
| decode failure | the decoder raises, or returns zero frames |
| empty | every sample is zero |
| flat | constant value, or DC with no variance |

Everything else is admitted. There is no fourth row, and in particular no "too quiet" row.

## Why threshold-free

A margin separating "no signal" from "quiet signal" is an unfitted number, and this project has been
bitten twice by exactly that: a silhouette coefficient read as a probability, and a 2->10 dB HNR ramp
under which ordinary voiced speech read as only partly voiced.

There is also direct evidence against any such margin from the only labelled recording available.
Its verified-*empty* stretches are not silent — they are room tone, and every instrument measured on
them reads content: YAMNet returns `Silence` above 0.5 in 12 of 29 windows but not in all of them, and
HeAR's `Snore` reaches 0.86 inside stretches carrying no event. A margin tuned to call room tone "no
signal" would reject this file. A margin loose enough to admit it rejects nothing that the degenerate
conditions do not already catch.

So the threshold would be either harmful or inert, and there is nothing to fit it on. The node stays
threshold-free until a labelled corpus exists that could fit one, at which point adding it is a
measured decision rather than a guess.

## Why there is no speech test here

This is the load-bearing constraint, and it follows from what the workflow is for rather than from any
convention.

An off-target speaker is **quiet and incidental by construction** — that is what makes it off-target.
A non-lexical vocalization carries **no words at all**. A speech gate at admission therefore discards
precisely the recordings this workflow exists to catch, and it discards them silently, because a
rejected file produces no evidence of what it contained.

Presence of speech is a measurement that carries uncertainty. It belongs where uncertainty can be
represented and where a doubtful answer can flag rather than delete.

## What ADMIT does not do

- No models of any kind.
- No speech, voice or event detection.
- No enhancement, and no second version of the audio. Enhancement is an operation a later node may
  invoke for its own purpose; it is not a product of this one.
- No quality verdict. It emits the tracks a quality judgement would need and makes none.
- No resampling or channel reduction as a *decision*: whatever normalisation a consumer needs, that
  consumer performs, so the admitted audio stays the recording as supplied.

## Consequence for the graph

Because the only rejection is degenerate, **almost every real recording passes**, and the first
substantive decision belongs to the node after it. That is deliberate. Every judgement that is a
matter of degree — is there speech, is this good enough, is a second voice present — happens at a node
that can return `flag`, so a doubtful answer reaches a human instead of being resolved by a threshold.
ADMIT keeps only the decisions that need no judgement at all, which is why it needs no third
outcome.
