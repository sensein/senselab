# TAXONOMY — what is in this recording

Decided 2026-08-19/20. This file governs.

## What it decides

**Which kinds of sound this recording contains.** Not where they are, not how good they are, not how
well any of them can subsequently be measured. Localisation is a branch question.

`kinds` are the branches: one presence answer each, and a branch runs when its kind is present. So
TAXONOMY discriminates at exactly branch granularity — finer would produce a distinction nothing
consumes, coarser could not route.

## Signature

```
taxonomy(audio) -> fail(reason) | flag(reason, kinds) | pass(kinds, residual_windows)
```

| port | direction | type | meaning |
| --- | --- | --- | --- |
| `audio` | in | decoded audio | from ADMIT, the recording as supplied |
| `fail` | out | reason | no branch would run |
| `flag` | out | reason, kinds | a human decides; the partial answer travels with it |
| `pass` | out | kinds, residual_windows | one presence `Estimate` per kind, plus the voiced windows the residual claimed |

Each `Estimate` carries its evidence count and spread, so `kinds` is the whole product — there is no
separate evidence port for a consumer to join back.

## The three kinds

| kind | contains | reached by |
| --- | --- | --- |
| **airway** — non-voice, non-speech | inhalation, exhalation, cough, throat clear | detection |
| **speech** | syllable repetition, word production, connected speech, singing | detection |
| **voice / no-words** | sustained vowel, pitch glide, loud phonation, vocal imitation, laughter, crying | **residual**, with an acoustic gate |

Elements are each branch's internal vocabulary. TAXONOMY does not separate them.

**Order defines the residual.** Airway and speech claim what they detect; voicing that neither claimed
is voice/no-words. Nothing needs a label for "sustained vowel" or "imitation", which is what makes the
residual worth having: no label space contains those concepts, and a residual does not need one.

## The screening set

| detector | contributes | limit |
| --- | --- | --- |
| **YAMNet** | 521 AudioSet labels; the broadest vocabulary and the only explicit `Silence` class. Resolves events 1.1 s apart. | No human-vocalic roll-up exists in the 521 — `Human sounds`, `Human voice`, `Respiratory sounds` are all absent — so only a union of specific labels is available. |
| **AST** | A second AudioSet opinion, and it does disagree usefully: it called one verified cough `Throat clearing` 0.96 where YAMNet said `Cough` 1.000. | Same corpus and label space as YAMNet, so the two can be wrong together. |
| **CrisperWhisper** | The only source of **words**, so the speech kind rests on it. Also emits `[breath]`, `[cough]`, `[UH]` with timings. | Its non-lexical *labels* are unreliable while its timings are not: it bounded one verified cough to onset −26 ms and offset −14 ms, and split the other into `[UH]` plus `[breath]` covering 440 of its 640 ms. |
| **HeAR** | Strongest on breath — 0.998 and 0.997 on two verified breaths, where YAMNet reads 0.726 and 0.893. | **Barred from the speech kind.** On verified speech it reports `Snore` 0.88 and `Speech` 0.01 across six measurements: not a weak vote, a wrong one. Input fixed at 2 s and needs ~160 ms of an event, so it is a presence gate, never a locator. |

## Windows: each detector on its own default

| detector | window | hop | windows over 14.03 s | shape |
| --- | --- | --- | --- | --- |
| YAMNet | 0.96 s | 0.48 s | 29 | score series |
| HeAR | 2 s fixed | 0.25 s | 50 | score series |
| AST | 10.24 s | — | 1 | one file-level score |
| CrisperWhisper | — | — | tokens with timings | no grid |

AST takes 1024 frames at a 10 ms hop and its feature extractor pads or truncates every clip to exactly
that, so feeding it less buys nothing: 0.96 s in is 9% real and 91% padding.

**Aggregate detection mode resolves the grid mismatch rather than complicating it.** Nothing here is
localised, so the grids never share a timeline. Each detector answers one question on its own terms and
the verdicts combine. A detector whose window spans the file answers presence directly rather than by
counting.

## Eligibility, before any threshold

| kind | YAMNet | AST | CrisperWhisper | HeAR | acoustic gate |
| --- | --- | --- | --- | --- | --- |
| **airway** | 10 labels | same space | `[breath]`, `[cough]` | 6 of its 8 classes | — |
| **speech** | `Speech` + 17 | same space | words | **barred** | — |
| **voice / no-words** | — | — | absence of words is a precondition, not evidence | — | **the gate** |

## Each detector's own verdict

**Series detectors — YAMNet, HeAR.** Fold the kind's family by max per window, threshold, count:

```
n = |{ w : max_{l in family(d,k)} score(w, l) >= tau[d][k] }|
present  if n >= min_n[d][k]
absent   if n == 0
unsure   otherwise
```

The fold is deliberately blind to *which* family member fired: one verified exhalation returned
`Breathing` 0.89, `Sigh` 0.77, `Gasp` 0.72 and `Sneeze` 0.70 at once, and an argmax over those names
reads noise. But the family must be drawn to the kind and no wider. `Snore` is an airway-family member in both
label spaces, and HeAR's `Snore` cleared 0.5 in **16 of 136 windows** on a file containing no snoring —
more than its `Cough`, which was correct in all 12 of its own. A max over a family imports that at full
strength. (Those counts are HeAR's, from a 500 ms sweep at 100 ms hop, not YAMNet's; YAMNet's `Snoring`
reached 0.819 in 2 of its 29 native windows on the same file.)

**File-level detector — AST.** One window covers the recording, so there is nothing to count. It needs
a band, or every score becomes a decision:

```
present  if s >= tau_hi[k]
absent   if s <  tau_lo[k]
unsure   otherwise
```

**Token detector — CrisperWhisper.** For speech, `present` on any words. For airway, `present` on a
token of the kind's own type — but a wrong-type token inside the family reads `unsure`, not `present`.

## Independence: count families, not detectors

| family | members |
| --- | --- |
| A — AudioSet | YAMNet, AST |
| B — lexical | CrisperWhisper |
| C — health-acoustic | HeAR |

**Airway has three families; speech has two.** So "two families must agree" is a modest bar for airway
and near-unanimity for speech. `min_families` is per kind and cannot be one global number.

## The residual's gate

A residual defined only by exclusion would absorb silence, room tone and environmental sound. Its gate
is acoustic rather than label-based, and asks the one question no label union can: **did a vocal tract
make this.** `Human voice` and `Respiratory sounds` are not in YAMNet's 521, so no set of labels can.

Normalised autocorrelation, with an RMS floor so periodic room tone cannot pass:

Measured on **`streaming-audio-2026-08-20T01-51-14-067Z.wav`**, which is **not** the human-verified
recording:

| region | RMS | F0 | periodicity |
| --- | --- | --- | --- |
| 3.20-3.40 s | 0.0188 | 87.4 Hz | **0.933** |
| 4.40-4.60 s | 0.0161 | 88.1 Hz | **0.934** |
| quiet stretches | 0.0004-0.0007 | unstable | **0.22-0.44** |

**The provenance matters and weakens the claim.** That recording carries no human labels, and
"sustained voicing" is an inference *from these very numbers* — F0 stable near 87 Hz at periodicity
0.93 for about four seconds — not something anyone verified by ear. Using them to justify a voicing
gate is therefore circular for the purpose of identifying voicing.

The numbers also do not transfer. At the identical timestamps the human-verified recording
`streaming-audio-2026-07-30T04-21-56-487Z.wav` reads periodicity **0.558** and **0.134**, because those
windows fall in a verified breath and a verified-empty stretch respectively. Two recordings, two
different worlds.

What survives is narrower than "an empirical basis" and worth stating exactly: **within one unlabelled
recording, normalised autocorrelation separates a sustained periodic stretch from the quiet stretches
around it by a wide margin.** That is a real acoustic contrast and a reasonable place for a derivation
to start. It is not a fitted threshold, and it is not validated against any label. HNR is an
alternative or an addition and is unmeasured.

**F0 dispersion discriminates within the residual, it does not gate it.** A sustained vowel holds F0
roughly constant, a glide sweeps it smoothly, and unvoiced noise scatters F0 estimates incoherently.
That separates the residual's members once admitted, which is a branch question.

**Voicing alone cannot separate airway from the residual, and detection order is what does.** Cough is
voiced — a diarizer's raw posterior reads 0.574 and 0.906 on the two verified coughs — so the gate would
admit it if airway had not already claimed it. Breath is the mirror case: unvoiced, so the gate would
never admit it, and it must come from airway detection or not at all.

### Two tracks carry the whole residual

The gate and the discrimination inside it are both functions of the same pair: an **energy** track and
a **periodicity/F0** track. Nothing about this category needs a label space.

| residual member | what distinguishes it, from those two tracks |
| --- | --- |
| sustained vowel | high periodicity held for a long run, F0 roughly constant |
| pitch glide | high periodicity, F0 sweeping monotonically — a trajectory, not a level |
| loud phonation | energy high *relative to the rest of the recording*; periodicity unremarkable |
| maximum phonation time | the **duration** of the voiced run is itself the measurement |
| laughter | periodicity intermittent in bursts, energy amplitude-modulated |
| crying | high periodicity at a high absolute F0, sustained with modulation |
| vocal imitation | nothing reliable — the trajectory can be anything |

So six features, all cheap and all derived from those two tracks: periodicity level, F0 level, F0
trajectory, energy level, energy modulation rate, and voiced-run duration. That set is the residual
branch's instrument, and TAXONOMY needs only the first to gate.

**Task hints resolve what the tracks cannot.** A hint names the task, so it says which member to expect
— `prolonged-vowel`, `glides`, `loudness`, `maximum-phonation-time` — and the features then confirm or
contradict it rather than having to identify it unaided. Imitation is the case that needs this: a child
imitating a dog passes the gate, because the vocal tract made it, and no trajectory distinguishes it
from any other wordless voicing.

The hint conditions the **decision**, never the measurement. The two tracks are computed the same way
with or without one, so a hint can never manufacture evidence — it can only choose among readings of
evidence that already exists, and a hint contradicted by the tracks is itself a finding.

**The residual's window set is published, not recomputed.** The overlap bookkeeping below produces a
concrete result — which voiced windows neither airway nor speech claimed — and that result leaves on
the `residual_windows` port. A branch that had to recompute it would become downstream of its siblings
and would stall whenever airway was absent, which is the same producer-consumer gap that has bitten
this project before: a value the code is written to honour never reaching the code that honours it.

**The residual is the one place grids must be compared.** Deciding that voicing was *not* claimed needs
the gate's voiced windows checked for time overlap against the airway and speech detectors' confident
windows. That is window-level bookkeeping, not event localisation, but it is a comparison across grids
and the rest of this node has none.

## What defines a kind as present

| state | condition |
| --- | --- |
| **present** | at least `min_families[k]` eligible families say present — or, for the residual, the gate admits voicing that neither other kind claimed |
| **absent** | **every** eligible family says absent |
| **undecided** | families disagree, or any is unsure |

Presence needs agreement; absence needs **unanimity**. A low score means either "not there" or "there
but quiet or masked", and masked is the case this workflow exists to catch, so no single family may
retire a kind alone.

## Pass, flag and fail

| outcome | condition |
| --- | --- |
| **fail** | every kind is `absent` |
| **flag** | any kind is `undecided` |
| **pass** | every kind is decided, and at least one is `present` |

**`flag` is per file, not per kind.** One undecided kind flags the whole recording. A partial run —
certain kinds proceeding while an uncertain one waits — would publish results whose completeness depends
on something a human has not yet looked at, and nothing downstream could distinguish that from a file
where the uncertain kind was genuinely absent.

## What TAXONOMY does not do

- No localisation: no spans, onsets or offsets. Branch questions.
- No per-kind flagging.
- No judgement about how measurable a present kind will turn out to be. Whether a branch can do its job
  on this recording is that branch's finding.
- No enhancement or separation. If a branch wants a channel in which one element survives, that is an
  operation inside the branch.

## Parameters

`tau[d][k]`, `min_n[d][k]`, `tau_hi[k]`, `tau_lo[k]`, `min_families[k]`, and the gate's periodicity and
RMS floors. Only the gate has a measurement behind it. The rest have no labelled corpus to fit them on
and their derivation slots stay empty rather than holding invented literals.

Three things keep that honest: the node can `flag`, so doubt is not forced into a guess; absence needs
unanimity, so the destructive outcome is hardest to reach; and span detection downstream adjudicates and
can withdraw what presence admitted.

## The live risk

False-positive accumulation is a property of aggregate mode: more windows means more chances for a
spurious confident one. HeAR's `Snore` clearing 0.5 in 16 of 136 windows on a file containing no
snoring is the measured case, and it is why `min_families` and `min_n` are real parameters rather than
formalities.
