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

## The aggregation rule: count confident windows, do not average scores

**Supersedes the quantile-versus-mean discussion above.** The rule is the one the raster plots already
use: reduce each window to a binary judgement, then count.

Per window, per kind: take the maximum over that kind's family labels. If it clears the confidence
threshold, the window is a detection for that kind. Over the waveform, the kind is present when the
number of confident windows reaches a named per-kind minimum. The evidence is the list of those
windows, which is exactly what a raster row displays.

**This removes the duration dependence for misses**, which is why it is the right rule. Averaging a
score series makes a 0.3 s cough vanish in a ten-minute recording; counting confident windows does not
— one confident window is one confident window whatever surrounds it. The threshold sits on the
per-window score, where the event is locally strong, instead of on an aggregate the recording length
has already diluted.

**It moves the duration dependence onto false positives instead, and that is the cost to carry.** A
longer recording offers more windows and therefore more chances for a spurious confident one, so a
"one window is enough" rule grows less safe as recordings lengthen. Measured, on the verified
recording at a 500 ms window: **`Snore` cleared 0.5 in 16 windows and there is no snoring in the
file** — more windows than `Cough`, which was correct in all 12 of its own. A one-window rule reports
`Snore` present.

So the per-kind minimum count is a real parameter and not a formality, and it is unfitted like the
rest. Two things keep that safe here: the node can `flag` rather than having to choose, and per D6 span
detection downstream adjudicates and can withdraw what this stage admitted. Presence is deliberately
a liberal pre-filter; it is not the verdict.

## The aggregation function

Each detector runs on its **own default window**. They do not align, and that is what shapes the
function.

| detector | window | hop | windows over a 14.03 s file |
| --- | --- | --- | --- |
| YAMNet | 0.96 s | 0.48 s | 29 |
| AST | 0.96 s | 0.48 s | 29 |
| HeAR | **2 s, fixed** | 0.25 s | **50** |
| CrisperWhisper | — | — | not a grid: tokens with timings |

```
taxonomy(audio) -> {kind: Estimate}

for detector d in {yamnet, ast, hear, crisperwhisper}:
    for kind k:
        # 1. fold labels into the kind, per window: max over the family
        series[d][k] = [ max(w.scores[l] for l in family(d, k)) for w in windows(d) ]

        # 2. threshold per (detector, kind) -- HeAR's 0.5 is not YAMNet's 0.5
        hits[d][k] = [ s >= tau[d][k] for s in series[d][k] ]

        # 3. count, and the detector's own verdict
        n[d][k]    = count(hits[d][k])
        says[d][k] = present  if n[d][k] >= min_n[d][k]
                     absent   if n[d][k] == 0
                     unsure   otherwise

# 4. combine verdicts, not counts
kind[k] = Estimate(
    present  if enough detectors say present,
    absent   if every detector that can see k says absent,
    unsure   otherwise,
    evidence = the confident windows per detector,
)
```

**Counts are combined at the verdict level, never summed across detectors.** A 2 s HeAR window and a
0.96 s YAMNet window are not the same unit, so `n[hear][k] + n[yamnet][k]` is a number with no
meaning. Each detector reaches its own verdict on its own grid; the verdicts combine.

**CrisperWhisper has no grid**, so steps 1–3 do not apply to it. Its contribution is the presence of
token types — did any `[cough]` or `[breath]` appear, were there words at all — and it enters step 4
as a verdict like the others.

**A raw window count is not hop-invariant, and this is the trap the table above sets up.** HeAR's 2 s
window on a 0.25 s hop means a single event sits inside roughly eight consecutive windows, so it
produces a run of about eight hits; the same event gives YAMNet two or three. Measured on the verified
file, the same 14.03 s yields 50 HeAR windows against 29 for YAMNet. So `min_n = 3` is a far weaker
requirement for HeAR than for YAMNet, and setting one number for both would be an unmeasured decision
disguised as a shared default.

Two ways to remove the dependence, and the second is better:

- express `min_n` in units of the detector's own hop, so the parameter means the same duration
  everywhere;
- or **count contiguous runs rather than windows.** One event produces one run whatever the hop, which
  is what "a detection" should mean, and it makes `min_n` a count of events rather than of frames.

Runs also carry the evidence a reader wants — where, and how long — for free.

**What the family fold must not do.** Step 1 takes a max over the kind's labels, so it is deliberately
insensitive to which member of a family fired. That is the point: one verified exhalation returned
`Breathing` 0.89, `Sigh` 0.77, `Gasp` 0.72 and `Sneeze` 0.70 at once, and an argmax over those names
is reading noise. But the fold must be over the *kind's own* family only. `Snore` is in the airway
family and fired 16 times on a file with no snoring, so a family drawn too wide imports its own false
positives at full strength through the max.

## AST's window is 10.24 s, and aggregate mode is why that is fine

AST takes 1024 frames at a 10 ms hop, so its input is **10.24 s**, and its feature extractor pads or
truncates every clip to exactly that. Measured:

| audio fed | model input | real | padding |
| --- | --- | --- | --- |
| 0.96 s | 1024 × 128 | 9% | **91%** |
| 2.00 s | 1024 × 128 | 20% | 80% |
| **10.24 s** | 1024 × 128 | **100%** | **0%** |

So feeding AST anything shorter than its window buys nothing and pads the remainder with silence. Use
10.24 s.

**Grids, on each detector's own default:**

| detector | window | hop | windows over 14.03 s | role |
| --- | --- | --- | --- | --- |
| YAMNet | 0.96 s | 0.48 s | 29 | window series |
| HeAR | 2 s fixed | 0.25 s | 50 | window series |
| AST | 10.24 s | — | 1 | file-level verdict |
| CrisperWhisper | — | — | tokens | file-level verdict |

**This is aggregate detection mode, and that resolves the grid mismatch rather than complicating it.**
Nothing here is localised, so the four grids never share a timeline and do not need reconciling. Each
detector answers one question on its own terms — *is this kind present in this recording* — and the
verdicts combine. A detector whose window spans the whole file is not handicapped for a presence
question; it answers directly what the others answer by counting.

That also settles the hop-invariance point recorded above: it would matter if a window count were read
as a count of events, which is a localisation claim. Here the count is evidence for a presence verdict
and nothing more.

**What stays live is false-positive accumulation**, because it is a property of aggregate mode itself:
more windows means more chances for a spurious confident one. `Snore` clearing 0.5 in 16 windows on a
file containing no snoring is the measured case, and it is why the per-kind minimum count is a real
parameter rather than a formality.

## The kinds

Four, and they are the branches. Elements are the branch's internal vocabulary and TAXONOMY does not
separate them.

| kind | elements |
| --- | --- |
| **airway** | inhalation, exhalation, cough, throat clear |
| **phonation** | sustained vowel, pitch glide, loud phonation |
| **speech** | syllable repetition, word production, connected speech, singing |
| **imitation** | vocal imitation of a non-vocal target |

Measured but never branch-selecting, because each changes a decision without being anyone's target:
other-speaker speech, laughter, crying, environmental sound, device and handling noise, silence.

## The aggregator in detail

### 1. Who may vote on what

Eligibility comes before any threshold. A detector is eligible for a kind only if it has labels that
can express the kind and is not disqualified on measurement.

| kind | YAMNet | AST | CrisperWhisper | HeAR | eligible |
| --- | --- | --- | --- | --- | --- |
| **airway** | 10 labels | same label space | `[breath]`, `[cough]` tokens | 6 of its 8 classes | 4 |
| **speech** | `Speech` + 17 | same label space | words — the only source | **barred** | 3 |
| **phonation** | no label | no label | absence of words is not evidence *for* it | no label | **0** |
| **imitation** | fires the imitated class | same | no words | no label | **0** |

HeAR is barred from speech, not merely weak on it: on the verified speech span it reports `Snore` 0.88
and `Speech` 0.01 across six independent measurements. A detector that confidently assigns speech
elsewhere casts a wrong vote, not a weak one.

For phonation, only `loud phonation` has any labels at all (`Shout`, `Yell`, `Bellow`, `Screaming`,
`Whoop`) and there are none for sustained vowel or pitch glide. Whether a prolonged vowel fires
something anyway is unmeasured, so this is an open measurement rather than a settled zero.

### 2. Each detector's own verdict

**Series detectors — YAMNet, HeAR.** Fold the kind's family by max per window, threshold, count:

```
n = |{ w : max_{l in family(d,k)} score(w, l) >= tau[d][k] }|
present  if n >= min_n[d][k]
absent   if n == 0
unsure   otherwise
```

**File-level detector — AST.** One 10.24 s window covers the recording, so there is nothing to count.
It needs a band, or every score becomes a decision:

```
present  if s >= tau_hi[k]
absent   if s <  tau_lo[k]
unsure   otherwise
```

**Token detector — CrisperWhisper.** No grid, no score. For speech, `present` if it returned any words.
For airway, `present` on a token of the kind's own type — but a token of the *wrong* type inside the
family reads `unsure`, because it split one verified cough into `[UH]` and `[breath]`. Its timing is
trustworthy where its label is not: onset −26 ms and offset −14 ms on the other cough.

### 3. Independence: count families, not detectors

YAMNet and AST share the AudioSet corpus and label space and can be wrong together.

| family | members |
| --- | --- |
| A — AudioSet | YAMNet, AST |
| B — lexical | CrisperWhisper |
| C — health-acoustic | HeAR |

**Airway has three families; speech has two.** So "two families must agree" is a modest bar for airway
and near-unanimity for speech. `min_families` is per kind and cannot be one global number.

### 4. What defines a kind as present

| state | condition |
| --- | --- |
| **present** | at least `min_families[k]` eligible families say present |
| **absent** | **every** eligible family says absent |
| **undecided** | families disagree, or any is unsure |
| **unobservable** | the eligible set is empty |

Presence needs agreement; absence needs **unanimity**. A low score means either "not there" or "there
but quiet or masked", and masked is the case this workflow exists to catch, so no single family may
retire a kind alone.

**`unobservable` is not a synonym for absent.** Phonation and imitation have no eligible detector.
"Everything looked and found nothing" and "nothing here can see this" are different claims, and
recording them identically is how a kind gets silently dropped.

### 5. Pass, flag and fail

Over the observable kinds:

| outcome | condition |
| --- | --- |
| **fail** | every observable kind is `absent` |
| **flag** | any observable kind is `undecided` |
| **pass** | every observable kind is decided, and at least one is `present` |

**Unobservable kinds are excluded from the gate.** Counted as undecided they would flag every file
forever — phonation is unobservable on every recording, so the gate would be a constant and the flag
would carry no information. Excluding them has a consequence that must be published rather than
hidden: **an unobservable kind is never screened**, so its branch is hint-selected or never runs. The
output names which kinds were never examined.

### 6. The parameters, none of them fitted

`tau[d][k]`, `min_n[d][k]`, `tau_hi[k]`, `tau_lo[k]`, `min_families[k]`. No labelled corpus exists to
fit any of them, and the derivation slots stay empty rather than holding invented literals.

Three things keep that honest: the node can `flag`, so doubt is not forced into a guess; absence needs
unanimity, so the destructive outcome is hardest to reach; and span detection downstream adjudicates
and can withdraw what presence admitted.

## Revision: three kinds, the third a residual

**Supersedes the four-kind table and the `unobservable` state.** Two kinds are detected; the third
takes what neither claimed, subject to a positive gate of its own.

| kind | how it is reached |
| --- | --- |
| **airway** — non-voice, non-speech | detected: breath, cough, throat clear |
| **speech** | detected: words |
| **voice / no-words** | **residual** — voiced, and claimed by neither of the above |

Order matters, because it is what makes the residual well-defined: airway and speech each claim what
they detect, and whatever voicing remains is voice/no-words.

**This removes `unobservable`.** Phonation and imitation had no eligible detector and so could never be
decided; as members of the residual they need none. Sustained vowel, pitch glide and a child imitating
a dog are all *voiced and wordless*, which is exactly what the residual is. The label space never had
to contain the concept.

### The residual needs a positive gate, or it collects everything

A residual defined only by exclusion would absorb silence, room tone and environmental sound. The gate
is acoustic rather than label-based: **is this vocal-tract produced.** Candidates are F0 presence, HNR,
and F0 dispersion.

Periodicity is measured, and this is the **only parameter in this node with any empirical basis**:

| region | RMS | F0 | periodicity |
| --- | --- | --- | --- |
| sustained voicing, 3.20-3.40 s | 0.0188 | 87.4 Hz | **0.933** |
| sustained voicing, 4.40-4.60 s | 0.0161 | 88.1 Hz | **0.934** |
| quiet stretches | 0.0004-0.0007 | unstable | **0.22-0.44** |

Normalised autocorrelation separates the two by a wide margin on this recording, with an RMS floor
needed as well so that periodic room tone cannot pass. One recording is one recording — but every
other threshold here has nothing behind it at all, so this is the one place a derivation can start.

**F0 dispersion is the within-category discriminator, not the gate.** A sustained vowel holds F0
roughly constant, a pitch glide sweeps it smoothly, and unvoiced noise produces F0 estimates that
scatter incoherently. So dispersion separates the residual's own members once the gate has admitted
them, which is a branch question rather than a screening one.

### What this changes elsewhere

**Laughter and crying move.** They were listed as measured-but-never-branch-selecting. They are voiced
and wordless, so under a residual they now select voice/no-words. That may be right — a cry is a target
in a pediatric protocol — but it is a change of behaviour and not a tidy-up.

**Cough is voiced, so voicing alone cannot separate airway from the residual.** Measured, a diarizer's
raw posterior responds to the verified coughs at 0.574 and 0.906. Detection order is what resolves it:
airway claims the cough before the residual sees it. Breath is the opposite case — unvoiced, so the
gate would never admit it, and it too must come from airway detection rather than from voicing.

**The gate does not need a label space**, which is why it closes the hole that the label mapping could
not. `Human voice`, `Human sounds` and `Respiratory sounds` are all absent from YAMNet's 521, so no
union of labels can ask "did a vocal tract make this". An acoustic gate asks it directly.
