# BRANCH — airway. Which types, and where.

Drafted 2026-08-19. This file governs the airway branch; where
`design.md` disagrees it is stale. An earlier airway draft, now deleted, was a
superseded draft of this same branch and is not a starting point.

## What it decides

TAXONOMY has already established that airway sound is present. This branch answers the two questions
it deliberately left alone: **which types of non-voice, non-speech sound are in this recording, and
where each one is.**

It is entitled to disagree with the presence claim that started it. TAXONOMY records that "span
detection downstream adjudicates and can withdraw what presence admitted", and withdrawal is one of
this node's outcomes — as a `flag`, never silently.

## Signature

```
airway(audio, hint?) -> fail(reason) | flag(reason, events) | pass(events)
```

| port | direction | type | meaning |
| --- | --- | --- | --- |
| `audio` | in | decoded audio | from ADMIT, the recording as supplied |
| `hint` | in | task name, or absent | conditions decisions; never reaches a measurement |
| `fail` | out | reason | the instrument cannot measure this recording |
| `flag` | out | reason, `events` | a human decides; the same product travels with it |
| `pass` | out | `events` | typed, onset-anchored spans |

**One product, not two.** The goal names types and spans, and a naive reading gives two ports — a type
set and a span list — which a consumer then has to join. Every published type has at least one located
event (that is the necessity rule below), so the type set is a projection of `events` and does not need
its own port. This follows TAXONOMY, where each `Estimate` carries its own evidence "so `kinds` is the
whole product".

**No port for the TAXONOMY verdict.** The branch reads the audio, not the presence answer. Score series
already computed upstream may be reused through the content-addressed cache — recomputing YAMNet on the
same samples at the same window is a cache hit, not a second measurement — but the *verdict* does not
cross the boundary. If it did, a presence claim would raise the confidence of the localisation that is
supposed to be able to withdraw it.

**No port for VAD, diarization or segmentation.** Measured, not stylistic: a diarizer's raw posterior
reads exactly 0.0000 on both verified breaths and a VAD reads 0.005, so a speech mask admits nothing for
breath; and the same diarizer reads 0.574 and 0.906 on the two verified coughs against 0.790 on real
speech, so a "not speech here" mask deletes cough preferentially. Either direction of use is refuted.

**No port for enhanced audio.** Not a prohibition — nothing routes it — but a property of the tools: the
repo default takes breath 1 to −26.4 dB and `MossFormer2_SE_48K` takes the two breaths to −37 and
−40 dB, so no node needing breath can use them.

## The instruments, and the one structural fact that organises them

| instrument | what it is measured to do | where it fails |
| --- | --- | --- |
| **DSP envelope** | cough onsets to ±5 ms across both verified coughs over a rise sweep Δ ∈ [3, 20] dB; 0 ms at Δ = 6 dB. 6 of 6 verified events found — the only instrument with complete recall | offsets refuted: −449 to +20 ms, moving ~180 ms across the same sweep, one cough never within 267 ms. Precision unmeasured; as a global peak-picker a slightly small delta produced 119–120 false onsets at a uniform 85 ms spacing. Cannot name anything |
| **YAMNet** 0.96 s / 0.48 s | resolves the two coughs 1.1 s apart — above 0.5 at 7.20, 7.68, 8.16, a dip at 8.64, then 9.12, 9.60 — and the two breaths 1.81 s apart. Explicit `Silence` class, which fires in the verified-empty gaps | 0.48 s response width at 10%, unfixable by hop: shrinking the hop tenfold made the speech leading edge *worse* (1.06 s → 1.34 s early). A breath returns family mass, not a label: `Breathing` 0.89 with `Sigh` 0.77, `Gasp` 0.72, `Sneeze` 0.70 on one event. `Snore` cleared 0.5 in 16 windows on a file containing no snoring — more than `Cough`, which was correct in all 12 of its own |
| **HeAR**, native 2 s / 0.25 s | a near-ideal presence gate: 40 ms of real cough inside the window saturates it, so false negatives are rare | cannot localise or count. An event of duration D elevates every window centre within D + 2 s, so the two breaths (1.81 s apart) and the two coughs (1.12 s) each merge into one plateau. Inter-cough intervals within a bout run 0.3–0.5 s |
| **HeAR**, excerpt mode | fed a short excerpt embedded in an otherwise-silent 2 s buffer it both localises and separates the 1.1 s cough pair: at a 160 ms window swept at 20 ms hop, `Cough` fires in two sharp plateaus peaking 0.998 and all 24 detections have their centre inside a verified cough | needs ~160 ms — a cliff, 0.063 → 0.220 → 0.999 at 40/80/160 ms, and 40 ms produces nothing at any of 700 positions. Above 160 ms false positives grow faster than true ones: `Breathe` centres landing in verified-empty audio go 3 → 23 → 38 across 160/320/640 ms |
| **AST** 10.24 s, file-level | the only instrument that ever proposed `Throat clearing`, at 0.93–0.96 | it proposed it on a verified **cough**, where YAMNet read `Cough` 1.000. Same corpus and label space as YAMNet, so it is not an independent family. One window over the file: no localisation at all |
| **CrisperWhisper** | timings: cough 1 bounded to onset −26 ms, offset −14 ms, 580 ms against a verified 568 ms, 97.5% coverage | labels: it split cough 2 into `[UH]` (9.60–9.94) plus `[breath]` (10.12–10.22), two mislabelled tokens covering 440 of its 640 ms with a 180 ms gap. A speech prior imposed on a non-speech event |
| **Energy modulation** | available, unmeasured here | see the section on it below |

**The structural fact: the full-recall instrument cannot name, and every instrument that can name
under-covers.** The envelope found 6 of 6 verified events and identified none; CrisperWhisper, HeAR and
YAMNet/AST each found 5 of 6, all missing the same 202 ms mouth sound, and each named what it found with
much better precision than the envelope's. That is a conjunction, not a vote, and it is the branch's
central rule:

> **An envelope onset is necessary. Classifier type support is necessary. Neither is sufficient.**

The rule is derived rather than chosen. Requiring the envelope costs nothing in recall on the only
labelled file — it had all six events — and it is the only defence against the one measured shared false
positive: on 6.60–7.10 s, HeAR scored `Breathe` 0.49 and YAMNet's `Breathing` fires across 6.72–7.68,
and there is no envelope onset anywhere in that stretch (the recorded onsets are 0.893, 2.275, 5.308,
7.924, 9.609). Two models agreeing does not make it safe, and an ensemble of the two would not have
caught it; the third instrument would. Conversely, requiring a classifier is the only defence against
the envelope's unmeasured precision, whose known failure mode is 120 spurious onsets, not one.

## What the node does, in order

### 1. Propose regions — YAMNet, folded by family

Fold the airway family by max per window, threshold, and take maximal runs of firing windows as candidate
regions, dilated by the half-window on each side. The family is the ten airway labels: `Breathing`,
`Cough`, `Gasp`, `Pant`, `Sniff`, `Snort`, `Wheeze`, `Snoring`, `Sigh`, `Throat clearing`.

The fold is blind to *which* member fired, for the reason TAXONOMY gives: one verified exhalation
returned `Breathing` 0.89, `Sigh` 0.77, `Gasp` 0.72 and `Sneeze` 0.70 at once, and an argmax over those
names reads noise. The consequence here is stronger than it is upstream — **no YAMNet label ever names a
type in this branch.** It proposes where to look and nothing else. That is what makes the 16 false
`Snoring` windows harmless: they cost a scan, not a label.

This step is deliberately over-generating. Region proposal is the one place in the branch where a false
positive is cheap, because two independent stages downstream can kill a region and neither can resurrect
one that was never proposed.

`Silence` is read on the negative side only, and only for the offset bracket in step 4.

### 2. Localise the onset — the envelope, inside proposed regions only

Rise detection on a 1 ms envelope at Δ dB above a floor estimated over the region's leading context.
Confined to proposed regions, which is what makes the 119–120-false-onset flood unreachable: that failure
was global peak-picking with a slightly small delta, and inside a region proposed by an independent
instrument the same delta has almost no unproposed audio to fire in.

The onset is a point with a tolerance, and **the tolerance is per type, because only cough's is
measured**:

| type | onset tolerance | what supports it |
| --- | --- | --- |
| cough | ±5 ms | both verified coughs across Δ ∈ [3, 20] dB; exactly 0 ms at Δ = 6 dB |
| breath | not better than ~25 ms | the reference itself disagrees by that much: the verified breath onsets are quoted as 2.275 s and 5.308 s, while the verified breath *windows* begin at 2.2995 s and 5.3285 s — 24.5 ms and 20.5 ms apart. No breath onset claim can be tighter than the slack inside the label it is scored against |
| mouth non-speech sound | not better than ~114 ms | the single verified instance has a DSP onset at 0.893 s inside a verified window of 0.779–0.981 s, so the envelope entered 114 ms late on the one non-cough event where both numbers exist |

The ±5 ms figure is therefore a **cough** result, n = 2, one healthy adult, dry close mic (Brouhaha C50
median 28.5 dB). Publishing it as the branch's onset accuracy would generalise a cough measurement to a
class of events where the two available comparisons are 25 ms and 114 ms.

Rise time and level step travel with the event as continuous features and never as thresholds:
9–17 ms with a 45–49 dB step for the coughs, 60–127 ms with 20–29 dB for the breaths, n = 2 per class
from one healthy adult. They describe a *healthy adult voluntary* cough and are expected to fail hardest
where the signal matters most — reduced peak cough flow, absent glottic closure, infant cough,
prolonged expiration in COPD.

A second onset estimate comes free from CrisperWhisper token edges (+20, +32, −26, −10 ms against the
four verified windows). Two independent instruments agreeing to within ~30 ms is the strongest onset
evidence available, so disagreement beyond that is a flag, not an averaging opportunity. HeAR's native
onsets are barred from this step: on breath 2 its error is **+532 ms**, its 2 s window showing through.

### 3. Type the event — HeAR in excerpt mode

For each proposed region, sweep a 160 ms window at a 20 ms hop, each excerpt embedded in an
otherwise-silent 2 s buffer, and read the detector's class posterior. Maximal runs above threshold are
plateaus; a plateau's class is the event's type candidate.

Three things about this mode have to be stated plainly, because it is the one step whose supporting
measurement contradicts an earlier one:

**It reads the detector head, never the encoder.** The measurement that padding "destroys" HeAR is about
embedding geometry: the centred cosine between the same event under different framings runs 0.0–0.5 and
`native|real_context` runs −0.21 to +0.26 against a class margin of ~0.9. That refutes padded input for
embeddings, nearest-neighbour work and any use of the 512-d vector. It does not refute the posterior,
which in excerpt mode peaks at 0.998 with all 24 detections centred inside a verified cough. This branch
uses the posterior and never the embedding, and that restriction is what keeps both measurements true at
once.

**The 40 ms and 160 ms requirements are not in conflict; they are the same fact from two sides.** With
*real* surrounding context, 40 ms of cough saturates the window — which is exactly why native mode cannot
localise: it is firing on neighbouring material. With *silence* as the surround, 40 ms produces nothing at
any of 700 positions and the cliff runs 0.063 → 0.220 → 0.999 across 40/80/160 ms. The window fires on
whatever content is in it; native mode simply always has 2 s of it.

**The window does not grow.** 160 ms is the operating point with a measured cliff below it and measured
false-positive growth above it: `Breathe` centres landing in verified-empty audio go 3 → 23 → 38 across
160/320/640 ms. Breath is measured to need 160–320 ms, so 160 ms sits at the *bottom* of breath's
requirement band while being the minimum of its false-positive curve. That is a choice the evidence does
not force — see the open questions.

The window geometry also fixes what a plateau can and cannot say. Because 160 ms of content is required,
the first firing excerpt lies up to ~160 ms after the true onset and the last up to ~160 ms before the
true offset. So a plateau is **systematically inside the event at both ends**: it cannot improve on the
envelope's onset, and its right edge is a lower bound on the offset and never an offset.

**AST contributes candidates, not confidence.** It is the only instrument that has ever proposed
`Throat clearing`, and it did so at 0.96 on a verified cough. It may raise `throat_clear` into a region's
candidate set; it may not raise or lower any candidate's evidence count, because it shares YAMNet's
corpus and label space and is not an independent family.

**CrisperWhisper is barred from typing entirely.** It called a voiced cough phase `[UH]` and the aspirate
tail `[breath]`, which is a speech prior imposed on a non-speech event rather than a random error.

### 4. Bracket the offset — because every instrument stops early

This is the branch's hardest honest problem: **no instrument measured bounds an offset**, and the product
is spans.

What *is* measured is that the errors share a direction. Coverage of the verified windows runs 87–98% for
speech, 64–98% for cough and **10–52% for breath**, and five independent detectors "agree in direction and
all fall short" — every instrument marks where a turbulent event begins and then loses it. The envelope's
offset errors run −449 to +20 ms, so all but one are early. The one +20 ms overshoot is the measured limit
of that agreement.

A one-directional error is not a measurement of the offset, but it is a valid **lower bound on the
extent**. So the branch reports the offset as a bracket:

| side | source | what it means |
| --- | --- | --- |
| `t_last_witnessed_s` | the latest of: the last firing excerpt centre from step 3, the last CrisperWhisper token edge inside the region, and the envelope's offset at the strictest Δ in the sweep | the last time at which something of this type was still observed. A lower bound on the true end, to within the single measured overshoot of +20 ms — not an inequality the measurement guarantees |
| `t_not_after_s` | the earlier of: the first subsequent YAMNet window in which `Silence` clears its threshold, and the next event's onset. Absent if neither exists | the true end is not later than this |

Neither number is an end, and the type below makes them unusable as one.

The lower bound is deliberately a max over three instruments rather than a consensus, because a lower
bound is monotone in the right direction — each additional witness can only push it later, toward the
truth. The risk it carries is a spurious late witness, and the one measured instance is CrisperWhisper's
`[breath]` at 10.12–10.22 s: a wrong *label*, but a token that genuinely lies inside cough 2. That is
why the witnesses are pooled by time and not by label: the question at this step is "was something still
happening", which is the one question CrisperWhisper answers well on this file.

### 5. Fuse, and decide each region

A region becomes a published event only if it has an envelope onset **and** classifier type support from
at least one family other than the one that proposed it. Otherwise it stays in the product as an
unresolved region with its reason, and the branch flags.

| region state | condition | outcome |
| --- | --- | --- |
| event | envelope onset, and a HeAR excerpt plateau over it | published in `events` |
| ambiguous type | two type candidates carry mass and no non-classifier feature separates them | published with both, and flags |
| classifier-only | classifier support, no envelope onset | unresolved region, flags — the 6.60–7.10 s case |
| envelope-only | envelope onset, no classifier will name it | unresolved region, flags — the 202 ms mouth sound |
| grouping-unresolved | more than one onset inside one plateau | one region, all onsets, no count, flags |

Each type candidate is an `Estimate` (`senselab.utils.data_structures.estimate.Estimate`), with
`n_evidence` the number of *independent families* supporting it — A = AudioSet (YAMNet, AST),
B = lexical (CrisperWhisper, barred from typing so never a member here), C = health-acoustic (HeAR),
D = acoustic (envelope features) — and `population` set to the only population any of this was measured
on. Every `Estimate` this branch emits carries the same population string, which is how a consumer sees
that a reverberant field recording of a child is outside the validated range without the branch having to
invent a threshold to reject it.

### 6. Grouping — declared, unsolved, and therefore no counts

Whether a broadband burst, a voiced phase and an aspirate tail are one cough or three events is a third
problem, distinct from timing and from labelling, and nothing measured solves it. HeAR's native detector
fragmented 3 of 4 verified events and CrisperWhisper split 1; HeAR cannot be fixed by better thresholding
because events closer than 2 s are merged before any threshold sees them, and inter-cough intervals within
a bout run 0.3–0.5 s. The physiological rule once proposed for this — a voiced phase within ~400 ms of a
burst belongs to that burst — has no instrument behind it: Praat HNR returns `nan` almost everywhere on
this recording, valid only at the two cough onsets, and pyin rails at its 60 Hz floor through the quiet
stretches.

So the branch **reports no event counts**, no cough-bout structure, and no respiratory rate. A region with
several onsets is emitted as one region carrying all of them and marked grouping-unresolved. `min_gap` is
not a parameter awaiting a number; grouping is a missing instrument.

## The span type

The product's shape is the place a consumer is most likely to be misled, so it is designed against that
specifically.

| field | type | note |
| --- | --- | --- |
| `type` | `cough` \| `throat_clear` \| `breath` \| `mouth_non_speech` \| `unattributed`, or a set of two | see the vocabulary below |
| `type_estimate` | `{type: Estimate}` | `n_evidence` = independent families; `population` = the validated population |
| `onset_s` | float | from the envelope, inside a proposed region |
| `onset_tolerance_ms` | float | per type, per the table in step 2 |
| `extent` | tagged union | `open` \| `witnessed_until { t_last_witnessed_s, overshoot_slack_ms }` \| `bracketed { t_last_witnessed_s, t_not_after_s }` \| `measured { t_end_s, tolerance_ms }` |
| `onsets` | list, when grouping is unresolved | present only in that case |
| `features` | rise ms, level step dB | continuous, never thresholded |
| `resolved_by` | `evidence` \| `hint` | present when a hint chose among readings |

Four rules make the unbounded offset unmistakable:

1. **There is no `end`, `offset_s` or `duration` field, and no two-element `[start, end]` array anywhere
   in the product** — that array is the shape every span consumer already knows how to misread.
2. **`extent` is a tagged union whose variants have different field names**, so `t_last_witnessed_s`
   cannot be read as an end by a consumer who forgot to check a tag, in Python or in JSON.
3. **The `measured` variant exists and nothing can currently construct it.** Its slot is there so that an
   instrument with demonstrated offset accuracy has somewhere to put its answer, and its emptiness is the
   finding. Today the closest thing to a filled slot is CrisperWhisper's −14 ms offset on cough 1, which
   is one event from one superseded checkpoint that mislabelled the other cough; one event is not an
   accuracy.
4. **The branch ships no interval arithmetic.** No `overlaps`, no `contains`, no `duration` helper, because
   each is undefined on an open extent and every one of them is where a fabricated end would be born.

The consequence is deliberate: inter-onset intervals are computable and extent-derived measures are not.
Respiratory rate from onsets is supportable; breath duration, inspiratory-to-expiratory ratio and any
maximum-phonation-style measure are uncomputable from this product rather than wrong in it. Given
10–52% breath coverage across every instrument tried, uncomputable is the correct state.

## The type vocabulary, and the three distinctions it refuses

| type | emitted when | refused distinction |
| --- | --- | --- |
| `cough` | a HeAR excerpt plateau reads `Cough`, with an envelope onset | — |
| `throat_clear` | as above for the throat-clearing class, or AST raised it | **cough vs throat clear is not separable by classifier vote on this evidence**: AST read `Throat clearing` 0.93–0.96 on a verified cough where YAMNet read `Cough` 1.000. Both candidates travel together, and the branch flags |
| `breath` | a plateau reads the breathing class, with an envelope onset | **inhalation vs exhalation has no instrument.** The taxonomy separates them; none of the detectors used here does, and the one plausible inhalation on the labelled file is precisely the unresolved 6.60–7.10 s stretch. `breath` is emitted with its phase unresolved rather than guessed |
| `mouth_non_speech` | never on classifier evidence, because none exists | the single verified instance — 202 ms, 0.779–0.981 s — was missed by HeAR, YAMNet, AST and CrisperWhisper alike, and its miss is not window geometry: its elevated interval is covered, HeAR's top-1 posterior there is ~0.05, the lowest point in the recording. It reaches the product as an **unresolved region**, which is what "a sub-threshold single-model score is a question, not a label" requires of a zero-model score |
| `unattributed` | an envelope onset in a proposed region that no plateau covers | may be a sub-160 ms event, which is exactly the blind spot the mouth sound sits in |

The eight airway labels not in this vocabulary — `Snoring`, `Wheeze`, `Sneeze`, `Gasp`, `Sigh`, `Pant`,
`Sniff`, `Snort` — are in the *proposal* family and may never name a type. On the only labelled file the
one non-target label that was confidently wrong was the snoring class, 16 windows on a file containing no
snoring, against 12 `Cough` windows that were all correct. A vocabulary that could emit those types would
have emitted that one.

## The 6.60–7.10 s stretch, which the design refuses to resolve

The evidence about this stretch is contradictory and the design must not pick a side.

- It is recorded as human-verified "nothing — no breath here", with HeAR's `Breathe` 0.49 called a false
  positive credited as a catch in error.
- It is then reopened in the same file as "possibly an inhalation … *a class nothing in the fold
  distinguishes*. **Status: unresolved**", with HeAR's 0.49 re-read as genuine uncertainty at a
  Snore/Breathe boundary rather than a weak breath detection.
- YAMNet's `Breathing` fires across 6.72–7.68 there.

Under this branch's rules it lands in exactly one place regardless: classifier support with no envelope
onset is an unresolved region and the branch flags. That is the right answer under both readings — it is
not published as a breath, and it is not deleted either. A design that resolved it would be picking one
of two contradictory verifications by threshold.

## The hint

A hint names the task, so it says which elements to expect. It conditions decisions and never reaches a
measurement: the envelope, the YAMNet fold and the excerpt scan are computed identically with or without
one.

| the hint may | the hint may not |
| --- | --- |
| choose between two type candidates that both already carry mass, recording `resolved_by: hint` so the choice is reversible | create a type no plateau supports |
| turn the absence of an expected type into a flag — a respiration-and-cough task with no breath type is a finding | create, move or extend a span |
| turn its own contradiction into a flag — a voluntary-cough task where no cough survives is a finding about the recording or the hint | raise an `Estimate`'s `n_evidence`, which counts families, not expectations |

## Energy modulation — what it is for, unmeasured

Named as available and unmeasured. Three uses, in order of what they would settle:

**The offset, attacked as a coherence change rather than a level.** Every refuted offset here is a level
threshold crossing, and the refutation is that the reported end moves with the threshold — ~180 ms across
Δ ∈ [3, 20] dB. Turbulent flow has a characteristic broadband amplitude modulation, so the end of a
breath can be defined as the point at which modulation coherence returns to its pre-onset baseline, which
is not a level and does not move with a dB choice. **The acceptance test is stated in advance**: a
criterion qualifies only if its reported end moves less across its own parameter sweep than the
envelope's 180 ms, and if it covers more of the verified breath windows than the 10–52% every instrument
tried has managed. Until then, `extent` stays `witnessed_until` for breath.

**The grouping problem, attacked as a rate.** Inter-cough intervals within a bout run 0.3–0.5 s, i.e.
2–3 Hz, which is a modulation frequency rather than a segmentation. A bout might be detectable as a
modulation peak in a band where no instrument here can resolve the individual coughs. This would not
give counts; it would give a defensible statement that the region is a bout.

**Cough versus throat clear, without the refuted classifier vote.** The measured tiebreaker on this file
was non-classifier — rise time and level step, plus descending harmonic striations after each burst — and
those figures cannot become thresholds at n = 2. A modulation-envelope contrast is a second
non-classifier feature and would be worth measuring against the same pair.

## Outcomes

| outcome | condition |
| --- | --- |
| **fail** | an instrument the necessity rule requires cannot run: sample rate below 16 kHz, so both classifiers run outside their trained band and a cough's burst energy is out of band; a model load failure; the pinned revision unavailable. Or no candidate region has a measurable onset at all — no valid floor anywhere, or clipping through every onset — which is the instrument-failure form of "quality too poor to measure" |
| **flag** | any region is unresolved: classifier-only, envelope-only, type-ambiguous, grouping-unresolved, onset instruments disagreeing beyond ~30 ms, a hint contradicted, or every region withdrawn against a TAXONOMY presence claim |
| **pass** | at least one published event, and no unresolved region |

**`flag` is per file, not per event**, and for a sharper reason than the analogous rule in TAXONOMY: an
unresolved region can add a *type*, and "which types are present" is a file-level answer. A product that
published four clean events while one region waited on a human would be asserting a type set whose
completeness depended on something nobody had looked at.

The product is the same shape on `flag` and on `pass` — the events found, plus the unresolved regions with
their reasons. `flag` is the outcome, not a different payload.

## Parameters

| parameter | value | derivation |
| --- | --- | --- |
| `envelope_rise_delta_db` | 6 | onset error exactly 0 ms at Δ = 6 dB and within ±5 ms across Δ ∈ [3, 20] dB, both verified coughs. n = 2, one healthy adult, dry close mic (C50 median 28.5 dB) |
| `hear_excerpt_window_ms` | 160 | cliff below: posterior 0.063 → 0.220 → 0.999 at 40/80/160 ms, and nothing at 40 ms across 700 positions. False-positive growth above: `Breathe` centres in verified-empty audio 3 → 23 → 38 at 160/320/640 ms |
| `hear_excerpt_hop_ms` | 20 | the hop of the sweep that produced the two sharp plateaus and 24 centred detections. A hop sets sample density, not resolution — shrinking AST's hop tenfold made its leading edge worse |
| `crisper_revision` | `831f87e1d69c` | pinned deliberately. The 2026-08-17 retrain `de0369c8a680` emits one token, `[cough]`, where this revision emits five; the offset witness for breath exists only under the older weights. Every CrisperWhisper number in these notes was produced by it, reached through a stale `refs/main` |
| `hear_tau` | — | the only threshold-shaped measurement available is a gap: plateaus peak at 0.998–0.999 and the one measured shared false positive scores 0.49. Any value in that gap separates on one file. A value just above 0.5 is refused on principle — a 0.49 read as a detection is the recorded mechanism of the phantom breath |
| `yamnet_tau_airway` | — | 0.5 is a demonstrated non-separator: 12 correct `Cough` windows and 16 false snoring-class windows both cleared it |
| `yamnet_min_windows` | — | |
| `silence_tau` | — | |
| `onset_agreement_ms` | — | the ~30 ms figure is an observation over four events, not a fitted tolerance |
| `clipped_fraction_max` | — | the labelled file's clipped fraction is 0.000, so there is no measurement of how much clipping destroys a rise |
| `grouping_min_gap` | none | not a parameter. Grouping is a missing instrument |

Three things keep the empty slots honest, and they are the same three TAXONOMY relies on: the node can
`flag`, so doubt need not become a guess; the necessity rule means no single instrument can publish
alone; and `Estimate` shrinks a claim toward its prior by the evidence behind it, so a one-family type
publishes as a weak claim rather than as a number.

**Revision discipline.** Every model load resolves a ref to a 40-hex commit and then loads again by that
SHA, and the resolved SHA enters the cache key. This is repo policy and it has already bitten these
measurements: two runs of "the same model" produced different non-speech annotations because a cached
`refs/main` still pointed at pre-retrain weights one day after the push.

## What this branch does not do

- **No counts, and no bout structure.** Grouping is unsolved.
- **No duration, and no extent-derived measure.** Not withheld — uncomputable from the product's type.
- **No inhalation/exhalation split.** No instrument.
- **No speaker attribution.** A superseded draft consumed `c50_db`, `rms_db` and a segmentation product
  for this and none of the three had a producer anywhere in the graph. Attribution is a speaker question
  and there is no port carrying speakers into this branch.
- **No quality verdict.** The branch says whether it could measure, which is not the same claim.
- **No enhancement.** Two accidental element filters exist — `MossFormer2_SS_16K` src1 keeps cough
  (−1.2, −0.1 dB) and drops everything else by 31–50 dB, `sepformer-dns4-16k-enh` keeps breath (−1.7,
  +1.7 dB) and drops cough by 17–21 dB — and they would form a genuinely independent fourth confirmation
  channel. They rest on n = 1 and using an enhancer as a class filter is not what either was built for,
  so they are named candidates and not part of the design.

## The choices the evidence does not force

1. **160 ms for breath as well as cough.** Breath is measured to need 160–320 ms and 160 ms is the bottom
   of that band; it is chosen because it is also the minimum of the false-positive curve. What would
   settle it: a breath-side sweep at 160/200/240/280/320 ms scored for recall *and* false-positive centres
   over the verified-empty stretches. Only the false-positive side of that curve exists.
2. **A high `hear_tau` rather than 0.5.** Justified by a gap on one file, not by a distribution. What would
   settle it: the scores, not just the count, of the false-positive centres at 160 ms.
3. **Pinning a superseded CrisperWhisper.** It is the only source of an offset witness for breath, and its
   extra tokens align with verified events on this file — but one file cannot establish that the newer
   weights are worse, and the pinned revision's own `[UH]` is a mislabelled cough. What would settle it:
   both revisions scored against the verified windows on a second labelled recording.
4. **Requiring an envelope onset for every published event.** It costs nothing in recall on the one file
   that has labels, and it is the only thing that catches the measured shared false positive. On a
   reverberant or noisy recording the envelope's behaviour is entirely unmeasured, and if it loses recall
   there this rule loses events rather than false positives. What would settle it: recall over the six
   verified events and false positives per minute over the ~8.5 s of verified-empty audio, on the original
   and on copies degraded with added noise and reverberation.
5. **Reading `Silence` as an offset bound.** It fires in the verified-empty gaps on this file, which is a
   qualitative observation and not a scored one.
6. **Including `mouth_non_speech` in the vocabulary at all.** The taxonomy's airway kind lists inhalation,
   exhalation, cough and throat clear; the mouth sound is verified, is vocal-tract produced and non-lexical,
   and would fail the residual branch's periodicity gate because it is unvoiced — so this branch is the
   only place it can land, and nothing upstream routes it here. It reaches the product only as an
   unresolved region, which is the weakest form in which it can be represented at all.

## Contradictions in the source material, and where this design stands

| contradiction | how this design stands on it |
| --- | --- |
| Envelope offsets are refuted here (−449 to +20 ms, ~180 ms across the sweep, one cough never within 267 ms), while `ground-truth-2026-08-18.md` concludes the earlier 1.04–1.10 s ambiguity "was an artifact of the envelope method, not a property of the event", on the strength of CrisperWhisper's 14 ms offset on cough 1 | Both survive if read narrowly: the envelope method is an artifact *and* no instrument has demonstrated offset accuracy on more than one event. The design stands on the weaker claim — offsets are lower bounds — and leaves the `measured` variant unconstructible |
| "No instrument measured bounds an offset" versus CrisperWhisper's −14 ms offset on cough 1 with 97.5% coverage | An n = 1 success, on the event the same model got right, from a checkpoint that mislabelled the other cough. It is a witness for the lower bound, not a bound |
| HeAR: padding "moves an embedding about as far as substituting unrelated audio" versus excerpt-in-silence localising at 0.998 | Different heads. The branch reads the posterior and never the embedding, which is the only way both hold |
| HeAR fires on 40 ms of cough (real-context slide) versus 40 ms producing nothing at 700 positions (silent-buffer excerpt) | Same fact from two sides: the window fires on whatever content it holds, and native mode always holds 2 s of it. This is *why* native mode cannot localise and excerpt mode can |
| 6.60–7.10 s is "human-verified nothing" and also "reopened as possibly an inhalation, status unresolved" | Refused. Classifier support with no envelope onset is an unresolved region under either reading |
| The 16 false snoring-class windows are attributed to YAMNet here and in `taxonomy.md`, and to HeAR's 8-class space in `ground-truth-2026-08-18.md` (`Snore` reaching 0.86 around 6.5–7.0 s); 16 fits neither grid uniquely — 29 YAMNet windows, 50 HeAR windows over 14.03 s | The design is insensitive to which model it was: no single label names a type in either instrument, in either mode |
| Verified breath onsets given as 2.275 s and 5.308 s, verified breath windows beginning at 2.2995 s and 5.3285 s | Taken as ~25 ms of slack inside the reference, which is why onset tolerance is per type and only cough's is stated as ±5 ms |
| The mouth sound's onset is given as 0.893 s and its verified window as 0.779–0.981 s | Taken as a 114 ms late entry by the envelope on the one non-cough event where both numbers exist — the second reason the ±5 ms figure does not generalise |

## The measurement that would move this branch most

One more labelled recording, from a different speaker on a different microphone, and one degraded copy of
each. Almost every number here is n = 1 file, n = 2 events per class, one healthy adult, close-miked and
dry, with background music. The three quantities that would change the design rather than confirm it are:
false positives per minute for the envelope under reverberation, whether any modulation-based criterion
beats the envelope's 180 ms of threshold drift on an offset, and whether the cough/throat-clear confusion
survives a second speaker.
