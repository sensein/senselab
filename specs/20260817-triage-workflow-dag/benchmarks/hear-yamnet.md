# Classifier characterisation

## HeAR takes the whole span; YAMNet takes its own windows

HeAR's graph accepts exactly 2 s and rejects every other length. Spans here are 352–1408 ms, so the
span's audio is placed in a 2 s buffer containing nothing else.

| span | duration | whole span as input | runner-up | 500 ms gated sweep, by coverage |
| --- | --- | --- | --- | --- |
| 2.32–3.29 s | 970 ms | `Breathe` 0.989 | `Speech` 0.04 | `Breathe` 36% |
| 5.32–6.22 s | 900 ms | `Breathe` 0.940 | `Snore` 0.41 | `Breathe` 36% |
| 7.92–8.51 s | 590 ms | `Cough` 0.989 | `Baby Cough` 0.04 | `Cough` 64% |
| 9.61–9.96 s | 350 ms | `Cough` 0.996 | `Throat Clear` 0.02 | `Cough` 62% |
| 11.75–13.16 s | 1410 ms | `Speech` 0.146 — no airway label fires | `Laugh` 0.05 | `Cough` 0% |

Whole-span input gives four decisive labels and a correct rejection of the speech span. The sweep
reaches the same four at 36–64% coverage, a weak statement about spans that are in fact unambiguous.

**YAMNet must not be fed a padded span.** Padded up to its 0.96 s minimum:

| span | duration | YAMNet on the padded span | YAMNet on its own windows |
| --- | --- | --- | --- |
| 7.92–8.51 s | 590 ms | **`Laughter` 0.131** | `Cough`, 100% coverage |
| 9.61–9.96 s | 350 ms | `Cough` 0.311 | `Cough`, 100% coverage |

A cough read as laughter is a corrupted input, not a weak confirmation.

## Coverage, not peak

| span | HeAR coverage | YAMNet by maximum | YAMNet by coverage |
| --- | --- | --- | --- |
| 2.32–3.29 s | `Breathe` 36% | **`Gasp` 0.740** | **`Breathing` 75%** |
| 5.32–6.22 s | `Breathe` 36% | `Breathing` 0.915 | `Breathing` 67% |
| 7.92–8.51 s | `Cough` 64% | `Cough` 0.863 | `Cough` 100% |
| 9.61–9.96 s | `Cough` 62% | `Cough` 1.000 | `Cough` 100% |
| 11.75–13.16 s | `Cough` **0%** | `Speech` 0.993 | `Speech` 80% |

A maximum lets one window name a 1.4 s span, and on the first exhalation it names `Gasp` off a single
loud window where coverage names `Breathing`.

## Confidence thresholds sit in empty intervals

YAMNet scores across the 29 windows leave a gap containing 0.5 for every label that matters:
`Cough` 0.84 → 0.27, `Speech` 0.92 → 0.14, `Breathing` 0.59 → 0.36, `Silence` bimodal with every score
≤ 0.36 or ≥ 0.62. `Speech` has the widest gap in the design at **0.77** (0.919 → 0.145).

### RETRACTED for `hear.label_floor`: "0.5 sits in a wide empty interval"

The claim was that, given a whole span, the winning HeAR label scores 0.940–0.996 while the runner-up
scores 0.02–0.41, so 0.5 falls in a wide empty interval rather than being tuned to either side. That
reads five spans of one recording. The b2ai-28 campaign is the first corpus wide enough to see the
distribution, and there is no empty interval in it:

- **170 spans produced 14 labels.** Every other span scored under the floor on every channel.
- The 14 winners span **0.501–0.821**. Nothing scored above 0.821, so the "0.940–0.996" band the
  claim rests on is not where this corpus lives.
- **Two of the 14 sit within 0.005 of the floor.** A threshold two thousandths away from a winner is
  not in an empty interval; it is on top of the data.
- **Both dedicated cough files got zero labels** — the corpus's least ambiguous airway material,
  and the floor rejected all of it.

`hear.label_floor` 0.5 is therefore **unmeasured on this corpus**. It is neither confirmed nor
refuted as a value; what is refuted is the reason given for it. The five whole-span readings above
stand as measurements of those five spans and license nothing about where a floor belongs.

## What each reports where there is certified nothing

16 of 136 gated 500 ms windows lie wholly inside YAMNet-silence.

| HeAR channel | max inside silence | max outside | gap |
| --- | --- | --- | --- |
| `Cough` | 0.009 | 1.000 | clean |
| `Breathe` | 0.764 | 0.933 | **0.17** |

This is a property of **sweeping a gate across whole audio**, a detection problem. It does not describe
classifying an already-bounded span: given the whole span, `Breathe` scores 0.989 and 0.940. An earlier
draft concluded from these numbers that breath is intrinsically harder than cough and that `Breathe`
cannot support a decision alone; that conclusion was wrong for this branch. What survives is narrower —
YAMNet is confident a breath span is breath-*family* but not which member (`Breathing` 0.925 vs `Gasp`
0.91), which is why the confirmation mapping treats `Breathing`, `Sigh` and `Gasp` as one.

## Presence gates are not locators

**YAMNet `Speech`** region at ≥ 0.5 runs 10.08–13.44 s against a label of 11.62–13.20: **−1.54 s** at
onset, +0.24 s at offset. The window at 10.08–11.04 s scores 0.9194 over a stretch the labels call
nothing, where the envelope carries real energy and an early span rule proposed a span. Unresolved: the
label file is not exhaustive, and a coarse region with no label is not evidence of a false positive.

**HeAR** fires strongly on speech in its own airway label space — over 11.6–13.2 s: `Snore` 0.864,
`Throat Clear` 0.732, `Cough` 0.372. Restricting `labels_of_interest` is what keeps that out of an
airway verdict; it is not cosmetic.
