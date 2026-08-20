# AIRWAY branch — spans from the envelope, labels from HeAR

Decided 2026-08-20. This file governs and replaces the earlier six-stage design outright.

## What it decides

Whether the recording contains airway events, where they are, and what kind each is. Nothing else.
Types, counts, severity, and grouping into bouts are not this branch's business.

## Signature

```
airway(derivatives, labels_of_interest={"Cough", "Breathe"})
    -> fail(reason) | flag(reason, spans) | pass(spans, figure)
```

Every input is a PREPROCESS derivative. The branch runs no model of its own except HeAR, and computes
no view of the signal that PREPROCESS does not already share.

| input | from | used for |
| --- | --- | --- |
| `energy_envelope` | PREPROCESS, pre-emphasised | span proposal and both boundaries |
| `silence` | PREPROCESS, YAMNet `Silence` | the envelope's floor; negative evidence in classification |
| `spectrogram_wb`, `gammatone` | PREPROCESS, pre-emphasised | the figure only — no decision reads them |

## What it does, in three steps

### 1. Spans, from the floor and the envelope

The floor is the median envelope level over YAMNet-silence windows; PREPROCESS derives it and its
justification lives there. Given the floor, a span is proposed and bounded by three rules, each measured
against the labelled recording's six events:

| rule | value | why not otherwise |
| --- | --- | --- |
| propose | peaks ≥ `floor + 18 dB`, separated by ≥ 150 ms | at `floor + 12 dB` a low-contrast peak's offset threshold sits near the floor, so its walk is effectively unbounded and two coughs 1.7 s apart merged into one 6 s span |
| onset | walk back from the peak to `peak − 15 dB` | peak-anchored lands 5/5 airway onsets inside the labels' declared windows; the same envelope with a floor-referenced threshold lands 2/6, and every one of its errors is early |
| offset | walk forward to `peak − 0.7 × (peak − floor)`, closing only after 120 ms continuously below | the events do not share a dynamic range — 20.0 dB for a mouth sound against 56.8 dB for a cough — so a fixed drop is at once too shallow for one and unreachable for the other. Median offset error 84.3 ms, against 573.9 ms for a fixed `peak − 10 dB` |

**The anchor, not the threshold, is what makes the onset work**, and this is why PREPROCESS emits a
40 Hz modulation envelope while explicitly claiming no onset accuracy: widening the envelope's bandwidth
makes onsets *worse* (median 144 ms at 320 Hz against 63 ms at 40 Hz), because a wider band tracks
pre-event fluctuation that a fixed threshold then fires on. Accuracy is a property of this rule, and
therefore of this branch.

**Two limits on those figures.** They come from one recording, one healthy adult, six events, and they
justify the *shape* of each rule rather than its constant. And the peak was located inside a labelled
span; here the same rules run unsupervised over the whole envelope, which is the harder problem the
figure exists to expose.

### 2. Labels, from HeAR restricted to the labels of interest

HeAR runs as a **gated sweep** — a 500 ms rectangular gate stepped at 100 ms, each gate placed inside a
2 s buffer, because the detector's graph rejects every input length except 2 s outright. A span's label
is the label of interest with the highest score among sweep windows overlapping that span.

`labels_of_interest` is configurable and **defaults to `{"Cough", "Breathe"}`** — HeAR's own names, of
its eight. Restricting the label space is not cosmetic: it is what keeps `Snore` and `Throat Clear`,
which fire strongly on ordinary speech, from entering an airway verdict at all.

**The two default channels are not equally trustworthy, and the silence mask is what shows it.** Asked
what HeAR reports where PREPROCESS certifies nothing — 16 of 136 gated windows lie wholly inside
YAMNet-silence:

| channel | max inside certified silence | max outside | gap |
| --- | --- | --- | --- |
| `Cough` | **0.009** | 1.000 | clean |
| `Breathe` | **0.764** | 0.933 | **0.17** |

**No threshold separates 0.764 from 0.933**, so `Breathe` cannot support a decision on its own: it
reaches 0.764 at 4.15 s, 0.600 at 4.25 s and 0.481 at 1.15 s in windows containing nothing. `Cough`
separates cleanly. Two consequences the branch is built around: a span whose only evidence is `Breathe`
carries that weakness in its product rather than being reported as an equal, and **the silence mask
enters as negative evidence** — a firing wholly inside certified silence is evidence against, available
for every channel at no cost.

This measurement needs no labels: a recording, its silence mask, and the instrument's own output. Any
channel added to `labels_of_interest` should be characterised this way before a verdict rests on it.

### 3. Lexical contamination — are there words among the airway events?

Any ASR word whose interval intersects **`[first span start, last span end]`** contaminates the branch's
own working interval, and **such a file is flagged for review**. `asr_crisperwhisper` already carries
word edges from PREPROCESS, so this costs nothing here.

The interval is deliberately the whole stretch spanned by airway activity, gaps included, rather than
each span separately. Speech interleaved *between* airway events is the case worth catching: it means
either the recording is not the single-purpose airway capture it was taken for, or a span the envelope
proposed is speech rather than an airway event, and neither is something this branch should resolve on
its own.

**The labelled recording exercises this rule.** Its spans run 2.32–13.16 s and it contains speech at
11.62–13.20 s, so it is flagged — correctly. That is worth stating plainly: the recording used to derive
every span rule above is itself a file this branch refuses to pass.

Note what this check is *not*. It reads word presence, not word timing, and it plays no part in locating
any span. An earlier design used CrisperWhisper token edges as a second onset estimate; that is still
unused, for the reason at the end of this file.

### 4. Outcome

| outcome | when | why this and not the other |
| --- | --- | --- |
| `fail` | no span proposed at all, **and** no hint declares airway content | there is nothing to classify, so the branch cannot produce its product. Not a finding about the recording — a statement that this branch has none |
| `flag` | spans exist but none carries a label of interest; **or** words fall inside the span interval; **or** a hint declares airway content the branch did not find | each is a case where a human resolves faster than any rule here would |
| `pass` | at least one span carries a label of interest, no lexical contamination, and nothing the hint asserts is missing | the product below |

**The hint changes what an absence means, which is the only thing it is allowed to do.** With no hint,
finding no cough is simply a finding. With a hint declaring a voluntary cough task, finding no cough is a
contradiction between what the recording was taken for and what is in it — and a contradiction is a
`flag`, because the alternatives (the participant did not comply, the mic missed it, the branch missed
it) are not separable by anything this branch measures. **A hint never creates a span, relabels one, or
lowers a threshold.** It cannot promote a `fail` to a `pass`; it can only turn silence about an absence
into a flag about it.

**The `flag`/`fail` split is a decision, not a measurement**, and one caveat bounds it: concluding "no
respiration" leans on `Breathe`, the channel that cannot separate presence from silence. An absence of
cough is well supported by the table above; an absence of breath is not. Until that channel is
characterised on more than one recording, a `flag` raised for want of breath should be read as "this
branch could not find breath", never as "there is none."

## The product

```
spans: [ { start, end, label, score, inside_silence, peak_over_floor_db } ]
figure: one aligned figure per recording
```

`inside_silence` and `peak_over_floor_db` travel with each span because both are what a reader needs to
discount it: the first says the classifier fired where nothing is, the second says how much dynamic
range the span actually had, which is what governs whether its offset means anything.

**The figure is a product, not a debugging aid.** It carries the waveform, the envelope with its floor
and the proposed spans, YAMNet `Silence`, the wideband spectrogram, the gammatone view, and the HeAR
channels in use — all on one aligned time axis, so a span can be checked against every derivative that
produced it. The generating script is `plot3.py` beside this file.

One honest limit on it, learned by producing it: at 14 s across a page the 5 ms and 20 ms spectrograms
are visually indistinguishable, so **the figure cannot be used to verify anything that depends on the
analysis window**. It shows where events are, not what resolution resolved them. A claim of that kind
needs a zoomed span.

## What this branch does not do

No counts and no bouts — grouping adjacent spans into events requires a merge criterion nothing here
measures, and the labelled recording already contains a cough whose two expulsive phases the envelope
separates at 9.61–9.96 and 10.04–10.78 s. Whether that is one cough or two is a question this branch
declines. No severity, no type vocabulary beyond `labels_of_interest`, and no onset estimate from ASR.
PREPROCESS offers CrisperWhisper token edges and this design reads only word *presence* from them, never
their times: a second onset source needs a rule for disagreement, and there is no measurement here to
write one from.
