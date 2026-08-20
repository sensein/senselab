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
| `asr_crisperwhisper` | PREPROCESS, plain | word *presence* inside the span interval; never word times |

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

### 2. Labels — the whole span, as one input to HeAR

**The span itself is the model's input.** HeAR's graph accepts exactly 2 s and rejects every other length
outright, and every span this branch produces is shorter than that (352–1408 ms measured), so the span's
audio is placed in a 2 s buffer with **nothing else in it** — the model sees the span and silence, never a
neighbouring event.

That is the padding path HeAR's own API warns about for embeddings, so it is justified by measurement
rather than by assumption. Against the alternative — sweeping a 500 ms gate across the recording and
aggregating the windows overlapping each span — whole-span input is not marginally better but decisively
so:

| span | duration | whole span as input | runner-up | sweep, by coverage |
| --- | --- | --- | --- | --- |
| 2.32–3.29 s | 970 ms | **`Breathe` 0.989** | `Speech` 0.04 | `Breathe` 36% |
| 5.32–6.22 s | 900 ms | **`Breathe` 0.940** | `Snore` 0.41 | `Breathe` 36% |
| 7.92–8.51 s | 590 ms | **`Cough` 0.989** | `Baby Cough` 0.04 | `Cough` 64% |
| 9.61–9.96 s | 350 ms | **`Cough` 0.996** | `Throat Clear` 0.02 | `Cough` 62% |
| 11.75–13.16 s | 1410 ms | `Speech` 0.146 — no airway label fires | `Laugh` 0.05 | `Cough` 0% |

Four decisive labels with the runner-up 0.02–0.41 behind, and a correct rejection of the speech span,
where the top label is `Speech` at 0.146 and nothing airway rises at all. The sweep reached the same
four labels but at 36–64% coverage, which is a weak statement about a span whose identity is in fact
unambiguous.

`labels_of_interest` is configurable and **defaults to `{"Cough", "Breathe"}`** — HeAR's own names, of
its eight. Restricting the label space is not cosmetic: it is what keeps `Snore` and `Throat Clear`,
which fire on ordinary speech, out of an airway verdict. A span whose best label of interest fails to
clear 0.5 carries no airway label, which is how the speech span above is rejected without a special case.

**This revises a claim an earlier draft of this file made.** Characterising HeAR against certified
silence — `Cough` maxing at 0.009 inside silence against 1.000 outside, `Breathe` at 0.764 against
0.933 — the draft concluded that breath is intrinsically harder than cough and that `Breathe` cannot
support a decision alone. That measurement is real, but it is a property of **sweeping a gate across
whole audio**, which is a detection problem. This branch does not detect with HeAR; the envelope
detects, and HeAR only names an already-bounded span. Given the whole span, `Breathe` scores 0.989 and
0.940 — as decisive as cough. The silence characterisation stands as a warning about sweep-based
detection and no longer as a limit on this branch's breath labels.

### 2b. YAMNet confirms or contests — but on its own windows, never on a padded span

YAMNet's airway labels are independent of HeAR's, so each span's label is put to it as a second opinion.
**YAMNet is given the windows it produced natively over the recording, the ones overlapping the span —
not the span as an input.** The asymmetry with HeAR is deliberate and measured: YAMNet's window is 0.96 s
and padding a shorter span up to it destroys the evidence.

| span | duration | YAMNet on the padded span | YAMNet on its own overlapping windows |
| --- | --- | --- | --- |
| 7.92–8.51 s | 590 ms | **`Laughter` 0.131** | `Cough`, 100% coverage |
| 9.61–9.96 s | 350 ms | **`Cough` 0.311** | `Cough`, 100% coverage |

A cough read as `Laughter` at 0.131 is not a weak confirmation, it is a corrupted input, and it is the
reason the two models are fed differently: HeAR's fixed 2 s graph leaves no alternative to a buffer,
while YAMNet has already windowed the audio and those windows contain real context.

Over its native windows, aggregated by **coverage over the whole span** — the fraction of overlapping
windows clearing 0.5 — YAMNet votes only when confident and abstains otherwise. An unconfident 0.96 s
window is not evidence against HeAR.

| YAMNet's coverage winner | effect |
| --- | --- |
| maps to HeAR's label — `Cough`→`Cough`, `Breathe`→{`Breathing`, `Sigh`, `Gasp`} | **confirm**; two independent instruments behind the span |
| a confident label outside that mapping — `Speech`, `Silence`, another airway kind | **contest**; the span is flagged rather than relabelled, because two instruments disagreeing is a majority for neither |
| nothing reaches confidence anywhere in the span | **abstain**; HeAR's label stands, recorded as single-source |

**Coverage rather than peak, because a peak mislabels.** On the 2.32–3.29 s exhalation YAMNet's maximum
is `Gasp` 0.740 off a single loud window, while its coverage winner is `Breathing` at 75% — the correct
reading. Cough spans reach 100% coverage, breath spans 67–75%.

**0.5 is the confidence bar and it is not fitted.** For each label that matters, YAMNet's scores across
the recording leave an empty interval containing it: `Cough` jumps 0.84 → 0.27, `Speech` 0.92 → 0.14,
`Breathing` 0.59 → 0.36.

**All four airway-labelled spans confirm.** The fifth proposed span, 11.75–13.16 s, never reaches this
step: HeAR gave it no label of interest, so there is nothing for YAMNet to confirm or contest. YAMNet
would in fact call it `Speech` at 80% coverage, which is the same conclusion by a different route — but
the branch does not need that second opinion to reject it, and an earlier draft of this file was wrong to
present the two as independent mechanisms catching one span. HeAR alone rejects it, at step 2.

**What survives of the breath/cough asymmetry** is narrower than the earlier draft claimed and still
worth knowing: YAMNet is confident *that* a breath span is breath-family but not *which* member —
`Breathing` 0.925 against `Gasp` 0.91 on one span, `Sigh` 0.896 against `Gasp` 0.78 on the other. That is
why the mapping treats `Breathing`, `Sigh` and `Gasp` as one family for confirmation. The difficulty is
in breath's internal label granularity, not in detecting breath.

### 3. Lexical contamination — words *among* the airway events

Any ASR word whose interval intersects **`[first airway-labelled span start, last airway-labelled span
end]`** flags the file for review. `asr_crisperwhisper` carries word edges from PREPROCESS, so this
costs nothing here.

**The interval is defined over airway-labelled spans only** — spans that came out of step 2 carrying a
label of interest. A proposed span that HeAR declined to label is not an airway event and must not
extend the interval, because letting it do so makes the rule circular: a speech span would define the
very interval in which its own words are then found, and every recording containing both airway events
and speech would flag itself regardless of how they are arranged.

The interval spans gaps between airway events deliberately. Speech *interleaved among* airway events is
the case worth catching — it means either the recording is not the single-purpose capture it was taken
for, or a span the envelope proposed is speech. Speech *before or after* all of them is neither.

**The labelled recording shows the difference, and it passes.** Its four airway-labelled spans run
2.32–9.96 s; its speech is at 11.62–13.20 s, entirely after them. The interval is `[2.32, 9.96]`, no word
falls inside it, and the file is not flagged — the correct outcome, since a cough sequence followed by
unrelated speech is not a contaminated cough recording.

Note what this check is *not*. It reads word presence, not word timing, and plays no part in locating any
span.

### 3b. Proposed spans that carry no airway label

The envelope proposes on energy, so it proposes spans that are not airway events — the 11.75–13.16 s
speech span is one. **Only spans carrying a label of interest go through.** An unlabelled span is inert
here: it does not extend the lexical interval, is not put to YAMNet, and does not affect the outcome.

It is carried in the output as a separate field in case another node finds it useful — the envelope has
already paid for it — but it is **not part of this branch's product and not part of its goal**. Nothing
in this branch reads it, and no verdict here depends on it.

### 4. Outcome

| outcome | when | why this and not the other |
| --- | --- | --- |
| `fail` | no span proposed at all, **and** no hint declares airway content | there is nothing to classify, so the branch cannot produce its product. Not a finding about the recording — a statement that this branch has none |
| `flag` | spans exist but none carries a label of interest; **or** words fall inside the airway-labelled interval; **or** a hint declares airway content the branch did not find | each is a case where a human resolves faster than any rule here would |
| `pass` | at least one span carries a label of interest, no lexical contamination inside that interval, and nothing the hint asserts is missing | the product below |

**The labelled recording is a `pass`** — four airway spans, two breath and two cough, all confirmed by
YAMNet, no words among them. Worth stating because an earlier draft of this file had it flagging itself.

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
airway_spans:     [ { start, end, label, coverage, yamnet: confirm|contest|abstain,
                      inside_silence, peak_over_floor_db } ]     # the product
unlabelled_spans: [ { start, end, peak_over_floor_db } ]         # carried, not claimed
figure:           one aligned figure per recording
```

**`airway_spans` is the product.** Every entry carries a label of interest, and the two extra fields are
what a reader needs to discount one: `inside_silence` says the classifier fired where PREPROCESS
certifies nothing, and `peak_over_floor_db` says how much dynamic range the span had, which governs
whether its offset means anything.

`unlabelled_spans` is a by-product with no consumer in this branch. It is emitted because the envelope
already computed it and a later node may want it; it carries no label, no YAMNet verdict, and no claim.
A reader should treat an empty `unlabelled_spans` and a full one as equally uninformative about the
recording's airway content.

**The figure is a product, not a debugging aid.** It carries the waveform, the envelope with its floor
and both span sets — labelled and unlabelled, distinguishable at a glance — YAMNet `Silence`, the wideband spectrogram, the gammatone view, and the HeAR
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
