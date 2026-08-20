# AIRWAY branch

## Signature

```
airway(store, hint?, labels_of_interest={"Cough","Breathe"})
    -> fail(reason) | flag(reason, spans) | pass(airway_spans, unlabelled_spans, figure)
```

Reads and writes the [element store](store.md). It **proposes no elements of its own**: it `label`s,
`confirm`s and `contest`s the `span` elements PREPROCESS wrote.

| element read | author | used for |
| --- | --- | --- |
| `span` elements at `K` = 18 dB | PREPROCESS | the candidates this branch classifies |
| `silence` | PREPROCESS | negative evidence in classification |
| `asr_crisperwhisper` words | PREPROCESS | word *presence* only, never word times |
| `spectrogram_wb`, `gammatone` | PREPROCESS | the figure only |
| `hint` | caller | conditions the outcome only |

## 1. Classify each span — HeAR

The **whole span** is the model input: its audio placed in a 2 s buffer containing nothing else, because
HeAR's graph accepts only 2 s.

- Label = the `labels_of_interest` member with the highest score, if it clears **0.5**.
- `labels_of_interest` is configurable; default `{"Cough", "Breathe"}`, from HeAR's eight.
- A span whose best label of interest fails to clear 0.5 carries **no label**.

## 2. Confirm or contest — YAMNet

YAMNet is read from **its own native 0.96 s windows** overlapping the span, never from the span as an
input. Aggregation is **coverage**: the fraction of overlapping windows scoring ≥ 0.5.

| YAMNet coverage winner | effect |
| --- | --- |
| maps to HeAR's label — `Cough`→`Cough`, `Breathe`→{`Breathing`,`Sigh`,`Gasp`} | **confirm** |
| a confident label outside that mapping | **contest** — flag the span, do not relabel |
| nothing reaches 0.5 anywhere in the span | **abstain** — HeAR's label stands, marked single-source |

## 3. Lexical contamination

Any ASR word intersecting `[first airway-labelled span start, last airway-labelled span end]` flags the
file. The interval spans the gaps between airway events. Unlabelled spans do not extend it.

## 4. Outcome

| outcome | when |
| --- | --- |
| `fail` | no span proposed, or PREPROCESS reported `no_contrast`, and no hint declares airway content |
| `flag` | no span carries a label of interest; or YAMNet contests a label; or a word falls inside the interval; or a hint declares airway content not found |
| `pass` | at least one span carries a label of interest, and none of the above. The labels are in the store; the verdict says what the branch concluded |

A hint changes only what an absence means. It never creates a span, relabels one, alters a threshold,
or promotes a `fail` to a `pass`.

## Product

**The store holds the content; the product is the verdict and a named view over it.**

```
outcome:  fail(reason) | flag(reason, spans) | pass
verdict:  { labelled_n, by_label{}, contested_n, flags[] }
view:     the span element ids this branch labelled, confirmed or contested
figure:   one aligned figure per recording          # an artifact, not in the store
```

Each span read through the view carries its `label`, the coverage behind it, YAMNet's
`confirm`/`contest`/abstain, whether it lies inside certified silence, and its `peak_over_floor_db` —
the last two being what a reader needs to discount it.

Spans this branch attached no label to are simply spans without a `label` assertion. They need no
separate field and no separate list.

The figure carries the waveform, the envelope with its floor and every span, YAMNet `Silence`, the
wideband spectrogram, the gammatone view and the HeAR channels in use, on one time axis, with labelled
and unlabelled spans distinguishable.
## Out of scope

Counts and bouts (no measured merge criterion), severity, any type beyond `labels_of_interest`, and
onset estimation from ASR.

Every label, confirmation and contest goes to the [element store](store.md) with its provenance.
Derivations live in [`benchmarks/`](benchmarks/).
