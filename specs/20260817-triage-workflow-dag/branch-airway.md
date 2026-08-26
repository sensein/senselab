# AIRWAY branch

Runs when [`routing.md`](routing.md) says the airway kind is present or uncertain, or when a hint
forces it.

## Signature

```
airway(store, hint?, labels_of_interest={"Cough","Breathe"})
    -> fail(reason) | flag(reason, spans) | pass(airway_spans, unlabelled_spans)
```

Reads and writes the [element store](store.md). It **proposes no candidate spans of its own**: it
re-evaluates every eligible `span` PREPROCESS wrote with HeAR, then `label`s, `confirm`s and
`contest`s those candidates.

| element read | author | used for |
| --- | --- | --- |
| `span` elements at `K` | PREPROCESS | the candidates this branch classifies |
| `hear_windows` | PREPROCESS | TAXONOMY's whole-file health-acoustic evidence; **not used to label a candidate** |
| `yamnet_windows`, `silence` | PREPROCESS | confirmation, contest, and negative evidence |
| `word` and `event` elements | PREPROCESS | which spans already carry a transcript, and lexical contamination |
| `spectrogram_wb`, `gammatone` | PREPROCESS | the [report](report.md) only |
| `hint` | caller | conditions the outcome only |

**This branch re-runs HeAR per candidate span.** It never reuses a whole-file HeAR label as span
evidence.

## 1. Label each span — HeAR, as confirmation

**HeAR is re-evaluated only here.** No other branch reads health-acoustic labels: its cough and breath
evidence is airway evidence and nothing else.

**HeAR confirms a span; it does not find one.** The candidate is the `span` element; AIRWAY puts a
short candidate in the detector's two-second silent buffer, or scans a longer isolated candidate,
then asks whether that input carries cough or breath.

- A span is eligible for a HeAR label only if it carries **no non-cough/breath transcript**. A span
  overlapping `word` entities from the consensus is transcribed content and is not offered to HeAR; a
  span overlapping only bracketed or onomatopoeic `event` entities remains eligible, because those
  are the events this branch is looking for.
- The label is the `labels_of_interest` member whose score clears its configured HeAR threshold in the
  span's fresh evaluation. The resulting `hear_span_window` records the scores and is derived from
  the candidate span.
- `labels_of_interest` is configurable; default `{"Cough", "Breathe"}`.
- A span with no confident member of interest in its windows carries **no label**.

## 2. Confirm or contest — YAMNet

YAMNet is read from `yamnet_windows`, on its own grid, never from the span as an input.

**A contest requires overlap with the candidate.** A YAMNet label may contest a fresh HeAR label only
when its native window intersects the candidate span. A label elsewhere is a different event, not a
disagreement about this one. The eligible contesting labels are the config key
`airway.contest_labels`, a declared set rather than all 521.

| the candidate's overlapping YAMNet windows | effect |
| --- | --- |
| carry a label mapping to HeAR's — `Cough`→`Cough`, `Breathe`→{`Breathing`,`Sigh`,`Gasp`} | **confirm** |
| carry a member of `airway.contest_labels` | **contest** — flag the span, do not relabel |
| carry neither | **abstain** — HeAR's label stands, marked single-source |

**A label may not both support and contest the same conclusion.** A label in
`taxonomy.audioset_airway_labels` is airway evidence and is therefore barred from
`airway.contest_labels`; the two sets are disjoint and the config is refused if they intersect.

## 3. Lexical contamination

Any consensus `word` intersecting `[first airway-labelled span start, last airway-labelled span end]`
flags the file. The interval spans the gaps between airway events; unlabelled spans do not extend it.
Bracketed and onomatopoeic `event` entities are not words and do not contaminate.

## 4. `K` is adjustable, and a span near it flags

The span gate `K` is the config key `spans.k_db`, and this branch reads it at its own setting: an
airway event is level-limited, and one value fitted on coughs does not serve quiet breaths.

- **`K` is per task.** `airway.k_db` overrides `spans.k_db` for this branch, and
  `airway.k_db_by_task` may override it per declared task.
- **A labelled span whose `peak_over_floor_db` sits within `airway.k_margin_db` of the gate is
  flagged**, with its margin reported. A span that only just cleared the gate, and any span the gate
  would have kept out under a slightly different setting, is a decision a human should see.
- **The merge rate is reported.** Where the offset rule merged adjacent proposals into one span, the
  span records how many proposals it absorbed, so a span covering several events is legible as one.

## 5. Outcome

| outcome | when |
| --- | --- |
| `fail` | no airway was established: either no span was proposed (or PREPROCESS reported `no_contrast`), or spans exist and none carries a label |
| `flag` | YAMNet contests a label; a word falls inside the interval; a labelled span sits within `airway.k_margin_db` of the gate |
| `pass` | at least one span carries a label of interest, and none of the above |

A hint reaches this branch's gate and nothing else. It never creates a span, relabels one, alters a
threshold, promotes a `fail` to a `pass`, or raises an absence to a `flag`: a flag resolves the kind
`present` in the fold, so a branch with no subject may not write one. A declaration this branch's
absence contradicts is named by VERDICT — [`verdict.md`](verdict.md).

**AIRWAY concludes about the airway kind and no other.** Its labels do not withdraw a diarizer
segment, do not remove energy from another branch's analysis, and do not refute the speech or voice
kinds — [`verdict.md`](verdict.md).

## Product

**The store holds the content; the product is the verdict and a named view over it.**

```
outcome:  fail(reason) | flag(reason, spans) | pass
verdict:  { labelled_n, by_label{}, contested_n, near_gate_n, merged_n, k_db, flags[] }
view:     the span element ids this branch labelled, confirmed or contested
```

Each span read through the view carries its `label`, the fresh HeAR evaluation behind it, YAMNet's
`confirm`/`contest`/abstain and the overlapping YAMNet windows, whether it lies inside
certified silence, how many proposals it absorbed, and its `peak_over_floor_db` with its margin over
`K`.

Spans this branch attached no label to are simply spans without a `label` assertion.

The [report](report.md) renders labelled and unlabelled spans distinguishably, on the shared axis.

## Out of scope

Running any classifier, counts and bouts (no measured merge criterion), severity, any type beyond
`labels_of_interest`, onset estimation from ASR, and any conclusion about a kind that is not airway.

Every label, confirmation and contest goes to the [element store](store.md) with its provenance.
Derivations live in [`benchmarks/`](benchmarks/).

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `airway.k_db`, `airway.k_db_by_task` | the span gate for airway, per task; **null** until quiet breaths and coughs are both represented in the fit |
| `airway.k_margin_db` | how close to the gate a labelled span must sit to flag; **null** |
| `airway.contest_labels` | the declared set of YAMNet labels that may contest a HeAR label, disjoint from the airway evidence labels; owed the corpus behind it |
