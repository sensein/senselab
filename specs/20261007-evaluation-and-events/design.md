# Evaluation utilities and behaviour events

## Scope

`senselab.utils.evaluation` splits labelled data and scores classifiers and event detectors.
`senselab.utils.data_structures.events` holds the behaviour-event types those scorers read, and
`events_io` reads and writes them in common interchange formats.

All of it is dataset-agnostic. A caller supplies rows, subject keys, unit keys and label
vocabularies; the package never sees a path layout, a study's label set or a manifest. Code that
knows about a particular study lives with that study and imports this package.

Three rules run through every function:

1. Every split and every report names its **protocol**.
2. Every classification report leads with **balanced accuracy** and gives each metric's chance or
   trivial **floor**, so a number is never read without its null.
3. Thresholds are tuned on a validation split, never on the split being scored.

Ported functions and their origins are listed in `provenance.md`.

## Protocols

`protocol` takes one of two values, and it is required: no splitter or scorer has a default.

- **`unseen_subject`.** No subject appears in two partitions. This asks whether the model works on
  people it has never seen.
- **`seen_subject`.** Every test subject also appears in train, no unit appears in two partitions,
  and every subject has at least two units. This asks whether the model works on new material from
  people it has already seen.

Both are legitimate questions. What makes a seen-subject result misleading is not the protocol but
leaving it unlabelled, so a report from one can be mistaken for the other.

**The unit is declared by the caller and recorded** in the split report's `unit` field.

- The default unit is the source recording (or session, or video).
- A clip unit is allowed. It lets clips of one recording fall into different partitions, which a
  recording unit forbids. That straddling is a property of the split, so the report says
  `unit: clip`, and a reader can tell it apart from a recording-level split.
- Units are keyed by the composite `(subject, unit)`. The check is then correct whether unit ids are
  unique across subjects or only within one.
- Under `unseen_subject` the report's unit is `subject`.

A violation raises `ProtocolViolation`. Its message gives counts only, never key values, so an
error message cannot leak an identifier into a log.

## Imbalance-aware reporting

On a 90/10 binary set, a classifier that always answers the majority class scores 0.9 accuracy and
has learned nothing. Every metric has some such trivial value. A report therefore gives, for each
metric, its **floor** (the best a trivial predictor achieves), the **lift** (value − floor) and the
**normalised lift** ((value − floor) / (1 − floor)).

| Metric | Floor | Why |
|---|---|---|
| balanced accuracy | 1/k (0.5 binary) | A constant predictor has recall 1 on one class and 0 on the other k − 1; the mean is 1/k. A random predictor has the same expectation (Brodersen et al. 2010). |
| positive-class F1 | 2p/(1+p) | The all-positive predictor has precision p and recall 1. Every other constant scores 0. |
| average precision | p (binary), 1/k (macro one-vs-rest) | A random scorer's expected AP equals the positive rate (Saito & Rehmsmeier 2015). Averaged one-vs-rest, the class prevalences sum to 1, so the mean is 1/k. |
| ROC-AUC | 0.5 | Any constant or random scorer ranks positives above negatives half the time. |
| MCC | 0 | A constant predictor has an empty row or column in the confusion matrix and is scored 0 by convention; a random one has expectation 0 (Chicco & Jurman 2020). |
| accuracy | majority-class prevalence | Always answering the majority class. |
| macro-F1 | the best constant predictor's macro-F1 | Answering class c gives that class an F1 of 2p_c/(1+p_c) and every other class 0, so the macro mean is 2p_c/(1+p_c)/k. The floor is the maximum over c. |

**Balanced accuracy is the only headline.** Raw accuracy is unreadable without its null rate, and
balanced accuracy's null is fixed at 1/k whatever the prevalence. Macro-F1, MCC and accuracy are
reported under `secondary`, each with its floor and lift. The report has no accuracy accessor at
the top level and no parameter that picks a different headline.

Also in every report:

- per-class and per-split prevalence, and the predicted prevalence, so a reader sees the class mix
  each number was computed on
- per-class precision, recall, F1 and support, and the confusion matrix
- ROC-AUC and average precision when scores are supplied; average precision because ROC-AUC alone
  is optimistic under heavy imbalance (Saito & Rehmsmeier 2015)
- an invalid-output rate, with invalid outputs counted as misses rather than dropped

**Confidence intervals** come from a cluster bootstrap that resamples subjects, not rows (Davison &
Hinkley 1997, ch. 3.8). Rows from one subject are correlated, so a row bootstrap understates the
interval width. **Model comparison** uses a paired cluster bootstrap of the difference, or DeLong's
test for two correlated ROC-AUCs (DeLong et al. 1988) with the midrank algorithm of Sun & Xu (2014).

References:

- Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). The balanced accuracy and
  its posterior distribution. *ICPR 2010*, 3121–3124.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC
  plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
- Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC)
  over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21, 6.
- Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge
  University Press.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or
  more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*,
  44(3), 837–845.
- Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under
  correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11),
  1389–1393.

## Split methods

`grouped_splits` is the one entry point. Its methods:

- stratified group k-fold
- the same with combined strata (for example label × timepoint), still grouped by the individual
- group-aware iterative multi-label stratification
- leave-one-subject-out
- grouped shuffle
- nested cross-validation, with inner splits drawn from the outer training partition only
- seen-subject k-fold and seen-subject holdout

**Grouping is by the individual, never by individual × timepoint.** The counter-example is a fold
assignment grouped by (subject, timepoint): one visit of a person is tested while another visit of
the same person is trained on. A check keyed on (subject, timepoint) passes it; the unseen-subject
invariant, keyed on the subject, rejects it.

Every method returns a split report: `protocol`, `unit`, `stratify`, and per fold and partition the
row, group and unit counts and the prevalence (with per-stratum prevalence when strata are given),
plus the largest absolute deviation of any fold's prevalence from the overall prevalence and the
invariant that was checked. The invariant is checked when the report is built, so a report that
exists is a report that passed. `SplitReport.to_reproduce()` reduces it to the flat shape a
reproduction driver records: each fold's test-partition group count and prevalence.

**Library choice.** scikit-learn, already a core dependency, supplies `StratifiedGroupKFold`,
`LeaveOneGroupOut` and `GroupShuffleSplit`; they are used unchanged. Two pieces are reimplemented,
because no maintained library provides them:

1. **Group-aware multi-label iterative stratification.** The iterative algorithm of Sechidis et al.
   (2011) applied to per-group label-count vectors. The `iterative-stratification` package
   implements the algorithm on rows and has no groups option.
2. **The seen-subject splitters.** No scikit-learn splitter stratifies within each subject while
   grouping by unit.

No dependency is added. The full justification and the measurements that fix the test tolerances
are recorded with the splitters' implementation.

**Why the seen-subject k-fold keeps every subject in train.** Each subject's units are dealt
round-robin over the n folds, so a fold holds at most ⌈u/n⌉ of a subject's u units. For u ≥ 2 and
n ≥ 2, ⌈u/n⌉ ≤ ⌈u/2⌉ ≤ (u + 1)/2 < u, so at least one unit is left for training in every fold.

**Seen-subject holdout.** Within each subject (and stratum, when given) units go to train,
validation and test by stratified proportional allocation, with at least one unit kept in train.
With clips as the unit and fractions 0.6/0.2/0.2, this reproduces a seen-subject clip-level
protocol used in the lab's own audio work. Declaring the unit as `clip` is what makes that split
pass the seen-subject invariant, and the report records it.

Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data.
*ECML PKDD 2011*, LNCS 6913, 145–158.

## Threshold grid

`data/threshold_grid.yaml` holds the candidate thresholds for `tune_threshold`: 0.05 to 0.95 in
steps of 0.05, ties broken towards the threshold closest to 0.5, the result rounded to four
decimals.

**Derivation.** This is a ported convention, not a fitted value: it is the grid and tie-break the
lab's earlier threshold-tuning code used (`provenance.md`, row `tune_threshold`), kept so that
results tuned by either implementation agree. It was not fitted to any data. Among equally good
thresholds, the tie-break picks the one nearest 0.5, the least extreme operating point. The grid
lives in `data/` rather than in code so that changing it is a reviewed data change.

The default objective is balanced accuracy. `f1` and `mcc` are available only as an explicit,
non-default choice that the report records; the balanced-accuracy headline is still computed at
whatever threshold was chosen.

## Event metrics

**Onset-tolerance F1.** A predicted event matches a reference event of the same label when their
onsets differ by no more than a tolerance in seconds. Matching is greedy and one-to-one, so two
predictions near one reference give one true positive and one false positive. Ported from the lab's
earlier implementation (`provenance.md`, row `event_f1`).

**Segment mAP.** Implemented clean-room from the ActivityNet temporal-localisation definition; no
lab implementation was read.

- Predictions of each class are ranked by confidence across all videos.
- Each prediction matches the best unmatched reference of the same class and video whose temporal
  IoU is at least the threshold; a duplicate detection is a false positive.
- Average precision is all-point interpolated.
- The mean is taken over classes present in the references.
- The report gives mAP at each tIoU threshold and their average separately, so a single number is
  never labelled with the wrong threshold.

## Event interchange formats

`events_io` reads and writes six formats: a per-frame label table, a segment CSV, RTTM, BIDS
`events.tsv`, ActivityNet/OpenTAD ground-truth JSON and a Label Studio export. `FORMAT_FIELDS`
declares which event fields each format carries. "Lossless" means every carried field
round-trips; a reader takes the fields its format lacks (schema id, source, confidence kind) as
arguments. A reader of a format that holds confidences but not their kind raises unless the caller
names the kind, so a vote fraction is never relabelled as a probability. Files are UTF-8 whatever
the locale.

**Per-frame label table.** A generic wide format: a `Frame` column of 0-based frame numbers and
one column per category, each cell holding that category's label or a background token (`N/A` by
default). An event `[onset, offset)` covers frames `[onset·fps, offset·fps)`. The writer refuses
what the table cannot hold without loss: times that are not frame-aligned, an event that covers no
frame, overlapping events in one category, adjacent events with the same label (which would read
back as one), and a track duration other than `n_frames / fps`. It is lossless up to `k / fps`: a
time within the alignment tolerance of frame `k` reads back as exactly `k / fps`.

**Format limits.**
- RTTM has no record for an empty track, so an empty track's recording id does not come back.
- BIDS stores a duration per row. A zero duration (an impulse event) is refused on read, because
  a `BehaviorEvent` needs `offset > onset`. Spaces are legal in `trial_type`; tabs and line breaks
  are refused on write.
- Label Studio: only `labels` results with start and end in seconds are covered; video
  `timelinelabels` results with frame ranges are not. The duration travels as each result's
  `original_length`, so an empty track cannot carry it, and `duration_s` is not declared for this
  format.

**Float exactness.** RTTM and BIDS store durations, so the offset is rebuilt as `onset + duration`.
Times are written as the shortest decimal that round-trips to the same float, so the reader gets
`d = fl(b − a)` exactly back, for onset `a` and offset `b` with `0 ≤ a < b`. Then:

- `d = (b − a)(1 + δ₁)` with `|δ₁| ≤ u = 2⁻⁵³`, so `a + d = b + (b − a)δ₁` and
  `|a + d − b| ≤ u(b − a) ≤ u·b`.
- For `b` in `[2^e, 2^(e+1))`, `ulp(b) = 2^(e−52)`, so `u·b < ulp(b)`: `a + d` lies within one ulp
  of `b`, and rounding it to the nearest float gives `|fl(a + d) − b| ≤ ulp(b)`.
- When `a ≥ b/2`, `b − a` is exact (Sterbenz), so the offset comes back exactly. Times that are
  multiples of a power of two (the fixtures use multiples of 1/64 s) also come back exactly.

The test checks the bound on pairs known to be inexact and on a power-of-two offset, where the
float spacing below `b` is half the spacing above it.
