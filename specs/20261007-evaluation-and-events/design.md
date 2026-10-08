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
| balanced accuracy | 1/k (0.5 binary), k = labels present in `y_true` | A constant predictor has recall 1 on one class and 0 on the other k − 1; the mean is 1/k. A random predictor has the same expectation (Brodersen et al. 2010). |
| positive-class F1 | 2p/(1+p) | The all-positive predictor has precision p and recall 1. Every other constant scores 0. |
| average precision | p (binary), 1/k (macro one-vs-rest) | A random scorer's expected AP equals the positive rate (Saito & Rehmsmeier 2015). Averaged one-vs-rest, the class prevalences sum to 1, so the mean is 1/k. |
| ROC-AUC | 0.5 | Any constant or random scorer ranks positives above negatives half the time. |
| MCC | 0 | A constant predictor has an empty row or column in the confusion matrix and is scored 0 by convention; a random one has expectation 0 (Chicco & Jurman 2020). |
| accuracy | majority-class prevalence | Always answering the majority class. |
| macro-F1 (over present labels) | 1/k | A label-blind predictor that draws labels at the true prevalences has precision and recall p_c for class c, so its expected F1 for c is p_c and its macro mean is 1/k. That beats the best constant predictor (max 2p_c/(1+p_c), over k), which was the floor at first; a floor must not sit below what a label-blind predictor reaches. |

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

**Bootstrap details.** `bootstrap.py`; B = 2000 by default; percentile intervals (`np.percentile`,
linear), 2.5 and 97.5 for 95 %.

- Groups are ordered by first appearance and keep their rows in input order; each resample is one
  `Generator.integers(0, n_groups, n_groups)` call, so a seed fixes every resample. The resampling is
  not stratified: the groups carry the label mix, and the convention ported here does not stratify.
- Predictions are fixed before resampling (a tuned threshold is applied once, with `>=`); nothing is
  re-tuned inside a resample. Balanced accuracy in a resample is `metrics.balanced_accuracy`,
  the headline function, so a resample that drops one of several classes averages over the classes
  it holds and is not biased low.
- A resample whose truth holds one class is skipped and counted (`n_skipped`, reported on every
  interval); a metric undefined on a kept resample is left out for that metric (`n_used`). The skip
  stays because the earlier lab bootstrap skips, and parity with it requires the same resamples.
  A high skip fraction means few groups carry the minority class, and then the interval
  under-covers, so an interval is flagged `unreliable` when more than 5 % of resamples gave it no value,
  `(n_resamples − n_used) / n_resamples > 0.05`, counting skipped resamples and those where that metric
  was undefined. Synthetic illustration (groups of 3–12 rows, a share q of groups
  holding positives, true BA 0.659, B = 1000, about 400 replicates per row; "keep" scores single-class
  resamples with present-label BA instead of skipping):

  | groups | q | mean skip | coverage, skip | width, skip | coverage, keep | width, keep |
  |---|---|---|---|---|---|---|
  | 8 | 0.25 | 0.136 | 0.659 | 0.283 | 0.885 | 0.358 |
  | 12 | 0.25 | 0.084 | 0.713 | 0.285 | 0.864 | 0.324 |
  | 20 | 0.25 | 0.021 | 0.839 | 0.275 | 0.892 | 0.284 |
  | 40 | 0.25 | 0.001 | 0.895 | 0.213 | 0.897 | 0.214 |
  | 12 | 0.5 | 0.007 | 0.897 | 0.265 | 0.907 | 0.268 |

- A statistic undefined on all rows (ROC-AUC or AP when the scored split holds one class) has
  `estimate: None` and no interval; the report's own `roc_auc` and `average_precision` are `None`
  there, computed without calling scikit-learn, so no warning is raised. No value is ever NaN, and
  `to_dict()` is strict JSON.
- The resampling depends on numpy's `Generator.integers` stream, which NEP 19 does not guarantee across
  numpy releases; a golden-value test pins the first draws, and the X2 parity record states both
  environments' numpy versions.
- `metrics.py` holds the label metrics (`balanced_accuracy`, `legacy_macro_recall` and the
  confusion-matrix forms); the report encodes labels to integers once and every resample computes
  balanced accuracy, macro-F1 and MCC from its confusion matrix with the same functions as the
  point estimates.
- `score_classification(groups=...)` adds intervals for balanced accuracy, macro-F1, MCC, and ROC-AUC
  and average precision when scores exist, each computed exactly as the report computes the metric.
  Timing at B = 2000: ranking
  metrics (scikit-learn per resample) dominate when scores are given.
- `legacy_cluster_bootstrap` reproduces the lab's earlier binary bootstrap (`provenance.md`, row
  `cluster_bootstrap_ci`): fixed threshold, F1, balanced accuracy, ROC-AUC and average precision.
  A test reimplements that algorithm and requires identical interval bounds.
- `paired_cluster_bootstrap_delta` resamples groups once per draw and scores both models on the
  same rows.

**DeLong's test** (`delong.py`). `delong_test(scores_a, scores_b, labels)` compares two scorers on
the same rows. Each scorer's structural components (DeLong et al. 1988) are computed from midranks in
O(N log N) (Sun & Xu 2014): a positive's component is the share of negatives it outranks, ties counting
one half, and a negative's the share of positives that outrank it. Variances and the covariance use
the sample covariance (ddof 1) over positives and over negatives; `z` is the AUROC difference over the
standard deviation of the difference, and the p-value is two-sided normal (`2·sf(|z|)`, which keeps
precision for large `|z|` where `2·(1 − cdf)` rounds to 0). When the variance of the difference is not
positive, there is no variance estimate: `z` and `p_value` are `None` rather than NaN, so results
serialise to strict JSON. That can happen even when the AUCs differ: a perfect scorer against a constant
one has AUCs 1 and 0.5 but zero variance in both. Labels must be exactly two distinct values, none
missing, with `positive_label` among them. Field names use the report's spelling (`roc_auc_a`,
`roc_auc_b`), as do the legacy bootstrap's keys (`roc_auc`, `average_precision`). Ported from
the lab's earlier implementation (`provenance.md`, row `delong_test`); a test reimplements that
algorithm and requires agreement within 1e-9, and checks the variances against the quadratic-time
definition within 1e-12. Each class needs at least two rows.

**Balanced accuracy's formula.** Recall averaged over the labels present in `y_true`, with invalid
outputs counted as misses (`balanced_accuracy`): the same as scikit-learn's
`balanced_accuracy_score`, and the definition whose trivial value is the `1/k` floor a reproduction
driver checks (coordinator decision, 2026-10-07). Threshold tuning (T6) and resampled intervals (T7)
call the same function. The per-class recalls are averaged with `np.mean` over the present labels in
scikit-learn's sorted order (numbers numerically, strings lexically; `metrics.native_order`), so the
value equals scikit-learn's exactly and does not depend on the order in which labels are declared. (A
mean depends on the order of its terms in the last bit; an earlier `math.fsum` version was order-free
but differed from scikit-learn by up to one ulp.) Macro-F1 also averages over present labels.

The lab's earlier evaluation (`provenance.md`, row `score_classification`) used
`recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)`, macro recall over
every declared label, where an absent label contributes 0. That formula is kept as
`legacy_macro_recall` for parity checks with earlier results only; it is never a report headline.
The two agree when every declared label occurs, and differ otherwise: with labels a, b, c and
y_true 6 × a + 4 × b all predicted a, the headline is 0.5 and the legacy value 0.333.

Per class, recall and F1 of an absent label are undefined and reported as `None`; precision is
`None` for a label that was never predicted. The report also lists `absent_labels`.

**Thresholds.** A report built with a tuned threshold records `threshold{value, objective,
split_name}` and `threshold_tuning: validation`. Predictions made elsewhere declare
`threshold_tuning` themselves: `validation`, or `none` (the default); `test` is refused, because a
threshold tuned on the scored split inflates the score. A threshold whose split name matches the
scored split (ignoring case and surrounding whitespace) is refused, as are NaN thresholds and
non-finite scores. The headline is balanced accuracy whatever objective set the threshold.

**Export to a reproduction driver's `metrics.json`** (written by the CLI, T10):

| Report | Driver entry | Floor agreement |
|---|---|---|
| `balanced_accuracy` | headline `balanced_accuracy` | same: 1/k over present classes |
| `secondary.accuracy` | secondary `accuracy` | same: majority-class prevalence |
| `secondary.mcc` | secondary `mcc` | same: 0 |
| `secondary.f1` (binary, positive label) | secondary `f1` | same: 2p/(1+p) |
| `secondary.f1_macro` (over present labels) | secondary `macro_f1` | driver checks presence only; ours is 1/k over present labels |
| `roc_auc` | secondary `auroc` | same: 0.5 |
| `average_precision`, binary with positive label | secondary `pr_auc` | same: p |
| `average_precision`, multi-class (macro one-vs-rest, floor 1/k) | not exported | the driver defines `pr_auc` for a declared positive class only |
| `per_class` | `per_class.{recall, precision, f1, prevalence}` | `None` values (absent labels, never-predicted precision) are mapped in T10 |
| `confusion_matrix`, binary | `confusion.{tp, fp, fn, tn}` | how invalid outputs map onto the four cells is settled in T10 |
| `threshold_tuning` | `threshold_tuning` | — |
| `split_report` | `SplitReport.to_reproduce()` | — |

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

`grouped_splits` is the one entry point for making splits; `check_split` checks a split made
elsewhere. The methods:

- stratified group k-fold
- the same with combined strata (for example label × timepoint), still grouped by the individual
- group-aware multi-label stratification
- leave-one-subject-out
- grouped shuffle
- nested cross-validation (`nested_grouped_splits`), with inner splits drawn from the outer
  training partition only
- seen-subject k-fold and seen-subject holdout

**Grouping is by the individual, never by individual × timepoint.** The counter-example is a fold
assignment grouped by (subject, timepoint): one visit of a person is tested while another visit of
the same person is trained on. A check keyed on (subject, timepoint) passes it; the unseen-subject
invariant, keyed on the subject, rejects it.

**Split report.** Every method returns `protocol`, `unit`, `stratify`, and per fold and partition
the row, group and unit counts and the prevalence (each label's share of the partition's rows, or
each column's positive rate for a multi-label matrix; per-stratum joint shares when strata are
given), plus the largest absolute deviation of any partition's prevalence from the overall
prevalence and the invariant that was checked. The invariant is checked when the report is built,
so a report that exists is a report that passed.

- `unit` is `subject` under `unseen_subject` and the caller's `unit_name` under `seen_subject`.
  `unit_name` has no default: a clip-level split cannot be reported as a recording-level one by
  omission.
- `stratify` is `none` (leave-one-subject-out, grouped shuffle), `class`, or `class_<strata_name>`
  when strata are given; the caller names the strata.
- `SplitReport.to_reproduce()` reduces the report to the flat shape a reproduction driver records:
  each fold's test-partition group count and prevalence. That schema has fixed vocabularies for
  `unit` (`subject`, `family`, `session`, `video`, `clip`, `recording`, `row`) and `stratify`
  (`none`, `class`, `class_timepoint`, `class_site`, `class_age`, `timepoint`). A report whose unit
  or stratify falls outside them raises instead of emitting an unknown token.

**Validation partitions.** `val_inner_splits=k` carves a validation partition from each fold's
training pool as the first of `k` inner folds, seeded with `seed + 10000·(fold + 1)`: the convention
of the lab's earlier grouped k-fold (`provenance.md`, row `grouped_splits`).

**Library choice.** scikit-learn, already a core dependency, supplies `StratifiedGroupKFold`,
`LeaveOneGroupOut` and `GroupShuffleSplit`; they are used unchanged. Two pieces are written here,
because no maintained library provides them:

1. **Group-aware multi-label stratification.** The `iterative-stratification` package stratifies
   rows and has no groups option. The first implementation applied the iterative algorithm of
   Sechidis et al. (2011) to per-subject label-count vectors (rarest label first, each subject to
   the fold that most wants that label). On the test cohort it left the common label (rate ≈ 0.5)
   0.147 from its overall rate in one test fold, outside the 0.05 tolerance fixed beforehand: it
   commits large subjects early on account of rare labels, and later small subjects cannot even
   the folds out. It was replaced by a greedy placement in the spirit of scikit-learn's own
   `StratifiedGroupKFold`: subjects largest first, each into the fold that minimises the summed
   squared difference between the fold's share of every label's positives and its share of rows,
   plus its row share's distance from `1/k`. On the cohort, over seeds 0–19, the worst test-fold
   deviation was 0.031 (rate ≈ 0.5), 0.014 (rate ≈ 0.2) and 0.004 (rate ≈ 0.05), and the rare label
   reached every test fold. The tolerances were not changed.
2. **The seen-subject splitters.** No scikit-learn splitter stratifies within each subject while
   grouping by unit.

No dependency is added.

**Seen-subject k-fold.** Each subject's units are ordered by majority label (then majority stratum),
shuffled within ties with the seed, and dealt round-robin over the n folds from a seeded offset.
**Why every subject stays in train:** a fold holds at most ⌈u/n⌉ of a subject's u units, and for
u ≥ 2 and n ≥ 2, ⌈u/n⌉ ≤ ⌈u/2⌉ ≤ (u + 1)/2 < u, so at least one unit is left for training in every
fold. With `val_inner_splits=k`, a subject's remaining r units are dealt over k slots the same way
and slot 0 becomes validation, but only when r ≥ 2, which keeps at least one unit in train by the
same argument. Slot 0 is certain to be filled only when r ≥ k; the worst fold leaves
r = m − ⌈m/n⌉, so a subject with m ≥ ⌈k·n/(n − 1)⌉ units (and at least 3) always fills it. That
bound is sufficient, not necessary: cohorts below it usually fill every fold's validation partition
through other subjects (for example 50 subjects × 4 units with n = 5, k = 4). The split is
therefore not refused up front; if a fold's validation partition comes out empty, which is
deterministic for a given seed, the error names the sufficient bound as advice.

**Seen-subject holdout.** Each subject's units, in the same order, are split by fractions
(default 0.6/0.2/0.2): the counts come from largest-remainder rounding of `u·fraction`, every
partition with a positive fraction gets at least one unit when the subject has as many units as
there are such partitions, and train always keeps at least one. The units are then interleaved
proportionally over the ordered list, so each partition draws from every label (and stratum) in
proportion; strata therefore shape the order within a subject rather than forming separate cells,
which keeps each subject's counts within one unit of `u·fraction`. With clips as the unit this is
the shape of a seen-subject clip-level holdout used in the lab's own audio work; declaring the unit
as `clip` is what makes such a split pass the seen-subject invariant, and the report records it.

**Test cohort and tolerances.** The tests use a seeded synthetic cohort (`imbalanced_cohort`): 60
subjects with 4–64 clips each in 2–4 recordings, binary label rate about 0.10 (per-subject rates
0, 0.05, 0.10 or 0.30), timepoints alternating by recording, and three multi-label columns around
0.5, 0.2 and 0.05. Before any splitter existed, with scikit-learn only, cohort seed 0 gave:

| Check | Required | Measured |
|---|---|---|
| (a) `StratifiedKFold(5, shuffle=True, random_state=0)` on rows shares subjects across partitions | ≥ 1 subject | all 60 subjects, in 5 of 5 folds |
| (b) label-blind group k-fold `max_abs_prevalence_deviation` (over all partitions) | > 0.04 | 0.0408 (see below) |
| (c) `StratifiedGroupKFold(5, shuffle=True, random_state=0)` max deviation | ≤ 0.02 | 0.0023 |
| (d) a 60/20/20 clip-level holdout per (subject, timepoint) puts clips of a recording in two partitions | ≥ 1 recording | 148 of 182 recordings |

All four held at the first seed tried, so the cohort seed is 0.

**The (b) baseline is deterministic.** It was first scikit-learn's `GroupKFold(5)` (0.0591), but
`GroupKFold` orders groups with an unstable `np.argsort` on group sizes, so ties between
equal-sized subjects depend on the sort kernel: a review reproduced a different result on the
locked numpy by disabling CPU features (`NPY_DISABLE_CPU_FEATURES="X86_V4,X86_V3"`), and older numpy
releases differ too. The test now uses the same label-blind, size-balancing idea written out with
stable ordering (largest subject first, ties by subject key, each into the fold with the fewest
rows, fold ties to the lowest index). Re-measured at cohort seed 0: 0.0408, still above the 0.04
tolerance, which was not changed.

(b) is one deterministic instance, 0.0008 above its threshold, so any change to the cohort generator,
its seed or this baseline must be re-measured before the test is trusted. It is sensitive: in the T4
re-review, 73 % of random tie orders for the same size-greedy assignment exceeded 0.04, and cohort
seeds 0–19 ranged from 0.030 to 0.149 under this baseline.

**Input checks.** Every entry point refuses, with a ValueError, rather than ignoring or failing
later: missing subject ids (None, NaN, empty); labels, units or strata of the wrong length; empty
partitions; duplicate, negative or out-of-range indices; non-integer index arrays, including boolean
masks (refused, not converted, since a mask and a 0/1 index array are easy to confuse); fractions
that are negative, at least 1, not three, or not summing to 1; fewer than 2 folds or inner folds; and
any argument the scheme or protocol would ignore (fractions with a k-fold, `n_splits` with
leave-one-subject-out or the holdout, `val_inner_splits` outside the k-folds, units or `unit_name`
under `unseen_subject`, strata with leave-one-subject-out, grouped shuffle or a multi-label matrix).
A nested seen-subject split that leaves a subject with one unit in an outer training pool raises a
`ProtocolViolation` naming that inner pool.

Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data.
*ECML PKDD 2011*, LNCS 6913, 145–158.

## Threshold grid

`src/senselab/utils/evaluation/data/threshold_grid.yaml` holds the candidate thresholds for
`tune_threshold`: 0.05 to 0.95 in steps of 0.05 (19 values, each `round(start + i·step, 4)`), ties
broken towards the threshold closest to 0.5 and then towards the lower one, the result rounded to
four decimals. The file holds only those five keys, with `#` descriptions; there is no
`derivation:` key, since that would be hashed with the values.

**Derivation.** This is a ported convention, not a fitted value: it is the grid and tie-break the
lab's earlier threshold-tuning code used (`provenance.md`, row `tune_threshold`), kept so that
results stay comparable. It was not fitted to any data. The earlier code built the grid with an
unrounded `arange`, so a score lying exactly on a grid value (0.65 against 0.6500000000000001) could
fall below its threshold there but not here; for scores off the grid, the choices agree (tested on
200 seeded cases against a reimplementation of the earlier rule). Among equally good
thresholds, the tie-break picks the one nearest 0.5, the least extreme operating point, and between
two equally near (0.3 and 0.7) the lower; the comparison is on integers in units of 10^-round, so float
error cannot decide it. The grid
lives in `data/` rather than in code so that changing it is a reviewed data change.

The default objective is balanced accuracy, computed by `metrics.balanced_accuracy`, the
same function as the report headline. `f1` (positive class) and `mcc` are available only as an
explicit, non-default choice; `ThresholdChoice.objective` records it, the report copies it into
`threshold.objective`, and the balanced-accuracy headline is still computed at whatever threshold
was chosen. Tuning is refused on a split name that marks held-out data: case-folded and cut into
letter and digit runs, a run that is `test`, `holdout` or `heldout`, adjacent runs joining to
`holdout`/`heldout` ("hold-out"), a run starting with `test` ("testset"), or `fold`/`outer`/`inner`
plus `test` ("outerTest"). "latest", "contest", "pretest" and "attestation" pass; "test_free_val"
does not. That
is a guard against slips, not a guarantee: the protection is the caller passing validation rows and
the report refusing a threshold on the split it was tuned on. Under nested cross-validation the
inner held-out partition is named `test`; tune on it under a name such as `inner_val`. On the T6 test's synthetic 90/10 data, the
BA-tuned threshold (0.5) gave balanced accuracy 0.858 on a fresh draw, against 0.793 for the
F1-tuned one (0.6), which is why F1 is not the default.

## Event metrics

**Onset-tolerance F1.** Predictions are taken in onset order; each takes the unused reference whose
onset is nearest its own, if the difference is within the tolerance in seconds (ties to the earlier
reference), and is a true positive, otherwise a false positive; references left unused are false
negatives. Matching is greedy and one-to-one, so two predictions near one reference give one true
positive and one false positive. With `label=None` labels are ignored altogether, as in the lab's
earlier implementation (`provenance.md`, row `event_f1`), which matched on onsets alone; with a label,
only events carrying it take part on either side. Precision is `None` without predictions, recall
`None` without references, and F1 `None` unless both are defined (the earlier code reported 0.0 and
pooled clips; to reproduce its numbers, filter labels on each side, call per clip, sum `tp`/`fp`/`fn`
over clips and compute the shares from the sums). A test reimplements the earlier matching and requires
identical counts on 300 random cases.

**Segment mAP.** Implemented clean-room from the ActivityNet temporal-localisation definition; no
lab implementation was read.

- Predictions of each class are ranked by confidence across all videos.
- Each prediction matches the unmatched reference of the same class and video with the highest
  temporal IoU, if that is at least the threshold; a duplicate detection is a false positive.
- Temporal IoU uses the evaluator's union, `len(ref) + len(pred) − intersection` (not
  `max(end) − min(start)`, which differs in the last bit at boundaries such as tIoU exactly 0.6).
- Ties follow the evaluator's order, for parity with published numbers: ranking is
  `confidences.argsort()[::-1]`, so tied confidences come out in reversed input order (videos in
  mapping order, then events); each prediction walks references by `tiou.argsort()[::-1]`, so on tied
  tIoU the later reference is taken first.
- Average precision is all-point interpolated.
- The mean (`np.mean`) is taken over classes present in the references, in their order of first
  appearance; thresholds that format to the same key ("0.6") are refused.
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
