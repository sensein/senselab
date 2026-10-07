# Evaluation utilities

Dataset-agnostic splitting and scoring for classification and event detection. Every split and
every report carries a `protocol` (`seen_subject` or `unseen_subject`), and every classification
report leads with balanced accuracy, giving each metric's chance or trivial floor beside it.

The design and its reasoning are in `specs/20261007-evaluation-and-events/design.md`.

This page is a stub. The functions below are added module by module; each section is filled in
when its module lands.

## Public functions

Splits (`splits.py`, `protocol.py`):

- `grouped_splits`: grouped, protocol-checked splits (stratified group k-fold, multi-label,
  combined strata, leave-one-subject-out, grouped shuffle, seen-subject k-fold and holdout).
- `nested_grouped_splits`: inner splits drawn from each outer training partition only.
- `check_split`: checks a split made elsewhere against a protocol and returns its report.

Classification (`classification.py`, `thresholds.py`):

- `score_classification`: the imbalance-aware report for one split.
- `score_splits`: reports for several splits, with prevalence per split.
- `tune_threshold`: a threshold chosen on a validation split from a grid kept in `data/`.

Uncertainty and comparison (`bootstrap.py`, `delong.py`):

- `cluster_bootstrap_ci`: percentile intervals from resampling groups.
- `paired_cluster_bootstrap_delta`: the paired group-level difference between two models.
- `delong_test`: DeLong's test for two correlated ROC-AUCs.

Events (`event_metrics.py`):

- `event_f1`: onset-tolerance F1 with one-to-one matching.
- `segment_map`: temporal-localisation mean average precision over tIoU thresholds.

Command line (`cli.py`):

- `senselab-evaluate <predictions.csv> --config <file.yaml> [--out DIR]`.
