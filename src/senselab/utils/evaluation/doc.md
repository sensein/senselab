# Evaluation utilities

Dataset-agnostic splitting and scoring for classification and event detection. Every split and every
report names its **protocol**, every classification report leads with **balanced accuracy** (the only
headline), and every metric comes with its chance or trivial **floor**. The design and its reasoning
are in `specs/20261007-evaluation-and-events/design.md`.

## Protocols

- `unseen_subject`: no subject appears in two partitions. Does the model work on new people?
- `seen_subject`: every test subject is also in train, no unit is in two partitions, and every
  subject has at least two units. Does it work on new material from people it has seen? The unit is
  declared by the caller (`unit_name`, required, for example `recording` or `clip`) and recorded in the
  split report.

A violation raises `ProtocolViolation` (counts only, never ids).

## Modules

| Module | Public API |
|---|---|
| `splits` | `grouped_splits`, `check_split`, `nested_grouped_splits`, `Split`, `SplitReport` (`to_reproduce()`) |
| `protocol` | `Protocol`, `ProtocolViolation`, `check_protocol` |
| `metrics` | `balanced_accuracy` (the headline definition), `legacy_macro_recall` (parity only), confusion-matrix helpers |
| `classification` | `score_classification`, `score_splits`, `ClassificationReport`, `EvaluationReport` (both `to_dict()`) |
| `thresholds` | `tune_threshold`, `load_threshold_grid`, `ThresholdChoice` |
| `bootstrap` | `cluster_bootstrap_ci`, `paired_cluster_bootstrap_delta`, `legacy_cluster_bootstrap`, `BootstrapCI` |
| `delong` | `delong_test`, `DeLongResult` |
| `event_metrics` | `event_f1`, `segment_map` |
| `export` | `reproduce_metrics` (one report in a reproduction driver's `metrics.json` vocabulary) |
| `cli` | `senselab-evaluate` |

## Report fields

A `ClassificationReport` holds: `protocol`, `split_name`, `labels`, `n`, `n_dropped`, `invalid_rate`,
`prevalence`, `predicted_prevalence`, `reference_prevalence`, `absent_labels`, `headline`
(`balanced_accuracy`), `headline_value`, `balanced_accuracy`, `secondary` (`accuracy`, `f1_macro`,
`mcc`, and `f1` for a binary task with a positive label), `per_class`, `confusion_matrix` (invalid
outputs as a column), `roc_auc` and `average_precision` (with scores), `threshold` and
`threshold_tuning`, `split_report`, and `ci` (cluster-bootstrap intervals when groups are given). Each
metric is a `MetricValue{value, floor, lift, normalized_lift}`; undefined values are `None`, never NaN.

## Floors

| Metric | Floor |
|---|---|
| balanced accuracy | 1/k, k = labels present in `y_true` |
| accuracy | majority-class prevalence |
| macro-F1 (present labels) | 1/k |
| MCC | 0 |
| positive-class F1 | 2p/(1+p) |
| ROC-AUC | 0.5 |
| average precision | p (binary), 1/k (macro one-vs-rest) |

## Example on synthetic data

```python
import numpy as np
from senselab.utils.evaluation.classification import score_classification
from senselab.utils.evaluation.splits import grouped_splits

rng = np.random.default_rng(0)
subjects = np.repeat([f"s{i}" for i in range(30)], 10)
y = np.where(rng.random(300) < 0.2, "pos", "neg")
scores = np.clip(np.where(y == "pos", 0.7, 0.35) + rng.normal(0, 0.15, 300), 0, 1)

splits, split_report = grouped_splits(
    y, subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0
)
test = splits[0].partitions["test"]
report = score_classification(
    y[test], np.where(scores[test] >= 0.5, "pos", "neg"),
    labels=["neg", "pos"], protocol="unseen_subject", split_name="fold0", n_dropped=0,
    split_report=split_report, y_score=scores[test], positive_label="pos",
    groups=subjects[test], n_resamples=500, seed=0,
)
print(report.headline, report.headline_value, report.balanced_accuracy.floor)
```

## Command line

```bash
senselab-evaluate predictions.csv --config my_eval.yaml --out results/
```

Every setting is in the config, deep-merged over `data/default.yaml` (keys: `schema`, `protocol`,
`columns`, `ground_truth`, `positive_label`, `threshold`, `bootstrap`, `reader`, `export`); `schema` and
`protocol` are required. The command writes `metrics.json` (the evaluation report, `config_sha256`,
`senselab_version`, row counts), `config.yaml` (the merged config) and `reproduce_metrics.json` for the
exported split. It exits 1 when the export is refused (for example a declared class absent from the
split), 2 on a usage or input error.
