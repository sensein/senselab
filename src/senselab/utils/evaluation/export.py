"""Export a classification report in a reproduction driver's `metrics.json` vocabulary.

`reproduce_metrics` maps one `ClassificationReport` onto the driver's flat schema: a single
`balanced_accuracy` headline, secondary entries named `accuracy`, `macro_f1`, `mcc`, `f1`, `auroc` and
`pr_auc`, `per_class` recall / precision / f1 / prevalence, a binary `confusion` block, the
`split_report`, `threshold_tuning` and `n`. Every entry carries its floor and lift, and its interval
when the report has one.

The driver needs a numeric recall and precision for every declared class, so:

- a declared class absent from the scored split is refused: its recall is undefined, and inventing one
  would misreport the split; drop the class from the schema or score a split that holds it;
- a present class that was never predicted exports precision 0.0: there were no predictions of it, so
  none was correct (scikit-learn's `zero_division=0` convention);
- the confusion block is left out when invalid outputs occur, since a 2 x 2 table has no cell for them,
  and for multi-class reports;
- multi-class average precision is not exported (the driver's `pr_auc` needs a declared positive class).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from senselab.utils.evaluation.bootstrap import BootstrapCI
from senselab.utils.evaluation.classification import ClassificationReport, MetricValue

# Report metric name to the driver's name, in the order the entries are written.
SECONDARY_NAMES = (("accuracy", "accuracy"), ("f1_macro", "macro_f1"), ("mcc", "mcc"), ("f1", "f1"))


def _entry(name: str, metric: MetricValue, n: int, ci: Optional[BootstrapCI]) -> Dict[str, Any]:
    """One headline or secondary entry with floor, lift, row count and interval bounds when known."""
    entry: Dict[str, Any] = {"name": name, "value": metric.value, "floor": metric.floor, "lift": metric.lift, "n": n}
    if ci is not None and ci.low is not None and ci.high is not None:
        entry["ci_low"], entry["ci_high"] = ci.low, ci.high
    return entry


def reproduce_metrics(report: ClassificationReport) -> Dict[str, Any]:
    """The report as a reproduction driver's `metrics.json` object.

    Args:
        report: One split's classification report.

    Returns:
        The driver-vocabulary object (validated by the driver, not here).

    Raises:
        ValueError: If a declared class is absent from the scored split.
    """
    if report.absent_labels:
        raise ValueError(
            f"{len(report.absent_labels)} declared classes are absent from split {report.split_name!r}: "
            "their recall is undefined; drop them from the schema or score a split that holds them"
        )
    cis = report.ci or {}
    secondary: List[Dict[str, Any]] = [
        _entry(driver, report.secondary[name], report.n, cis.get(name))
        for name, driver in SECONDARY_NAMES
        if name in report.secondary
    ]
    if report.roc_auc is not None:
        secondary.append(_entry("auroc", report.roc_auc, report.n, cis.get("roc_auc")))
    if report.average_precision is not None and report.positive_label is not None:
        secondary.append(_entry("pr_auc", report.average_precision, report.n, cis.get("average_precision")))
    per_class = {
        label: {
            "recall": values["recall"],
            "precision": 0.0 if values["precision"] is None else values["precision"],
            "f1": values["f1"],
            "prevalence": values["prevalence"],
        }
        for label, values in report.per_class.items()
    }
    data: Dict[str, Any] = {
        "headline": [_entry("balanced_accuracy", report.balanced_accuracy, report.n, cis.get("balanced_accuracy"))],
        "secondary": secondary,
        "per_class": per_class,
        "threshold_tuning": report.threshold_tuning,
        "n": report.n,
    }
    counts = report.confusion_matrix["counts"]
    labels = list(report.labels)
    if report.positive_label is not None and report.invalid_rate == 0:
        pos = labels.index(report.positive_label)
        neg = 1 - pos
        data["confusion"] = {
            "tp": counts[pos][pos],
            "fp": counts[neg][pos],
            "fn": counts[pos][neg],
            "tn": counts[neg][neg],
        }
    if report.split_report is not None:
        data["split_report"] = report.split_report.to_reproduce()
    return data
