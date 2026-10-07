"""The ``metrics.json`` vocabulary and its validator.

A step's adapter writes ``metrics.json`` from a fixed vocabulary; the step's ``metrics_schema`` (in
the manifest) narrows it to declared headline names, headline labels, count names and class labels.
The shape is::

    {
      "headline": [{"name", "label"?, "value", "ci_low"?, "ci_high"?, "numdf"?, "dendf"?, "n"?,
                    "n_folds"?, "floor"?, "lift"?, "unit"?}],
      "secondary": [entries of the same shape],
      "per_class": {label: {"recall"?, "precision"?, "f1"?, "prevalence"?}},
      "confusion": {"tp"?, "fp"?, "fn"?, "tn"?},
      "counts": {declared name: int},
      "split_report": {"protocol", "unit"?, "stratify"?, "folds"?: [{"n_groups"?, "prevalence"?}],
                       "invariant_ok"},
      "threshold_tuning": "validation" | "none" | "test",
      "n"?, "n_folds"?
    }

Each kind allows its own headline names (`HEADLINES_BY_KIND`) and secondary names
(`SECONDARY_BY_KIND`). A classification step's only headline is ``balanced_accuracy``: raw accuracy
means little without its null rate, so accuracy, top-1, macro F1, AUROC, PR-AUC, MCC and F1 are
secondary metrics, each validated like a headline entry and carrying its floor (for accuracy and
top-1, the majority-class prevalence) and lift. Detection, statistic and count steps take no
secondary metrics. ``headline_labels`` label secondary entries too. ``per_class`` belongs only to
classification and detection, and ``confusion`` only to classification (otherwise
``E_METRICS_KIND_KEY``). Headline values are range-checked: rates in [0, 1] (or [0, 100] for
``unit: pp``), ``mcc`` in [-1, 1] (likewise scaled), ``f_stat`` at least 0, and ``ci_low <= value <=
ci_high`` (``E_METRICS_RANGE``); ``floor`` takes the value's bounds and ``lift`` lies within plus or
minus their width. Per-class recall, precision and f1 lie in [0, 1]. With a top-level ``n``, a class
prevalence above 0 but below ``1/n`` (less than one example) gives ``E_METRICS_PREVALENCE_SUBSAMPLE``.

``delta``, ``target``, ``tolerance`` and ``pass`` are written only by the driver, at the top level, in a
headline entry or in a ``per_class`` entry.

**Classification** (single-label) and **detection** steps must carry:

- ``prevalence`` in [0, 1] for every declared class (classification: also ``recall`` and
  ``precision``, and prevalences summing to 1)
- ``floor`` and ``lift`` on every headline and secondary entry; standard floors are checked (`expected_floor`),
  with k the number of declared classes present (prevalence above 0); a declared class with
  prevalence 0 gives the warning ``W_CLASS_ABSENT``
- ``threshold_tuning``; ``test`` is accepted and returned as the warning ``W_TUNED_ON_TEST``
- a ``split_report``, unless the schema opts out with ``split: {none: whole_dataset}``; for
  classification every fold's prevalence covers every declared class

``n``, ``counts.*``, ``confusion.*``, a ``count`` headline value and, when the split unit is a person
(``subject``, ``family``) or not given, a fold's ``n_groups`` below the minimum cell size are replaced
by ``"<{min_cell}"``.

Results are fixed codes only; nothing from the rejected content is echoed.

A **metric address** names one value: ``<name>``, ``<name>@<label>`` (a headline or secondary entry;
the two name sets are disjoint), ``counts.<name>``, ``per_class.<label>.<field>`` or
``confusion.<cell>``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, get_args

if TYPE_CHECKING:
    from senselab.utils.reproduce.spec import MetricsSchema

MetricName = Literal[
    "balanced_accuracy",
    "macro_f1",
    "auroc",
    "f1",
    "accuracy",
    "top1",
    "avg_map",
    "map_at",
    "f_stat",
    "count",
    "pr_auc",
    "mcc",
]
MetricsKind = Literal["classification", "detection", "statistic", "count"]
SplitOptOutReason = Literal["whole_dataset"]
METRIC_NAMES: tuple[str, ...] = get_args(MetricName)
METRICS_KINDS: tuple[str, ...] = get_args(MetricsKind)
CLASSIFICATION_HEADLINE = "balanced_accuracy"
HEADLINES_BY_KIND: dict[str, frozenset[str]] = {
    "classification": frozenset({CLASSIFICATION_HEADLINE}),
    "detection": frozenset({"avg_map", "map_at"}),
    "statistic": frozenset({"f_stat"}),
    "count": frozenset({"count"}),
}
SECONDARY_BY_KIND: dict[str, frozenset[str]] = {
    "classification": frozenset({"accuracy", "top1", "macro_f1", "auroc", "pr_auc", "mcc", "f1"}),
    "detection": frozenset(),
    "statistic": frozenset(),
    "count": frozenset(),
}
CLASSIFICATION_KINDS = frozenset({"classification", "detection"})
RATE_METRICS = (
    HEADLINES_BY_KIND["classification"] | SECONDARY_BY_KIND["classification"] | HEADLINES_BY_KIND["detection"]
) - {"mcc"}
PRESENCE_ONLY_FLOORS = frozenset({"macro_f1", "avg_map", "map_at"})
ENTRY_NUMBERS = ("value", "ci_low", "ci_high", "numdf", "dendf", "floor", "lift")
ENTRY_INTS = ("n", "n_folds")
ENTRY_UNITS = ("fraction", "pp")
PER_CLASS_FIELDS = ("recall", "precision", "f1", "prevalence")
CONFUSION_CELLS = ("tp", "fp", "fn", "tn")
SPLIT_PROTOCOLS = ("seen_subject", "unseen_subject", "none")
SPLIT_UNITS = ("subject", "family", "session", "video", "clip", "recording", "row")
PERSON_UNITS = frozenset({"subject", "family"})
SPLIT_STRATIFY = ("none", "class", "class_timepoint", "class_site", "class_age", "timepoint")
FOLD_KEYS = ("n_groups", "prevalence")
TUNING_SPLITS = ("validation", "none", "test")
DRIVER_KEYS = frozenset({"delta", "target", "tolerance", "pass"})
TOP_KEYS = frozenset(
    {"headline", "secondary", "per_class", "confusion", "counts", "split_report", "threshold_tuning", "n", "n_folds"}
)
SPLIT_KEYS = frozenset({"protocol", "unit", "stratify", "folds", "invariant_ok"})
ENTRY_KEYS = frozenset({"name", "label", "unit", *ENTRY_NUMBERS, *ENTRY_INTS})
TOLERANCE = 1e-6

CODES = frozenset(
    {
        "E_METRICS_NOT_OBJECT",
        "E_METRICS_BAD_SHAPE",
        "E_METRICS_UNKNOWN_KEY",
        "E_METRICS_DRIVER_KEY",
        "E_METRICS_NOT_NUMBER",
        "E_METRICS_NOT_INT",
        "E_METRICS_NOT_BOOL",
        "E_METRICS_NEGATIVE",
        "E_METRICS_BAD_TOKEN",
        "E_METRICS_UNDECLARED_HEADLINE",
        "E_METRICS_UNDECLARED_SECONDARY",
        "E_METRICS_UNDECLARED_LABEL",
        "E_METRICS_UNDECLARED_COUNT",
        "E_METRICS_MISSING_HEADLINE",
        "E_METRICS_MISSING_SECONDARY",
        "E_METRICS_MISSING_COUNT",
        "E_METRICS_NO_PREVALENCE",
        "E_METRICS_NO_CLASS_RATES",
        "E_METRICS_PREVALENCE_RANGE",
        "E_METRICS_PREVALENCE_SUM",
        "E_METRICS_PREVALENCE_SUBSAMPLE",
        "E_METRICS_NO_FLOOR",
        "E_METRICS_BAD_FLOOR",
        "E_METRICS_NO_LIFT",
        "E_METRICS_LIFT_MISMATCH",
        "E_METRICS_RANGE",
        "E_METRICS_KIND_KEY",
        "E_METRICS_NO_TUNING_SPLIT",
        "E_METRICS_BAD_TUNING_SPLIT",
        "E_SPLIT_MISSING",
        "E_SPLIT_UNEXPECTED",
        "E_SPLIT_NO_PROTOCOL",
        "E_SPLIT_BAD_PROTOCOL",
        "E_SPLIT_BAD_INVARIANT",
        "E_SPLIT_FOLD_PREVALENCE",
    }
)
WARNINGS = frozenset({"W_TUNED_ON_TEST", "W_CLASS_ABSENT"})
ADDRESS_CODES = frozenset(
    {
        "E_METRICS_BAD_ADDRESS",
        "E_METRICS_UNDECLARED_HEADLINE",
        "E_METRICS_UNDECLARED_LABEL",
        "E_METRICS_UNDECLARED_COUNT",
        "E_METRICS_ABSENT",
        "E_METRICS_AMBIGUOUS",
        "E_METRICS_SUPPRESSED",
    }
)

Writer = Literal["adapter", "driver"]


class MetricsError(Exception):
    """A metric could not be read; the message is the fixed code only."""

    def __init__(self, code: str) -> None:
        """Store the code as both the message and ``code``.

        Args:
            code: One of `ADDRESS_CODES`.
        """
        super().__init__(code)
        self.code = code


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def expected_floor(name: str, prevalence: Mapping[str, float], *, k: int, positive: str | None) -> float | None:
    """The trivial-classifier value of a headline or secondary metric, as a fraction.

    Args:
        name: The metric name.
        prevalence: Class label to prevalence on the scored split, for every declared class.
        k: The number of declared classes present on the scored split (prevalence above 0); an absent
            class does not count, as in scikit-learn's balanced accuracy.
        positive: The declared positive class of a binary step, or None.

    Returns:
        ``1/k`` for balanced accuracy, 0.5 for AUROC, 0 for MCC, the largest
        prevalence for accuracy and top-1, ``p`` for PR-AUC and ``2p/(1+p)`` for F1 when a positive
        class ``p`` is declared; None when the floor is checked for presence only.
    """
    if name == CLASSIFICATION_HEADLINE:
        return 1.0 / k
    if name == "auroc":
        return 0.5
    if name == "mcc":
        return 0.0
    if name in ("accuracy", "top1"):
        return max(prevalence.values())
    if positive is not None and name == "pr_auc":
        return prevalence[positive]
    if positive is not None and name == "f1":
        p = prevalence[positive]
        return 2 * p / (1 + p)
    return None


class _Check:
    """One validation pass: collects codes and builds the suppressed copy in place."""

    def __init__(self, schema: MetricsSchema, writer: Writer, min_cell: int) -> None:
        self.schema = schema
        self.writer = writer
        self.min_cell = min_cell
        self.token = f"<{min_cell}"
        self.codes: set[str] = set()
        self.classification = schema.kind == "classification"
        self.imbalance = schema.kind in CLASSIFICATION_KINDS

    def keys(self, obj: Mapping[str, Any], allowed: frozenset[str], *, driver: bool) -> None:
        for key in obj:
            if key in DRIVER_KEYS and self.writer == "adapter":
                self.codes.add("E_METRICS_DRIVER_KEY")
            elif key in DRIVER_KEYS and driver:
                if key == "pass" and not isinstance(obj[key], bool):
                    self.codes.add("E_METRICS_NOT_BOOL")
                elif key != "pass" and not _is_number(obj[key]):
                    self.codes.add("E_METRICS_NOT_NUMBER")
            elif key not in allowed:
                self.codes.add("E_METRICS_UNKNOWN_KEY")

    def number(self, obj: Mapping[str, Any], key: str) -> None:
        if key in obj and not _is_number(obj[key]):
            self.codes.add("E_METRICS_NOT_NUMBER")

    def integer(self, obj: dict[str, Any], key: str, *, suppress: bool) -> None:
        if key not in obj:
            return
        value = obj[key]
        if suppress and value == self.token:
            return
        if not _is_number(value):
            self.codes.add("E_METRICS_NOT_NUMBER")
        elif isinstance(value, float) and not value.is_integer():
            self.codes.add("E_METRICS_NOT_INT")
        elif value < 0:
            self.codes.add("E_METRICS_NEGATIVE")
        elif suppress and value < self.min_cell:
            obj[key] = self.token

    def word(self, obj: Mapping[str, Any], key: str, allowed: tuple[str, ...]) -> None:
        if key in obj and (not isinstance(obj[key], str) or obj[key] not in allowed):
            self.codes.add("E_METRICS_BAD_TOKEN")

    def mapping(self, value: object) -> bool:
        if isinstance(value, dict):
            return True
        self.codes.add("E_METRICS_BAD_SHAPE")
        return False

    def prevalences(self, values: Mapping[str, Any], *, complete: bool) -> None:
        """Range-check a label → prevalence map; for classification, also its sum when ``complete``."""
        numbers = [v for v in values.values() if _is_number(v)]
        if any(v < 0 or v > 1 for v in numbers):
            self.codes.add("E_METRICS_PREVALENCE_RANGE")
        if self.classification and complete and len(numbers) == len(values) and abs(sum(numbers) - 1) > TOLERANCE:
            self.codes.add("E_METRICS_PREVALENCE_SUM")

    def scored_prevalence(self, classes: object) -> dict[str, float] | None:
        if not isinstance(classes, dict):
            return None
        found = {
            c: classes.get(c, {}).get("prevalence") for c in self.schema.per_class if isinstance(classes.get(c), dict)
        }
        if len(found) != len(self.schema.per_class) or not all(_is_number(v) for v in found.values()):
            return None
        return {c: float(v) for c, v in found.items()}

    def entries(self, entries: object, prevalence: dict[str, float] | None, *, secondary: bool) -> None:
        if not isinstance(entries, list):
            self.codes.add("E_METRICS_BAD_SHAPE")
            return
        declared = self.schema.secondary if secondary else self.schema.headline
        for entry in entries:
            if not self.mapping(entry):
                continue
            self.keys(entry, ENTRY_KEYS, driver=True)
            name = entry.get("name")
            if name not in declared:
                self.codes.add("E_METRICS_UNDECLARED_SECONDARY" if secondary else "E_METRICS_UNDECLARED_HEADLINE")
            if "label" in entry and entry["label"] not in self.schema.headline_labels:
                self.codes.add("E_METRICS_UNDECLARED_LABEL")
            if "value" not in entry:
                self.codes.add("E_METRICS_BAD_SHAPE")
            for key in ENTRY_NUMBERS:
                self.number(entry, key)
            if name == "count":
                self.integer(entry, "value", suppress=True)
            self.integer(entry, "n", suppress=True)
            self.integer(entry, "n_folds", suppress=False)
            self.word(entry, "unit", ENTRY_UNITS)
            self.value_range(entry)
            if self.imbalance:
                self.floor(entry, prevalence)

    def value_range(self, entry: dict[str, Any]) -> None:
        name, value = entry.get("name"), entry.get("value")
        if not isinstance(name, str):
            return
        scale = 100.0 if entry.get("unit") == "pp" else 1.0
        if name == "mcc":
            low, high = -scale, scale
        elif name in RATE_METRICS:
            low, high = 0.0, scale
        elif name == "f_stat":
            low, high = 0.0, math.inf
        else:
            return
        slack = TOLERANCE * scale
        floor, lift = entry.get("floor"), entry.get("lift")
        width = high - low
        for bounded in (value, floor):
            if _is_number(bounded) and not low - slack <= bounded <= high + slack:
                self.codes.add("E_METRICS_RANGE")
        if _is_number(lift) and abs(lift) > width + slack:
            self.codes.add("E_METRICS_RANGE")
        if not _is_number(value):
            return
        ci_low, ci_high = entry.get("ci_low"), entry.get("ci_high")
        if _is_number(ci_low) and ci_low > value + slack:
            self.codes.add("E_METRICS_RANGE")
        if _is_number(ci_high) and ci_high < value - slack:
            self.codes.add("E_METRICS_RANGE")

    def floor(self, entry: dict[str, Any], prevalence: dict[str, float] | None) -> None:
        if "floor" not in entry:
            self.codes.add("E_METRICS_NO_FLOOR")
        if "lift" not in entry:
            self.codes.add("E_METRICS_NO_LIFT")
        value, floor, lift = (entry.get(k) for k in ("value", "floor", "lift"))
        scale = 100.0 if entry.get("unit") == "pp" else 1.0
        if _is_number(value) and _is_number(floor) and _is_number(lift):
            if abs(lift - (value - floor)) > TOLERANCE * max(scale, abs(value)):
                self.codes.add("E_METRICS_LIFT_MISMATCH")
        name = entry.get("name")
        if not _is_number(floor) or not isinstance(name, str) or prevalence is None:
            return
        present = sum(1 for p in prevalence.values() if p > 0)
        if not present:
            return
        expected = expected_floor(name, prevalence, k=present, positive=self.schema.positive)
        if expected is not None and abs(floor - expected * scale) > TOLERANCE * scale:
            self.codes.add("E_METRICS_BAD_FLOOR")

    def per_class(self, classes: object, n: object = None) -> None:
        if not self.mapping(classes):
            return
        assert isinstance(classes, dict)
        for label, fields in classes.items():
            if label not in self.schema.per_class:
                self.codes.add("E_METRICS_UNDECLARED_LABEL")
            if not self.mapping(fields):
                continue
            self.keys(fields, frozenset(PER_CLASS_FIELDS), driver=True)
            for key in PER_CLASS_FIELDS:
                self.number(fields, key)
            for key in ("recall", "precision", "f1"):
                rate = fields.get(key)
                if _is_number(rate) and not 0 <= rate <= 1:
                    self.codes.add("E_METRICS_RANGE")
        declared = {c: classes[c] for c in self.schema.per_class if isinstance(classes.get(c), dict)}
        given = {c: f["prevalence"] for c, f in declared.items() if "prevalence" in f}
        self.prevalences(given, complete=len(given) == len(self.schema.per_class))
        if any(_is_number(p) and p == 0 for p in given.values()):
            self.codes.add("W_CLASS_ABSENT")
        if _is_number(n) and n > 0 and any(_is_number(p) and 0 < p and p * n < 1 - TOLERANCE for p in given.values()):
            self.codes.add("E_METRICS_PREVALENCE_SUBSAMPLE")

    def class_fields(self, classes: object) -> None:
        declared = self.schema.per_class
        rows = [classes.get(c) if isinstance(classes, dict) else None for c in declared]
        if not declared or not all(isinstance(r, dict) and "prevalence" in r for r in rows):
            self.codes.add("E_METRICS_NO_PREVALENCE")
        if self.classification and not all(isinstance(r, dict) and {"recall", "precision"} <= set(r) for r in rows):
            self.codes.add("E_METRICS_NO_CLASS_RATES")

    def confusion(self, cells: object) -> None:
        if not self.mapping(cells):
            return
        assert isinstance(cells, dict)
        self.keys(cells, frozenset(CONFUSION_CELLS), driver=False)
        for cell in CONFUSION_CELLS:
            self.integer(cells, cell, suppress=True)

    def counts(self, counts: object) -> None:
        if not self.mapping(counts):
            return
        assert isinstance(counts, dict)
        for name in list(counts):
            if name not in self.schema.counts:
                self.codes.add("E_METRICS_UNDECLARED_COUNT")
            self.integer(counts, name, suppress=True)
        if set(self.schema.counts) - set(counts):
            self.codes.add("E_METRICS_MISSING_COUNT")

    def split_report(self, split: object) -> None:
        if not self.mapping(split):
            return
        assert isinstance(split, dict)
        self.keys(split, SPLIT_KEYS, driver=False)
        protocol = split.get("protocol")
        if "protocol" not in split:
            self.codes.add("E_SPLIT_NO_PROTOCOL")
        elif protocol not in SPLIT_PROTOCOLS:
            self.codes.add("E_SPLIT_BAD_PROTOCOL")
        invariant = split.get("invariant_ok")
        if not isinstance(invariant, bool) or (protocol == "none" and invariant):
            self.codes.add("E_SPLIT_BAD_INVARIANT")
        self.word(split, "unit", SPLIT_UNITS)
        self.word(split, "stratify", SPLIT_STRATIFY)
        if "folds" not in split:
            return
        folds = split["folds"]
        if not isinstance(folds, list):
            self.codes.add("E_METRICS_BAD_SHAPE")
            return
        unit = split.get("unit")
        people = unit is None or unit in PERSON_UNITS
        for fold in folds:
            if self.mapping(fold):
                self.fold(fold, people=people)

    def fold(self, fold: dict[str, Any], *, people: bool) -> None:
        self.keys(fold, frozenset(FOLD_KEYS), driver=False)
        self.integer(fold, "n_groups", suppress=people)
        prevalence = fold.get("prevalence")
        if isinstance(prevalence, dict):
            if set(prevalence) - set(self.schema.per_class):
                self.codes.add("E_METRICS_UNDECLARED_LABEL")
            for label in prevalence:
                self.number(prevalence, label)
            complete = set(self.schema.per_class) <= set(prevalence)
            if self.classification and not complete:
                self.codes.add("E_SPLIT_FOLD_PREVALENCE")
            self.prevalences(prevalence, complete=complete)
        elif "prevalence" in fold and not self.classification:
            self.number(fold, "prevalence")
            if _is_number(prevalence) and not 0 <= prevalence <= 1:
                self.codes.add("E_METRICS_PREVALENCE_RANGE")
        elif self.classification:
            self.codes.add("E_SPLIT_FOLD_PREVALENCE")

    def run(self, obj: dict[str, Any]) -> None:
        self.keys(obj, TOP_KEYS, driver=True)
        if "per_class" in obj and not self.imbalance:
            self.codes.add("E_METRICS_KIND_KEY")
        elif "per_class" in obj:
            self.per_class(obj["per_class"], obj.get("n"))
        prevalence = self.scored_prevalence(obj.get("per_class"))
        for key, declared, missing in (
            ("headline", self.schema.headline, "E_METRICS_MISSING_HEADLINE"),
            ("secondary", self.schema.secondary, "E_METRICS_MISSING_SECONDARY"),
        ):
            entries = obj.get(key, [])
            self.entries(entries, prevalence, secondary=key == "secondary")
            if isinstance(entries, list):
                names = {e.get("name") for e in entries if isinstance(e, dict) and isinstance(e.get("name"), str)}
                if set(declared) - names:
                    self.codes.add(missing)
        if "confusion" in obj and not self.classification:
            self.codes.add("E_METRICS_KIND_KEY")
        elif "confusion" in obj:
            self.confusion(obj["confusion"])
        self.counts(obj.get("counts", {}))
        if "split_report" in obj:
            if self.schema.split is not None:
                self.codes.add("E_SPLIT_UNEXPECTED")
            self.split_report(obj["split_report"])
        elif self.imbalance and self.schema.split is None:
            self.codes.add("E_SPLIT_MISSING")
        tuning = obj.get("threshold_tuning")
        if "threshold_tuning" in obj and tuning not in TUNING_SPLITS:
            self.codes.add("E_METRICS_BAD_TUNING_SPLIT")
        if tuning == "test":
            self.codes.add("W_TUNED_ON_TEST")
        self.integer(obj, "n", suppress=True)
        self.integer(obj, "n_folds", suppress=False)
        if self.imbalance:
            self.class_fields(obj.get("per_class"))
            if "threshold_tuning" not in obj:
                self.codes.add("E_METRICS_NO_TUNING_SPLIT")


def validate_metrics(
    obj: object, schema: MetricsSchema, *, writer: Writer = "adapter", min_cell: int = 5
) -> tuple[dict[str, Any] | None, list[str]]:
    """Check a ``metrics.json`` object against the vocabulary and a step's schema, and suppress small cells.

    The driver validates the adapter's output, then (after writing the ``expect`` results) the final
    object with ``writer="driver"``; both passes must use the same ``min_cell``.

    Args:
        obj: The parsed ``metrics.json``.
        schema: The step's ``metrics_schema``; its ``kind`` decides whether prevalence, floors, lift,
            the threshold-tuning split and a split report are required.
        writer: ``adapter`` for an adapter's output, which may not carry ``delta``, ``target``,
            ``tolerance`` or ``pass``; ``driver`` once the driver has evaluated ``expect``.
        min_cell: Suppressed cells below this become ``"<{min_cell}"``.

    Returns:
        ``(clean, codes)``: ``codes`` is sorted and unique, drawn from `CODES` (errors) and `WARNINGS`.
        ``clean`` is the suppressed copy when there is no error code, otherwise None. ``obj`` is not
        modified.
    """
    if not isinstance(obj, dict):
        return None, ["E_METRICS_NOT_OBJECT"]
    clean = copy.deepcopy(obj)
    check = _Check(schema, writer, min_cell)
    check.run(clean)
    codes = sorted(check.codes)
    if any(c.startswith("E_") for c in codes):
        return None, codes
    return clean, codes


def _parse_address(address: str) -> tuple[str, ...] | None:
    parts = address.split(".")
    if len(parts) == 1:
        name, _, label = address.partition("@")
        return ("headline", name, label) if name and (label or "@" not in address) else None
    if len(parts) == 2 and parts[0] == "counts" and parts[1]:
        return ("counts", parts[1])
    if len(parts) == 2 and parts[0] == "confusion" and parts[1] in CONFUSION_CELLS:
        return ("confusion", parts[1])
    if len(parts) == 3 and parts[0] == "per_class" and parts[1] and parts[2] in PER_CLASS_FIELDS:
        return ("per_class", parts[1], parts[2])
    return None


def metric_address_error(address: str, schema: MetricsSchema) -> str | None:
    """Check that a metric address names a value the schema allows.

    Args:
        address: ``<name>``, ``<name>@<label>`` (a headline or secondary metric), ``counts.<name>``,
            ``per_class.<label>.<field>`` or ``confusion.<cell>``.
        schema: The step's ``metrics_schema``.

    Returns:
        None when the address is valid, otherwise a fixed code from `ADDRESS_CODES`
        (``E_METRICS_UNDECLARED_HEADLINE`` for a name in neither ``headline`` nor ``secondary``).
    """
    parsed = _parse_address(address)
    if parsed is None:
        return "E_METRICS_BAD_ADDRESS"
    if parsed[0] == "headline":
        if parsed[1] not in (*schema.headline, *schema.secondary):
            return "E_METRICS_UNDECLARED_HEADLINE"
        if parsed[2] and parsed[2] not in schema.headline_labels:
            return "E_METRICS_UNDECLARED_LABEL"
    if parsed[0] == "counts" and parsed[1] not in schema.counts:
        return "E_METRICS_UNDECLARED_COUNT"
    if parsed[0] == "per_class" and parsed[1] not in schema.per_class:
        return "E_METRICS_UNDECLARED_LABEL"
    return None


Quantity = Literal["rate", "mcc", "statistic", "count"]
SCALED_QUANTITIES = frozenset({"rate", "mcc"})


def metric_quantity(address: str) -> Quantity | None:
    """The kind of number a metric address names, which fixes its range and whether it takes a unit.

    Args:
        address: A metric address (see `metric_address_error`).

    Returns:
        ``rate`` (a headline or secondary rate, or a per-class field, in [0, 1] or percentage points), ``mcc`` (in
        [-1, 1], likewise scaled), ``statistic`` (``f_stat``) or ``count`` (a ``count`` headline,
        ``counts.*`` or ``confusion.*``); None for a malformed address or an unknown headline name.
    """
    parsed = _parse_address(address)
    if parsed is None:
        return None
    if parsed[0] == "per_class":
        return "rate"
    if parsed[0] in ("counts", "confusion"):
        return "count"
    name = parsed[1]
    if name == "mcc":
        return "mcc"
    if name in RATE_METRICS:
        return "rate"
    if name == "f_stat":
        return "statistic"
    if name == "count":
        return "count"
    return None


def metric_value(obj: Mapping[str, Any], address: str) -> float:
    """Read one value from a validated ``metrics.json`` object.

    Args:
        obj: A clean object returned by `validate_metrics`.
        address: A metric address (see `metric_address_error`). A bare name must match one headline or
            secondary entry; ``<name>@<label>`` selects the entry with that label.

    Returns:
        The number.

    Raises:
        MetricsError: ``E_METRICS_BAD_ADDRESS``, ``E_METRICS_ABSENT``, ``E_METRICS_AMBIGUOUS`` or
            ``E_METRICS_SUPPRESSED``.
    """
    parsed = _parse_address(address)
    if parsed is None:
        raise MetricsError("E_METRICS_BAD_ADDRESS")
    value: object
    if parsed[0] == "headline":
        _, name, label = parsed
        found = [
            e
            for e in (*obj.get("headline", []), *obj.get("secondary", []))
            if e.get("name") == name and (e.get("label") == label if label else True) and "value" in e
        ]
        if len(found) > 1:
            raise MetricsError("E_METRICS_AMBIGUOUS")
        value = found[0]["value"] if found else None
    elif parsed[0] == "per_class":
        value = obj.get("per_class", {}).get(parsed[1], {}).get(parsed[2])
    else:
        value = obj.get(parsed[0], {}).get(parsed[1])
    if isinstance(value, str):
        raise MetricsError("E_METRICS_SUPPRESSED")
    if not _is_number(value):
        raise MetricsError("E_METRICS_ABSENT")
    assert isinstance(value, (int, float))
    return value
