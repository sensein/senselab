"""The ``metrics.json`` vocabulary and its validator.

A step's adapter writes ``metrics.json`` from a fixed vocabulary; the step's ``metrics_schema`` (in
the manifest) narrows it to declared headline names, headline labels, count names and class labels.
The shape is::

    {
      "headline": [{"name", "label"?, "value", "ci_low"?, "ci_high"?, "numdf"?, "dendf"?, "n"?,
                    "n_folds"?, "floor"?, "lift"?, "unit"?}],
      "per_class": {label: {"recall"?, "precision"?, "f1"?, "prevalence"?}},
      "confusion": {"tp"?, "fp"?, "fn"?, "tn"?},
      "counts": {declared name: int},
      "split_report": {"protocol", "unit"?, "stratify"?, "folds"?: [{"n_groups"?, "prevalence"?}],
                       "invariant_ok"},
      "threshold_tuning": "validation" | "none",
      "n"?, "n_folds"?
    }

``delta``, ``target``, ``tolerance`` and ``pass`` are written only by the driver, at the top level,
in a headline entry or in a ``per_class`` entry. Classification and detection steps must carry a
``prevalence`` for every declared class, a ``floor`` on every headline entry, and the split thresholds
were tuned on (``threshold_tuning``). ``n``, ``counts.*`` and ``confusion.*`` below the minimum cell
size are replaced by ``"<{min_cell}"``.

Results are fixed error codes only; nothing from the rejected content is echoed.

A **metric address** names one value: ``<headline>``, ``<headline>@<label>``, ``counts.<name>``,
``per_class.<label>.<field>`` or ``confusion.<cell>``.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from senselab.utils.reproduce.spec import MetricsSchema

HeadlineName = Literal[
    "balanced_accuracy",
    "macro_recall",
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
HEADLINE_NAMES: tuple[str, ...] = get_args(HeadlineName)
METRICS_KINDS: tuple[str, ...] = get_args(MetricsKind)
CLASSIFICATION_KINDS = frozenset({"classification", "detection"})
ENTRY_NUMBERS = ("value", "ci_low", "ci_high", "numdf", "dendf", "floor", "lift")
ENTRY_INTS = ("n", "n_folds")
ENTRY_UNITS = ("fraction", "pp")
PER_CLASS_FIELDS = ("recall", "precision", "f1", "prevalence")
CONFUSION_CELLS = ("tp", "fp", "fn", "tn")
SPLIT_PROTOCOLS = ("seen_subject", "unseen_subject")
FOLD_KEYS = ("n_groups", "prevalence")
TUNING_SPLITS = ("validation", "none")
DRIVER_KEYS = frozenset({"delta", "target", "tolerance", "pass"})
TOP_KEYS = frozenset(
    {"headline", "per_class", "confusion", "counts", "split_report", "threshold_tuning", "n", "n_folds"}
)
SPLIT_KEYS = frozenset({"protocol", "unit", "stratify", "folds", "invariant_ok"})
ENTRY_KEYS = frozenset({"name", "label", "unit", *ENTRY_NUMBERS, *ENTRY_INTS})
LIFT_TOLERANCE = 1e-6
_TOKEN = re.compile(r"^[a-z]+(_[a-z]+)*$")
_TOKEN_MAX = 40

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
        "E_METRICS_UNDECLARED_LABEL",
        "E_METRICS_UNDECLARED_COUNT",
        "E_METRICS_MISSING_HEADLINE",
        "E_METRICS_MISSING_COUNT",
        "E_METRICS_NO_PREVALENCE",
        "E_METRICS_NO_FLOOR",
        "E_METRICS_NO_TUNING_SPLIT",
        "E_METRICS_BAD_TUNING_SPLIT",
        "E_METRICS_LIFT_MISMATCH",
        "E_SPLIT_NO_PROTOCOL",
        "E_SPLIT_BAD_PROTOCOL",
        "E_SPLIT_BAD_INVARIANT",
    }
)
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


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class _Check:
    """One validation pass: collects codes and builds the suppressed copy in place."""

    def __init__(self, schema: MetricsSchema, writer: Writer, min_cell: int) -> None:
        self.schema = schema
        self.writer = writer
        self.min_cell = min_cell
        self.token = f"<{min_cell}"
        self.codes: set[str] = set()

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

    def word(self, obj: Mapping[str, Any], key: str, allowed: tuple[str, ...] | None = None) -> None:
        if key not in obj:
            return
        value = obj[key]
        ok = isinstance(value, str) and len(value) <= _TOKEN_MAX and bool(_TOKEN.match(value))
        if not ok or (allowed is not None and value not in allowed):
            self.codes.add("E_METRICS_BAD_TOKEN")

    def mapping(self, value: object) -> bool:
        if isinstance(value, dict):
            return True
        self.codes.add("E_METRICS_BAD_SHAPE")
        return False

    def headline(self, entries: object) -> None:
        if not isinstance(entries, list):
            self.codes.add("E_METRICS_BAD_SHAPE")
            return
        classification = self.schema.kind in CLASSIFICATION_KINDS
        for entry in entries:
            if not self.mapping(entry):
                continue
            self.keys(entry, ENTRY_KEYS, driver=True)
            name = entry.get("name")
            if name not in self.schema.headline:
                self.codes.add("E_METRICS_UNDECLARED_HEADLINE")
            if "label" in entry and entry["label"] not in self.schema.headline_labels:
                self.codes.add("E_METRICS_UNDECLARED_LABEL")
            if "value" not in entry:
                self.codes.add("E_METRICS_BAD_SHAPE")
            for key in ENTRY_NUMBERS:
                self.number(entry, key)
            if name == "count":
                self.integer(entry, "value", suppress=False)
            self.integer(entry, "n", suppress=True)
            self.integer(entry, "n_folds", suppress=False)
            self.word(entry, "unit", ENTRY_UNITS)
            if classification and "floor" not in entry:
                self.codes.add("E_METRICS_NO_FLOOR")
            parts = [entry.get(k) for k in ("value", "floor", "lift")]
            if all(_is_number(p) for p in parts):
                value, floor, lift = parts
                if abs(lift - (value - floor)) > LIFT_TOLERANCE * max(1.0, abs(value)):
                    self.codes.add("E_METRICS_LIFT_MISMATCH")
        names = {e.get("name") for e in entries if isinstance(e, dict) and isinstance(e.get("name"), str)}
        if set(self.schema.headline) - names:
            self.codes.add("E_METRICS_MISSING_HEADLINE")

    def per_class(self, classes: object) -> None:
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

    def prevalence_present(self, classes: object) -> bool:
        if not isinstance(classes, dict) or not self.schema.per_class:
            return False
        return all(isinstance(classes.get(c), dict) and "prevalence" in classes[c] for c in self.schema.per_class)

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
        if "protocol" not in split:
            self.codes.add("E_SPLIT_NO_PROTOCOL")
        elif split["protocol"] not in SPLIT_PROTOCOLS:
            self.codes.add("E_SPLIT_BAD_PROTOCOL")
        if not isinstance(split.get("invariant_ok"), bool):
            self.codes.add("E_SPLIT_BAD_INVARIANT")
        self.word(split, "unit")
        self.word(split, "stratify")
        if "folds" not in split:
            return
        folds = split["folds"]
        if not isinstance(folds, list):
            self.codes.add("E_METRICS_BAD_SHAPE")
            return
        for fold in folds:
            if not self.mapping(fold):
                continue
            self.keys(fold, frozenset(FOLD_KEYS), driver=False)
            self.integer(fold, "n_groups", suppress=False)
            prevalence = fold.get("prevalence")
            if isinstance(prevalence, dict):
                if set(prevalence) - set(self.schema.per_class):
                    self.codes.add("E_METRICS_UNDECLARED_LABEL")
                for label in prevalence:
                    self.number(prevalence, label)
            elif "prevalence" in fold:
                self.number(fold, "prevalence")

    def run(self, obj: dict[str, Any]) -> None:
        self.keys(obj, TOP_KEYS, driver=True)
        self.headline(obj.get("headline", []))
        if "per_class" in obj:
            self.per_class(obj["per_class"])
        if "confusion" in obj:
            self.confusion(obj["confusion"])
        self.counts(obj.get("counts", {}))
        if "split_report" in obj:
            self.split_report(obj["split_report"])
        if "threshold_tuning" in obj and obj["threshold_tuning"] not in TUNING_SPLITS:
            self.codes.add("E_METRICS_BAD_TUNING_SPLIT")
        self.integer(obj, "n", suppress=True)
        self.integer(obj, "n_folds", suppress=False)
        if self.schema.kind in CLASSIFICATION_KINDS:
            if not self.prevalence_present(obj.get("per_class")):
                self.codes.add("E_METRICS_NO_PREVALENCE")
            if "threshold_tuning" not in obj:
                self.codes.add("E_METRICS_NO_TUNING_SPLIT")


def validate_metrics(
    obj: object, schema: MetricsSchema, *, writer: Writer = "adapter", min_cell: int = 5
) -> tuple[dict[str, Any] | None, list[str]]:
    """Check a ``metrics.json`` object against the vocabulary and a step's schema, and suppress small cells.

    Args:
        obj: The parsed ``metrics.json``.
        schema: The step's ``metrics_schema``; its ``kind`` decides whether prevalence, floor and the
            threshold-tuning split are required.
        writer: ``adapter`` for an adapter's output, which may not carry ``delta``, ``target``,
            ``tolerance`` or ``pass``; ``driver`` once the driver has evaluated ``expect``.
        min_cell: ``n``, ``counts.*`` and ``confusion.*`` below this become ``"<{min_cell}"``.

    Returns:
        ``(clean, codes)``: the suppressed copy and ``[]`` when valid, otherwise ``None`` and the
        sorted, unique error codes from `CODES`. ``obj`` is not modified.
    """
    if not isinstance(obj, dict):
        return None, ["E_METRICS_NOT_OBJECT"]
    clean = copy.deepcopy(obj)
    check = _Check(schema, writer, min_cell)
    check.run(clean)
    if check.codes:
        return None, sorted(check.codes)
    return clean, []


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
        address: ``<headline>``, ``<headline>@<label>``, ``counts.<name>``, ``per_class.<label>.<field>``
            or ``confusion.<cell>``.
        schema: The step's ``metrics_schema``.

    Returns:
        None when the address is valid, otherwise a fixed code from `ADDRESS_CODES`.
    """
    parsed = _parse_address(address)
    if parsed is None:
        return "E_METRICS_BAD_ADDRESS"
    if parsed[0] == "headline":
        if parsed[1] not in schema.headline:
            return "E_METRICS_UNDECLARED_HEADLINE"
        if parsed[2] and parsed[2] not in schema.headline_labels:
            return "E_METRICS_UNDECLARED_LABEL"
    if parsed[0] == "counts" and parsed[1] not in schema.counts:
        return "E_METRICS_UNDECLARED_COUNT"
    if parsed[0] == "per_class" and parsed[1] not in schema.per_class:
        return "E_METRICS_UNDECLARED_LABEL"
    return None


def metric_value(obj: Mapping[str, Any], address: str) -> float:
    """Read one value from a validated ``metrics.json`` object.

    Args:
        obj: A clean object returned by `validate_metrics`.
        address: A metric address (see `metric_address_error`). A bare headline name must match one
            entry; ``<name>@<label>`` selects the entry with that label.

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
            for e in obj.get("headline", [])
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
