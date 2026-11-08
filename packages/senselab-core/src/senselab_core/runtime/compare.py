"""Output agreement by standard comparisons: hash equality, exact equality, and `isclose` within named thresholds.

`runtime/data/comparison.yaml` (`comparison@1.0`) declares, per kernel type, how each field of its wire form
compares (`COMPARISONS`); the thresholds its rules name are the run config's (`Thresholds.load`, over the packaged
`runtime/data/run_config.yaml`). `compare` decides whether two outputs agree and returns the field-level
differences, each with the threshold it was judged by. Outputs are compared as their flattened leaves (`flatten`).
"""

import difflib
import functools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy
from pydantic import Field

from senselab_core.kernel._base import FiniteFloat, Frozen, value_digest
from senselab_core.kernel.formats import FORMAT_PATTERN, RUN_CONFIG_FORMAT
from senselab_core.kernel.wire import WIRE_FORMATS
from senselab_core.runtime.calibration import RunCalibration
from senselab_core.runtime.config import load_yaml

COMPARISON_FORMAT = "comparison@1.0"

Value = list[float | None] | str
"""One output leaf: numbers as a float list, a None where a value is absent, or anything else as its JSON."""
Fields = dict[str, Value]
"""An output's leaves by dotted path."""

FLATTEN_VERSION = "flatten@2"
"""The version of `flatten` and its path form, recorded with every flattened file; a change to either bumps it."""
_INDEX = re.compile(r"\[\d+\]")


def stale_flatten(raw: Mapping[str, object], fmt: str) -> str | None:
    """Return why a file of flattened fields is not of format `fmt` and `FLATTEN_VERSION`, or None when it is."""
    if raw.get("format") == fmt and raw.get("flatten") == FLATTEN_VERSION:
        return None
    return f"written by {raw.get('flatten', 'flatten@1')} as {raw.get('format')}, read {FLATTEN_VERSION} as {fmt}"


def index_free(path: str) -> str:
    """Return `path` with every `[n]` as `[*]`."""
    return _INDEX.sub("[*]", path)


def _numeric(value: object) -> bool:
    if isinstance(value, (bool, str)):
        return False
    if isinstance(value, numpy.ndarray):
        return value.dtype.kind in "biuf"
    try:
        numpy.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


class NotWireForm(TypeError):
    """A value `flatten` reads that is not a wire form, such as a decoded kernel value."""


def flatten(value: object, prefix: str = "") -> Fields:
    """Return the leaves of a wire form `value` by dotted path (`FLATTEN_VERSION`).

    Numbers, None and numeric arrays are float lists, a None (or NaN) where a value is absent; a non-numeric
    array is flattened element by element, like a list; a string or bool is its JSON.

    Raises:
        NotWireForm: a leaf is none of these (a kernel value compares as its `wire_form`); it names the path.
    """
    if isinstance(value, numpy.ndarray) and not _numeric(value):
        return flatten(value.tolist(), prefix)
    if isinstance(value, Mapping):
        out: Fields = {}
        for key in sorted(value, key=str):
            out |= flatten(value[key], f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(value, (list, tuple)) and not _numeric(value):
        out = {}
        for n, element in enumerate(value):
            out |= flatten(element, f"{prefix}[{n}]")
        return out
    if _numeric(value):
        floats = numpy.asarray(value, dtype=float).ravel().tolist()
        return {prefix: [None if math.isnan(x) else x for x in floats]}
    if isinstance(value, (str, bool)):
        return {prefix: json.dumps(value)}
    raise NotWireForm(f"{prefix or 'the value'} is a {type(value).__name__}, not a wire form")


class Threshold(Frozen):
    """`isclose` bounds: `actual` agrees with `expected` when `|actual - expected| <= atol + rtol * |expected|`."""

    rtol: Annotated[FiniteFloat, Field(ge=0)]
    atol: Annotated[FiniteFloat, Field(ge=0)]


class UnknownThreshold(ValueError):
    """A run config names a threshold, or a key, that the packaged run config does not define."""


class Thresholds(Frozen):
    """The named thresholds a comparison reads; `canonical_digest` is what a verify or parity decision records."""

    by_name: Mapping[str, Threshold]

    @classmethod
    def load(cls, override: Mapping[str, Any] | Path | None = None) -> "Thresholds":
        """Return the `thresholds` of `load_run_config(override)`."""
        return load_run_config(override).thresholds


READABLE_RUN_CONFIG_FORMATS = frozenset({"run-config@2.0", RUN_CONFIG_FORMAT})


class LoadedRunConfig(Frozen):
    """A run config's validated sections: the comparison `thresholds` and the `calibration` overrides."""

    thresholds: Thresholds
    calibration: RunCalibration


def load_run_config(override: Mapping[str, Any] | Path | None = None) -> LoadedRunConfig:
    """Return the packaged run config's sections with `override`'s (a run config mapping or YAML) merged over.

    A threshold of `override` may set `rtol`, `atol` or both; `calibration` may set `trust`.

    Raises:
        UnknownThreshold: `override` names a key, threshold or calibration key the packaged run config lacks, naming
            the closest.
        ValueError: `override` declares a format other than `run-config@2.0` or `@2.1`, the packaged file is
            not `run-config@2.1` with exactly `rtol` and `atol` per threshold, or a
            merged bound or calibration value is invalid.
    """
    merged = {name: dict(bound) for name, bound in _packaged_thresholds().items()}
    raw = load_yaml(override, "run config") if isinstance(override, Path) else dict(override or {})
    tag = raw.get("format", RUN_CONFIG_FORMAT)
    if tag not in READABLE_RUN_CONFIG_FORMATS:
        raise ValueError(f"run config format {tag!r} is not one of {sorted(READABLE_RUN_CONFIG_FORMATS)}")
    _known("run config key", set(raw) - {"format"}, {"thresholds", "calibration"})
    given = raw.get("thresholds") or {}
    _known("threshold", set(given), set(merged))
    for name, bound in given.items():
        _known(f"key of threshold {name!r}", set(bound), {"rtol", "atol"})
        merged[name] |= bound
    calibration = raw.get("calibration") or {}
    _known("calibration key", set(calibration), set(RunCalibration.model_fields))
    return LoadedRunConfig(
        thresholds=Thresholds(
            by_name={n: Threshold(**{k: _number(v) for k, v in b.items()}) for n, b in merged.items()}
        ),
        calibration=RunCalibration.model_validate(calibration),
    )


def _number(v: object) -> object:
    return float(v) if isinstance(v, int) and not isinstance(v, bool) else v


def _known(what: str, given: set[str], known: set[str]) -> None:
    for name in sorted(given - known):
        closest = difflib.get_close_matches(name, sorted(known), n=1)
        raise UnknownThreshold(f"unknown {what} {name!r}" + (f"; did you mean {closest[0]!r}?" if closest else ""))


@functools.cache
def _packaged_thresholds() -> dict[str, dict[str, object]]:
    raw = load_yaml(resources.files("senselab_core.runtime").joinpath("data", "run_config.yaml"), "run_config.yaml")
    if raw.get("format") != RUN_CONFIG_FORMAT or not isinstance(raw.get("thresholds"), dict):
        raise ValueError(f"run_config.yaml must carry format {RUN_CONFIG_FORMAT!r} and 'thresholds'")
    out = {}
    for name, entry in raw["thresholds"].items():
        if not isinstance(entry, dict) or set(entry) != {"rtol", "atol"}:
            raise ValueError(f"{RUN_CONFIG_FORMAT} threshold {name!r} is exactly 'rtol' and 'atol'")
        out[str(name)] = {"rtol": entry["rtol"], "atol": entry["atol"]}
    return out


@dataclass(frozen=True)
class Rule:
    """How one field compares: `exact`, `close` within the threshold named `threshold`, or `optional`.

    `optional` marks the parent of an optional nested value (`span: Span | None`): absent it is one `[None]` leaf,
    present it is the leaves below it, so None against None agrees and None against a value differs at the parent.
    """

    kind: Literal["exact", "close", "optional"]
    threshold: str | None = None


@dataclass(frozen=True)
class TypeComparison:
    """How a kernel type's fields compare: `(pattern, rule)` pairs, first match wins, and its label columns."""

    fields: tuple[tuple[str, Rule], ...]
    match_labels: tuple[str, str, str] | None = None

    def rule(self, relative: str) -> Rule | None:
        """Return the rule of the first pattern matching the index-free `relative` path, or None."""
        free = index_free(relative)
        return next((r for p, r in self.fields if _pattern(p).fullmatch(free)), None)


def _pattern(text: str) -> re.Pattern[str]:
    return re.compile(".*".join(map(re.escape, text.split("*"))))


def _rule(where: str, raw: object) -> Rule:
    if raw == "exact":
        return Rule("exact")
    if raw == "optional":
        return Rule("optional")
    if isinstance(raw, Mapping) and set(raw) == {"close"} and isinstance(raw["close"], str) and raw["close"]:
        return Rule("close", raw["close"])
    raise ValueError(
        f"{COMPARISON_FORMAT} {where}: a rule is 'exact', 'optional' or {{close: <threshold name>}}, got {raw!r}"
    )


def parse_comparisons(text: str) -> Mapping[str, TypeComparison]:
    """Return the per-type comparisons a `comparison@1.0` document declares, keyed by `<name>@<major>`.

    Raises:
        ValueError: the document is not of that format and form.
    """
    raw = load_yaml(text, "comparison table")
    if raw.get("format") != COMPARISON_FORMAT or not isinstance(raw.get("types"), dict):
        raise ValueError(f"a comparison table is a mapping of format {COMPARISON_FORMAT!r} and 'types'")
    out = {}
    for name, entry in raw["types"].items():
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("fields"), dict)
            or set(entry) - {"fields", "match_labels"}
        ):
            raise ValueError(f"{COMPARISON_FORMAT} type {name}: a mapping of 'fields' and an optional 'match_labels'")
        labels = entry.get("match_labels")
        if labels is not None and (not isinstance(labels, dict) or set(labels) != {"label", "start", "end"}):
            raise ValueError(f"{COMPARISON_FORMAT} type {name}: match_labels names 'label', 'start' and 'end'")
        out[str(name)] = TypeComparison(
            fields=tuple((str(p), _rule(f"{name}.{p}", r)) for p, r in entry["fields"].items()),
            match_labels=None if labels is None else (labels["label"], labels["start"], labels["end"]),
        )
    return out


COMPARISONS: Mapping[str, TypeComparison] = parse_comparisons(
    resources.files("senselab_core.runtime").joinpath("data", "comparison.yaml").read_text("utf-8")
)
"""The packaged per-type comparisons, by `<name>@<major>`."""


class UndeclaredComparison(LookupError):
    """A kernel type, or a path of one, that declares no comparison; `tag` is its format tag, `path` the path if any."""

    def __init__(self, tag: str, path: str | None = None) -> None:
        """Name the format tag and the path."""
        what = (
            f"kernel type {tag!r} declares no comparison"
            if path is None
            else f"{tag!r} declares no comparison for {path!r}"
        )
        super().__init__(f"{what} in {COMPARISON_FORMAT}")
        self.tag = tag
        self.path = path


Comparisons = Mapping[str, TypeComparison]
"""Per-type comparisons by `<name>@<major>`, as `parse_comparisons` returns them."""
_KERNEL_TYPES = frozenset(t.split(".", 1)[0] for t in WIRE_FORMATS)


def _type_of(leaf: Value | None, comparisons: Comparisons) -> str | None:
    if not isinstance(leaf, str):
        return None
    try:
        tag = json.loads(leaf)
    except ValueError:
        return None
    if not isinstance(tag, str) or re.fullmatch(FORMAT_PATTERN, tag) is None:
        return None
    key = tag.split(".", 1)[0]
    if key in comparisons:
        return key
    if key in _KERNEL_TYPES:
        raise UndeclaredComparison(tag)
    return None


def typed_values(fields: Mapping[str, Value], comparisons: Comparisons | None = None) -> dict[str, str]:
    """Return the path prefixes holding a kernel value's wire form, each with its type's `<name>@<major>`.

    A kernel value is found by its `format` leaf under any path; the root's prefix is `""`. `comparisons` defaults to
    `COMPARISONS`.

    Raises:
        UndeclaredComparison: a kernel value's type has no entry in `comparisons`.
    """
    table = COMPARISONS if comparisons is None else comparisons
    out = {}
    for path, leaf in fields.items():
        prefix, _, name = path.rpartition(".")
        if name == "format" and (kind := _type_of(leaf, table)) is not None:
            out[prefix] = kind
    return out


def _owner(path: str, prefixes: Sequence[str]) -> str | None:
    """Return the longest prefix in `prefixes` that `path` lies under."""
    return next((p for p in prefixes if p == "" or path.startswith(f"{p}.")), None)


def best_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return the one-to-one (row, column) pairs of largest total weight, as many as the smaller side.

    With a tie it returns the optimum whose sorted `(row, column)` sequence is lexically smallest.
    """
    n, m = len(weights), len(weights[0]) if weights else 0
    wanted = min(n, m)
    best = _total(weights, _hungarian(weights))
    chosen: list[tuple[int, int]] = []
    for i in range(n):
        if len(chosen) == wanted:
            break
        used = {c for _, c in chosen}
        free = [j for j in range(m) if j not in used]
        for j in free:
            rest_rows = list(range(i + 1, n))
            rest_cols = [c for c in free if c != j]
            if min(len(rest_rows), len(rest_cols)) != wanted - len(chosen) - 1:
                continue
            sub = [[weights[r][c] for c in rest_cols] for r in rest_rows]
            total = _total(weights, chosen) + weights[i][j] + _total(sub, _hungarian(sub))
            if math.isclose(total, best, rel_tol=1e-12, abs_tol=1e-12):
                chosen.append((i, j))
                break
    if len(chosen) != wanted:
        raise ValueError(f"no optimal assignment of {wanted} pairs found in the weights")
    return chosen


def _total(weights: Sequence[Sequence[float]], pairs: Sequence[tuple[int, int]]) -> float:
    return math.fsum(weights[i][j] for i, j in pairs)


def _hungarian(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return one optimal assignment of `weights` (Hungarian method), as many pairs as the smaller side."""
    n, m = len(weights), len(weights[0]) if weights else 0
    size = max(n, m)
    top = max((w for row in weights for w in row), default=0.0)
    cost = [[top - (weights[i][j] if i < n and j < m else 0.0) for j in range(size)] for i in range(size)]
    u, v, match, way = [0.0] * (size + 1), [0.0] * (size + 1), [0] * (size + 1), [0] * (size + 1)
    for i in range(1, size + 1):
        match[0], column = i, 0
        least, used = [float("inf")] * (size + 1), [False] * (size + 1)
        while match[column]:
            used[column], row, delta, nxt = True, match[column], float("inf"), 0
            for j in range(1, size + 1):
                if not used[j]:
                    reduced = cost[row - 1][j - 1] - u[row] - v[j]
                    if reduced < least[j]:
                        least[j], way[j] = reduced, column
                    if least[j] < delta:
                        delta, nxt = least[j], j
            for j in range(size + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    least[j] -= delta
            column = nxt
        while column:
            previous = way[column]
            match[column] = match[previous]
            column = previous
    return sorted((match[j] - 1, j - 1) for j in range(1, size + 1) if match[j] - 1 < n and j - 1 < m)


def _relabel_one(
    expected: Mapping[str, Value], actual: Mapping[str, Value], prefix: str, columns: tuple[str, str, str]
) -> Fields:
    base = f"{prefix}." if prefix else ""
    label = re.compile(re.escape(f"{base}{columns[0]}") + r"\[(\d+)\]")
    times = (f"{base}{columns[1]}", f"{base}{columns[2]}")

    def keys(fields: Mapping[str, object]) -> list[str]:
        found = [(int(m.group(1)), k) for k in fields if (m := label.fullmatch(k))]
        return [k for _, k in sorted(found)]

    ek, ak = keys(expected), keys(actual)
    if not ek or not ak or any(t not in expected or t not in actual for t in times):
        return dict(actual)
    elabel = [json.loads(str(expected[k])) for k in ek]
    alabel = [json.loads(str(actual[k])) for k in ak]
    rows, cols = sorted(set(elabel)), sorted(set(alabel))
    overlap = [[0.0] * len(cols) for _ in rows]
    spans = [[list(f[t]) for t in times] for f in (expected, actual)]
    if any(len(s) != len(k) for s, k in ((spans[0][0], ek), (spans[1][0], ak))):
        return dict(actual)
    for i, (es, ee) in enumerate(zip(*spans[0], strict=True)):
        for j, (a_s, a_e) in enumerate(zip(*spans[1], strict=True)):
            if None not in (es, ee, a_s, a_e):
                overlap[rows.index(elabel[i])][cols.index(alabel[j])] += max(0.0, min(ee, a_e) - max(es, a_s))
    mapped = {cols[c]: rows[r] for r, c in best_assignment(overlap) if overlap[r][c] > 0}
    out = dict(actual)
    for k, name in zip(ak, alabel, strict=True):
        out[k] = json.dumps(mapped.get(name, f"<unmatched:{name}>"))
    return out


def relabel(
    expected: Mapping[str, Value], actual: Mapping[str, Value], comparisons: Comparisons | None = None
) -> Fields:
    """Return `actual` with the labels of each kernel value that declares `match_labels` renamed to `expected`'s.

    Labels are arbitrary per run, so they are mapped one to one by most shared time (the map DER scores under); a
    label with no counterpart (no shared time) is renamed to one no expected label equals. `actual` is unchanged
    where either side has no labelled spans.
    """
    table = COMPARISONS if comparisons is None else comparisons
    out = dict(actual)
    typed = typed_values(expected, table) | typed_values(actual, table)
    for prefix, kind in sorted(typed.items()):
        columns = table[kind].match_labels
        if columns is not None:
            out = _relabel_one(expected, out, prefix, columns)
    return out


class Difference(Frozen):
    """One leaf that differs: its path, the rule and threshold that judged it, both sides and how far apart they are.

    `expected` and `actual` are the values (a numeric leaf of more than `SUMMARY_VALUES` as its length and range, a
    text leaf cut at `SUMMARY_CHARS`). `absolute` and `relative` are the largest `|actual - expected|` and
    `|actual - expected| / |expected|` over a numeric leaf's paired values, None where undefined.
    """

    path: str
    rule: Literal["exact", "close", "optional", "missing"]
    threshold: str | None
    bound: Threshold | None
    expected: str
    actual: str
    absolute: FiniteFloat | None
    relative: FiniteFloat | None
    detail: str


@dataclass(frozen=True)
class Comparison:
    """Whether two outputs agree, the leaves that differ, and the digest of the thresholds that judged them."""

    agree: bool
    differences: tuple[Difference, ...]
    thresholds: str

    @property
    def paths(self) -> tuple[str, ...]:
        """The paths that differ."""
        return tuple(d.path for d in self.differences)


SUMMARY_VALUES = 4
"""A numeric leaf of at most this many values is summarised as its values, a longer one as its length."""
SUMMARY_CHARS = 80
"""A text leaf is summarised as its first this many characters."""


def _summary(leaf: Value | None) -> str:
    if leaf is None:
        return "absent"
    if isinstance(leaf, str):
        return leaf if len(leaf) <= SUMMARY_CHARS else f"{leaf[:SUMMARY_CHARS]}..."
    if len(leaf) <= SUMMARY_VALUES:
        return repr(leaf)
    present = [x for x in leaf if x is not None]
    return f"{len(leaf)} values in [{min(present):g}, {max(present):g}]" if present else f"{len(leaf)} absent values"


def _gap(a: Value, b: Value) -> tuple[float | None, float | None]:
    """Return the largest absolute and relative difference over the paired numbers of `a` and `b`, None if undefined."""
    if isinstance(a, str) or isinstance(b, str) or len(a) != len(b):
        return None, None
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None, None
    want, got = (numpy.asarray(side, dtype=float) for side in zip(*pairs, strict=True))
    with numpy.errstate(invalid="ignore"):
        gap = numpy.abs(got - want)
        nonzero = want != 0
        absolute = float(gap.max())
        relative = float((gap[nonzero] / numpy.abs(want[nonzero])).max()) if nonzero.any() else None
    return tuple(x if x is not None and math.isfinite(x) else None for x in (absolute, relative))  # type: ignore[return-value]


def _exact(a: Value, b: Value) -> str | None:
    if a == b:
        return None
    if isinstance(a, str) or isinstance(b, str):
        return "not equal"
    if len(a) != len(b):
        return f"{len(a)} values against {len(b)}"
    unequal = [n for n, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    return f"{len(unequal)} of {len(a)} values not equal, first at [{unequal[0]}]"


def _close(a: Value, b: Value, bound: Threshold) -> str | None:
    if isinstance(a, str) or isinstance(b, str):
        return None if a == b else "not equal"
    if len(a) != len(b):
        return f"{len(a)} values against {len(b)}"
    if any((x is None) != (y is None) for x, y in zip(a, b, strict=True)):
        return "an absent value against a number"
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    want, got = (numpy.asarray(side, dtype=float) for side in zip(*pairs, strict=True))
    close = numpy.isclose(got, want, rtol=bound.rtol, atol=bound.atol)
    if close.all():
        return None
    gap = numpy.nan_to_num(numpy.abs(got - want), nan=numpy.inf)
    largest = float(numpy.where(close, 0.0, gap).max())
    return (
        f"{int((~close).sum())} of {len(pairs)} values beyond rtol {bound.rtol:g} atol {bound.atol:g},"
        f" largest |difference| {largest:g}"
    )


def _judge(path: str, a: Value | None, b: Value | None, rule: Rule, thresholds: Thresholds) -> Difference | None:
    if a is None or b is None:
        return Difference(
            path=path,
            rule="missing",
            threshold=None,
            bound=None,
            expected=_summary(a),
            actual=_summary(b),
            absolute=None,
            relative=None,
            detail="present on one side only",
        )
    bound = thresholds.by_name[rule.threshold] if rule.threshold is not None else None
    why = _close(a, b, bound) if bound is not None else _exact(a, b)
    if why is None:
        return None
    absolute, relative = _gap(a, b)
    return Difference(
        path=path,
        rule=rule.kind,
        threshold=rule.threshold,
        bound=bound,
        expected=_summary(a),
        actual=_summary(b),
        absolute=absolute,
        relative=relative,
        detail=why,
    )


def _optional_gaps(
    expected: Mapping[str, Value], actual: Mapping[str, Value], rules: Mapping[str, Rule]
) -> dict[str, Difference]:
    """Return a difference per `optional` parent that is a `[None]` leaf on one side and nested leaves on the other."""
    out = {}
    for path, rule in rules.items():
        if rule.kind != "optional" or (path in expected) == (path in actual):
            continue
        absent, present = (expected, actual) if path in expected else (actual, expected)
        if absent[path] != [None] or not any(k.startswith(f"{path}.") for k in present):
            continue
        out[path] = Difference(
            path=path,
            rule="optional",
            threshold=None,
            bound=None,
            expected="absent" if absent is expected else "present",
            actual="absent" if absent is actual else "present",
            absolute=None,
            relative=None,
            detail="absent on one side, present on the other",
        )
    return out


def compare_fields(
    expected: Mapping[str, Value],
    actual: Mapping[str, Value],
    thresholds: Thresholds,
    comparisons: Comparisons | None = None,
) -> Comparison:
    """Return whether flattened outputs agree, leaf by leaf under each kernel type's declared comparison.

    A leaf under a kernel value (found by its `format` leaf) is judged by its type's first matching rule in
    `comparisons` (default `COMPARISONS`); any other leaf is exact. Labels a type matches are mapped first
    (`relabel`). Equal leaves agree before any rule is looked up. A leaf on one side only is a difference; an
    `optional` parent absent on one side and present on the other is one difference at the parent.

    Raises:
        UndeclaredComparison: a kernel value's type has no entry in `comparisons`, or a leaf that differs lies under
            a type that declares no rule for its path.
        KeyError: a rule names a threshold `thresholds` lacks.
    """
    table = COMPARISONS if comparisons is None else comparisons
    digest = thresholds.canonical_digest()
    typed = typed_values(expected, table) | typed_values(actual, table)
    if expected == actual:
        return Comparison(agree=True, differences=(), thresholds=digest)
    actual = relabel(expected, actual, table)
    prefixes = sorted(typed, key=len, reverse=True)
    paths = sorted(expected.keys() | actual.keys())
    owners = {p: _owner(p, prefixes) for p in paths}
    rules: dict[str, Rule] = {}
    for path in paths:
        owner = owners[path]
        found_rule = (
            Rule("exact") if owner is None else table[typed[owner]].rule(path[len(owner) + 1 :] if owner else path)
        )
        if found_rule is None:
            if expected.get(path) == actual.get(path):
                continue
            raise UndeclaredComparison(typed[owner or ""], path)
        rules[path] = found_rule
    gaps = _optional_gaps(expected, actual, rules)
    found = list(gaps.values())
    for path in paths:
        if path not in rules or path in gaps or any(path.startswith(f"{p}.") for p in gaps):
            continue
        a, b = expected.get(path), actual.get(path)
        if a is not None and a == b:
            continue
        if (difference := _judge(path, a, b, rules[path], thresholds)) is not None:
            found.append(difference)
    found.sort(key=lambda d: d.path)
    return Comparison(agree=not found, differences=tuple(found), thresholds=digest)


def compare(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    thresholds: Thresholds,
    comparisons: Comparisons | None = None,
) -> Comparison:
    """Return whether two outputs agree: equal hashes agree at once; otherwise `compare_fields` of their leaves.

    Every kernel value in either output must have a comparison in `comparisons`, equal hashes or not.

    Raises:
        NotWireForm: either output holds a value that is not a wire form (`flatten`).
        UndeclaredComparison: a kernel value's type has no entry in `comparisons` (default `COMPARISONS`).
        KeyError: a rule names a threshold `thresholds` lacks.
    """
    leaves = flatten(expected), flatten(actual)
    for side in leaves:
        typed_values(side, comparisons)
    try:
        if value_digest("output", dict(expected)) == value_digest("output", dict(actual)):
            return Comparison(agree=True, differences=(), thresholds=thresholds.canonical_digest())
    except ValueError:
        pass
    return compare_fields(*leaves, thresholds, comparisons)
