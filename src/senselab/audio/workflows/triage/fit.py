"""Fit a dataset profile's corpus statistics from a pilot run's corpus table (plan step 10).

A corpus statistic (kind ``b`` in ``data/profile_values.yaml``) is a value computed from the dataset's own
recordings: a fence around a distribution, a percentile, a median. Each one's entry carries a ``method``: which
corpus-table column it reads, one value per recording, session or participant, which recordings, which statistic,
and how it is rounded. :func:`fit_values` computes every one from the table ``recording_vectors`` writes;
``scripts/fit_profile.py`` writes them into the profile's ``fitted.yaml``
(:data:`~senselab.audio.workflows.triage.dataset_profile.FITTED_FILE`).

The order for a new dataset: a pilot run (a profile with ``pilot: true`` may leave these values empty), the COHORT
pass, the corpus table, the fit, then a replay of the pilot so its verdicts read the fitted values.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from senselab.audio.workflows.triage.dataset_profile import CONFIG_FILE, FITTED_KIND, profile_values

METHOD_KEY = "method"
"""The ``data/profile_values.yaml`` entry key holding a corpus statistic's method."""

TUKEY_K = 1.5
"""The Tukey fence's multiple of the interquartile range."""

STATISTICS = ("median", "percentile", "percentile_band", "tukey_upper", "tukey_lower")
"""The statistics a method may name."""

PER = {"recording": (), "session": ("participant", "session"), "participant": ("participant",)}
"""The columns a method's ``per`` groups rows by: one value per group (COHORT repeats a group's reading on each of
its recordings)."""

OPERATORS: Mapping[str, Callable[[Any, Any], bool]] = {
    "eq": lambda cell, value: cell == value,
    "ne": lambda cell, value: cell != value,
    "gt": lambda cell, value: cell > value,
    "ge": lambda cell, value: cell >= value,
    "lt": lambda cell, value: cell < value,
    "le": lambda cell, value: cell <= value,
}

Row = Mapping[str, Any]
"""One corpus-table row, as ``pyarrow.Table.to_pylist`` gives it."""
"""The comparisons a method's ``where`` filter may use."""


@dataclass(frozen=True)
class Fit:
    """One fitted value.

    Attributes:
        file: The data file the value belongs in, e.g. ``cohort_stage.yaml``.
        path: Its key path in that file.
        value: The fitted value (a number, a ``[low, high]`` pair, or ``{family: number}``), or None where the
            table held nothing to fit it from.
        n: How many values it was computed over (per family, for a value by family).
        method: The method it followed.
    """

    file: str
    path: tuple[str, ...]
    value: Any
    n: int | dict[str, int]
    method: Mapping[str, Any]

    @property
    def key(self) -> str:
        """``<file> <dotted key path>``."""
        return f"{self.file} {'.'.join(self.path)}"


def fit_methods() -> list[tuple[str, tuple[str, ...], dict[str, Any]]]:
    """Every corpus statistic with a method: ``(data file, key path, method)``.

    Returns:
        The kind-``b`` entries of ``data/profile_values.yaml`` that carry a ``method``, in the file's order.

    Raises:
        ValueError: If a method names a statistic or ``per`` this module does not compute.
    """
    out = []
    for relative, entries in profile_values().items():
        for entry in entries:
            method = entry.get(METHOD_KEY)
            if entry.get("kind") != FITTED_KIND or not isinstance(method, Mapping):
                continue
            path = tuple(str(part) for part in entry["path"])
            if method.get("statistic") not in STATISTICS or method.get("per", "recording") not in PER:
                raise ValueError(f"{relative} {'.'.join(path)}: method {dict(method)} is not one fit.py computes")
            out.append((relative, path, dict(method)))
    return out


def task_families(tasks: Mapping[str, Any], expects: Mapping[str, Any]) -> set[str]:
    """The task-list families whose ``expects`` holds every key and value of ``expects``.

    Args:
        tasks: The task list's ``tasks`` mapping.
        expects: E.g. ``{"pattern": "free_response"}``.

    Returns:
        The matching family names.
    """
    return {
        name
        for name, entry in tasks.items()
        if isinstance(entry, Mapping)
        and all((entry.get("expects") or {}).get(key) == value for key, value in expects.items())
    }


def _present(value: Any) -> bool:  # noqa: ANN401 -- a table cell
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def select(
    rows: Sequence[Row],
    method: Mapping[str, Any],
    *,
    tasks: Mapping[str, Any],
    setting: Callable[[Sequence[str]], Any],
) -> list[Row]:
    """The rows a method computes over: filtered, one per group, with the column present.

    Args:
        rows: The corpus table's rows.
        method: The value's method.
        tasks: The task list's ``tasks`` mapping, for a ``tasks`` filter.
        setting: Reads another value of the same data file by key path, for a ``where`` filter's ``setting``.

    Returns:
        The rows, in table order.

    Raises:
        ValueError: If a ``where`` filter's setting is unset in the profile.
    """
    column = str(method["column"])
    chosen = [row for row in rows if _present(row.get(column))]
    for condition in method.get("where") or ():
        if not chosen:
            break
        bound = setting(condition["setting"]) if "setting" in condition else condition["value"]
        if bound is None:
            raise ValueError(f"the where filter's setting {condition['setting']} is unset in the profile")
        name, compare = str(condition["column"]), OPERATORS[str(condition["op"])]
        chosen = [row for row in chosen if _present(row.get(name)) and compare(row[name], bound)]
    if method.get("tasks"):
        families = task_families(tasks, method["tasks"])
        chosen = [row for row in chosen if row.get("declared_family") in families]
    keys = PER[str(method.get("per", "recording"))]
    if keys:
        seen: set[tuple[Any, ...]] = set()
        grouped = []
        for row in chosen:
            group = tuple(row.get(key) for key in keys)
            if any(not _present(part) for part in group) or group in seen:
                continue
            seen.add(group)
            grouped.append(row)
        chosen = grouped
    return chosen


def statistic(values: np.ndarray, method: Mapping[str, Any]) -> Any:  # noqa: ANN401 -- a number or a pair
    """A method's statistic over its values, clamped and rounded.

    Args:
        values: The values (finite).
        method: The method.

    Returns:
        A number, or ``[low, high]`` for ``percentile_band``; None where there are no values.
    """
    if values.size == 0:
        return None
    name = method["statistic"]
    decimals = int(method.get("decimals", 3))
    if name == "percentile_band":
        low, high = (float(np.percentile(values, float(q))) for q in method["q"])
        return [round(low, decimals), round(high, decimals)]
    if name == "median":
        value = float(np.median(values))
    elif name == "percentile":
        value = float(np.percentile(values, float(method["q"])))
    else:
        q1, q3 = (float(x) for x in np.quantile(values, [0.25, 0.75]))
        spread = TUKEY_K * (q3 - q1)
        value = q3 + spread if name == "tukey_upper" else q1 - spread
    clamp = method.get("clamp")
    if clamp is not None:
        value = min(max(value, float(clamp[0])), float(clamp[1]))
    return round(value, decimals)


def _values(rows: Sequence[Row], method: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray([float(row[str(method["column"])]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return np.abs(values) if method.get("absolute") else values


def fit_value(
    rows: Sequence[Row],
    relative: str,
    path: tuple[str, ...],
    method: Mapping[str, Any],
    *,
    tasks: Mapping[str, Any],
    setting: Callable[[Sequence[str]], Any],
) -> Fit:
    """One corpus statistic, fitted.

    Args:
        rows: The corpus table's rows.
        relative: The data file the value belongs in.
        path: Its key path.
        method: Its method.
        tasks: The task list's ``tasks`` mapping.
        setting: Reads another value of ``relative`` by key path.

    Returns:
        The fit.
    """
    chosen = select(rows, method, tasks=tasks, setting=setting)
    if method.get("by"):
        by = str(method["by"])
        groups: dict[str, list[Row]] = {}
        for row in chosen:
            groups.setdefault(str(row.get(by)), []).append(row)
        value: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for group in sorted(groups):
            values = _values(groups[group], method)
            if values.size:
                value[group], counts[group] = statistic(values, method), int(values.size)
        return Fit(relative, path, value or None, counts, method)
    values = _values(chosen, method)
    return Fit(relative, path, statistic(values, method), int(values.size), method)


def fit_values(
    rows: Sequence[Row],
    *,
    tasks: Mapping[str, Any],
    setting: Callable[[str, Sequence[str]], Any],
) -> list[Fit]:
    """Every corpus statistic, fitted from a corpus table.

    Args:
        rows: The corpus table's rows (``recording_vectors.parquet``, schema 32 on; :func:`columns_read` names
            the columns needed).
        tasks: The task list's ``tasks`` mapping.
        setting: Reads a value of the active profile by data file and key path.

    Returns:
        One fit per method, in ``data/profile_values.yaml``'s order.
    """
    return [
        fit_value(
            rows,
            relative,
            path,
            method,
            tasks=tasks,
            setting=functools.partial(setting, relative),
        )
        for relative, path, method in fit_methods()
    ]


def fitted_values(fits: Sequence[Fit]) -> dict[str, dict[str, Any]]:
    """The fitted values as a profile's ``fitted.yaml`` holds them: nested by data file and key path.

    Args:
        fits: The fits.

    Returns:
        ``{data file: {key: ... value}}``; a value the table could not fit is left out.
    """
    out: dict[str, dict[str, Any]] = {}
    for fit in fits:
        if fit.value is None or fit.file == CONFIG_FILE:
            continue
        node = out.setdefault(fit.file, {})
        for part in fit.path[:-1]:
            node = node.setdefault(part, {})
        node[fit.path[-1]] = fit.value
    return out


def columns_read() -> list[str]:
    """Every corpus-table column a fit reads, for a reader that loads only those.

    Returns:
        The columns, sorted.
    """
    names = {"participant", "session", "declared_family"}
    for _, _, method in fit_methods():
        names.add(str(method["column"]))
        names.update(str(condition["column"]) for condition in method.get("where") or ())
        if method.get("by"):
            names.add(str(method["by"]))
    return sorted(names)
