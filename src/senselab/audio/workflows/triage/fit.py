"""Fit a dataset profile's corpus statistics from a pilot run's corpus table (plan step 10).

A corpus statistic (kind ``b`` in ``data/profile_values.yaml``) is a value computed from the dataset's own
recordings: a fence around a distribution, a percentile, a median. Each one's entry carries a ``method``: which
corpus-table column it reads, one value per recording, session or participant, which recordings, which statistic,
and how it is rounded. :func:`fit_values` computes every one from the table ``recording_vectors`` writes;
``scripts/fit_profile.py`` writes them into the profile's ``fitted.yaml``
(:data:`~senselab.audio.workflows.triage.dataset_profile.FITTED_FILE`).

The order for a new dataset: a pilot run (a profile with ``pilot: true`` may leave these values empty), the COHORT
pass, the corpus table, the fit, a replay of the pilot so its branches read the fitted values, the COHORT pass again
over the replayed runs, then a second replay so its verdicts fold the new COHORT readings (``scripts/fit_profile.py``
lists the steps).

Each statistic is fitted over at least its ``n_min`` values (:func:`n_min_of`); a value with fewer is not fitted
unless the caller allows it, and its :attr:`Fit.short` names it either way.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.dataset_profile import (
    CONFIG_FILE,
    FITTED_KIND,
    PACKAGED_DATA,
    PROFILE_VALUES_FILE,
    profile_values,
)

METHOD_KEY = "method"
"""The ``data/profile_values.yaml`` entry key holding a corpus statistic's method."""

TUKEY_K = 1.5
"""The Tukey fence's multiple of the interquartile range."""

STATISTICS = ("median", "percentile", "percentile_band", "tukey_upper", "tukey_lower")
"""The statistics a method may name."""

N_MIN_KEY = "n_min"
"""The key of a method's minimum count, and of the per-statistic defaults at the top of ``data/profile_values.yaml``."""

WHOLE = ""
"""How :attr:`Fit.short` names a value without ``by`` that has too few values."""

PER = {"recording": (), "session": ("participant", "session"), "participant": ("participant",)}
"""The columns a method's ``per`` groups rows by: one value per group (COHORT repeats a group's reading on each of
its recordings). COHORT groups by target speaker, so a row that declares one (:data:`SPEAKER_COLUMN`) is grouped
under it in place of its participant (:func:`group_of`)."""

SPEAKER_COLUMN = "target_speaker"
"""The corpus-table column naming a recording's declared target speaker; null where the target is the participant."""


def group_of(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, ...]:
    """The group a ``per`` method files a row under: :data:`PER`'s columns, with the speaker COHORT read.

    Args:
        row: A corpus-table row.
        keys: The method's :data:`PER` columns.

    Returns:
        One value per key; ``participant`` is the declared target speaker where the row names one, so recordings of
        two target speakers of one participant are two groups, as COHORT reads them (plan step 9a). A session is
        then its speaker's session: ``(speaker, session)``, with the participant kept beside it.
    """
    speaker = row.get(SPEAKER_COLUMN)
    if not _present(speaker):
        return tuple(row.get(key) for key in keys)
    group = tuple(speaker if key == "participant" else row.get(key) for key in keys)
    return (*group, row.get("participant")) if "session" in keys else group


OPERATORS: Mapping[str, Callable[[Any, Any], bool]] = {
    "eq": lambda cell, value: cell == value,
    "ne": lambda cell, value: cell != value,
    "gt": lambda cell, value: cell > value,
    "ge": lambda cell, value: cell >= value,
    "lt": lambda cell, value: cell < value,
    "le": lambda cell, value: cell <= value,
}
"""The comparisons a method's ``where`` filter may use."""

Row = Mapping[str, Any]
"""One corpus-table row, as ``pyarrow.Table.to_pylist`` gives it."""


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
        n_min: The fewest values it may be fitted over (:func:`n_min_of`).
        short: What had some values but fewer than ``n_min``: the families of a value by family, else
            ``(WHOLE,)``; empty when nothing did. Unless the fit allowed it, these are left out of ``value``.
    """

    file: str
    path: tuple[str, ...]
    value: Any
    n: int | dict[str, int]
    method: Mapping[str, Any]
    n_min: int = 0
    short: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """``<file> <dotted key path>``."""
        return f"{self.file} {'.'.join(self.path)}"


@functools.cache
def n_min_defaults() -> dict[str, int]:
    """Each statistic's default minimum count, from the top of ``data/profile_values.yaml``.

    Returns:
        ``{statistic: count}``.

    Raises:
        ValueError: If a statistic has no default, or one that is not a positive whole number.
    """
    document = yaml.safe_load((PACKAGED_DATA / PROFILE_VALUES_FILE).read_text()) or {}
    defaults = dict(document.get(N_MIN_KEY) or {})
    for name in STATISTICS:
        if not isinstance(defaults.get(name), int) or isinstance(defaults[name], bool) or defaults[name] < 1:
            raise ValueError(f"{PROFILE_VALUES_FILE} {N_MIN_KEY}.{name} is {defaults.get(name)!r}, not a count >= 1")
    return {name: int(defaults[name]) for name in STATISTICS}


def n_min_of(method: Mapping[str, Any]) -> int:
    """The fewest values a method's statistic may be fitted over: its own ``n_min``, else its statistic's default.

    Args:
        method: The method.

    Returns:
        The count.

    Raises:
        ValueError: If the method's own ``n_min`` is not a positive whole number.
    """
    own = method.get(N_MIN_KEY)
    if own is None:
        return n_min_defaults()[str(method["statistic"])]
    if not isinstance(own, int) or isinstance(own, bool) or own < 1:
        raise ValueError(f"method {dict(method)}: {N_MIN_KEY} {own!r} is not a count >= 1")
    return own


def fit_methods() -> list[tuple[str, tuple[str, ...], dict[str, Any]]]:
    """Every corpus statistic with a method: ``(data file, key path, method)``.

    Returns:
        The kind-``b`` entries of ``data/profile_values.yaml`` that carry a ``method``, in the file's order.

    Raises:
        ValueError: If a method names a statistic or ``per`` this module does not compute, or an ``n_min`` that is
            not a count.
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
            n_min_of(method)
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
            group = group_of(row, keys)
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
    allow_short: bool = False,
) -> Fit:
    """One corpus statistic, fitted.

    Args:
        rows: The corpus table's rows.
        relative: The data file the value belongs in.
        path: Its key path.
        method: Its method.
        tasks: The task list's ``tasks`` mapping.
        setting: Reads another value of ``relative`` by key path.
        allow_short: Fit a value (or a family) over fewer than its ``n_min`` values too; it is named in
            :attr:`Fit.short` either way.

    Returns:
        The fit. No values at all leave the value (or family) unfitted, as before; between one and ``n_min - 1``
        leave it unfitted unless ``allow_short``.
    """
    n_min = n_min_of(method)
    chosen = select(rows, method, tasks=tasks, setting=setting)
    if method.get("by"):
        by = str(method["by"])
        groups: dict[str, list[Row]] = {}
        for row in chosen:
            groups.setdefault(str(row.get(by)), []).append(row)
        value: dict[str, Any] = {}
        counts: dict[str, int] = {}
        short: list[str] = []
        for group in sorted(groups):
            values = _values(groups[group], method)
            if not values.size:
                continue
            counts[group] = int(values.size)
            if values.size < n_min:
                short.append(group)
                if not allow_short:
                    continue
            value[group] = statistic(values, method)
        return Fit(relative, path, value or None, counts, method, n_min, tuple(short))
    values = _values(chosen, method)
    too_few = 0 < values.size < n_min
    fitted = None if too_few and not allow_short else statistic(values, method)
    return Fit(relative, path, fitted, int(values.size), method, n_min, (WHOLE,) if too_few else ())


def fit_values(
    rows: Sequence[Row],
    *,
    tasks: Mapping[str, Any],
    setting: Callable[[str, Sequence[str]], Any],
    allow_short: bool = False,
) -> list[Fit]:
    """Every corpus statistic, fitted from a corpus table.

    Args:
        rows: The corpus table's rows (``recording_vectors.parquet``, schema 32 on; :func:`columns_read` names
            the columns needed).
        tasks: The task list's ``tasks`` mapping.
        setting: Reads a value of the active profile by data file and key path.
        allow_short: Fit values over fewer than their ``n_min`` values too (:func:`fit_value`).

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
            allow_short=allow_short,
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
    names = {"participant", "session", SPEAKER_COLUMN, "declared_family"}
    for _, _, method in fit_methods():
        names.add(str(method["column"]))
        names.update(str(condition["column"]) for condition in method.get("where") or ())
        if method.get("by"):
            names.add(str(method["by"]))
    return sorted(names)
