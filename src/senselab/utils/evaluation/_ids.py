"""Id validation shared by the evaluation modules (subject, unit and group ids)."""

from __future__ import annotations

import math
import sys
from typing import Any, Sequence, Union

import numpy as np

MISSING_IDS = "None, NaN, NaT, pandas.NA or empty"


def is_missing(value: object, pandas_na: object, pandas_nat: object) -> bool:
    """Whether one id is missing: None, a float NaN of any width, pandas.NA or NaT, or the empty string."""
    if (
        value is None
        or (pandas_na is not None and value is pandas_na)
        or (pandas_nat is not None and value is pandas_nat)
    ):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(value)
    return isinstance(value, str) and value == ""


def key_array(values: Union[np.ndarray, Sequence[Any]], what: str) -> np.ndarray:
    """The ids as an array, refusing missing ones before coercion can turn a NaN into the string "nan".

    `pandas.NA` and `pandas.NaT` are recognised without importing pandas: an input can only hold them
    if the caller has already imported pandas, so the check looks the module up in `sys.modules`.

    Args:
        values: One id per row.
        what: The id's name for the error message (`subject`, `unit`, `group`).

    Returns:
        The ids as an array.

    Raises:
        ValueError: If any id is missing.
    """
    raw = values.tolist() if isinstance(values, np.ndarray) else list(values)
    pandas = sys.modules.get("pandas")
    na, nat = (getattr(pandas, "NA", None), getattr(pandas, "NaT", None)) if pandas is not None else (None, None)
    missing = sum(is_missing(v, na, nat) for v in raw)
    if missing:
        raise ValueError(f"{missing} rows have no {what} id ({MISSING_IDS}); every row needs one")
    return values if isinstance(values, np.ndarray) else np.asarray(raw)


def distinct_keys(ids: np.ndarray) -> int:
    """Distinct ids compared as strings, so 1 and "1" are one id."""
    return len({str(v) for v in ids.tolist()})
