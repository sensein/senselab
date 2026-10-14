"""Categorical extraction: check the values an adapter emits against the step's declared allow-lists.

A step that declares ``extract`` (`spec.ExtractSpec`) has an adapter that returns
``{"metrics": {...}, "extracted": {...}}``. `validate_extracted` checks the ``extracted`` object: every
declared name is present and no other; each value has its kind's shape and lies in its allow-list. A
failure is the single code ``E_EXTRACT``; no value is echoed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypeGuard

from senselab.utils.reproduce.spec import ExtractSpec

CODE = "E_EXTRACT"
YESNO = ("yes", "no")


def _integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid(value: object, spec: ExtractSpec) -> bool:
    if spec.kind == "model_id":
        return isinstance(value, str) and value in spec.allowed
    if spec.kind == "yesno":
        return isinstance(value, str) and value in YESNO
    if spec.kind == "index":
        return _integer(value) and 0 <= value < len(spec.allowed)
    if spec.kind == "series":
        assert spec.max_len is not None
        return isinstance(value, list) and len(value) <= spec.max_len and all(_number(v) for v in value)
    return _integer(value) and value >= 0


def validate_extracted(
    obj: object, specs: Sequence[ExtractSpec], *, min_cell: int
) -> tuple[dict[str, Any] | None, list[str]]:
    """Check an adapter's ``extracted`` object and suppress small counts.

    Args:
        obj: The ``extracted`` value the adapter returned.
        specs: The step's declarations.
        min_cell: The profile policy's minimum cell size; a ``count`` below it becomes ``"<min_cell>"``.

    Returns:
        ``(clean, codes)``: the checked copy and ``[]``, or ``None`` and ``["E_EXTRACT"]``.
    """
    declared = {s.name: s for s in specs}
    if not isinstance(obj, dict) or set(obj) != set(declared):
        return None, [CODE]
    if not all(_valid(obj[name], spec) for name, spec in declared.items()):
        return None, [CODE]
    clean: dict[str, Any] = {}
    for name, spec in declared.items():
        value = obj[name]
        if spec.kind == "count" and value < min_cell:
            value = f"<{min_cell}"
        clean[name] = list(value) if spec.kind == "series" else value
    return clean, []
