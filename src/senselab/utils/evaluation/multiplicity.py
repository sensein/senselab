"""Multiplicity adjustment across a family of tests: Holm-Bonferroni.

Sort the p-values ascending. The adjusted p-value of the test at rank `i` (1-based) among `m` is
`min(1, max over ranks j <= i of (m - j + 1) * p_j)`, so adjusted values never decrease with rank. A
test is rejected when its adjusted p-value is at most `alpha`. Ties keep the input order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class HolmResult:
    """One test's Holm outcome.

    Attributes:
        p_value: The raw p-value.
        adjusted_p: The Holm-adjusted p-value, at most 1.
        reject: Whether the adjusted p-value is at most `alpha`.
        rank: Position by ascending raw p-value, starting at 1.
    """

    p_value: float
    adjusted_p: float
    reject: bool
    rank: int


def holm(p_values: Mapping[str, float], alpha: float) -> Dict[str, HolmResult]:
    """Holm-Bonferroni adjustment of a family of p-values.

    Args:
        p_values: Test name to raw p-value in [0, 1]; at least one.
        alpha: Family-wise level in (0, 1).

    Returns:
        Test name to its result, in ascending rank order.

    Raises:
        ValueError: If the family is empty, a p-value is outside [0, 1] or not a number, or `alpha` is outside (0, 1).
    """
    if not p_values:
        raise ValueError("holm needs at least one p-value")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if any(not 0 <= p <= 1 for p in p_values.values()):  # NaN fails the comparison too
        raise ValueError("p-values must be in [0, 1]")
    m = len(p_values)
    result: Dict[str, HolmResult] = {}
    running = 0.0
    for rank, name in enumerate(sorted(p_values, key=lambda k: p_values[k]), start=1):
        running = max(running, (m - rank + 1) * p_values[name])
        adjusted = min(1.0, running)
        result[name] = HolmResult(p_value=p_values[name], adjusted_p=adjusted, reject=adjusted <= alpha, rank=rank)
    return result
