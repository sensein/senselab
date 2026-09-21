"""What every measurement the graph writes actually looks like, over a corpus.

A threshold is only derivable against the distribution of the reading it cuts. This reads a tree of
finished stores and reports, per measurement and per declared family, how that reading is
distributed — so a bound can be chosen against measured values rather than assumed ones, and a
count nobody asked for can be replaced by a measured central tendency.

Numeric readings get quantiles; everything else gets value counts. No transcript text and no
detected string is read: a measurement's ``value`` is a number, a flag or a short vocabulary term,
and nothing else is taken from the store.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

STORE_NAME = "store.jsonl"
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
"""Where a bound is usually argued about: the tails, the quartiles and the middle."""


@dataclass
class Distribution:
    """One measurement's readings within one family.

    Attributes:
        n: How many recordings carried a reading.
        numeric: The numeric readings, retained so quantiles can be taken.
        values: Counts per distinct value, for readings that are not numbers.
    """

    n: int = 0
    numeric: list[float] = field(default_factory=list)
    values: dict[str, int] = field(default_factory=dict)

    def add(self, value: Any) -> None:  # noqa: ANN401 -- a measurement's value is its own type
        """Record one reading.

        Args:
            value: The measurement's value, of whatever type it was written with.
        """
        self.n += 1
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            key = str(value)[:40]
            self.values[key] = self.values.get(key, 0) + 1
            return
        number = float(value)
        if math.isfinite(number):
            self.numeric.append(number)
        else:
            self.values["non-finite"] = self.values.get("non-finite", 0) + 1

    def summary(self) -> dict[str, Any]:
        """The distribution, as quantiles for numbers and counts for everything else.

        Returns:
            A mapping carrying ``n`` and either the quantiles or the value counts.
        """
        out: dict[str, Any] = {"n": self.n}
        if self.numeric:
            ordered = sorted(self.numeric)
            out["numeric_n"] = len(ordered)
            out["min"] = ordered[0]
            out["max"] = ordered[-1]
            for q in QUANTILES:
                index = min(len(ordered) - 1, int(q * len(ordered)))
                out[f"p{int(q * 100)}"] = round(ordered[index], 4)
        if self.values:
            out["values"] = dict(sorted(self.values.items(), key=lambda kv: -kv[1])[:8])
        return out


def readings(root: Path) -> Iterator[tuple[str, str, Any]]:
    """Every measurement in every store under a tree, with the family that produced it.

    Args:
        root: A directory holding run directories at any depth.

    Yields:
        The declared family, the measurement's name, and its value.
    """
    for store in sorted(root.rglob(STORE_NAME)):
        family, found = "", []
        try:
            lines = store.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("record") != "entity":
                continue
            attributes = record.get("attributes") or {}
            if record.get("prov_type") == "verdict" and attributes.get("node") == "VERDICT":
                family = str(attributes.get("declared_family") or "(undeclared)")
            elif record.get("prov_type") == "measurement" and attributes.get("name"):
                found.append((str(attributes["name"]), attributes.get("value")))
        for name, value in found:
            if value is not None:
                yield family, name, value


def collect(items: Iterator[tuple[str, str, Any]]) -> dict[str, dict[str, Distribution]]:
    """Group readings by measurement and family.

    Args:
        items: What :func:`readings` yields.

    Returns:
        Measurement name to family to its distribution.
    """
    out: dict[str, dict[str, Distribution]] = defaultdict(lambda: defaultdict(Distribution))
    for family, name, value in items:
        out[name][family].add(value)
    return {name: dict(families) for name, families in out.items()}


def render_markdown(stats: Mapping[str, Mapping[str, Distribution]], source: Path | str) -> str:
    """Render the distributions, widest-spread measurements first.

    Args:
        stats: What :func:`collect` returned.
        source: The tree the readings came from.

    Returns:
        The report.
    """
    lines = [f"# Measurement distributions — {source}", ""]
    lines.append(f"{len(stats)} distinct measurements. Numeric readings carry quantiles; the rest carry counts.")
    lines.append("A bound belongs against these, not against a guess.\n")
    for name in sorted(stats):
        families = stats[name]
        total = sum(d.n for d in families.values())
        lines += [f"## `{name}` — {total} readings over {len(families)} families", ""]
        numeric = any(d.numeric for d in families.values())
        if numeric:
            lines += ["| family | n | p5 | p25 | median | p75 | p95 |", "|---|---:|---:|---:|---:|---:|---:|"]
        else:
            lines += ["| family | n | values |", "|---|---:|---|"]
        for family in sorted(families, key=lambda f: -families[f].n)[:12]:
            s = families[family].summary()
            if numeric and "p50" in s:
                lines.append(
                    f"| `{family}` | {s['n']} | {s['p5']} | {s['p25']} | **{s['p50']}** | {s['p75']} | {s['p95']} |"
                )
            elif not numeric:
                shown = ", ".join(f"`{k}`×{v}" for k, v in (s.get("values") or {}).items())
                lines.append(f"| `{family}` | {s['n']} | {shown} |")
        lines.append("")
    return "\n".join(lines) + "\n"
