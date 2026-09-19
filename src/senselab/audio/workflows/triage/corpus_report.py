"""The corpus fold: what a tree of triage runs decided, counted.

One run writes one decision — :meth:`FileVerdict.record`, carried by the verdict entity and by the
``run.json`` beside every store. This module reads a corpus of those and counts them, so the
questions asked of a whole release — how many flagged, on what ground, which deviations, which gates
nobody could read — are answered without reopening a store.

Every value it reports is categorical or a count. No transcript text and no detected string reaches
it, because none reaches the decision it reads.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

ROW_GLOB = "*.row.json"
"""What a corpus driver names its per-recording row."""

LOG_GLOB = "run.json"
"""What the runner names its own log beside each store."""

UNREAD = "UNREAD"
"""A recording whose row exists but carries no decision — the run raised before VERDICT folded."""


DURATION_EDGES = (1.0, 3.0, 10.0, 30.0, 60.0)
"""Where the duration buckets divide, in seconds. An unusually short recording is a different
finding from a task that was attempted and failed, so the two are counted apart."""

UNKNOWN_DURATION = "unknown"
"""A recording whose header would not give a duration."""


def duration_bucket(seconds: float | None) -> str:
    """Name the bucket one duration falls in.

    Args:
        seconds: The recording's duration, or None when its header gave none.

    Returns:
        The bucket's name, as it appears in the report.
    """
    if seconds is None:
        return UNKNOWN_DURATION
    low = 0.0
    for edge in DURATION_EDGES:
        if seconds < edge:
            return f"{low:g}-{edge:g}s"
        low = edge
    return f">={low:g}s"


@dataclass(frozen=True)
class CorpusReport:
    """What a corpus of decisions holds, counted.

    Attributes:
        files: How many decisions were read.
        unread: How many rows carried no decision at all.
        errored: Node name to how many recordings recorded an error there.
        triage: Triage outcome to count.
        release: Release outcome to count.
        discard_ground: Ground to count, over the discarded.
        route_state: File route state to count.
        routes: Branch to its route state counts.
        findings: Branch to its finding counts.
        conformance: Reporting node to its conformance counts, keyed by the stringified value.
        conformance_by_family: Declared family to node to conformance counts.
        conformance_of: Reporting node to what its conformance was about, counted.
        deviations: Node to deviation type to count.
        unmeasured: Node to config path to count — a path nobody measured on this corpus.
        critical_absences: Branch to gate to count. Any entry is a run that reached no branch.
        llm_redaction: Field to value to count, over REDACT's re-read.
        reasons: ``node|outcome|kind`` to count, over every contributing verdict.
        ran: Node to run state to count.
        families: Declared family to count.
        flagged_families: Declared family to how many of its recordings did not pass.
        durations: Duration bucket to count, read from each record's header duration.
        triage_by_duration: Duration bucket to triage outcome counts, so an unusually short
            recording can be told from a task that was attempted and failed.
    """

    files: int = 0
    unread: int = 0
    errored: dict[str, int] = field(default_factory=dict)
    triage: dict[str, int] = field(default_factory=dict)
    release: dict[str, int] = field(default_factory=dict)
    discard_ground: dict[str, int] = field(default_factory=dict)
    route_state: dict[str, int] = field(default_factory=dict)
    routes: dict[str, dict[str, int]] = field(default_factory=dict)
    findings: dict[str, dict[str, int]] = field(default_factory=dict)
    conformance: dict[str, dict[str, int]] = field(default_factory=dict)
    conformance_by_family: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    conformance_of: dict[str, dict[str, int]] = field(default_factory=dict)
    deviations: dict[str, dict[str, int]] = field(default_factory=dict)
    unmeasured: dict[str, dict[str, int]] = field(default_factory=dict)
    critical_absences: dict[str, dict[str, int]] = field(default_factory=dict)
    llm_redaction: dict[str, dict[str, int]] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)
    ran: dict[str, dict[str, int]] = field(default_factory=dict)
    families: dict[str, int] = field(default_factory=dict)
    flagged_families: dict[str, int] = field(default_factory=dict)
    durations: dict[str, int] = field(default_factory=dict)
    triage_by_duration: dict[str, dict[str, int]] = field(default_factory=dict)


def _nested(counters: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    """Order a two-level count table, outer key then descending count.

    Args:
        counters: Outer key to its counter.

    Returns:
        The same table as plain dicts, each inner one most-frequent first.
    """
    return {key: dict(counters[key].most_common()) for key in sorted(counters)}


def decisions(root: Path) -> Iterator[tuple[str, dict[str, Any] | None, dict[str, Any]]]:
    """Read every decision under a tree, whether it is a driver's rows or the runner's own logs.

    Args:
        root: A directory holding ``*.row.json`` rows, ``run.json`` logs, or both, at any depth.

    Yields:
        The recording's name, its decision or None, and the whole record it came from.
    """
    for path in sorted(list(root.rglob(ROW_GLOB)) + list(root.rglob(LOG_GLOB))):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            yield path.stem, None, {}
            continue
        if not isinstance(record, dict):
            yield path.stem, None, {}
            continue
        name = str(record.get("stem") or Path(str(record.get("source") or path)).stem)
        decision = record.get("decision")
        yield name, decision if isinstance(decision, dict) else None, record


def aggregate(records: Iterable[tuple[str, dict[str, Any] | None, dict[str, Any]]]) -> CorpusReport:
    """Count a corpus of decisions.

    Args:
        records: What :func:`decisions` yields — a name, a decision or None, and its record.

    Returns:
        The corpus fold.
    """
    files = unread = 0
    errored: Counter[str] = Counter()
    triage: Counter[str] = Counter()
    release: Counter[str] = Counter()
    ground: Counter[str] = Counter()
    route_state: Counter[str] = Counter()
    families: Counter[str] = Counter()
    flagged: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    by_duration: dict[str, Counter[str]] = {}
    routes: dict[str, Counter[str]] = {}
    findings: dict[str, Counter[str]] = {}
    conformance: dict[str, Counter[str]] = {}
    conformance_of: dict[str, Counter[str]] = {}
    by_family: dict[str, dict[str, Counter[str]]] = {}
    deviations: dict[str, Counter[str]] = {}
    unmeasured: dict[str, Counter[str]] = {}
    absences: dict[str, Counter[str]] = {}
    redaction: dict[str, Counter[str]] = {}
    ran: dict[str, Counter[str]] = {}

    for _name, decision, record in records:
        files += 1
        raw = record.get("duration_s")
        bucket = duration_bucket(float(raw) if isinstance(raw, (int, float)) else None)
        durations[bucket] += 1
        for node in record.get("errors") or {}:
            errored[str(node)] += 1
        if decision is None:
            by_duration.setdefault(bucket, Counter())[UNREAD] += 1
            unread += 1
            triage[UNREAD] += 1
            continue
        family = str(decision.get("declared_family") or "undeclared")
        families[family] += 1
        outcome = str(decision.get("triage"))
        triage[outcome] += 1
        by_duration.setdefault(bucket, Counter())[outcome] += 1
        if outcome != "pass":
            flagged[family] += 1
        release[str(decision.get("release"))] += 1
        if decision.get("discard_ground"):
            ground[str(decision["discard_ground"])] += 1
        route_state[str(decision.get("route_state"))] += 1
        for branch, state in (decision.get("routes") or {}).items():
            routes.setdefault(str(branch), Counter())[str(state)] += 1
        for branch, found in (decision.get("findings") or {}).items():
            findings.setdefault(str(branch), Counter())[str(found)] += 1
        for node, value in (decision.get("conformance") or {}).items():
            conformance.setdefault(str(node), Counter())[str(value)] += 1
            by_family.setdefault(family, {}).setdefault(str(node), Counter())[str(value)] += 1
        for node, referent in (decision.get("conformance_of") or {}).items():
            conformance_of.setdefault(str(node), Counter())[str(referent)] += 1
        for node, names in (decision.get("deviations") or {}).items():
            deviations.setdefault(str(node), Counter()).update(str(n) for n in names)
        for node, paths in (decision.get("unmeasured") or {}).items():
            unmeasured.setdefault(str(node), Counter()).update(str(p) for p in paths)
        for branch, gates in (decision.get("critical_absences") or {}).items():
            absences.setdefault(str(branch), Counter()).update(str(g) for g in gates)
        for key, value in (decision.get("llm_redaction") or {}).items():
            if isinstance(value, (str, bool, int, float)) or value is None:
                redaction.setdefault(str(key), Counter())[str(value)] += 1
            elif isinstance(value, list):
                redaction.setdefault(str(key), Counter()).update(str(v) for v in value)
        for node, state in (decision.get("ran") or {}).items():
            ran.setdefault(str(node), Counter())[str(state)] += 1
        for reason in decision.get("reasons") or []:
            if isinstance(reason, dict):
                reasons[f"{reason.get('node')}|{reason.get('outcome')}|{reason.get('kind')}"] += 1

    return CorpusReport(
        files=files,
        unread=unread,
        errored=dict(errored.most_common()),
        triage=dict(triage.most_common()),
        release=dict(release.most_common()),
        discard_ground=dict(ground.most_common()),
        route_state=dict(route_state.most_common()),
        routes=_nested(routes),
        findings=_nested(findings),
        conformance=_nested(conformance),
        conformance_by_family={family: _nested(by_family[family]) for family in sorted(by_family)},
        conformance_of=_nested(conformance_of),
        deviations=_nested(deviations),
        unmeasured=_nested(unmeasured),
        critical_absences=_nested(absences),
        llm_redaction=_nested(redaction),
        reasons=dict(reasons.most_common()),
        ran=_nested(ran),
        families=dict(families.most_common()),
        flagged_families=dict(flagged.most_common()),
        durations={name: durations[name] for name in sorted(durations, key=_bucket_order)},
        triage_by_duration={
            name: dict(by_duration[name].most_common()) for name in sorted(by_duration, key=_bucket_order)
        },
    )


def _bucket_order(name: str) -> float:
    """Sort duration buckets by their lower edge rather than alphabetically.

    Args:
        name: The bucket's name.

    Returns:
        Its lower edge, with the unknown bucket last.
    """
    if name == UNKNOWN_DURATION:
        return float("inf")
    return float(name.lstrip(">=").split("-")[0].rstrip("s"))


def _table(title: str, counts: Mapping[str, int], total: int) -> list[str]:
    """Render one count table with its share of the corpus.

    Args:
        title: The table's heading.
        counts: Value to count.
        total: The denominator every share is taken against.

    Returns:
        The table's lines, empty when there is nothing to count.
    """
    if not counts:
        return []
    lines = [f"### {title}", "", "| value | files | share |", "|---|---:|---:|"]
    for value, count in counts.items():
        share = f"{100.0 * count / total:.2f}%" if total else "--"
        lines.append(f"| `{value}` | {count} | {share} |")
    return [*lines, ""]


def _two_level(title: str, table: Mapping[str, Mapping[str, int]], total: int) -> list[str]:
    """Render one node-keyed count table.

    Args:
        title: The section's heading.
        table: Outer key to value to count.
        total: The denominator every share is taken against.

    Returns:
        The section's lines, empty when there is nothing to count.
    """
    if not table:
        return []
    lines = [f"### {title}", "", "| key | value | files | share |", "|---|---|---:|---:|"]
    for key, counts in table.items():
        for value, count in counts.items():
            share = f"{100.0 * count / total:.2f}%" if total else "--"
            lines.append(f"| `{key}` | `{value}` | {count} | {share} |")
    return [*lines, ""]


def render_markdown(report: CorpusReport, source: Path | str) -> str:
    """Render the corpus fold as a report.

    Args:
        report: What :func:`aggregate` counted.
        source: The tree it was read from, named in the heading.

    Returns:
        The report.
    """
    total = report.files
    lines = [
        f"# Triage corpus decisions — {source}",
        "",
        f"**{total} recordings read**, {report.unread} of them carrying no decision "
        f"({100.0 * report.unread / total:.2f}% unread)."
        if total
        else "No decisions were read.",
        "",
        "Every value below is categorical or a count, as the decision it is read from is.",
        "",
        "---",
        "",
        "## What the graph decided",
        "",
    ]
    lines += _table("Triage", report.triage, total)
    lines += _table("Release", report.release, total)
    lines += _table("Discard ground", report.discard_ground, total)
    lines += _table("Declared family", report.families, total)
    lines += _table("Flagged, by declared family", report.flagged_families, total)
    lines += _table("Duration", report.durations, total)
    lines += _two_level("Triage by duration", report.triage_by_duration, total)
    lines += ["## Where it was routed", ""]
    lines += _table("File route state", report.route_state, total)
    lines += _two_level("Branch route", report.routes, total)
    lines += _two_level("Branch finding", report.findings, total)
    lines += ["## What the reporting nodes said", ""]
    lines += _two_level("Conformance", report.conformance, total)
    lines += _two_level("Conformance is about", report.conformance_of, total)
    lines += _two_level("Deviations", report.deviations, total)
    lines += ["## What nobody could read", ""]
    lines += _two_level("Unmeasured config paths", report.unmeasured, total)
    lines += _two_level("Critical absences", report.critical_absences, total)
    lines += _table("Node errors", report.errored, total)
    lines += _two_level("Node run state", report.ran, total)
    lines += ["## Redaction and reasons", ""]
    lines += _two_level("LLM re-read", report.llm_redaction, total)
    lines += _table("Contributing verdicts (node|outcome|kind)", report.reasons, total)
    if report.conformance_by_family:
        lines += ["## Conformance by declared family", ""]
        for family in report.conformance_by_family:
            lines += _two_level(f"family `{family}`", report.conformance_by_family[family], total)
    return "\n".join(lines).rstrip() + "\n"
