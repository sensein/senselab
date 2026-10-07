#!/usr/bin/env python3
r"""Score triage decisions against the owner's listening labels.

    uv run python scripts/evaluate_task_events.py LABELS.csv DECISIONS \
        [--split src/senselab/audio/workflows/triage/data/evaluation/task_events_split.yaml] \
        [--label-map src/senselab/audio/workflows/triage/data/evaluation/owner_label_map.yaml] \
        [--out DIR]

``LABELS.csv`` is the owner label table (``listen_set``, ``stem``, ``family``, ``owner_label``,
``owner_note``), or a ``.json`` exported from the triage review page, whose entries carry the same
keys. ``DECISIONS`` is either a ``recording_vectors.parquet`` (``stem``, ``verdict``,
``ground_keys``, ``declared_family``) or a directory of replay/refold row files (``*.jsonl``, each
row carrying ``stem`` and ``verdict``, or ``REFOLD``/``VERDICT`` as ``<verdict>/<release>``, and
optionally ``ground_keys``). On ORCD, point ``DECISIONS`` at a replay's rows directory.

The split file is read when it exists and written when it does not; a labelled subject missing
from it is held out. ``--out`` receives ``report.json`` and ``disagreements.csv`` (per-recording,
so keep it outside the repository). The summary is printed. The protocol is in
``specs/20261007-task-events-in-background/design.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DATA = Path(__file__).resolve().parents[1] / "src/senselab/audio/workflows/triage/data/evaluation"
SPLIT_SEED = 20261007
HELDOUT_FRACTION = 0.3
REVIEW_BAND_KEYS = ("breath_review_low_confidence", "cough_review_low_confidence", "voice_review_low_confidence")
VOICE_FAMILIES = (
    "maximum-phonation-time",
    "maximum-phonation-time-v2",
    "prolonged-vowel",
    "glides-low-to-high",
    "glides-high-to-low",
    "high-to-low",
)
REVIEW_SETS = ("breath_review_check", "discard_contested_check")
_TIMESTAMP = re.compile(r"_\d{8}-\d{6}$")
_OUTCOME = {"pass": "pass", "review": "review", "discard": "discard", "flag": "review", "rerun": "incomplete"}
"""Verdict to outcome. ``flag`` and ``rerun`` are the values tables written before schema 25 carry."""


def _outcome(decision: pd.Series) -> str:
    """A decision's outcome: its verdict, or ``incomplete`` where its run is."""
    if str(decision.get("run_status")) == "incomplete":
        return "incomplete"
    return _OUTCOME.get(str(decision["verdict"]), str(decision["verdict"]))


def recording_key(stem: str) -> str:
    """The stem without the run timestamp, so labels and decisions join."""
    return _TIMESTAMP.sub("", str(stem))


def subject(stem: str) -> str:
    """The 8-hex subject prefix of a BIDS stem."""
    match = re.match(r"sub-([0-9a-fA-F]{8})", str(stem))
    return match.group(1).lower() if match else str(stem)[:8]


def family_group(family: str, listen_set: str) -> str:
    """breath, cough, voice, contested_review or other."""
    if listen_set.startswith(REVIEW_SETS):
        return "contested_review"
    fam = str(family)
    if fam in VOICE_FAMILIES:
        return "voice"
    if "cough" in fam and "breath" not in fam:
        return "cough"
    if "breath" in fam:
        return "breath"
    return "other"


REVIEW_EXPORT_SCHEMA = "senselab.triage.review"
LABEL_COLUMNS = ("listen_set", "stem", "family", "owner_label", "owner_note")


def load_labels(path: Path) -> pd.DataFrame:
    """The labels from the owner table, or from a review page export.

    Args:
        path: A ``.csv`` label table, or a ``.json`` review export (``{"schema", "entries": [...]}``).

    Returns:
        One row per label, with at least the label table's columns.

    Raises:
        ValueError: When a JSON file is not a review page export.
    """
    if path.suffix.lower() != ".json":
        return pd.read_csv(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and payload.get("schema") not in (None, REVIEW_EXPORT_SCHEMA):
        raise ValueError(f"{path} is a {payload.get('schema')!r} file, not a triage review export")
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    table = pd.DataFrame(entries)
    for column in LABEL_COLUMNS:
        if column not in table.columns:
            table[column] = ""
    return table


def map_labels(labels: pd.DataFrame, label_map: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Add ``target`` and ``remark`` columns from the label map.

    Returns:
        The labels with their targets, and the descriptions of rows no rule mapped.
    """
    by_label = label_map.get("labels") or {}
    notes = label_map.get("notes") or []
    targets, remarks, unmapped = [], [], []
    for _, row in labels.iterrows():
        entry = by_label.get(str(row.get("owner_label") or "").strip())
        if entry is None:
            note = str(row.get("owner_note") or "").lower()
            entry = next((rule for rule in notes if str(rule["match"]).lower() in note), None)
        if entry is None:
            unmapped.append(f"{row['listen_set']} {subject(row['stem'])} {row.get('owner_label')!r}")
            targets.append("excluded")
            remarks.append("")
        else:
            targets.append(entry["target"])
            remarks.append(entry.get("remark", ""))
    out = labels.copy()
    out["target"] = targets
    out["remark"] = remarks
    return out, unmapped


def make_split(labels: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Fit and held-out subjects per listen set: stratified by target, grouped by subject, 30% held out."""
    rng = np.random.default_rng(SPLIT_SEED)
    split: dict[str, dict[str, list[str]]] = {}
    for listen_set in sorted(labels["listen_set"].unique()):
        rows = labels[labels["listen_set"] == listen_set]
        by_subject: dict[str, Counter[str]] = defaultdict(Counter)
        for _, row in rows.iterrows():
            by_subject[subject(row["stem"])][row["target"]] += 1
        strata: dict[str, list[str]] = defaultdict(list)
        for subj in sorted(by_subject):
            strata[by_subject[subj].most_common(1)[0][0]].append(subj)
        held: list[str] = []
        for target in sorted(strata):
            members = strata[target]
            order = rng.permutation(len(members))
            n_held = int(round(HELDOUT_FRACTION * len(members)))
            held.extend(members[i] for i in order[:n_held])
        split[str(listen_set)] = {"fit": sorted(set(by_subject) - set(held)), "heldout": sorted(held)}
    return split


def load_or_write_split(path: Path, labels: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Read the split file, or write it from these labels when it does not exist."""
    if path.exists():
        loaded = yaml.safe_load(path.read_text()) or {}
        return {
            str(ls): {part: [str(s) for s in (sides or {}).get(part) or []] for part in ("fit", "heldout")}
            for ls, sides in loaded.items()
        }
    split = make_split(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Fit and held-out subjects (8-hex prefixes) per listen set; numpy default_rng({SPLIT_SEED}), "
        f"{int(HELDOUT_FRACTION * 100)}% held out,\n# stratified by target, grouped by subject. "
        "Written once by scripts/evaluate_task_events.py;\n# subjects absent here are held out.\n"
    )
    path.write_text(header + yaml.safe_dump(split, sort_keys=True))
    return split


def side(row: pd.Series, split: dict[str, dict[str, list[str]]]) -> str:
    """``fit`` when the split file puts the row's subject there; otherwise ``heldout``."""
    return "fit" if subject(row["stem"]) in split.get(str(row["listen_set"]), {}).get("fit", []) else "heldout"


def _parse_verdict(record: dict[str, Any]) -> str | None:
    if record.get("verdict"):
        return str(record["verdict"])
    for key in ("VERDICT", "REFOLD"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value.split("/", 1)[0]
    return None


def load_decisions(source: Path) -> pd.DataFrame:
    """Decisions keyed by recording: ``verdict`` and ``ground_keys``, from a parquet or a rows directory."""
    if source.is_file():
        table = pd.read_parquet(source)
        keep = [c for c in ("stem", "verdict", "run_status", "ground_keys", "declared_family") if c in table.columns]
        table = table[keep].copy()
    else:
        records = []
        for path in sorted(source.rglob("*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                verdict = _parse_verdict(record)
                if record.get("stem") and verdict:
                    records.append(
                        {"stem": record["stem"], "verdict": verdict, "ground_keys": record.get("ground_keys") or []}
                    )
        table = pd.DataFrame(records, columns=["stem", "verdict", "ground_keys"])
    if "ground_keys" not in table.columns:
        table["ground_keys"] = [[] for _ in range(len(table))]
    table["key"] = table["stem"].map(recording_key)
    return table.drop_duplicates("key", keep="last").set_index("key")


def score(labels: pd.DataFrame, decisions: pd.DataFrame, label_map: dict[str, Any]) -> pd.DataFrame:
    """One row per scored label: target, outcome, agreement, side and group."""
    allowed = {k: set(v or []) for k, v in (label_map.get("targets") or {}).items()}
    rows = []
    for _, row in labels.iterrows():
        if row["target"] == "excluded":
            continue
        decision = decisions.loc[recording_key(row["stem"])] if recording_key(row["stem"]) in decisions.index else None
        outcome = "missing" if decision is None else _outcome(decision)
        keys = [] if decision is None else list(decision["ground_keys"])
        rows.append(
            {
                "listen_set": row["listen_set"],
                "subject": subject(row["stem"]),
                "stem": row["stem"],
                "group": family_group(row.get("family", ""), str(row["listen_set"])),
                "target": row["target"],
                "remark": row.get("remark", ""),
                "outcome": outcome,
                "agree": outcome in allowed.get(row["target"], set()),
                "side": row["side"],
                "keys": ";".join(str(k) for k in keys),
                "owner_note": row.get("owner_note", ""),
            }
        )
    return pd.DataFrame(rows)


def review_band_rate(decisions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per family group, the share of kept recordings in a review band; an outcome, not a target."""
    if "declared_family" not in decisions.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    kept = decisions[decisions["verdict"].isin(["pass", "review", "flag"])]
    groups = kept["declared_family"].map(lambda f: family_group(f, ""))
    for group in ("breath", "cough", "voice"):
        rows = kept[groups == group]
        if len(rows) == 0:
            continue
        in_band = rows["ground_keys"].map(lambda keys: any(k in REVIEW_BAND_KEYS for k in keys)).sum()
        rate = round(float(in_band) / len(rows), 4)
        out[group] = {"kept": int(len(rows)), "in_review_band": int(in_band), "rate": rate}
    return out


def summarise(scored: pd.DataFrame, decisions: pd.DataFrame, unmapped: list[str]) -> dict[str, Any]:
    """The report: agreement by side and group, confusion matrices, remarks and the review-band rate."""
    report: dict[str, Any] = {"n_scored": int(len(scored)), "unmapped": unmapped}
    report["agreement"] = {
        s: {
            "n": int((scored["side"] == s).sum()),
            "agree": int(scored.loc[scored["side"] == s, "agree"].sum()),
        }
        for s in ("fit", "heldout")
    }
    by_group: dict[str, Any] = {}
    for group, rows in scored.groupby("group"):
        by_group[str(group)] = {
            "fit": f"{int(rows.loc[rows.side == 'fit', 'agree'].sum())}/{int((rows.side == 'fit').sum())}",
            "heldout": f"{int(rows.loc[rows.side == 'heldout', 'agree'].sum())}/{int((rows.side == 'heldout').sum())}",
            "confusion": pd.crosstab(rows["target"], rows["outcome"]).to_dict(orient="index"),
        }
    report["by_group"] = by_group
    report["remarks"] = {
        str(k): int(v) for k, v in Counter(r for r in scored["remark"] if isinstance(r, str) and r).items()
    }
    report["review_band_rate"] = review_band_rate(decisions)
    return report


def main() -> int:
    """Score, print the summary and write the report.

    Returns:
        0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None)
    parser.add_argument("labels", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--split", type=Path, default=DATA / "task_events_split.yaml")
    parser.add_argument("--label-map", type=Path, default=DATA / "owner_label_map.yaml")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    label_map = yaml.safe_load(args.label_map.read_text())
    labels = load_labels(args.labels)
    labels = labels.drop_duplicates(["listen_set", "stem"], keep="last")
    labels, unmapped = map_labels(labels, label_map)
    split = load_or_write_split(args.split, labels[labels["target"] != "excluded"])
    labels["side"] = [side(row, split) for _, row in labels.iterrows()]
    decisions = load_decisions(args.decisions)
    scored = score(labels, decisions, label_map)
    report = summarise(scored, decisions, unmapped)

    print(f"scored {report['n_scored']} labels")
    for s, v in report["agreement"].items():
        print(f"  {s:8s} agree {v['agree']}/{v['n']}")
    for group, v in report["by_group"].items():
        print(f"  {group:17s} fit {v['fit']:>7s}  heldout {v['heldout']:>7s}")
    print("  review-band rate (outcome):", report["review_band_rate"])
    if unmapped:
        print("  unmapped labels:", *unmapped, sep="\n    ")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        scored[~scored["agree"]].to_csv(args.out / "disagreements.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
