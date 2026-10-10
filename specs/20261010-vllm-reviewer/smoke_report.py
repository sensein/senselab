#!/usr/bin/env python3
"""Labels, digests and the fold of two reviewer runs over copies of the same stores; no transcript text.

    python smoke_report.py ROWS_A ROWS_B [--out REPORT.json]

For each row it prints the reviewer's labels, quote counts and engine, whether run B's readings equal run
A's (every round's labels, quotes and reasoning, compared by sha256), the second opinion's chosen classes and
the fold's triage, release and reason. Prints counts, labels and ids only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.extend import read_store, run_root_of
from senselab.audio.workflows.triage.nodes.common import find_measurement, find_verdict
from senselab.audio.workflows.triage.nodes.review import LLM_REVIEW_MEASUREMENT
from senselab.audio.workflows.triage.vocabulary import REDACTION_LLM_ANNOTATION, SECOND_OPINION_ANSWERS

LABELS = (
    "status",
    "iterations",
    "converged",
    "redaction",
    "original",
    "speakers",
    "other_speaker",
    "off_task_speech",
    "phrases_instead_of_items",
)
QUOTES = ("phrase_quotes", "off_task_quotes", "other_speaker_quotes", "task_content_quotes", "proposal")
ROUND_KEYS = (*LABELS[3:], *QUOTES, "reasoning", "raw", "instructions_spoken", "conditions", "other_speakers")


def _digest(value: Any) -> str:  # noqa: ANN401 -- any JSON value
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def reading(row: dict[str, Any]) -> dict[str, Any]:
    """One row's reviewer reading, second opinion and fold, as labels, counts and digests."""
    store = read_store(run_root_of(Path(row["enhanced"])))
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    held = dict(annotation.attributes) if annotation is not None else {}
    rounds = [
        {key: entity.attributes.get(key) for key in ROUND_KEYS}
        for entity in store.entities("measurement")
        if entity.attributes.get("name") == LLM_REVIEW_MEASUREMENT and not store.is_invalidated(entity.id)
        and entity.attributes.get("prompt_version") == held.get("prompt_version")
    ]
    opinion = find_measurement(store, SECOND_OPINION_ANSWERS)
    verdict = find_verdict(store, "VERDICT")
    fold = dict(verdict.attributes) if verdict is not None else {}
    engine = dict(held.get("engine") or {})
    return {
        "stem": str(row["stem"])[4:12],
        "task": str(row["stem"]).split("task-", 1)[-1].rsplit("_", 1)[0],
        **{key: held.get(key) for key in LABELS},
        **{f"{key}_n": len(held.get(key) or ()) for key in QUOTES},
        "prompt_version": held.get("prompt_version"),
        "engine": engine.get("name"),
        "engine_args": " ".join(engine.get("server_args") or ()),
        "failure": (str(held.get("failure"))[:120] if held.get("failure") else None),
        "rounds_digest": _digest(rounds),
        "opinion": dict((opinion.attributes.get("choices") or {})) if opinion is not None else None,
        "opinion_status": opinion.attributes.get("status") if opinion is not None else None,
        "fold": {key: fold.get(key) for key in ("outcome", "triage", "release", "reason", "release_reason") if key in fold},
    }


def main(argv: list[str] | None = None) -> int:
    """Print and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows_a", type=Path)
    parser.add_argument("rows_b", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows_a = [json.loads(line) for line in args.rows_a.read_text().splitlines() if line.strip()]
    rows_b = {json.loads(line)["stem"]: json.loads(line) for line in args.rows_b.read_text().splitlines() if line}
    report = []
    for row in rows_a:
        a = reading(row)
        b = reading(rows_b[row["stem"]]) if row["stem"] in rows_b else None
        a["identical_in_b"] = None if b is None else a["rounds_digest"] == b["rounds_digest"]
        report.append(a)
        print(json.dumps({key: value for key, value in a.items() if key != "engine_args"}, sort_keys=False))
    same = [entry["identical_in_b"] for entry in report]
    print(f"identical readings: {sum(1 for flag in same if flag)}/{len(same)}")
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
