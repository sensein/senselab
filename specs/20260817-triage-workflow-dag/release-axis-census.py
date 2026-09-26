"""The release-axis census behind verdict.md's population tables.

Reads the evidence VERDICT's release fold reads — SPEECH's run state and lexical count, the
``pii_scan`` tri-state, the live ``pii`` count and REDACT's own outcome — straight off each
replayed store, recovering the latest generation from the invalidation edges the way
``replay_diff.split_generations`` does. Imports no senselab, so it runs on a bare interpreter.

Usage, one Slurm task per slice of subject directories::

    python release-axis-census.py <subject dir>... <out.jsonl>

Run 2026-09-22 over ``triage_replay_20260922/out/`` as a 64-task array on ``mit_preemptable``
(job 23515058): 62,548 recordings, zero read errors. The figures are in
``verdict.md`` § What the old rule was reporting.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPLAYED = {"TAXONOMY", "routing", "AIRWAY", "SPEECH", "VOICE", "QUALITY", "REDACT", "VERDICT"}


def read_store(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, list[str]]]:
    """Entities, activities and the two relations this probe reads, from one store.jsonl."""
    entities, acts, generated, invalidated = {}, {}, {}, {}
    with open(path) as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            record = obj.get("record")
            if record == "entity":
                entities[obj["id"]] = obj
            elif record == "activity":
                acts[obj["id"]] = obj
            elif record == "relation":
                rel, src, tgt = obj.get("relation"), obj.get("source"), obj.get("target")
                if rel == "wasGeneratedBy":
                    generated[src] = tgt
                elif rel == "wasInvalidatedBy":
                    invalidated.setdefault(src, []).append(tgt)
    return entities, acts, generated, invalidated


def probe(path: Path) -> dict[str, Any]:
    """The release evidence the latest generation of one store carries."""
    entities, acts, generated, invalidated = read_store(path)
    retiring = {i for i, a in acts.items() if a.get("node") == "REPLAY" and a.get("step") == "decision_superseded"}
    after = []
    for eid, ent in entities.items():
        if any(a in retiring for a in invalidated.get(eid, ())):
            continue
        act = acts.get(generated.get(eid))
        if act is not None and act.get("node") in REPLAYED:
            after.append(ent)

    decision, speech_report, scans, pii_n = None, None, [], 0
    redact_verdict = None
    for ent in after:
        kind, attrs = ent.get("prov_type"), ent.get("attributes", {})
        if kind == "verdict" and attrs.get("node") == "VERDICT":
            decision = attrs
        elif kind == "verdict" and attrs.get("node") == "REDACT":
            redact_verdict = attrs
        elif kind == "branch_report" and attrs.get("node") == "SPEECH":
            speech_report = attrs
        elif kind == "pii":
            pii_n += 1
        elif kind == "measurement" and attrs.get("name") == "pii_scan":
            scans.append(attrs)

    scanned = None if not scans else (not any(s.get("scanned") is False for s in scans))
    ran = (decision or {}).get("ran") or {}
    notes = [] if speech_report is None else [str(n) for n in (speech_report.get("notes") or ())]
    return {
        "stem": Path(path).parents[1].name,
        "speech_ran": ran.get("SPEECH"),
        "redact_ran": ran.get("REDACT"),
        "release_old": (decision or {}).get("release"),
        "scanned": scanned,
        "scans_n": len(scans),
        "pii_n": pii_n,
        "words_n": None if speech_report is None else speech_report.get("words_n"),
        "diarization": None if speech_report is None else speech_report.get("diarization"),
        "speech_reported": speech_report is not None,
        "no_lexical_note": any("no consensus word" in n for n in notes),
        "redact_outcome": None if redact_verdict is None else redact_verdict.get("outcome"),
        "has_decision": decision is not None,
    }


def main() -> None:
    """Probe every store under the given roots, one JSON object per line."""
    roots, out = sys.argv[1:-1], sys.argv[-1]
    bad = Counter()
    with open(out, "w") as handle:
        for root in roots:
            for store in sorted(Path(root).rglob("run/store.jsonl")):
                try:
                    handle.write(json.dumps(probe(store)) + "\n")
                except Exception as error:  # noqa: BLE001
                    bad[type(error).__name__] += 1
    sys.stderr.write(f"errors: {dict(bad)}\n")


if __name__ == "__main__":
    main()
