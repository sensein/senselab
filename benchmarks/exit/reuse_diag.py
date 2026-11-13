"""List the keys with a start-phase execution and no `senselab:Reuse` in a Step 4 run, with the status of each.

Reads the run as the driver's `prov` command does: the writer documents of `ProvStore.read_documents`, a document's
phase by its activities' times against the first event of the run's last epoch, a key as the digest the events name
(`provenance_check.activity_key`). The two sources are joined with `keyjoin.join`, which prints one key of each and
fails when none matches.

    uv run --no-sync python benchmarks/exit/reuse_diag.py RUN_DIR START_JSON RESUME_JSON
"""

import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import keyjoin
from senselab_core.runtime.execution.events import RESULTS, Event, read_events, terminal_events
from senselab_core.runtime.execution.provenance_check import activity_key
from senselab_core.runtime.provenance import ProvStore


class Row(NamedTuple):
    """One activity: its document's phase and name, its key digest, its kind, and who settled it (a Reuse's writer)."""

    phase: str
    document: str
    key: str
    kind: str
    writer: str


def is_settled_reuse(node: Mapping[str, Any]) -> bool:
    """Whether the Reuse `node` has the settled shape: the coordinator served an earlier outcome without running it.

    A worker's Reuse records its execution context and the agent it was associated with; a settled one records neither.
    """
    return "senselab:context" not in node and "prov:wasAssociatedWith" not in node


@dataclass(frozen=True)
class Finding:
    """A key executed before the interrupt and served by no Reuse, and what its events say became of it."""

    key: str
    document: str
    status: str


def seconds(node: Mapping[str, Any], key: str) -> float:
    """Return the epoch seconds of the `xsd:dateTime` literal that `node` holds under `key`."""
    return datetime.fromisoformat(node[key]["@value"].replace("Z", "+00:00")).timestamp()


def boundary_of(events: Sequence[Event]) -> float:
    """Return the time of the first event of the run's last epoch, in seconds."""
    last = max(e.epoch for e in events)
    return min(e.t_ns for e in events if e.epoch == last) / 1e9


def activity_rows(run: Path, boundary: float) -> list[Row]:
    """Return a row for each activity of the run's provenance documents."""
    rows: list[Row] = []
    for name, data in sorted(ProvStore.read_documents(run).items()):
        acts = ProvStore.from_documents({name: data}).activities()
        ended = [seconds(a.node(), "prov:endedAtTime") for a in acts]
        began = [seconds(a.node(), "prov:startedAtTime") for a in acts]
        phase = "start" if acts and max(ended) <= boundary else "resume" if acts and min(began) >= boundary else "other"
        for a in acts:
            reuse = "senselab:Reuse" in a.types()
            writer = "coordinator" if reuse and is_settled_reuse(a.node()) else "worker"
            rows.append(Row(phase, name, activity_key(a), "reuse" if reuse else "execution", writer))
    return rows


def diagnose(run: Path, out: Callable[[str], None] = print) -> list[Finding]:
    """Return the start-phase executions of `run` that have no Reuse, each with its status.

    A status is `committed` (a result event before the resume holds the key), `late` (no result before it, and the
    resume did not execute it) or `uncommitted-executed-in-resume`.

    Raises:
        keyjoin.JoinFailure: the provenance keys and the event keys share none.
    """
    events = read_events(run)
    last = max(e.epoch for e in events)
    boundary = boundary_of(events)
    out(f"events {len(events)}, epochs {sorted({e.epoch for e in events})}, boundary {boundary}")
    before = [e for e in events if e.epoch < last]
    held = {e.key for job, e in terminal_events(before).items() if e.kind in RESULTS and job[1] and e.key}
    finished = Counter(e.key for e in events if e.epoch == last and e.kind == "finished" and e.key)
    rows = activity_rows(run, boundary)
    start_exec = {r.key: r.document for r in rows if r.phase == "start" and r.kind == "execution"}
    keyjoin.join(
        "start-phase execution keys (prov)", start_exec.keys(), "event keys", {e.key for e in events if e.key}, out
    )
    reused = {r.key for r in rows if r.kind == "reuse"}
    missing = sorted(set(start_exec) - reused)
    out(
        f"start-phase execution keys {len(start_exec)}, with a Reuse {len(set(start_exec) & reused)}, "
        f"without {len(missing)}"
    )
    found = []
    for key in missing:
        status = "committed" if key in held else ("uncommitted-executed-in-resume" if finished[key] else "late")
        out(f"key {key}: prov document {start_exec[key]}; STATUS: {status}")
        found.append(Finding(key, start_exec[key], status))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """Print the diagnosis of the run named by `argv` (run directory, start.json, resume.json)."""
    run, start_json, resume_json = (Path(a) for a in (argv if argv is not None else sys.argv[1:4]))
    print("start.json", json.loads(start_json.read_text(encoding="utf-8")))
    print("resume.json", json.loads(resume_json.read_text(encoding="utf-8")))
    diagnose(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
