#!/usr/bin/env python3
r"""Measure the session background against the session floor over a sample of finished runs, read-only.

    uv run python scripts/measure_session_background.py MANIFEST --out-dir DIR \
        [--sessions 200] [--seed 0] [--slice-index I --slice-count N]

``MANIFEST`` is the JSONL every extend driver takes (``stem``, ``enhanced``). ``--sessions`` BIDS
sessions are drawn with ``--seed`` and sharded by ``--slice-index`` / ``--slice-count``. For every
member the script reads SESSION's speech-residual record off the stored streams, and for every session
the session background, beside the stored session floor and BACKGROUND's floor. For every airway, voice
and syllable-repetition member of a session with a background, it reads the branch's task-event decision
twice from the same code, against BACKGROUND's stored view and against that view with the session
background as its floor. Nothing is written into any run: the stores are read into memory and dropped.
One JSONL of per-recording rows per task goes under ``--out-dir``; transcript text is never written.
``summarise`` folds the rows into distributions.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from senselab.audio.workflows.triage import breath_pattern, cough_pattern, task_events, voice_phonation
from senselab.audio.workflows.triage.background_model import (
    BACKGROUND_MODEL,
    Floor,
    background_model_parameters,
    regions_of,
)
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.extend import RUN_SUBDIR, read_manifest, read_store, run_root_of, take_slice
from senselab.audio.workflows.triage.nodes import ddk as ddk_node
from senselab.audio.workflows.triage.ddk_task import DDK_READING
from senselab.audio.workflows.triage.nodes.airway_task import BREATH_READING, COUGH_READING, measure_airway_task
from senselab.audio.workflows.triage.nodes.background import (
    SESSION_FLOOR,
    declared_family,
    session_key,
    speech_residual_attributes,
)
from senselab.audio.workflows.triage.nodes.branches import AIRWAY_EXPECTATIONS, SPEECH_EXPECTATIONS
from senselab.audio.workflows.triage.nodes.common import describe_exception, find_measurement, software_agent
from senselab.audio.workflows.triage.nodes.voice import PHONATION_READING, read_phonation
from senselab.audio.workflows.triage.session_background import kind_of, session_background_of
from senselab.audio.workflows.triage.task_events import GenericView, generic_view_of
from senselab.utils.prov_store import ProvStore

DECIDED_KINDS = ("airway", "voice", "syllable_repetition")
READINGS = {
    "airway": (BREATH_READING, COUGH_READING),
    "voice": (PHONATION_READING,),
    "syllable_repetition": (DDK_READING,),
}
VIEW_READERS = (task_events, breath_pattern, cough_pattern, voice_phonation, ddk_node)


def build_parser() -> argparse.ArgumentParser:
    """The CLI.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
    summary = sub.add_parser("summarise", help="Fold the per-recording rows into distributions")
    summary.add_argument("out_dir", type=Path)
    parser.add_argument("manifest", type=Path, nargs="?", help="Manifest JSONL: one object per recording")
    parser.add_argument("--out-dir", type=Path, help="Where the rows go")
    parser.add_argument("--sessions", type=int, default=200, help="BIDS sessions drawn")
    parser.add_argument("--seed", type=int, default=0, help="The draw's seed")
    parser.add_argument("--slice-index", type=int, default=0, help="This task's index, 0-based")
    parser.add_argument("--slice-count", type=int, default=1, help="How many tasks the array has")
    return parser


def _broadband(band_db: Sequence[float] | None) -> float | None:
    if band_db is None:
        return None
    return float(10.0 * np.log10(np.sum(10.0 ** (np.asarray(band_db, dtype=np.float64) / 10.0)) + 1e-12))


def background_view(view: GenericView, band_db: Sequence[float]) -> GenericView:
    """The stored view with the session background as its floor, regions read again against it.

    Args:
        view: BACKGROUND's stored view.
        band_db: The session background per band.

    Returns:
        The view.
    """
    band = np.asarray(band_db, dtype=np.float64)
    old = view.floor
    floor = Floor(band, "speech_residual", old.quiet_s, old.own_db, old.residual_db)
    regions = tuple(regions_of(view.frames, band, view.impulses, background_model_parameters()))
    return GenericView(view.frames, floor, view.impulses, regions, float(_broadband(band) or 0.0), view.hop_s)


@contextmanager
def reading_view(view: GenericView | None) -> Iterator[None]:
    """Every branch reads ``view`` in place of BACKGROUND's stored one, while the block runs.

    Args:
        view: The view, or None to leave the stored one in place.

    Yields:
        Nothing.
    """
    if view is None:
        yield
        return
    held = {module: module.generic_view_of for module in VIEW_READERS}
    try:
        for module in VIEW_READERS:
            module.generic_view_of = lambda store, run_dir: view  # type: ignore[attr-defined]
        yield
    finally:
        for module, original in held.items():
            module.generic_view_of = original  # type: ignore[attr-defined]


def branch_decision(run_root: Path, family: str, view: GenericView | None, config: Any) -> str | None:  # noqa: ANN401
    """The branch's task-event decision for one recording, read on a fresh in-memory copy of its store.

    Args:
        run_root: The run root.
        family: The declared family.
        view: The view the branch reads, or None for BACKGROUND's stored one.
        config: The triage configuration.

    Returns:
        ``present``, ``review`` or ``absent``; None where the reading was not taken.
    """
    store = read_store(run_root)
    run_dir = run_root / RUN_SUBDIR
    kind = kind_of(family)
    with reading_view(view):
        if kind == "airway" and family in AIRWAY_EXPECTATIONS:
            activity = store.activity(node="AIRWAY", step="measure_session_background", parameters={})
            software = software_agent(store)
            measure_airway_task(
                store,
                activity,
                software,
                family=family,
                expectation=AIRWAY_EXPECTATIONS[family],
                sampling_hz=float(config.require("resample.target_hz")),
                language=None,
                run_dir=run_dir,
            )
            for name in READINGS["airway"]:
                found = find_measurement(store, name)
                if found is not None and found.attributes.get("reading"):
                    return _decision_of(found.attributes)
            return None
        if kind == "voice":
            read = read_phonation(store, run_dir, family)
            evidence = getattr(read, "evidence", None)
            return None if evidence is None else str(evidence.decision)
        if kind == "syllable_repetition" and family in SPEECH_EXPECTATIONS:
            reads = ddk_node.read_ddk(store, run_dir, "plain")
            from senselab.audio.workflows.triage.nodes.speech import branch_params  # noqa: PLC0415

            reading, _ = ddk_node.ddk_reading(SPEECH_EXPECTATIONS[family], store, branch_params(config), reads)
            return reading.decision
    return None


def stored_decision(store: ProvStore, kind: str | None) -> str | None:
    """The decision the run stored for its task reading.

    Args:
        store: The store.
        kind: The declared kind.

    Returns:
        The stored evidence decision, or None.
    """
    for name in READINGS.get(kind or "", ()):
        found = find_measurement(store, name)
        if found is None:
            continue
        decision = _decision_of(found.attributes)
        if decision is not None:
            return decision
    return None


def _decision_of(attributes: dict[str, Any]) -> str | None:
    for holder in (attributes.get("reading") or {}, attributes):
        evidence = holder.get("evidence") or {}
        if isinstance(evidence, dict) and evidence.get("decision") is not None:
            return str(evidence["decision"])
    value = attributes.get("decision", attributes.get("value"))
    return None if value is None else str(value)


def measure_session(session: str, rows: Sequence[dict[str, Any]], config: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """The per-recording rows of one session.

    Args:
        session: The session key.
        rows: Its members' manifest rows.
        config: The triage configuration.

    Returns:
        One row per member.
    """
    opened = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
            store = read_store(run_root)
            reading = speech_residual_attributes(store, run_root / RUN_SUBDIR)
            opened.append((row, run_root, store, {**reading, "stem": row["stem"]}, None))
        except Exception as error:  # noqa: BLE001 -- one unreadable member is a row, not a crash
            opened.append((row, None, None, {"missing": ["store"]}, describe_exception(error)))
    background = session_background_of([r for _, _, _, r, _ in opened if not r.get("missing")])
    out = []
    for row, run_root, store, reading, error in opened:
        record: dict[str, Any] = {"stem": row["stem"], "session": session, "error": error}
        if store is None or run_root is None:
            out.append(record)
            continue
        family = declared_family(store)
        kind = kind_of(family)
        floor = find_measurement(store, SESSION_FLOOR)
        model = find_measurement(store, BACKGROUND_MODEL)
        model_floor = ((model.attributes.get("floor") or {}) if model is not None else {}) or {}
        record.update(
            {
                "family": family,
                "kind": kind,
                "reading": {k: v for k, v in reading.items() if k not in ("residual_band_db", "band_edges_hz")},
                "session_background": {k: v for k, v in background.items() if k != "members"},
                "old_session_floor_db": _broadband(None if floor is None else floor.attributes.get("band_db")),
                "old_session_floor_source": None if floor is None else floor.attributes.get("floor_source"),
                "old_background_floor_db": _broadband(model_floor.get("band_db")),
                "old_background_floor_source": model_floor.get("source"),
                "new_background_db": _broadband(background.get("band_db")),
            }
        )
        if kind in DECIDED_KINDS and family is not None:
            record["stored_decision"] = stored_decision(store, kind)
            band = background.get("band_db")
            stored_view = generic_view_of(store, run_root / RUN_SUBDIR)
            try:
                record["old_decision"] = branch_decision(run_root, family, None, config)
                record["new_decision"] = (
                    None
                    if band is None or stored_view is None
                    else branch_decision(run_root, family, background_view(stored_view, band), config)
                )
            except Exception as error:  # noqa: BLE001 -- a reading that raises is recorded, not fatal
                record["decision_error"] = describe_exception(error)
        out.append(record)
    return out


def draw_sessions(manifest: Path, sessions: int, seed: int) -> list[tuple[str, list[dict[str, Any]]]]:
    """The drawn sessions and their rows.

    Args:
        manifest: The manifest.
        sessions: How many to draw.
        seed: The seed.

    Returns:
        ``(session, rows)`` pairs in sorted order.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in read_manifest(manifest, required=("stem", "enhanced")):
        groups.setdefault(session_key(str(row["stem"])) or str(row["stem"]), []).append(row)
    keys = sorted(groups)
    chosen = sorted(random.Random(seed).sample(keys, min(sessions, len(keys))))
    return [(key, groups[key]) for key in chosen]


def run(args: argparse.Namespace) -> int:
    """Measure one shard of the drawn sessions.

    Args:
        args: The parsed command line.

    Returns:
        0.
    """
    started = time.time()
    config = load_triage_config()
    drawn = draw_sessions(args.manifest, args.sessions, args.seed)
    mine = take_slice([{"session": s, "rows": r} for s, r in drawn], args.slice_index, args.slice_count)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"rows-{args.slice_index}-of-{args.slice_count}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for group in mine:
            for record in measure_session(group["session"], group["rows"], config):
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
    print(f"{len(mine)} sessions in {time.time() - started:.0f}s -> {path}")
    return 0


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": len(values), **{f"p{p}": round(float(v), 2) for p, v in zip((5, 25, 50, 75, 95), q)}}


def summarise(out_dir: Path) -> dict[str, Any]:
    """Fold every rows file under a directory into distributions.

    Args:
        out_dir: The directory.

    Returns:
        The summary.
    """
    rows = [json.loads(line) for path in sorted(out_dir.glob("rows-*.jsonl")) for line in path.read_text().splitlines()]
    sessions: dict[str, dict[str, Any]] = {}
    for row in rows:
        sessions.setdefault(row["session"], row)
    session_rows = list(sessions.values())
    with_bg = [r for r in session_rows if r.get("new_background_db") is not None]
    diffs = [
        r["new_background_db"] - r["old_session_floor_db"]
        for r in with_bg
        if r.get("old_session_floor_db") is not None
    ]
    floor_diffs = [
        r["new_background_db"] - r["old_background_floor_db"]
        for r in rows
        if r.get("kind") in DECIDED_KINDS
        and r.get("new_background_db") is not None
        and r.get("old_background_floor_db") is not None
    ]
    by_family: dict[str, dict[str, list[float]]] = {}
    not_applicable: dict[str, int] = {}
    for row in rows:
        reading = row.get("reading") or {}
        if reading.get("missing") or "family" not in row:
            continue
        family = str(row.get("family"))
        slot = by_family.setdefault(family, {})
        if reading.get("not_applicable"):
            not_applicable[family] = not_applicable.get(family, 0) + 1
        for measure in ("foreground_minus_residual_db", "plain_minus_residual_db"):
            for frames, value in (reading.get(measure) or {}).items():
                if value is not None:
                    slot.setdefault(f"{measure}.{frames}", []).append(float(value))
    transitions: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("kind") not in DECIDED_KINDS or "old_decision" not in row:
            continue
        key = f"{row.get('old_decision')}->{row.get('new_decision')}"
        slot2 = transitions.setdefault(str(row["kind"]), {})
        slot2[key] = slot2.get(key, 0) + 1
    consistency: dict[str, int] = {}
    for row in rows:
        if "old_decision" in row and row.get("stored_decision") is not None:
            key = "same" if row["stored_decision"] == row["old_decision"] else "differs"
            consistency[key] = consistency.get(key, 0) + 1
    return {
        "recordings": len(rows),
        "sessions": len(session_rows),
        "sessions_with_background": len(with_bg),
        "fallback": {
            str(k): sum(1 for r in session_rows if (r.get("session_background") or {}).get("fallback") == k)
            for k in {(r.get("session_background") or {}).get("fallback") for r in session_rows}
        },
        "members_n": _quantiles([float((r.get("session_background") or {}).get("members_n", 0)) for r in session_rows]),
        "new_minus_old_session_floor_db": _quantiles(diffs),
        "new_minus_background_floor_db_non_speech": _quantiles(floor_diffs),
        "quality_by_family": {
            family: {name: _quantiles(values) for name, values in sorted(slot.items())}
            for family, slot in sorted(by_family.items())
        },
        "not_applicable_by_family": dict(sorted(not_applicable.items())),
        "decision_transitions": transitions,
        "old_decision_vs_stored": consistency,
        "errors": sum(1 for r in rows if r.get("error") or r.get("decision_error")),
    }


def main(argv: list[str] | None = None) -> int:
    """Measure, or summarise.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 on success, 2 on bad arguments.
    """
    args = build_parser().parse_args(argv)
    if args.command == "summarise":
        print(json.dumps(summarise(args.out_dir), indent=2, sort_keys=True))
        return 0
    if args.manifest is None or args.out_dir is None or not args.manifest.exists():
        print("ERROR: a manifest that exists and --out-dir are required", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
