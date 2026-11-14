"""Cross-node probe of the layered measures roots, on the shared filesystem they are meant for.

Three nodes at one commit share one measures root ``<shared>``; barriers and event logs live beside it in
``<shared>.events`` (so stamped hostnames never enter a measures record). Roles:

* ``write``: ``--probes`` rounds. Each round writes a fresh unit (the round number is one of its params), so each is a
  first-writer race; the writers meet at a barrier on the shared storage (``go-<round>/<name>``, released when
  ``--writers`` entries exist) and then write one probe each. After the last round every writer waits for the others
  and reads every unit back, requiring all writers' observations.
* ``hold``: node A takes the probe lease of a lease unit, beats for ``--hold`` seconds (more than ``--stale``), then
  SIGKILLs itself.
* ``break``: nodes B and C look for a live lease in every root and try to take it every second; a taker proves it,
  writes one probe's observations, holds it two bounds and releases it.
* ``readonly``: started with ``<shared>`` mounted read-only and ``--own`` writable. It waits for A's hold, logs
  ``peer_holds`` of the lease unit (the lease lies in a root it cannot write), then writes and falls back to ``--own``.
* ``verdict``: reads ``<shared>`` and the events and prints JSON with ``pass``.

``verdict`` passes when every writer read every unit back complete, no unit lacks a writer's probe, each written unit
has one ``unit.json``, nothing is quarantined, no take precedes the kill, the first take follows it within
``--stale`` plus ``--slack``, no two holds overlap, no holder lost its lease, ``readonly`` saw A's lease and wrote
only to its own root, three distinct hosts took part, and at least half the rounds overlapped. A round overlaps when
every writer had staged its observation file before any writer committed it: each writer's window runs from staging to
commit on the storage clock, and the windows have a common point. The commit (`StorageDir.link`) is held at a barrier on
the shared storage until all writers have staged, for `unit.json` and for the observation file alike. ``<checkout>`` is a clone at the commit under test::

    export PYTHONPATH=<checkout>/packages/senselab-core/src:<checkout>/packages/senselab/src:<checkout>/packages/senselab-worker/src
    S=<checkout>/scripts/probe_measures_cross_node.py
    python $S write <shared> --name A --probes 20 --writers 3 --stale 120      # nodes A, B, C
    python $S hold <shared> --stale 120 --hold 360                              # node A, with the write role
    python $S break <shared> --stale 120 --name B --until 1200                  # nodes B and C
    apptainer exec --bind <shared>:<shared>:ro <image> python $S readonly <shared> --own <own> --stale 120
    python $S verdict <shared> --stale 120 --slack 75 --hosts 3 --writers 3 --probes 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from senselab_core.kernel.identity import ModelIdentity
from senselab_core.runtime.locks import Lease, LeaseTimeout
from senselab_core.runtime.measures import UNIT_FILE, Build, MeasureRoots, MeasureUnit, PeerProbing, Provenance
from senselab_core.runtime.storage import StorageDir, StorePolicy
from senselab_core.testing.calibration_samples import UNIT, observation

LEASE_UNIT = UNIT.model_copy(update={"model": ModelIdentity(id="openai/whisper-base", identity="e" * 40)})
OBSERVATIONS_PER_PROBE = 3
COMMIT_POLL_S = 0.02
PROVENANCE = Provenance(
    job=os.environ.get("SLURM_JOB_ID"),
    run="cross-node",
    senselab=Build(version="probe"),
    profile="d" * 64,
    hardware=UNIT.hardware,
)


def _unit(round_: int) -> MeasureUnit:
    return UNIT.model_copy(update={"params": {"word_timestamps": False, "round": round_}})


def _roots(*roots: Path, stale: float) -> MeasureRoots:
    return MeasureRoots(roots, StorePolicy.load_default(), "group", os.geteuid(), stale_after_s=stale)


def _now(events: Path) -> float:
    with StorageDir.open_root(events, create=True) as directory:
        return directory.dir_now()


def _log(events: Path, name: str, **event: object) -> None:
    event.update(role=name, host=socket.gethostname(), storage_t=_now(events), local_t=time.time())
    with open(events / f"events-{name}.jsonl", "a") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _barrier(events: Path, label: str, name: str, parties: int, bound: float, poll: float) -> None:  # noqa: PLR0913 - the role's command-line options
    """Make `events/<label>/<name>` and wait until `parties` entries exist on the shared storage."""
    here = events / label
    here.mkdir(parents=True, exist_ok=True)
    (here / name).mkdir(exist_ok=True)
    deadline = time.monotonic() + bound
    while len(list(here.iterdir())) < parties:
        if time.monotonic() > deadline:
            raise TimeoutError(f"barrier {label}: fewer than {parties} entries")
        time.sleep(poll)


def _items(round_: int, writers: list[str]) -> set[str]:
    return {f"{w}-{round_}-{i}" for w in writers for i in range(OBSERVATIONS_PER_PROBE)}


class _CommitRendezvous:
    """Hold every writer's `StorageDir.link` (the commit of a staged file) until all writers have staged theirs.

    A writer enters `commit-<round>-<kind>/<name>` when its temp file is staged and links only once every writer has;
    `kind` is `unit` for the first-writer `unit.json` race and `obs` for the observation file. A writer that finds
    `unit.json` already committed never stages it, so that barrier gives up after `grace` seconds and goes on.
    """

    def __init__(self, events: Path, name: str, writers: int, grace: float, poll: float) -> None:
        self.events, self.name, self.writers, self.grace, self.poll = events, name, writers, grace, poll
        self.round = 0
        self.windows: dict[str, tuple[float, float]] = {}
        self._link = StorageDir.link

    def __enter__(self) -> "_CommitRendezvous":
        rendezvous = self

        def link(stage: StorageDir, name: str, dst: StorageDir, dst_name: str, *, grammar: re.Pattern[str]) -> None:
            kind = {UNIT_FILE: "unit", f"{rendezvous.name}-{rendezvous.round}.json": "obs"}.get(dst_name)
            if kind is None:
                rendezvous._link(stage, name, dst, dst_name, grammar=grammar)
                return
            staged = _now(rendezvous.events)
            try:
                _barrier(
                    rendezvous.events, f"commit-{rendezvous.round}-{kind}", rendezvous.name,
                    rendezvous.writers, rendezvous.grace, rendezvous.poll,
                )  # fmt: skip
            except TimeoutError:
                pass
            try:
                rendezvous._link(stage, name, dst, dst_name, grammar=grammar)
            finally:
                rendezvous.windows[kind] = (staged, _now(rendezvous.events))

        StorageDir.link = link  # type: ignore[assignment, method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        StorageDir.link = self._link  # type: ignore[method-assign]


def write(  # noqa: PLR0913 - the role's command-line options
    shared: Path, events: Path, name: str, probes: int, writers: int, stale: float, bound: float, poll: float
) -> None:
    """Write one probe of a fresh unit per round: stage, meet at a commit barrier, commit together; then read back."""
    roots = _roots(shared, stale=stale)
    with _CommitRendezvous(events, name, writers, grace=10.0, poll=COMMIT_POLL_S) as commits:
        for n in range(probes):
            _barrier(events, f"go-{n}", name, writers, bound, poll)
            commits.round, commits.windows = n, {}
            landed = roots.put_observations(
                _unit(n),
                f"{name}-{n}",
                [observation(probe=None, item=f"{name}-{n}-{i}") for i in range(OBSERVATIONS_PER_PROBE)],
                PROVENANCE,
            )
            start, end = commits.windows.get("obs", (None, None))
            _log(events, name, event="wrote", round=n, landed=landed is not None, start_t=start, end_t=end,
                 unit_window=commits.windows.get("unit"))  # fmt: skip
    _barrier(events, "done", name, writers, bound, poll)
    names = sorted(p.name for p in (events / "done").iterdir())
    missing = [n for n in range(probes) if _items(n, names) - {o.item for o in roots.observations(_unit(n))}]
    _log(events, name, event="read", writers=names, missing_rounds=missing)


def hold(shared: Path, events: Path, stale: float, fraction: float, seconds: float, poll: float) -> None:  # noqa: PLR0913 - the role's command-line options
    """Take the probe lease, beat for `seconds`, then SIGKILL this process while holding it."""
    lease = _roots(shared, stale=stale).probe_lease(LEASE_UNIT)
    if not isinstance(lease, Lease):
        _log(events, "A", event="no-lease", why=type(lease).__name__)
        return
    with lease, lease.keep_alive(fraction) as state:
        _log(events, "A", event="held")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(poll)
            _log(events, "A", event="beat", beats=state.beats, lost=state.lost, owns=lease.owns())
        _log(events, "A", event="kill", beats=state.beats, lost=state.lost)
        os.kill(os.getpid(), signal.SIGKILL)


def breaker(shared: Path, events: Path, stale: float, fraction: float, name: str, until: float, poll: float) -> None:  # noqa: PLR0913 - the role's command-line options
    """Look for a live lease and try to take it every `poll` seconds; a taker proves it, writes, holds two bounds."""
    roots = _roots(shared, stale=stale)
    deadline = time.monotonic() + until
    while time.monotonic() < deadline:
        lease = roots.probe_lease(LEASE_UNIT)
        if isinstance(lease, Lease):
            try:
                with lease, lease.keep_alive(fraction) as state:
                    _log(events, name, event="took", broke=[str(p) for p in lease.broke], proved=lease.prove())
                    roots.put_observations(
                        LEASE_UNIT, f"lease-{name}", [observation(probe=None, item=name)], PROVENANCE
                    )
                    time.sleep(2 * stale)
                    owned = lease.owns()
                _log(events, name, event="released", owns=owned, lost=state.lost)
                return
            except LeaseTimeout:
                pass
        _log(events, name, event="refused", peer=isinstance(lease, PeerProbing))
        time.sleep(poll)
    _log(events, name, event="gave-up")


def readonly(shared: Path, events: Path, own: Path, stale: float, until: float, poll: float) -> None:  # noqa: PLR0913 - the role's command-line options
    """During A's hold, with `shared` mounted read-only, see its lease, then write one probe and fall back to `own`."""
    roots = _roots(shared, own, stale=stale)
    deadline = time.monotonic() + until
    log = events / "events-A.jsonl"
    while not (log.exists() and '"held"' in log.read_text()) and time.monotonic() < deadline:
        time.sleep(poll)
    sees = roots.peer_holds(LEASE_UNIT)
    before = sorted(str(p) for p in shared.rglob("*"))
    landed = roots.put_observations(_unit(-1), "readonly-0", [observation(probe=None, item="readonly")], PROVENANCE)
    _log(
        events,
        "readonly",
        event="wrote",
        peer_holds=sees,
        in_own=landed is not None and own in landed.parents,
        shared_unchanged=before == sorted(str(p) for p in shared.rglob("*")),
        write_root_is_own=roots.write_root == own,
        read_back=len(roots.observations(_unit(-1))) > 0,
    )


def _spans_overlap(spans: list[tuple[float, float]]) -> bool:
    return max(s for s, _ in spans) <= min(e for _, e in spans)


def _check_writes(shared: Path, events: list[dict[str, Any]], writers: int, probes: int) -> dict[str, Any]:
    wrote = [e for e in events if e["event"] == "wrote" and e["role"] != "readonly"]
    reads = [e for e in events if e["event"] == "read"]
    rounds = [[(e["start_t"], e["end_t"]) for e in wrote if e["round"] == n] for n in range(probes)]
    unit_files = list((shared / "measures@1" / "units").rglob("unit.json"))
    return {
        "probes_written": len(wrote),
        "all_landed": all(e["landed"] for e in wrote),
        "readers": len(reads),
        "rounds_missing_at_a_reader": sorted({n for e in reads for n in e["missing_rounds"]}),
        "release_spread_s": [round(max(s for s, _ in r) - min(s for s, _ in r), 3) for r in rounds if r],
        "overlapping_rounds": sum(len(r) == writers and _spans_overlap(r) for r in rounds),
        "unit_files": len(unit_files),
        "quarantined": len(list((shared / "measures@1" / "quarantine").glob("*"))),
        "complete": len(wrote) == writers * probes and len(reads) == writers,
    }


def verdict(shared: Path, events_dir: Path, args: argparse.Namespace) -> bool:
    """Print the probe's verdict as JSON; return whether it passed."""
    events = [
        json.loads(line) for p in sorted(events_dir.glob("events-*.jsonl")) for line in p.read_text().splitlines()
    ]
    kill = next((e["storage_t"] for e in events if e["role"] == "A" and e["event"] == "kill"), None)
    held = next((e["storage_t"] for e in events if e["role"] == "A" and e["event"] == "held"), None)
    spans, unreleased = [], 0
    for t in (e for e in events if e["event"] == "took"):
        released = next((r["storage_t"] for r in events if r["role"] == t["role"] and r["event"] == "released"), None)
        if released is None:
            unreleased += 1
        else:
            spans.append((t["storage_t"], released))
    spans.sort()
    took = sorted(e["storage_t"] for e in events if e["event"] == "took")
    overlaps = sum(later[0] < earlier[1] for earlier, later in zip(spans, spans[1:]))
    early = [t for t in took if kill is not None and t < kill]
    lost = [e for e in events if e.get("lost")]
    fallback: dict[str, Any] = next((e for e in events if e["role"] == "readonly" and e["event"] == "wrote"), {})
    writes = _check_writes(shared, events, args.writers, args.probes)
    seen_hosts = sorted({e["host"] for e in events})
    units_expected = args.probes + 1
    passed = bool(
        kill is not None and held is not None and bool(took) and len(seen_hosts) == args.hosts and not unreleased
        and not early and not lost and not overlaps and kill - held > args.stale and took[0] - kill <= args.stale + args.slack
        and bool(writes["complete"]) and writes["all_landed"] and not writes["rounds_missing_at_a_reader"]
        and writes["unit_files"] == units_expected and not writes["quarantined"]
        and writes["overlapping_rounds"] * 2 >= args.probes
        and bool(fallback) and fallback["peer_holds"] and fallback["in_own"] and fallback["shared_unchanged"]
        and fallback["write_root_is_own"] and fallback["read_back"]
    )  # fmt: skip
    print(
        json.dumps(
            {
                "hosts": len(seen_hosts),
                "take_after_kill_s": (took[0] - kill) if took and kill is not None else None,
                "takes_before_kill": len(early),
                "overlapping_holds": overlaps,
                "lost": len(lost),
                "writes": writes,
                "readonly_fallback": fallback,
                "pass": passed,
            }
        )
    )
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run one role; `verdict` exits 1 when the probe failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("role", choices=["write", "hold", "break", "readonly", "verdict"])
    parser.add_argument("shared", type=Path)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--name", default="B")
    parser.add_argument("--own", type=Path, default=None)
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--hosts", type=int, default=3)
    parser.add_argument("--writers", type=int, default=3)
    parser.add_argument("--stale", type=float, required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--hold", type=float, default=360.0)
    parser.add_argument("--until", type=float, default=1200.0)
    parser.add_argument("--slack", type=float, default=75.0)
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--barrier-poll", type=float, default=0.25)
    args = parser.parse_args(argv)
    events = args.events or args.shared.with_name(args.shared.name + ".events")
    if args.role == "write":
        write(args.shared, events, args.name, args.probes, args.writers, args.stale, args.until, args.barrier_poll)
    elif args.role == "hold":
        hold(args.shared, events, args.stale, args.fraction, args.hold, args.poll)
    elif args.role == "break":
        breaker(args.shared, events, args.stale, args.fraction, args.name, args.until, args.poll)
    elif args.role == "readonly":
        if args.own is None:
            parser.error("readonly needs --own")
        readonly(args.shared, events, args.own, args.stale, args.until, args.poll)
    else:
        return 0 if verdict(args.shared, events, args) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
