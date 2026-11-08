"""Cross-node probe of the layered measures roots, on the shared filesystem they are meant for.

Three nodes at one commit share one measures root ``<shared>``. Roles, each logging to ``<events>/events-<name>.jsonl``
(``<shared>.events``, beside the root, so the stamped hostnames never enter a measures record):

* ``write``: wait for ``--start-at``, then ``--probes`` times write one probe's observations of one unit, concurrently
  with the other nodes.
* ``hold``: node A takes the probe lease of a second unit, beats for ``--hold`` seconds (more than ``--stale``), then
  SIGKILLs itself.
* ``break``: nodes B and C look for a live lease in every root and try to take it every second; a taker proves it,
  writes one probe's observations, holds it two bounds and releases it.
* ``readonly``: one process with ``<shared>`` mounted read-only and ``--own`` writable falls back to its own root.
* ``verdict``: reads ``<shared>`` and the events and prints JSON with ``pass``.

``verdict`` passes when every written probe is read back exactly once from every node, each unit has one
``unit.json``, nothing is quarantined, no take precedes the kill, the first take follows it within ``--stale`` plus
``--slack``, no two holds overlap, no holder lost its lease, the ``readonly`` process wrote only to its own root,
and three distinct hosts took part. Run ``<checkout>`` at the commit under test, ``<shared>`` a fresh directory::

    export PYTHONPATH=<checkout>/packages/senselab-core/src:<checkout>/packages/senselab/src:<checkout>/packages/senselab-worker/src
    S=<checkout>/scripts/probe_measures_cross_node.py
    python $S write <shared> --name A --start-at <epoch>        # on each of three nodes, A B C
    python $S hold <shared> --stale 120 --hold 360               # node A, concurrently with the break roles
    python $S break <shared> --stale 120 --name B --until 1200   # nodes B and C
    apptainer exec --bind <shared>:<shared>:ro <image> python $S readonly <shared> --own <own>
    python $S verdict <shared> --stale 120 --slack 75 --hosts 3 --writers 3 --probes <n>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

from senselab_core.kernel.identity import ModelIdentity
from senselab_core.runtime.calibration import Observation
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.measures import Build, MeasureRoots, Provenance
from senselab_core.runtime.storage import StorageDir, StorePolicy
from senselab_core.testing.calibration_samples import UNIT, observation

WRITE_UNIT = UNIT
LEASE_UNIT = UNIT.model_copy(update={"model": ModelIdentity(id="openai/whisper-base", identity="e" * 40)})
EVENT_POLL_S = 5.0
TAKE_POLL_S = 1.0
OBSERVATIONS_PER_PROBE = 3
PROVENANCE = Provenance(
    job=os.environ.get("SLURM_JOB_ID"),
    run="cross-node",
    senselab=Build(version="probe"),
    profile="d" * 64,
    hardware=UNIT.hardware,
)


def _observation(item: str) -> Observation:
    return observation(probe=None, item=item)


def _roots(*roots: Path, stale: float) -> MeasureRoots:
    return MeasureRoots(roots, StorePolicy.load_default(), "group", os.geteuid(), stale_after_s=stale)


def _log(events: Path, name: str, **event: object) -> None:
    with StorageDir.open_root(events, create=True) as directory:
        stamp = directory.dir_now()
    event.update(role=name, host=socket.gethostname(), storage_t=stamp, local_t=time.time())
    with open(events / f"events-{name}.jsonl", "a") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write(shared: Path, events: Path, name: str, probes: int, start_at: float) -> None:
    """Write `probes` probes of one unit, starting together at `start_at` (epoch seconds)."""
    roots = _roots(shared, stale=120)
    while time.time() < start_at:
        time.sleep(0.05)
    for n in range(probes):
        probe = f"{name}-{n}"
        landed = roots.put_observations(
            WRITE_UNIT, probe, [_observation(f"{probe}-{i}") for i in range(OBSERVATIONS_PER_PROBE)], PROVENANCE
        )
        _log(events, name, event="wrote", probe=probe, landed=landed is not None)


def hold(shared: Path, events: Path, stale: float, fraction: float, seconds: float) -> None:
    """Take the probe lease, beat for `seconds`, then SIGKILL this process while holding it."""
    lease = _roots(shared, stale=stale).probe_lease(LEASE_UNIT)
    if lease is None:
        _log(events, "A", event="no-lease")
        return
    with lease, lease.keep_alive(fraction) as state:
        _log(events, "A", event="held")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(EVENT_POLL_S)
            _log(events, "A", event="beat", beats=state.beats, lost=state.lost, owns=lease.owns())
        _log(events, "A", event="kill", beats=state.beats, lost=state.lost)
        os.kill(os.getpid(), signal.SIGKILL)


def breaker(shared: Path, events: Path, stale: float, fraction: float, name: str, until: float) -> None:  # noqa: PLR0913 - the role's command-line options
    """Look for a live lease and try to take it every second; a taker proves it, writes, holds two bounds, releases."""
    roots = _roots(shared, stale=stale)
    deadline = time.monotonic() + until
    while time.monotonic() < deadline:
        lease = None if roots.peer_holds(LEASE_UNIT) else roots.probe_lease(LEASE_UNIT)
        if lease is None:
            _log(events, name, event="refused")
            time.sleep(TAKE_POLL_S)
            continue
        try:
            with lease, lease.keep_alive(fraction) as state:
                _log(events, name, event="took", broke=[str(p) for p in lease.broke], proved=lease.prove())
                roots.put_observations(LEASE_UNIT, f"lease-{name}", [_observation(name)], PROVENANCE)
                time.sleep(2 * stale)
                owned = lease.owns()
            _log(events, name, event="released", owns=owned, lost=state.lost)
            return
        except LeaseTimeout:
            _log(events, name, event="refused")
            time.sleep(TAKE_POLL_S)
    _log(events, name, event="gave-up")


def readonly(shared: Path, events: Path, own: Path) -> None:
    """With `shared` mounted read-only, write one probe and record where it landed."""
    before = sorted(str(p) for p in shared.rglob("*"))
    roots = _roots(shared, own, stale=120)
    landed = roots.put_observations(WRITE_UNIT, "readonly-0", [_observation("readonly")], PROVENANCE)
    after = sorted(str(p) for p in shared.rglob("*"))
    _log(
        events,
        "readonly",
        event="wrote",
        in_own=landed is not None and own in landed.parents,
        shared_unchanged=before == after,
        write_root_is_own=roots.write_root == own,
        read_back=len(roots.observations(WRITE_UNIT)) > 0,
    )


def _check_writes(shared: Path, events: list[dict], writers: int, probes: int) -> dict[str, object]:
    roots = _roots(shared, stale=120)
    items = [o.item for o in roots.observations(WRITE_UNIT)]
    expected = {
        f"{e['probe']}-{i}"
        for e in events
        if e["event"] == "wrote" and e["role"] != "readonly"
        for i in range(OBSERVATIONS_PER_PROBE)
    }
    units = list((shared / "measures@1" / "units").rglob("unit.json"))
    quarantine = list((shared / "measures@1" / "quarantine").glob("*"))
    return {
        "probes_written": len(expected) // OBSERVATIONS_PER_PROBE,
        "observations_missing": len(expected - set(items)),
        "observations_duplicated": len(items) - len(set(items)),
        "units": len(units),
        "quarantined": len(quarantine),
        "complete": len(expected) == writers * probes * OBSERVATIONS_PER_PROBE and expected <= set(items),
    }


def verdict(shared: Path, events_dir: Path, stale: float, slack: float, hosts: int, writers: int, probes: int) -> bool:  # noqa: PLR0913 - the role's command-line options
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
    fallback: dict = next((e for e in events if e["role"] == "readonly" and e["event"] == "wrote"), {})
    writes = _check_writes(shared, events, writers, probes)
    seen_hosts = sorted({e["host"] for e in events})
    passed = (
        kill is not None and held is not None and bool(took) and len(seen_hosts) == hosts and not unreleased and not early
        and not lost and not overlaps and kill - held > stale and took[0] - kill <= stale + slack
        and bool(writes["complete"]) and not writes["observations_duplicated"] and writes["units"] == 2
        and not writes["quarantined"]
        and bool(fallback) and fallback["in_own"] and fallback["shared_unchanged"] and fallback["write_root_is_own"]
        and fallback["read_back"]
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
    parser.add_argument("--start-at", type=float, default=0.0)
    parser.add_argument("--stale", type=float, default=120.0)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--hold", type=float, default=360.0)
    parser.add_argument("--until", type=float, default=1200.0)
    parser.add_argument("--slack", type=float, default=75.0)
    args = parser.parse_args(argv)
    events = args.events or args.shared.with_name(args.shared.name + ".events")
    if args.role == "write":
        write(args.shared, events, args.name, args.probes, args.start_at)
    elif args.role == "hold":
        hold(args.shared, events, args.stale, args.fraction, args.hold)
    elif args.role == "break":
        breaker(args.shared, events, args.stale, args.fraction, args.name, args.until)
    elif args.role == "readonly":
        if args.own is None:
            parser.error("readonly needs --own")
        readonly(args.shared, events, args.own)
    else:
        return 0 if verdict(args.shared, events, args.stale, args.slack, args.hosts, args.writers, args.probes) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
