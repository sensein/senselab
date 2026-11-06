"""Cross-node lease probe: a beating holder is never broken, and a SIGKILLed one is broken within one bound.

Three roles share one directory on the filesystem under test, each on its own node, at one commit. Node A takes the
lease on ``<dir>/resource`` and beats for ``--hold`` seconds (more than ``--stale``), then SIGKILLs itself. Nodes B and
C try to take it every second; a taker holds it, beating, for two bounds and releases it. ``verdict`` passes when no
take precedes the kill, the first take follows the kill within ``--stale`` plus ``--slack``, no two holds overlap,
and no holder lost its lease. Every event is stamped on the storage clock (a probe directory's mtime) and appended
to ``<dir>/events-<role>.jsonl``.

The directory is the positional argument, or ``SENSELAB_LEASE_PROBE_DIR`` when it is omitted. Run (``<checkout>`` is
a clone at the commit under test, ``<shared>`` a fresh directory on the shared filesystem; A, B and C on three
distinct nodes, B and C started within one renewal interval of A)::

    export PYTHONPATH=<checkout>/packages/senselab-core/src:<checkout>/packages/senselab/src:<checkout>/packages/senselab-worker/src
    # node A
    <checkout>/.venv/bin/python <checkout>/scripts/probe_lease_cross_node.py hold <shared> --stale 120 --fraction 0.25 --hold 360
    # nodes B and C
    <checkout>/.venv/bin/python <checkout>/scripts/probe_lease_cross_node.py break <shared> --stale 120 --fraction 0.25 --name B --until 1200
    <checkout>/.venv/bin/python <checkout>/scripts/probe_lease_cross_node.py break <shared> --stale 120 --fraction 0.25 --name C --until 1200
    # once all three have ended; prints JSON with "pass" and the three hosts
    <checkout>/.venv/bin/python <checkout>/scripts/probe_lease_cross_node.py verdict <shared> --stale 120 --slack 75

``--fraction × --stale`` plus the clients' attribute-cache delay must stay below ``--stale``.
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

from senselab_core.runtime.locks import Lease, LeaseTimeout
from senselab_core.runtime.storage import StorageDir

DIR_VARIABLE = "SENSELAB_LEASE_PROBE_DIR"
EVENT_POLL_S = 5.0
TAKE_POLL_S = 1.0


def _now(root: Path) -> float:
    with StorageDir.open_root(root) as directory:
        return directory.dir_now()


def _log(root: Path, role: str, **event: object) -> None:
    event.update(role=role, host=socket.gethostname(), storage_t=_now(root), local_t=time.time())
    with open(root / f"events-{role}.jsonl", "a") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def hold(root: Path, stale: float, fraction: float, seconds: float) -> None:
    """Take the lease, beat for `seconds`, then SIGKILL this process while holding it."""
    with Lease(root / "resource", timeout=60, stale_after=stale) as lease, lease.keep_alive(fraction) as state:
        _log(root, "A", event="held")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(EVENT_POLL_S)
            _log(root, "A", event="beat", beats=state.beats, lost=state.lost, owns=lease.owns())
        _log(root, "A", event="kill", beats=state.beats, lost=state.lost)
        os.kill(os.getpid(), signal.SIGKILL)


def breaker(root: Path, stale: float, fraction: float, name: str, until: float) -> None:
    """Try to take the lease every second until `until` seconds pass; hold a taken lease, beating, for two bounds."""
    deadline = time.monotonic() + until
    while time.monotonic() < deadline:
        try:
            with Lease(root / "resource", timeout=0, stale_after=stale) as lease, lease.keep_alive(fraction) as state:
                _log(root, name, event="took", broke=[str(p) for p in lease.broke])
                time.sleep(2 * stale)
                _log(root, name, event="released", owns=lease.owns(), lost=state.lost)
                return
        except LeaseTimeout:
            _log(root, name, event="refused")
        time.sleep(TAKE_POLL_S)
    _log(root, name, event="gave-up")


def verdict(root: Path, stale: float, slack: float) -> bool:
    """Print the probe's verdict as JSON; return whether it passed."""
    events = [json.loads(line) for p in sorted(root.glob("events-*.jsonl")) for line in p.read_text().splitlines()]
    kill = next(e["storage_t"] for e in events if e["role"] == "A" and e["event"] == "kill")
    held = next(e["storage_t"] for e in events if e["role"] == "A" and e["event"] == "held")
    took = sorted(e["storage_t"] for e in events if e["event"] == "took")
    spans = sorted(
        (t["storage_t"], next(r["storage_t"] for r in events if r["role"] == t["role"] and r["event"] == "released"))
        for t in events
        if t["event"] == "took"
    )
    overlaps = sum(later[0] < earlier[1] for earlier, later in zip(spans, spans[1:]))
    early = [t for t in took if t < kill]
    lost = [e for e in events if e.get("lost")]
    passed = bool(took) and not early and not lost and not overlaps and kill - held > stale
    passed = passed and took[0] - kill <= stale + slack
    report = {
        "hosts": sorted({e["host"] for e in events}),
        "held_s": kill - held,
        "take_after_kill_s": (took[0] - kill) if took else None,
        "takes_before_kill": len(early),
        "overlapping_holds": overlaps,
        "lost": len(lost),
        "takers": len(took),
        "pass": passed,
    }
    print(json.dumps(report))
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run one role; `verdict` exits 1 when the probe failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("role", choices=["hold", "break", "verdict"])
    parser.add_argument("dir", type=Path, nargs="?", default=os.environ.get(DIR_VARIABLE))
    parser.add_argument("--stale", type=float, default=120.0)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--hold", type=float, default=360.0)
    parser.add_argument("--name", default="B")
    parser.add_argument("--until", type=float, default=1200.0)
    parser.add_argument("--slack", type=float, default=75.0)
    args = parser.parse_args(argv)
    if args.dir is None:
        parser.error(f"give the shared directory, or set {DIR_VARIABLE}")
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.role == "hold":
        hold(root, args.stale, args.fraction, args.hold)
    elif args.role == "break":
        breaker(root, args.stale, args.fraction, args.name, args.until)
    else:
        return 0 if verdict(root, args.stale, args.slack) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
