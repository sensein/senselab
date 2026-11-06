"""Cross-node queue probe: concurrent claimers on several nodes commit every queued item exactly once.

One node puts ``--items`` fake plan fragments into the queue ``--queue`` under ``<dir>/queues``. Each node then runs
``claim`` with ``--procs`` claimer processes; a claimer claims one item at a time, waits ``--work`` seconds and
commits it, until the queue has no unfinished item. With ``--crash-after K`` the first claimer of the node SIGKILLs
itself while holding its K-th claim, so another claimer breaks the stale claim and commits the item. Every claim and
outcome is appended to ``<dir>/events/<host>-<pid>.jsonl``.

``verdict`` passes when every item put is committed exactly once (one ``done`` entry, one ``committed`` event, no
``dead`` entry), each claimer's claims within a segment follow the segment's order (a claim that broke a stale claim
excepted), and at least ``--hosts`` distinct hosts claimed; it counts duplicate commits refused and claims lost.

The directory is the positional argument, or ``SENSELAB_QUEUE_PROBE_DIR`` when it is omitted. Run (``<checkout>`` is
a clone at the commit under test, ``<shared>`` a fresh directory on the shared filesystem; the three ``claim`` runs on
three distinct nodes)::

    export PYTHONPATH=<checkout>/packages/senselab-core/src:<checkout>/packages/senselab/src:<checkout>/packages/senselab-worker/src
    # once, on any node
    <checkout>/.venv/bin/python <checkout>/scripts/probe_queue_cross_node.py put <shared> --items 10000 --profile orcd
    # on each of three nodes
    <checkout>/.venv/bin/python <checkout>/scripts/probe_queue_cross_node.py claim <shared> --procs 8 --profile orcd --crash-after 100
    # once all three have ended; prints JSON with "pass" and the hosts
    <checkout>/.venv/bin/python <checkout>/scripts/probe_queue_cross_node.py verdict <shared> --hosts 3
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.execution.executor import JobRef, PlanFragment
from senselab_core.runtime.execution.queue import ITEM_TAG, Queue, QueueSet
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_worker.digest import identity_digest

DIR_VARIABLE = "SENSELAB_QUEUE_PROBE_DIR"
PLAN = "0" * 64
PUT_BATCH = 500


def _queue(root: Path, args: argparse.Namespace) -> Queue:
    override: dict[str, Any] = {}
    if args.stale is not None:
        override["lease"] = {
            "stale_after_s": args.stale,
            "renew_fraction": args.fraction,
            "attribute_delay_s": args.delay,
        }
    return QueueSet(root / "queues", ExecutionProfile.load(args.profile, override=override)).queue(args.queue)


def _fragments(n: int) -> list[PlanFragment]:
    placement = NodePlacement(
        backend="probe",
        backend_version="1",
        backend_params={},
        stack="probe",
        env="host",
        device="cpu",
        batch_size=1,
        resource_class="cpu",
    )
    job = (JobRef(plan_hash=PLAN, node="probe", element=(0,)),)
    return [
        PlanFragment(plan_hash=PLAN, node="probe", admission=f"probe-{i}", jobs=job, inputs=({},), placement=placement)
        for i in range(n)
    ]


def put(root: Path, args: argparse.Namespace) -> None:
    """Put `--items` fragments and record their digests in `<dir>/items.json`."""
    queue = _queue(root, args)
    fragments = _fragments(args.items)
    for start in range(0, len(fragments), PUT_BATCH):
        queue.put_all(fragments[start : start + PUT_BATCH])
    digests = [identity_digest(ITEM_TAG, f.model_dump(mode="json")) for f in fragments]
    (root / "items.json").write_text(json.dumps({"queue": args.queue, "items": digests}))


def _claimer(root: str, args: argparse.Namespace, index: int) -> None:
    """Claim and commit one item at a time until the queue has no unfinished item or `--until` seconds pass."""
    queue = _queue(Path(root), args)
    events_dir = Path(root) / "events"
    events_dir.mkdir(exist_ok=True)
    host, pid = socket.gethostname(), os.getpid()
    worker = f"{host}-{pid}"
    deadline = time.monotonic() + args.until
    claims = 0
    with open(events_dir / f"{worker}.jsonl", "a", buffering=1) as log:

        def record(**event: object) -> None:
            log.write(json.dumps({**event, "host": host, "pid": pid, "t": time.time()}) + "\n")

        while time.monotonic() < deadline:
            claim = queue.claim(worker, 1)
            if claim is None:
                if queue.depth() == 0:
                    break
                time.sleep(args.poll)
                continue
            (digest,) = claim.items
            claims += 1
            record(event="claim", item=digest, segment=claim.segment(digest), broke=claim.broke(digest))
            if index == 0 and claims == args.crash_after:
                record(event="crash", item=digest)
                os.kill(pid, signal.SIGKILL)
            time.sleep(args.work)
            claim.commit()
            for item, outcome in claim.outcomes.items():
                record(event=outcome, item=item)
        record(event="end", claims=claims)


def claim(root: Path, args: argparse.Namespace) -> None:
    """Run `--procs` claimer processes on this node and wait for them."""
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_claimer, args=(str(root), args, i)) for i in range(args.procs)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def _positions(queue_dir: Path) -> dict[str, tuple[int, int]]:
    """Return each item's segment and line in the index, at its first line."""
    positions: dict[str, tuple[int, int]] = {}
    for path in sorted((queue_dir / "segments").glob("*.jsonl")):
        seq = int(path.name.split(".")[0])
        for line, text in enumerate(path.read_text().splitlines()):
            try:
                positions.setdefault(json.loads(text)["item"], (seq, line))
            except (ValueError, KeyError):
                continue
    return positions


def verdict(root: Path, args: argparse.Namespace) -> bool:
    """Print the probe's verdict as JSON; return whether it passed."""
    put_record = json.loads((root / "items.json").read_text())
    items = set(put_record["items"])
    queue_dir = _queue(root, argparse.Namespace(**{**vars(args), "queue": put_record["queue"]})).path
    events = [
        json.loads(line) for p in sorted((root / "events").glob("*.jsonl")) for line in p.read_text().splitlines()
    ]
    kinds = collections.Counter(e["event"] for e in events)
    committed = collections.Counter(e["item"] for e in events if e["event"] == "committed")
    done = {p.name.removesuffix(".json") for p in (queue_dir / "done").glob("*.json")}
    dead = {p.name.removesuffix(".json") for p in (queue_dir / "dead").glob("*.json")}
    positions = _positions(queue_dir)
    inversions = 0
    last: dict[tuple[str, int, int], int] = {}
    for event in events:
        if event["event"] != "claim" or event.get("broke"):
            continue
        seq, line = positions[event["item"]]
        key = (event["host"], event["pid"], seq)
        inversions += line < last.get(key, -1)
        last[key] = line
    hosts = sorted({e["host"] for e in events if e["event"] == "claim"})
    once = sum(committed[i] == 1 for i in items)
    report = {
        "items": len(items),
        "committed_once": once,
        "committed_twice_or_more": sum(committed[i] > 1 for i in items),
        "not_committed": sorted(items - set(committed))[:10],
        "done_entries": len(done & items),
        "dead_entries": len(dead),
        "duplicate_commits": kinds["duplicate"],
        "lost_claims": kinds["lost"],
        "claims": kinds["claim"],
        "reclaims": sum(1 for e in events if e["event"] == "claim" and e.get("broke")),
        "crashes": kinds["crash"],
        "order_inversions": inversions,
        "hosts": hosts,
        "distinct_hosts": len(hosts),
    }
    passed = once == len(items) == len(done & items) and not dead and not inversions and len(hosts) >= args.hosts
    report["pass"] = passed
    print(json.dumps(report))
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run one mode; `verdict` exits 1 when the probe failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["put", "claim", "verdict"])
    parser.add_argument("dir", type=Path, nargs="?", default=os.environ.get(DIR_VARIABLE))
    parser.add_argument("--queue", default="cpu")
    parser.add_argument("--profile", default="laptop")
    parser.add_argument("--items", type=int, default=10_000)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--work", type=float, default=0.0)
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--until", type=float, default=3600.0)
    parser.add_argument("--crash-after", type=int, default=0)
    parser.add_argument("--stale", type=float, default=None)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--hosts", type=int, default=3)
    args = parser.parse_args(argv)
    if args.dir is None:
        parser.error(f"give the shared directory, or set {DIR_VARIABLE}")
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "put":
        put(root, args)
    elif args.mode == "claim":
        claim(root, args)
    else:
        return 0 if verdict(root, args) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
