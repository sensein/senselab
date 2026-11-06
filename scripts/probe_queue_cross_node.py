"""Cross-node queue probe: concurrent claimers on several nodes give every queued item exactly one `done` outcome.

``put`` puts ``--items`` fake plan fragments into queue ``--queue`` under ``<dir>/queues``, with segments of
``--segment-items`` slots, so a run crosses and retires many segments. With ``--crash-after K`` a first putter is
SIGKILLed at its K-th slot (its item written, its slot not taken), and a second put and a reconcile repair it.

``claim`` runs ``--procs`` claimer processes on this node. A claimer claims one item at a time, waits ``--work``
seconds and commits it, until the queue has no item without an outcome. With ``--crash-after K``, claimer 0 of the
node is disturbed at its K-th claim, as ``--crash-at`` says: ``hold`` SIGKILLs it while it holds the claim, ``mark``
SIGKILLs it between its outcome link and its slot's mark, and ``stall`` makes its outcome link, after the proof,
sleep ``--stall`` seconds (longer than the stale bound), so another claimer breaks the claim and commits first.
Every claim and outcome is appended to ``<dir>/events/<host>-<pid>.jsonl``.

``verdict`` passes when every item put has exactly one outcome, `done`, with at most one ``committed`` event (none
only for an item whose committer was killed after its link); no item is dead; and at least ``--hosts`` distinct hosts
claimed. Claim order is compared per claimer by ``(segment, slot)`` across segments: a later claim of an earlier
slot is explained when that item was claimed before (a crash, stall or release) or the claim broke a stale claim;
the claimer's back-offs (items found under another live claim) are reported beside the rest, which must not exceed
them. Duplicate commits refused and claims lost are counted.

The directory is the positional argument, or ``SENSELAB_QUEUE_PROBE_DIR`` when it is omitted. Run (``<checkout>`` is
a clone at the commit under test, ``<shared>`` a fresh directory on the shared filesystem; the three ``claim`` runs on
three distinct nodes; the same ``--segment-items`` and lease flags on every mode)::

    export PYTHONPATH=<checkout>/packages/senselab-core/src:<checkout>/packages/senselab/src:<checkout>/packages/senselab-worker/src
    P="<checkout>/.venv/bin/python <checkout>/scripts/probe_queue_cross_node.py"
    L="--profile orcd --segment-items 64 --stale 150 --fraction 0.25 --delay 60"
    # once, on any node
    $P put <shared> --items 10000 --crash-after 500 $L
    # on three nodes, one disturbance each
    $P claim <shared> --procs 8 --crash-after 100 --crash-at hold $L
    $P claim <shared> --procs 8 --crash-after 150 --crash-at mark $L
    $P claim <shared> --procs 8 --crash-after 200 --crash-at stall --stall 200 $L
    # once all three have ended; prints JSON with "pass"
    $P verdict <shared> --hosts 3 $L
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


def _queues(root: Path, args: argparse.Namespace) -> QueueSet:
    override: dict[str, Any] = {"pool": {"segment_items": args.segment_items}}
    if args.stale is not None:
        override["lease"] = {
            "stale_after_s": args.stale,
            "renew_fraction": args.fraction,
            "attribute_delay_s": args.delay,
        }
    return QueueSet(root / "queues", ExecutionProfile.load(args.profile, override=override))


def _queue(root: Path, args: argparse.Namespace) -> Queue:
    return _queues(root, args).queue(args.queue)


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


def _kill_at(function: str, prefix: str, after: int = 1) -> None:
    """SIGKILL this process at its `after`-th `os.<function>` call on a name starting with `prefix`."""
    real, seen = getattr(os, function), [0]

    def killed(*a: Any, **k: Any) -> Any:  # noqa: ANN401
        if any(isinstance(x, str) and os.path.basename(x).startswith(prefix) for x in a[:2]):
            seen[0] += 1
            if seen[0] >= after:
                os.kill(os.getpid(), signal.SIGKILL)
        return real(*a, **k)

    setattr(os, function, killed)


def _put_all(root: str, args: argparse.Namespace, crash_after: int) -> None:
    if crash_after:
        _kill_at("symlink", "slot-", crash_after)
    queue = _queue(Path(root), args)
    fragments = _fragments(args.items)
    for start in range(0, len(fragments), PUT_BATCH):
        queue.put_all(fragments[start : start + PUT_BATCH])


def put(root: Path, args: argparse.Namespace) -> None:
    """Record the items in `<dir>/items.json`, put them (a first putter killed at `--crash-after`), and repair."""
    digests = [identity_digest(ITEM_TAG, f.model_dump(mode="json")) for f in _fragments(args.items)]
    (root / "items.json").write_text(json.dumps({"queue": args.queue, "items": digests}))
    if args.crash_after:
        child = multiprocessing.get_context("spawn").Process(target=_put_all, args=(str(root), args, args.crash_after))
        child.start()
        child.join()
        print(json.dumps({"crashed_putter_exit": child.exitcode, "reconciled": _queues(root, args).reconcile()}))
    _put_all(str(root), args, 0)


def _disturb(args: argparse.Namespace) -> None:
    """Install claimer 0's disturbance for its next commit."""
    if args.crash_at == "mark":
        _kill_at("open", "res-")
    elif args.crash_at == "stall":
        real, done, real_mkdir = os.link, [False], os.mkdir

        def mkdir(path: Any, *a: Any, **k: Any) -> None:  # noqa: ANN401
            if os.path.basename(os.fsdecode(path)).startswith("beat-"):
                time.sleep(args.stall)
            real_mkdir(path, *a, **k)

        os.mkdir = mkdir

        def link(*a: Any, **k: Any) -> None:  # noqa: ANN401
            if not done[0]:
                done[0] = True
                time.sleep(args.stall)
            real(*a, **k)

        os.link = link


def _claimer(root: str, args: argparse.Namespace, index: int) -> None:
    """Claim and commit one item at a time until no item lacks an outcome or `--until` seconds pass."""
    queue = _queue(Path(root), args)
    events_dir = Path(root) / "events"
    events_dir.mkdir(exist_ok=True)
    host, pid = socket.gethostname(), os.getpid()
    worker = f"{host}-{pid}"
    deadline, claims = time.monotonic() + args.until, 0
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
            record(event="claim", item=digest, slot=list(claim.slot(digest)), broke=claim.broke(digest))
            if index == 0 and claims == args.crash_after:
                record(event="disturb", item=digest, kind=args.crash_at)
                if args.crash_at == "hold":
                    os.kill(pid, signal.SIGKILL)
                _disturb(args)
            time.sleep(args.work)
            claim.commit()
            for item, outcome in claim.outcomes.items():
                record(event=outcome, item=item)
        record(event="end", claims=claims, backoffs=queue.backoffs)


def claim(root: Path, args: argparse.Namespace) -> None:
    """Run `--procs` claimer processes on this node and wait for them."""
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_claimer, args=(str(root), args, i)) for i in range(args.procs)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def _order(events: list[dict[str, Any]]) -> tuple[int, int, dict[str, int]]:
    """Return the explained and unexplained order inversions, and the claimers whose unexplained ones pass back-offs."""
    claimed: set[str] = set()
    last: dict[tuple[str, int], tuple[int, int]] = {}
    explained = 0
    unexplained: collections.Counter[tuple[str, int]] = collections.Counter()
    backoffs = {(e["host"], e["pid"]): e.get("backoffs", 0) for e in events if e["event"] == "end"}
    for event in sorted((e for e in events if e["event"] == "claim"), key=lambda e: e["t"]):
        key, slot = (event["host"], event["pid"]), tuple(event["slot"])
        if key in last and slot < last[key]:
            if event["item"] in claimed or event["broke"]:
                explained += 1
            else:
                unexplained[key] += 1
        last[key] = max(slot, last.get(key, slot))
        claimed.add(event["item"])
    over = {f"{h}-{p}": n for (h, p), n in unexplained.items() if n > backoffs.get((h, p), 0)}
    return explained, sum(unexplained.values()), over


def verdict(root: Path, args: argparse.Namespace) -> bool:
    """Print the probe's verdict as JSON; return whether it passed."""
    put_record = json.loads((root / "items.json").read_text())
    items = set(put_record["items"])
    queue = _queues(root, args).queue(put_record["queue"])
    events = [
        json.loads(line) for p in sorted((root / "events").glob("*.jsonl")) for line in p.read_text().splitlines()
    ]
    kinds = collections.Counter(e["event"] for e in events)
    committed = collections.Counter(e["item"] for e in events if e["event"] == "committed")
    killed_after_link = {e["item"] for e in events if e["event"] == "disturb" and e["kind"] == "mark"}
    outcomes = {d: (queue.outcome(d) or {}).get("outcome") for d in items}
    done = {d for d, o in outcomes.items() if o == "done"}
    dead = {d for d, o in outcomes.items() if o == "dead"}
    silent = {d for d in done if committed[d] == 0} - killed_after_link
    explained, unexplained, over = _order(events)
    hosts = sorted({e["host"] for e in events if e["event"] == "claim"})
    segments = {tuple(e["slot"])[0] for e in events if e["event"] == "claim"}
    report = {
        "items": len(items),
        "done_outcomes": len(done),
        "dead_outcomes": len(dead),
        "no_outcome": sorted(items - done - dead)[:10],
        "committed_twice_or_more": sum(committed[i] > 1 for i in items),
        "done_without_a_commit_event": sorted(silent)[:10],
        "duplicate_commits": kinds["duplicate"],
        "lost_claims": kinds["lost"],
        "claims": kinds["claim"],
        "reclaims": sum(1 for e in events if e["event"] == "claim" and e.get("broke")),
        "disturbances": kinds["disturb"],
        "segments_claimed_from": len(segments),
        "retired_segments": sum(1 for _ in (queue.path / "retired").iterdir()),
        "order_inversions_explained": explained,
        "order_inversions_unexplained": unexplained,
        "claimers_over_their_backoffs": over,
        "hosts": hosts,
        "distinct_hosts": len(hosts),
    }
    report["pass"] = (
        len(done) == len(items)
        and not dead
        and not silent
        and report["committed_twice_or_more"] == 0
        and not over
        and len(hosts) >= args.hosts
    )
    print(json.dumps(report))
    return bool(report["pass"])


def main(argv: list[str] | None = None) -> int:
    """Run one mode; `verdict` exits 1 when the probe failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["put", "claim", "verdict"])
    parser.add_argument("dir", type=Path, nargs="?", default=os.environ.get(DIR_VARIABLE))
    parser.add_argument("--queue", default="cpu")
    parser.add_argument("--profile", default="laptop")
    parser.add_argument("--segment-items", type=int, default=64)
    parser.add_argument("--items", type=int, default=10_000)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--work", type=float, default=0.0)
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--until", type=float, default=3600.0)
    parser.add_argument("--crash-after", type=int, default=0)
    parser.add_argument("--crash-at", choices=["hold", "mark", "stall"], default="hold")
    parser.add_argument("--stall", type=float, default=0.0)
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
