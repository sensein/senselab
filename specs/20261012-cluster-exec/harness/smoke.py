"""Live smoke of the work_pool controller on real Slurm partitions, with one worker cancelled mid-item.

    python smoke.py POOL CLASSES [--items 20] [--seconds 60] [--tick 30] [--timeout 1500] [--log-dir D]

Publishes ``--items`` items running ``sleep --seconds`` in the first class of the CLASSES file, then runs
``control POOL --classes CLASSES --once`` every ``--tick`` seconds. As soon as an item is leased, the worker job
holding it is cancelled with ``scancel`` (a preemption): the item must be released and later done at attempt 2.
At the end, this pool's leftover workers are cancelled (only jobs named ``work-pool-<tag>-<class>``). Prints:

    smoke items=N done=D dead=X
    preempted item=<id> job=<job> interrupted=<reason> done_attempt=<a>
    partition=<p> submitted=<n> started=<n>      (one line per partition of the class)

and exits 0 when done=N, dead=0, done_attempt=2 and some partition started a worker.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from senselab.utils.work_pool import Controller, Item, WorkPool, load_classes


def sh(*argv: str) -> str:
    return subprocess.run(list(argv), check=True, capture_output=True, text=True).stdout


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def victim(pool: WorkPool, name: str) -> tuple[str, str]:
    """The first leased item whose worker job is one of this pool's (checked by job name), or empty strings."""
    for lease in sorted(p for p in (pool.root / "leases").iterdir() if not p.name.startswith(".")):
        owner = read(lease / "owner.json")
        job = str(read(pool.root / "workers" / f"{owner.get('worker')}.json").get("job") or "")
        found = subprocess.run(["squeue", "-h", "-j", job, "-o", "%j"], capture_output=True, text=True) if job else None
        if found is not None and found.stdout.strip() == name:
            return lease.name, job
    return "", ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pool")
    parser.add_argument("classes")
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--tick", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=1500.0)
    parser.add_argument("--log-dir")
    args = parser.parse_args()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    cls = load_classes(args.classes)[0]
    pool = WorkPool.create(args.pool, lease_s=60, heartbeat_s=5)
    pool.publish(
        [Item(id=f"smoke-{i:03d}", resource_class=cls.name, command=("sleep", str(args.seconds))) for i in range(args.items)]
    )
    name = Controller(pool, [cls]).job_name(cls.name)
    control = [sys.executable, "-m", "senselab.utils.work_pool", "control", str(pool.root), "--classes", args.classes]
    control += ["--once", "--worker-poll", "5", "--worker-linger", "2"]
    if args.log_dir:
        control += ["--log-dir", args.log_dir]
    item, job, last, deadline = "", "", 0.0, time.time() + args.timeout
    while time.time() < deadline:
        if time.time() - last >= args.tick:
            subprocess.run(control, check=True)
            last = time.time()
        if not job:
            item, job = victim(pool, name)
            if job:
                sh("scancel", job)
                print(f"cancelled job={job} holding item={item}", flush=True)
        counts = pool.status()["counts"]
        if counts["done"] + counts["dead"] == args.items:
            break
        time.sleep(2)
    leftovers = sh("squeue", "--me", "-h", f"--name={name}", "-o", "%i").split()
    if leftovers:
        sh("scancel", *leftovers)
    counts = pool.status()["counts"]
    print(f"smoke items={args.items} done={counts['done']} dead={counts['dead']} leftovers_cancelled={len(leftovers)}")
    interrupted = read(pool.root / "interrupted" / f"{item}.1.json").get("reason")
    attempt = read(pool.root / "done" / f"{item}.json").get("attempt")
    print(f"preempted item={item} job={job} interrupted={interrupted} done_attempt={attempt}")
    rows = sh("sacct", "-X", "-n", "-P", f"--name={name}", "-S", started_at, "-o", "JobID,Partition").split()
    submitted: dict[str, int] = {}
    for row in rows:
        part = row.split("|")[1]
        submitted[part] = submitted.get(part, 0) + 1
    started: dict[str, set] = {}
    for record in (pool.root / "workers").glob("*.json"):
        data = read(record)
        if data.get("job"):
            started.setdefault(str(data.get("partition")), set()).add(str(data["job"]))
    for spec in cls.partitions:
        p = spec.partition
        print(f"partition={p} submitted={submitted.get(p, 0)} started={len(started.get(p, ()))}")
    ok = counts["done"] == args.items and counts["dead"] == 0 and attempt == 2 and any(started.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
