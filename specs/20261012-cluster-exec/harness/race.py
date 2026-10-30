"""Cross-node races on a shared directory: the rename primitives, then senselab.utils.work_pool itself.

    python race.py prim-setup DIR --names N          # N files in DIR/queue/
    python race.py prim DIR --names N --start T      # each process: try to win every name by mkdir, staged dir
                                                     # rename and file-out-of-queue rename
    python race.py prim-check DIR --names N          # every name must have exactly one winner per primitive
    python race.py publish POOL --items N --lease-s S --heartbeat-s H
    python race.py work POOL --items N --start T [--hold S] [--kill-third] [--hold-range LO,HI --beats]
    python race.py check POOL --items N [--overlap]  # last line: race items=N done=D double=X lost=Y [overlap=Z]

Ported from sensein/senselab@feat/task-events (f2a806af), specs/20261011-slice-pool/pool/harness/race.py, with
the file-out-of-queue rename added and the pool legs on top. ``--kill-third`` makes every third worker (by
SLURM_PROCID) SIGKILL itself while it holds a lease, so its lease must be reaped while the others claim.
``--hold-range LO,HI --beats`` holds each lease a random LO..HI seconds (longer than lease_s) while a thread sends
real heartbeats; each run takes ``<pool>-running/<id>`` with mkdir, so a second run of an id that starts while
another still holds it (on any node, at any attempt) is counted as an overlap.
"""

import argparse
import json
import os
import random
import signal
import socket
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from typing import Callable

from senselab.utils.work_pool import Item, Lease, WorkPool


def me() -> str:
    host = socket.gethostname().split(".")[0]
    return f"{host}.{os.environ.get('SLURM_PROCID', 'x')}.{os.getpid()}"


def wait_until(start: float) -> None:
    while time.time() < start:
        time.sleep(0.01)


def prim_setup(root: Path, names: int) -> None:
    for sub in ("mkdir", "rename", "tmp", "queue", "taken", "wins"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for i in range(names):
        (root / "queue" / str(i)).write_text(str(i))


def prim(root: Path, names: int, start: float) -> None:
    who = me()
    taken = root / "taken" / who
    taken.mkdir()
    won: dict[str, list[int]] = {"mkdir": [], "rename": [], "file": []}
    wait_until(start)
    for i in range(names):
        try:
            os.mkdir(root / "mkdir" / str(i))
            won["mkdir"].append(i)
        except FileExistsError:
            pass
        token = uuid.uuid4().hex
        staged = root / "tmp" / token
        staged.mkdir()
        (staged / "owner").write_text(token)
        try:
            os.rename(staged, root / "rename" / str(i))
        except OSError:
            pass
        try:
            if (root / "rename" / str(i) / "owner").read_text() == token:
                won["rename"].append(i)
            else:
                (staged / "owner").unlink()
                staged.rmdir()
        except FileNotFoundError:
            pass
        try:
            os.rename(root / "queue" / str(i), taken / str(i))
        except OSError:
            pass
        if (taken / str(i)).exists():
            won["file"].append(i)
    (root / "wins" / f"{who}.json").write_text(json.dumps(won))


def prim_check(root: Path, names: int) -> int:
    files = list((root / "wins").glob("*.json"))
    counts = {k: [0] * names for k in ("mkdir", "rename", "file")}
    for path in files:
        for k, v in json.loads(path.read_text()).items():
            for i in v:
                counts[k][i] += 1
    hosts = sorted({p.name.split(".")[0] for p in files})
    ok = True
    print(f"primitives processes={len(files)} hosts={len(hosts)}")
    for k, c in counts.items():
        one, zero, more = sum(x == 1 for x in c), sum(x == 0 for x in c), sum(x > 1 for x in c)
        ok = ok and one == names
        print(f"primitive {k} names={names} exactly_one={one} zero={zero} more={more}")
    return 0 if ok else 1


def publish(root: Path, items: int, lease_s: float, heartbeat_s: float) -> None:
    pool = WorkPool.create(root, lease_s=lease_s, heartbeat_s=heartbeat_s)
    pool.publish([Item(id=f"item-{i:05d}", resource_class="main", command=("true",)) for i in range(items)])
    (root.parent / f"{root.name}-log").mkdir(exist_ok=True)
    print(f"published {items} into {root.name} lease_s={lease_s}")


def held(pool: WorkPool, lease: Lease, seconds: float, running: Path, note: Callable[..., None]) -> None:
    """Hold a lease for ``seconds`` with real heartbeats, marking the run in ``running/`` to catch overlaps."""
    item = lease.item
    marker = running / item.id
    try:
        os.mkdir(marker)
        mine = True
    except FileExistsError:
        mine = False
        note(overlap=item.id, attempt=item.attempt)
    stop, lost = threading.Event(), []

    def beat() -> None:
        while not stop.wait(pool.heartbeat_s):
            if not pool.heartbeat(lease):
                lost.append(1)
                return

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    time.sleep(seconds)
    stop.set()
    thread.join()
    if lost:
        note(leaselost=item.id, attempt=item.attempt)
    if mine:
        os.rmdir(marker)


def work(
    root: Path, items: int, start: float, hold: float, kill_third: bool, hold_range: str = "", beats: bool = False
) -> None:
    pool, who = WorkPool(root), me()
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    kill_after = random.Random(rank).randint(5, 40) if kill_third and rank % 3 == 0 else None
    fd = os.open(root.parent / f"{root.name}-log" / f"{who}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)

    def note(**rec: object) -> None:
        os.write(fd, (json.dumps(rec) + "\n").encode())

    running = root.parent / f"{root.name}-running"
    running.mkdir(exist_ok=True)
    rng = random.Random(f"{who}")
    lo, hi = (float(x) for x in hold_range.split(",")) if hold_range else (0.0, 0.0)

    wait_until(start)
    claimed, deadline = 0, time.time() + 900
    while time.time() < deadline:
        lease = pool.claim(["main"], who)
        if lease is None:
            if len(os.listdir(root / "done")) + len(os.listdir(root / "dead")) >= items:
                break
            time.sleep(0.5)
            continue
        note(claim=lease.item.id, attempt=lease.item.attempt)
        claimed += 1
        if kill_after is not None and claimed == kill_after:
            note(killed=lease.item.id, attempt=lease.item.attempt)
            os.kill(os.getpid(), signal.SIGKILL)
        if beats:
            held(pool, lease, rng.uniform(lo, hi), running, note)
        elif hold:
            time.sleep(hold)
        note(complete=lease.item.id, attempt=lease.item.attempt, ok=pool.complete(lease, {"exit_code": 0}))
    os.close(fd)


def check(root: Path, items: int, overlap_leg: bool = False) -> int:
    claims: Counter[tuple[str, int]] = Counter()
    completes: Counter[str] = Counter()
    overlaps: Counter[str] = Counter()
    leaselost = 0
    killed, hosts, logs = 0, set(), list((root.parent / f"{root.name}-log").glob("*.jsonl"))
    for path in logs:
        hosts.add(path.name.split(".")[0])
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if "claim" in rec:
                claims[(rec["claim"], rec["attempt"])] += 1
            elif "killed" in rec:
                killed += 1
            elif "overlap" in rec:
                overlaps[rec["overlap"]] += 1
            elif "leaselost" in rec:
                leaselost += 1
            elif rec.get("ok"):
                completes[rec["complete"]] += 1
    expected = {f"item-{i:05d}" for i in range(items)}
    done = {n.removesuffix(".json") for n in os.listdir(root / "done") if not n.startswith(".")}
    double = sum(1 for c in claims.values() if c > 1) + sum(1 for c in completes.values() if c > 1)
    double += len(set(completes) ^ done)  # a completion without a done record, or the reverse
    lost = len(expected - done)
    st = WorkPool(root).status()["counts"]
    print(f"leg {root.name} workers={len(logs)} hosts={len(hosts)} claims={sum(claims.values())} killed={killed} "
          f"interrupted={st['interrupted']} dead={st['dead']} queued={st['queued']} leased={st['leased']} "
          f"conflicts={st['conflicts']} leaselost={leaselost}")
    tail = f" overlap={len(overlaps)}" if overlap_leg else ""
    print(f"race items={items} done={len(done & expected)} double={double} lost={lost}{tail}")
    return 0 if double == 0 and lost == 0 and not overlaps and done == expected else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prim-setup", "prim", "prim-check", "publish", "work", "check"])
    p.add_argument("root", type=Path)
    p.add_argument("--names", type=int, default=400)
    p.add_argument("--items", type=int, default=2000)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--hold", type=float, default=0.0)
    p.add_argument("--lease-s", type=float, default=60.0)
    p.add_argument("--heartbeat-s", type=float, default=5.0)
    p.add_argument("--kill-third", action="store_true")
    p.add_argument("--hold-range", default="")
    p.add_argument("--beats", action="store_true")
    p.add_argument("--overlap", action="store_true")
    a = p.parse_args()
    if a.mode == "prim-setup":
        prim_setup(a.root, a.names)
    elif a.mode == "prim":
        prim(a.root, a.names, a.start)
    elif a.mode == "prim-check":
        return prim_check(a.root, a.names)
    elif a.mode == "publish":
        publish(a.root, a.items, a.lease_s, a.heartbeat_s)
    elif a.mode == "work":
        work(a.root, a.items, a.start, a.hold, a.kill_third, a.hold_range, a.beats)
    else:
        return check(a.root, a.items, a.overlap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
