#!/usr/bin/env python3
"""slice_pool: a Slurm worker pool over the slices of one stage (stdlib only, Python 3.12).

A stage is a pool directory holding N slice indices. A few workers (one small array, or one array per GPU
type) each claim the next free index, run the stage's per-slice command for it, mark it, and repeat.

    slice_pool.py init   POOL --slices N [--retries 2] [--stale-minutes 10] [--heartbeat 60]
                              [--success-codes 0] [--max-interrupts 6] [--same-host-grace 300]
                              [--linger 2] [--poll 30] [--host-check CMD] [--force] -- COMMAND ARG ...
    slice_pool.py work   POOL                    # one worker (pool/worker.sbatch runs this)
    slice_pool.py status POOL                    # counts -> POOL/status.json, printed
    slice_pool.py submit POOL --workers W [--after JOB[:JOB]] [--name NAME] [-- SBATCH_ARG ...]
    slice_pool.py gate   POOL [--check-only]     # after the worker arrays end (afterany)

COMMAND may carry ``{i}`` (slice index) and ``{n}`` (slice count); the child also gets POOL_DIR,
POOL_SLICE_INDEX, POOL_SLICE_COUNT and POOL_ATTEMPT in its environment. Its output goes to
POOL/logs/slice-<i>.<attempt>.<token>.log.

Layout of POOL:
    pool.json            the stage definition (init)
    claims/<i>/          a claim: owner.json, heartbeat, and done.json once the slice succeeded
    done/<i>             success marker (exit code, host, times)
    failed/<i>.<k>       the k-th failed attempt of slice i (exit code, host, log)
    interrupted/<i>.<t>  a claim taken over as stale, released on SIGTERM, or lost
    tmp/ trash/ linger/  claim staging, released claims, standby slots
    submissions.<job>.json  each worker array `submit` made (job id, sbatch args) -- what `gate` resubmits
    resubmitted, halted  written by `gate`
    status.json          written by `status` and `gate`

gate exit codes: 0 complete, 10 incomplete and workers resubmitted (the new job ids are printed on the
last line as ``RESUBMITTED <id>[:<id>]``), 1 incomplete after the one resubmission (``halted`` written),
3 incomplete under --check-only. The design is in specs/20261011-slice-pool/design.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

TOOL = Path(__file__).resolve()
WORKER_SBATCH = TOOL.parent / "worker.sbatch"
DEFAULT_PY = "/orcd/scratch/bcs/002/satra/senselab-r28/.venv/bin/python"
SUBDIRS = ("claims", "done", "failed", "interrupted", "tmp", "trash", "linger", "logs")
GATE_RESUBMITTED = 10
GATE_HALTED = 1
GATE_INCOMPLETE = 3


def now() -> float:
    """Wall-clock seconds."""
    return time.time()


def stamp(t: float | None = None) -> str:
    """ISO local time."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now() if t is None else t))


def log(msg: str) -> None:
    """One timestamped line to stdout."""
    print(f"[{stamp()}] {msg}", flush=True)


def write_json(path: Path, data: Any) -> None:
    """Write JSON through a per-writer temporary file and an atomic rename."""
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex[:12]}")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    """Read JSON; None when the file is absent or half-visible."""
    try:
        with open(path) as handle:
            return json.load(handle)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None


def fresh_mtime(path: Path) -> float | None:
    """The file's mtime, revalidated with the server (an open forces NFS close-to-open revalidation)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except (FileNotFoundError, NotADirectoryError):
        return None
    try:
        return os.fstat(fd).st_mtime
    finally:
        os.close(fd)


def exists_fresh(path: Path) -> bool:
    """Whether a file exists, asked through an open rather than a cached stat."""
    try:
        os.close(os.open(path, os.O_RDONLY))
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False


def listdir(path: Path) -> list[str]:
    """Directory entries, ignoring dot-files (temporary writes)."""
    try:
        return [name for name in os.listdir(path) if not name.startswith(".")]
    except FileNotFoundError:
        return []


def identity() -> dict[str, Any]:
    """Who this process is: host, Slurm job/array ids, restart count, pid."""
    env = os.environ
    job = env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID") or "local"
    task = env.get("SLURM_ARRAY_TASK_ID")
    return {
        "host": socket.gethostname().split(".")[0],
        "job": f"{job}_{task}" if task is not None else job,
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "restart": int(env.get("SLURM_RESTART_COUNT", "0") or 0),
        "partition": env.get("SLURM_JOB_PARTITION"),
        "gpus": env.get("SLURM_JOB_GPUS") or env.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
    }


class Pool:
    """One stage's pool directory."""

    def __init__(self, root: Path) -> None:
        """Bind to an initialised pool directory."""
        self.root = root.resolve()
        cfg = read_json(self.root / "pool.json")
        if cfg is None:
            raise SystemExit(f"{self.root}: no pool.json (run init first)")
        self.cfg: dict[str, Any] = cfg
        self.n: int = int(cfg["slices"])

    def d(self, name: str) -> Path:
        """A subdirectory."""
        return self.root / name

    # -- reading the state -------------------------------------------------------------------------------------
    def failures(self) -> dict[int, list[dict[str, Any]]]:
        """Failed attempts per slice (from failed/<i>.<k>)."""
        out: dict[int, list[dict[str, Any]]] = {}
        for name in listdir(self.d("failed")):
            head = name.split(".")[0]
            if head.isdigit():
                rec = read_json(self.d("failed") / name) or {"host": None, "end": 0}
                out.setdefault(int(head), []).append(rec)
        return out

    def interruptions(self) -> dict[int, int]:
        """Interrupted attempts per slice (stale takeovers, SIGTERM releases, lost claims)."""
        out: dict[int, int] = {}
        for name in listdir(self.d("interrupted")):
            head = name.split(".")[0]
            if head.isdigit():
                out[int(head)] = out.get(int(head), 0) + 1
        return out

    def exhausted(self, i: int, fails: dict[int, list[dict[str, Any]]], ints: dict[int, int]) -> bool:
        """Whether slice i has used its attempt budget (1 + retries failures, or max_interrupts interruptions)."""
        return len(fails.get(i, [])) >= 1 + int(self.cfg["retries"]) or ints.get(i, 0) >= int(
            self.cfg["max_interrupts"]
        )

    def claim_state(self, i: int) -> str:
        """'none', 'done', 'live' or 'stale' for slice i's claim directory."""
        claim = self.d("claims") / str(i)
        if exists_fresh(claim / "done.json"):
            return "done"
        beat = fresh_mtime(claim / "heartbeat")
        if beat is None:
            if not claim.exists():
                return "none"
            owner = fresh_mtime(claim / "owner.json")
            beat = owner if owner is not None else 0.0
        return "stale" if now() - beat > 60 * float(self.cfg["stale_minutes"]) else "live"

    def status(self) -> dict[str, Any]:
        """One state per slice and the counts, written to status.json."""
        fails, ints = self.failures(), self.interruptions()
        done_marks = {int(n) for n in listdir(self.d("done")) if n.isdigit()}
        claimed = {int(n) for n in listdir(self.d("claims")) if n.isdigit()}
        per: dict[str, list[int]] = {k: [] for k in ("done", "running", "stale", "failed", "retry", "unclaimed")}
        for i in range(self.n):
            state = self.claim_state(i) if i in claimed else "none"
            if i in done_marks or state == "done":
                per["done"].append(i)
            elif state == "live":
                per["running"].append(i)
            elif state == "stale":
                per["stale"].append(i)
            elif self.exhausted(i, fails, ints):
                per["failed"].append(i)
            elif i in fails or i in ints:
                per["retry"].append(i)
            else:
                per["unclaimed"].append(i)
        result = {
            "pool": str(self.root),
            "name": self.cfg.get("name"),
            "slices": self.n,
            "time": stamp(),
            "counts": {k: len(v) for k, v in per.items()},
            "complete": len(per["done"]) == self.n,
            "failed_attempts": sum(len(v) for v in fails.values()),
            "interrupted_attempts": sum(ints.values()),
            "missing": [i for i in range(self.n) if i not in set(per["done"])],
            "slices_by_state": {k: v for k, v in per.items() if k != "done"},
            "halted": (self.root / "halted").exists(),
            "resubmitted": (self.root / "resubmitted").exists(),
        }
        write_json(self.root / "status.json", result)
        return result

    # -- claiming ----------------------------------------------------------------------------------------------
    def _stage_claim(self, owner: dict[str, Any]) -> Path:
        """A populated private directory (owner.json + heartbeat) ready to be renamed into claims/."""
        staged = self.d("tmp") / owner["token"]
        staged.mkdir()
        write_json(staged / "owner.json", owner)
        (staged / "heartbeat").touch()
        return staged

    def _install(self, staged: Path, i: int, token: str) -> bool:
        """Rename the staged directory onto claims/<i>; exclusive because a claim is never an empty directory."""
        target = self.d("claims") / str(i)
        try:
            os.rename(staged, target)
        except OSError:
            pass  # EEXIST / ENOTEMPTY: someone holds it (a spurious error on a retransmit is checked below)
        held = read_json(target / "owner.json")
        if held is not None and held.get("token") == token:
            return True
        shutil.rmtree(staged, ignore_errors=True)
        return False

    def _take_over(self, i: int, me: dict[str, Any], reason: str) -> bool:
        """Move a stale claims/<i> aside (only one renamer wins) and record the interruption."""
        claim = self.d("claims") / str(i)
        aside = self.d("trash") / f"{i}.{reason}.{uuid.uuid4().hex[:12]}"
        try:
            os.rename(claim, aside)
        except OSError:
            pass
        if not aside.exists():
            return False
        if exists_fresh(aside / "done.json"):  # it finished between our check and the rename: put it back
            try:
                os.rename(aside, claim)
            except OSError:
                log(f"slice {i}: finished claim moved aside and could not be restored; done/{i} still marks it")
            return False
        old = read_json(aside / "owner.json") or {}
        write_json(
            self.d("interrupted") / f"{i}.{uuid.uuid4().hex[:12]}",
            {
                "slice": i,
                "reason": reason,
                "previous_owner": old,
                "taken_by": me,
                "time": stamp(),
                "heartbeat_age_s": round(now() - (fresh_mtime(aside / "heartbeat") or 0), 1),
            },
        )
        shutil.rmtree(aside, ignore_errors=True)
        log(f"slice {i}: {reason} claim of {old.get('job')}@{old.get('host')} taken over")
        return True

    def next_slice(self, me: dict[str, Any]) -> tuple[int | None, bool]:
        """Claim the next runnable slice. Returns (index or None, whether work is still outstanding elsewhere)."""
        fails, ints = self.failures(), self.interruptions()
        claimed = {int(n) for n in listdir(self.d("claims")) if n.isdigit()}
        grace = float(self.cfg["same_host_grace"])
        outstanding = False
        fresh, retry, deferred, stale = [], [], [], []
        for i in range(self.n):
            if i in claimed:
                state = self.claim_state(i)
                if state == "stale":
                    stale.append(i)
                elif state == "live":
                    outstanding = True
                elif state == "none":
                    claimed.discard(i)
                if state != "none":
                    continue
            if exists_fresh(self.d("done") / str(i)) or self.exhausted(i, fails, ints):
                continue
            prior = fails.get(i, [])
            if not prior and not ints.get(i):
                fresh.append(i)
            elif any(r.get("host") == me["host"] and now() - float(r.get("end", 0)) < grace for r in prior):
                deferred.append(i)
            else:
                retry.append(i)
        for i in stale:
            if self._take_over(i, me, "stale"):
                ints[i] = ints.get(i, 0) + 1
                if not self.exhausted(i, fails, ints):
                    retry.append(i)
        for i in fresh + sorted(retry):
            attempt = len(fails.get(i, [])) + ints.get(i, 0) + 1
            owner = {**me, "slice": i, "attempt": attempt, "token": uuid.uuid4().hex, "claimed": stamp()}
            if self._install(self._stage_claim(owner), i, owner["token"]):
                return i, True
            outstanding = True
        return None, outstanding or bool(deferred)

    def release(self, i: int, token: str) -> None:
        """Give slice i back: move my claim aside (only if it is still mine) and delete it."""
        claim = self.d("claims") / str(i)
        held = read_json(claim / "owner.json")
        if held is None or held.get("token") != token:
            return
        aside = self.d("trash") / f"{i}.release.{token[:12]}"
        try:
            os.rename(claim, aside)
        except OSError:
            return
        shutil.rmtree(aside, ignore_errors=True)

    # -- standby slots -----------------------------------------------------------------------------------------
    def take_linger(self, me: dict[str, Any]) -> Path | None:
        """Hold one of `linger` standby slots (claimed and taken over like slices), or None when all are held."""
        for k in range(int(self.cfg["linger"])):
            slot = self.d("linger") / str(k)
            beat = fresh_mtime(slot / "heartbeat")
            if beat is not None and now() - beat > 60 * float(self.cfg["stale_minutes"]):
                aside = self.d("trash") / f"linger{k}.{uuid.uuid4().hex[:12]}"
                try:
                    os.rename(slot, aside)
                    shutil.rmtree(aside, ignore_errors=True)
                except OSError:
                    pass
            token = uuid.uuid4().hex
            staged = self._stage_claim({**me, "linger": k, "token": token, "claimed": stamp()})
            try:
                os.rename(staged, slot)
            except OSError:
                pass
            held = read_json(slot / "owner.json")
            if held is not None and held.get("token") == token:
                return slot
            shutil.rmtree(staged, ignore_errors=True)
        return None

    def drop_linger(self, slot: Path) -> None:
        """Leave a standby slot."""
        aside = self.d("trash") / f"{slot.name}.drop.{uuid.uuid4().hex[:12]}"
        try:
            os.rename(slot, aside)
            shutil.rmtree(aside, ignore_errors=True)
        except OSError:
            pass


# -- worker ----------------------------------------------------------------------------------------------------
class Worker:
    """Claim, run, mark, repeat."""

    def __init__(self, pool: Pool) -> None:
        """Set up signal handling."""
        self.pool = pool
        self.me = identity()
        self.child: subprocess.Popen[bytes] | None = None
        self.terminating = False
        self.lost = threading.Event()
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_term)
        signal.signal(signal.SIGUSR1, self._on_term)

    def _on_term(self, signum: int, _frame: Any) -> None:
        log(f"signal {signum}: stopping the slice in hand and releasing it")
        self.terminating = True
        self._kill_child()

    def _kill_child(self) -> None:
        child = self.child
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = now() + 20
        while child.poll() is None and now() < deadline:
            time.sleep(0.5)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _beat(self, path: Path, token: str, stop: threading.Event, owner_file: Path) -> None:
        """Touch the heartbeat every `heartbeat` seconds; if the claim is no longer ours, stop the child."""
        period = float(self.pool.cfg["heartbeat"])
        while not stop.wait(period):
            held = read_json(owner_file)
            if held is None or held.get("token") != token:
                log(f"claim {owner_file.parent} is no longer mine; stopping the child")
                self.lost.set()
                self._kill_child()
                return
            try:
                os.utime(path)
            except FileNotFoundError:
                pass

    def host_ok(self) -> bool:
        """Run the pool's host check (e.g. GPU driver), once per worker."""
        check = self.pool.cfg.get("host_check")
        if not check:
            return True
        rc = subprocess.run(["bash", "-c", check]).returncode
        if rc != 0:
            log(f"host check failed on {self.me['host']} (exit {rc}); this worker takes no slices")
        return rc == 0

    def run_slice(self, i: int) -> None:
        """Run the stage command for slice i under my claim, then mark done/failed or release."""
        claim = self.pool.d("claims") / str(i)
        owner = read_json(claim / "owner.json") or {}
        token, attempt = owner["token"], owner["attempt"]
        n = self.pool.n
        cmd = [a.replace("{i}", str(i)).replace("{n}", str(n)) for a in self.pool.cfg["command"]]
        logfile = self.pool.d("logs") / f"slice-{i}.{attempt}.{token[:8]}.log"
        env = {
            **os.environ,
            "POOL_DIR": str(self.pool.root),
            "POOL_SLICE_INDEX": str(i),
            "POOL_SLICE_COUNT": str(n),
            "POOL_ATTEMPT": str(attempt),
        }
        if self.terminating:
            self.pool.release(i, token)
            return
        start = now()
        log(f"slice {i}/{n} attempt {attempt}: {shlex.join(cmd)} > {logfile.name}")
        stop = threading.Event()
        self.lost.clear()
        beat = threading.Thread(
            target=self._beat, args=(claim / "heartbeat", token, stop, claim / "owner.json"), daemon=True
        )
        with open(logfile, "ab") as out:
            self.child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            beat.start()
            if self.terminating:
                self._kill_child()
            rc = self.child.wait()
        if rc < 0 or rc in (137, 143):
            time.sleep(3)  # a job-wide SIGTERM can reach the child before this process's handler has run
        stop.set()
        beat.join(timeout=5)
        self.child = None
        end = now()
        record = {
            "slice": i,
            "rc": rc,
            "attempt": attempt,
            "start": stamp(start),
            "end": end,
            "end_iso": stamp(end),
            "seconds": round(end - start, 1),
            "log": str(logfile),
            **self.me,
            "token": token,
        }
        success = rc in [int(c) for c in self.pool.cfg["success_codes"]]
        if success and not self.terminating:
            if self.lost.is_set():
                log(f"slice {i}: succeeded after its claim was taken over; marking done anyway")
            else:
                write_json(claim / "done.json", record)
            write_json(self.pool.d("done") / str(i), record)
            log(f"slice {i}: done (rc {rc}, {record['seconds']}s)")
            return
        if self.terminating or self.lost.is_set():
            reason = "sigterm" if self.terminating else "lost"
            write_json(
                self.pool.d("interrupted") / f"{i}.{uuid.uuid4().hex[:12]}",
                {**record, "reason": reason, "time": stamp()},
            )
            self.pool.release(i, token)
            log(f"slice {i}: {reason} (rc {rc}); released")
            return
        k = len(self.pool.failures().get(i, [])) + 1
        target = self.pool.d("failed") / f"{i}.{k}"
        while target.exists():
            k += 1
            target = self.pool.d("failed") / f"{i}.{k}"
        write_json(target, record)
        self.pool.release(i, token)
        log(f"slice {i}: FAILED rc {rc} on {self.me['host']} (failure {k}); released")

    def loop(self) -> int:
        """Work until nothing is left (or a standby slot is not needed)."""
        log(f"worker {self.me['job']} on {self.me['host']} pool {self.pool.root} ({self.pool.n} slices)")
        if not self.host_ok():
            return 0
        slot: Path | None = None
        ran = 0
        try:
            while not self.terminating:
                if (self.pool.root / "halted").exists():
                    log("pool halted; exiting")
                    break
                i, outstanding = self.pool.next_slice(self.me)
                if i is not None:
                    if slot is not None:
                        self.pool.drop_linger(slot)
                        slot = None
                    self.run_slice(i)
                    ran += 1
                    continue
                if not outstanding:
                    log("no slice left to run or to wait for")
                    break
                if slot is None:
                    slot = self.pool.take_linger(self.me)
                    if slot is None:
                        log("work outstanding elsewhere; standby slots are held; exiting")
                        break
                    log(f"standing by in {slot.name} while other claims run")
                try:
                    os.utime(slot / "heartbeat")
                except FileNotFoundError:
                    slot = None
                time.sleep(float(self.pool.cfg["poll"]))
        finally:
            if slot is not None:
                self.pool.drop_linger(slot)
        st = self.pool.status()
        log(f"worker exits after {ran} slice(s); pool {st['counts']}")
        return 143 if self.terminating else 0


# -- slurm -----------------------------------------------------------------------------------------------------
def sbatch(args: list[str]) -> str:
    """Run sbatch --parsable and return the job id."""
    out = subprocess.run(["sbatch", "--parsable", *args], check=True, capture_output=True, text=True).stdout
    return out.strip().split(";")[0]


def submit(pool: Pool, workers: int, after: str | None, name: str | None, extra: list[str]) -> str:
    """Submit one worker array over the pool and record it."""
    workers = max(1, min(workers, pool.n))
    args = [
        f"--array=0-{workers - 1}%{workers}",
        f"--job-name={name or pool.cfg.get('name') or 'pool'}",
        f"--output={pool.d('logs')}/worker-%A_%a.out",
    ]
    if after:
        args.append(f"--dependency=afterany:{after}")
    job = sbatch([*args, *extra, str(WORKER_SBATCH), str(pool.root)])
    with open(pool.root / f"submissions.{job}.json", "w") as handle:
        json.dump({"job": job, "workers": workers, "name": name, "extra": extra, "time": stamp()}, handle)
    return job


def submissions(pool: Pool) -> list[dict[str, Any]]:
    """Every recorded worker array, oldest first."""
    recs = [
        read_json(pool.root / n) for n in listdir(pool.root) if n.startswith("submissions.") and n.endswith(".json")
    ]
    return sorted((r for r in recs if r), key=lambda r: int(str(r["job"]).split("_")[0]))


def queued(jobs: list[str]) -> bool | None:
    """Whether any of the jobs is still in the queue; None when squeue could not say."""
    if not jobs:
        return False
    run = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(jobs)], capture_output=True, text=True)
    if run.returncode != 0:
        return False if "Invalid job id" in run.stderr else None
    return bool(run.stdout.strip())


def gate(pool: Pool, check_only: bool) -> int:
    """After every worker array has left the queue: complete -> 0; else resubmit once (10); else halt (1)."""
    st = pool.status()
    if st["complete"]:
        log(f"gate: {pool.root.name} complete ({st['counts']})")
        return 0
    log(f"gate: {pool.root.name} incomplete: {st['counts']} missing {st['missing'][:40]}")
    if check_only:
        return GATE_INCOMPLETE
    subs = submissions(pool)
    alive = queued([str(s["job"]) for s in subs])
    if alive:
        log("gate: a worker array is still queued; not resubmitting (run gate after afterany on every array)")
        return GATE_INCOMPLETE
    if (pool.root / "resubmitted").exists():
        write_json(pool.root / "halted", {"time": stamp(), "status": st})
        pool.status()
        log(f"gate: HALTED {pool.root} after one resubmission; {len(st['missing'])} slice(s) missing")
        return GATE_HALTED
    if alive is False:  # no worker alive: claims left behind are orphans; release them now instead of waiting
        me = {**identity(), "role": "gate"}
        for i in st["slices_by_state"]["running"] + st["slices_by_state"]["stale"]:
            pool._take_over(i, me, "orphan")
    archive = pool.d("failed") / "round1"
    for i in st["slices_by_state"]["failed"]:  # one more full budget for slices that used theirs
        archive.mkdir(exist_ok=True)
        for name in listdir(pool.d("failed")):
            if name.split(".")[0] == str(i):
                os.replace(pool.d("failed") / name, archive / name)
        for name in listdir(pool.d("interrupted")):
            if name.split(".")[0] == str(i):
                (pool.d("interrupted") / "round1").mkdir(exist_ok=True)
                os.replace(pool.d("interrupted") / name, pool.d("interrupted") / "round1" / name)
    remaining = len(st["missing"])
    new = []
    for s in subs if subs else [{"workers": 1, "name": None, "extra": []}]:
        if s.get("resubmission"):
            continue
        job = submit(pool, min(int(s["workers"]), remaining), None, s.get("name"), list(s.get("extra", [])))
        rec_path = pool.root / f"submissions.{job}.json"
        rec = read_json(rec_path) or {}
        write_json(rec_path, {**rec, "resubmission": True})
        new.append(job)
    write_json(pool.root / "resubmitted", {"time": stamp(), "jobs": new, "missing": st["missing"]})
    pool.status()
    log(f"gate: resubmitted workers {new} for {remaining} missing slice(s)")
    print(f"RESUBMITTED {':'.join(new)}", flush=True)
    return GATE_RESUBMITTED


# -- cli -------------------------------------------------------------------------------------------------------
def cmd_init(a: argparse.Namespace) -> int:
    """Create or confirm a pool directory."""
    root = Path(a.pool).resolve()
    command = a.command
    if not command:
        raise SystemExit("init: give the per-slice command after --")
    cfg = {
        "name": a.name or root.name,
        "slices": a.slices,
        "retries": a.retries,
        "stale_minutes": a.stale_minutes,
        "heartbeat": a.heartbeat,
        "success_codes": [int(c) for c in a.success_codes.split(",")],
        "max_interrupts": a.max_interrupts,
        "same_host_grace": a.same_host_grace,
        "linger": a.linger,
        "poll": a.poll,
        "host_check": a.host_check,
        "command": command,
    }
    if a.heartbeat * 3 > a.stale_minutes * 60:
        raise SystemExit("init: --stale-minutes must be at least three heartbeats")
    root.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (root / sub).mkdir(exist_ok=True)
    old = read_json(root / "pool.json")
    if old is not None and old != cfg and not a.force:
        diff = sorted(k for k in set(old) | set(cfg) if old.get(k) != cfg.get(k))
        raise SystemExit(f"init: {root}/pool.json exists and differs in {diff}; pass --force to replace it")
    if old != cfg:
        write_json(root / "pool.json", cfg)
    print(json.dumps(Pool(root).status()["counts"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="create a pool")
    i.add_argument("pool")
    i.add_argument("--slices", type=int, required=True)
    i.add_argument("--name")
    i.add_argument("--retries", type=int, default=2)
    i.add_argument("--stale-minutes", type=float, default=10.0)
    i.add_argument("--heartbeat", type=float, default=60.0)
    i.add_argument("--success-codes", default="0")
    i.add_argument("--max-interrupts", type=int, default=6)
    i.add_argument("--same-host-grace", type=float, default=300.0)
    i.add_argument("--linger", type=int, default=2)
    i.add_argument("--poll", type=float, default=30.0)
    i.add_argument("--host-check")
    i.add_argument("--force", action="store_true")
    w = sub.add_parser("work", help="run one worker")
    w.add_argument("pool")
    s = sub.add_parser("status", help="counts -> status.json")
    s.add_argument("pool")
    b = sub.add_parser("submit", help="submit a worker array")
    b.add_argument("pool")
    b.add_argument("--workers", type=int, required=True)
    b.add_argument("--after")
    b.add_argument("--name")
    g = sub.add_parser("gate", help="completeness check / one resubmission / halt")
    g.add_argument("pool")
    g.add_argument("--check-only", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    tail: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, tail = argv[:cut], argv[cut + 1 :]
    a = p.parse_args(argv)
    a.command = a.sbatch_args = tail
    if a.cmd == "init":
        return cmd_init(a)
    pool = Pool(Path(a.pool))
    if a.cmd == "work":
        return Worker(pool).loop()
    if a.cmd == "status":
        print(json.dumps(pool.status(), indent=1))
        return 0
    if a.cmd == "submit":
        print(submit(pool, a.workers, a.after, a.name, a.sbatch_args))
        return 0
    return gate(pool, a.check_only)


if __name__ == "__main__":
    sys.exit(main())
