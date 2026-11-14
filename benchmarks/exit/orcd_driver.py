#!/usr/bin/env python3
"""The driver of the cluster evidence steps (`benchmarks/exit/orcd/`): the verdict of each step and its helpers.

Run with `uv run --no-sync python benchmarks/exit/orcd_driver.py <command> ...` from a checkout. Every verdict command
prints one `PASS <check>` or `FAIL <check>: <why>` line per check and, last, one JSON line
`{"pass": bool, "failed": [...], "checks": n}`; it exits 0 only when every check passed.

    profile      copy a site profile, refusing a worker class whose time_s exceeds its partition's limit
    identity     verdict of Step 2 (resample) and Step 3 (stt): plan hash, counts, identities, placement
    interrupt    verdict of Step 4's start job
    resume       verdict of Step 4's resume job
    suite        verdict of one tier's pytest run, from its junit file
    envs         build every shipped env lock under SENSELAB_ENV_ROOT and import its packages
    r253         per-item cost `a` against the jobs of a probe item, over a grid of quantum_items and single_item_jobs
    crossnode    verdict of the lease and queue cross-node probes, from their verdict files
    prov         verdict of the post-hoc provenance read of Step 4's store
    combine      one verdict over the `*-verdict.json` files of a directory
    agent-prepared, agent-ran    verdicts of Step 6's two phases
    watch        run a command under the no-progress watchdog (`stall.py`)
    prefix       the Slurm job-name prefix of a Step 4 run
    get          one value of a JSON or YAML file
    tree         lay a small synthetic BIDS tree
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from senselab_core.runtime.config import dump_yaml, load_yaml

CHECKOUT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CHECKOUT / "benchmarks" / "exit"), str(CHECKOUT / "benchmarks" / "scale")]

HASH = re.compile(r"[0-9a-f]{64}")


class Checks:
    """The checks of one verdict."""

    def __init__(self) -> None:
        """Start with none."""
        self.failed: list[str] = []
        self.count = 0

    def __call__(self, name: str, ok: bool, why: str = "") -> bool:
        """Record a check and print its line."""
        self.count += 1
        if not ok:
            self.failed.append(name)
        print(f"PASS {name}" if ok else f"FAIL {name}: {why or 'false'}", flush=True)
        return ok

    def verdict(self, out: Path, **extra: Any) -> int:  # noqa: ANN401
        """Print and write the JSON verdict; return the exit status."""
        document = {"pass": not self.failed, "failed": self.failed, "checks": self.count, **extra}
        out.write_text(json.dumps(document) + "\n", encoding="utf-8")
        print(json.dumps(document), flush=True)
        return 1 if self.failed else 0


def _read(path: Path) -> Any:  # noqa: ANN401
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def slurm_limit_s(partition: str) -> float | None:
    """Return the partition's time limit in seconds from `sinfo`, None for no limit, raising when unreadable."""
    out = subprocess.run(
        ["sinfo", "-h", "-p", partition, "-o", "%l"], capture_output=True, text=True, check=True, timeout=60
    ).stdout.split()
    if not out:
        raise ValueError(f"sinfo lists no partition {partition!r}")
    text = out[0]
    if text.lower() in {"infinite", "unlimited"}:
        return None
    days, _, clock = text.rpartition("-")
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:  # noqa: PLR2004 - H:M:S
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def cmd_profile(args: argparse.Namespace) -> int:
    """Copy the site profile to `--out`; refuse a checked class whose time_s passes its partition's limit."""
    document = load_yaml(args.src, "site profile")
    classes = document["resource_classes"]
    chosen = set(args.classes.split(",")) if args.classes else set(classes)
    refused = []
    for name, spec in classes.items():
        if args.worker_time_s is not None:
            spec["time_s"] = min(spec["time_s"], args.worker_time_s)
        if name not in chosen:
            continue
        partition = os.path.expandvars(str(spec.get("partition", "")))
        if not partition or "$" in partition:
            refused.append(f"{name} names partition {partition!r}, which is empty or an unset variable")
            continue
        limit = slurm_limit_s(partition)
        print(f"class {name}: time_s {spec['time_s']}, partition {partition!r}, partition limit {limit}")
        if limit is not None and spec["time_s"] > limit:
            refused.append(f"{name} asks {spec['time_s']} s of a partition that allows {limit:.0f} s")
    if refused:
        print("REFUSED: " + "; ".join(refused) + "; set WORKER_TIME_S to a value the partition allows", file=sys.stderr)
        return 2
    args.out.write_text(dump_yaml(document), encoding="utf-8")
    return 0


def _keyed_outputs(cache: Path, plan_hash: str) -> dict[str, tuple[str, str]]:
    """Return each record id's (key digest, output identity) from the run's terminal events under cache root `cache`."""
    from senselab_core.runtime.execution.events import (  # noqa: PLC0415
        RESULTS,
        read_events,
        run_dir_of,
        terminal_events,
    )
    from senselab_core.runtime.execution.run import load_plan  # noqa: PLC0415

    os.environ["SENSELAB_CACHE"] = str(cache)
    run_dir = run_dir_of(plan_hash)
    rows = load_plan(run_dir).manifest.rows
    terminal = terminal_events(read_events(run_dir))
    return {
        rows[element[0]].id: (e.key, e.output)
        for (_, element), e in terminal.items()
        if e.kind in RESULTS and element and e.key is not None and e.output is not None
    }


def _array_differences(
    laptop: dict[str, Any], run: dict[str, Any], laptop_cache: Path | None, cache: Path | None
) -> list[str]:
    """Return what separates the laptop run from this one.

    That is another step or plan hash, a record one side lacks, and each record whose outputs disagree under the run
    config's thresholds (`runtime.compare`; series data by `samples`).

    Equal identities agree without a read. A record whose identities differ is read from both cache roots' result
    stores, under the key this run's events give it, and compared array by array.
    """
    from senselab_core.runtime.compare import RunConfigFile, compare  # noqa: PLC0415
    from senselab_core.runtime.execution.run import RESULTS_DIR  # noqa: PLC0415
    from senselab_core.runtime.store import ResultStore  # noqa: PLC0415

    lines = [f"step: {laptop['step']} != {run['step']}"] if laptop["step"] != run["step"] else []
    lines += [f"hash: {laptop['hash']} != {run['hash']}"] if laptop["hash"] != run["hash"] else []
    left, right = laptop["identities"], run["identities"]
    lines += [f"{i}: only on the laptop" for i in sorted(left.keys() - right.keys())]
    lines += [f"{i}: only on this run" for i in sorted(right.keys() - left.keys())]
    differing = sorted(i for i in left.keys() & right.keys() if left[i] != right[i])
    if lines or not differing:
        return lines
    if laptop_cache is None or cache is None:
        return [f"{len(differing)} identities differ; give --laptop-cache and --cache to compare their arrays"]
    thresholds = RunConfigFile.load().thresholds
    keys = _keyed_outputs(cache, run["hash"])
    stores = ResultStore(laptop_cache / RESULTS_DIR), ResultStore(cache / RESULTS_DIR)
    agreed = 0
    for record in differing:
        key = keys[record][0]
        ours, theirs = stores[0].outputs_of(key, left[record]), stores[1].outputs_of(key, right[record])
        if ours is None or theirs is None:
            lines.append(
                f"{record}: outputs not in {'the laptop' if ours is None else 'this run'}'s store under key {key}"
            )
            continue
        found = compare(ours, theirs, thresholds)
        if found.agree:
            agreed += 1
        else:
            lines += [f"{record}: {d.path} differs by {d.absolute} under {d.threshold}" for d in found.differences[:3]]
    print(
        f"{len(differing)} of {len(right)} identities differ; {agreed} of them agree under the run config's thresholds"
    )
    return lines


def cmd_identity(args: argparse.Namespace) -> int:
    """Verdict of an identity step: plan, run, hashes, counts, identities and (Step 3) the H100 placement."""
    check = Checks()
    plan_text = args.plan_log.read_text(encoding="utf-8") if args.plan_log.is_file() else ""
    seen = re.search(rf"one plan hash ({HASH.pattern})", plan_text)
    check(
        "plan: one hash under every profile, differences only in placement",
        args.plan_status == 0 and bool(seen),
        plan_text[-300:],
    )
    run = _read(args.run_json)
    check("run wrote its identities", run is not None, f"run exit {args.run_status}")
    if run is not None:
        counts, items = run["counts"], run["items"]
        check("every item done, none failed", counts.get("done") == items and not counts.get("failed"), str(counts))
        check(
            "an identity for every item",
            len(run["identities"]) == items == args.items,
            f"{len(run['identities'])} of {items}",
        )
        check(
            "run hash equals the plan's",
            seen is not None and run["hash"] == seen.group(1),
            f"{run['hash']} vs {seen and seen.group(1)}",
        )
        if args.expect_hash:
            check(
                "run hash equals the expected hash",
                run["hash"] == args.expect_hash,
                f"{run['hash']} vs {args.expect_hash}",
            )
        if args.laptop is not None:
            other = _read(args.laptop)
            differences = (
                _array_differences(other, run, args.laptop_cache, args.cache)
                if other
                else ["laptop identities file missing"]
            )
            check(
                "outputs agree with the laptop's under the run config's `samples` threshold (runtime.compare)",
                not differences,
                "; ".join(differences[:5]),
            )
        if args.require_class:
            classes = {p["resource_class"] for p in run["placement"].values()}
            check(
                f"placed on {args.require_class}", any(c.startswith(args.require_class) for c in classes), str(classes)
            )
        print(f"wall_s {run['wall_s']:.0f}, watch {run['watch']}, placement {run['placement']}")
    return check.verdict(args.out, hash=run and run["hash"], wall_s=run and run["wall_s"])


def cmd_interrupt(args: argparse.Namespace) -> int:
    """Verdict of Step 4's start job: the run was interrupted part way, at or past the share."""
    check = Checks()
    start = _read(args.start_json)
    check("start wrote its report", start is not None, f"start exit {args.status}")
    if start is not None:
        target = math.ceil(start["items"] / 3)
        check("the run hash is one hash", bool(HASH.fullmatch(start["hash"])), start["hash"])
        check(
            "interrupted part way, at or past a third",
            target <= start["committed"] < start["items"],
            f"{start['committed']} of {start['items']}",
        )
        print(f"hash {start['hash']}, committed {start['committed']} of {start['items']}, cancelled {start['workers']}")
    return check.verdict(args.out, hash=start and start["hash"], committed=start and start["committed"])


def cmd_resume(args: argparse.Namespace) -> int:
    """Verdict of Step 4's resume job: executions plus late hits are the items not committed; the rest are reuses."""
    check = Checks()
    resumed, start = _read(args.resume_json), _read(args.start_json)
    check(
        "resume wrote its accounting (the driver asserts no re-execution and no ref resolution)",
        resumed is not None,
        f"resume exit {args.status}",
    )
    if resumed is not None and start is not None:
        left = resumed["items"] - resumed["committed"]
        check(
            "executions + late hits = items not committed before the resume",
            resumed["executions"] + resumed["late"] == left,
            str(resumed),
        )
        check(
            "reuses = items - executions", resumed["reuses"] == resumed["items"] - resumed["executions"], str(resumed)
        )
        check(
            "no fewer committed than at the interrupt",
            resumed["committed"] >= start["committed"],
            f"{resumed['committed']} < {start['committed']}",
        )
        check(
            "the resume is of the interrupted run",
            resumed["hash"] == start["hash"],
            f"{resumed['hash']} vs {start['hash']}",
        )
        print(f"accounting {resumed}")
    return check.verdict(args.out, accounting=resumed)


def cmd_suite(args: argparse.Namespace) -> int:
    """Verdict of one tier: its junit file holds tests, none failed or errored, and the pytest exit status is 0."""
    check = Checks()
    root = ET.parse(args.junit).getroot() if args.junit.is_file() else None
    check(f"{args.tier}: pytest exited 0", args.status == 0, f"exit {args.status}")
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    if root is not None:
        for suite in root.iter("testsuite"):
            for key in counts:
                counts[key] += int(suite.get(key, 0))
    check(f"{args.tier}: tests ran", counts["tests"] > counts["skipped"], str(counts))
    check(f"{args.tier}: none failed or errored", counts["failures"] + counts["errors"] == 0, str(counts))
    print(f"{args.tier} {counts}")
    return check.verdict(args.out, tier=args.tier, counts=counts)


ENV_IMPORTS = {
    "crisperwhisper": ("isolated", ["torch", "crisperwhisper"]),
    "pii-spans": ("isolated", ["torch", "transformers"]),
    "pyannote": ("isolated", ["torch", "pyannote.audio"]),
    "speechbrain": ("isolated", ["torch", "speechbrain"]),
    "vllm-server": ("server", ["vllm"]),
}


def cmd_envs(args: argparse.Namespace) -> int:
    """Build each shipped lock (`inputs.yaml`) under the profile's env roots and import its packages in it."""
    from senselab_core.kernel.declare import Env  # noqa: PLC0415
    from senselab_core.runtime.envs import Uv, ensure_env, read_lock_inputs  # noqa: PLC0415
    from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415

    lock_dir = CHECKOUT / "packages/senselab/src/senselab/data/env_locks"
    profile = ExecutionProfile.load(args.profile)
    names = sorted(read_lock_inputs(lock_dir / "inputs.yaml"))
    check = Checks()
    check(
        "every shipped lock is covered by this driver",
        set(names) == set(ENV_IMPORTS),
        f"{names} vs {sorted(ENV_IMPORTS)}",
    )
    for name in names:
        kind, modules = ENV_IMPORTS.get(name, ("isolated", []))
        env = Env.server(name) if kind == "server" else Env.isolated(name)
        root = profile.env_root("server" if kind == "server" else "isolated")
        try:
            handle = ensure_env(
                env,
                lock_dir / f"{name}.txt",
                root.expanduser(),
                settings=profile.envs,
                lease=profile.lease,
                uv=Uv.find(),
            )
        except Exception as error:  # noqa: BLE001 - a build failure is the check's finding
            check(f"env {name} builds", False, f"{type(error).__name__}: {str(error)[-400:]}")
            continue
        check(f"env {name} builds", True)
        probe = (
            "import importlib,sys,json; print(json.dumps({m: bool(importlib.import_module(m)) for m in sys.argv[1:]}))"
        )
        done = subprocess.run(
            [handle.python, "-c", probe, *modules], capture_output=True, text=True, timeout=600, check=False
        )
        check(f"env {name} imports {modules}", done.returncode == 0, done.stderr[-400:])
        print(
            f"env {name}: python {handle.python_version}, torch index {handle.torch_index}, "
            f"wheel_set {handle.wheel_set[:16]}"
        )
    return check.verdict(args.out)


R253_JOBS = (2, 4, 8, 16, 32, 64)
"""The jobs of a probe's single item (`calibration.single_item_jobs`): the item sizes whose `a` is measured."""
R253_QUANTA = (4, 256)
"""`pool.quantum_items` values run beside the profile's own, at the smallest single item, to see whether it matters."""


def item_costs(units: Sequence[dict[str, Any]]) -> list[tuple[int, float]]:
    """Return (jobs, a) for every whole probe item whose requests were all ok: a = item wall - sum of request walls."""
    found = []
    for unit in units:
        items: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for o in unit["observations"]:
            if o["probe"] is not None:
                items.setdefault((o["probe"]["id"], o["item"]), []).append(o)
        for requests in items.values():
            walls = [o["item_wall_s"] for o in requests if o["item_wall_s"] is not None]
            if walls and all(o["outcome"] == "ok" for o in requests):
                found.append(
                    (sum(o["jobs"] for o in requests), max(0.0, walls[0] - sum(o["wall_s"] for o in requests)))
                )
    return found


def summarize(pairs: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Return a per job count (n, median, quartiles) and the least-squares slope of a on jobs, with its error."""
    by: dict[int, list[float]] = {}
    for jobs, a in pairs:
        by.setdefault(jobs, []).append(a)
    rows = {}
    for jobs in sorted(by):
        values = sorted(by[jobs])
        q = statistics.quantiles(values, n=4) if len(values) >= 2 else [values[0]] * 3  # noqa: PLR2004
        rows[jobs] = {
            "n": len(values),
            "median_s": q[1],
            "q1_s": q[0],
            "q3_s": q[2],
            "min_s": values[0],
            "max_s": values[-1],
        }
    slope = stderr = None
    if len(by) >= 2 and len(pairs) > 2:  # noqa: PLR2004
        xs = [float(j) for j, _ in pairs]
        ys = [a for _, a in pairs]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
        resid = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys, strict=True))
        stderr = math.sqrt(resid / (len(xs) - 2) / sxx)
    return {"per_jobs": rows, "slope_s_per_job": slope, "slope_stderr": stderr}


def cmd_r253(args: argparse.Namespace) -> int:
    """Run the signal.resample point over the grid on a local pool; report `a` per job count, pooled and per run."""
    import ladder  # noqa: PLC0415
    from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415

    site = load_yaml(args.profile, "site profile")
    laptop = load_yaml(
        CHECKOUT / "packages/senselab-core/src/senselab_core/runtime/data/profiles/laptop.yaml", "laptop"
    )
    check = Checks()
    pooled: list[tuple[int, float]] = []
    report: dict[str, Any] = {}
    base = site["pool"]["quantum_items"]
    grid = [(base, single, rep) for single in R253_JOBS for rep in range(args.reps)] + [
        (q, R253_JOBS[0], 0) for q in R253_QUANTA
    ]
    for quantum, single, rep in grid:
        label = f"q{quantum}-j{single}-r{rep}"
        document = json.loads(json.dumps(site))
        document["substrate"] = laptop["substrate"]
        document["pool"]["quantum_items"] = quantum
        document["calibration"]["single_item_jobs"] = single
        document["resource_classes"]["cpu"]["max_workers"] = args.workers
        path = args.work / f"profile-{label}.yaml"
        path.write_text(dump_yaml(document), encoding="utf-8")
        suite = args.work / f"suite-{label}.yaml"
        suite.write_text(
            dump_yaml(
                {
                    "format": "scale-suite@2.0",
                    "name": f"r253-{label}",
                    "seed": 0,
                    "long_s": 7200,
                    "calibration": {"length_strata": 3, "batched_sizes": 2},
                    "point_wall_s": 900,
                    "tasks": [
                        {
                            "task": "signal.resample",
                            "port": "series",
                            "data": "synthetic",
                            "params": {"target_rate": 8000},
                            "points": [args.items],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        os.environ["SENSELAB_CACHE"] = str(args.work / f"cache-{label}")
        try:
            results = ladder.run(suite, ExecutionProfile.load(path), args.work / f"measures-{label}")
        except Exception as error:  # noqa: BLE001 - a failed grid point is a finding
            check(f"{label} ran", False, f"{type(error).__name__}: {str(error)[-300:]}")
            continue
        check(f"{label} ran", bool(results) and results[0]["done"] is True, "not done")
        pairs = item_costs([u for r in results for u in r["units"]])
        check(f"{label} has whole probe items with an item wall", len(pairs) > 0, "none")
        report[label] = {"quantum_items": quantum, "single_item_jobs": single, "rep": rep, **summarize(pairs)}
        print(f"{label}: {json.dumps(report[label]['per_jobs'], sort_keys=True)}", flush=True)
        pooled += pairs
    report["pooled"] = summarize(pooled)
    counts = report["pooled"]["per_jobs"]
    check("every single-item size measured", {1, *R253_JOBS} <= set(counts), str(sorted(counts)))
    (args.work / "r253.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("R253 a per job count (n, median s):", {j: (r["n"], round(r["median_s"], 5)) for j, r in counts.items()})
    print(
        "R253 slope of a on jobs (s per job):",
        report["pooled"]["slope_s_per_job"],
        "+-",
        report["pooled"]["slope_stderr"],
    )
    return check.verdict(
        args.out, slope_s_per_job=report["pooled"]["slope_s_per_job"], slope_stderr=report["pooled"]["slope_stderr"]
    )


def cmd_crossnode(args: argparse.Namespace) -> int:
    """Verdict of the Task 12 (lease) and Task 13 (queue) probes: each verdict file says pass over three hosts."""
    check = Checks()
    for name, hosts in (("lease", 3), ("queue", 3)):
        verdict = _read(args.dir / f"{name}-verdict.json")
        check(f"{name} probe wrote a verdict", verdict is not None, "missing")
        if verdict is not None:
            check(f"{name} probe passed", verdict.get("pass") is True, json.dumps(verdict)[:400])
            seen = verdict.get("hosts")
            count = len(seen) if isinstance(seen, list) else seen
            check(f"{name} probe ran on {hosts} distinct hosts", count is None or count >= hosts, str(seen))
    return check.verdict(args.out)


def cmd_prov(args: argparse.Namespace) -> int:  # noqa: PLR0915 - one verdict, one check per line
    """Verdict of the post-hoc provenance read of Step 4's store, against the start and resume accounting.

    The completeness rules are the runtime's own (`check_provenance`, ADR-010): every committed key has an execution or
    a `senselab:Reuse`, and no writer records more Reuse of a key than the run has jobs with it. What this adds is
    Step 4's accounting. A writer document is assigned to a phase as a whole: the start phase's documents hold only
    activities that ended before the resume's first event (the first event of the run's last epoch), the resume's only
    activities that started after it. A Reuse is the coordinator's when it has the settled shape (no `senselab:context`
    and no `prov:wasAssociatedWith`: the coordinator settles an earlier epoch's outcome without running it), a worker's
    otherwise; the split is told per activity, since a worker document may hold only store hits. Resume-phase
    executions are the items neither committed at the interrupt nor late hits; `reuses` are the Reuse activities the
    resume counted. The accounting's `executions` counts `finished` events, which is not the number of executions, so
    it is not compared. A kill loses the activities of the claim in flight, so the keys without an execution activity
    number at most `quantum_items` for each cancelled worker.
    """
    from reuse_diag import activity_rows, boundary_of  # noqa: PLC0415 - needs the checkout on the path
    from senselab_core.runtime.execution.events import read_events  # noqa: PLC0415
    from senselab_core.runtime.execution.provenance_check import check_provenance, committed_keys  # noqa: PLC0415
    from senselab_core.runtime.provenance import ProvStore  # noqa: PLC0415

    check = Checks()
    run = args.run_dir
    start, resumed = _read(args.start_json), _read(args.resume_json)
    check(
        "start.json and the resume accounting exist",
        start is not None and resumed is not None,
        f"{args.start_json} {args.resume_json}",
    )
    check("the run's prov directory exists", (run / "prov").is_dir(), str(run / "prov"))
    if start is None or resumed is None or not (run / "prov").is_dir():
        return check.verdict(args.out, hash=args.hash)
    items, committed = start["items"], start["committed"]
    events = read_events(run)
    rows = activity_rows(run, boundary_of(events))
    report = check_provenance(run)
    print(f"{report.committed} committed keys, {report.executions} executions, {report.reuses} reuses")
    crossing = sorted({r.document for r in rows if r.phase == "other"})
    reuses = [r for r in rows if r.kind == "reuse"]
    executions = [r for r in rows if r.kind == "execution"]
    resume_exec = [r for r in executions if r.phase == "resume"]
    split = Counter(r.writer for r in reuses)
    print(
        f"{len(rows)} activities: {len(reuses)} senselab:Reuse ({split['coordinator']} settled by the coordinator, "
        f"{split['worker']} by workers), {len(executions)} executions ({len(executions) - len(resume_exec)} before the "
        f"resume, {len(resume_exec)} after); items {items}, committed at the interrupt {committed}; "
        f"resume accounting {resumed}"
    )
    check(
        "provenance complete (ADR-010: an execution or Reuse per committed key; one Reuse per item per attempt)",
        report.complete,
        "; ".join(report.problems[:5]),
    )
    check("no document straddles the resume's first event", not crossing, str(crossing))
    check(
        f"the run committed {items} keys (one per item)",
        report.committed == items,
        f"{report.committed} committed keys",
    )
    lost = committed_keys(run) - {r.key for r in executions}
    lost_allowed = args.quantum_items * len(start["workers"])
    check(
        f"keys without an execution activity <= {lost_allowed} (ADR-010: a kill loses the claim in flight, at most "
        f"quantum_items {args.quantum_items} per cancelled worker)",
        len(lost) <= lost_allowed,
        f"{len(lost)}",
    )
    expected = items - committed - resumed["late"]
    check(
        f"resume-phase executions = items - committed - late start executions = {expected}",
        len(resume_exec) == expected,
        f"{len(resume_exec)}",
    )
    check(
        f"senselab:Reuse activities = the resume's reuses ({resumed['reuses']})",
        len(reuses) == resumed["reuses"],
        f"{len(reuses)} ({split['coordinator']} coordinator, {split['worker']} worker)",
    )
    plans = [r.id for r in ProvStore.from_documents(ProvStore.read_documents(run)).plans()]
    check("exactly one plan entity", plans == [f"plan:{args.hash}"], str(plans))
    claims = Counter((e.epoch, e.job.element) for e in events if e.kind == "claimed" and e.job is not None)
    twice_claimed = Counter(epoch for (epoch, _), n in claims.items() if n > 1)
    print(
        f"informational: items claimed twice within one attempt (epoch), from events: {dict(twice_claimed) or 'none'}"
    )
    return check.verdict(
        args.out,
        hash=args.hash,
        activities=len(rows),
        reuses=len(reuses),
        reuses_coordinator=split["coordinator"],
        reuses_worker=split["worker"],
        executions=len(executions),
        resume_executions=len(resume_exec),
        lost_keys=len(lost),
        items_claimed_twice={str(k): v for k, v in twice_claimed.items()},
    )


def cmd_agent_prepared(args: argparse.Namespace) -> int:
    """Verdict of Step 6's prepare phase: the agent's input is written and describes every task."""
    check = Checks()
    document = _read(args.input)
    check("agent-input.json written", args.status == 0 and document is not None, f"exit {args.status}")
    if document is not None:
        check(
            "describe holds every task",
            len(document.get("describe", {})) >= args.tasks,
            str(len(document.get("describe", {}))),
        )
        print(f"agent input: {args.input}")
    return check.verdict(args.out)


def cmd_agent_ran(args: argparse.Namespace) -> int:
    """Verdict of Step 6's run phase: the agent's call validated, planned and ran."""
    check = Checks()
    result = _read(args.result) or {}
    check(
        "the agent's call ran",
        bool(result.get("ok")),
        f"{result.get('error') or result.get('counts')} (exit {args.status})",
    )
    return check.verdict(args.out, call=result.get("call"))


def cmd_watch(args: argparse.Namespace) -> int:
    """Run the command after `--` under the no-progress watchdog; exit with its status, or 3 and a failing verdict.

    With `--runs-glob` (the run directories the command creates) and `--profile`, the watchdog counts only the Slurm
    jobs of those runs, from the moment a run directory exists; until then, and without it, any job of `--prefix`.
    """
    import stall  # noqa: PLC0415 - found through the checkout

    argv = args.argv_rest[1:] if args.argv_rest[:1] == ["--"] else args.argv_rest
    if not argv:
        print("watch needs a command after --", file=sys.stderr)
        return 2
    prefixes = stall.run_prefixes(args.runs_glob, args.profile) if args.runs_glob else None
    done = stall.watched(
        argv,
        args.log,
        stall_s=args.stall_s,
        poll_s=args.poll_s,
        prefix=args.prefix,
        timeout_s=args.timeout_s,
        prefixes=prefixes,
    )
    if done.stalled is not None:
        stall.write_stall_verdict(args.verdict, done.stalled)
    return done.status


def cmd_prefix(args: argparse.Namespace) -> int:
    """Print the start of the name of every Slurm job of Step 4's run `--hash`."""
    import exitlib  # noqa: PLC0415 - found through the checkout
    import interrupt_resume  # noqa: PLC0415
    from senselab_core.runtime.execution.events import run_dir_of  # noqa: PLC0415

    exitlib.isolate(args.work, interrupt_resume.STEP)
    print(exitlib.slurm_prefix(exitlib.load_profile(str(args.profile)), run_dir_of(args.hash), args.hash))
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    """Print the value at the dotted `--key` of a JSON or YAML file; exit 1 when it is absent."""
    value: Any = load_yaml(args.file, "document")
    for part in args.key.split("."):
        if not isinstance(value, dict) or part not in value:
            print(f"{args.file} has no {args.key}", file=sys.stderr)
            return 1
        value = value[part]
    print(value)
    return 0


def cmd_tree(args: argparse.Namespace) -> int:
    """Lay a synthetic BIDS tree of `--items` recordings under `--out`."""
    import bids_tree  # noqa: PLC0415 - found through the checkout

    bids_tree.synthetic(args.out, args.items, args.seed)
    return 0


def cmd_combine(args: argparse.Namespace) -> int:
    """One verdict over every `*-verdict.json` of `--dir`: it passes when each of them does and at least one exists."""
    check = Checks()
    files = sorted(args.dir.glob("*-verdict.json"))
    check("some verdicts exist", bool(files), f"none under {args.dir}")
    for path in files:
        verdict = _read(path)
        check(path.stem.removesuffix("-verdict"), bool(verdict and verdict.get("pass")), json.dumps(verdict)[:300])
    return check.verdict(args.out, parts=[p.name for p in files])


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of every command."""
    parser = argparse.ArgumentParser(prog="orcd_driver.py")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, function: Callable[[argparse.Namespace], int], *flags: tuple[str, dict[str, Any]]) -> None:
        sub = commands.add_parser(name)
        sub.set_defaults(function=function)
        for flag, options in flags:
            sub.add_argument(flag, **options)

    path: dict[str, Any] = {"type": Path, "required": True}
    status: dict[str, Any] = {"type": int, "required": True}
    add(
        "profile",
        cmd_profile,
        ("--src", path),
        ("--out", path),
        ("--classes", {}),
        ("--worker-time-s", {"type": float}),
    )
    add(
        "identity",
        cmd_identity,
        ("--out", path),
        ("--plan-log", path),
        ("--plan-status", status),
        ("--run-json", path),
        ("--run-status", status),
        ("--items", status),
        ("--expect-hash", {}),
        ("--laptop", {"type": Path}),
        ("--laptop-cache", {"type": Path}),
        ("--cache", {"type": Path}),
        ("--require-class", {}),
    )
    add("interrupt", cmd_interrupt, ("--out", path), ("--start-json", path), ("--status", status))
    add("resume", cmd_resume, ("--out", path), ("--start-json", path), ("--resume-json", path), ("--status", status))
    add("suite", cmd_suite, ("--out", path), ("--tier", {"required": True}), ("--junit", path), ("--status", status))
    add("envs", cmd_envs, ("--out", path), ("--profile", path))
    add(
        "r253",
        cmd_r253,
        ("--out", path),
        ("--profile", path),
        ("--work", path),
        ("--items", {"type": int, "default": 400}),
        ("--workers", {"type": int, "default": 8}),
        ("--reps", {"type": int, "default": 4}),
    )
    add(
        "prov",
        cmd_prov,
        ("--out", path),
        ("--run-dir", path),
        ("--hash", {"required": True}),
        ("--start-json", path),
        ("--resume-json", path),
        ("--quantum-items", status),
    )
    add("combine", cmd_combine, ("--out", path), ("--dir", path))
    add("crossnode", cmd_crossnode, ("--out", path), ("--dir", path))
    add(
        "agent-prepared",
        cmd_agent_prepared,
        ("--out", path),
        ("--input", path),
        ("--status", status),
        ("--tasks", {"type": int, "default": 10}),
    )
    add("agent-ran", cmd_agent_ran, ("--out", path), ("--result", path), ("--status", status))
    add(
        "watch",
        cmd_watch,
        ("--log", path),
        ("--verdict", path),
        ("--prefix", {"default": "senselab-"}),
        ("--runs-glob", {}),
        ("--profile", {"type": Path}),
        ("--stall-s", {"type": float, "default": 1800.0}),
        ("--poll-s", {"type": float, "default": 60.0}),
        ("--timeout-s", {"type": float, "default": 60.0}),
        ("argv_rest", {"nargs": argparse.REMAINDER}),
    )
    add("prefix", cmd_prefix, ("--work", path), ("--profile", path), ("--hash", {"required": True}))
    add("get", cmd_get, ("--file", path), ("--key", {"required": True}))
    add("tree", cmd_tree, ("--out", path), ("--items", status), ("--seed", {"type": int, "default": 0}))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command."""
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    sys.exit(main())
