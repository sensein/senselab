"""One model load for N jobs: a task mapped over N recordings on one warm pool worker, checked from its store entries.

`uv run python benchmarks/scale/one_load.py --profile PROFILE --audio DIR --items N --task TASK [--stack STACK]
[--local] [--out FILE]`

`PROFILE` is a whole profile file (YAML or JSON) whose one class of the task's device has `max_workers: 1`; its
substrate starts the worker (`slurm`: one `sbatch` with the class's typed `--gres`, from the job this runs in), or, with
`--local`, a process of this allocation does. The first N audio files of `DIR` (sorted, `.wav` or
`.flac`) are the records. The run makes its own pool over its run directory and closes it when it ends. Each executed
job's store entry records the pool facts of the call that computed it (`pool-facts@1`): the check is that every job
completed, on one worker, under one residency, with exactly one cold call (the load) and N-1 warm ones. One JSON
document is printed, with the run's jobs by state, failures by fault class and unplaceable messages beside the check,
and written to `FILE` when given; the exit status is 0 only when the check passes.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from senselab_core.runtime.call import TaskCall, packaged
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.datasets import from_records
from senselab_core.runtime.planner import one_node_graph, plan
from senselab_core.runtime.profiles import ExecutionProfile


def _facts(run: Any, keys: Sequence[str]) -> list[dict[str, Any]]:  # noqa: ANN401 - a Run
    """Return the pool facts of the store entry of each key digest the run gave."""
    store = run.store()
    found = []
    for key in keys:
        found += [context["pool"] for _, context, _ in store.stored(key) if "pool" in context]
    return found


def check(facts: Sequence[dict[str, Any]], items: int) -> dict[str, Any]:
    """Return the verdict over the pool facts of `items` executed jobs: one worker, one residency, one cold call."""
    states = Counter(r["state"] for f in facts for r in f.get("residency", []))
    workers = sorted({f["worker"] for f in facts})
    residencies = sorted({r["key"] for f in facts for r in f.get("residency", [])})
    passed = len(facts) == items and len(workers) == 1 and len(residencies) == 1 and states.get("cold") == 1
    return {
        "pass": passed,
        "items": items,
        "entries": len(facts),
        "workers": len(workers),
        "residencies": len(residencies),
        "cold": states.get("cold", 0),
        "warm": states.get("warm", 0),
        "classes": sorted({f["class"] for f in facts}),
        "hardware": sorted({json.dumps(f.get("hardware"), sort_keys=True) for f in facts}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Plan the task over the recordings, run it on the profile's pool, check its entries and print the verdict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--profile", type=Path, required=True, help="a whole execution profile file")
    parser.add_argument("--audio", type=Path, required=True, help="a directory of .wav or .flac recordings")
    parser.add_argument("--items", type=int, required=True, help="how many recordings to map the task over")
    parser.add_argument("--task", default="speaker_embeddings", help="the task, whose first input is audio")
    parser.add_argument("--stack", default=None, help="the stack to prefer for the task")
    parser.add_argument("--local", action="store_true", help="start the worker on this host, not by the substrate")
    parser.add_argument("--out", type=Path, default=None, help="where to write the verdict")
    args = parser.parse_args(argv)
    profile = ExecutionProfile.model_validate(load_yaml(args.profile, "execution profile"))
    if args.local:
        profile = profile.model_copy(update={"substrate": profile.substrate.model_copy(update={"kind": "local"})})
    if args.stack is not None:
        preferences = {**profile.stack_preferences, args.task: (args.stack,)}
        profile = profile.model_copy(update={"stack_preferences": preferences})
    files = sorted(p for p in args.audio.iterdir() if p.suffix in {".wav", ".flac"})[: args.items]
    if len(files) < args.items:
        parser.error(f"{args.audio} holds {len(files)} recordings, fewer than {args.items}")
    registry = packaged().registry
    spec = registry.spec(args.task)
    port = next(iter(spec.inputs))
    task = TaskCall.model_validate({"task": args.task, "task_version": spec.version, "inputs": {}, "params": {}})
    graph = one_node_graph(task, registry, every_port=True)
    rows = [{"id": f"r{i}", "inputs": {port: path}} for i, path in enumerate(files)]
    placed = plan(graph, from_records(rows), execution=profile)
    run = placed.run()
    status = run.wait()
    keys = sorted({k for _, given in run.results() for k in getattr(given, "keys", {}).values()})
    verdict = {"hash": placed.hash, "done": status.done, **check(_facts(run, keys), args.items)}
    verdict.update(jobs=dict(status.counts), faults=dict(status.faults), unplaceable=list(status.unplaceable))
    verdict["pass"] = bool(verdict["pass"] and status.done)
    text = json.dumps(verdict, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
