"""Batch invariance probe: each benchmark's held-out items through `execute` at batch 1 and at batch N, compared.

`uv run python benchmarks/batch_invariance.py SUITE --execution orcd [--override FILE] --out DIR --sha SHA --job JOB`

For each benchmark of the suite (`benchmark-suite@1.0`), its held-out items are keyed and sent as one node's jobs,
once in batches of one and once in batches of N, each into a store of its own under `DIR`. N is the largest number of
those jobs sharing one formation (`formation` over the packaged bucket grid), so every bucket is sent whole. Each
item's outputs are compared (`compare`) under the run config's thresholds. One line per backend is printed:
`<sha> <job> <task>/<backend>: agree k/n, formation <mixed|bucket>, batch N, max diff <path>=<absolute> ...` (`none`
when nothing differs); `... : not batched — no evidence (formation <kind>, batch N)` when no request would hold more
than one item, the formation being `single` or N 1, and nothing is run; or `<sha> <job> <task>: not placeable:
<reason>` for a task no device here runs.
"""

import argparse
import importlib.util
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from senselab_core.runtime.call import RunConfig, TaskCall, bind_input, packaged, placement_of
from senselab_core.runtime.compare import compare, index_free
from senselab_core.runtime.execution.batching import BucketGrid, formation
from senselab_core.runtime.execution.executor import Admission, JobRef
from senselab_core.runtime.execution.in_process import fragment_jobs, node_fragment, process_workers
from senselab_core.runtime.execution.job import Completed, Job, JobOutcome, execute
from senselab_core.runtime.fetch import DataManifest, default_cache_root, fetch
from senselab_core.runtime.keys import SINGLE, Placement
from senselab_core.runtime.planner import plan
from senselab_core.runtime.profiles import ExecutionProfile, NoDevice
from senselab_core.runtime.store import ResultStore

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = "heldout"


def harness() -> ModuleType:
    """Return `benchmarks/harness.py`, the owner of the suites, the adapters and the item split."""
    found = sys.modules.get("benchmark_harness")
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location("benchmark_harness", ROOT / "benchmarks" / "harness.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Probe:
    """One backend's comparison: agreeing items of all held-out items, formation kind, batch N, largest gap per field.

    It is evidence only when N > 1 and the kind is not `single` (`batched`).
    """

    task: str
    backend: str
    agree: int
    items: int
    formation: str
    batch: int
    largest: Mapping[str, float | None]

    @property
    def batched(self) -> bool:
        """Whether any request held more than one item, so the comparison is evidence."""
        return self.batch > 1 and self.formation != SINGLE

    def line(self, sha: str, job: str) -> str:
        """Return the probe's report line."""
        head = f"{sha} {job} {self.task}/{self.backend}"
        shape = f"formation {self.formation}, batch {self.batch}"
        if not self.batched:
            return f"{head}: not batched — no evidence ({shape})"
        gaps = " ".join(f"{p}={'n/a' if v is None else f'{v:.3g}'}" for p, v in sorted(self.largest.items()))
        return f"{head}: agree {self.agree}/{self.items}, {shape}, max diff {gaps or 'none'}"


@dataclass(frozen=True)
class NotPlaceable:
    """A benchmark whose task this profile places on no device here."""

    task: str
    reason: str

    def line(self, sha: str, job: str) -> str:
        """Return the benchmark's report line."""
        return f"{sha} {job} {self.task}: not placeable: {' '.join(self.reason.split())}"


def node_jobs(  # noqa: PLR0913 - the call, then keyword-only run inputs
    task: str,
    inputs: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    model: str | None,
    *,
    run: RunConfig,
    execution: ExecutionProfile,
) -> tuple[list[Job], Placement]:
    """Return one job per input mapping of one call of `task`, as one node's jobs, and the placement they share.

    Each call is planned as `InProcess.call` plans it.

    Raises:
        ValueError: `inputs` is empty.
    """
    spec = packaged().registry.spec(task)
    jobs: list[Job] = []
    placement = None
    for given in inputs:
        call = TaskCall.model_validate({"task": task, "inputs": given, "params": params, "model": model})
        bound = {port: bind_input(port, spec.inputs[port], value)[:2] for port, value in call.inputs.items()}
        values = {port: value for port, (_, value) in bound.items()}
        placed = plan(call.model_copy(update={"inputs": values}), config=run, execution=execution)
        fragment = node_fragment(placed, Admission(id="probe", plan_hash=placed.hash))
        [job], placement = fragment_jobs(fragment, placed, dict(bound.values()))
        jobs.append(job)
    if placement is None:
        raise ValueError(f"{task!r}: no held-out item to probe")
    plan_hash = jobs[0].ref.plan_hash
    return [
        replace(j, ref=JobRef(plan_hash=plan_hash, node="probe", element=(i,))) for i, j in enumerate(jobs)
    ], placement


def probe(jobs: list[Job], placement: Placement, *, run: RunConfig, execution: ExecutionProfile, work: Path) -> Probe:
    """Run `jobs` at batch 1 and at the largest formation population, and compare each item's outputs.

    Nothing runs when no request would hold more than one item (`Probe.batched`).
    """
    grid = BucketGrid.load_default()
    [(top, batch)] = Counter(formation(j, placement.backend, grid) for j in jobs).most_common(1)
    kind = top.split(":")[0]
    backend = placement.backend
    unrun = Probe(backend.task, backend.name, 0, len(jobs), kind, batch, {})
    if not unrun.batched:
        return unrun
    workers = process_workers(execution)

    def sent(size: int) -> list[JobOutcome]:
        store = ResultStore(work / f"batch-{size}")
        return execute(
            jobs, replace(placement, batch_size=size), store=store, run=run, profile=execution, workers=workers
        )

    agree, largest = 0, dict[str, float | None]()
    for one, many in zip(sent(1), sent(batch), strict=True):
        if not (isinstance(one, Completed) and isinstance(many, Completed)):
            continue
        result = compare(one.result.outputs, many.result.outputs, run.thresholds)
        agree += result.agree
        for d in result.differences:
            path = index_free(d.path)
            gaps = [g for g in (largest.get(path), d.absolute) if g is not None]
            largest[path] = max(gaps) if gaps else None
    return Probe(backend.task, backend.name, agree, len(jobs), kind, batch, largest)


def probe_suite(
    suite: Path, *, execution: ExecutionProfile, out: Path, manifests: Path, data_cache: Path
) -> list[Probe | NotPlaceable]:
    """Return one `Probe` per benchmark of `suite` over its held-out items, or why its task is not placeable."""
    bench = harness()
    run = RunConfig()
    probes: list[Probe | NotPlaceable] = []
    for entry in bench.Suite.load(suite).benchmarks:
        try:
            placement_of(entry.task, TaskCall.parse_model(entry.model), execution)
        except NoDevice as err:
            probes.append(NotPlaceable(entry.task, str(err)))
            continue
        files = (
            {}
            if entry.manifest is None
            else fetch(DataManifest.load(manifests / f"{entry.manifest}.yaml"), cache_root=data_cache)
        )
        items = bench.ADAPTERS[entry.adapter]().items(files, entry.items, out / "work" / entry.name, entry.max_seconds)
        held = [i.inputs for i in items if i.split == HELDOUT and i.fault is None]
        jobs, placement = node_jobs(entry.task, held, entry.params, entry.model, run=run, execution=execution)
        probes.append(probe(jobs, placement, run=run, execution=execution, work=out / entry.name))
    return probes


def main(argv: Sequence[str] | None = None) -> int:
    """Probe every benchmark of the suite and print one line per backend."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--execution", default="laptop")
    parser.add_argument("--override", type=Path, default=None, help="profile YAML deep-merged over --execution")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--manifests", type=Path, default=ROOT / "benchmarks" / "manifests")
    parser.add_argument("--data-cache", type=Path, default=None, help="data cache (default: $SENSELAB_DATA_CACHE)")
    args = parser.parse_args(argv)
    probes = probe_suite(
        args.suite,
        execution=ExecutionProfile.load(args.execution, args.override),
        out=args.out,
        manifests=args.manifests,
        data_cache=args.data_cache or default_cache_root(),
    )
    for found in probes:
        print(found.line(args.sha, args.job))
    return 0


if __name__ == "__main__":
    sys.exit(main())
