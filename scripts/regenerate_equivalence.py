"""Sample verify evidence into a result store, and regenerate `runtime/data/equivalence.yaml` from stores' evidence.

`sample` runs the held-out items of one suite benchmark through `run_call` under `--reuse` and `--verify`, compared
under the thresholds of `--config` (a run config merged over the packaged one), against `--store`, and prints how many
calls executed, were reused and were verified. `regenerate` reads the `equivalence-evidence@4.0` records of every
store named and writes `equivalence@3.0`; the table is never hand-edited.
"""

import argparse
import importlib
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from senselab_core.runtime.equivalence import render
from senselab_core.runtime.store import ResultStore

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "packages" / "senselab-core" / "src" / "senselab_core" / "runtime" / "data" / "equivalence.yaml"
HELDOUT = "heldout"


def _parity_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("parity_reference", ROOT / "benchmarks" / "parity_reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def regenerate(stores: Sequence[Path], out: Path) -> Path:
    """Write the table the evidence in `stores` measures to `out`."""
    out.write_text(render(e for root in stores for e in ResultStore(root).evidence()), "utf-8")
    return out


def _sample_cli(args: argparse.Namespace) -> None:
    from senselab_core.kernel.identity import ReusePolicy  # noqa: PLC0415 - only a real run needs the runtime
    from senselab_core.runtime.call import RunConfig, TaskCall, run_call  # noqa: PLC0415
    from senselab_core.runtime.compare import Thresholds  # noqa: PLC0415
    from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415

    bench, items = _parity_script().load_items(args.suite, args.benchmark, args.data_cache, args.work, HELDOUT)
    store = ResultStore(args.store)
    before = len(store.evidence())
    profile = ExecutionProfile.load(args.execution)
    run = RunConfig(reuse=ReusePolicy(args.reuse), verify=args.verify, thresholds=Thresholds.load(args.config))
    executed = reused = 0
    for item in items:
        call = TaskCall.model_validate({"task": bench.task, "inputs": dict(item.inputs), "params": bench.params})
        result = run_call(call, execution=profile, run=run, store=store)
        executed += result.reused is None
        reused += result.reused is not None
    evidence = store.evidence()
    print(
        f"{bench.name}: {executed} executed, {reused} reused, {len(evidence) - before} verified;"
        f" the store holds {len(evidence)} evidence records, {sum(not e.agree for e in evidence)} disagreeing"
    )


def _regenerate_cli(args: argparse.Namespace) -> None:
    print(regenerate(args.stores, args.out))


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: `sample` one context into a store, `regenerate` the table from stores."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    s = sub.add_parser("sample")
    s.add_argument("--suite", type=Path, default=ROOT / "benchmarks" / "suites" / "equivalence.yaml")
    s.add_argument("--benchmark", required=True)
    s.add_argument("--execution", default="orcd")
    s.add_argument("--store", type=Path, required=True)
    s.add_argument("--reuse", default="equivalent")
    s.add_argument("--verify", type=float, default=0.0)
    s.add_argument("--config", type=Path, help="a run config YAML whose thresholds are merged over the packaged ones")
    s.add_argument("--data-cache", type=Path, required=True)
    s.add_argument("--work", type=Path, required=True)
    s.set_defaults(func=_sample_cli)
    g = sub.add_parser("regenerate")
    g.add_argument("stores", type=Path, nargs="*")
    g.add_argument("--out", type=Path, default=TABLE)
    g.set_defaults(func=_regenerate_cli)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
