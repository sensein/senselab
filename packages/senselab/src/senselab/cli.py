"""The ``senselab`` command line: ``data fetch``, ``status`` and ``explain`` from core, and ``bench run``."""

import argparse
import importlib.util
import os
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from senselab_core.cli import mount_data, mount_runs
from senselab_core.runtime.fetch import FetchError, default_cache_root
from senselab_core.runtime.profiles import DEFAULT_PROFILE, ExecutionProfile
from senselab_core.runtime.revision import cache_root

BENCH_DIR = "benchmarks"


def _find_benchmarks(start: Path) -> Path:
    """Return the `benchmarks/` directory holding `harness.py`, searching `start` and its parents."""
    for parent in (start, *start.parents):
        if (parent / BENCH_DIR / "harness.py").is_file():
            return parent / BENCH_DIR
    raise FileNotFoundError(
        f"no {BENCH_DIR}/harness.py in {start} or above; run from a senselab checkout or pass --benchmarks DIR"
    )


def _load_harness(directory: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("senselab_bench_harness", directory / "harness.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile(name: str) -> ExecutionProfile:
    path = Path(name)
    return (
        ExecutionProfile.load(DEFAULT_PROFILE, path)
        if path.suffix in {".yaml", ".yml"}
        else ExecutionProfile.load(name)
    )


def run_bench(args: argparse.Namespace) -> int:
    """Run a parsed ``bench run``; return the exit code (0 on success, 1 on a reported error)."""
    try:
        directory = args.benchmarks or _find_benchmarks(Path.cwd())
        harness = _load_harness(directory)
        suite_path = Path(args.suite) if Path(args.suite).suffix else directory / "suites" / f"{args.suite}.yaml"
        suite = harness.Suite.load(suite_path)
        run = harness.run_suite(
            suite,
            execution=_profile(args.execution),
            manifests=directory / "manifests",
            data_cache=args.data_cache or default_cache_root(),
            out=args.out or cache_root() / "2x" / "benchmarks",
            run_id=os.environ.get("SENSELAB_RUN_ID") or uuid.uuid4().hex[:12],
        )
    except (ValueError, FetchError, OSError) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1
    for entry in harness.load_result(run / "result.json").entries:
        if entry.not_placeable is not None:
            print(f"senselab: not placeable: {entry.benchmark}: {entry.not_placeable}", file=sys.stderr)
    print(run / "result.json")
    return 0


def _mount_bench(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    bench = groups.add_parser("bench", help="benchmark suites")
    commands = bench.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a benchmark suite and write its result with provenance")
    run.add_argument("suite", help="a suite name under benchmarks/suites/, or a path to a suite YAML file")
    run.add_argument("--execution", default="laptop", help="a packaged execution profile name, or a profile YAML file")
    run.add_argument(
        "--benchmarks", type=Path, default=None, help="the benchmarks directory (default: found from the cwd)"
    )
    run.add_argument("--data-cache", type=Path, default=None, help="data cache (default: $SENSELAB_DATA_CACHE)")
    run.add_argument(
        "--out", type=Path, default=None, help="results directory (default: $SENSELAB_CACHE/2x/benchmarks)"
    )
    bench.set_defaults(run=run_bench)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="senselab")
    groups = parser.add_subparsers(dest="group", required=True)
    mount_data(groups)
    mount_runs(groups)
    _mount_bench(groups)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; return the process exit code (0 on success, 1 on a reported error)."""
    args = _build_parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
