"""Step 6: a fresh agent plans from `describe()` alone; this runs the plan it writes.

`agent_plan.py prepare --dataset PATH --out FILE` writes the agent's input, `describe` of every task and the dataset
path, with the instructions in `agent_instructions.txt`.
`run --call FILE --work DIR --profile P --executor E --out FILE` validates the agent's call, plans it over the dataset
and runs it, and writes what happened, including a refusal.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import exitlib  # noqa: E402 - beside this script
from senselab._api.call import ConfiguredTask  # noqa: E402

INSTRUCTIONS = Path(__file__).resolve().parent / "agent_instructions.txt"
INPUT_FORMAT = "agent-input@1.0"
CALL_FORMAT = "agent-plan@1.0"
RESULT_FORMAT = "agent-result@1.0"


class BadCall(ValueError):
    """The agent's call is not a runnable `agent-plan@1.0` document; the message says why."""


def agent_input(dataset: Path) -> dict[str, Any]:
    """Return what the agent is given: the instructions, the dataset path and `describe` of every task."""
    import senselab  # noqa: PLC0415 - the agent's own toolkit

    if not dataset.is_dir():
        raise ValueError(f"the dataset {dataset} is not a directory")
    return {
        "format": INPUT_FORMAT,
        "instructions": INSTRUCTIONS.read_text(encoding="utf-8"),
        "dataset": str(dataset),
        "describe": {t["name"]: senselab.describe(t["name"]) for t in senselab.tasks()},
    }


def task_classes() -> dict[str, type[ConfiguredTask]]:
    """Return the generated task classes by declared task name and by class name."""
    import senselab  # noqa: PLC0415 - the agent's own toolkit

    found: dict[str, type[ConfiguredTask]] = {}
    for name in senselab.generated.__all__:
        each = getattr(senselab.generated, name)
        declared = getattr(each, "name", None)
        if isinstance(each, type) and issubclass(each, ConfiguredTask) and isinstance(declared, str):
            found[declared] = found[each.__name__] = each
    return found


def configured(call: Any) -> tuple[Any, Path]:  # noqa: ANN401 - the agent's JSON, checked here
    """Return the configured task and dataset path an `agent-plan@1.0` document names.

    Raises:
        BadCall: the document is malformed, names no task or no dataset, or its params and model do not construct.
    """
    if not isinstance(call, dict) or call.get("format") != CALL_FORMAT:
        raise BadCall(f"the call is not a {CALL_FORMAT} object")
    classes = task_classes()
    task = classes.get(str(call.get("task")))
    if task is None:
        raise BadCall(f"task {call.get('task')!r} is none of {sorted({c.name for c in classes.values()})}")
    params, model, dataset = call.get("params", {}), call.get("model"), call.get("dataset")
    if not isinstance(params, dict) or not isinstance(dataset, str) or not Path(dataset).is_dir():
        raise BadCall("params is an object and dataset is the path of a directory")
    try:
        return task(model, **params), Path(dataset)
    except Exception as error:  # noqa: BLE001 - the task's own typed refusal is the agent's evidence
        raise BadCall(f"{type(error).__name__}: {error}") from error


def run_call(
    call_path: Path, profile_argument: str, executor: str, work: Path, config: exitlib.Config
) -> dict[str, Any]:
    """Run the agent's call over its dataset and return the `agent-result@1.0` document; a refused call is a result.

    Raises:
        BoundExceeded, FileExistsError: as `exitlib.execute`.
    """
    import senselab  # noqa: PLC0415 - imported after the cache root is set

    base = exitlib.load_profile(profile_argument)
    profile = exitlib.profile_for(executor, base, config)
    exitlib.isolate(work, f"agent-{base.name}-{executor}")
    result: dict[str, Any] = {"format": RESULT_FORMAT, "call": None, "ok": False}
    try:
        call = json.loads(call_path.read_text(encoding="utf-8"))
        result["call"] = call
        task, dataset = configured(call)
        placed = senselab.plan(task, senselab.Dataset.bids(dataset), execution=profile)
    except Exception as error:  # noqa: BLE001 - a refusal is recorded, not raised
        return {**result, "error": {"type": type(error).__name__, "message": str(error)}}
    done = exitlib.execute(placed, executor, config)
    identities = exitlib.output_identities(done.events, placed.manifest.rows)
    failed = done.counts.get("failed", 0)
    return {
        **result,
        "ok": failed == 0 and len(identities) == len(placed.manifest.rows),
        "hash": placed.hash,
        "items": len(placed.manifest.rows),
        "counts": dict(done.counts),
        "identities": identities,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the step's command line; return 1 when the agent's call was refused or failed."""
    parser = argparse.ArgumentParser(prog="agent_plan.py")
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="write the agent's input")
    preparing.add_argument("--dataset", type=Path, required=True)
    preparing.add_argument("--out", type=Path, required=True)
    running = commands.add_parser("run", help="run the agent's call")
    running.add_argument("--call", type=Path, required=True)
    running.add_argument("--work", type=Path, required=True)
    running.add_argument("--profile", required=True, help="a packaged execution profile, or a YAML file over one")
    running.add_argument("--executor", required=True, choices=exitlib.EXECUTORS)
    running.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        exitlib.write_json(args.out, agent_input(args.dataset))
        print(f"wrote {args.out}")
        return 0
    result = run_call(args.call, args.profile, args.executor, args.work, exitlib.load_config())
    exitlib.write_json(args.out, result)
    print(f"{'ok' if result['ok'] else 'FAILED'}: {result.get('error') or result.get('counts')}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
