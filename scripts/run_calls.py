"""Run a JSON array of calls on `InProcess`, each validated as a user call is.

uv run python scripts/run_calls.py calls.json --execution laptop

Prints `task: ok <output types>` or `task: FAILED <error type>: <message>` per call; exits 1 if any call fails.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from senselab_core.runtime.call import TaskCall, packaged
from senselab_core.runtime.execution.in_process import InProcess
from senselab_core.runtime.profiles import ExecutionProfile


def run_calls(calls: Sequence[Any], execution: ExecutionProfile) -> list[str]:
    """Return one result line per call: `task: ok <output types>` or `task: FAILED <error type>: <message>`."""
    lines = []
    executor = InProcess(profile=execution)
    for call in calls:
        task = call.get("task", "?") if isinstance(call, dict) else "?"
        try:
            result = executor.call(TaskCall.model_validate(call))
        except Exception as error:  # noqa: BLE001 - the harness reports every failure of a call
            lines.append(f"{task}: FAILED {type(error).__name__}: {' '.join(str(error).split())}")
            continue
        spec = packaged().registry.spec(task)
        lines.append(f"{task}: ok {' '.join(f'{p}:{t}' for p, t in spec.outputs.items())}")
        assert set(result.outputs) == set(spec.outputs)
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Read the calls file, run the calls and print their lines; return 1 if any failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "calls", type=Path, help="a JSON array of calls `{task, inputs, params, model, backend_params}`"
    )
    parser.add_argument("--execution", default="laptop", help="a packaged profile name, or a path to a whole profile file, to run under")
    args = parser.parse_args(argv)
    calls = json.loads(args.calls.read_text())
    if not isinstance(calls, list):
        parser.error("the calls file must hold a JSON array")
    lines = run_calls(calls, ExecutionProfile.load(args.execution))
    print("\n".join(lines))
    return 1 if any(": FAILED " in line for line in lines) else 0


if __name__ == "__main__":
    sys.exit(main())
