r"""Re-run the in-family branch over a finished corpus run's stores and record what it read.

``specs/20260921-gates-in-verdict/corpus-replay.py``, with one change: the readings are recorded as
a name-to-value mapping rather than a list of names, so the A/B covers every value VERDICT could
read and not only the conformance it reaches. Both sides of this change are post-gates, so
``GATED`` is True on both and the conformance is reached the same way on each.

Read-only over the run tree: it loads each recording's ``store.jsonl`` and its PREPROCESS
derivatives, re-runs the one branch that owns the declared family, and writes what that branch's
task conformance would be. No model is loaded, no audio is decoded, and nothing under the run tree
is written.

The same file runs on both sides of the change. On the pre-change code a branch answers the
conformance itself (``Result.done``); on the post-change code VERDICT's gates answer it from the
readings the branch reported. Which side is running is read off ``Result._fields``, so one script
produces both halves of the A/B and any difference is attributable to the change rather than to a
difference between two scripts.

Usage, as a Slurm array task::

    REPLAY_ROWS=<run>/rows REPLAY_OUT=<dir> REPLAY_HINTS=<dir holding hints.py> \\
      python corpus-replay.py

``SLURM_ARRAY_TASK_ID`` and ``SLURM_ARRAY_TASK_COUNT`` select this task's slice.
"""

from __future__ import annotations

import importlib.util
import json
import os
import traceback
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.extend import read_store
from senselab.audio.workflows.triage.nodes.airway import align_airway
from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS, Result, branch_params
from senselab.audio.workflows.triage.nodes.ddk import read_ddk
from senselab.audio.workflows.triage.nodes.speech import align_speech
from senselab.audio.workflows.triage.nodes.voice import align_voice
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK, task_family, task_id_of

GATED = "done" not in Result._fields
"""Whether this checkout is the post-change one, in which VERDICT's gates answer the conformance."""

if GATED:
    from senselab.audio.workflows.triage.nodes.gates import (
        apply_gates,
        conformance_gate_names,
        load_gate_bounds,
    )

CONFIG = load_triage_config(None)
"""The packaged configuration of whichever checkout is running."""


def _hints_module(directory: str) -> Any:  # noqa: ANN401 — a module
    """The corpus run's own hint populator, loaded from beside the driver that used it.

    Args:
        directory: The directory holding ``hints.py``.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location("corpus_hints", Path(directory) / "hints.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owner_of(family: str) -> str | None:
    """The branch whose expectation table holds this family.

    Args:
        family: The declared task family.

    Returns:
        The branch name, or None when no branch evaluates the family in family.
    """
    for branch, table in EXPECTATIONS.items():
        if family in table:
            return branch
    return None


def evaluate(branch: str, family: str, store: Any, hint: Any, run_dir: Path) -> Result:  # noqa: ANN401
    """Run the in-family mode of the branch that owns this family.

    Args:
        branch: The owning branch.
        family: The declared task family.
        store: The provenance store the run left behind.
        hint: What the recording was declared to contain.
        run_dir: The run's own directory, which the derivatives are relative to.

    Returns:
        What the mode returned.

    Raises:
        KeyError: If the branch is not one of the three.
    """
    params = branch_params(CONFIG)
    if branch == "VOICE":
        return align_voice(family, store, hint, params, run_dir=run_dir)
    if branch == "AIRWAY":
        return align_airway(family, store, hint, params, run_dir=run_dir)
    if branch == "SPEECH":
        return align_speech(family, store, hint, params, reads=read_ddk(store, run_dir, "plain"))
    raise KeyError(branch)


def conformance_of(result: Result, branch: str, family: str) -> Any:  # noqa: ANN401 — a Conformance
    """What the task's conformance is, on whichever side of the change is running.

    Args:
        result: What the mode returned.
        branch: The owning branch.
        family: The declared task family.

    Returns:
        True, False, or ``"UNDETERMINED"``.
    """
    if not GATED:
        # The pre-change `Result` carries the conformance itself; the post-change one has no field for it.
        return getattr(result, "done")  # noqa: B009
    row = EXPECTATIONS[branch][family]
    readings = {finding.name: finding.evidence["value"] for finding in result.deviations if finding.kind == "measure"}
    names = conformance_gate_names(row.pattern, anti_pattern=row.anti_pattern)
    return apply_gates(names, load_gate_bounds(CONFIG, row.pattern, family), readings)[0]


def replay(row_path: Path, build_hint: Any) -> dict[str, Any]:  # noqa: ANN401 — the populator
    """One recording's before-and-after row.

    Args:
        row_path: The corpus run's own row for the recording.
        build_hint: The run's hint populator.

    Returns:
        What to write for this recording. ``conformance`` is None when nothing could be replayed.
    """
    row = json.loads(row_path.read_text())
    stem = str(row["stem"])
    # The row's own `family` is the stem's task id, which carries the trailing index; the
    # expectation tables are keyed by the family that id resolves to.
    resolved = task_family(task_id_of(stem))
    family = "" if resolved == UNKNOWN_TASK else resolved
    out: dict[str, Any] = {
        "stem": stem,
        "family": family,
        "branch": owner_of(family),
        "recorded": (row.get("decision") or {}).get("conformance", {}),
        "conformance": None,
        "error": None,
    }
    branch = out["branch"]
    if branch is None or not row.get("ok"):
        return out
    run_root = Path(str(row["run_dir"])).parent
    try:
        store = read_store(run_root)
        hint, _ = build_hint(Path(str(row["source"] if "source" in row else _source_of(run_root))))
        result = evaluate(branch, family, store, hint, run_root / "run")
        out["conformance"] = conformance_of(result, branch, family)
        out["readings"] = {
            finding.name: finding.evidence["value"] for finding in result.deviations if finding.kind == "measure"
        }
    except Exception:  # noqa: BLE001 — a replay that cannot run is data, not a crash
        out["error"] = traceback.format_exc(limit=3)
    return out


def _source_of(run_root: Path) -> str:
    """The recording the run was over, from its own ``run.json``.

    Args:
        run_root: The run root.

    Returns:
        The source path.
    """
    return str(json.loads((run_root / "run" / "run.json").read_text())["source"])


def main() -> None:
    """Replay this array task's slice of the corpus."""
    rows_dir = Path(os.environ["REPLAY_ROWS"])
    out_dir = Path(os.environ["REPLAY_OUT"])
    build_hint = _hints_module(os.environ["REPLAY_HINTS"]).build_hint
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    rows = sorted(rows_dir.glob("*/*/*.row.json"))
    mine = rows[index::count]
    out_dir.mkdir(parents=True, exist_ok=True)
    side = "new" if GATED else "old"
    target = out_dir / f"{side}-{index:04d}.jsonl"
    with target.open("w") as handle:
        for position, row_path in enumerate(mine):
            handle.write(json.dumps(replay(row_path, build_hint), default=str) + "\n")
            if position % 200 == 0:
                handle.flush()
                print(f"{position}/{len(mine)}", flush=True)
    print(f"wrote {target} ({len(mine)} rows)", flush=True)


if __name__ == "__main__":
    main()
