"""Which stages a triage run drives, and which of them a dataset's workflow leaves off (plan step 11).

The runner drives :data:`STAGES` in order. A dataset names the stages it runs in the ``workflow.stages``
setting; a stage left off is recorded ``skipped`` with :data:`OFF_IN_WORKFLOW` as its note, the way a branch
routing declined is, and a stage that needs one that is off is skipped with :func:`off_note`'s
"needs" note. The stages that every later stage or the verdict reads may not be left off.

This module imports nothing from the nodes, so the configuration and the profile check can read it; a test
holds :data:`STAGES` to :data:`~senselab.audio.workflows.triage.vocabulary.GRAPH_ORDER`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

WORKFLOW_SETTING = "workflow.stages"
"""The configuration key listing the stages a run drives."""

OFF_IN_WORKFLOW = "off in this workflow"
"""The note on a stage the workflow leaves off."""


@dataclass(frozen=True)
class Stage:
    """One stage of the triage graph.

    Attributes:
        name: The node name the run records.
        needs: The stages it reads, which must run for it to run.
        may_be_off: Whether a workflow may leave it off; False for stages every later stage or the verdict reads.
    """

    name: str
    needs: tuple[str, ...] = ()
    may_be_off: bool = False


STAGES: tuple[Stage, ...] = (
    Stage("ADMIT"),
    Stage("PREPROCESS"),
    Stage("SESSION"),
    Stage("BACKGROUND"),
    Stage("TAXONOMY"),
    Stage("routing"),
    Stage("AIRWAY", may_be_off=True),
    Stage("SPEECH", may_be_off=True),
    Stage("VOICE", may_be_off=True),
    Stage("QUALITY", may_be_off=True),
    Stage("COHORT", may_be_off=True),
    Stage("REDACT", needs=("SPEECH",), may_be_off=True),
    Stage("REVIEW", needs=("SPEECH",), may_be_off=True),
    Stage("SECOND_OPINION", needs=("REVIEW",), may_be_off=True),
    Stage("VERDICT"),
)
"""Every stage, in the order the runner drives it."""

STAGE_NAMES = tuple(stage.name for stage in STAGES)
OPTIONAL_STAGES = frozenset(stage.name for stage in STAGES if stage.may_be_off)
"""The stages a workflow may leave off, and so the only ones a profile value may name."""


def check_stages(listed: Iterable[str]) -> tuple[str, ...]:
    """A ``workflow.stages`` list, checked.

    Args:
        listed: The stage names the workflow runs.

    Returns:
        The names, in graph order.

    Raises:
        ValueError: If a name is not a stage, or a stage that may not be off is missing.
    """
    names = [str(name) for name in listed]
    unknown = sorted(set(names) - set(STAGE_NAMES))
    if unknown:
        raise ValueError(
            f"{WORKFLOW_SETTING} names {unknown}, which are not triage stages; the stages are {STAGE_NAMES}"
        )
    required = [stage.name for stage in STAGES if not stage.may_be_off and stage.name not in names]
    if required:
        raise ValueError(
            f"{WORKFLOW_SETTING} leaves off {required}, which every later stage or the verdict reads; "
            f"only {sorted(OPTIONAL_STAGES)} may be off"
        )
    return tuple(name for name in STAGE_NAMES if name in names)


def listed_stages(config_values: Mapping[str, Any]) -> tuple[str, ...]:
    """The stages a configuration's workflow lists; every stage when it sets none.

    Args:
        config_values: The merged configuration (``TriageConfig.values``).

    Returns:
        The checked names, in graph order.
    """
    listed = (config_values.get("workflow") or {}).get("stages")
    return STAGE_NAMES if listed is None else check_stages(listed)


def off_notes(config_values: Mapping[str, Any]) -> dict[str, str]:
    """Why each stage the workflow does not run is off.

    A stage not listed is :data:`OFF_IN_WORKFLOW`; a listed stage whose need is off is off too, with a note
    naming that need. A stage absent from the result runs.

    Args:
        config_values: The merged configuration.

    Returns:
        ``{stage: note}`` for every stage that is off.
    """
    listed = set(listed_stages(config_values))
    notes: dict[str, str] = {}
    for stage in STAGES:
        if stage.name not in listed:
            notes[stage.name] = OFF_IN_WORKFLOW
            continue
        off = [need for need in stage.needs if need in notes]
        if off:
            notes[stage.name] = f"{OFF_IN_WORKFLOW}: needs {', '.join(off)}"
    return notes


def running_stages(config_values: Mapping[str, Any]) -> frozenset[str]:
    """The stages a run of this configuration drives.

    Args:
        config_values: The merged configuration.

    Returns:
        Every stage :func:`off_notes` leaves on.
    """
    notes = off_notes(config_values)
    return frozenset(name for name in STAGE_NAMES if name not in notes)


def require_stage(config_values: Mapping[str, Any], stage: str) -> None:
    """Refuse a corpus pass for a stage the workflow leaves off.

    Args:
        config_values: The merged configuration.
        stage: The stage the pass writes.

    Raises:
        ValueError: If the stage is off, naming why.
    """
    note = off_notes(config_values).get(stage)
    if note is not None:
        raise ValueError(f"{stage} is {note} ({WORKFLOW_SETTING}), so this pass has nothing to write")
