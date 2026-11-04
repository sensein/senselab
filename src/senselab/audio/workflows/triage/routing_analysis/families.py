"""Grouping a task id into the family it belongs to.

Nothing here reads a file name (plan step 6a): a task id is what a recording's hint declared. Which tasks are of
which kind is the dataset's task list's (:data:`~senselab.audio.workflows.triage.task_catalogue.TASKS_BY_KIND`),
not this module's.

Only trailing numeric segments are collapsed, which folds a repeat index and the Harvard list
index and nothing else. A ``v2`` marker is not numeric and therefore survives, so
``respiration-and-cough`` v1 and v2 stay different families.
"""

from __future__ import annotations

import re

_TRAILING_INDEX = re.compile(r"(?:-\d+)+$")

UNKNOWN_TASK = "unknown"
"""The task id of a recording that declares none."""


def task_family(task_id: str) -> str:
    """The family a task id collapses into.

    Args:
        task_id: A sanitized task id.

    Returns:
        The id with every trailing numeric segment removed.
    """
    return _TRAILING_INDEX.sub("", task_id)
