"""Grouping a BIDS-like stem's ``task-`` id into the family it belongs to.

These read a file name, so only a tool working over file names uses them (the identity check comparing what a
store recorded against its stem). Which tasks are of which kind is the dataset's task list's
(:data:`~senselab.audio.workflows.triage.task_catalogue.TASKS_BY_KIND`), not this module's.

Only trailing numeric segments are collapsed, which folds a repeat index and the Harvard list
index and nothing else. A ``v2`` marker is not numeric and therefore survives, so
``respiration-and-cough`` v1 and v2 stay different families.
"""

from __future__ import annotations

import re

_TASK = re.compile(r"_task-(?P<task>[^_]+)$")
_TRAILING_INDEX = re.compile(r"(?:-\d+)+$")

UNKNOWN_TASK = "unknown"
"""The task id of a stem carrying no ``task-`` entity."""


def task_id_of(stem: str) -> str:
    """The sanitized task id in a BIDS stem.

    Args:
        stem: A stem such as ``sub-a_ses-b_task-harvard-sentences-list-10-3``.

    Returns:
        The lowercased task id, or :data:`UNKNOWN_TASK` when the stem carries no ``task-`` entity.
    """
    match = _TASK.search(stem)
    return match.group("task").lower() if match else UNKNOWN_TASK


def task_family(task_id: str) -> str:
    """The family a task id collapses into.

    Args:
        task_id: A sanitized task id.

    Returns:
        The id with every trailing numeric segment removed.
    """
    return _TRAILING_INDEX.sub("", task_id)
