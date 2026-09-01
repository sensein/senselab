"""Span proposal from an envelope."""

from senselab.audio.tasks.spans.api import NoContrast, Span, group_extents_into_runs, propose_spans

__all__ = [
    "NoContrast",
    "Span",
    "group_extents_into_runs",
    "propose_spans",
]
