"""Uncertainty-driven adaptive analysis loop (prototype).

Spec: ``specs/20260723-225523-dynamic-uncertainty-workflow/``.

This subpackage keeps imports light on purpose: no torch / model backends are
imported at module level, so the loop's pure core (belief store, region
proposal, policy engine, fusion, evaluation) runs in minimal environments.
Interventions that need live model backends import them lazily inside their
``execute`` functions and degrade to ``blocked_guard`` when unavailable.

Public API::

    from senselab.audio.workflows.audio_analysis.adaptive import run_adaptive_loop
"""

from typing import Any

__all__ = ["run_adaptive_loop"]


def __getattr__(name: str) -> Any:  # noqa: ANN401 — lazy re-export
    """Lazily resolve public symbols so importing the package stays light."""
    if name == "run_adaptive_loop":
        from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop

        return run_adaptive_loop
    raise AttributeError(name)
