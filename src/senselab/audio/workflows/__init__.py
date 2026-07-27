"""Workflows and pipelines for audio processing and analysis.

``explore_conversation`` is resolved lazily (PEP 562): importing this package —
or any pure submodule under ``audio_analysis`` — must not pull the four model
task stacks that ``explore_conversation`` depends on (architecture-review.md
F1 / T046).
"""

from typing import Any

__all__ = ["explore_conversation"]


def __getattr__(name: str) -> Any:  # noqa: ANN401 — lazy re-export
    """Resolve ``explore_conversation`` on first access."""
    if name == "explore_conversation":
        from senselab.audio.workflows.explore_conversation import explore_conversation

        return explore_conversation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy exports to introspection / pdoc."""
    return __all__
