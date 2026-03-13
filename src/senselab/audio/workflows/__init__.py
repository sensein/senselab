"""Workflows and pipelines for audio processing and analysis."""

from typing import Any


def analyze_conversation_recordings(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy export for the conversation analysis workflow."""
    from .conversation_analysis import analyze_conversation_recordings as _analyze_conversation_recordings

    return _analyze_conversation_recordings(*args, **kwargs)


def explore_conversation(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy export for the original conversation exploration workflow."""
    from .explore_conversation import explore_conversation as _explore_conversation

    return _explore_conversation(*args, **kwargs)
