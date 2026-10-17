""".. include:: ./doc.md"""  # noqa: D415

from .accuracy_estimate import (  # noqa: F401
    AccuracyEstimate,
    TranscriptAccuracyEstimate,
    estimate_transcript_accuracy,
)
from .utils import calculate_cer, calculate_mer, calculate_wer, calculate_wil, calculate_wip  # noqa: F401

__all__ = [
    "AccuracyEstimate",
    "TranscriptAccuracyEstimate",
    "calculate_cer",
    "calculate_mer",
    "calculate_wer",
    "calculate_wil",
    "calculate_wip",
    "estimate_transcript_accuracy",
]
