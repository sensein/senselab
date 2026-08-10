"""Standalone PII detection over transcripts."""

# NOTE: detect_pii / PiiReport are added in Task 2 of plan-b. Until then this
# __init__ exports only what api.py actually defines.
from senselab.text.tasks.pii_detection.api import (
    PiiSpan,
    report_to_dict,
)
from senselab.text.tasks.pii_detection.subprocess_backend import (
    DETECTOR_GLINER,
    DETECTOR_PRESIDIO,
)

__all__ = [
    "DETECTOR_GLINER",
    "DETECTOR_PRESIDIO",
    "PiiSpan",
    "report_to_dict",
]
