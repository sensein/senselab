"""Standalone PII detection over text, ``ScriptLine``, and transcripts."""

from senselab.text.tasks.pii_detection.api import (
    PiiReport,
    PiiSpan,
    detect_pii,
    flatten_script_line,
    report_to_dict,
)
from senselab.text.tasks.pii_detection.subprocess_backend import (
    DETECTOR_GLINER,
    DETECTOR_PRESIDIO,
)

__all__ = [
    "DETECTOR_GLINER",
    "DETECTOR_PRESIDIO",
    "PiiReport",
    "PiiSpan",
    "detect_pii",
    "flatten_script_line",
    "report_to_dict",
]
