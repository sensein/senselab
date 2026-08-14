"""Standalone PII detection over text, ``ScriptLine``, and transcripts."""

from senselab.text.tasks.pii_detection.api import (
    PiiReport,
    PiiSpan,
    default_detectors,
    detect_pii,
    flatten_script_line,
)
from senselab.text.tasks.pii_detection.local_llm import LocalLlmConfig
from senselab.text.tasks.pii_detection.subprocess_backend import (
    DETECTOR_GLINER,
    DETECTOR_LLM,
    DETECTOR_PRESIDIO,
    DETECTOR_RULES,
)

__all__ = [
    "DETECTOR_GLINER",
    "DETECTOR_LLM",
    "DETECTOR_PRESIDIO",
    "DETECTOR_RULES",
    "LocalLlmConfig",
    "PiiReport",
    "PiiSpan",
    "default_detectors",
    "detect_pii",
    "flatten_script_line",
]
