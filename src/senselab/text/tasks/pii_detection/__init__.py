""".. include:: ./doc.md"""  # noqa: D415

from senselab.text.tasks.pii_detection.api import (
    PiiReport,
    PiiScan,
    PiiSpan,
    decide_pii,
    default_detectors,
    detect_pii,
    flatten_script_line,
    scan_for_pii,
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
    "PiiScan",
    "PiiSpan",
    "decide_pii",
    "default_detectors",
    "detect_pii",
    "flatten_script_line",
    "scan_for_pii",
]
