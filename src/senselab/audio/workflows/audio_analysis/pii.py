"""Temporary shim — replaced by the real adapter in Task 6 of plan-b."""

from senselab.text.tasks.pii_detection.api import (  # noqa: F401
    PiiPassReport,
    PiiSpan,
    detect_pii_in_pass,
    report_to_dict,
)
