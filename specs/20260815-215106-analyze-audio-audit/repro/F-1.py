"""Reproduce F-1: __init__.py's module docstring claims three uncertainty axes
(speech_presence, speaker, asr) and a 5-row timeline; axes.py's own AXES tuple has
four members (adds background_mask). No model load, no network.

Run from the repository root:
    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-1.py
"""

import re
import sys

from senselab.audio.workflows.audio_analysis import __doc__ as pkg_doc
from senselab.audio.workflows.audio_analysis.axes import AXES

claimed_axes = ("speech_presence", "speaker", "asr")
claimed_count = len(claimed_axes)
actual_count = len(AXES)
actual_names = tuple(a.name for a in AXES)

# Sanity-check the docstring actually says what the finding claims, so this script
# fails loudly if the prose is ever fixed instead of silently passing.
assert pkg_doc is not None
normalized = re.sub(r"\s+", " ", pkg_doc)
assert "emits three" in normalized and "uncertainty time series" in normalized, "docstring wording changed"
assert "5-row timeline" in normalized, "docstring wording changed"
for name in claimed_axes:
    assert name in pkg_doc, f"{name!r} missing from docstring"

print(f"claimed (docstring): {claimed_count} axes -> {claimed_axes}")
print(f"actual  (axes.AXES): {actual_count} axes -> {actual_names}")

if actual_count != claimed_count and "background_mask" in actual_names:
    print("DEFECT REPRODUCED: docstring says 3 axes, axes.AXES has "
          f"{actual_count} ({actual_names}); background_mask is undocumented at the top level.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: axes.AXES matches the docstring's claim.")
    sys.exit(1)
