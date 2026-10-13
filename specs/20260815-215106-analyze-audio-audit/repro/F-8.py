"""Reproduce F-8: adaptive/interventions.py's module docstring describes
I4_overlap_detection as using "segmentation-3.0 per-class posteriors (gated model...)".
_i4_execute actually calls backends.overlap_track_from_spans, whose own docstring says
it derives overlap "from cross-diarizer spans rather than one model's channels" and
returns a 1.0/0.0 decision, not a posterior -- confirmed via an AST sweep of
_i4_execute's own call sites (zero calls to any segmentation-3.0-posterior API, one
call to overlap_track_from_spans).

No model load, no network: pure AST/source inspection.

Run from the repository root:
    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-8.py
"""

import ast
import inspect
import sys

import senselab.audio.workflows.audio_analysis.adaptive.backends as backends
import senselab.audio.workflows.audio_analysis.adaptive.interventions as interventions

module_doc = inspect.getdoc(interventions) or ""
normalized = " ".join(module_doc.split())
claimed = "segmentation-3.0 per-class posteriors"
assert claimed in normalized, "interventions.py module docstring wording changed"

i4_source = inspect.getsource(interventions._i4_execute)
tree = ast.parse(i4_source)
called_names = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            called_names.add(func.attr)
        elif isinstance(func, ast.Name):
            called_names.add(func.id)

backend_doc = inspect.getdoc(backends.overlap_track_from_spans) or ""
normalized_backend_doc = " ".join(backend_doc.split())

calls_span_backend = "overlap_track_from_spans" in called_names
uses_cross_diarizer = "cross-diarizer" in normalized_backend_doc
is_decision_not_posterior = "not a posterior" in normalized_backend_doc.replace("**", "").lower()

print(f"claimed (module docstring): I4_overlap_detection uses "
      f"'segmentation-3.0 per-class posteriors'")
print(f"actual  (_i4_execute call sites): {sorted(called_names)}")
print(f"actual  (overlap_track_from_spans docstring): uses cross-diarizer spans = "
      f"{uses_cross_diarizer}, decision-not-posterior = {is_decision_not_posterior}")

if calls_span_backend and uses_cross_diarizer and is_decision_not_posterior:
    print("DEFECT REPRODUCED: docstring claims segmentation-3.0 per-class posteriors, but "
          "_i4_execute calls overlap_track_from_spans, which derives overlap from "
          "cross-diarizer spans and returns a decision, not a posterior.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: _i4_execute uses segmentation-3.0 posteriors as described.")
    sys.exit(1)
