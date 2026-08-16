"""Reproduce F-6: adaptive/loop.py's module docstring says "round 1 is the ingested
analyze_audio run" (fixed numbering, artifact-only). _baseline_round's own docstring,
730 lines later in the same file, says this replaced a scheme where "the adaptive loop
used to call its ingest 'round 1' while the fusion loop called the same iteration
'round 0'" -- i.e. the baseline round is *adopted* (last_round(out_dir) or 0), not fixed
at 1. Also demonstrate the second staleness point: run_adaptive_loop accepts an
in-memory ingest path (harvests=/unharvested_votes=/summary=) the "Artifact-driven"
docstring never mentions.

No model load, no network: pure source/signature inspection, plus a synthetic call to
_baseline_round against an empty directory (no rounds written) to show it returns 0,
not 1.

Run from the repository root:
    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-6.py
"""

import inspect
import sys
import tempfile
from pathlib import Path

import senselab.audio.workflows.audio_analysis.adaptive.loop as loop

module_doc = inspect.getdoc(loop) or ""
claimed = "round 1 is the\ningested analyze_audio run".replace("\n", " ")
normalized = " ".join(module_doc.split())
assert "round 1 is the ingested analyze_audio run" in normalized, "loop.py module docstring wording changed"
assert "Artifact-driven" in normalized, "loop.py module docstring wording changed"

baseline_doc = inspect.getdoc(loop._baseline_round) or ""
assert "round 1" in baseline_doc and "round 0" in baseline_doc, "_baseline_round docstring wording changed"

# Show the actual behavior: on a directory with no fusion rounds written, the loop adopts
# round 0 as baseline, not round 1 as the module docstring's "round 1 is the ingested run"
# framing implies.
with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp)
    actual_baseline = loop._baseline_round(out_dir)

sig = inspect.signature(loop.run_adaptive_loop)
in_memory_params = {"harvests", "unharvested_votes", "summary"} & set(sig.parameters)

print(f"claimed (module docstring): 'round 1 is the ingested analyze_audio run' "
      f"(framed as fixed, artifact-only numbering)")
print(f"actual  (_baseline_round on an empty out_dir): baseline round = {actual_baseline}")
print(f"actual  (run_adaptive_loop signature): in-memory ingest params present = "
      f"{sorted(in_memory_params)}")

if actual_baseline != 1 and in_memory_params == {"harvests", "unharvested_votes", "summary"}:
    print("DEFECT REPRODUCED: the module docstring's 'round 1' / artifact-only framing does not "
          f"match _baseline_round's actual result ({actual_baseline}, not 1) or "
          f"run_adaptive_loop's fully supported in-memory ingest path ({sorted(in_memory_params)}).")
    sys.exit(0)
else:
    print("NOT REPRODUCED: behavior matches the docstring's claim.")
    sys.exit(1)
