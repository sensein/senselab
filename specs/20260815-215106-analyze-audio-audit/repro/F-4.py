"""Reproduce F-4: adaptive/plot.py's _fused_axis docstring claims "the belief store
ingests L1's per-pass axis folds". belief.py's own VoteStore.from_run_dir docstring
says that path ("L1/<pass>/uncertainty/<axis>.parquet") was removed, and the store now
ingests "L2/round/0/derivatives/votes/<axis>.parquet" instead -- the same quantity the
in-process path sees, not a per-pass axis fold.

No model load, no network: pure docstring inspection.

Run from the repository root:
    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-4.py
"""

import inspect
import re
import sys

from senselab.audio.workflows.audio_analysis.adaptive.belief import VoteStore
from senselab.audio.workflows.audio_analysis.adaptive.plot import _fused_axis

plot_doc = inspect.getdoc(_fused_axis) or ""
belief_doc = inspect.getdoc(VoteStore.from_run_dir) or ""

claimed = "the belief store ingests L1's per-pass axis folds"
assert claimed in plot_doc, f"_fused_axis docstring wording changed; expected {claimed!r}"

normalized_belief = " ".join(belief_doc.split())
removed_path = "L1/<pass>/uncertainty/<axis>.parquet"
real_path = "L2/round/0/derivatives/votes/<axis>.parquet"
assert removed_path in normalized_belief, "VoteStore.from_run_dir docstring wording changed"
assert real_path in normalized_belief, "VoteStore.from_run_dir docstring wording changed"
assert "gone" in normalized_belief.lower(), "VoteStore.from_run_dir docstring wording changed"

print(f"claimed (_fused_axis docstring): {claimed!r}")
print(f"actual  (VoteStore.from_run_dir docstring): {removed_path!r} is removed ('Both are gone'); "
      f"real ingest path is {real_path!r}")

if removed_path in normalized_belief and "gone" in normalized_belief.lower():
    print("DEFECT REPRODUCED: _fused_axis's docstring claims the belief store ingests "
          f"{removed_path!r}, but from_run_dir's own docstring says that path was removed "
          f"and it now ingests {real_path!r}.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: belief.py does not disown the path plot.py claims.")
    sys.exit(1)
