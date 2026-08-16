"""Reproduce F-2: io.py's write_linked_votes docstring claims the file is written to
"L2/round0/votes/<axis>.parquet". The real destination (scripts/analyze_audio.py's call
site, using derivatives_dir) is "L2/round/<n>/derivatives/votes/<axis>.parquet" — missing
the "derivatives/" segment and misspelling "round/0" as "round0".

No model load, no network: derivatives_dir is a pure path-building function.

Run from the repository root:
    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-2.py
"""

import inspect
import sys
from pathlib import Path

from senselab.audio.workflows.audio_analysis.io import write_linked_votes
from senselab.audio.workflows.audio_analysis.layout import derivatives_dir

doc = inspect.getdoc(write_linked_votes) or ""
claimed_path = "L2/round0/votes/<axis>.parquet"
assert claimed_path in doc, f"docstring wording changed, expected {claimed_path!r} in it"

# Build the real path exactly as scripts/analyze_audio.py does at its write_linked_votes
# call site: derivatives_dir(run_dir, 0) / "votes" / f"{axis_name}.parquet".
run_dir = Path("/tmp/fake_run")
axis_name = "speaker"
real_path = derivatives_dir(run_dir, 0) / "votes" / f"{axis_name}.parquet"
real_relative = real_path.relative_to(run_dir).as_posix()

print(f"claimed (docstring): {claimed_path}")
print(f"actual  (derivatives_dir(run_dir, 0) / 'votes' / f'{{axis}}.parquet'): {real_relative}")

claimed_relative = claimed_path.replace("<axis>", axis_name)
if real_relative != claimed_relative:
    print(f"DEFECT REPRODUCED: docstring path {claimed_relative!r} != actual path {real_relative!r}")
    sys.exit(0)
else:
    print("NOT REPRODUCED: paths match.")
    sys.exit(1)
