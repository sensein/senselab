"""Reproduction for F-153 (raised-by B-15).

occupancy.py's `_union_length` is generic interval algebra (clip intervals to a window, sum union
length over `list[tuple[float, float]]`) with zero workflow-type coupling; `occupancy`'s *only*
workflow coupling is its `Spans`/`Span` dataclass-typed signature — the actual algorithm
underneath is reusable as-is. Promotion candidate for `senselab/utils/tasks/`.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-153.py
(from the repository root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ast_sweep import function_param_types  # noqa: E402

from senselab.audio.workflows.audio_analysis.occupancy import _union_length, occupancy  # noqa: E402
from senselab.audio.workflows.audio_analysis.shapes import Span, Spans  # noqa: E402

repo_root = Path(__file__).resolve().parents[3]
occupancy_py = repo_root / "src/senselab/audio/workflows/audio_analysis/occupancy.py"

union_length_params = function_param_types(occupancy_py, "_union_length")
occupancy_params = function_param_types(occupancy_py, "occupancy")
print(f"_union_length parameter annotations: {union_length_params!r}")
print(f"occupancy parameter annotations: {occupancy_params!r}")

# _union_length runs on plain tuples with no import of Spans/Span needed at all.
result = _union_length([(0.0, 1.0), (0.5, 1.5), (2.0, 3.0)])
print(f"_union_length([(0,1),(0.5,1.5),(2,3)]) = {result}")

spans = Spans(spans=(Span(start=0.0, end=1.0, label="A"), Span(start=0.5, end=1.5, label="A")))
occ = occupancy(spans, start=0.0, end=2.0)
print(f"occupancy(...) [uses the same algorithm, but requires Spans/Span] = {occ}")

union_is_generic = not any("Span" in p for p in union_length_params)
occupancy_is_coupled = any("Span" in p for p in occupancy_params)
correct_value = abs(result - 2.5) < 1e-9  # union(0,1)+(0.5,1.5) -> (0,1.5)=1.5, plus (2,3)=1.0 -> 2.5

if union_is_generic and occupancy_is_coupled and correct_value:
    print(
        "DEFECT REPRODUCED (promotion-candidate): _union_length's own parameter types are plain "
        f"{union_length_params!r} (no Span/Spans) and it computes the correct union length "
        f"({result}) on bare tuples; the *only* thing tying this interval algebra to the "
        f"workflow is occupancy()'s Spans/Span-typed wrapper ({occupancy_params!r}). The "
        "generic core belongs in senselab/utils/tasks/, with only the thin adapter staying here."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
