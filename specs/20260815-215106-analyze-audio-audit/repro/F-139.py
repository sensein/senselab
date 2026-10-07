"""Reproduction for F-139 (raised-by B-1).

fuse.py:559 `derive_mask_from_axes(..., settled_below: float = 0.35)` is a bare, undecided
default (see `keys.py`'s own named paradigm case). A bucket whose presence+mask uncertainty sits
at 0.34 becomes a `target_free` mask region (discounting later signals there); a bucket at 0.36,
qualitatively indistinguishable, gets no region at all. No caller overrides it and it is not
present in `default.yaml`.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-139.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.fuse import Derivatives, derive_mask_from_axes

BUCKET = {"start": 0.0, "end": 1.0}


def rows_at(uncertainty: float) -> dict:
    """One presence axis row and one agreeing background_mask row at `uncertainty`."""
    presence_row = {**BUCKET, "uncertainty": uncertainty}
    mask_row = {**BUCKET, "uncertainty": uncertainty}
    return {"speech_presence": [presence_row], "background_mask": [mask_row]}


current = Derivatives()

just_below = derive_mask_from_axes(rows_at(0.34), current)  # below the bare 0.35 default
just_above = derive_mask_from_axes(rows_at(0.36), current)  # above it

below_regions = just_below.mask_regions if just_below is not None else ()
above_regions = just_above.mask_regions if just_above is not None else ()

print(f"uncertainty=0.34 -> mask_regions={below_regions!r}")
print(f"uncertainty=0.36 -> mask_regions={above_regions!r}")

# Confirm no caller in the package ever overrides settled_below.
import ast  # noqa: E402
from pathlib import Path  # noqa: E402

repo_root = Path(__file__).resolve().parents[3]
overridden = False
for path in (repo_root / "src/senselab/audio/workflows/audio_analysis").rglob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and any(getattr(kw, "arg", None) == "settled_below" for kw in node.keywords):
            overridden = True
print(f"settled_below= passed anywhere in the package: {overridden}")

flips = (
    len(below_regions) == 1
    and below_regions[0]["state"] == "target_free"
    and len(above_regions) == 0
    and not overridden
)

if flips:
    print(
        "DEFECT REPRODUCED: settled_below=0.35 (bare default, no derivation, no override path) "
        "decides the mask verdict. uncertainty=0.34 -> target_free region (wrong: should not be "
        "a sharp cliff); uncertainty=0.36 -> no region at all, for a qualitatively identical "
        f"bucket. Right behavior: this boundary should come from a fitted value in data/, "
        f"not a code literal. below_regions={below_regions!r} above_regions={above_regions!r}"
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
