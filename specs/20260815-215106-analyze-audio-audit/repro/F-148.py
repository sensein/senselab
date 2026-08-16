"""Reproduction for F-148 (raised-by B-10).

statistics.py's `confidence`, `variability`, `entropy_uncertainty`, `epistemic_uncertainty` are
pure statistics over generic `Sequence[float]`/`Mapping[str, float]` inputs (weighted-vote
probability, population std-dev, normalized Shannon entropy, entropy mutual-information
decomposition) with zero `audio_analysis` coupling — a promotion candidate for
`senselab/utils/tasks/` (matching the codebase's own `project_mc_dropout_optional` want).

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-148.py
(from the repository root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ast_sweep import workflow_imports  # noqa: E402

from senselab.audio.workflows.audio_analysis.statistics import (  # noqa: E402
    confidence,
    entropy_uncertainty,
    epistemic_uncertainty,
    variability,
)

repo_root = Path(__file__).resolve().parents[3]
statistics_py = repo_root / "src/senselab/audio/workflows/audio_analysis/statistics.py"
imports = workflow_imports(statistics_py)
print(f"statistics.py's imports from audio_analysis: {imports!r}")

c = confidence([True, True, False], weights=[0.5, 0.3, 0.2])
v = variability([1.0, 2.0, 3.0])
e = entropy_uncertainty({"a": 0.5, "b": 0.5})
total, epi = epistemic_uncertainty([{"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}])

print(f"confidence(votes, weights)={c}")
print(f"variability([1,2,3])={v}")
print(f"entropy_uncertainty(even split)={e}")
print(f"epistemic_uncertainty(two disagreeing signals)=({total}, {epi})")

no_coupling = imports == []
ran_on_plain_types = all(x is not None for x in (c, v, e, total, epi))

if no_coupling and ran_on_plain_types:
    print(
        "DEFECT REPRODUCED (promotion-candidate): statistics.py imports only `math`/`typing` — "
        "nothing from audio_analysis — and every function above ran on plain "
        "list[bool]/list[float]/dict[str, float] inputs with no Region/VoteStore/policy coupling. "
        "This generic uncertainty-decomposition math belongs in senselab/utils/tasks/, not the "
        "audio_analysis workflow package."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
