"""Reproduction for F-160 (raised-by B-22).

adaptive/identity_repair.py's `_agglomerative_cosine`, `_l2`, `change_point_trajectory` are
deterministic average-linkage clustering, L2-normalization, and a fixed-smoothing adjacent-window
cosine trajectory — generic numerical routines with zero dependency on Region/VoteStore/policy
dicts. The whole module imports only `floors.MIN_EVIDENCE_WEIGHT` (a bare float constant), so the
promotion candidate covers the entire file, not just the three named functions.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-160.py
(from the repository root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ast_sweep import workflow_imports  # noqa: E402

import numpy as np  # noqa: E402

from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import (  # noqa: E402
    _agglomerative_cosine,
    _l2,
    change_point_trajectory,
)

repo_root = Path(__file__).resolve().parents[3]
identity_repair_py = repo_root / "src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py"
imports = workflow_imports(identity_repair_py)
print(f"identity_repair.py's imports from audio_analysis: {imports!r}")

# _l2: plain array normalization.
normed = _l2(np.array([[3.0, 4.0], [1.0, 0.0]]))
print(f"_l2([[3,4],[1,0]]) = {normed.tolist()}")

# _agglomerative_cosine: plain vectors, plain float threshold, no Region/VoteStore in sight.
vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
labels = _agglomerative_cosine(vectors, threshold=0.05)
print(f"_agglomerative_cosine(4 plain vectors) = {labels}")

# change_point_trajectory: dict[str, list[dict]] with generic "vector"/"start_s"/"end_s" keys.
window_embeddings = {
    "model_a": [
        {"vector": [1.0, 0.0], "start_s": 0.0, "end_s": 1.0},
        {"vector": [1.0, 0.0], "start_s": 1.0, "end_s": 2.0},
        {"vector": [0.0, 1.0], "start_s": 2.0, "end_s": 3.0},
    ]
}
times, dist = change_point_trajectory(window_embeddings)
print(f"change_point_trajectory(...) = times={times}, dist={dist}")

no_coupling = imports == ["senselab.audio.workflows.audio_analysis.floors.MIN_EVIDENCE_WEIGHT"]
correct_l2 = abs(float(np.linalg.norm(normed[0])) - 1.0) < 1e-9
ran_clustering = len(labels) == 4 and labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
ran_trajectory = len(times) == 2 and len(dist) == 2

if no_coupling and correct_l2 and ran_clustering and ran_trajectory:
    print(
        "DEFECT REPRODUCED (promotion-candidate): identity_repair.py's only import from "
        f"audio_analysis is {imports!r} (a bare constant, not a workflow type), and all three "
        "functions ran correctly on plain numpy arrays / list[dict] with generic 'vector'/"
        "'start_s'/'end_s' keys. This clustering/trajectory math belongs in "
        "senselab/utils/tasks/ or senselab/audio/tasks/speaker_embeddings/, blocked only by "
        "leading-underscore naming."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
