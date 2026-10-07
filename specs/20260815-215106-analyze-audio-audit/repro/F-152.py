"""Reproduction for F-152 (raised-by B-14).

acoustic.py's `lufs_track`/`level_above_floor_track` are pure `(waveform, sampling_rate) ->
(times, values)` numpy computations (BS.1770-style short-term loudness; bias-corrected percentile
floor excess), only `math`/`numpy` imported — a promotion candidate for
`senselab/audio/tasks/features_extraction/` (new `loudness.py`).

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-152.py
(from the repository root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ast_sweep import workflow_imports  # noqa: E402

import numpy as np  # noqa: E402

from senselab.audio.workflows.audio_analysis.acoustic import (  # noqa: E402
    level_above_floor_track,
    lufs_track,
)

repo_root = Path(__file__).resolve().parents[3]
acoustic_py = repo_root / "src/senselab/audio/workflows/audio_analysis/acoustic.py"
imports = workflow_imports(acoustic_py)
print(f"acoustic.py's imports from audio_analysis: {imports!r}")

rng = np.random.default_rng(0)
sr = 16000
waveform = (0.1 * rng.standard_normal(sr * 2)).astype(np.float64)

times1, lufs_vals = lufs_track(waveform, sr, hop_s=0.1)
times2, floor_vals = level_above_floor_track(waveform, sr, hop_s=0.1)

print(f"lufs_track: {len(times1)} frames, first values={lufs_vals[:3]}")
print(f"level_above_floor_track: {len(times2)} frames, first values={floor_vals[:3]}")

no_coupling = imports == []
ran_on_plain_arrays = len(times1) > 0 and len(times2) > 0

if no_coupling and ran_on_plain_arrays:
    print(
        "DEFECT REPRODUCED (promotion-candidate): acoustic.py imports only `math`/`numpy` — "
        "nothing from audio_analysis — and both functions ran correctly on a bare "
        "(waveform, sampling_rate) pair. This loudness math belongs in "
        "senselab/audio/tasks/features_extraction/, not the audio_analysis workflow package."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
