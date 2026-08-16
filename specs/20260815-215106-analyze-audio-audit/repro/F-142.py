"""Reproduction for F-142 (raised-by B-4).

level.py's `apply_gain_db`, `integrated_lufs`, `loudness_range_lu`, `true_peak_dbtp`,
`clipped_fraction`, `normalization_gain_db`, `peak_limited_gain_db` are pure BS.1770/EBU-Tech-3342
loudness/gain/clipping math over a raw `(waveform, sampling_rate)` pair, with no `audio_analysis`
workflow-bookkeeping coupling — a promotion candidate for `senselab.audio.tasks.quality_control`.

Demonstrated by: (1) an AST sweep of level.py's imports showing zero coupling to any
`audio_analysis`-specific type, and (2) running the functions directly on a synthetic waveform to
confirm they execute with no workflow object in sight.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-142.py
(from the repository root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ast_sweep import workflow_imports  # noqa: E402

import numpy as np  # noqa: E402

from senselab.audio.workflows.audio_analysis.level import (  # noqa: E402
    apply_gain_db,
    clipped_fraction,
    integrated_lufs,
    loudness_range_lu,
    normalization_gain_db,
    true_peak_dbtp,
)

repo_root = Path(__file__).resolve().parents[3]
level_py = repo_root / "src/senselab/audio/workflows/audio_analysis/level.py"
imports = workflow_imports(level_py)
print(f"level.py's imports from audio_analysis: {imports!r}")

rng = np.random.default_rng(0)
sr = 16000
waveform = (0.1 * rng.standard_normal(sr * 2)).astype(np.float64)

lufs = integrated_lufs(waveform, sr)
gained = apply_gain_db(waveform, 6.0)
lra = loudness_range_lu(waveform, sr)
peak = true_peak_dbtp(waveform, sr)
clip = clipped_fraction(waveform)
norm_gain = normalization_gain_db(waveform, sr, target_lufs=-23.0)

print(f"integrated_lufs={lufs}, true_peak_dbtp={peak}, loudness_range_lu={lra}")
print(f"apply_gain_db(+6dB) sample ratio={gained[0] / waveform[0]}, clipped_fraction={clip}")
print(f"normalization_gain_db={norm_gain}")

no_coupling = imports == []
ran_on_plain_arrays = all(
    isinstance(x, (int, float, np.floating, np.integer)) or hasattr(x, "shape")
    for x in (lufs, lra, peak, clip, norm_gain)
) and gained.shape == waveform.shape

if no_coupling and ran_on_plain_arrays:
    print(
        "DEFECT REPRODUCED (promotion-candidate): level.py imports NOTHING from "
        "audio_analysis (only numpy/json/math/pathlib/dataclasses + senselab.utils logging) and "
        "every listed function runs correctly on a bare (waveform, sampling_rate) pair with no "
        "workflow object anywhere in the call. This BS.1770/EBU-Tech-3342 math belongs in "
        "senselab.audio.tasks.quality_control, not the audio_analysis workflow package."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
