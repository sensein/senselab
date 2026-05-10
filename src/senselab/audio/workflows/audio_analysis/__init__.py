"""Three-axis uncertainty workflow for analyze_audio outputs.

Reads cached / in-memory results from senselab's per-task audio pipeline
(diarization, ASR, scene classification, alignment, PPG) and emits three
per-bucket uncertainty time series — `presence`, `identity`, and `utterance` —
plus a ranked `disagreements.json` index and a 5-row timeline plot.

See ``specs/20260508-173136-compare-uncertainty/spec.md`` for the full design.
The reusable workflow is consumed by ``scripts/analyze_audio.py`` as a thin
wrapper, but it is also importable standalone:

    from senselab.audio.workflows.audio_analysis import compute_uncertainty_axes
"""

from senselab.audio.workflows.audio_analysis.aggregators import (
    AGGREGATORS,
    apply_aggregator,
)
from senselab.audio.workflows.audio_analysis.compute import compute_uncertainty_axes
from senselab.audio.workflows.audio_analysis.disagreements import build_disagreements_index
from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    extract_per_window_embeddings,
)
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.io import write_axis_parquet
from senselab.audio.workflows.audio_analysis.labelstudio import (
    attach_uncertainty_tracks_to_ls,
    uncertainty_to_label_bin,
)
from senselab.audio.workflows.audio_analysis.plot import build_aligned_timeline_plot
from senselab.audio.workflows.audio_analysis.types import (
    AxisResult,
    UncertaintyAxis,
    UncertaintyRow,
)

__all__ = [
    "AGGREGATORS",
    "AxisResult",
    "BucketGrid",
    "UncertaintyAxis",
    "UncertaintyRow",
    "WindowEmbedding",
    "apply_aggregator",
    "attach_uncertainty_tracks_to_ls",
    "build_aligned_timeline_plot",
    "build_disagreements_index",
    "compute_uncertainty_axes",
    "extract_per_window_embeddings",
    "uncertainty_to_label_bin",
    "write_axis_parquet",
]
