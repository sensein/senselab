"""Three-axis uncertainty workflow for analyze_audio outputs.

Reads cached / in-memory results from senselab's per-task audio pipeline
(diarization, ASR, scene classification, alignment, PPG) and emits three
per-bucket uncertainty time series — `speech_presence`, `speaker`, and `asr` —
plus a ranked `disagreements.json` index and a 5-row timeline plot.

See ``specs/20260508-173136-compare-uncertainty/spec.md`` for the full design.
The reusable workflow is consumed by ``scripts/analyze_audio.py`` as a thin
wrapper, but it is also importable standalone:

    from senselab.audio.workflows.audio_analysis import compute_uncertainty_axes

Public symbols are resolved lazily (PEP 562, same pattern as ``adaptive/``) so
that importing the *pure* submodules (``aggregate``, ``aggregators``, ``grid``,
``types``, ``votes``, ``harvesters``, ``adaptive.*``) never pulls
torch / speechbrain / transformers — those load only when a model-touching
symbol (``compute_uncertainty_axes``, ``extract_per_window_embeddings``, the
plot) is actually requested (architecture-review.md F1 / T046).
"""

from typing import Any

# symbol name → defining submodule (all imports deferred to first attribute access).
_LAZY_EXPORTS = {
    "AGGREGATORS": "aggregators",
    "apply_aggregator": "aggregators",
    "compute_uncertainty_axes": "compute",
    "harvest_pass": "compute",
    "build_disagreements_index": "disagreements",
    "WindowEmbedding": "embeddings",
    "extract_per_window_embeddings": "embeddings",
    "BucketGrid": "grid",
    "STAGE_VERSIONS": "stage_context",
    "PassPlan": "stage_context",
    "StageContext": "stage_context",
    "run_pass": "stages",
    "stage_code_version": "stage_context",
    "write_signal_parquet": "io",
    "write_signal_stability": "io",
    "attach_uncertainty_tracks_to_ls": "labelstudio",
    "uncertainty_to_label_bin": "labelstudio",
    "build_aligned_timeline_plot": "plot",
    "FusedAxis": "types",
    "SignalResult": "types",
    "SignalRow": "types",
    "UncertaintyAxis": "types",
    "LinkedPass": "votes",
    "PassHarvest": "votes",
    "link_pass": "votes",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 — lazy re-export
    """Resolve public symbols on first access without importing heavy submodules."""
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{submodule}"), name)


def __dir__() -> list[str]:
    """Expose lazy exports to introspection / pdoc."""
    return __all__
