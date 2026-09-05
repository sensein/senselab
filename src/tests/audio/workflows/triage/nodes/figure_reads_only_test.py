"""FIGURE reads what the run persisted and derives nothing of its own."""

import ast
from pathlib import Path
from typing import Callable

import numpy as np

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.figure import _continuity
from senselab.utils.prov_store import ProvStore

#: Anything that would let FIGURE measure instead of read. Importing one of these is the regression
#: this guard exists to catch: a curve derived here can differ from the one the spans in the same
#: store were proposed against, and the page would then annotate them with a value that never
#: produced them.
FORBIDDEN_IMPORTS = {
    "classify_audios",
    "detect_clip_events",
    "detect_disruptions",
    "extract_spectrogram_from_audios",
    "gammatone_filterbank",
    "global_floor_dbfs",
    "hilbert_envelope_dbfs",
    "propose_spans",
    "segments_between_change_points",
    "spectral_continuity",
    "transcribe_audios",
}


class TestTheContinuityTraceIsRead:
    """PREPROCESS persists the trace; FIGURE reads that array and no other."""

    def test_the_persisted_trace_comes_back_verbatim(
        self,
        store: ProvStore,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """A ramp no spectral analysis of a silent stream could produce is returned unchanged."""
        ramp = np.linspace(0.0, 1.0, 16000, dtype="float64")
        seed_preprocess_store(store, duration_s=1.0, continuity_trace=ramp, continuity_cut_level=0.25)

        trace, level, percentile = _continuity(store, tmp_path)

        assert trace is not None
        np.testing.assert_allclose(trace, ramp)

    def test_the_recorded_cut_is_read_and_not_recalculated(
        self,
        store: ProvStore,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The level PREPROCESS recorded is used even where it is not the rank cut of the trace."""
        ramp = np.linspace(0.0, 1.0, 16000, dtype="float64")
        seed_preprocess_store(store, duration_s=1.0, continuity_trace=ramp, continuity_cut_level=0.9)

        _, level, percentile = _continuity(store, tmp_path)

        assert (level, percentile) == (0.9, 5.0)

    def test_a_store_predating_the_derivative_reports_absence(
        self,
        store: ProvStore,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """Every run recorded before PREPROCESS persisted the trace yields None, never a fresh curve."""
        seed_preprocess_store(store, duration_s=1.0)

        assert _continuity(store, tmp_path) == (None, None, None)


class TestTheModuleCannotMeasure:
    """The guard is structural: FIGURE may not import a producer at all."""

    def test_it_imports_nothing_that_measures(self) -> None:
        """No measuring API is reachable from the module, so no panel can quietly derive its data."""
        import senselab.audio.workflows.triage.nodes.figure as figure_module

        tree = ast.parse(Path(figure_module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[-1] for alias in node.names}

        offending = sorted(FORBIDDEN_IMPORTS & imported)
        assert not offending, f"FIGURE must read, not measure; it imports {offending}"


class TestPreprocessPersistsTheTrace:
    """The other half of the contract: the trace reaches the store as its own derivative."""

    def test_the_trace_is_registered_as_a_writable_derivative(self, config: TriageConfig) -> None:
        """PREPROCESS declares a ``continuity_trace`` block, so a completed run carries the array."""
        import senselab.audio.workflows.triage.nodes.preprocess as preprocess_module

        source = Path(preprocess_module.__file__).read_text()
        assert '("continuity_trace", _continuity_trace)' in source
        assert 'run_dir / "derivatives" / "continuity_trace.npz"' in source
