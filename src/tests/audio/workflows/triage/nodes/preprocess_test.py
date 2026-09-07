"""PREPROCESS v2: every whole-file model here, sets not winners, phonation spans, bracket-aware words."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.yamnet import YAMNET_WINDOW_SECONDS
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import preprocess as preprocess_module
from senselab.audio.workflows.triage.nodes.common import find_measurement, find_measurements, live_entities
from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID, QWEN_ID, preprocess
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import (
    SR,
    _audio,
    _default_samples,
    _line,
    _seed_admit,
    _stub_models,
    window,
)


def _clipped_at_44k() -> np.ndarray:
    """2 s of a 220 Hz tone driven 3.5 dB past full scale, so it clips in flat plateaus."""
    grid = np.arange(int(2.0 * 44100)) / 44100
    return np.clip(1.5 * np.sin(2 * np.pi * 220.0 * grid), -1.0, 1.0).astype(np.float32)


def _long_burst_samples(burst_s: float = 1.2, total_s: float = 3.5) -> np.ndarray:
    """A quiet noise bed with one loud tone burst long enough to meet YAMNet's own native frame."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(total_s * SR)) * 1e-4).astype(np.float32)
    start = int(1.0 * SR)
    stop = start + int(burst_s * SR)
    grid = np.arange(stop - start) / SR
    samples[start:stop] += (0.5 * np.sin(2 * np.pi * 440.0 * grid)).astype(np.float32)
    return samples


def _samples_with_a_long_gap() -> np.ndarray:
    """A quiet noise bed with a short burst, long enough overall that a gap clears the native frame."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(8.0 * SR)) * 1e-4).astype(np.float32)
    start = int(4.0 * SR)
    stop = start + int(0.15 * SR)
    grid = np.arange(stop - start) / SR
    samples[start:stop] += (0.5 * np.sin(2 * np.pi * 440.0 * grid)).astype(np.float32)
    return samples


def _merging_bursts() -> np.ndarray:
    """Three tone bursts close enough that the offset rule merges all three into one span."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(3.0 * SR)) * 1e-4).astype(np.float32)
    for start, stop, amplitude in ((1.0, 1.15, 0.5), (1.16, 1.31, 0.3), (1.32, 1.47, 0.5)):
        i0, i1 = int(start * SR), int(stop * SR)
        grid = np.arange(i1 - i0) / SR
        samples[i0:i1] += (amplitude * np.sin(2 * np.pi * 440.0 * grid)).astype(np.float32)
    return samples


class TestWindowClassificationsAreSets:
    """A window carries every label over its own threshold, and pooling is set-union."""

    def test_a_window_may_carry_several_labels(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two labels clearing their thresholds in one window are both members; nothing wins."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, yamnet=[window(0.0, 0.96, {"Speech": 0.9, "Cough": 0.7, "Music": 0.1})])
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        pooled = find_measurement(store, "yamnet_windows")
        assert pooled is not None
        assert pooled.attributes["labels"] == ["Cough", "Speech"]
        per_window = find_measurements(store, "yamnet_window")
        assert len(per_window) == 1
        assert sorted(per_window[0].attributes["labels"]) == ["Cough", "Speech"]
        assert set(per_window[0].attributes["scores"]) == {"Cough", "Speech"}

    def test_a_per_label_threshold_overrides_the_default(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Speech at 0.45 clears its own 0.4 while Cough at 0.45 misses the 0.5 default."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, yamnet=[window(0.0, 0.96, {"Speech": 0.45, "Cough": 0.45})])
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert find_measurements(store, "yamnet_window")[0].attributes["labels"] == ["Speech"]

    def test_an_empty_window_is_still_written(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A window nobody's threshold cleared is not the same fact as a window never classified."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(
            monkeypatch,
            yamnet=[window(0.0, 0.96, {"Speech": 0.9}), window(0.48, 1.44, {"Speech": 0.01})],
        )
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        per_window = find_measurements(store, "yamnet_window")
        assert len(per_window) == 2
        assert per_window[1].attributes["labels"] == []
        pooled = find_measurement(store, "yamnet_windows")
        assert pooled is not None
        assert pooled.attributes["n_windows"] == 2

    def test_an_all_subthreshold_hear_window_retains_its_raw_scores(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HeAR display evidence survives even when no label joins the decision set."""
        _seed_admit(store, tmp_path, wav_writer)
        raw_scores = {"Speech": 0.49, "Breathe": 0.35, "Cough": 0.12}
        _stub_models(monkeypatch, hear=[window(0.0, 2.0, raw_scores)])

        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)

        [hear_window] = find_measurements(store, "hear_window")
        assert hear_window.attributes["raw_scores"] == raw_scores
        assert hear_window.attributes["labels"] == []
        assert hear_window.attributes["scores"] == {}
        pooled = find_measurement(store, "hear_windows")
        assert pooled is not None
        assert pooled.attributes["labels"] == []

    def test_pooling_is_union_and_the_windows_are_retained(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The union names the labels; windows_by_label names where each one was."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(
            monkeypatch,
            yamnet=[window(0.0, 0.96, {"Speech": 0.9}), window(0.48, 1.44, {"Cough": 0.9})],
        )
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        pooled = find_measurement(store, "yamnet_windows")
        per_window = find_measurements(store, "yamnet_window")
        assert pooled is not None
        assert pooled.attributes["labels"] == ["Cough", "Speech"]
        assert pooled.attributes["windows_by_label"]["Speech"] == [per_window[0].id]
        assert pooled.attributes["windows_by_label"]["Cough"] == [per_window[1].id]

    def test_the_scores_survive_a_null_threshold(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The packaged config folds nothing, but the model output is still in the store (V3)."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, yamnet=[window(0.0, 0.96, {"Speech": 0.9})])
        result = preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert find_measurement(store, "yamnet_scores") is not None
        assert find_measurement(store, "yamnet_windows") is None
        assert "yamnet_windows" in result.absent

    def test_ast_runs_at_the_owner_directed_window(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AST reads the recording in 10 s windows; 10.24 s is the nearest realisable width."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, ast=[window(0.0, 10.24, {"Speech": 0.9})], record=seen)
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert seen["ast"]["win_length"] == pytest.approx(10.24)

    def test_ast_is_asked_for_its_whole_vocabulary(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """classify_audios does `top_k or 5`, so None would rank 527 labels down to five (C2)."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, ast=[window(0.0, 0.96, {"Speech": 0.9})], record=seen)
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert seen["ast"]["top_k"] == 527

    def test_a_truncating_top_k_would_lose_a_confident_label(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A window carrying six labels over threshold keeps all six; a top-5 rank would drop one."""
        _seed_admit(store, tmp_path, wav_writer)
        scores = {f"L{i}": 0.9 - i * 0.01 for i in range(6)}
        _stub_models(monkeypatch, ast=[window(0.0, 0.96, scores)])
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        ast_window = find_measurements(store, "ast_window")[0]
        assert len(ast_window.attributes["labels"]) == 6

    def test_hear_runs_on_its_fixed_window_at_the_configured_hop(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HeAR's 2 s window is model-imposed; hop_s is the only key."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, hear=[window(0.0, 2.0, {"Cough": 0.9})], record=seen)
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert seen["hear"]["hop_length"] == pytest.approx(2.0)
        pooled = find_measurement(store, "hear_windows")
        assert pooled is not None
        assert pooled.attributes["labels"] == ["Cough"]


class TestSpansCarryTheirMergeRate:
    """A span covering several events says so, and the count comes from production."""

    def test_a_merged_span_reports_every_proposal_it_absorbed(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three bursts, one span, and the stored entity names every proposal that span absorbed.

        The count is raw threshold-crossings rather than events: the three bursts' own inter-burst
        gaps never drop the envelope back below k_db, so the three bursts alone read as one
        continuous crossing, plus a brief pre-onset and post-offset ring each from the zero-phase
        Butterworth envelope -- three crossings in total, close enough together (well under
        min_separation_ms) to be absorbed as one proposal. The count is written by ``propose_spans``
        and copied onto the entity by the node, so this is the assertion that keeps sibling T6's
        merge-rate report reading production rather than a fixture. Asserting the exact number is
        what makes it discriminating: a node that hard-coded the field, or a fixture that supplied
        it, would read one.

        Filtered to the pre-emphasised signal specifically: normalization is on by default now, and
        its own AGC-boosted envelope genuinely finds two more standalone spans elsewhere in this
        fixture's noise bed (its own quiet lead-in and tail, not the bursts this test is about) --
        real, additive supplementary evidence, not a bug, but not what this assertion is testing.
        """
        _seed_admit(store, tmp_path, wav_writer, samples=_merging_bursts())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        spans = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") is None
            and e.attributes.get("measure") == "amplitude"
            and e.attributes.get("signal") == "preemphasised"
        ]
        assert len(spans) == 1
        assert spans[0].attributes["merged_proposals"] == 3

    def test_an_unmerged_span_reports_one(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The contrast the merged case needs: one burst absorbs a small, non-zero proposal count.

        Three, not one: the tone's abrupt onset and offset each ring the zero-phase Butterworth
        envelope (the same overshoot ``TestAnUnmeasurableSampleHasNoDecibelValue`` documents),
        each ring briefly crossing k_db above the floor as its own ~12 ms run before and after the
        166 ms sustained crossing the tone itself produces -- three raw crossings, all well under
        min_separation_ms's 150 ms gate, absorbed as one proposal into the one span they produce.
        """
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        spans = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") is None and e.attributes.get("measure") == "amplitude"
        ]
        assert [e.attributes["merged_proposals"] for e in spans] == [3]


def _burst_that_also_clips() -> np.ndarray:
    """A quiet bed with one loud, hard-clipped burst and one loud, clean burst elsewhere."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(4.0 * SR)) * 1e-4).astype(np.float32)
    grid = np.arange(int(0.15 * SR)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * grid)).astype(np.float32)
    clipped_i0 = int(1.5 * SR)
    samples[clipped_i0 : clipped_i0 + len(tone)] += np.clip(3.0 * tone, -1.0, 1.0)
    clean_i0 = int(3.0 * SR)
    samples[clean_i0 : clean_i0 + len(tone)] += tone
    return samples


def _quiet_sustained_tone() -> np.ndarray:
    """A 500 ms tone too soft to clear the amplitude gate, in an otherwise quiet noise bed.

    1e-3 amplitude against a 1e-4 noise bed: measured directly (through the same pre-emphasis
    PREPROCESS applies), this stays below spans.k_db=6 on the pre-emphasised envelope -- no
    amplitude span at all -- while its steady harmonic content survives the continuity rank cut as
    a run between change points. The scenario the continuity pass exists for.
    """
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(3.0 * SR)) * 1e-4).astype(np.float32)
    start = int(1.0 * SR)
    stop = start + int(0.5 * SR)
    grid = np.arange(stop - start) / SR
    samples[start:stop] += (1e-3 * np.sin(2 * np.pi * 440.0 * grid)).astype(np.float32)
    return samples


class TestSpectralContinuitySpans:
    """A third span source: a sustained tone too soft for either amplitude pass, caught on shape."""

    def test_a_span_too_quiet_for_amplitude_is_found_by_continuity(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No amplitude span exists for this fixture; the continuity pass is what finds anything."""
        _seed_admit(store, tmp_path, wav_writer, samples=_quiet_sustained_tone())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        assert spans
        proposed = [e for e in spans if e.attributes["measure"] != "gap"]
        assert proposed, "continuity should have proposed something to leave gaps around"
        assert all(e.attributes["measure"] == "continuity" for e in proposed)
        assert "continuity_cut_percentile" in spans[0].attributes
        assert "k_db" not in spans[0].attributes
        assert "peak_over_floor_continuity" not in spans[0].attributes, "a rank cut references no floor"


class TestAsrSpans:
    """A fourth span source: the consensus transcript's own word timings, no threshold at all."""

    def test_asr_finds_a_span_neither_amplitude_nor_continuity_did(
        self,
        store: ProvStore,
        asr_span_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A plain noise bed has one broad continuity span elsewhere; ASR alone covers its own gap.

        A stationary noise bed's own spectral shape is steady enough that continuity claims nearly
        the entire recording (accepted, by-design behavior for background/silence, per the owner) --
        re-verified directly against this exact seed and duration after continuity's smoothing
        switched from its own MedianSmoothing(0.2 s) to the shared ButterworthSmoothing(envelope.
        lowpass_hz/.filter_order): continuity spans (0.0003, 2.089), (2.120, 2.993), leaving only
        (2.089, 2.120) -- 31 ms -- genuinely uncovered. A proper Butterworth lowpass has a smoother
        transient response than the retired median window, so it now bridges nearly every stochastic
        dip in stationary noise; a sweep of seeds 0-7 found no wider gap anywhere. The two consensus
        words below are placed, and sized, to fit entirely inside this real but narrow gap -- shorter
        than a real spoken word, a direct consequence of how effectively the new smoothing closes
        gaps in noise, not a choice made for its own sake.
        """
        samples = (np.random.default_rng(0).standard_normal(int(3.0 * SR)) * 1e-4).astype(np.float32)
        _seed_admit(store, tmp_path, wav_writer, samples=samples)
        # Contiguous, so the two words are one run: grouping merges where extents touch, there
        # being no gap threshold to bridge a hole between them.
        first = ScriptLine(text="one", start=2.093, end=2.102, score=0.9)
        second = ScriptLine(text="two", start=2.102, end=2.111, score=0.9)
        line = ScriptLine(text="one two", start=2.093, end=2.111, chunks=[first, second], score=0.9)
        _stub_models(monkeypatch, crisper=line, qwen=line)
        preprocess(store, _audio(tmp_path), asr_span_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        asr_spans = [e for e in spans if e.attributes["measure"] == "asr"]
        assert asr_spans
        assert asr_spans[0].attributes["signal"] == "consensus"
        assert asr_spans[0].extent == pytest.approx((2.093, 2.111), abs=1e-3)
        assert asr_spans[0].attributes["merged_proposals"] == 2
        assert "peak_over_floor_db" not in asr_spans[0].attributes
        assert "continuity_cut_percentile" not in asr_spans[0].attributes

    def test_asr_fully_covered_by_an_existing_span_contributes_nothing(
        self,
        store: ProvStore,
        asr_span_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A consensus word landing entirely inside the amplitude burst's span adds no ASR span."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        word = ScriptLine(text="word", start=1.5, end=1.6, score=0.9)
        line = ScriptLine(text="word", start=1.5, end=1.6, chunks=[word], score=0.9)
        _stub_models(monkeypatch, crisper=line, qwen=line)
        preprocess(store, _audio(tmp_path), asr_span_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        assert spans
        assert all(e.attributes["measure"] != "asr" for e in spans)
        assert any(e.attributes["measure"] == "amplitude" for e in spans)

    def test_asr_spans_are_absent_when_consensus_is_absent(
        self,
        store: ProvStore,
        asr_span_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both recognizers failing leaves no consensus_transcript; the amplitude span still exists."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch)

        def _broken_transcribe(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise ValueError("no recognizer available")

        monkeypatch.setattr(preprocess_module, "transcribe_audios", _broken_transcribe)
        preprocess(store, _audio(tmp_path), asr_span_config, run_dir=tmp_path)
        assert find_measurement(store, "consensus_transcript") is None
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        assert spans
        assert all(e.attributes["measure"] != "asr" for e in spans)
        assert any(e.attributes["measure"] == "amplitude" for e in spans)


class TestClipAndSpans:
    """ClipDaT-derived clip spans, and the foreground-event spans that flag overlap with one."""

    def test_a_hard_clipped_recording_yields_a_clip_span(
        self,
        store: ProvStore,
        spans_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A recording driven past full scale produces a family=clip span over the plateau."""
        _seed_admit(store, tmp_path, wav_writer, samples=_clipped_at_44k(), sampling_rate=44100)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), spans_config, run_dir=tmp_path)
        clips = [e for e in live_entities(store, "span") if e.attributes.get("family") == "clip"]
        assert clips
        assert clips[0].attributes["signal"] == "recording"

    def test_a_clean_recording_has_no_clip_spans(
        self,
        store: ProvStore,
        spans_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A quiet bed and one ordinary burst never touch the recording's own extreme repeatedly."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), spans_config, run_dir=tmp_path)
        assert not [e for e in live_entities(store, "span") if e.attributes.get("family") == "clip"]

    def test_a_burst_yields_a_span_naming_which_signal_found_it(
        self,
        store: ProvStore,
        spans_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A span carries no family beyond clip-overlap; it names which signal it was proposed over."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), spans_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        assert spans
        assert spans[0].attributes["signal"] in {"preemphasised", "normalized"}

    def test_a_span_containing_a_clip_is_flagged_not_excluded(
        self,
        store: ProvStore,
        spans_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Overlap with a clip span is recorded on the span; it is still measured, not dropped."""
        _seed_admit(store, tmp_path, wav_writer, samples=_burst_that_also_clips())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), spans_config, run_dir=tmp_path)
        clips = [e for e in live_entities(store, "span") if e.attributes.get("family") == "clip"]
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        assert clips and spans
        flagged = [e for e in spans if e.attributes["contains_clip"]]
        clean = [e for e in spans if not e.attributes["contains_clip"]]
        assert flagged, "the clipped burst's own span must be flagged"
        assert clean, "the untouched burst elsewhere must not be flagged"

    def test_a_supplementary_span_is_added_only_where_the_primary_pass_missed(
        self,
        store: ProvStore,
        spans_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A span the normalized pass finds is added only where it does not overlap a primary one.

        ``dynamic_range_normalize`` is monkeypatched (real AGC parameters do not reliably boost a
        short, isolated quiet blip enough to demonstrate this deterministically -- measured directly,
        see the comment this replaces in git history) to a fixed transform: it amplifies a quiet
        region the primary pass cannot see at all, leaving the rest of the recording, including the
        main burst, untouched. Real envelope, floor and span-proposal code runs throughout; only the
        normalization step itself is a stand-in.
        """
        samples = _default_samples()
        quiet_start = int(0.3 * SR)
        quiet_stop = quiet_start + int(0.1 * SR)
        grid = np.arange(quiet_stop - quiet_start) / SR
        samples[quiet_start:quiet_stop] = (1e-4 * np.sin(2 * np.pi * 300.0 * grid)).astype(np.float32)

        def fake_normalize(audio: Audio, **kwargs: Any) -> Audio:  # noqa: ANN401
            boosted = audio.waveform.clone()
            tone = (0.5 * np.sin(2 * np.pi * 300.0 * grid)).astype(np.float32)
            boosted[:, quiet_start:quiet_stop] = torch.as_tensor(tone, dtype=boosted.dtype)
            return Audio(waveform=boosted, sampling_rate=audio.sampling_rate)

        monkeypatch.setattr(preprocess_module, "dynamic_range_normalize", fake_normalize)
        _seed_admit(store, tmp_path, wav_writer, samples=samples)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), spans_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        by_signal = {e.attributes["signal"] for e in spans}
        assert "preemphasised" in by_signal, "the main burst must still be found by the primary pass"
        assert "normalized" in by_signal, "the quiet burst must be added by the supplementary pass"
        quiet_span = next(e for e in spans if e.attributes["signal"] == "normalized")
        assert quiet_span.extent is not None
        assert quiet_span.extent[0] < quiet_stop / SR
        assert quiet_span.extent[1] > quiet_start / SR


class TestSpanQuality:
    """Per-span SQUIM, HeAR and YAMNet: raw measurements, no labelling decision."""

    def test_squim_measures_one_assertion_per_span(
        self,
        store: ProvStore,
        span_quality_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SQUIM's objective scores land on the plain signal, one assertion per span."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), span_quality_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        measured = find_measurements(store, "squim")
        assertions = [e for e in live_entities(store, "assertion") if e.attributes.get("name") == "squim"]
        assert spans
        assert len(assertions) == len(spans)
        assert not measured, "SQUIM writes assertions, not measurements"
        assert assertions[0].attributes["stoi"] == pytest.approx(0.91)

    def test_hear_measures_the_plain_signal_with_raw_and_thresholded_scores(
        self,
        store: ProvStore,
        span_quality_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A span's HeAR window carries the full raw distribution, not only what cleared."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch, hear=[window(0.0, 2.0, {"Cough": 0.9, "Breathe": 0.2})])
        preprocess(store, _audio(tmp_path), span_quality_config, run_dir=tmp_path)
        windows = find_measurements(store, "span_hear")
        assert windows
        assert windows[0].attributes["signal"] == "plain"
        assert windows[0].attributes["labels"] == ["Cough"]
        assert set(windows[0].attributes["raw_scores"]) == {"Cough", "Breathe"}
        assert windows[0].attributes["span_id"] in {
            e.id for e in live_entities(store, "span") if e.attributes.get("family") is None
        }

    def test_a_native_span_is_classified_directly_with_no_buffering(
        self,
        store: ProvStore,
        span_quality_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A span at least a native frame long: unlike HeAR, its window sits inside its own extent."""
        _seed_admit(store, tmp_path, wav_writer, samples=_long_burst_samples())
        _stub_models(monkeypatch, yamnet=[window(0.0, 0.96, {"Speech": 0.9})])
        preprocess(store, _audio(tmp_path), span_quality_config, run_dir=tmp_path)
        spans = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") is None and e.attributes.get("measure") == "amplitude"
        ]
        assert spans
        measurement = next(w for w in find_measurements(store, "span_yamnet") if w.attributes["span_id"] == spans[0].id)
        assert measurement.attributes["signal"] == "plain"
        assert measurement.attributes["attribution"] == "native"
        assert measurement.attributes["labels"] == ["Speech"]
        span_start = min(e.extent[0] for e in spans if e.extent is not None)
        assert measurement.extent is not None
        assert measurement.extent[0] == pytest.approx(span_start, abs=1e-3)

    def test_a_short_span_is_attributed_from_its_covering_windows(
        self,
        store: ProvStore,
        span_quality_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A span under the native frame is never classified directly; it takes the covering windows' mean.

        With exactly one covering window, the overlap-weighted mean reduces to that window's own
        scores -- the arithmetic itself is pinned separately in ``TestCoveringWindowAttribution``.
        """
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        covering = window(1.0, 2.5, {"Synthesizer": 0.9, "Speech": 0.2})
        _stub_models(monkeypatch, yamnet=[covering])
        preprocess(store, _audio(tmp_path), span_quality_config, run_dir=tmp_path)
        spans = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") is None and e.attributes.get("measure") == "amplitude"
        ]
        assert spans
        span = spans[0]
        assert span.extent is not None
        assert span.extent[1] - span.extent[0] < YAMNET_WINDOW_SECONDS
        measurement = next(w for w in find_measurements(store, "span_yamnet") if w.attributes["span_id"] == span.id)
        assert measurement.attributes["attribution"] == "covering_windows"
        assert measurement.attributes["covering_windows_n"] == 1
        assert measurement.attributes["covering_seconds"] == pytest.approx(span.extent[1] - span.extent[0])
        assert measurement.attributes["raw_scores"] == {"Synthesizer": 0.9, "Speech": 0.2}
        assert measurement.attributes["labels"] == ["Synthesizer"]

    def test_a_short_span_with_no_covering_window_is_unmeasured(
        self,
        store: ProvStore,
        span_quality_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No window overlaps a short span (the whole-file pass produced none): unmeasured, not padded.

        A native span with the same empty result is a different, already-covered fact
        (``no_native_window``); this pins the short span's own reason distinctly.
        """
        _seed_admit(store, tmp_path, wav_writer, samples=_samples_with_a_long_gap())
        _stub_models(monkeypatch, yamnet=[])
        preprocess(store, _audio(tmp_path), span_quality_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        assert spans
        unmeasured_by_extent = {
            e.extent: e.attributes["unmeasured"]
            for e in live_entities(store, "assertion")
            if e.attributes.get("name") == "span_yamnet" and e.attributes.get("unmeasured")
        }
        assert len(unmeasured_by_extent) == len(spans)
        amplitude_span = next(e for e in spans if e.attributes.get("measure") == "amplitude")
        native_gap = next(
            e
            for e in spans
            if e.attributes.get("measure") == "gap" and e.extent is not None and e.extent[1] - e.extent[0] >= 0.96
        )
        assert unmeasured_by_extent[amplitude_span.extent] == "no_covering_window"
        assert unmeasured_by_extent[native_gap.extent] == "no_native_window"


class TestCoveringWindowAttribution:
    """The overlap-weighted mean, pinned directly against hand-computed numbers."""

    def test_a_single_covering_window_reduces_to_its_own_scores(self) -> None:
        """One window entirely covering the span: the weighted mean is just that window's scores."""
        result = preprocess_module._covering_window_attribution(
            (1.0, 1.5), [window(0.0, 2.0, {"Speech": 0.8, "Music": 0.2})]
        )
        assert result is not None
        scores, n, covering_seconds = result
        assert scores == pytest.approx({"Speech": 0.8, "Music": 0.2})
        assert n == 1
        assert covering_seconds == pytest.approx(0.5)

    def test_two_covering_windows_are_weighted_by_their_overlap_seconds(self) -> None:
        """0.3 s of one window and 0.1 s of another: the mean leans toward the larger overlap."""
        windows = [
            window(0.9, 1.3, {"Speech": 1.0, "Music": 0.0}),  # overlaps [1.0, 1.5) by 0.3 s
            window(1.4, 2.0, {"Speech": 0.0, "Music": 1.0}),  # overlaps [1.0, 1.5) by 0.1 s
        ]
        result = preprocess_module._covering_window_attribution((1.0, 1.5), windows)
        assert result is not None
        scores, n, covering_seconds = result
        assert n == 2
        assert covering_seconds == pytest.approx(0.4)
        # sum(score * overlap) / sum(overlap): Speech = (1.0*0.3 + 0.0*0.1) / 0.4 = 0.75
        assert scores["Speech"] == pytest.approx(0.75)
        assert scores["Music"] == pytest.approx(0.25)

    def test_a_non_overlapping_window_contributes_nothing(self) -> None:
        """A window elsewhere in the file must not be counted or bias the mean."""
        windows = [
            window(0.9, 1.3, {"Speech": 1.0}),
            window(5.0, 6.0, {"Speech": 0.0}),
        ]
        result = preprocess_module._covering_window_attribution((1.0, 1.5), windows)
        assert result is not None
        scores, n, covering_seconds = result
        assert n == 1
        assert covering_seconds == pytest.approx(0.3)
        assert scores["Speech"] == pytest.approx(1.0)

    def test_no_covering_window_returns_none(self) -> None:
        """Nothing overlaps the span: the caller must not invent a score."""
        assert preprocess_module._covering_window_attribution((1.0, 1.5), []) is None
        assert preprocess_module._covering_window_attribution((1.0, 1.5), [window(5.0, 6.0, {"Speech": 1.0})]) is None


class TestThePackagedConfigStillRunsEveryClassifier:
    """V3's split, for all three classifiers: the model runs, the threshold fold is what goes absent."""

    def test_every_classifier_scores_survive_the_packaged_config(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A null threshold must not cost the expensive model output, and a null hop must not either.

        The per-span passes are held to the same rule as the whole-file ones: they are documented
        "raw scores only -- no labelling decision", so a null labelling threshold leaves them
        unlabelled rather than unrun.

        The hops are what made this worth pinning: while ``windows.ast.hop_s`` and
        ``windows.hear.hop_s`` were null, ``require`` raised inside the scores block, so AST and HeAR
        never ran at all under the packaged config and V3 held for one classifier out of three.
        """
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(
            monkeypatch,
            yamnet=[window(0.0, 0.96, {"Speech": 0.9})],
            ast=[window(0.0, 10.24, {"Speech": 0.9})],
            hear=[window(0.0, 2.0, {"Cough": 0.9})],
            crisper=_line("hello world"),
            qwen=_line("hello world"),
        )
        result = preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        for name in ("yamnet_scores", "ast_scores", "hear_scores"):
            assert find_measurement(store, name) is not None, name
        for name in ("yamnet_windows", "ast_windows", "hear_windows"):
            assert find_measurement(store, name) is None, name
        assert set(result.absent) == {
            "yamnet_windows",
            "ast_windows",
            "hear_windows",
            "phonation_tracks",
        }
        for name in ("span_hear", "span_yamnet"):
            windows = find_measurements(store, name)
            assert windows, f"{name} must run: its scores do not depend on a labelling threshold"
            for measurement in windows:
                assert measurement.attributes["raw_scores"], f"{name} kept no scores"
                assert measurement.attributes["labelled"] is False
                assert "labels" not in measurement.attributes

    def test_the_shipped_hops_are_non_overlapping(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each classifier reads the recording once end to end until a hop is fitted."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, record=seen)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert seen["ast"]["hop_length"] == pytest.approx(10.24)
        assert seen["ast"]["win_length"] == pytest.approx(10.24)
        assert seen["hear"]["hop_length"] == pytest.approx(2.0)


class TestPhonationTracks:
    """F0 and formant tracks, measured once over the whole stream. No span, no boundary, no decision.

    Detection over these tracks (sustained-phonation and glide spans) moved to TAXONOMY — see
    ``TestPhonationSpans`` in ``taxonomy_test.py``, which runs PREPROCESS then TAXONOMY together and
    asserts on the spans TAXONOMY proposes from what this node measures.
    """

    def test_the_tracks_are_measured_with_no_span_written(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``phonation_tracks`` measurement exists; no ``span`` of any family does."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        tracks = find_measurement(store, "phonation_tracks")
        assert tracks is not None
        assert tracks.attributes["hop_s"] == pytest.approx(0.01)
        npz = np.load(tmp_path / "derivatives" / "phonation_tracks.npz")
        assert len(npz["f0_hz"]) == len(npz["times_s"])
        assert len(npz["f1_hz"]) == len(npz["formant_times_s"])
        assert not [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert not find_measurements(store, "formant_tracks")

    def test_a_null_criterion_leaves_the_tracks_absent(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The packaged config leaves ``voice.f0_range_hz`` null, so the pass is absent, not guessed."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch)
        result = preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert "phonation_tracks" in result.absent
        assert find_measurement(store, "phonation_tracks") is None


class TestAnAbsenceIsAttributedNotJustClassified:
    """A class name says which of three kinds of failure; it never says which key or which input."""

    def _absent(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict[str, str]:
        """Run PREPROCESS and return the verdict's ``absent`` mapping."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        entity = next(
            e
            for e in store.entities("verdict")
            if e.attributes["node"] == "PREPROCESS" and not store.is_invalidated(e.id)
        )
        return dict(entity.attributes["absent"])

    def test_a_raising_block_records_the_class_and_its_first_line(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bare class name leaves a reader unable to tell which of eleven null keys was read."""
        monkeypatch.setattr(
            preprocess_module,
            "detect_disruptions",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("disruptions.clip_headroom is null\nsecond line")),
        )
        absent = self._absent(store, config, tmp_path, wav_writer, monkeypatch)
        assert absent["disruptions_file"] == "ValueError: disruptions.clip_headroom is null"

    def test_a_message_free_exception_records_the_class_alone(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bare raise must not record a dangling colon."""
        monkeypatch.setattr(
            preprocess_module, "detect_disruptions", lambda *a, **k: (_ for _ in ()).throw(LookupError())
        )
        absent = self._absent(store, config, tmp_path, wav_writer, monkeypatch)
        assert absent["disruptions_file"] == "LookupError"

    def test_a_long_message_is_capped(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cap is what bounds how much of an audio-derived string a message can carry."""
        monkeypatch.setattr(
            preprocess_module,
            "detect_disruptions",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("x" * 500)),
        )
        absent = self._absent(store, config, tmp_path, wav_writer, monkeypatch)
        recorded = absent["disruptions_file"]
        assert len(recorded) <= len("ValueError: ") + 200
        assert recorded.endswith("...")


class TestTheConsensusTranscript:
    """The consensus stream is aligned by ``triage.consensus.align_sources`` over the store's hypotheses."""

    def test_the_consensus_is_the_aligned_stream_with_a_flat_provenance(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The measurement carries the §4.2 fields and the plain join of the words."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        attributes = consensus.attributes
        assert attributes["role"] == "consensus"
        assert attributes["algorithm"] == "star_sequence_alignment"
        assert attributes["routine"] == "senselab.audio.workflows.triage.consensus.align_sources"
        assert attributes["source_order"] == "lexicographic_by_source_name"
        assert [row["name"] for row in attributes["sources"]] == ["asr_crisperwhisper", "asr_qwen"]
        assert attributes["n_sources"] == 2
        assert attributes["text"] == "hello world"
        assert attributes["outcomes"] == {"agreement": 2, "variant": 0, "insertion": 0}
        assert attributes["time_fit"] == "weighted_isotonic_median"
        assert len(attributes["word_ids"]) == attributes["n_words"] == 2
        for retired in ("words", "provenance", "systems", "timing_authority", "event_ids"):
            assert retired not in attributes

    def test_each_hypothesis_measurement_carries_its_role_source_and_model(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 19: the §4.1 field set, with ``model_id`` and ``commit_sha`` matching the agent."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello there"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        for name, model_id in (("asr_crisperwhisper", CRISPERWHISPER_ID), ("asr_qwen", QWEN_ID)):
            measurement = find_measurement(store, name)
            assert measurement is not None
            attributes = measurement.attributes
            assert attributes["role"] == "asr_hypothesis"
            assert attributes["source"] == name
            assert attributes["model_id"] == model_id
            assert attributes["commit_sha"] == "a" * 40
            assert attributes["n_words"] == len(attributes["words"]) == 2
            assert "timestamp_model" in attributes and "duration_s" in attributes
            assert "recognizer" not in attributes
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        rows = {row["name"]: row for row in consensus.attributes["sources"]}
        assert rows["asr_qwen"]["model_id"] == QWEN_ID
        assert rows["asr_qwen"]["commit_sha"] == "a" * 40
        assert rows["asr_qwen"]["measurement_id"] == find_measurement(store, "asr_qwen").id  # type: ignore[union-attr]
        assert rows["asr_qwen"]["agent_id"] is not None

    def test_word_entities_are_the_stream_positions(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two recognizers agreeing on two words yield two word entities carrying the §3.1 attributes."""
        _seed_admit(store, tmp_path, wav_writer)
        seen: dict[str, Any] = {}
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"), record=seen)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert "align" not in seen
        assert find_measurement(store, "alignment") is None
        assert not (tmp_path / "derivatives" / "alignment.json").exists()
        words = live_entities(store, "word")
        assert [w.attributes["index"] for w in words] == [0, 1]
        for word in words:
            assert word.extent is not None
            assert set(word.attributes) == {
                "text",
                "bracketed",
                "outcome",
                "sources",
                "readings",
                "timings",
                "onset_spread_s",
                "offset_spread_s",
                "temporal_uncertainty_s",
                "variants",
                "agreement",
                "index",
            }
            assert word.attributes["outcome"] == "agreement"
            assert word.attributes["sources"] == ["asr_crisperwhisper", "asr_qwen"]
            assert set(word.attributes["timings"]) == {"asr_crisperwhisper", "asr_qwen"}

    def test_the_consensus_reads_every_hypothesis_in_the_store(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 18: a third ``asr_hypothesis`` seeded directly, with no block, joins the consensus."""
        _seed_admit(store, tmp_path, wav_writer)
        extra = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "asr_extra",
                "signal": "plain",
                "role": "asr_hypothesis",
                "source": "asr_extra",
                "model_id": "seeded/extra",
                "commit_sha": None,
                "transcript": "hello world",
                "words": [
                    {"text": "hello", "start": 0.5, "end": 0.7, "score": None},
                    {"text": "world", "start": 0.8, "end": 1.0, "score": None},
                ],
                "n_words": 2,
                "untimed_chunks_n": 0,
                "out_of_bounds_chunks_n": 0,
                "timestamp_source": "native",
                "timestamp_model": None,
                "duration_s": 3.0,
            },
        )
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello there"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        assert [row["name"] for row in consensus.attributes["sources"]] == [
            "asr_crisperwhisper",
            "asr_extra",
            "asr_qwen",
        ]
        assert consensus.attributes["n_sources"] == 3
        assert extra in {row["measurement_id"] for row in consensus.attributes["sources"]}
        [world] = [w for w in live_entities(store, "word") if w.attributes["text"] == "world"]
        assert world.attributes["outcome"] == "variant"
        assert world.attributes["agreement"] == pytest.approx(2 / 3)

    def test_a_single_recognizer_is_an_absence_not_a_consensus(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 20 (R-2): one block raising leaves no consensus and no word; the other blocks still run."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"))
        real = preprocess_module.transcribe_audios

        def _one_fails(audios: list, model: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if str(model.path_or_uri) == QWEN_ID:
                raise LookupError("qwen unavailable")
            return real(audios, model=model, **kwargs)

        monkeypatch.setattr(preprocess_module, "transcribe_audios", _one_fails)
        result = preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        absent = dict(store.get_entity(result.verdict_entity_id).attributes["absent"])
        assert absent["consensus_transcript"].startswith("LookupError: consensus needs at least two")
        assert "found 1: ['asr_crisperwhisper']" in absent["consensus_transcript"]
        assert find_measurement(store, "consensus_transcript") is None
        assert live_entities(store, "word") == []
        assert find_measurement(store, "asr_crisperwhisper") is not None
        assert find_measurement(store, "energy_envelope") is not None

    def test_a_defect_in_the_consensus_fails_the_node(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 21: a RuntimeError from align_sources is a hard failure naming the block."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"))

        def _broken(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise RuntimeError("alignment blew up")

        monkeypatch.setattr(preprocess_module, "align_sources", _broken)
        with pytest.raises(RuntimeError, match="consensus_transcript: RuntimeError: alignment blew up"):
            preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)

    def test_provenance_links_words_to_the_sources_that_produced_them(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 22: the activity used every hypothesis; an insertion derives from its source only."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("I uh think"), qwen=_line("I think"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        crisper = find_measurement(store, "asr_crisperwhisper")
        qwen = find_measurement(store, "asr_qwen")
        assert crisper is not None and qwen is not None
        activity = store.generated_by(consensus.id)
        assert activity is not None
        assert set(store.uses_of(activity)) == {crisper.id, qwen.id}
        assert set(store.derived_from(consensus.id)) == {crisper.id, qwen.id}
        by_text = {w.attributes["text"]: w for w in live_entities(store, "word")}
        assert store.derived_from(by_text["uh"].id) == [crisper.id]
        assert set(store.derived_from(by_text["think"].id)) == {crisper.id, qwen.id}

    def test_a_wordless_run_still_writes_the_consensus(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 24: two hypotheses with no word is an aligned nothing, not a missing alignment."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line(""), qwen=_line(""))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        assert consensus.attributes["n_words"] == 0
        assert consensus.attributes["word_ids"] == []
        assert consensus.attributes["text"] == ""
        assert consensus.attributes["reference_source"] is None
        assert consensus.attributes["n_sources"] == 2
        assert live_entities(store, "word") == []

    def test_the_stream_order_is_the_column_order_not_the_time_order(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The finding: a repetition is emitted verbatim in column order, and no word is re-sorted."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("and the and the d- the"), qwen=_line("and the"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        assert consensus.attributes["text"] == "and the and the d- the"
        words = sorted(live_entities(store, "word"), key=lambda w: int(w.attributes["index"]))
        # The seeded chunks put Qwen's "and"/"the" exactly on the first of each repeated pair, so
        # those are the copies the alignment pairs and the later copies are the insertions.
        assert [w.attributes["outcome"] for w in words] == [
            "agreement",
            "agreement",
            "insertion",
            "insertion",
            "insertion",
            "insertion",
        ]
        onsets = [w.extent[0] for w in words if w.extent is not None]
        assert onsets == sorted(onsets)


class TestWordsAreBracketAware:
    """A bracketed or onomatopoeic token is a bracketed word: in the stream, not lexical."""

    def test_a_bracketed_token_is_a_bracketed_word_in_stream_order(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 23: hello [COUGH] world from both is three words, the middle one bracketed and agreed."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello [COUGH] world"), qwen=_line("hello [COUGH] world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        words = sorted(live_entities(store, "word"), key=lambda w: int(w.attributes["index"]))
        assert [w.attributes["text"] for w in words] == ["hello", "[COUGH]", "world"]
        assert [w.attributes["bracketed"] for w in words] == [False, True, False]
        assert words[1].attributes["outcome"] == "agreement"
        assert "event" not in {e.prov_type for e in store.entities()}

    def test_an_onomatopoeic_token_is_normalised_into_a_bracketed_word(
        self,
        store: ProvStore,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the vocabulary supplied, 'khh' becomes [KHH] and the raw token stays in the readings."""
        override = tmp_path / "tokens.yaml"
        override.write_text("words:\n  onomatopoeic_tokens: [khh, ahem]\n")
        config = load_triage_config(override)
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello khh world"), qwen=_line("hello khh world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        words = sorted(live_entities(store, "word"), key=lambda w: int(w.attributes["index"]))
        assert [w.attributes["text"] for w in words] == ["hello", "[KHH]", "world"]
        assert words[1].attributes["bracketed"] is True
        assert words[1].attributes["readings"] == {"asr_crisperwhisper": "khh", "asr_qwen": "khh"}

    def test_a_null_vocabulary_leaves_an_onomatopoeic_token_a_word(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The honest unfitted state: nobody drew the vocabulary, so nothing is normalised."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello khh world"), qwen=_line("hello khh world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        words = sorted(live_entities(store, "word"), key=lambda w: int(w.attributes["index"]))
        assert [w.attributes["text"] for w in words] == ["hello", "khh", "world"]
        assert not any(w.attributes["bracketed"] for w in words)

    def test_the_asr_span_source_reads_lexical_words_and_changes_no_word(
        self,
        store: ProvStore,
        asr_span_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 25 (R-4): a bracketed word proposes no span; the stream is the same either way."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        cough = ScriptLine(text="[COUGH]", start=2.5, end=2.7, score=0.9)
        line = ScriptLine(text="[COUGH]", start=2.5, end=2.7, chunks=[cough], score=0.9)
        _stub_models(monkeypatch, crisper=line, qwen=line)
        preprocess(store, _audio(tmp_path), asr_span_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        assert all(e.attributes["measure"] != "asr" for e in spans)
        words = live_entities(store, "word")
        assert [w.attributes["text"] for w in words] == ["[COUGH]"]
        assert words[0].extent == (2.5, 2.7)


class TestDisruptionsAreMeasuredOnTheOriginal:
    """The file-level reading exists whatever the transcript says (V9, V10)."""

    def test_a_wordless_file_still_carries_a_file_level_disruption_reading(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No words is not no measurement; that confusion is what this row exists to remove."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line(""), qwen=_line(""))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        measurement = find_measurement(store, "disruptions_file")
        assert measurement is not None
        assert measurement.attributes["clipped_runs"] == 0
        assert "zero_crossing_rate" in measurement.attributes

    def test_the_reading_is_taken_at_the_original_rate_and_level(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hard-clipped 44.1 kHz recording reports its clipping, and reports it at 44100.

        Naming the stream is not enough on its own: rewriting the block to read ``plain`` leaves the
        ``signal`` attribute untouched and every other assertion in this class passes. What the plain
        stream cannot fake is the evidence -- it is peak-normalised, which lifts the samples off full
        scale, and resampled to 16 kHz, which rounds the flat plateaus clipping consists of into
        ripple. So the mutation reads sampling_rate 16000 and clipped_runs 0, and both are pinned.
        """
        _seed_admit(store, tmp_path, wav_writer, samples=_clipped_at_44k(), sampling_rate=44100)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        measurement = find_measurement(store, "disruptions_file")
        assert measurement is not None
        assert measurement.attributes["sampling_rate"] == 44100
        assert measurement.attributes["clipped_runs"] > 0
        assert measurement.attributes["clipped_s"] > 0.0

    def test_the_reading_names_the_original_recording_stream(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Peak normalisation and resampling destroy the defects, so the stream must be the original."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        measurement = find_measurement(store, "disruptions_file")
        assert measurement is not None
        assert measurement.attributes["signal"] == "recording"


def _burst_with_a_sharp_offset() -> np.ndarray:
    """3 s of digital silence around one 0.5 s tone, whose offset makes the envelope undershoot."""
    grid = np.arange(int(3.0 * SR)) / SR
    samples = np.zeros_like(grid)
    voiced = (grid >= 1.0) & (grid < 1.5)
    samples[voiced] = 0.6 * np.sin(2 * np.pi * 440.0 * grid[voiced])
    return samples.astype(np.float32)


class TestTheEnvelopeSidecarHoldsMeasurementsOnly:
    """An undershooting filter has no dB value to write there, and a clamp is not a measurement."""

    def test_the_written_envelope_carries_no_fabricated_floor(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """-240 dBFS in the sidecar is 20*log10(1e-12), which REPORT then drew as the panel's floor."""
        _seed_admit(store, tmp_path, wav_writer, samples=_burst_with_a_sharp_offset())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        loaded = np.load(tmp_path / "derivatives" / "energy_envelope.npz")
        envelope = loaded["envelope_dbfs"]
        assert not np.any(envelope <= -240.0)
        assert np.isnan(envelope).any(), "digital silence and the offset undershoot are unmeasurable"
        assert float(np.nanmax(envelope)) > -20.0

    def test_the_written_floor_never_reads_as_unmeasurably_low(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mostly digital silence makes the 10th percentile of |samples| exactly 0; the floor must not be -inf."""
        _seed_admit(store, tmp_path, wav_writer, samples=_burst_with_a_sharp_offset())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        loaded = np.load(tmp_path / "derivatives" / "energy_envelope.npz")
        assert not np.any(loaded["floor_dbfs"] <= -240.0)

    def test_no_span_extent_reaches_the_end_of_the_recording(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unmeasurable hangover window used to keep the offset open to the last sample."""
        _seed_admit(store, tmp_path, wav_writer, samples=_burst_with_a_sharp_offset())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        # A gap span reaching the end is the trailing background, which is the point of gaps; the
        # hangover this guards against would show as a *proposed* span running to the last sample.
        for span in spans:
            assert span.extent is not None
            assert np.isfinite(span.extent).all()
        for span in (e for e in spans if e.attributes["measure"] != "gap"):
            assert span.extent[1] < 2.9, "the burst ends at 1.5 s; a span to 3.0 s is the NaN hangover"


class TestAnUnexpectedBlockFailureIsNotAbsorbed:
    """A failure that is neither a null-config ValueError nor a missing-prerequisite LookupError."""

    def test_every_other_block_still_runs_before_preprocess_raises(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The loop does not abort early: unrelated derivatives, before and after the break, survive."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch)

        def _broken(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise RuntimeError("torchaudio blew up")

        monkeypatch.setattr(preprocess_module, "extract_spectrogram_from_audios", _broken)

        with pytest.raises(RuntimeError) as excinfo:
            preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)

        message = str(excinfo.value)
        assert "spectrogram_wideband" in message
        assert "spectrogram_narrowband" in message
        # Before the failing blocks in `blocks`' own order:
        assert find_measurement(store, "energy_envelope") is not None
        # After the failing blocks: `spans` still proposes from the amplitude sources alone, and
        # `gammatone` does not touch the spectrogram at all.
        assert live_entities(store, "span")
        assert find_measurement(store, "gammatone") is not None
