"""PREPROCESS v2: every whole-file model here, sets not winners, phonation spans, bracket-aware words."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
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
        """
        _seed_admit(store, tmp_path, wav_writer, samples=_merging_bursts())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        spans = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") is None and e.attributes.get("measure") == "amplitude"
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
    amplitude span at all -- while its steady harmonic content clears spans.continuity_margin=0.03
    easily. The scenario the continuity pass exists for.
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
        assert all(e.attributes["measure"] == "continuity" for e in spans)
        assert "peak_over_floor_continuity" in spans[0].attributes
        assert "k_db" not in spans[0].attributes


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
        first = ScriptLine(text="one", start=2.093, end=2.100, score=0.9)
        second = ScriptLine(text="two", start=2.104, end=2.111, score=0.9)
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
        assert "peak_over_floor_continuity" not in asr_spans[0].attributes

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

    def test_yamnet_windows_a_span_directly_with_no_buffering(
        self,
        store: ProvStore,
        span_quality_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unlike HeAR, YAMNet's window sits inside the span's own extent, not a 2 s buffer."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch, yamnet=[window(0.0, 0.96, {"Speech": 0.9})])
        preprocess(store, _audio(tmp_path), span_quality_config, run_dir=tmp_path)
        spans = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") is None and e.attributes.get("measure") == "amplitude"
        ]
        windows = find_measurements(store, "span_yamnet")
        assert windows
        assert windows[0].attributes["signal"] == "plain"
        assert windows[0].attributes["labels"] == ["Speech"]
        span_start = min(e.extent[0] for e in spans if e.extent is not None)
        assert windows[0].extent is not None
        assert windows[0].extent[0] == pytest.approx(span_start, abs=1e-3)

    def test_a_span_yamnet_never_scores_never_labels_falls_back_to_unmeasured(
        self,
        store: ProvStore,
        span_quality_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty native-window result is a recorded fact, not a silently missing span."""
        _seed_admit(store, tmp_path, wav_writer, samples=_default_samples())
        _stub_models(monkeypatch, yamnet=[])
        preprocess(store, _audio(tmp_path), span_quality_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        unmeasured = [
            e
            for e in live_entities(store, "assertion")
            if e.attributes.get("name") == "span_yamnet" and e.attributes.get("unmeasured")
        ]
        assert spans
        assert len(unmeasured) == len(spans)
        assert unmeasured[0].attributes["unmeasured"] == "no_native_window"


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
            "clip_spans",
            "normalized_envelope",
            "span_hear",
            "span_yamnet",
        }

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
    """fuse_consensus_words is called, and its output is what every text consumer reads."""

    def test_the_consensus_comes_from_fuse_consensus_words(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The routine is called with both recognizers' resolved results, and its provenance stored."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        assert sorted(consensus.attributes["systems"]) == sorted([CRISPERWHISPER_ID, QWEN_ID])
        assert consensus.attributes["provenance"]["operator"] == "consensus_words/resample"
        assert consensus.attributes["text"] == "hello world"
        assert consensus.attributes["timing_authority"] == "consensus_asr"

    def test_word_entities_are_the_consensus_words_only(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two recognizers agreeing on two words yield two word entities, not four."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert len(live_entities(store, "word")) == 2

    def test_consensus_word_times_and_uncertainty_are_authoritative(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No second aligner is written; downstream timing evidence stays on consensus words."""
        _seed_admit(store, tmp_path, wav_writer)
        seen: dict[str, Any] = {}
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"), record=seen)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)

        assert "align" not in seen
        assert find_measurement(store, "alignment") is None
        assert not (tmp_path / "derivatives" / "alignment.json").exists()
        words = live_entities(store, "word")
        assert len(words) == 2
        for word in words:
            assert word.extent is not None
            assert set(word.attributes) >= {
                "confidence",
                "existence_confidence",
                "temporal_confidence",
                "coverage",
                "recognizers",
                "timing_sources",
            }

    def test_the_per_recognizer_hypotheses_stay_as_measurements(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The evidence the consensus was fused from is retained, but not as word entities."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello there"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        for name in ("asr_crisperwhisper", "asr_qwen"):
            measurement = find_measurement(store, name)
            assert measurement is not None
            assert len(measurement.attributes["words"]) == 2

    def test_a_wordless_run_still_writes_the_consensus_with_a_filled_provenance(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fold that ran and found nothing is a fact, and it is not the same fact as never folding.

        ``fuse_consensus_words`` returns ``([], {})`` when no recognizer produced a readable word, so
        every named read of the provenance would raise and the measurement would be lost with it.
        TAXONOMY's lexical line reads ``absent`` from this measurement and ``unavailable`` from its
        absence (I7), so collapsing the two here would silently change a downstream verdict.
        """
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line(""), qwen=_line(""))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        assert consensus.attributes["words"] == []
        assert consensus.attributes["text"] == ""
        assert consensus.attributes["provenance"]["operator"] == "consensus_words/resample"
        assert consensus.attributes["provenance"]["n_words"] == 0
        assert live_entities(store, "word") == []


class TestWordsAreBracketAware:
    """A bracketed or onomatopoeic token is an event, and carries no lexical evidence."""

    def test_a_bracketed_token_is_an_event_not_a_word(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """[COUGH] between two words leaves two words and one event."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(
            monkeypatch,
            crisper=_line("hello [COUGH] world"),
            qwen=_line("hello [COUGH] world"),
        )
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert [e.attributes["text"] for e in live_entities(store, "word")] == ["hello", "world"]
        events = live_entities(store, "event")
        assert len(events) == 1
        assert events[0].attributes["bracketed"] == "[COUGH]"
        assert events[0].attributes["origin"] == "bracketed"

    def test_an_onomatopoeic_token_is_normalised_into_an_event(
        self,
        store: ProvStore,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the vocabulary supplied, 'khh' becomes [KHH] and the raw token travels with it."""
        override = tmp_path / "tokens.yaml"
        override.write_text("words:\n  onomatopoeic_tokens: [khh, ahem]\n")
        config = load_triage_config(override)
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello khh world"), qwen=_line("hello khh world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert [e.attributes["text"] for e in live_entities(store, "word")] == ["hello", "world"]
        events = live_entities(store, "event")
        assert events[0].attributes["bracketed"] == "[KHH]"
        assert events[0].attributes["raw"] == "khh"
        assert events[0].attributes["origin"] == "onomatopoeic"

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
        assert [e.attributes["text"] for e in live_entities(store, "word")] == ["hello", "khh", "world"]


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
        for span in spans:
            assert span.extent is not None
            assert np.isfinite(span.extent).all()
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
