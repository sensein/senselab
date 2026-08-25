"""PREPROCESS v2: every whole-file model here, sets not winners, phonation spans, bracket-aware words."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from scipy.signal import lfilter

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import preprocess as preprocess_module
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import find_measurement, find_measurements, live_entities
from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID, QWEN_ID, preprocess
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import window

SR = 16000


class _FakeModel:
    """A model spec stub carrying exactly what the node reads: path_or_uri and commit_sha."""

    def __init__(self, path_or_uri: str) -> None:
        """Stub a resolved model."""
        self.path_or_uri = path_or_uri
        self.commit_sha = "a" * 40


def _resonate(excitation: np.ndarray, formants: np.ndarray, bandwidth: float = 80.0) -> np.ndarray:
    """Pass an excitation through one two-pole resonator per column of ``formants``."""
    out = excitation.astype(np.float64)
    r = float(np.exp(-np.pi * bandwidth / SR))
    for column in range(formants.shape[1]):
        track = formants[:, column]
        if float(track.min()) == float(track.max()):
            centre = float(track[0])
            out = lfilter([1.0 - r], [1.0, -2 * r * np.cos(2 * np.pi * centre / SR), r * r], out)
            continue
        filtered = np.zeros_like(out)
        previous, before = 0.0, 0.0
        for index, centre in enumerate(track):
            filtered[index] = (
                (1.0 - r) * out[index] + 2 * r * np.cos(2 * np.pi * centre / SR) * previous - r * r * before
            )
            before, previous = previous, filtered[index]
        out = filtered
    return np.asarray(out / (np.abs(out).max() + 1e-12), dtype=np.float64)


def _pulses(rate_hz: np.ndarray) -> np.ndarray:
    """A pulse train whose instantaneous rate follows ``rate_hz``."""
    out = np.zeros(len(rate_hz))
    phase = 0.0
    for index, rate in enumerate(rate_hz):
        phase += rate / SR
        if phase >= 1.0:
            phase -= 1.0
            out[index] = 1.0
    return out


def _fixed(centres: tuple[float, ...], n_samples: int) -> np.ndarray:
    """A constant formant track, one column per centre frequency."""
    return np.tile(np.array([list(centres)]), (n_samples, 1))


def _steady_vowel() -> np.ndarray:
    """1.5 s of a 200 Hz buzz through fixed resonators: a sustained, voiced production."""
    n_samples = int(1.5 * SR)
    return _resonate(_pulses(np.full(n_samples, 200.0)), _fixed((700.0, 1200.0, 2600.0), n_samples))


def _steady_noise() -> np.ndarray:
    """1.5 s of noise through the same resonators: a sustain with no periodicity at all."""
    n_samples = int(1.5 * SR)
    excitation = np.random.default_rng(0).standard_normal(n_samples)
    return _resonate(excitation, _fixed((700.0, 1200.0, 2600.0), n_samples))


def _rising_glide() -> np.ndarray:
    """0.3 s in which F0 and the resonances both sweep upward faster than either limb tolerates."""
    n_samples = int(0.3 * SR)
    ramp = np.linspace(0.0, 1.0, n_samples)
    resonances = np.stack([300.0 + 3300.0 * ramp, 900.0 + 3500.0 * ramp], axis=1)
    swept = _resonate(_pulses(150.0 * (480.0 / 150.0) ** ramp), resonances)
    return np.concatenate([swept, np.zeros(int(0.2 * SR))])


def _line(text: str) -> ScriptLine:
    """One recognizer's result: a chunk per whitespace-separated token, 0.3 s apart."""
    tokens = [token for token in text.split() if token]
    chunks = [
        ScriptLine(text=token, start=0.5 + index * 0.3, end=0.5 + index * 0.3 + 0.2, score=0.9)
        for index, token in enumerate(tokens)
    ]
    if not chunks:
        return ScriptLine(text="", start=0.0, end=0.0, chunks=None, score=0.9)
    return ScriptLine(text=text, start=chunks[0].start, end=chunks[-1].end, chunks=chunks, score=0.9)


def _seed_admit(
    store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path], samples: np.ndarray | None = None
) -> None:
    """Write the fixture recording and run ADMIT over it, so the ``recording`` stream exists."""
    path = wav_writer("input.wav", _default_samples() if samples is None else samples)
    admitted = admit(store, path, load_triage_config(), run_dir=tmp_path)
    assert admitted.audio is not None


def _default_samples() -> np.ndarray:
    """A quiet noise bed with one loud burst — enough contrast for one envelope span."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(3.0 * SR)) * 1e-4).astype(np.float32)
    start = int(1.5 * SR)
    stop = start + int(0.15 * SR)
    grid = np.arange(stop - start) / SR
    samples[start:stop] += (0.5 * np.sin(2 * np.pi * 440.0 * grid)).astype(np.float32)
    return samples


def _audio(tmp_path: Path) -> Audio:
    """The fixture recording, as ADMIT returned it."""
    return Audio(filepath=str(tmp_path / "input.wav"))


def _stub_models(
    monkeypatch: pytest.MonkeyPatch,
    *,
    yamnet: list[dict[str, Any]] | None = None,
    ast: list[dict[str, Any]] | None = None,
    hear: list[dict[str, Any]] | None = None,
    crisper: ScriptLine | None = None,
    qwen: ScriptLine | None = None,
    record: dict[str, Any] | None = None,
) -> None:
    """Replace every model call PREPROCESS makes, on the node module, and record each one's kwargs."""
    seen = record if record is not None else {}

    def fake_classify(audios: list, model: Any, **kwargs: Any) -> list:  # noqa: ANN401
        """YAMNet or AST, told apart by the model the node passed."""
        which = "yamnet" if model == "yamnet" else "ast"
        seen[which] = {"model": model, **kwargs}
        return [list(yamnet or []) if which == "yamnet" else list(ast or [])]

    def fake_hear(audios: list, **kwargs: Any) -> list:  # noqa: ANN401
        """HeAR's event detector, on its fixed 2 s window."""
        seen["hear"] = dict(kwargs)
        return [list(hear or [])]

    def fake_transcribe(audios: list, model: _FakeModel, **kwargs: Any) -> list:  # noqa: ANN401
        """Whichever recognizer the node asked for."""
        seen.setdefault("transcribe", []).append(str(model.path_or_uri))
        return [(crisper if str(model.path_or_uri) == CRISPERWHISPER_ID else qwen) or _line("")]

    def fake_align(items: list, **kwargs: Any) -> list:  # noqa: ANN401
        """One aligned line per input tuple."""
        seen["align"] = len(items)
        return [[ScriptLine(text="hello", start=0.5, end=0.7)] for _ in items]

    def fake_squim(audios: list, device: Any = None) -> list:  # noqa: ANN401
        """One objective-head dict per input."""
        return [{"stoi": 0.91, "pesq": 1.8, "si_sdr": 7.5} for _ in audios]

    monkeypatch.setattr(preprocess_module, "_crisperwhisper_model", lambda: _FakeModel(CRISPERWHISPER_ID))
    monkeypatch.setattr(preprocess_module, "_qwen_model", lambda: _FakeModel(QWEN_ID))
    monkeypatch.setattr(preprocess_module, "_ast_model", lambda: _FakeModel(preprocess_module.AST_ID))
    monkeypatch.setattr(preprocess_module, "classify_audios", fake_classify)
    monkeypatch.setattr(preprocess_module, "detect_health_acoustic_events", fake_hear)
    monkeypatch.setattr(preprocess_module, "transcribe_audios", fake_transcribe)
    monkeypatch.setattr(preprocess_module, "align_transcriptions", fake_align)
    monkeypatch.setattr(preprocess_module, "extract_objective_quality_features_from_audios", fake_squim)


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

    def test_ast_runs_at_the_configured_window_not_at_10_24_s(
        self,
        store: ProvStore,
        windows_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The 'native frame' reasoning is retracted in this repo; the window is a key defaulting to 0.96."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, ast=[window(0.0, 0.96, {"Speech": 0.9})], record=seen)
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert seen["ast"]["win_length"] == pytest.approx(10.24)
        assert seen["ast"]["hop_length"] == pytest.approx(0.48)

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
        assert set(result.absent) == {"yamnet_windows", "ast_windows", "hear_windows", "phonation_spans"}

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


class TestPhonationSpans:
    """Sustains and glides, voiced, unvoiced or mixed, with duration_s as the primary feature."""

    def test_a_sustained_vowel_yields_a_span_carrying_its_duration(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 1.5 s steady tone is one sustained span whose duration_s is its extent."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert spans
        best = max(spans, key=lambda e: e.attributes["duration_s"])
        assert best.attributes["member"] == "sustained"
        assert best.extent is not None
        assert best.attributes["duration_s"] == pytest.approx(best.extent[1] - best.extent[0])
        assert best.attributes["duration_s"] > 1.0

    def test_an_unvoiced_sustain_is_a_span_like_any_other(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Steady band-limited noise sustains with no periodicity and is not refused."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_noise())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert spans
        assert any(e.attributes["production"] in ("unvoiced", "mixed") for e in spans)

    def test_a_glide_is_a_span_with_a_direction_and_an_excursion(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rising sweep is a glide, not a sustain, and carries where it went."""
        _seed_admit(store, tmp_path, wav_writer, samples=_rising_glide())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        glides = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") == "phonation" and e.attributes["member"] == "glide"
        ]
        assert glides
        assert glides[0].attributes["glide_direction"] == "rising"
        assert glides[0].attributes["glide_extent_cents"] > 0.0

    def test_formant_tracks_are_written_per_span_and_sliced_from_the_stream(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One formant_tracks measurement per span, each derived from the span it covers."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        tracks = find_measurements(store, "formant_tracks")
        assert len(tracks) == len(spans)
        assert set(store.derived_from(tracks[0].id)) & {e.id for e in spans}
        assert len(tracks[0].attributes["f1_hz"]) == len(tracks[0].attributes["times_s"])

    def test_a_null_criterion_leaves_the_spans_absent(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The packaged config fits nothing, so the pass is absent rather than run on invented floors."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        result = preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert "phonation_spans" in result.absent
        assert not [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]


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
