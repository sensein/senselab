"""Tests for the breathing-pattern measure."""

import json
from pathlib import Path

import numpy as np

from senselab.audio.workflows.triage.breath_pattern import (
    ALTERNATING_BREATHS,
    NO_BREATHING,
    SINGLE_BREATH,
    VETO_LITTLE_ACTIVITY,
    VETO_NOISE,
    VETO_SILENCE,
    VETO_SPEECH,
    BreathVeto,
    ModulationReading,
    breath_pattern_parameters,
    breath_veto_of,
    measure_breath_pattern,
    measure_modulation,
    modulation_parameters,
)
from senselab.utils.prov_store import ProvStore

HOP_S = 0.005
BIN_HZ = 50.0
BINS = 161


def _spectrogram(duration_s: float, breaths: list[tuple[float, float]], level_db: float = 30.0) -> np.ndarray:
    """A noise floor with flat broadband bursts at each (start, length), level_db above the floor."""
    rng = np.random.default_rng(0)
    frames = int(duration_s / HOP_S)
    power = rng.uniform(0.5, 1.5, size=(BINS, frames)) * 1e-6
    for start, length in breaths:
        a, b = int(start / HOP_S), int((start + length) / HOP_S)
        power[:, a:b] *= 10 ** (level_db / 10)
    return power


def _measure(power: np.ndarray, voiced: np.ndarray | None = None) -> str:
    frames = power.shape[1]
    mask = np.zeros(frames, dtype=bool) if voiced is None else voiced
    return measure_breath_pattern(
        power, hop_s=HOP_S, bin_hz=BIN_HZ, voiced=mask, parameters=breath_pattern_parameters()
    ).pattern


def test_alternating_breaths_read_as_a_pattern() -> None:
    """Five 1 s breaths every 3 s over a quiet floor are alternating breaths."""
    assert _measure(_spectrogram(18.0, [(1 + 3 * k, 1.0) for k in range(5)])) == ALTERNATING_BREATHS


def test_one_long_breath_is_a_single_breath() -> None:
    """One 1.6 s breath, as 7c169ccc held where three quick ones were asked."""
    assert _measure(_spectrogram(8.5, [(3.0, 1.6)])) == SINGLE_BREATH


def test_a_silent_recording_has_no_breathing() -> None:
    """The noise floor alone holds no breath."""
    assert _measure(_spectrogram(20.0, [])) == NO_BREATHING


def test_breaths_below_the_rise_are_no_breathing() -> None:
    """Breaths only 3 dB above the floor are too quiet to count: the owner discards them."""
    assert _measure(_spectrogram(18.0, [(1 + 3 * k, 1.0) for k in range(5)], level_db=3.0)) == NO_BREATHING


def test_a_voiced_burst_is_not_a_breath() -> None:
    """A burst voiced over most of its length is speech, not breath."""
    power = _spectrogram(10.0, [(4.0, 1.0)])
    voiced = np.zeros(power.shape[1], dtype=bool)
    voiced[int(4.0 / HOP_S) : int(5.0 / HOP_S)] = True
    assert _measure(power, voiced) == NO_BREATHING


def _modulation(power: np.ndarray):  # noqa: ANN202
    return measure_modulation(
        power,
        hop_s=HOP_S,
        bin_hz=BIN_HZ,
        smooth_s=breath_pattern_parameters().smooth_s,
        parameters=modulation_parameters(),
    )


def test_five_breath_cycles_read_as_five_breaths() -> None:
    """An inhale and an exhale per breath make two peaks each; ten peaks are five breaths."""
    bursts = [(2 + k * 5.5 + offset, length) for k in range(5) for offset, length in ((0.5, 1.0), (2.0, 1.4))]
    reading = _modulation(_spectrogram(30.0, bursts))
    assert reading is not None
    assert reading.breathing
    assert reading.estimated_breaths == 5


def test_syllabic_bursts_are_not_breathing_cycles() -> None:
    """Bursts at a syllabic rate put the energy in the syllabic band, so no cycles count."""
    bursts = [(2 + k * 0.25, 0.12) for k in range(100)]
    reading = _modulation(_spectrogram(30.0, bursts))
    assert reading is not None
    assert not reading.breathing


def test_silence_has_no_breathing_cycles() -> None:
    """A floor with no burst has no active span, so no cycles count."""
    reading = _modulation(_spectrogram(30.0, []))
    assert reading is not None
    assert not reading.breathing and reading.active_span_s == 0


def test_a_recording_shorter_than_the_reading_needs_has_none() -> None:
    """Under min_duration_s the modulation spectrum is not read."""
    assert _modulation(_spectrogram(2.0, [(0.5, 1.0)])) is None


def test_an_odd_peak_count_rounds_up_to_the_unpaired_breath() -> None:
    """Nine peaks are an unpaired burst beside four full breaths: five breaths."""
    bursts = [(2 + k * 5.5 + offset, length) for k in range(5) for offset, length in ((0.5, 1.0), (2.0, 1.4))][:9]
    reading = _modulation(_spectrogram(30.0, bursts))
    assert reading is not None
    assert reading.modulation_peaks == 9
    assert reading.estimated_breaths == 5


def _scored_store(tmp_path: Path, yamnet: list[dict[str, float]] | None, words: int = 0) -> ProvStore:
    store = ProvStore(run_id="t")
    if yamnet is not None:
        windows = [
            {"start": k, "end": k + 1, "label_scores": [{label: v} for label, v in scores.items()]}
            for k, scores in enumerate(yamnet)
        ]
        (tmp_path / "yamnet_scores.json").write_text(json.dumps(windows))
        store.entity(
            prov_type="measurement", extent=None, attributes={"name": "yamnet_scores", "path": "yamnet_scores.json"}
        )
    for k in range(words):
        store.entity(prov_type="word", extent=(k, k + 0.5), attributes={"index": k, "text": "word", "bracketed": False})
    return store


def _cycles(ratio_db: float = 10.0, span_s: float = 25.0) -> ModulationReading:
    return ModulationReading(
        breath_vs_syllabic_db=ratio_db, active_span_s=span_s, modulation_peaks=8, estimated_breaths=4, breathing=True
    )


def _veto(tmp_path: Path, yamnet: list[dict[str, float]] | None, *, words: int = 0, **kw: float) -> object:
    reading = _cycles(**kw)
    return breath_veto_of(_scored_store(tmp_path, yamnet, words), tmp_path, modulation=reading, duration_s=30.0)


def test_a_breath_neither_classifier_scores_is_not_vetoed(tmp_path: Path) -> None:
    """324cd5d0 / 2337c1e6: no YAMNet or HeAR breath score, and nothing says it is not breathing."""
    veto = _veto(tmp_path, [{"Breathing": 0.0, "Silence": 0.2}] * 4)
    assert isinstance(veto, BreathVeto) and veto.vetoed_by is None


def test_lexical_words_veto_as_speech(tmp_path: Path) -> None:
    """a03b5325: ten consensus words ("Just hit record again...") are speech, not breathing."""
    veto = _veto(tmp_path, [{"Speech": 0.9}] * 4, words=10)
    assert isinstance(veto, BreathVeto) and veto.vetoed_by == VETO_SPEECH


def test_vehicle_noise_vetoes(tmp_path: Path) -> None:
    """ae2a7223: YAMNet hears a vehicle throughout."""
    veto = _veto(tmp_path, [{"Vehicle": 0.5}] * 4)
    assert isinstance(veto, BreathVeto) and veto.vetoed_by == VETO_NOISE


def test_too_little_activity_vetoes(tmp_path: Path) -> None:
    """4d596bce: a 2.4 s active span in a 20-30 s recording is too little sound to be breathing."""
    veto = _veto(tmp_path, [{"Silence": 1.0}] * 4, span_s=2.4)
    assert isinstance(veto, BreathVeto) and veto.vetoed_by == VETO_LITTLE_ACTIVITY


def test_silence_with_weak_cycles_vetoes_but_quiet_breathing_does_not(tmp_path: Path) -> None:
    """5cc93330 (silence 0.76, 1.2 dB) is vetoed; 324cd5d0 (silence 0.998, 3.7 dB) has breathing."""
    weak = _veto(tmp_path, [{"Silence": 0.76}] * 4, ratio_db=1.2)
    quiet = _veto(tmp_path, [{"Silence": 0.998}] * 4, ratio_db=3.7)
    assert isinstance(weak, BreathVeto) and weak.vetoed_by == VETO_SILENCE
    assert isinstance(quiet, BreathVeto) and quiet.vetoed_by is None


def test_absent_yamnet_windows_name_their_input(tmp_path: Path) -> None:
    """Without YAMNet's windows the vetoes cannot be read: the reading names the absent input."""
    assert _veto(tmp_path, None) == ("yamnet_scores",)
