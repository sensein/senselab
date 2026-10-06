"""Tests for the breathing-pattern measure."""

import json
from pathlib import Path

import numpy as np

from senselab.audio.workflows.triage.breath_pattern import (
    ALTERNATING_BREATHS,
    EXTENT_AIRWAY_EVENTS,
    EXTENT_MEASURE_EVENTS,
    EXTENT_MODULATION,
    NO_BREATHING,
    OVER_TASK_EXTENT,
    SINGLE_BREATH,
    VETO_LITTLE_ACTIVITY,
    VETO_SPEECH,
    BreathExtent,
    BreathVeto,
    breath_extent_fallback,
    breath_pattern_parameters,
    breath_veto_of,
    extent_parameters,
    measure_breath_extent,
    measure_breath_pattern,
    measure_modulation,
    modulation_parameters,
    tighten_to_events,
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


EXTENT = (2.0, 12.0)


def _scored_store(
    tmp_path: Path,
    yamnet: list[dict[str, float]] | None = None,
    words: tuple[tuple[str, float], ...] = (),
    extent: tuple[float, float] | None = EXTENT,
) -> ProvStore:
    """A store with YAMNet windows one second apart, timed consensus words and a task-extent span."""
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
    for k, (text, at) in enumerate(words):
        store.entity(prov_type="word", extent=(at, at + 0.4), attributes={"index": k, "text": text, "bracketed": False})
    if extent is not None:
        store.entity(prov_type="span", extent=extent, attributes={"role": "task_extent"})
    return store


def _veto(store: ProvStore, tmp_path: Path, fraction: float | None = 0.9, language: str | None = "en") -> BreathVeto:
    return breath_veto_of(
        store,
        tmp_path,
        active_fraction=fraction,
        active_over=OVER_TASK_EXTENT,
        extent=EXTENT,
        language=language,
    )


def test_a_breath_neither_classifier_scores_is_not_vetoed(tmp_path: Path) -> None:
    """324cd5d0 / 2337c1e6: no YAMNet or HeAR breath score; scores are context and veto nothing."""
    veto = _veto(_scored_store(tmp_path, [{"Breathing": 0.0, "Silence": 0.99}] * 20), tmp_path)
    assert veto.vetoed_by is None
    assert veto.silence_mean == 0.99


def test_speech_words_inside_the_task_extent_veto(tmp_path: Path) -> None:
    """517381e9: words spoken across the task extent are speech, not breathing."""
    words = tuple((w, 3.0 + k) for k, w in enumerate("I am just going to keep talking here".split()))
    veto = _veto(_scored_store(tmp_path, words=words), tmp_path)
    assert veto.vetoed_by == VETO_SPEECH and veto.lexical_words_n == 8


def test_speech_outside_the_task_extent_does_not_veto(tmp_path: Path) -> None:
    """6ca9935e: "breath followed by speech. does not overlap in task extent"."""
    words = tuple((w, 14.0 + k) for k, w in enumerate("I forgot a lot of that".split()))
    veto = _veto(_scored_store(tmp_path, words=words), tmp_path)
    assert veto.vetoed_by is None and veto.lexical_words_n == 0


def test_interjections_and_other_scripts_are_not_speech(tmp_path: Path) -> None:
    """c9b77a28 / 988c1609: repeated 唉, 啊 and 呜 are a recognizer's rendering of a sigh, not words."""
    words = tuple((w, 3.0 + k) for k, w in enumerate(["唉", "啊", "呜", "啊", "呜", "唉"]))
    veto = _veto(_scored_store(tmp_path, words=words), tmp_path, language="en")
    assert veto.vetoed_by is None and veto.lexical_words_n == 0


def test_too_little_activity_over_the_extent_vetoes(tmp_path: Path) -> None:
    """167ac3f5 ("has little breath"): an active span a fifth of the task extent is too little."""
    veto = _veto(_scored_store(tmp_path), tmp_path, fraction=0.2)
    assert veto.vetoed_by == VETO_LITTLE_ACTIVITY


def test_noise_and_silence_do_not_veto(tmp_path: Path) -> None:
    """0b9b68cf (vehicle 0.40) and b26208dd (silence 0.74) both hold breathing the owner heard."""
    noisy = _veto(_scored_store(tmp_path, [{"Vehicle": 0.6}] * 20), tmp_path)
    quiet = _veto(_scored_store(tmp_path, [{"Silence": 0.8}] * 20), tmp_path)
    assert noisy.vetoed_by is None and noisy.noise_mean == 0.6
    assert quiet.vetoed_by is None


def test_absent_classifier_windows_do_not_stop_the_veto(tmp_path: Path) -> None:
    """The classifier scores are context only: without YAMNet's windows the veto still reads."""
    veto = _veto(_scored_store(tmp_path, None), tmp_path)
    assert veto.vetoed_by is None and veto.speech_mean is None


def _extent(power: np.ndarray):  # noqa: ANN202
    return measure_breath_extent(
        power,
        hop_s=HOP_S,
        bin_hz=BIN_HZ,
        smooth_s=breath_pattern_parameters().smooth_s,
        modulation=modulation_parameters(),
        parameters=extent_parameters(),
    )


def _cycles(start: float, n: int) -> list[tuple[float, float]]:
    return [(start + k * 5.5 + offset, length) for k in range(n) for offset, length in ((0.5, 1.0), (2.0, 1.4))]


def test_breathing_in_the_first_half_bounds_the_extent_there() -> None:
    """6f73b8d2: the breathing sits in the first half; the extent ends before the silent second half."""
    extent = _extent(_spectrogram(60.0, _cycles(1.0, 5)))
    assert extent is not None
    assert extent.source == EXTENT_MODULATION
    assert extent.start_s < 3.0
    assert 25.0 < extent.end_s < 35.0


def test_the_extent_stops_where_speech_takes_over() -> None:
    """6ca9935e: breathing followed by speech; syllabic bursts after the breaths are left outside."""
    speech = [(32.0 + k * 0.25, 0.12) for k in range(100)]
    extent = _extent(_spectrogram(60.0, _cycles(1.0, 5) + speech))
    assert extent is not None
    assert extent.end_s < 36.0


def test_silence_has_no_modulation_extent() -> None:
    """No window breathes, so the modulation gives no extent."""
    assert _extent(_spectrogram(30.0, [])) is None


def test_a_recording_shorter_than_a_window_has_no_modulation_extent() -> None:
    """Shorter than one window: the fallbacks decide."""
    assert _extent(_spectrogram(3.0, [(0.5, 1.0)])) is None


def test_the_fallback_is_the_measures_events_then_airways_hull() -> None:
    """Without a modulation extent, the measure's events (padded) stand, then AIRWAY's own hull."""
    pad = extent_parameters().pad_s
    events = breath_extent_fallback(((2.0, 3.0), (5.0, 6.0)), (0.0, 1.0), duration_s=10.0, pad_s=pad)
    assert events is not None and events.source == EXTENT_MEASURE_EVENTS
    assert events.bounds == (2.0 - pad, 6.0 + pad)
    airway = breath_extent_fallback((), (1.0, 4.0), duration_s=10.0, pad_s=pad)
    assert airway is not None and airway.source == EXTENT_AIRWAY_EVENTS and airway.bounds == (1.0, 4.0)
    assert breath_extent_fallback((), None, duration_s=10.0, pad_s=pad) is None


def test_a_modulation_extent_narrows_to_the_breath_events_inside_it() -> None:
    """Talk at a window's edge is left outside: the extent runs from the first to the last event, padded."""
    wide = BreathExtent(start_s=0.0, end_s=20.0, source=EXTENT_MODULATION, estimated_breaths=3)
    narrowed = tighten_to_events(wide, ((12.0, 13.0), (15.0, 16.5)), duration_s=20.0, pad_s=1.0)
    assert narrowed.bounds == (11.0, 17.5)
    assert narrowed.source == EXTENT_MODULATION
    assert tighten_to_events(wide, (), duration_s=20.0, pad_s=1.0) == wide

