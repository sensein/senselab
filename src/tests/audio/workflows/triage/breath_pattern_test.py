"""Tests for the breathing-pattern measure."""

import json
from pathlib import Path

import numpy as np

from senselab.audio.workflows.triage.breath_pattern import (
    ALTERNATING_BREATHS,
    EXTENT_AIRWAY_EVENTS,
    EXTENT_BREATH_TRAIN,
    EXTENT_MEASURE_EVENTS,
    NO_BREATHING,
    OVER_TASK_EXTENT,
    SINGLE_BREATH,
    VETO_LITTLE_ACTIVITY,
    VETO_SPEECH,
    BreathTrain,
    BreathVeto,
    breath_extent_fallback,
    breath_pattern_parameters,
    breath_veto_of,
    in_review_band,
    measure_breath_pattern,
    measure_breath_train,
    measure_modulation,
    modulation_parameters,
    review_parameters,
    speech_runs,
    train_extent,
    train_parameters,
    voice_segments,
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


def test_syllabic_bursts_put_the_modulation_in_the_syllabic_band() -> None:
    """Bursts at a syllabic rate read a breathing-over-syllabic ratio well under breathing's."""
    syllabic = _modulation(_spectrogram(30.0, [(2 + k * 0.25, 0.12) for k in range(100)]))
    breathing = _modulation(_spectrogram(30.0, _cycles(2.0, 5)))
    assert syllabic is not None and breathing is not None
    assert syllabic.breath_vs_syllabic_db < breathing.breath_vs_syllabic_db - 10


def test_silence_has_no_active_span() -> None:
    """A floor with no burst has no active span."""
    reading = _modulation(_spectrogram(30.0, []))
    assert reading is not None
    assert reading.active_span_s == 0


def test_a_recording_shorter_than_the_reading_needs_has_none() -> None:
    """Under min_duration_s the modulation spectrum is not read."""
    assert _modulation(_spectrogram(2.0, [(0.5, 1.0)])) is None


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


def _cycles(start: float, n: int) -> list[tuple[float, float]]:
    return [(start + k * 5.5 + offset, length) for k in range(n) for offset, length in ((0.5, 1.0), (2.0, 1.4))]


def _train(power: np.ndarray, words: tuple[tuple[float, float], ...] = (), family: str | None = None) -> BreathTrain:
    return measure_breath_train(
        power, hop_s=HOP_S, bin_hz=BIN_HZ, voiced=np.zeros(power.shape[1], dtype=bool), words=words, family=family
    )


QUICK = "respiration-and-cough-threequickbreaths"


def test_quick_breaths_are_counted_phase_by_phase() -> None:
    """3d889bf8: three quick breaths are six short bursts in four seconds, so three breaths."""
    train = _train(_spectrogram(6.0, [(0.3 + 0.6 * k, 0.35) for k in range(6)]), family=QUICK)
    assert (train.phases, train.breaths) == (6, 3)


def test_five_breath_cycles_count_ten_phases() -> None:
    """An inhale and an exhale per breath: ten phases are five breaths."""
    train = _train(_spectrogram(30.0, _cycles(2.0, 5)))
    assert (train.phases, train.breaths) == (10, 5)
    assert train.rate_cpm is not None and abs(train.rate_cpm - 60 / 5.5) < 1.0


def test_an_odd_phase_count_rounds_up_to_the_unpaired_breath() -> None:
    """Nine phases are an unpaired burst beside four full breaths: five breaths."""
    train = _train(_spectrogram(30.0, _cycles(2.0, 5)[:9]))
    assert (train.phases, train.breaths) == (9, 5)


def test_one_long_breath_stays_a_single_breath() -> None:
    """7c169ccc: one 1.6 s breath where three quick ones were asked is one phase, one breath."""
    train = _train(_spectrogram(8.5, [(3.0, 1.6)]), family=QUICK)
    assert (train.phases, train.breaths) == (1, 1)


def test_silence_has_no_breath_train() -> None:
    """The noise floor alone holds no burst, so no train and no extent."""
    train = _train(_spectrogram(30.0, []))
    assert train.phases == 0 and train.extent_s is None


def test_breathing_in_the_first_half_bounds_the_extent_there() -> None:
    """6f73b8d2: the breathing sits in the first half; the extent ends at the last breath, padded."""
    train = _train(_spectrogram(60.0, _cycles(1.0, 5)))
    assert train.extent_s is not None
    last = _cycles(1.0, 5)[-1]
    assert train.extent_s[0] < 1.5
    assert last[0] + last[1] <= train.extent_s[1] < last[0] + last[1] + 1.0


def test_every_counted_peak_and_overlapping_event_lies_inside_the_extent() -> None:
    """The invariant the owner's figures were checked on: no counted burst or train event falls outside."""
    power = _spectrogram(30.0, _cycles(2.0, 5))
    train = _train(power)
    events = measure_breath_pattern(
        power,
        hop_s=HOP_S,
        bin_hz=BIN_HZ,
        voiced=np.zeros(power.shape[1], dtype=bool),
        parameters=breath_pattern_parameters(),
    ).event_spans_s
    extent = train_extent(train, events + ((train.bursts[-1].end_s, train.bursts[-1].end_s + 2.0),))
    assert extent.source == EXTENT_BREATH_TRAIN and extent.breaths == 5
    assert all(extent.start_s <= burst.start_s and burst.end_s <= extent.end_s for burst in train.bursts)
    assert all(extent.start_s <= e[0] and e[1] <= extent.end_s for e in events if e[1] > extent.start_s)
    assert extent.end_s == train.bursts[-1].end_s + 2.0


def _words(start: float, n: int) -> tuple[tuple[float, float], ...]:
    return tuple((start + 0.4 * k, start + 0.4 * k + 0.3) for k in range(n))


def test_speech_runs_need_enough_words() -> None:
    """A run of five words is speech; two stray words are not."""
    assert speech_runs(_words(10.0, 5) + _words(20.0, 2), words_min=5, gap_s=1.0) == [(10.0, 10.0 + 0.4 * 4 + 0.3)]


def test_speech_after_the_breaths_ends_the_extent() -> None:
    """6ca9935e: breathing then reading aloud; the extent stops at the speech, and bursts in it are left out."""
    words = _words(24.0, 10)
    power = _spectrogram(30.0, _cycles(1.0, 4) + [(25.0, 0.5), (27.0, 0.5)])
    train = _train(power, words=words)
    assert train.extent_s is not None and train.extent_s[1] <= 24.0
    assert train.phases == 8


def test_the_fallback_is_the_measures_events_then_airways_hull() -> None:
    """Without a breath train, the measure's events (padded) stand, then AIRWAY's own hull."""
    pad = train_parameters().pad_s
    events = breath_extent_fallback(((2.0, 3.0), (5.0, 6.0)), (0.0, 1.0), duration_s=10.0, pad_s=pad)
    assert events is not None and events.source == EXTENT_MEASURE_EVENTS
    assert events.bounds == (2.0 - pad, 6.0 + pad)
    airway = breath_extent_fallback((), (1.0, 4.0), duration_s=10.0, pad_s=pad)
    assert airway is not None and airway.source == EXTENT_AIRWAY_EVENTS and airway.bounds == (1.0, 4.0)
    assert breath_extent_fallback((), None, duration_s=10.0, pad_s=pad) is None


def test_the_review_band_holds_weak_irregular_and_short_trains() -> None:
    """fac74f45 and 81873ca0 the owner was fine to see flagged; a clean train stays out."""
    p = review_parameters()
    clean = BreathTrain(phases=10, breaths=5, cycle_cv=0.1, rise_db=20.0)
    assert not in_review_band(clean, p)
    assert in_review_band(BreathTrain(phases=10, breaths=5, cycle_cv=0.6, rise_db=p.irregular_rise_db_max - 1), p)
    assert in_review_band(BreathTrain(phases=10, breaths=5, cycle_cv=0.1, rise_db=p.weak_rise_db_max - 1), p)
    assert in_review_band(BreathTrain(phases=1, breaths=1, rise_db=20.0), p)
    assert in_review_band(None, p)


TRACK_HOP_S = 0.01


def _track(duration_s: float, runs: list[tuple[float, float, float, float]]) -> tuple[np.ndarray, ...]:
    """A phonation track voiced over each (start, length, f0, strength); unvoiced elsewhere."""
    times = np.arange(0.0, duration_s, TRACK_HOP_S)
    f0, strength = np.zeros_like(times), np.zeros_like(times)
    for start, length, hz, power in runs:
        held = (times >= start) & (times < start + length)
        f0[held], strength[held] = hz, power
    return times, f0, strength


def _segments(
    track: tuple[np.ndarray, ...], words: tuple[tuple[float, float], ...] = ()
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    return voice_segments(
        *track,
        words,
        voicing_strength_min=breath_pattern_parameters().voicing_strength_min,
        parameters=train_parameters(),
    )


def test_a_steady_vocalisation_is_sustained_and_not_speech() -> None:
    """988c1609: a held 0.5 s "aah" at steady pitch is a sustained vocalisation."""
    sustained, speech = _segments(_track(4.0, [(1.0, 0.5, 150.0, 0.9)]))
    assert speech == () and len(sustained) == 1
    assert abs(sustained[0][0] - 1.0) < 0.02 and abs(sustained[0][1] - 1.5) < 0.02


def test_voicing_flickers_on_breath_noise_are_neither_speech_nor_sustained() -> None:
    """78e40278: turbulence trips the pitch tracker in short runs; with no word they are not speech."""
    flickers = [(0.5 + 0.15 * k, 0.04, 140.0, 0.6) for k in range(12)]
    assert _segments(_track(4.0, flickers)) == ((), ())


def test_a_word_chains_the_voiced_runs_beside_it_into_one_speech_segment() -> None:
    """ba1d1459: two words then voiced syllables under half a second apart are one speech segment."""
    track = _track(20.0, [(16.4, 0.2, 180.0, 0.6), (17.5, 0.13, 160.0, 0.6), (18.15, 0.11, 125.0, 0.6)])
    sustained, speech = _segments(track, words=((16.36, 16.68), (16.64, 16.92), (18.08, 18.48)))
    assert sustained == ()
    assert len(speech) == 1 and abs(speech[0][0] - 16.36) < 0.02 and abs(speech[0][1] - 18.48) < 0.02


def _voiced_frames(power: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(power.shape[1], dtype=bool)
    for start, length in spans:
        mask[int(start / HOP_S) : int((start + length) / HOP_S)] = True
    return mask


def test_a_vocalised_exhale_and_the_inhale_beside_it_are_two_phases() -> None:
    """988c1609: an unvoiced inhale running straight into a held "aah" exhale counts as both phases."""
    inhales = [(1.0 + 3.0 * k, 0.6) for k in range(3)]
    exhales = [(1.6 + 3.0 * k, 0.6) for k in range(3)]
    power = _spectrogram(11.0, [(1.0 + 3.0 * k, 1.2) for k in range(3)])
    voiced = _voiced_frames(power, exhales)
    sustained = tuple((start, start + length) for start, length in exhales)
    plain = measure_breath_train(power, hop_s=HOP_S, bin_hz=BIN_HZ, voiced=voiced)
    split = measure_breath_train(power, hop_s=HOP_S, bin_hz=BIN_HZ, voiced=voiced, sustained=sustained)
    assert plain.phases == 3
    assert (split.phases, split.breaths) == (6, 3)
    assert sum(burst.voiced for burst in split.bursts) == 3
    assert all(a.end_s <= b.start_s for a, b in zip(split.bursts, split.bursts[1:]))
    assert all(inhale[0] - 0.2 <= burst.start_s for inhale, burst in zip(inhales, split.bursts[::2]))


def test_scattered_voicing_inside_one_long_breath_leaves_it_one_phase() -> None:
    """7c169ccc: voicing scattered through one long breath is no sustained run, so the breath stays whole."""
    power = _spectrogram(8.5, [(3.0, 1.6)])
    sustained, _ = _segments(_track(8.5, [(3.1 + 0.3 * k, 0.05, 140.0, 0.85) for k in range(5)]))
    train = measure_breath_train(
        power,
        hop_s=HOP_S,
        bin_hz=BIN_HZ,
        voiced=np.zeros(power.shape[1], dtype=bool),
        family=QUICK,
        sustained=sustained,
    )
    assert sustained == () and (train.phases, train.breaths) == (1, 1)


def test_speech_segments_hold_no_phase_and_end_the_extent() -> None:
    """ba1d1459: bursts under an acoustic speech segment are not phases, and the extent stops at it."""
    power = _spectrogram(30.0, _cycles(1.0, 4) + [(25.0, 0.5), (27.0, 0.5)])
    train = measure_breath_train(
        power, hop_s=HOP_S, bin_hz=BIN_HZ, voiced=np.zeros(power.shape[1], dtype=bool), speech=((24.0, 28.0),)
    )
    assert train.phases == 8
    assert train.extent_s is not None and train.extent_s[1] <= 24.0


def _shaped(power: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """Give each burst the same falling spectral tilt, so the bursts share a template a phase can match."""
    tilt = np.linspace(1.0, 0.1, power.shape[0])[:, None]
    for start, length in spans:
        power[:, int(start / HOP_S) : int((start + length) / HOP_S)] *= tilt
    return power


def test_a_recording_opening_mid_inhale_counts_that_phase_from_zero() -> None:
    """ecc63817: the file opens on an inhale already raised over the floor; it counts, and the extent starts at 0."""
    phases = [(0.0, 0.5)] + [(1.0 + 0.8 * k, 0.4) for k in range(5)]
    train = _train(_shaped(_spectrogram(6.5, phases), phases), family=QUICK)
    assert (train.phases, train.breaths) == (6, 3)
    assert train.bursts[0].start_s == 0.0 and train.extent_s is not None and train.extent_s[0] == 0.0


def test_a_quiet_opening_is_no_edge_phase() -> None:
    """42442f80: an opening under the floor (a filter edge) adds no phase."""
    phases = [(1.0 + 0.8 * k, 0.4) for k in range(5)]
    power = _shaped(_spectrogram(6.5, phases), phases)
    power[:, : int(0.4 / HOP_S)] *= 1e-3
    train = _train(power, family=QUICK)
    assert train.phases == 5
