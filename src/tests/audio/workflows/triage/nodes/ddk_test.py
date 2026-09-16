"""DDK: the train it proposes, the task it evaluates, and the corpus it evaluates nothing on.

Most tests still override every operating point DDK reads through the mechanism a campaign would
use, so the numbers are fixture values, not fits: what is pinned is that the body reads the key,
not where the key falls. The packaged configuration now ships a reasoned value for each of them, so
a body no longer refuses over the packaged file alone — that case is exercised separately, by
clearing a key with an explicit override.
"""

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.features_extraction.ppg import PHONEME_LABELS
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.branches import (
    UNDETERMINED,
    Result,
    branch_params,
)
from senselab.audio.workflows.triage.nodes.common import BranchResult, find_branch_report, live_entities
from senselab.audio.workflows.triage.nodes.ddk import (
    CYCLES_OR_SYLLABLES_PER_S,
    KIND,
    NO_INSTRUMENT,
    NO_PPG,
    NODE,
    PPG_EXPECTED_PLACE,
    PPG_PLACE_AGREEMENT,
    PPG_RATE,
    PPG_UNITS,
    Posteriorgram,
    PpgReading,
    RATE,
    SYLLABLES_PER_S,
    align_ddk,
    ddk,
    detect_ddk,
    dispersion,
    dispersion_by_position,
    ppg_reading,
    read_ddk,
    trend,
)
from senselab.audio.workflows.triage.vocabulary import (
    BRANCHES,
    BranchDecision,
    RunState,
    fold_file_verdict,
)
from senselab.utils.prov_store import Entity, ProvStore

SR = 16000
"""The working rate PREPROCESS resamples to, which the spectrogram's hop and bins are in."""

ENVELOPE_HZ = 1000.0
FLOOR_DBFS = -60.0
PEAK_DBFS = -35.0
SILENT_DBFS = -70.0

CYCLE = ("labial", "alveolar", "velar")
BANDS = {"labial": (0.0, 1500.0), "velar": (1500.0, 3500.0), "alveolar": (3500.0, 8000.0)}

FRAME_S = 0.01
"""The posteriorgram's frame period in these fixtures. ppgs' own is near this; nothing reads it."""

STOP_OF = {"labial": "p", "alveolar": "t", "velar": "k"}
"""One stop per place, to write a raster whose expected places are known by construction."""


def _raster(syllables: Sequence[tuple[str, float]]) -> np.ndarray:
    """A one-hot argmax raster over a phoneme sequence with per-phoneme durations.

    Args:
        syllables: ``(phoneme, seconds)`` in order. A phoneme repeated back to back would collapse
            into one run, so a caller that wants two runs of one phoneme separates them.

    Returns:
        The posteriorgram, ``(frame, phoneme)``, one-hot on the named phoneme.
    """
    indices: list[int] = []
    for label, seconds in syllables:
        indices.extend([PHONEME_LABELS.index(label)] * max(1, int(round(seconds / FRAME_S))))
    frames = np.zeros((len(indices), len(PHONEME_LABELS)), dtype=float)
    frames[np.arange(len(indices)), indices] = 1.0
    return frames


def _cv_raster(
    places: Sequence[str], intervals: Sequence[float], *, lead_s: float = 0.2, stop_s: float = 0.05
) -> np.ndarray:
    """A raster of consonant-vowel syllables whose onsets are separated by given intervals.

    Args:
        places: One place of articulation per syllable; its stop opens that syllable.
        intervals: The interval from each onset to the next; one shorter than ``places``.
        lead_s: Silence before the first onset, so the train does not start at frame zero.
        stop_s: How long each stop run is; the vowel fills the rest of its interval.

    Returns:
        The raster.
    """
    sequence: list[tuple[str, float]] = [("<silent>", lead_s)]
    for index, place in enumerate(places):
        gap = intervals[index] if index < len(intervals) else stop_s + 0.2
        sequence.append((STOP_OF[place], stop_s))
        sequence.append(("aa", max(FRAME_S, gap - stop_s)))
    return _raster(sequence)


@pytest.fixture
def ddk_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with every ``branch`` key DDK's two modes read supplied.

    The packaged file now ships a reasoned value for each of them, but these fixture values are what
    this suite measures against — chosen for the test signals, not fits on any corpus.

    ``smoothing_window_s`` is 0.011 rather than 0.010 for one reason, pinned by
    :class:`TestTheEventWalkIsSensitiveToTheSmoothingWindowsParity`: an even width in samples makes
    ``boxcar`` a half-sample shift, which ties every other maximum and loses it. 0.011 at the 1 kHz
    envelope of these fixtures is eleven samples, so the tests measure DDK and not that artefact.
    """
    override = tmp_path / "ddk.yaml"
    override.write_text(
        "branch:\n"
        "  train_min_s: 1.5\n"
        "  modulation_band_hz: [2.0, 12.0]\n"
        "  rate_prominence_min: 2.0\n"
        "  smoothing_window_s: 0.011\n"
        "  peak_prominence_db: 6.0\n"
        "  trough_return_db: 3.0\n"
        "  event_min_s: 0.02\n"
        "  repeat_min_occurrences: 3\n"
        "  burst_window_ms: 10.0\n"
        "  place_margin_db: 3.0\n"
        "  place_centroid_bands_hz:\n"
        "    labial: [0.0, 1500.0]\n"
        "    velar: [1500.0, 3500.0]\n"
        "    alveolar: [3500.0, 8000.0]\n"
    )
    return load_triage_config(override)


def _train_envelope(duration_s: float, carrier: tuple[float, float], rate_hz: float, phase_s: float) -> np.ndarray:
    """An envelope silent outside ``carrier`` and modulated at ``rate_hz`` inside it.

    Args:
        duration_s: The recording's duration.
        carrier: The extent the train occupies.
        rate_hz: The repetition rate.
        phase_s: Where inside the carrier the first maximum falls.

    Returns:
        The envelope in dBFS, one value per sample at :data:`ENVELOPE_HZ`.
    """
    samples = int(duration_s * ENVELOPE_HZ)
    envelope = np.full(samples, SILENT_DBFS)
    lo, hi = int(carrier[0] * ENVELOPE_HZ), int(carrier[1] * ENVELOPE_HZ)
    times = np.arange(lo, hi) / ENVELOPE_HZ
    shape = 0.5 + 0.5 * np.cos(2.0 * np.pi * rate_hz * (times - carrier[0] - phase_s))
    envelope[lo:hi] = FLOOR_DBFS + (PEAK_DBFS - FLOOR_DBFS) * shape
    return envelope


def _flat_envelope(duration_s: float, carriers: Sequence[tuple[float, float]]) -> np.ndarray:
    """An envelope raised over its floor inside each carrier and constant there.

    A constant slice has no modulation at all: its mean-removed spectrum is identically zero, so
    ``train_rate_hz`` reads no rate. That is the no-modulation case, stated without a random draw.

    Args:
        duration_s: The recording's duration.
        carriers: The extents to raise.

    Returns:
        The envelope in dBFS.
    """
    envelope = np.full(int(duration_s * ENVELOPE_HZ), SILENT_DBFS)
    for start, end in carriers:
        envelope[int(start * ENVELOPE_HZ) : int(end * ENVELOPE_HZ)] = PEAK_DBFS
    return envelope


def _wideband(duration_s: float, places: Sequence[tuple[tuple[float, float], str]]) -> np.ndarray:
    """A wideband power spectrogram whose energy sits in one place band per named slot.

    Args:
        duration_s: The recording's duration.
        places: ``((start, end), place)`` slots; frames inside a slot carry that place's band.

    Returns:
        Power, ``(bin, frame)``, at a 5 ms window and a 5 ms hop.
    """
    n_fft = int(SR * 0.005)
    hop_s = n_fft / SR
    frames = int(duration_s / hop_s)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    power = np.full((freqs.size, frames), 1e-6)
    for (start, end), place in places:
        lo, hi = BANDS[place]
        bins = (freqs >= lo) & (freqs < hi)
        first, last = int(start / hop_s), min(frames, int(np.ceil(end / hop_s)))
        if last <= first:
            continue
        power[np.ix_(bins, np.arange(first, last))] = 1.0
    return power


def _pataka_slots(carrier: tuple[float, float], rate_hz: float, collapse: str | None = None) -> list[Any]:
    """One place slot per repetition period, cycling through /p/, /t/, /k/ unless collapsed.

    Args:
        carrier: The train's extent.
        rate_hz: The repetition rate, whose period is one slot.
        collapse: A single place every slot carries instead, which is the sequential task produced
            as an alternating one.

    Returns:
        The slots, for :func:`_wideband`.
    """
    period = 1.0 / rate_hz
    slots: list[Any] = []
    index = 0
    start = carrier[0]
    while start < carrier[1]:
        slots.append(((start, min(start + period, carrier[1])), collapse or CYCLE[index % len(CYCLE)]))
        start += period
        index += 1
    return slots


@pytest.fixture
def seed_ddk_store(tmp_path: Path) -> Callable[..., dict[str, str]]:
    """Write the store surface DDK reads: the streams, the envelope, the spans, the words.

    Every argument defaults to writing nothing for that derivative, which is how a test sets up an
    absence — a different state from an empty reading.
    """

    def _seed(
        store: ProvStore,
        *,
        stem: str = "recording",
        duration_s: float = 6.0,
        envelope: np.ndarray | None = None,
        spans: Sequence[tuple[float, float]] = (),
        words: Sequence[tuple[str, tuple[float, float]]] = (),
        wideband: np.ndarray | None = None,
        posteriorgram: np.ndarray | None = None,
        seconds_per_frame: float = FRAME_S,
    ) -> dict[str, str]:
        """Seed one store; returns the ids it wrote, keyed by what they are."""
        (tmp_path / "derivatives").mkdir(exist_ok=True)
        activity = store.activity(node="PREPROCESS", step="seed-ddk", parameters={})
        agent = store.agent(agent_type="software", version="senselab test-seed")
        store.was_associated_with(activity, agent)
        ids: dict[str, str] = {}

        def _write(prov_type: str, extent: tuple[float, float] | None, attributes: dict[str, Any]) -> str:
            """One seeded entity with PREPROCESS's generating activity."""
            entity_id = store.entity(prov_type=prov_type, extent=extent, attributes=attributes)  # type: ignore[arg-type]
            store.was_generated_by(entity_id, activity)
            store.was_attributed_to(entity_id, agent)
            return entity_id

        for name in ("recording", "plain"):
            ids[name] = _write(
                "stream",
                (0.0, duration_s),
                {"name": name, "path": f"{stem}.wav", "sampling_rate": SR, "channels": 1},
            )
        if envelope is not None:
            np.savez(
                tmp_path / "derivatives" / "energy_envelope.npz",
                envelope_dbfs=envelope,
                floor_dbfs=np.full_like(envelope, FLOOR_DBFS),
            )
            ids["energy_envelope"] = _write(
                "measurement",
                None,
                {
                    "name": "energy_envelope",
                    "signal": "preemphasised",
                    "path": "derivatives/energy_envelope.npz",
                    "sampling_rate": ENVELOPE_HZ,
                },
            )
        if wideband is not None:
            np.savez(tmp_path / "derivatives" / "spectrogram_wideband.npz", spectrogram=wideband)
            ids["spectrogram_wideband"] = _write(
                "measurement",
                None,
                {
                    "name": "spectrogram_wideband",
                    "signal": "preemphasised",
                    "path": "derivatives/spectrogram_wideband.npz",
                    "n_fft": int(SR * 0.005),
                    "hop_length": int(SR * 0.005),
                    "win_length": int(SR * 0.005),
                },
            )
        if posteriorgram is not None:
            np.savez(
                tmp_path / "derivatives" / "ppg_posteriorgram.npz",
                posteriorgram=posteriorgram.astype(np.float16),
                phonemes=np.asarray(PHONEME_LABELS, dtype=np.str_),
                seconds_per_frame=np.float64(seconds_per_frame),
                duration_s=np.float64(posteriorgram.shape[0] * seconds_per_frame),
                sampling_rate=np.int64(SR),
            )
            ids["ppg_posteriorgram"] = _write(
                "measurement",
                (0.0, posteriorgram.shape[0] * seconds_per_frame),
                {
                    "name": "ppg_posteriorgram",
                    "signal": "enhanced",
                    "path": "derivatives/ppg_posteriorgram.npz",
                    "frames": int(posteriorgram.shape[0]),
                    "n_phonemes": int(posteriorgram.shape[1]),
                    "phonemes": list(PHONEME_LABELS),
                    "seconds_per_frame": seconds_per_frame,
                    "layout": "frames_by_phonemes",
                },
            )
        for index, (start, end) in enumerate(spans):
            ids[f"span-{index}"] = _write(
                "span",
                (start, end),
                {"signal": "preemphasised", "measure": "amplitude", "merged_proposals": 1, "k_db": 18.0},
            )
        if words:
            ids["consensus_transcript"] = _write(
                "measurement", None, {"name": "consensus_transcript", "signal": "consensus", "n_words": len(words)}
            )
            for index, (text, extent) in enumerate(words):
                ids[f"word-{index}"] = _write(
                    "word", extent, {"index": index, "text": text, "bracketed": text.startswith("[")}
                )
        return ids

    return _seed


def _spans_of(store: ProvStore, family: str = KIND) -> list[Entity]:
    """Every live span one family proposed, earliest first."""
    found = [span for span in live_entities(store, "span") if span.attributes.get("family") == family]
    return sorted(found, key=lambda span: span.extent or (0.0, 0.0))


def _measurements(store: ProvStore, name: str) -> list[Entity]:
    """Every live measurement of one name."""
    return [e for e in live_entities(store, "measurement") if e.attributes.get("name") == name]


def _report(store: ProvStore) -> Entity:
    """DDK's own branch report entity."""
    found = find_branch_report(store, NODE)
    assert found is not None
    return found


def _run(store: ProvStore, config: TriageConfig, tmp_path: Path, hint: AudioHints | None = None) -> BranchResult:
    """Call the node the way the runner does."""
    return ddk(store, "plain", config, hint, run_dir=tmp_path)


class TestARealTrainProposesOneSpanWithARate:
    """D1: the train is one span, its rate comes from the envelope's modulation spectrum."""

    def test_a_declared_alternating_train_proposes_exactly_one_span(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Twenty syllables over one carrier are one ``task_extent`` span, not twenty."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, ddk_config, tmp_path)
        spans = _spans_of(store)
        assert len(spans) == 1
        assert spans[0].attributes["role"] == "task_extent"
        assert spans[0].attributes["production"] == "syllable_train"
        assert spans[0].attributes["syllables_n"] > 1
        assert result.report.conformance is True
        assert result.report.kind == KIND

    def test_the_rate_is_the_modulation_peak_and_states_its_unit(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A 5 Hz train reads 5 Hz, in syllables per second because the family is alternating."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        [rate] = _measurements(store, RATE)
        assert rate.attributes["value"] == pytest.approx(5.0, abs=0.3)
        assert rate.attributes["unit"] == SYLLABLES_PER_S
        assert rate.attributes["support_syllables"] > 1
        assert rate.attributes["uncalibrated"] is True

    def test_the_train_carries_its_onsets_intervals_and_dispersion(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """D3 carries regularity: the onset series, its intervals and their dispersion and trend."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        [counts] = _measurements(store, "counts")
        entries = counts.attributes["entries"]
        assert len(entries["syllable_onset_s"]["found"]) == entries["expected_event_count"]["found"]
        assert entries["expected_event_count"]["declared"] == 10
        assert entries["expected_event_count"]["found"] == 20
        intervals = entries["inter_onset_interval_s"]["found"]
        assert intervals == pytest.approx([0.2] * len(intervals), abs=0.01)
        [spread] = _measurements(store, "interval_dispersion")
        assert spread.attributes["value"] == pytest.approx(0.0, abs=0.05)
        assert spread.attributes["support_intervals"] == len(intervals)
        assert spread.attributes["trend_s_per_step"] == pytest.approx(0.0, abs=0.001)

    def test_a_declared_sequence_reads_its_cycle_and_proposes_one_span(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """D6 over ``pataka``: the places come from the burst spectrum, and the cycle is realised."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
            wideband=_wideband(6.0, _pataka_slots((1.0, 5.0), 5.0)),
        )
        result = _run(store, ddk_config, tmp_path)
        [span] = _spans_of(store)
        assert span.attributes["production"] == "syllable_sequence"
        assert span.attributes["realised_cycles"] >= 1
        [rate] = _measurements(store, RATE)
        assert rate.attributes["unit"] == CYCLES_OR_SYLLABLES_PER_S
        assert result.report.conformance is True

    def test_a_collapsed_sequence_deviates_rather_than_failing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """``/pa-pa-pa/`` is the finding, not a fault: a mismatch per syllable and no realised cycle."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
            wideband=_wideband(6.0, _pataka_slots((1.0, 5.0), 5.0, collapse="labial")),
        )
        result = _run(store, ddk_config, tmp_path)
        assertions = [e for e in live_entities(store, "assertion") if e.attributes.get("deviation_type") is not None]
        kinds = {str(e.attributes["deviation_type"]) for e in assertions}
        assert "syllable_sequence_mismatch" in kinds
        assert "stimulus_mismatch" not in kinds
        [collapse] = _measurements(store, "sequence_collapse_fraction")
        assert collapse.attributes["value"] == pytest.approx(1.0)
        assert collapse.attributes["dominant_place"] == "labial"
        assert _spans_of(store)[0].attributes["realised_cycles"] == 0
        assert result.report.conformance is False


class TestTheEventWalkNoLongerDependsOnTheSmoothingWindowsParity:
    """A property of the ported instrument, measured here because DDK's count depends on it.

    ``events_in_extent`` requires a maximum to stand over a trough on *both* sides. ``boxcar`` is
    ``np.convolve(..., mode="same")``, which for an **even** width centres the window between two
    samples, so a smooth maximum comes back as two equal samples: the strict-maximum test picks the
    second, the left walk stops immediately on its equal neighbour, and the prominence computes as
    0 dB. Half or more of an exactly periodic train is then discarded, and the syllable count — this
    branch's own measurement — is wrong by that much.

    The foundation records the same arithmetic for a clipped, flat-topped cough. What is measured
    here is that it also fires on an ordinary sampled sinusoid, which is what a DDK train is, and
    that the trigger is the window's parity in samples rather than anything about the production.
    Fixing it is a measurement (smooth first, or accept a one-sided trough) and belongs in
    ``branches.py``, so this test pins the behaviour rather than working around it.
    """

    @pytest.mark.parametrize(
        ("window_s", "complete"),
        [(0.009, True), (0.010, True), (0.011, True), (0.012, True), (0.015, True)],
    )
    def test_the_same_twenty_syllable_train_counts_the_same_at_every_width(
        self,
        store: ProvStore,
        tmp_path: Path,
        seed_ddk_store: Callable[..., Any],
        window_s: float,
        complete: bool,
    ) -> None:
        """One signal, five widths, twenty syllables each — the defect this class found is fixed.

        It was found here and fixed in ``branches.boxcar`` on 2026-09-16: an even width made
        ``np.convolve(..., mode="same")`` centre between samples, so every smooth maximum returned
        as two equal values and the strict-maximum test found neither. The packaged
        ``smoothing_window_s`` of 0.01 s is 160 samples at 16 kHz -- even -- and lost 12 of these
        20. ``boxcar`` now raises an even width to the next odd one, so parity decides nothing.
        This class is kept as the regression guard, since the failure was silent and directional:
        a halved onset series doubles the intervals it does find, inflating dispersion at the
        highest rates.
        """
        override = tmp_path / "parity.yaml"
        override.write_text(
            "branch:\n"
            "  train_min_s: 1.5\n"
            "  modulation_band_hz: [2.0, 12.0]\n"
            "  rate_prominence_min: 2.0\n"
            f"  smoothing_window_s: {window_s}\n"
            "  peak_prominence_db: 6.0\n"
            "  trough_return_db: 3.0\n"
            "  event_min_s: 0.02\n"
            "  repeat_min_occurrences: 3\n"
        )
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, load_triage_config(override), tmp_path)
        [counts] = _measurements(store, "counts")
        onsets = counts.attributes["entries"]["expected_event_count"]["found"]
        assert onsets == 20 if complete else onsets < 20


class TestButtercupIsServedByTheConsensusWords:
    """The one lexical DDK family: the recognisers produce the count directly."""

    def test_the_repeated_word_becomes_one_span_over_its_own_hull(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Ten ``buttercup`` tokens are one span, derived from the transcript and every word."""
        words = [("buttercup", (1.0 + 0.4 * index, 1.3 + 0.4 * index)) for index in range(10)]
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            envelope=_train_envelope(6.0, (1.0, 5.0), 2.5, 0.2),
            spans=[(1.0, 5.0)],
            words=words,
        )
        result = _run(store, ddk_config, tmp_path)
        [span] = _spans_of(store)
        assert span.attributes["production"] == "lexical_repetition"
        assert span.attributes["token"] == "buttercup"
        assert span.attributes["repeats_n"] == 10
        assert span.extent == pytest.approx((1.0, 4.9))
        [counts] = _measurements(store, "counts")
        assert counts.attributes["entries"]["expected_event_count"] == {"found": 10, "declared": 10}
        assert result.report.conformance is True

    def test_the_word_nobody_said_is_a_fail_and_not_a_span(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """No token matched is an absence of detected content, and mints nothing."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            envelope=_train_envelope(6.0, (1.0, 5.0), 2.5, 0.2),
            spans=[(1.0, 5.0)],
            words=[("hello", (1.0, 1.4)), ("there", (1.5, 1.9))],
        )
        result = _run(store, ddk_config, tmp_path)
        assert _spans_of(store) == []
        assert result.report.conformance is False


class TestConnectedSpeechRoutedByTheLexicalGateFindsNoTrain:
    """Most of DDK's corpus. The branch says what it found; it does not evaluate someone's task."""

    def test_repeated_function_words_do_not_become_a_train(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The gate that routed this fires on ``the`` said eight times. No train is invented."""
        words = [("the", (1.0 + 0.5 * index, 1.2 + 0.5 * index)) for index in range(8)]
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-rainbow-passage",
            envelope=_flat_envelope(6.0, [(1.0 + 0.5 * index, 1.2 + 0.5 * index) for index in range(8)]),
            spans=[(1.0 + 0.5 * index, 1.2 + 0.5 * index) for index in range(8)],
            words=words,
        )
        result = _run(store, ddk_config, tmp_path)
        roles = [str(span.attributes["role"]) for span in _spans_of(store)]
        assert "repetition" not in roles
        assert "task_extent" not in roles
        assert roles == ["lexical_repetition"]
        assert _report(store).attributes["trains_n"] == 0
        assert result.report.conformance == UNDETERMINED

    def test_a_long_carrier_with_no_modulation_proposes_no_train(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A carrier clearing the train minimum is still not a train without a modulation peak."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-rainbow-passage",
            envelope=_flat_envelope(6.0, [(1.0, 5.0)]),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, ddk_config, tmp_path)
        assert _spans_of(store) == []
        assert result.report.conformance == UNDETERMINED

    def test_an_acoustic_train_out_of_family_is_annotated_and_evaluates_no_task(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Rapid repetition inside a Harvard sentence is reported as what it is, not as a DDK task."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-harvard-sentences-list",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, ddk_config, tmp_path)
        [span] = _spans_of(store)
        assert span.attributes["role"] == "repetition"
        assert span.attributes["evaluates_no_task"] is True
        assert span.attributes["production"] == "acoustic_repetition"
        [rate] = _measurements(store, RATE)
        assert rate.attributes["reading"] == "acoustic_repetition_not_a_declared_ddk_task"
        assert _report(store).attributes["mode"] == "detect"
        assert result.report.conformance == UNDETERMINED


class TestTheDispatchGoesBothWays:
    """The declared family picks the mode and never supplies the answer."""

    def test_a_ddk_family_takes_align(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A declared ``diadochokinesis-ka`` is in family, so the task is evaluated."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-ka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        report = _report(store)
        assert report.attributes["mode"] == "align"
        assert report.attributes["task_family"] == "diadochokinesis-ka"
        assert _spans_of(store)[0].attributes["role"] == "task_extent"

    def test_another_branchs_family_takes_detect(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """``prolonged-vowel`` is VOICE's kind, so DDK annotates and concludes nothing about it."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-prolonged-vowel",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        report = _report(store)
        assert report.attributes["mode"] == "detect"
        assert report.attributes["task_family"] == "prolonged-vowel"
        assert _spans_of(store)[0].attributes["role"] == "repetition"

    def test_no_derivable_task_family_takes_detect(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A path that is not a BIDS stem names no family, and the safe arm evaluates no task."""
        seed_ddk_store(store, stem="whatever", envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1), spans=[(1.0, 5.0)])
        _run(store, ddk_config, tmp_path)
        report = _report(store)
        assert report.attributes["mode"] == "detect"
        assert report.attributes["task_family"] is None

    def test_the_hint_task_token_selects_align_over_the_path(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The clean carrier wins: a hint naming a DDK task puts an unnamed path into align.

        Asserted on the span role, which only align mints, and not only on the recorded ``mode``.
        The report's ``mode`` comes from ``mode_of`` while the branch that ran comes from
        ``dispatch``; both derive it from ``declared_task_family(store, hint)``, so they agree only
        while both are handed the same hint. Dropping the hint from the ``dispatch`` call alone
        leaves the recorded mode saying ``align`` over a result ``detect_ddk`` produced, and a test
        reading the field would not notice.
        """
        seed_ddk_store(store, stem="whatever", envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1), spans=[(1.0, 5.0)])
        hint = AudioHints(metadata={"task_token": "diadochokinesis-ta"})
        _run(store, ddk_config, tmp_path, hint)
        [span] = _spans_of(store)
        assert span.attributes["role"] == "task_extent"
        assert span.attributes["production"] == "syllable_train"
        assert _report(store).attributes["mode"] == "align"
        assert _report(store).attributes["task_family"] == "diadochokinesis-ta"

    def test_detect_never_concludes_about_a_task(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """``detect_ddk``'s only answer is UNDETERMINED, whatever it found."""
        ids = seed_ddk_store(
            store, stem="whatever", envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1), spans=[(1.0, 5.0)]
        )
        assert ids
        result = detect_ddk(store, branch_params(ddk_config), reads=read_ddk(store, tmp_path, "plain"))
        assert result.done == UNDETERMINED
        assert isinstance(result, Result)


class TestAnAbsentInstrumentIsNotANegativeReading:
    """UNDETERMINED where a rate cannot be read, and a flag rather than a fail."""

    def test_an_absent_envelope_makes_align_undetermined(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Neither rate instrument is present, so each says so and no task was evaluated."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa", spans=[(1.0, 5.0)])
        result = align_ddk(
            "diadochokinesis-pa", store, None, branch_params(ddk_config), reads=read_ddk(store, tmp_path, "plain")
        )
        assert result.done == UNDETERMINED
        assert result.components == []
        assert [(f.name, f.evidence.get("unavailable")) for f in result.deviations if f.kind == "measure"] == [
            (RATE, "energy_envelope"),
            (PPG_RATE, "ppg_posteriorgram"),
        ]

    def test_an_absent_envelope_notes_rather_than_failing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A missing derivative must not be reported as the speaker having produced no train."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa", spans=[(1.0, 5.0)])
        result = _run(store, ddk_config, tmp_path)
        assert result.report.conformance == UNDETERMINED
        assert NO_INSTRUMENT in _report(store).attributes["notes"]

    def test_an_absent_spectrogram_leaves_every_place_unresolved(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """No burst spectrum is no substitution: the sequence is unread, not collapsed."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        assert _measurements(store, "sequence_collapse_fraction") == []
        [place] = _measurements(store, "syllable_place")
        assert place.attributes["unavailable"] == "spectrogram_wideband"
        assert _spans_of(store)[0].attributes["resolved_n"] == 0
        deviations = {str(e.attributes.get("deviation_type")) for e in live_entities(store, "assertion")}
        assert "syllable_sequence_mismatch" not in deviations


class TestEveryOperatingPointComesFromTheConfig:
    """No number in the body: a body reads every threshold from ``branch.*``, never a literal.

    The packaged file now ships a reasoned value for every key DDK reads, so a body never refuses
    on the packaged config alone. What is still pinned is that clearing a key by override does not
    raise: the qualifier it would have gated is skipped, the key is named in the report's
    ``unmeasured``, and the dependent conformance is what the body's own control flow produces —
    for DDK that is ``False`` when the carrier search itself is gated (``train_min_s``), because an
    ungated carrier search finds none, and :data:`UNDETERMINED` when only the lexical loop is gated
    (``repeat_min_occurrences``), because ``detect_ddk`` never answers anything else.
    """

    def test_an_unmeasured_train_minimum_is_recorded_rather_than_raised(
        self, store: ProvStore, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """``train_min_s`` cleared by override: no carrier ever qualifies, and the ask is named."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        override = tmp_path / "null_train_min.yaml"
        override.write_text("branch:\n  train_min_s: null\n")
        result = _run(store, load_triage_config(override), tmp_path)
        # UNDETERMINED, not False: an unmeasured qualifier could neither admit nor reject, so
        # claiming the instruction was not met would rest on a number nobody chose.
        assert result.report.conformance == UNDETERMINED
        assert "train_min_s" in result.report.unmeasured
        assert _spans_of(store) == []

    def test_an_unmeasured_repetition_minimum_is_recorded_rather_than_raised(
        self, store: ProvStore, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The lexical loop's own key, cleared the same way: no repetition is minted."""
        seed_ddk_store(store, stem="whatever", words=[("the", (1.0, 1.2)), ("the", (1.5, 1.7))])
        override = tmp_path / "null_repeat_min.yaml"
        override.write_text("branch:\n  repeat_min_occurrences: null\n")
        result = _run(store, load_triage_config(override), tmp_path)
        assert result.report.conformance == UNDETERMINED
        assert "repeat_min_occurrences" in result.report.unmeasured
        assert _spans_of(store) == []


class TestTheRegularityStatisticsAreParameterFree:
    """Dispersion and trend need no threshold, and say nothing where they are undefined."""

    def test_dispersion_is_zero_for_a_perfectly_even_train(self) -> None:
        """Equal intervals have no dispersion."""
        assert dispersion([0.2, 0.2, 0.2, 0.2]) == pytest.approx(0.0)

    def test_dispersion_is_none_below_its_own_domain(self) -> None:
        """A sample deviation needs two values; one interval gets no number invented for it."""
        assert dispersion([0.2]) is None
        assert dispersion([]) is None

    def test_the_trend_signs_festination_and_slowing(self) -> None:
        """Shortening intervals are negative, lengthening positive; neither is named a deficit."""
        shortening = trend([0.3, 0.25, 0.2, 0.15])
        lengthening = trend([0.15, 0.2, 0.25, 0.3])
        assert shortening is not None and shortening < 0.0
        assert lengthening is not None and lengthening > 0.0
        assert trend([0.2]) is None

    def test_a_sequential_train_reports_dispersion_within_syllable_position(self) -> None:
        """A pooled dispersion over /pa-ta-ka/ has a floor set by syllable identity, not control."""
        intervals = [0.10, 0.15, 0.20] * 4
        pooled = dispersion(intervals)
        assert pooled is not None and pooled > 0.2
        by_position = dispersion_by_position(intervals, 3)
        assert set(by_position) == {"0", "1", "2"}
        assert all(value == pytest.approx(0.0) for value in by_position.values())


class TestTheNodeRunsRatherThanBeingRecordedNoNode:
    """The standing argument for building this branch was the flag a missing node raises."""

    def test_the_runner_dispatches_ddk(self) -> None:
        """``run.py``'s branch table now names all four branches the vocabulary declares."""
        import inspect

        from senselab.audio.workflows.triage import run as run_module

        source = inspect.getsource(run_module._drive_branches)
        assert '"DDK": lambda: ddk(' in source
        assert set(BRANCHES) == {"AIRWAY", "SPEECH", "VOICE", "DDK"}

    def test_a_ddk_report_stops_the_asked_to_run_and_never_ran_flag(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The node writes a report carrying a kind, which is what the fold joins the decision to."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, ddk_config, tmp_path)
        folded = fold_file_verdict(
            [],
            branch_reports=(result.report,),
            spans_by_node={"DDK": len(_spans_of(store))},
            branch_decisions={
                "DDK": BranchDecision(branch="DDK", will_run=True, route_state="routed", forced_by_declaration=False)
            },
            ran={"DDK": RunState.COMPLETED},
            hint_claims={},
            route_state="routed",
        )
        assert not any("never ran" in reason.why for reason in folded.reasons)
        assert folded.findings["DDK"] == "present"
        assert folded.agreement["DDK"] == "agree"

    def test_without_a_report_the_flag_still_fires(self) -> None:
        """The flag is not wrong and is not edited: a routed branch that reported nothing flags."""
        folded = fold_file_verdict(
            [],
            branch_decisions={
                "DDK": BranchDecision(branch="DDK", will_run=True, route_state="routed", forced_by_declaration=False)
            },
            ran={"DDK": RunState.SKIPPED},
            hint_claims={},
            route_state="routed",
        )
        assert any(reason.node == "DDK" and "never ran" in reason.why for reason in folded.reasons)


def _reading(frames: np.ndarray, config: TriageConfig) -> PpgReading:
    """The CV walk over one raster, at the packaged segmentation points."""
    ppg = Posteriorgram(frames=frames, phonemes=PHONEME_LABELS, seconds_per_frame=FRAME_S)
    found = ppg_reading(ppg, branch_params(config))
    assert found is not None
    return found


class TestThePosteriorgramCvWalk:
    """The instrument itself: runs, units, trains. Task-agnostic — nothing here reads a family."""

    def test_a_clean_four_hertz_train_reads_its_period_and_rate(self, ddk_config: TriageConfig) -> None:
        """Eight onsets a quarter second apart are one train at 4 Hz with no jitter."""
        reading = _reading(_cv_raster(["labial"] * 8, [0.25] * 7), ddk_config)
        assert len(reading.units) == 8
        assert reading.train is not None
        assert reading.train.repetitions == 8
        assert reading.train.period_s == pytest.approx(0.25, abs=0.005)
        assert reading.train.rate_hz == pytest.approx(4.0, abs=0.05)
        assert reading.train.jitter == pytest.approx(0.0, abs=0.01)

    def test_connected_speech_like_onsets_are_not_one_train(self, ddk_config: TriageConfig) -> None:
        """Stops occur in ordinary speech; what does not occur is their intervals staying regular."""
        reading = _reading(
            _cv_raster(
                ["labial", "alveolar", "labial", "velar", "labial", "alveolar", "labial"],
                [0.2, 0.9, 0.25, 1.5, 0.2, 0.35],
            ),
            ddk_config,
        )
        assert len(reading.units) == 7
        assert reading.train is None

    def test_a_stop_free_raster_yields_no_cv_units(self, ddk_config: TriageConfig) -> None:
        """Sustained phonation, a glide, a breath: no stop, so no CV unit, structurally."""
        reading = _reading(_raster([("<silent>", 0.2), ("aa", 2.0), ("s", 0.3), ("aa", 1.0)]), ddk_config)
        assert reading.units == ()
        assert reading.train is None

    def test_a_slow_train_is_reported_with_its_rate_and_nothing_flags_it(self, ddk_config: TriageConfig) -> None:
        """0.9 Hz is implausibly slow for DDK and the branch says so by measuring it, not by judging."""
        reading = _reading(_cv_raster(["labial"] * 5, [1.11] * 4), ddk_config)
        assert reading.train is not None
        assert reading.train.rate_hz == pytest.approx(0.9, abs=0.02)
        assert reading.train.repetitions == 5

    def test_the_scan_stops_at_the_next_consonant_rather_than_pairing_across_it(self, ddk_config: TriageConfig) -> None:
        """``/p/ /t/ /aa/`` is one syllable, not two: the /p/ has no nucleus of its own."""
        reading = _reading(_raster([("<silent>", 0.2), ("p", 0.05), ("t", 0.05), ("aa", 0.2)]), ddk_config)
        assert [(unit.consonant, unit.vowel) for unit in reading.units] == [("t", "aa")]

    def test_vowel_variation_does_not_cost_a_unit(self, ddk_config: TriageConfig) -> None:
        """/pa/, /paw/ and /puh/ are the same syllable for this instrument; only the stop is fixed."""
        raster = _raster(
            [
                ("<silent>", 0.2),
                ("p", 0.05),
                ("aa", 0.2),
                ("p", 0.05),
                ("ao", 0.2),
                ("p", 0.05),
                ("uh", 0.2),
                ("p", 0.05),
                ("uw", 0.2),
            ]
        )
        reading = _reading(raster, ddk_config)
        assert [unit.vowel for unit in reading.units] == ["aa", "ao", "uh", "uw"]
        assert reading.train is not None
        assert reading.train.repetitions == 4


class TestThePosteriorgramAnswersTheDeclaredTask:
    """The instrument is task-agnostic; the comparison is against the declared expectation."""

    def test_a_declared_sequence_scores_full_per_position_accuracy(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """/pa-ta-ka/ produced as asked reads 1.0 overall and 1.0 at each of the three positions."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(list(CYCLE) * 4, [0.2] * 11),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [place] = _measurements(store, PPG_EXPECTED_PLACE)
        assert place.attributes["value"] == pytest.approx(1.0)
        assert place.attributes["by_position"] == {"0": 1.0, "1": 1.0, "2": 1.0}
        assert place.attributes["expected_sequence"] == list(CYCLE)

    def test_a_collapsed_sequence_scores_low_without_failing_conformance(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """/pa-pa-pa/ for /pa-ta-ka/ is the clinically meaningful finding, not a task not performed."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial"] * 12, [0.2] * 11),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [place] = _measurements(store, PPG_EXPECTED_PLACE)
        assert place.attributes["value"] == pytest.approx(1.0 / 3.0, abs=0.01)
        assert place.attributes["by_position"] == {"0": 1.0, "1": 0.0, "2": 0.0}
        assert result.report.conformance is True

    def test_a_single_syllable_family_reads_the_place_its_own_instruction_names(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """/ta/ expects alveolar; a train of /pa/ under that instruction scores zero, not one."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-ta", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-ta"}))
        [place] = _measurements(store, PPG_EXPECTED_PLACE)
        assert place.attributes["expected_sequence"] == ["alveolar"]
        assert place.attributes["value"] == pytest.approx(0.0)

    def test_the_train_span_names_the_posteriorgram_as_its_evidence(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Under propose-only the derivation is the whole record of where the extent came from."""
        ids = seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [span] = [proposed for proposed in _spans_of(store) if proposed.attributes["role"] == "ppg_train"]
        assert span.attributes["production"] == "syllable_train_from_ppg"
        assert span.attributes["repetitions"] == 8
        assert store.derived_from(span.id) == [ids["ppg_posteriorgram"]]

    def test_the_two_place_instruments_are_compared_when_the_spectrogram_is_present(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The agreement is the only thing that can settle which place reading to trust."""
        onsets = [0.2 + 0.2 * index for index in range(12)]
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(list(CYCLE) * 4, [0.2] * 11),
            wideband=_wideband(4.0, [((onset, onset + 0.05), CYCLE[index % 3]) for index, onset in enumerate(onsets)]),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [agreement] = _measurements(store, PPG_PLACE_AGREEMENT)
        assert agreement.attributes["support_onsets"] == 12
        assert agreement.attributes["value"] == pytest.approx(1.0)
        assert "not the place decision" in agreement.attributes["reading"]

    def test_a_declared_task_with_no_train_is_a_non_conformance_not_an_open_question(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The instruction asked for a train and the content holds none; that is an answer."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_raster([("<silent>", 0.5), ("aa", 3.0)]),
        )
        result = align_ddk(
            "diadochokinesis-pa", store, None, branch_params(ddk_config), reads=read_ddk(store, tmp_path, "plain")
        )
        assert result.done is False
        assert result.components == []
        assert [f.evidence["found"] for f in result.deviations if f.name == PPG_UNITS] == [0]

    def test_a_readable_posteriorgram_answers_a_task_the_envelope_could_not(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """No envelope is not no train. The same store without the raster reads UNDETERMINED."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        result = align_ddk(
            "diadochokinesis-pa", store, None, branch_params(ddk_config), reads=read_ddk(store, tmp_path, "plain")
        )
        assert result.done is True
        [rate] = [f for f in result.deviations if f.name == PPG_RATE and f.kind == "measure"]
        assert rate.evidence["value"] == pytest.approx(4.0, abs=0.05)
        assert rate.evidence["unit"] == SYLLABLES_PER_S
        assert rate.evidence["repetitions"] == 8

    def test_an_absent_posteriorgram_is_noted_rather_than_raised(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """An absent instrument is an absence; a branch never refuses over one."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        assert NO_PPG in _report(store).attributes["notes"]
        [absent] = _measurements(store, PPG_RATE)
        assert absent.attributes["unavailable"] == "ppg_posteriorgram"
        assert absent.attributes["value"] is None
