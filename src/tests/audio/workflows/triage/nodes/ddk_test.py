"""The syllable-train instrument: the train it proposes and the ten SPEECH tasks it evaluates.

Not a branch. Every test here drives the product path SPEECH takes — ``dispatch`` over
``align_speech``, which serves the ten ``diadochokinesis-*`` rows through ``align_ddk``.

Most tests still override every operating point the body reads through the mechanism a campaign
would use, so the numbers are fixture values, not fits: what is pinned is that the body reads the
key, not where the key falls. The packaged configuration now ships a reasoned value for each of
them, so a body no longer refuses over the packaged file alone — that case is exercised separately,
by clearing a key with an explicit override.
"""

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.features_extraction.ppg import PHONEME_LABELS
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.branches import (
    BUTTERCUP,
    LABIAL_LOW,
    SPEECH_EXPECTATIONS,
    UNDETERMINED,
    UNMEASURED_POINTS,
    BranchParams,
    Result,
    Syllable,
    branch_params,
    dispatch,
    mode_of,
    propose_spans,
    write_findings,
)
from senselab.audio.workflows.triage.nodes.common import live_entities, software_agent
from senselab.audio.workflows.triage.nodes.ddk import (
    CYCLES_OR_SYLLABLES_PER_S,
    NO_ENVELOPE,
    NO_PPG,
    PPG_CYCLE_NUCLEUS,
    PPG_CYCLE_RATE,
    PPG_PLACE_AGREEMENT,
    PPG_RATE,
    PPG_UNITS,
    RATE,
    SYLLABLES_PER_S,
    Posteriorgram,
    PpgReading,
    align_ddk,
    cycle_scan,
    dispersion,
    dispersion_by_position,
    ppg_reading,
    read_ddk,
    syllable_detail,
    trend,
)
from senselab.audio.workflows.triage.nodes.speech import align_speech, detect_speech
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


BUTTERCUP_NUCLEI = ("ah", "er", "ah")
"""/bʌ-tər-kʌp/: the nucleus of each of its three syllables, in the posteriorgram's ARPAbet."""


def _buttercup_raster(
    repeats: int, *, nuclei: Sequence[str] = BUTTERCUP_NUCLEI, coda: bool = True, lead_s: float = 0.2
) -> np.ndarray:
    """A raster of ``buttercup`` repetitions: /b/+/t/+/k/ onsets, and the coda /p/ of "cup".

    Args:
        repeats: How many times the word is said.
        nuclei: The three nuclei, so a test can substitute one without touching the onsets.
        coda: Whether the final /p/ of "cup" is realised, which opens no syllable of its own.
        lead_s: Silence before the first onset.

    Returns:
        The raster.
    """
    sequence: list[tuple[str, float]] = [("<silent>", lead_s)]
    for _ in range(repeats):
        for stop, nucleus in zip(("b", "t", "k"), nuclei):
            sequence.extend([(stop, 0.05), (nucleus, 0.15)])
        if coda:
            sequence.append(("p", 0.05))
        sequence.append(("<silent>", 0.05))
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


def _spans_of(store: ProvStore, family: str = "speech") -> list[Entity]:
    """Every live span one family proposed, earliest first."""
    found = [span for span in live_entities(store, "span") if span.attributes.get("family") == family]
    return sorted(found, key=lambda span: span.extent or (0.0, 0.0))


def _measurements(store: ProvStore, name: str) -> list[Entity]:
    """Every live measurement of one name."""
    return [e for e in live_entities(store, "measurement") if e.attributes.get("name") == name]


def _run(store: ProvStore, config: TriageConfig, tmp_path: Path, hint: AudioHints | None = None) -> Result:
    """Drive the syllable body the way SPEECH's node does, and write what it proposed.

    The four product calls SPEECH makes around the body: load the derivatives, dispatch on the
    declared family, write the proposals, and write the findings.
    """
    params = branch_params(config)
    reads = read_ddk(store, tmp_path, "plain")

    def _align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
        """Bind the loaded derivatives to the in-family mode."""
        return align_speech(task_family, store, hint, params, reads=reads)

    result = dispatch("SPEECH", store, params, hint, align=_align, detect=detect_speech)
    activity = store.activity(node="SPEECH", step="expect", parameters={})
    agent = software_agent(store)
    store.was_associated_with(activity, agent)
    propose_spans(store, activity, agent, result.components)
    write_findings(store, activity, agent, [*result.deviations, *params.record()], signal="plain")
    return result


def _detail(result: Result) -> dict[str, Any]:
    """The fields SPEECH's report carries for a syllable task."""
    return syllable_detail(result)


def _unmeasured(store: ProvStore) -> list[str]:
    """The operating points a body asked for and the configuration did not carry."""
    found = _measurements(store, UNMEASURED_POINTS)
    return [str(key) for entity in found for key in entity.attributes.get("value") or ()]


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
        assert result.done is True

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
        assert result.done is True

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
        assert result.done is False


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


class TestButtercupIsAThreeSyllableTemplate:
    """The defect the template fixes: a global vowel set finds two of buttercup's three syllables."""

    def test_every_one_of_its_three_syllables_is_found_including_the_rhotic_one(self, ddk_config: TriageConfig) -> None:
        """/bʌ/ /tər/ /kʌp/ is three units per repetition; a low-vowel set drops the middle one."""
        reading = _reading(_buttercup_raster(6), ddk_config, BUTTERCUP)
        assert [(unit.consonant, unit.vowel) for unit in reading.units[:3]] == [("b", "ah"), ("t", "er"), ("k", "ah")]
        assert len(reading.units) == 18

    def test_the_declared_template_is_what_admits_the_rhotic_nucleus(self, ddk_config: TriageConfig) -> None:
        """Extraction is permissive over the classes the template names, and over no others.

        ``/pa/`` names only ``low``, so the same raster read under that template loses the /t/ unit —
        which is what a single global vowel set did to buttercup under every template.
        """
        under_pa = _reading(_buttercup_raster(6), ddk_config, (LABIAL_LOW,))
        assert [(unit.consonant, unit.vowel) for unit in under_pa.units[:2]] == [("b", "ah"), ("k", "ah")]
        assert len(under_pa.units) == 12

    def test_the_coda_p_of_cup_opens_no_syllable_and_is_counted_as_none(self, ddk_config: TriageConfig) -> None:
        """A consonant with no following nucleus yields no unit; coda modelling is deliberately absent."""
        with_coda = _reading(_buttercup_raster(6), ddk_config, BUTTERCUP)
        without = _reading(_buttercup_raster(6, coda=False), ddk_config, BUTTERCUP)
        assert len(with_coda.units) == len(without.units) == 18
        assert "p" not in {unit.consonant for unit in with_coda.units}

    def test_the_nucleus_conformance_is_per_position_against_the_template(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Said as asked, buttercup reads low-rhotic-low at its three positions and scores 1.0."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            posteriorgram=_buttercup_raster(6),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        [nucleus] = _measurements(store, PPG_CYCLE_NUCLEUS)
        assert nucleus.attributes["expected_sequence"] == ["low", "rhotic", "low"]
        assert nucleus.attributes["value"] == pytest.approx(1.0)
        assert nucleus.attributes["by_position"] == {"0": 1.0, "1": 1.0, "2": 1.0}
        assert nucleus.attributes["support_syllables"] == 18

    def test_a_substituted_nucleus_is_a_finding_and_not_a_dropped_unit(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A "buttercap" keeps all three syllables and scores 2/3, which is the clinical reading."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            posteriorgram=_buttercup_raster(6, nuclei=("ah", "ah", "ah")),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        assert _detail(result)["ppg_cv_units_n"] == 18
        [nucleus] = _measurements(store, PPG_CYCLE_NUCLEUS)
        assert nucleus.attributes["value"] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert nucleus.attributes["by_position"] == {"0": 1.0, "1": 0.0, "2": 1.0}
        assert result.done is True

    def test_a_pa_recording_is_unaffected_by_any_of_this(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The control: a one-position low template counts every syllable as its own cycle."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [cycles] = _measurements(store, PPG_CYCLE_RATE)
        [nucleus] = _measurements(store, PPG_CYCLE_NUCLEUS)
        assert cycles.attributes["expected_sequence"] == ["labial"]
        assert cycles.attributes["cycles"] == 8
        assert cycles.attributes["consumed"] == pytest.approx(1.0)
        assert nucleus.attributes["expected_sequence"] == ["low"]
        assert nucleus.attributes["value"] == pytest.approx(1.0)
        assert nucleus.attributes["realised_nuclei"] == ["low"] * 8

    def test_it_counts_the_same_thirty_syllables_pataka_does(self) -> None:
        """Ten repetitions of three syllables, so the row is structurally the sequential one's."""
        assert SPEECH_EXPECTATIONS["diadochokinesis-buttercup"].expected_event_count == 30
        assert SPEECH_EXPECTATIONS["diadochokinesis-pataka"].expected_event_count == 30
        assert SPEECH_EXPECTATIONS["diadochokinesis-v2-buttercup"].declared_duration_s == 5.0

    def test_the_syllable_spans_are_minted_into_speechs_own_family(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """One minting family per branch is what ``dispatch`` enforces; a second would weaken it."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            posteriorgram=_buttercup_raster(6),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        minted = [entity for entity in live_entities(store, "span") if entity.attributes.get("role")]
        assert minted and {entity.attributes["family"] for entity in minted} == {"speech"}

    def test_a_participant_who_said_nothing_at_all_is_still_noticed(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """What the deleted token match answered, the carrier and the CV walk answer without it."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            envelope=np.full(int(6.0 * ENVELOPE_HZ), SILENT_DBFS),
            posteriorgram=_raster([("<silent>", 6.0)]),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        assert _spans_of(store) == []
        assert result.done is False


class TestTheDeclaredFamilySelectsTheBody:
    """The declared family picks which of SPEECH's bodies runs, and never supplies the answer."""

    def test_a_syllable_family_takes_the_syllable_body(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A declared ``diadochokinesis-ka`` is in family for SPEECH, so the train is evaluated."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-ka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, ddk_config, tmp_path)
        assert mode_of("SPEECH", store) == ("align", "diadochokinesis-ka")
        assert _spans_of(store)[0].attributes["role"] == "task_extent"
        assert _spans_of(store)[0].attributes["production"] == "syllable_train"
        assert _detail(result)["trains_n"] == 1

    def test_a_syllable_recording_reaches_no_second_branch(
        self, store: ProvStore, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """SPEECH is the only branch a ``diadochokinesis-*`` family is in family for."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa")
        assert mode_of("SPEECH", store)[0] == "align"
        assert [branch for branch in BRANCHES if mode_of(branch, store)[0] == "align"] == ["SPEECH"]

    def test_another_branchs_family_takes_the_out_of_family_mode(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """``prolonged-vowel`` is VOICE's kind, so no syllable body runs and no train is minted."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-prolonged-vowel",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, ddk_config, tmp_path)
        assert mode_of("SPEECH", store) == ("detect", "prolonged-vowel")
        assert result.done == UNDETERMINED
        assert [span.attributes["role"] for span in _spans_of(store)] == []

    def test_the_hint_task_token_selects_the_syllable_body_over_the_path(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The clean carrier wins: a hint naming a syllable task puts an unnamed path into align.

        Asserted on the span role, which only the in-family body mints, and not only on the mode
        ``mode_of`` reports: both derive the family from ``declared_task_family(store, hint)``, so
        they agree only while both are handed the same hint.
        """
        seed_ddk_store(store, stem="whatever", envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1), spans=[(1.0, 5.0)])
        hint = AudioHints(metadata={"task_token": "diadochokinesis-ta"})
        _run(store, ddk_config, tmp_path, hint)
        [span] = _spans_of(store)
        assert span.attributes["role"] == "task_extent"
        assert span.attributes["production"] == "syllable_train"
        assert mode_of("SPEECH", store, hint) == ("align", "diadochokinesis-ta")


class TestAnAbsentInstrumentIsNotANegativeReading:
    """UNDETERMINED where a rate cannot be read, and a flag rather than a fail."""

    def test_an_absent_envelope_makes_align_undetermined(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Neither rate instrument is present, so each says so and no task was evaluated."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa", spans=[(1.0, 5.0)])
        result = align_ddk(
            SPEECH_EXPECTATIONS["diadochokinesis-pa"],
            store,
            branch_params(ddk_config),
            reads=read_ddk(store, tmp_path, "plain"),
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
        assert result.done == UNDETERMINED
        assert read_ddk(store, tmp_path, "plain").envelope is None, NO_ENVELOPE

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
    for the syllable body that is :data:`UNDETERMINED` when the carrier search itself is gated
    (``train_min_s``), because an ungated carrier search finds none.
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
        assert result.done == UNDETERMINED
        assert "train_min_s" in _unmeasured(store)
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


class TestThereIsNoSecondBranchForASyllableTask:
    """The DDK branch is dissolved: one recording, one branch, one report about the task."""

    def test_the_runner_dispatches_the_three_branches_the_vocabulary_declares(self) -> None:
        """``run.py``'s branch table names exactly ``BRANCHES``, and DDK is not one of them."""
        import inspect

        from senselab.audio.workflows.triage import run as run_module

        source = inspect.getsource(run_module._drive_branches)
        assert set(BRANCHES) == {"AIRWAY", "SPEECH", "VOICE"}
        assert "DDK" not in source
        for branch in BRANCHES:
            assert f'"{branch}": lambda:' in source

    def test_a_syllable_recording_is_routed_to_speech_and_to_no_second_branch(
        self, store: ProvStore, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The fold is handed one branch decision for a declared train, and it is SPEECH's."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa")
        aligned = [branch for branch in BRANCHES if mode_of(branch, store)[0] == "align"]
        assert aligned == ["SPEECH"]

    def test_a_routed_branch_that_reported_nothing_still_flags(self) -> None:
        """The flag is not wrong and is not edited: a routed branch that reported nothing flags."""
        folded = fold_file_verdict(
            [],
            branch_decisions={
                "SPEECH": BranchDecision(
                    branch="SPEECH", will_run=True, route_state="routed", forced_by_declaration=False
                )
            },
            ran={"SPEECH": RunState.SKIPPED},
            hint_claims={},
            route_state="routed",
        )
        assert any(reason.node == "SPEECH" and "never ran" in reason.why for reason in folded.reasons)


def _reading(frames: np.ndarray, config: TriageConfig, template: Sequence[Syllable] | None = None) -> PpgReading:
    """The CV walk over one raster, at the packaged segmentation points."""
    ppg = Posteriorgram(frames=frames, phonemes=PHONEME_LABELS, seconds_per_frame=FRAME_S)
    found = ppg_reading(ppg, branch_params(config), template)
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

    def test_nucleus_variation_inside_one_class_does_not_cost_a_unit(self, ddk_config: TriageConfig) -> None:
        """/pa/, /pah/, /paw/ and /pie/ are the same syllable for this instrument: all ``low``."""
        raster = _raster(
            [
                ("<silent>", 0.2),
                ("p", 0.05),
                ("aa", 0.2),
                ("p", 0.05),
                ("ao", 0.2),
                ("p", 0.05),
                ("aw", 0.2),
                ("p", 0.05),
                ("ay", 0.2),
            ]
        )
        reading = _reading(raster, ddk_config, (LABIAL_LOW,))
        assert [unit.vowel for unit in reading.units] == ["aa", "ao", "aw", "ay"]
        assert reading.train is not None
        assert reading.train.repetitions == 4


class TestThePosteriorgramAnswersTheDeclaredTask:
    """The instrument is task-agnostic; the comparison is against the declared expectation."""

    def test_a_declared_sequence_counts_one_cycle_per_realised_repeat(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """/pa-ta-ka/ produced as asked yields four complete cycles accounting for every unit."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(list(CYCLE) * 4, [0.2] * 11),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [cycles] = _measurements(store, PPG_CYCLE_RATE)
        assert cycles.attributes["cycles"] == 4
        assert cycles.attributes["consumed"] == pytest.approx(1.0)
        assert cycles.attributes["insertions_n"] == 0
        assert cycles.attributes["expected_sequence"] == list(CYCLE)

    def test_a_collapsed_sequence_completes_no_cycle_without_failing_conformance(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """/pa-pa-pa/ for /pa-ta-ka/ is the clinically meaningful finding, not a task not performed."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial"] * 12, [0.2] * 11),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [cycles] = _measurements(store, PPG_CYCLE_RATE)
        assert cycles.attributes["cycles"] == 0
        assert cycles.attributes["consumed"] == pytest.approx(0.0)
        assert cycles.attributes["value"] is None
        assert result.done is True

    def test_a_single_syllable_family_counts_cycles_of_the_place_its_own_instruction_names(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """/ta/ expects alveolar; a train of /pa/ under that instruction completes no cycle."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-ta", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-ta"}))
        [cycles] = _measurements(store, PPG_CYCLE_RATE)
        assert cycles.attributes["expected_sequence"] == ["alveolar"]
        assert cycles.attributes["cycles"] == 0
        assert cycles.attributes["consumed"] == pytest.approx(0.0)

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
            SPEECH_EXPECTATIONS["diadochokinesis-pa"],
            store,
            branch_params(ddk_config),
            reads=read_ddk(store, tmp_path, "plain"),
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
            SPEECH_EXPECTATIONS["diadochokinesis-pa"],
            store,
            branch_params(ddk_config),
            reads=read_ddk(store, tmp_path, "plain"),
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
        assert read_ddk(store, tmp_path, "plain").ppg is None, NO_PPG
        [absent] = _measurements(store, PPG_RATE)
        assert absent.attributes["unavailable"] == "ppg_posteriorgram"
        assert absent.attributes["value"] is None
