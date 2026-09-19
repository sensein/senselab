"""The syllable-train instrument: the extent it proposes and the ten SPEECH tasks it evaluates.

Not a branch. Every test here drives the product path SPEECH takes — ``dispatch`` over
``align_speech``, which serves the ten ``diadochokinesis-*`` rows through ``align_ddk``.

Most tests still override every operating point the body reads through the mechanism a campaign
would use, so the numbers are fixture values, not fits: what is pinned is that the body reads the
key, not where the key falls. The posteriorgram instrument is a cyclic decode of the token's
phoneme sequence and its two parameters are derived at read time rather than configured, so the
only points a decode test can clear are the two class vocabularies.
"""

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.features_extraction.ppg import PHONEME_LABELS
from senselab.audio.workflows.triage.config import (
    DATA_MAP_PATHS,
    TriageConfig,
    UnknownConfigKey,
    load_triage_config,
)
from senselab.audio.workflows.triage.nodes import ddk
from senselab.audio.workflows.triage.nodes.branches import (
    BUTTERCUP,
    DEVIATION_TYPES,
    PA,
    PARAM_KEYS,
    PATAKA,
    SPEECH_EXPECTATIONS,
    UNDETERMINED,
    UNMEASURED_POINTS,
    BranchParams,
    Result,
    branch_params,
    deviation_names,
    dispatch,
    mode_of,
    propose_spans,
    write_findings,
)
from senselab.audio.workflows.triage.nodes.common import (
    BRANCH_MEASURES,
    lexical_words,
    live_entities,
    software_agent,
)
from senselab.audio.workflows.triage.nodes.ddk import (
    CONTRADICTED,
    CV_AUTHORITY,
    CYCLES_OR_SYLLABLES_PER_S,
    INSTRUMENT_READING,
    NO_ENVELOPE,
    NO_PPG,
    NO_REPETITIONS,
    PPG_DISPERSION,
    PPG_MASS,
    PPG_RATE,
    PPG_REPETITIONS,
    RATE,
    SYLLABLES_PER_S,
    TASK_FROM_DECODE,
    TRANSCRIPT_CLAIM,
    Decode,
    Posteriorgram,
    Visit,
    align_ddk,
    arcs,
    class_masses,
    decode_template,
    dispersion,
    min_phone_frames,
    phoneme_classes,
    read_ddk,
    repetitions_of,
    syllable_detail,
    trend,
    visits,
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
"""The working rate PREPROCESS resamples to."""

ENVELOPE_HZ = 1000.0
FLOOR_DBFS = -60.0
PEAK_DBFS = -35.0
SILENT_DBFS = -70.0

FRAME_S = 0.01
"""The posteriorgram's frame period in these fixtures, and ppgs' own to within 0.26 ms."""

STOP_OF = {"labial": "p", "alveolar": "t", "velar": "k"}
"""One stop per place, to write a raster whose expected classes are known by construction."""


def _raster(syllables: Sequence[tuple[str, float]]) -> np.ndarray:
    """A one-hot raster over a phoneme sequence with per-phoneme durations.

    Args:
        syllables: ``(phoneme, seconds)`` in order.

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
    repeats: int,
    *,
    nuclei: Sequence[str] = BUTTERCUP_NUCLEI,
    coda: bool = True,
    lead_s: float = 0.2,
    flap: str = "t",
) -> np.ndarray:
    """A raster of ``buttercup`` repetitions: /b/+/t/+/k/ onsets, and the coda /p/ of "cup".

    Args:
        repeats: How many times the word is said.
        nuclei: The three nuclei, so a test can substitute one without touching the onsets.
        coda: Whether the final /p/ of "cup" is realised.
        lead_s: Silence before the first onset.
        flap: What the intervocalic /t/ of "butter" surfaces as, which ARPAbet-40 spells three ways.

    Returns:
        The raster.
    """
    sequence: list[tuple[str, float]] = [("<silent>", lead_s)]
    for _ in range(repeats):
        for stop, nucleus in zip(("b", flap, "k"), nuclei):
            sequence.extend([(stop, 0.05), (nucleus, 0.15)])
        if coda:
            sequence.append(("p", 0.05))
        sequence.append(("<silent>", 0.05))
    return _raster(sequence)


@pytest.fixture
def ddk_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with every ``branch`` key DDK's two modes read supplied.

    ``smoothing_window_s`` is 0.011 rather than 0.010 for the reason ``branches_test`` records: an
    even width in samples made ``boxcar`` a half-sample shift. The decode reads neither it nor any
    other numeric point — ``D`` and the emission floor come off the stored derivative itself.
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
        "  burst_window_ms: 20.0\n"
    )
    return load_triage_config(override)


def _train_envelope(duration_s: float, carrier: tuple[float, float], rate_hz: float, phase_s: float) -> np.ndarray:
    """An envelope silent outside ``carrier`` and modulated at ``rate_hz`` inside it.

    Args:
        duration_s: The recording's duration.
        carrier: The extent the train occupies.
        rate_hz: The repetition rate inside it.
        phase_s: Where the first peak sits relative to the carrier's start.

    Returns:
        The envelope in dBFS at :data:`ENVELOPE_HZ`.
    """
    samples = int(duration_s * ENVELOPE_HZ)
    times = np.arange(samples) / ENVELOPE_HZ
    envelope = np.full(samples, SILENT_DBFS)
    inside = (times >= carrier[0]) & (times < carrier[1])
    phase = 2.0 * np.pi * rate_hz * (times[inside] - carrier[0] - phase_s)
    envelope[inside] = PEAK_DBFS - 0.5 * (PEAK_DBFS - FLOOR_DBFS) * (1.0 - np.cos(phase))
    return envelope


def _flat_envelope(duration_s: float, carriers: Sequence[tuple[float, float]]) -> np.ndarray:
    """An envelope loud inside each carrier and silent outside, with no modulation at all.

    Args:
        duration_s: The recording's duration.
        carriers: The extents to fill.

    Returns:
        The envelope in dBFS.
    """
    samples = int(duration_s * ENVELOPE_HZ)
    times = np.arange(samples) / ENVELOPE_HZ
    envelope = np.full(samples, SILENT_DBFS)
    for start, end in carriers:
        envelope[(times >= start) & (times < end)] = PEAK_DBFS
    return envelope


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
        posteriorgram: np.ndarray | None = None,
        seconds_per_frame: float = FRAME_S,
        dtype: str = "float16",
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
        if posteriorgram is not None:
            np.savez(
                tmp_path / "derivatives" / "ppg_posteriorgram.npz",
                posteriorgram=posteriorgram.astype(np.dtype(dtype)),
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
                    "dtype": dtype,
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
    """Drive the syllable body the way SPEECH's node does, and write what it proposed."""
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


def _decode(frames: np.ndarray, config: TriageConfig, template: Sequence[str] | None) -> Decode:
    """Run the decode over a raster under one template, outside the store."""
    ppg = Posteriorgram(frames=frames, phonemes=PHONEME_LABELS, seconds_per_frame=FRAME_S)
    decoded = decode_template(ppg, branch_params(config), template)
    assert decoded is not None
    return decoded


class TestTheDecodeCountsRepetitionsOfThePhonemeSequence:
    """The mechanism: one state tier per template phoneme, a filler between, a wrap arc closing."""

    def test_a_clean_eight_repetition_pa_train_decodes_eight(self, ddk_config: TriageConfig) -> None:
        """Eight /pa/ syllables against the two-position template ``p aa`` are eight repetitions."""
        decoded = _decode(_cv_raster(["labial"] * 8, [0.25] * 7), ddk_config, PA)
        assert decoded.template == ("p", "aa")
        assert decoded.classes == ("labial", "open")
        assert decoded.count == 8
        assert decoded.syllables == 8

    def test_the_period_is_the_time_between_repetition_starts(self, ddk_config: TriageConfig) -> None:
        """Onsets 0.25 s apart give a 0.25 s period and four syllables a second."""
        decoded = _decode(_cv_raster(["labial"] * 8, [0.25] * 7), ddk_config, PA)
        assert decoded.period_s == pytest.approx(0.25, abs=0.02)
        assert decoded.rate_hz == pytest.approx(4.0, abs=0.2)
        assert decoded.cycle_rate_hz == pytest.approx(4.0, abs=0.2)

    def test_every_position_carries_the_mass_of_the_class_it_expects(self, ddk_config: TriageConfig) -> None:
        """A one-hot raster puts all the mass in the expected class at every position."""
        decoded = _decode(_cv_raster(["labial"] * 8, [0.25] * 7), ddk_config, PA)
        assert list(decoded.realised_mass) == pytest.approx([1.0, 1.0])
        assert all(frames > 0 for frames in decoded.occupancy)

    def test_the_class_is_summed_rather_than_the_phoneme_matched(self, ddk_config: TriageConfig) -> None:
        """A frame split between /p/ and /b/ scores their sum, which is what voicing error costs."""
        frames = _cv_raster(["labial"] * 8, [0.25] * 7)
        labial = frames[:, PHONEME_LABELS.index("p")] > 0.0
        frames[labial] = 0.0
        frames[np.ix_(labial, [PHONEME_LABELS.index("p")])] = 0.5
        frames[np.ix_(labial, [PHONEME_LABELS.index("b")])] = 0.3
        decoded = _decode(frames, ddk_config, PA)
        assert decoded.count == 8
        assert decoded.realised_mass[0] == pytest.approx(0.8, abs=0.01)

    def test_a_three_syllable_sequence_decodes_its_six_positions(self, ddk_config: TriageConfig) -> None:
        """``pataka`` is ``p aa t aa k aa``: six positions, three syllables per repetition."""
        places = ["labial", "alveolar", "velar"] * 6
        decoded = _decode(_cv_raster(places, [0.2] * (len(places) - 1)), ddk_config, PATAKA)
        assert decoded.classes == ("labial", "open", "alveolar", "open", "velar", "open")
        assert decoded.count == 6
        assert decoded.vowel_positions == 3
        assert decoded.syllables == 18

    def test_a_collapsed_sequence_names_which_positions_collapsed(self, ddk_config: TriageConfig) -> None:
        """/pa-pa-pa/ produced for /pa-ta-ka/ reads high at position 0 and at floor at 2 and 4."""
        places = ["labial"] * 18
        decoded = _decode(_cv_raster(places, [0.2] * (len(places) - 1)), ddk_config, PATAKA)
        assert decoded.realised_mass[0] == pytest.approx(1.0)
        assert decoded.realised_mass[2] == pytest.approx(0.0, abs=0.01)
        assert decoded.realised_mass[4] == pytest.approx(0.0, abs=0.01)

    def test_silence_falls_in_the_filler_and_completes_no_repetition(self, ddk_config: TriageConfig) -> None:
        """A recording of nothing is a reading of zero, not an absence of one."""
        decoded = _decode(_raster([("<silent>", 6.0)]), ddk_config, PA)
        assert decoded.count == 0
        assert decoded.extent is None
        assert decoded.filler_fraction == pytest.approx(1.0)

    def test_a_leading_partial_repetition_is_not_counted(self, ddk_config: TriageConfig) -> None:
        """A recording that begins mid-performance is decodable; the partial is simply not one."""
        whole = _decode(_cv_raster(["labial"] * 8, [0.25] * 7), ddk_config, PA)
        started_late = _decode(_raster([("aa", 0.2), *[("p", 0.05), ("aa", 0.2)] * 8]), ddk_config, PA)
        assert whole.count == started_late.count == 8

    def test_a_trailing_consonant_with_no_nucleus_closes_no_repetition(self, ddk_config: TriageConfig) -> None:
        """The wrap arc has to be traversed; a template left half-run is not a repetition."""
        decoded = _decode(_raster([*[("p", 0.05), ("aa", 0.2)] * 8, ("p", 0.05)]), ddk_config, PA)
        assert decoded.count == 8


class TestTheTwoParametersAreDerivedAndNeitherIsAConfigKey:
    """``D`` off the stored frame period, the emission floor off the stored dtype."""

    @pytest.mark.parametrize(
        ("seconds_per_frame", "expected"), [(0.01, 2), (0.01026, 2), (0.02, 1), (0.005, 4), (0.0, 1)]
    )
    def test_the_chain_length_is_the_burst_window_in_frames(self, seconds_per_frame: float, expected: int) -> None:
        """``ceil(20 ms / frame period)``, at least one, which is 2 everywhere on the corpus."""
        assert min_phone_frames(seconds_per_frame, 20.0) == expected

    def test_a_longer_chain_would_make_the_fastest_reported_rate_undecodable(self) -> None:
        """The boundary the floor must never cross: 8 cycles/s of /pataka/ is 21 ms per phone."""
        phones_per_s = 8.0 * 6.0
        assert min_phone_frames(0.01, 20.0) * 0.01 < 1.0 / phones_per_s
        assert min_phone_frames(0.01, 30.0) * 0.01 > 1.0 / phones_per_s

    def test_the_emission_floor_is_the_stored_dtypes_smallest_subnormal(self) -> None:
        """A storage fact, not a knob: it moves on its own if the sidecar's dtype ever moves."""
        frames = np.zeros((4, len(PHONEME_LABELS)))
        half = Posteriorgram(frames=frames, phonemes=PHONEME_LABELS, seconds_per_frame=FRAME_S, dtype="float16")
        single = Posteriorgram(frames=frames, phonemes=PHONEME_LABELS, seconds_per_frame=FRAME_S, dtype="float32")
        assert half.emission_floor == pytest.approx(float(np.finfo(np.float16).smallest_subnormal))
        assert single.emission_floor < half.emission_floor

    def test_the_floor_keeps_an_empty_class_finite_rather_than_minus_infinity(self, ddk_config: TriageConfig) -> None:
        """A class holding no mass at all must score a number, or the whole path score is ``-inf``."""
        decoded = _decode(_raster([("<silent>", 2.0)]), ddk_config, PA)
        assert decoded.score_per_frame is not None
        assert np.isfinite(decoded.score_per_frame)

    def test_neither_parameter_is_a_branch_operating_point(self) -> None:
        """A key here would be an unmeasured decision with a public interface."""
        assert "min_phone_frames" not in PARAM_KEYS
        assert "emission_floor" not in PARAM_KEYS


class TestTheEmissionsAreOnePartitionOfThePhonemeInventory:
    """Every state emits the likelihood of one observation under a different part of one partition.

    That is what makes the log-scores comparable across states.
    """

    def test_the_named_classes_and_the_remainder_sum_to_one(self, ddk_config: TriageConfig) -> None:
        """The parts are mutually exclusive and exhaustive because the posteriorgram sums to one."""
        frames = _buttercup_raster(3)
        ppg = Posteriorgram(frames=frames, phonemes=PHONEME_LABELS, seconds_per_frame=FRAME_S)
        params = branch_params(ddk_config)
        lookup = phoneme_classes(params.point("phoneme_place_classes"), params.point("phoneme_vowel_classes"))
        masses = class_masses(ppg, ("labial", "open", "alveolar", "rhotic", "velar"), lookup)
        assert masses.shape == (frames.shape[0], 6)
        assert masses.sum(axis=1) == pytest.approx(np.ones(frames.shape[0]))

    def test_silence_falls_in_the_remainder_and_needs_no_state_of_its_own(self, ddk_config: TriageConfig) -> None:
        """No silence tier, no garbage tier: the filler between positions is all three."""
        ppg = Posteriorgram(frames=_raster([("<silent>", 0.5)]), phonemes=PHONEME_LABELS, seconds_per_frame=FRAME_S)
        params = branch_params(ddk_config)
        lookup = phoneme_classes(params.point("phoneme_place_classes"), params.point("phoneme_vowel_classes"))
        masses = class_masses(ppg, ("labial", "open"), lookup)
        assert masses[:, -1] == pytest.approx(np.ones(masses.shape[0]))

    def test_the_vowel_partition_is_total_over_the_ten_arpabet_vowels(self, ddk_config: TriageConfig) -> None:
        """A template that is a phoneme sequence must classify every vowel, not only the low ones."""
        classes = branch_params(ddk_config).point("phoneme_vowel_classes")
        assert classes is not None
        members = {phoneme for phonemes in classes.values() for phoneme in phonemes}
        vowels = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}
        assert vowels <= members

    def test_no_vowel_is_in_two_classes(self, ddk_config: TriageConfig) -> None:
        """A partition, so the summed masses cannot double-count and cannot exceed one."""
        classes = branch_params(ddk_config).point("phoneme_vowel_classes")
        assert classes is not None
        members = [phoneme for phonemes in classes.values() for phoneme in phonemes]
        assert len(members) == len(set(members))

    def test_the_rhotic_is_its_own_class_and_not_a_height(self, ddk_config: TriageConfig) -> None:
        """Merging ``er`` into ``open`` would give buttercup three interchangeable vowel slots."""
        classes = branch_params(ddk_config).point("phoneme_vowel_classes")
        assert classes is not None
        assert classes["rhotic"] == ("er",)
        assert "er" not in classes["open"]
        decoded = _decode(_buttercup_raster(4), ddk_config, BUTTERCUP)
        assert decoded.classes[1] != decoded.classes[3]

    def test_the_open_class_holds_exactly_what_the_low_class_held_plus_two(self, ddk_config: TriageConfig) -> None:
        """The extension makes the inventory total; it does not move the DDK reading."""
        classes = branch_params(ddk_config).point("phoneme_vowel_classes")
        assert classes is not None
        assert set(classes["open"]) == {"aa", "ae", "ah", "ao", "aw", "ay"} | {"eh", "oy"}


class TestTheTopologyIsCyclicAndEveryArcCostsZero:
    """The topology constrains which paths are legal; it is not a prior over which are likely."""

    def test_the_state_count_is_n_times_d_plus_one(self) -> None:
        """Twenty-one states for buttercup at ``D = 2``, which is nothing to decode over."""
        table, _ = arcs(7, 2)
        assert table.shape[0] == 7 * (2 + 1)

    def test_only_the_last_sub_state_of_a_position_self_loops(self) -> None:
        """The chain is how a minimum phone duration is enforced structurally, not by a penalty."""
        table, valid = arcs(3, 2)
        for position in range(3):
            first, last = position * 2, position * 2 + 1
            assert first not in set(table[first][valid[first]])
            assert last in set(table[last][valid[last]])

    def test_position_zero_is_reachable_from_the_last_position_and_from_its_filler(self) -> None:
        """The wrap arc is the repetition boundary, and the filler on it is lead-in and lead-out."""
        table, valid = arcs(3, 2)
        preds = set(table[0][valid[0]])
        assert preds == {2 * 2 + 1, 3 * 2 + 2}

    def test_a_filler_follows_its_position_and_itself(self) -> None:
        """It holds the frames the template does not account for and nothing else."""
        table, valid = arcs(3, 2)
        filler = 3 * 2 + 1
        assert set(table[filler][valid[filler]]) == {1 * 2 + 1, filler}

    def test_visits_charge_a_filler_frame_to_no_position(self) -> None:
        """A filler frame closes the stay it follows and opens none of its own."""
        found = visits(np.asarray([0, 1, 1, 6, 6, 2, 3]), 3, 2)
        assert found == [Visit(0, 0, 2), Visit(1, 5, 6)]

    def test_a_repetition_is_positions_zero_through_n_minus_one_in_order(self) -> None:
        """One tuple per traversal that completed; the trailing partial is not one."""
        found = [Visit(index % 3, index, index) for index in range(8)]
        complete = repetitions_of(found, 3)
        assert [run[0].position for run in complete] == [0, 0]
        assert [run[-1].position for run in complete] == [2, 2]
        assert len(complete) == 2


class TestButtercupHasSevenPositionsIncludingItsCoda:
    """The defect the phoneme template fixes: a syllable template has no position for a coda."""

    def test_the_template_is_the_tokens_phoneme_sequence_with_no_syllable_layer(self) -> None:
        """/bʌ.tər.kʌp/ is CVCVCVC — seven positions, where a syllable template carried three."""
        assert BUTTERCUP == ("b", "ah", "t", "er", "k", "ah", "p")
        assert SPEECH_EXPECTATIONS["diadochokinesis-buttercup"].sequence == BUTTERCUP
        assert SPEECH_EXPECTATIONS["diadochokinesis-v2-buttercup"].sequence == BUTTERCUP

    def test_the_coda_p_of_cup_is_reached_and_carries_real_mass(self, ddk_config: TriageConfig) -> None:
        """The capability claim A1: the shipped instrument read this position on 0% by construction."""
        decoded = _decode(_buttercup_raster(6), ddk_config, BUTTERCUP)
        assert decoded.count == 6
        assert decoded.occupancy[6] > 0
        assert decoded.realised_mass[6] == pytest.approx(1.0)

    def test_a_buttercup_said_without_its_coda_costs_the_last_repetition_not_the_position(
        self, ddk_config: TriageConfig
    ) -> None:
        """The coda and the next repetition's /b/ are the same class, so the decode borrows it.

        A deleted phone is never a structural deletion: position 6 is still traversed and still
        reported. What the deletion costs is the final repetition, which has no following /b/ to
        borrow — the honest reading, since without a coda there is nothing there to read.
        """
        with_coda = _decode(_buttercup_raster(6), ddk_config, BUTTERCUP)
        without = _decode(_buttercup_raster(6, coda=False), ddk_config, BUTTERCUP)
        assert with_coda.count == 6
        assert without.count == 5
        assert without.occupancy[6] > 0

    def test_the_rhotic_position_is_reached_and_distinguishes_the_middle_syllable(
        self, ddk_config: TriageConfig
    ) -> None:
        """What keeping ``er`` out of ``open`` buys: position 3 is not positions 1 and 5."""
        decoded = _decode(_buttercup_raster(6), ddk_config, BUTTERCUP)
        assert decoded.occupancy[3] > 0
        assert decoded.realised_mass[3] == pytest.approx(1.0)

    def test_a_substituted_nucleus_reads_at_floor_at_its_own_position(self, ddk_config: TriageConfig) -> None:
        """A "buttercap" keeps all seven positions and reads position 3 at the emission floor."""
        said = _decode(_buttercup_raster(6, nuclei=("ah", "ah", "ah")), ddk_config, BUTTERCUP)
        asked = _decode(_buttercup_raster(6), ddk_config, BUTTERCUP)
        assert said.count == asked.count == 6
        assert said.realised_mass[3] == pytest.approx(0.0, abs=0.01)
        assert asked.realised_mass[3] == pytest.approx(1.0)
        assert said.realised_mass[1] is not None and said.realised_mass[1] > 0.8

    @pytest.mark.parametrize("flap", ["t", "d", "r"])
    def test_the_alveolar_class_admits_all_three_spellings_of_butters_flap(
        self, ddk_config: TriageConfig, flap: str
    ) -> None:
        """ARPAbet-40 has no flap symbol, so /ɾ/ surfaces as ``t``, ``d`` or ``r`` — one segment."""
        decoded = _decode(_buttercup_raster(6, flap=flap), ddk_config, BUTTERCUP)
        assert decoded.count == 6
        assert decoded.realised_mass[2] == pytest.approx(1.0)

    def test_the_admission_is_declared_in_the_class_and_not_in_the_template(self, ddk_config: TriageConfig) -> None:
        """The template stays the stimulus text; the tolerance lives in the equivalence class."""
        places = branch_params(ddk_config).point("phoneme_place_classes")
        assert places is not None
        assert set(places["alveolar"]) == {"t", "d", "r"}
        assert "r" not in BUTTERCUP

    def test_it_counts_the_same_thirty_syllables_pataka_does(self) -> None:
        """Ten repetitions of three syllables, so the row is structurally the sequential one's."""
        assert SPEECH_EXPECTATIONS["diadochokinesis-buttercup"].expected_event_count == 30
        assert SPEECH_EXPECTATIONS["diadochokinesis-pataka"].expected_event_count == 30
        assert SPEECH_EXPECTATIONS["diadochokinesis-v2-buttercup"].declared_duration_s == 5.0


class TestEveryFamilysTemplateIsItsStimulusText:
    """It is data with no fitted content: the derivation of every entry is what the task says."""

    @pytest.mark.parametrize(
        ("family", "template"),
        [
            ("diadochokinesis-pa", ("p", "aa")),
            ("diadochokinesis-ta", ("t", "aa")),
            ("diadochokinesis-ka", ("k", "aa")),
            ("diadochokinesis-v2-puh", ("p", "ah")),
            ("diadochokinesis-v2-tuh", ("t", "ah")),
            ("diadochokinesis-v2-kuh", ("k", "ah")),
            ("diadochokinesis-pataka", ("p", "aa", "t", "aa", "k", "aa")),
            ("diadochokinesis-v2-puhtuhkuh", ("p", "ah", "t", "ah", "k", "ah")),
            ("diadochokinesis-buttercup", ("b", "ah", "t", "er", "k", "ah", "p")),
            ("diadochokinesis-v2-buttercup", ("b", "ah", "t", "er", "k", "ah", "p")),
        ],
    )
    def test_the_row_carries_the_tokens_phonemes(self, family: str, template: tuple[str, ...]) -> None:
        """Ten rows, nine sequences; the v2 rows are the same token timed rather than counted."""
        assert SPEECH_EXPECTATIONS[family].sequence == template

    def test_the_three_place_families_differ_only_in_their_first_phoneme(self) -> None:
        """The place contrast is the whole of what /pa/ /ta/ /ka/ ask differently."""
        firsts = {SPEECH_EXPECTATIONS[f"diadochokinesis-{name}"].sequence for name in ("pa", "ta", "ka")}
        assert {sequence[1] for sequence in firsts} == {"aa"}
        assert {sequence[0] for sequence in firsts} == {"p", "t", "k"}

    def test_a_row_survives_a_round_trip_through_its_mapping(self) -> None:
        """A run records which expectation it applied, and a phoneme tuple has to come back."""
        row = SPEECH_EXPECTATIONS["diadochokinesis-buttercup"]
        assert type(row).from_mapping(row.as_mapping()).sequence == BUTTERCUP


class TestTheCountsAreHeuristicsAndNothingFoldsThem:
    """The owner's governing constraint: expected count is not a target, it is a heuristic."""

    def test_the_declared_count_and_the_decoded_one_are_written_beside_each_other(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Found beside declared, in the same unit: both are syllables."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [counts] = _measurements(store, "counts")
        entry = counts.attributes["entries"]["expected_event_count"]
        assert entry["declared"] == 10
        assert entry["found"] == 8

    def test_the_found_count_is_in_syllables_because_the_declaration_is(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Six repetitions of a three-syllable token is eighteen syllables against a declared 30."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            posteriorgram=_buttercup_raster(6),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        [counts] = _measurements(store, "counts")
        entry = counts.attributes["entries"]["expected_event_count"]
        assert entry["declared"] == 30
        assert entry["found"] == 18
        [repetitions] = _measurements(store, PPG_REPETITIONS)
        assert repetitions.attributes["value"] == 6
        assert repetitions.attributes["declared_repetitions"] == 10

    def test_a_count_short_of_its_declaration_is_not_a_deviation_and_not_a_non_conformance(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Every extractor is imperfect, so a shortfall is not evidence about the participant."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            posteriorgram=_buttercup_raster(2),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        assert result.done is True
        assert deviation_names(result.deviations) == ()

    def test_no_finding_compares_the_two_counts(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """No conformance term, no score and no gate reads the pair — only the pair itself ships."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            posteriorgram=_buttercup_raster(2),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        scores = [
            finding
            for finding in result.deviations
            if finding.kind == "measure" and {"declared_event_count", "declared_repetitions"} & set(finding.evidence)
        ]
        assert [finding.name for finding in scores] == [PPG_REPETITIONS]
        assert set(scores[0].evidence) & {"conformance", "agreement", "shortfall", "ratio"} == set()

    def test_the_field_keeps_its_name(self) -> None:
        """The owner kept ``expected_event_count``: for 3 heys or 5 breaths the number is meaningful.

        Which kind of count a task declares is a per-task property that does not exist yet and is
        sequenced separately; until it does, a DDK reader has no field saying the number is a guide.
        """
        assert SPEECH_EXPECTATIONS["diadochokinesis-pataka"].expected_event_count == 30
        assert not hasattr(SPEECH_EXPECTATIONS["diadochokinesis-pataka"], "declared_event_count")


class TestTheTaskExtentIsTheDecodedRepetitionSpan:
    """Exactly one span, and it is a boundary rather than a mask over matched positions."""

    def test_the_extent_runs_from_the_first_repetitions_start_to_the_last_ones_end(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The only span this family proposes, and the decode is what mints it."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        spans = _spans_of(store)
        assert len(spans) == 1
        assert spans[0].attributes["role"] == "task_extent"
        assert spans[0].attributes["production"] == TASK_FROM_DECODE
        assert spans[0].attributes["repetitions"] == 6

    def test_it_includes_the_filler_between_repetitions_rather_than_masking_it_out(
        self, ddk_config: TriageConfig
    ) -> None:
        """A convex boundary: the span's duration is the whole stretch, not the matched frames."""
        decoded = _decode(_buttercup_raster(6), ddk_config, BUTTERCUP)
        extent = decoded.extent
        assert extent is not None
        held = sum(decoded.occupancy) * FRAME_S
        assert extent[1] - extent[0] > held

    def test_a_pause_inside_the_performance_does_not_split_the_extent(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The subject pausing mid-task was still attempting it, so the boundary spans the pause."""
        places = ["labial", "alveolar", "velar"] * 6
        intervals = [0.2] * 5 + [2.5] + [0.2] * (len(places) - 7)
        seed_ddk_store(
            store,
            duration_s=12.0,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(places, intervals),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [span] = _spans_of(store)
        assert span.extent is not None
        assert span.extent[1] - span.extent[0] > 2.5

    def test_it_names_the_posteriorgram_it_was_read_off(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Under propose-only the derivation is the whole record of where the extent came from."""
        ids = seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [span] = _spans_of(store)
        assert store.derived_from(span.id) == [ids["ppg_posteriorgram"]]

    def test_a_carrier_the_decode_never_read_proposes_no_task_extent(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """An amplitude carrier is not phonetic evidence, so it cannot say where the task was."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        assert _spans_of(store) == []
        assert [span.attributes.get("production") for span in _spans_of(store)] == []

    def test_no_task_extent_carries_a_production_the_envelope_minted(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The two envelope productions are gone outright, not left unreachable behind a branch."""
        assert not hasattr(ddk, "TASK_FROM_ENVELOPE")
        assert not hasattr(ddk, "TASK_FROM_ENVELOPE_SEQUENCE")
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        productions = {span.attributes.get("production") for span in live_entities(store, "span")}
        assert productions & {"syllable_train", "syllable_sequence"} == set()

    def test_the_absent_decode_reads_as_an_absent_instrument_not_as_found_nothing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Both propose no extent, and the two states stay told apart by their own records."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert _spans_of(store) == []
        [absent] = _measurements(store, PPG_RATE)
        assert absent.attributes["unavailable"] == "ppg_posteriorgram"
        assert "reason" not in absent.attributes
        assert _measurements(store, PPG_REPETITIONS) == []

    def test_a_decode_that_read_nothing_still_reads_as_found_nothing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The other side of the same distinction, on the same seeded carrier."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
            posteriorgram=_raster([("<silent>", 6.0)]),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert _spans_of(store) == []
        [rate] = _measurements(store, PPG_RATE)
        assert rate.attributes["reason"] == NO_REPETITIONS
        assert "unavailable" not in rate.attributes
        [repetitions] = _measurements(store, PPG_REPETITIONS)
        assert repetitions.attributes["value"] == 0

    def test_the_decode_takes_precedence_over_a_carrier_that_also_read_the_task(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """One extent survives, and where both read it the decode is the one that knows where."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (4.0, 5.8), 5.0, 0.1),
            spans=[(4.0, 5.8)],
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [span] = _spans_of(store)
        assert span.attributes["production"] == TASK_FROM_DECODE
        assert span.extent is not None
        assert span.extent[1] < 3.0

    def test_a_recording_neither_instrument_read_proposes_no_extent(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Inventing an extent over a task that was not performed is the error the gate avoids."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=np.full(int(6.0 * ENVELOPE_HZ), SILENT_DBFS),
            posteriorgram=_raster([("<silent>", 6.0)]),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        assert _spans_of(store) == []
        assert result.done is False

    def test_the_span_is_minted_into_speechs_own_family_like_every_other_role(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """DDK dissolved into SPEECH, so a ``ddk`` family would be a family no reader knows."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        assert _spans_of(store, "ddk") == []
        assert {str(span.attributes["role"]) for span in _spans_of(store)} == {"task_extent"}

    def test_a_task_extent_touching_the_recordings_edge_is_a_truncation(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The deviation keys on the span that ships, so it cannot go quiet when the input moves."""
        raster = _cv_raster(["labial"] * 8, [0.25] * 7, lead_s=0.0)
        seed_ddk_store(
            store,
            duration_s=raster.shape[0] * FRAME_S,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=raster,
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert "truncation" in deviation_names(result.deviations)


class TestTheEnvelopeChannelIsTheModulationSpectrumAndNothingElse:
    """The peak-walk onset channel is retired; the modulation-rate channel survives."""

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
        assert rate.attributes["uncalibrated"] is True

    def test_a_sequential_family_keeps_the_ambiguous_unit(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A sequential train modulates at the cycle rate as well as the syllable rate."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        [rate] = _measurements(store, RATE)
        assert rate.attributes["unit"] == CYCLES_OR_SYLLABLES_PER_S

    def test_the_decodes_two_rates_travel_beside_the_peak_so_the_ambiguity_is_checkable(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Nothing new is measured; the reader can see which of the two the peak matched."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [rate] = _measurements(store, RATE)
        assert rate.attributes["decoded_syllable_rate_hz"] is not None
        assert rate.attributes["decoded_cycle_rate_hz"] is not None
        assert rate.attributes["decoded_syllable_rate_hz"] > rate.attributes["decoded_cycle_rate_hz"]

    def test_no_onset_channel_measurement_survives(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A median of 4 onsets against a declared 10 is not repairable by re-deriving a threshold."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        for name in ("interval_dispersion", "sequence_collapse_fraction", "syllable_place"):
            assert _measurements(store, name) == []
        entries = {key for counts in _measurements(store, "counts") for key in counts.attributes["entries"]}
        assert entries & {"syllable_onset_s", "inter_onset_interval_s", "realised_cycles"} == set()

    def test_the_rate_is_read_over_the_carrier_and_mints_nothing_from_it(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The carrier locates the modulation reading and nothing else; no span comes off it."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (2.237, 4.114), 5.0, 0.1),
            spans=[(2.237, 4.114)],
        )
        _run(store, ddk_config, tmp_path)
        [rate] = _measurements(store, RATE)
        assert rate.extent == pytest.approx((2.237, 4.114))
        assert _spans_of(store) == []

    def test_a_carrier_with_no_readable_modulation_is_no_carrier(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Envelope-first, then qualified by modulation: a flat carrier holds no train."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_flat_envelope(6.0, [(1.0, 5.0)]),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, ddk_config, tmp_path)
        assert _spans_of(store) == []
        assert result.done is False

    def test_the_train_fraction_is_taken_over_the_extent_that_ships(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Keyed on the surviving span, so it cannot describe an extent no reader ever sees."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
            posteriorgram=_cv_raster(["labial"] * 17, [0.25] * 16, lead_s=1.0),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [span] = _spans_of(store)
        assert span.extent is not None
        [fraction] = _measurements(store, "train_fraction_of_recording")
        assert fraction.attributes["train_s"] == pytest.approx(span.extent[1] - span.extent[0], abs=0.001)
        assert fraction.attributes["value"] == pytest.approx(fraction.attributes["train_s"] / 6.0, abs=0.001)


class TestTheBurstPlaceInstrumentIsGone:
    """Under the decode there is no place decision to make, only mass to sum."""

    def test_no_place_agreement_measurement_is_taken(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The covariate was least interpretable exactly where a check would have mattered.

        The burst spectrum degrades in noisier recordings.
        """
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert _measurements(store, "ddk_place_agreement_ppg_vs_burst") == []

    def test_its_two_operating_points_are_not_config_keys(self, ddk_config: TriageConfig) -> None:
        """``ddk_places`` was their only consumer in ``src/``, so they go with it."""
        for key in ("branch.place_centroid_bands_hz", "branch.place_margin_db"):
            with pytest.raises(UnknownConfigKey):
                ddk_config.require(key)

    def test_the_wideband_spectrogram_is_no_longer_read_by_this_module(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """DDK's only use of it was the burst place path; AIRWAY's own reading is untouched."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        reads = read_ddk(store, tmp_path, "plain")
        assert not hasattr(reads, "wideband")

    def test_the_burst_window_survives_with_its_consumer_changed(self, ddk_config: TriageConfig) -> None:
        """It stops feeding the burst spectrum and starts deriving the decode's chain length."""
        window = branch_params(ddk_config).point("burst_window_ms")
        assert window == pytest.approx(20.0)
        assert min_phone_frames(FRAME_S, window) == 2


class TestTheDeviationVocabularyLostItsUnwritableEntry:
    """A threshold on per-position mass has no derivation, so the deviation it needed goes."""

    def test_syllable_sequence_mismatch_is_no_longer_declared(self) -> None:
        """``write_findings`` refuses any deviation not in the vocabulary, so this cannot be raised."""
        assert "syllable_sequence_mismatch" not in DEVIATION_TYPES

    def test_a_collapsed_sequence_reports_mass_per_position_rather_than_a_deviation(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Which positions collapsed is more informative than that some syllable did not match."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial"] * 18, [0.2] * 17),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        assert deviation_names(result.deviations) == ()
        [mass] = _measurements(store, PPG_MASS)
        assert mass.attributes["value"][0] == pytest.approx(1.0)
        assert mass.attributes["value"][2] == pytest.approx(0.0, abs=0.01)
        assert mass.attributes["positions"] == list(PATAKA)


class TestTheClassMappingsAreCampaignOverridable:
    """Two shipped documents said so and the schema rejected it, which was a live defect."""

    @pytest.mark.parametrize("key", ["branch.phoneme_place_classes", "branch.phoneme_vowel_classes"])
    def test_the_key_is_declared_as_a_data_mapping(self, key: str) -> None:
        """Every other mapping is schema and an override may only change keys it already has."""
        assert key in DATA_MAP_PATHS

    def test_an_override_adding_a_place_class_is_admitted(self, tmp_path: Path) -> None:
        """A campaign may add a class without editing the installed package."""
        override = tmp_path / "campaign.yaml"
        override.write_text("branch:\n  phoneme_place_classes:\n    palatal: [ch, jh]\n")
        classes = branch_params(load_triage_config(override)).point("phoneme_place_classes")
        assert classes is not None
        assert classes["palatal"] == ("ch", "jh")
        assert classes["labial"] == ("p", "b")

    def test_an_override_adding_a_vowel_class_is_admitted(self, tmp_path: Path) -> None:
        """The same idiom, and the same reason."""
        override = tmp_path / "campaign.yaml"
        override.write_text("branch:\n  phoneme_vowel_classes:\n    nasalised: [ah]\n")
        classes = branch_params(load_triage_config(override)).point("phoneme_vowel_classes")
        assert classes is not None
        assert "nasalised" in classes

    def test_an_override_of_a_schema_mapping_is_still_refused(self, tmp_path: Path) -> None:
        """The control: the admission is per key, not a general loosening of the schema."""
        override = tmp_path / "campaign.yaml"
        override.write_text(
            "branch:\n  label_sets:\n    cough: [Cough]\n  modulation_band_hz: [1.0, 10.0]\n  nope: 1\n"
        )
        with pytest.raises(ValueError):
            load_triage_config(override)


class TestAnAbsentInstrumentIsNotANegativeReading:
    """Three states stay distinct and each keeps its own record."""

    def test_an_absent_envelope_makes_align_undetermined(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """No envelope derivative is an absent instrument, not an absent performance."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa")
        result = _run(store, ddk_config, tmp_path)
        assert result.done is UNDETERMINED
        assert _spans_of(store) == []

    def test_an_absent_envelope_notes_rather_than_failing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The measurement has no value and names the derivative that was missing."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa")
        _run(store, ddk_config, tmp_path)
        [rate] = _measurements(store, RATE)
        assert rate.attributes["value"] is None
        assert rate.attributes["unavailable"] == "energy_envelope"
        assert read_ddk(store, tmp_path, "plain").envelope is None, NO_ENVELOPE

    def test_an_absent_posteriorgram_is_noted_rather_than_raised(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The decode has no reading, and the envelope's conformance is unchanged by that."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [rate] = _measurements(store, PPG_RATE)
        assert rate.attributes["value"] is None
        assert rate.attributes["unavailable"] == "ppg_posteriorgram"
        assert result.done is True
        assert read_ddk(store, tmp_path, "plain").ppg is None, NO_PPG

    def test_a_decode_that_completed_nothing_reports_zero_and_not_an_absence(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """This is a reading, and it is the only one of the three that can support a false."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_raster([("<silent>", 6.0)]),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [repetitions] = _measurements(store, PPG_REPETITIONS)
        assert repetitions.attributes["value"] == 0
        [mass] = _measurements(store, PPG_MASS)
        assert mass.attributes["value"] == [None, None]
        [rate] = _measurements(store, PPG_RATE)
        assert rate.attributes["reason"] == NO_REPETITIONS
        assert result.done is False

    def test_an_unmeasured_class_vocabulary_leaves_the_decode_silent_and_names_the_key(
        self, store: ProvStore, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A branch does not refuse over an unmeasured point; ``params.missing`` is the record."""
        override = tmp_path / "cleared.yaml"
        override.write_text("branch:\n  phoneme_place_classes: null\n")
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        _run(store, load_triage_config(override), tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert _measurements(store, PPG_REPETITIONS) == []
        assert "phoneme_place_classes" in _unmeasured(store)

    def test_the_decode_has_no_numeric_operating_point_that_can_be_null(
        self, store: ProvStore, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A real narrowing of the unmeasured surface: only the class mappings can reach it."""
        override = tmp_path / "cleared.yaml"
        override.write_text("branch:\n  train_min_s: null\n")
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        _run(store, load_triage_config(override), tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [repetitions] = _measurements(store, PPG_REPETITIONS)
        assert repetitions.attributes["value"] == 8
        assert repetitions.attributes["min_phone_frames"] == 2


class TestConformanceNarrowsAndDoesNotWiden:
    """``Done`` stays three-valued and the only honest ``false`` is unchanged."""

    def test_a_repetition_found_by_either_instrument_is_conformance(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A readable decode answers a task the envelope could not."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_flat_envelope(6.0, [(1.0, 5.0)]),
            spans=[(1.0, 5.0)],
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert result.done is True

    def test_no_repetition_and_no_carrier_is_the_one_honest_false(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The instrument ran and found no evidence the task was performed at all."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=np.full(int(6.0 * ENVELOPE_HZ), SILENT_DBFS),
            posteriorgram=_raster([("<silent>", 6.0)]),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert result.done is False

    def test_a_weak_position_is_never_a_false(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A collapsed sequence is a reading about production, not a failure to perform the task."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial"] * 18, [0.2] * 17),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        assert result.done is True

    def test_a_weak_path_is_not_an_acceptance_gate(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The decode makes a tempting new ``false`` available and it must not be taken.

        A posteriorgram with no structure at all holds more mass outside the template's classes
        than inside any of them, so the path never leaves the filler and no repetition completes.
        The envelope's carrier still answers the conformance question, and the weak decode neither
        overrides it nor gates it.
        """
        noisy = np.full((400, len(PHONEME_LABELS)), 1.0 / len(PHONEME_LABELS))
        seed_ddk_store(
            store,
            duration_s=4.0,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(4.0, (1.0, 3.5), 5.0, 0.1),
            spans=[(1.0, 3.5)],
            posteriorgram=noisy,
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [repetitions] = _measurements(store, PPG_REPETITIONS)
        assert repetitions.attributes["value"] == 0
        assert repetitions.attributes["filler_fraction"] == pytest.approx(1.0)
        assert result.done is True

    def test_an_unmeasured_train_minimum_is_recorded_rather_than_raised(
        self, store: ProvStore, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """No carrier has two causes and only one of them is a reading of the recording."""
        override = tmp_path / "cleared.yaml"
        override.write_text("branch:\n  train_min_s: null\n")
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        result = _run(store, load_triage_config(override), tmp_path)
        assert result.done is UNDETERMINED
        assert "train_min_s" in _unmeasured(store)


class TestTheRegularityStatisticsAreParameterFree:
    """Applied now to the decode's own period series rather than to an onset series."""

    def test_dispersion_is_zero_for_a_perfectly_even_train(self) -> None:
        """The coefficient of variation of a constant series is zero."""
        assert dispersion([0.2] * 6) == pytest.approx(0.0)

    def test_dispersion_is_none_below_its_own_domain(self) -> None:
        """A sample deviation needs two values; one interval is not a dispersion."""
        assert dispersion([0.2]) is None
        assert dispersion([]) is None

    def test_the_trend_signs_festination_and_slowing(self) -> None:
        """Negative is festination and positive is slowing; no threshold names either."""
        festinating = trend([0.30, 0.28, 0.26, 0.24])
        slowing = trend([0.24, 0.26, 0.28, 0.30])
        assert festinating is not None and festinating < 0.0
        assert slowing is not None and slowing > 0.0

    def test_there_is_no_dispersion_by_position(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The decode measures one period per repetition, start to start, so nothing is split.

        The by-position split existed because envelope onsets land differently per place.
        """
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [spread] = _measurements(store, PPG_DISPERSION)
        assert "by_position" not in spread.attributes
        assert spread.attributes["support_intervals"] == 5

    def test_the_occupancy_is_a_duration_reading_per_position(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """How long the decode held each position, which is about the phones and not correctness."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [mass] = _measurements(store, PPG_MASS)
        consonant, vowel = mass.attributes["occupancy_s"]
        assert vowel > consonant > 0.0


class TestEveryReportKeyAReaderSelectsHasAWriter:
    """A key the table names with no writer is a claim about the graph that is not true."""

    def test_the_syllable_body_writes_every_ppg_key_the_table_names(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The fullest reading this instrument can take: both instruments, a contested word."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            envelope=_train_envelope(6.0, (0.2, 4.5), 5.0, 0.1),
            spans=[(0.2, 4.5)],
            posteriorgram=_buttercup_raster(6),
            words=[("buttercup", (0.5, 1.0))],
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        detail = _detail(result)
        named = [key for key in BRANCH_MEASURES["SPEECH"] if key.startswith("ppg_")]
        assert named
        assert [key for key in named if detail.get(key) is None] == []

    def test_the_detail_carries_exactly_the_keys_the_table_names(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A key the writer emits that no reader selects is invisible in the report."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert set(_detail(result)) <= set(BRANCH_MEASURES["SPEECH"])

    def test_the_removed_keys_are_named_by_neither_reader_nor_writer(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Twenty of the table's twenty-seven entries were this instrument's and are replaced."""
        removed = {
            "interval_dispersion",
            "interval_trend_s_per_step",
            "ppg_trains_n",
            "ppg_rate_hz",
            "ppg_jitter_over_median",
            "ppg_cv_units_n",
            "ppg_interval_trend_s_per_step",
            "ppg_cycles",
            "ppg_declared_cycles",
            "ppg_cycle_gap_cv",
            "ppg_cycle_consumed",
            "ppg_cycle_insertions_n",
            "ppg_cycle_nucleus_fraction",
            "ppg_place_agreement",
        }
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert removed & set(BRANCH_MEASURES["SPEECH"]) == set()
        assert removed & set(_detail(result)) == set()

    def test_the_report_names_one_realised_mass_number_per_template_position(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The first time the report says anything about which part of the sequence was realised."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            posteriorgram=_buttercup_raster(6),
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-buttercup"}))
        detail = _detail(result)
        assert detail["ppg_positions"] == list(BUTTERCUP)
        assert len(detail["ppg_realised_mass"]) == len(BUTTERCUP)

    def test_two_rates_are_reported_and_not_five(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The decoded syllable rate and the envelope modulation rate."""
        rate_keys = [key for key in BRANCH_MEASURES["SPEECH"] if key.endswith("_hz")]
        assert rate_keys == ["modulation_peak_hz", "ppg_syllable_rate_hz", "ppg_cycle_rate_hz"]


class TestTheDeclaredFamilySelectsTheBody:
    """The declared family picks which of SPEECH's bodies runs, and never supplies the answer."""

    def test_a_syllable_family_takes_the_syllable_body(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The in-family mode, over the row the declaration names."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        assert mode_of("SPEECH", store) == ("align", "diadochokinesis-pa")
        result = _run(store, ddk_config, tmp_path)
        assert result.done is True
        assert [branch for branch in BRANCHES if mode_of(branch, store)[0] == "align"] == ["SPEECH"]

    def test_another_branchs_family_takes_the_out_of_family_mode(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A declared cough is AIRWAY's row, so SPEECH detects rather than expecting."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-respiration-and-cough",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        assert mode_of("SPEECH", store) == ("detect", "respiration-and-cough")

    def test_the_hint_task_token_selects_the_syllable_body_over_the_path(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A recording whose stem says nothing, declared by its sidecar metadata instead."""
        seed_ddk_store(
            store,
            stem="recording",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        hint = AudioHints(metadata={"task_token": "diadochokinesis-pataka"})
        assert mode_of("SPEECH", store, hint) == ("align", "diadochokinesis-pataka")
        result = _run(store, ddk_config, tmp_path, hint)
        assert result.done is True


class TestThereIsNoSecondBranchForASyllableTask:
    """DDK dissolved into SPEECH, so the runner dispatches three branches and not four."""

    def test_the_runner_dispatches_the_three_branches_the_vocabulary_declares(self) -> None:
        """A fourth would be a branch with no node and no report section."""
        import inspect

        from senselab.audio.workflows.triage import run as run_module

        source = inspect.getsource(run_module._drive_branches)
        assert set(BRANCHES) == {"AIRWAY", "SPEECH", "VOICE"}
        assert "DDK" not in source

    def test_a_syllable_recording_is_routed_to_speech_and_to_no_second_branch(
        self, store: ProvStore, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The fold is handed one branch decision for a declared train, and it is SPEECH's."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa")
        assert [branch for branch in BRANCHES if mode_of(branch, store)[0] == "align"] == ["SPEECH"]

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


PA_TRAIN = (["labial"] * 8, [0.25] * 7)
"""Eight /pa/ onsets 0.25 s apart, covering roughly 0.2 s to 2.2 s."""

INSIDE = [("papapapa", (0.5, 1.0)), ("pataca", (1.2, 1.8)), ("[cough]", (0.8, 0.9))]
"""Two recogniser words over the syllable train, and a bracketed token that claims no word."""

OUTSIDE = ("hello", (3.0, 3.5))
"""A word the instrument's extent does not reach."""


def _contest_assertions(store: ProvStore) -> list[Entity]:
    """Every live contest against the recogniser's lexical claim.

    Args:
        store: The provenance store.

    Returns:
        The assertion entities, in write order.
    """
    return [
        entity
        for entity in live_entities(store, "assertion")
        if entity.attributes.get("verb") == "contest" and entity.attributes.get("claim") == TRANSCRIPT_CLAIM
    ]


class TestTheInstrumentIsTheAuthorityOverTheRecogniserText:
    """On a declared syllable task the instrument says what was produced and the ASR does not.

    Nothing here deletes, replaces or hides recogniser text: the safety constraint in
    ``specs/20260817-triage-workflow-dag/ddk-instrument-over-asr.md`` forbids narrowing what the
    PII scan is given, and the interlock that measures that is in ``pii_interlock_test.py``.
    """

    def test_the_instruments_reading_is_recorded_and_says_it_is_the_authority(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Without this record the two readings sit side by side and neither claims the task."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [reading] = _measurements(store, INSTRUMENT_READING)
        assert reading.attributes["authority"] == CV_AUTHORITY
        assert reading.attributes["supersedes"] == TRANSCRIPT_CLAIM
        assert reading.attributes["repetitions"] == 8
        assert len(reading.attributes["onsets_s"]) == 8

    def test_its_value_is_the_realised_class_series_per_repetition(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """One inner list per repetition, one number per template position, and no word text."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [reading] = _measurements(store, INSTRUMENT_READING)
        value = reading.attributes["value"]
        assert len(value) == 8
        assert all(len(row) == 2 for row in value)
        assert reading.attributes["classes"] == ["labial", "open"]

    def test_the_reading_is_taken_over_the_extent_the_instrument_covers(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The decoded repetition span, so nothing is claimed where the decode read nothing."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        [reading] = _measurements(store, INSTRUMENT_READING)
        assert reading.extent is not None
        assert reading.extent[0] == pytest.approx(0.2, abs=0.05)
        assert reading.extent[1] < OUTSIDE[1][0]

    def test_every_lexical_word_inside_that_extent_is_contested(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The half that makes the reading legible: which recogniser words it contradicts."""
        ids = seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        contested = _contest_assertions(store)
        assert [entity.attributes["reason"] for entity in contested] == [CONTRADICTED, CONTRADICTED]
        assert [entity.attributes["authority"] for entity in contested] == [CV_AUTHORITY, CV_AUTHORITY]
        assert [entity.attributes["index"] for entity in contested] == [0, 1]
        assert [store.derived_from(entity.id)[0] for entity in contested] == [ids["word-0"], ids["word-1"]]

    def test_a_bracketed_token_inside_the_extent_is_not_contested(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """``[cough]`` never claimed a word was said, so the instrument contradicts nothing."""
        ids = seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        derived = {store.derived_from(entity.id)[0] for entity in _contest_assertions(store)}
        assert ids["word-2"] not in derived

    def test_a_word_outside_the_extent_is_left_alone(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The instrument read nothing there, so it is the authority on nothing there."""
        ids = seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        derived = {store.derived_from(entity.id)[0] for entity in _contest_assertions(store)}
        assert ids["word-3"] not in derived
        [reading] = _measurements(store, INSTRUMENT_READING)
        assert reading.attributes["contradicted_words_n"] == 2

    def test_the_contested_words_survive_verbatim_and_stay_live(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Mark, do not delete. A word this pass retired would be a word the PII scan lost."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        live = live_entities(store, "word")
        assert [str(word.attributes["text"]) for word in live] == [text for text, _ in [*INSIDE, OUTSIDE]]

    def test_no_contest_carries_the_words_text(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A second copy of possibly-identifying text is a second thing redaction must cover."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        blob = " ".join(str(entity.attributes) for entity in _contest_assertions(store))
        blob += " ".join(str(entity.attributes) for entity in _measurements(store, INSTRUMENT_READING))
        for text, _ in [*INSIDE, OUTSIDE]:
            assert text not in blob

    def test_the_contradiction_is_not_a_deviation_and_reaches_no_verdict_fold(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The instrument disagreeing with a recogniser is not a departure by the participant."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        subjects = {word.id for word in lexical_words(store)}
        about_a_word = [
            finding for finding in result.deviations if subjects & set(finding.derived_from) and finding.start
        ]
        kinds = {finding.kind for finding in about_a_word}
        assert "contest" in kinds
        assert "deviation" not in kinds
        assert CONTRADICTED not in deviation_names(result.deviations)
        assert result.done is True

    def test_an_instrument_that_completed_no_repetition_claims_nothing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """No evidence the task was performed is no authority over anything the recogniser said."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_raster([("<silent>", 0.2), ("p", 0.05), ("aa", 0.3), ("<silent>", 0.4)]),
            words=[("pa", (0.3, 0.5))],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pataka"}))
        [repetitions] = _measurements(store, PPG_REPETITIONS)
        assert repetitions.attributes["value"] == 0
        assert _measurements(store, INSTRUMENT_READING) == []
        assert _contest_assertions(store) == []

    def test_the_branch_report_carries_how_many_words_the_instrument_contradicted(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The one consumer that gains from this; ``BRANCH_MEASURES`` names the key."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        result = _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        assert _detail(result)["ppg_contradicted_words_n"] == 2
        assert "ppg_contradicted_words_n" in BRANCH_MEASURES["SPEECH"]


class TestOnlyADeclaredSyllableTaskMakesTheInstrumentTheAuthority:
    """Authority comes from the declaration, never from a train the decode happened to find."""

    def test_an_undeclared_recording_records_no_reading_and_contests_nothing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The same syllable raster under no declaration: ``detect_speech``, and no decode."""
        seed_ddk_store(store, stem="recording", posteriorgram=_cv_raster(*PA_TRAIN), words=[*INSIDE, OUTSIDE])
        _run(store, ddk_config, tmp_path)
        assert _measurements(store, INSTRUMENT_READING) == []
        assert _contest_assertions(store) == []

    def test_a_declared_lexical_task_over_the_same_raster_is_untouched(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A train found incidentally on connected speech overrides no recogniser."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-free-speech",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "free-speech"}))
        assert _measurements(store, INSTRUMENT_READING) == []
        assert _contest_assertions(store) == []

    def test_the_lexical_count_the_routing_gate_reads_is_the_same_on_both(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """``speech.lexical`` is not discounted: the count the gate reads is untouched either way."""
        undeclared = ProvStore(run_id="test-run-undeclared")
        for target, stem in ((store, "sub-a_ses-1_task-diadochokinesis-pa"), (undeclared, "recording")):
            seed_ddk_store(target, stem=stem, posteriorgram=_cv_raster(*PA_TRAIN), words=[*INSIDE, OUTSIDE])
        _run(store, ddk_config, tmp_path, AudioHints(metadata={"task_token": "diadochokinesis-pa"}))
        _run(undeclared, ddk_config, tmp_path)
        assert len(_contest_assertions(store)) == 2
        assert _contest_assertions(undeclared) == []
        assert len(lexical_words(store)) == len(lexical_words(undeclared)) == 3
