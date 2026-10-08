"""The syllable-train task: the extent it proposes and the ten SPEECH tasks it evaluates.

Not a branch. Every test here drives the product path SPEECH takes — ``dispatch`` over
``align_speech``, which serves the ten ``diadochokinesis-*`` rows through ``align_ddk``. The task
events are the task layer's (``ddk_task``) over a seeded background view; the posteriorgram scores
their identity. The task layer's own units are in ``ddk_task_test.py``.
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
from senselab.audio.workflows.triage.ddk_task import DDK_READING
from senselab.audio.workflows.triage.nodes.branches import (
    BUTTERCUP,
    DEVIATION_TYPES,
    PARAM_KEYS,
    PATAKA,
    SPEECH_EXPECTATIONS,
    UNDETERMINED,
    UNMEASURED_POINTS,
    BranchParams,
    CountUnit,
    RequiredCount,
    Result,
    TypicalCount,
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
    CYCLE_RATE,
    CYCLES_OR_SYLLABLES_PER_S,
    EVENTS_FOUND,
    IDENTITY,
    INSTRUMENT_READING,
    NO_ENVELOPE,
    NO_PPG,
    PERIOD_CV,
    POSITION_MASS,
    RATE,
    SYLLABLE_RATE,
    SYLLABLES_PER_S,
    TASK_FROM_EVENTS,
    TRANSCRIPT_CLAIM,
    Posteriorgram,
    class_masses,
    phoneme_classes,
    position_masses,
    read_ddk,
    syllable_detail,
)
from senselab.audio.workflows.triage.nodes.gates import (
    CONFORMANCE_GATES,
    GATE_SPECS,
    INSTRUCTED_COUNT_FRACTION,
    REQUIRED_COUNT,
    Pattern,
)
from senselab.audio.workflows.triage.nodes.speech import align_speech, detect_speech
from senselab.audio.workflows.triage.vocabulary import (
    BRANCHES,
    BranchDecision,
    RunState,
    fold_file_verdict,
)
from senselab.utils.prov_store import Entity, ProvStore
from tests.audio.workflows.triage.nodes.conftest import gated_from_store, level_of_raster, seed_background_view

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
        The raster, with half a second of silence after the last syllable.
    """
    sequence: list[tuple[str, float]] = [("<silent>", lead_s)]
    for index, place in enumerate(places):
        gap = intervals[index] if index < len(intervals) else stop_s + 0.2
        sequence.append((STOP_OF[place], stop_s))
        sequence.append(("aa", max(FRAME_S, gap - stop_s)))
    sequence.append(("<silent>", 0.5))
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
    pause_s: float = 0.3,
) -> np.ndarray:
    """A raster of ``buttercup`` repetitions, each word followed by a pause.

    Args:
        repeats: How many times the word is said.
        nuclei: The three nuclei, so a test can substitute one without touching the onsets.
        coda: Whether the final /p/ of "cup" is realised.
        lead_s: Silence before the first onset.
        flap: What the intervocalic /t/ of "butter" surfaces as, which ARPAbet-40 spells three ways.
        pause_s: The silence after each word.

    Returns:
        The raster.
    """
    sequence: list[tuple[str, float]] = [("<silent>", lead_s)]
    for _ in range(repeats):
        for stop, nucleus in zip(("b", flap, "k"), nuclei):
            sequence.extend([(stop, 0.05), (nucleus, 0.15)])
        if coda:
            sequence.append(("p", 0.05))
        sequence.append(("<silent>", pause_s))
    return _raster(sequence)


@pytest.fixture
def ddk_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with every ``branch`` key the syllable body's envelope instrument reads.

    ``smoothing_window_s`` is 0.011 rather than 0.010 for the reason ``branches_test`` records: an
    even width in samples made ``boxcar`` a half-sample shift.
    """
    override = tmp_path / "ddk.yaml"
    override.write_text(
        "branch:\n"
        "  modulation_band_hz: [2.0, 12.0]\n"
        "  smoothing_window_s: 0.011\n"
        "  peak_prominence_db: 6.0\n"
        "  trough_return_db: 3.0\n"
        "  event_min_s: 0.02\n"
        "verdict:\n"
        "  gates:\n"
        "    by_group:\n"
        "      SYLLABLE_TRAIN:\n"
        "        train_min_s: 1.5\n"
        "        rate_prominence_min: 2.0\n"
        "      SYLLABLE_SEQUENCE:\n"
        "        train_min_s: 1.5\n"
        "        rate_prominence_min: 2.0\n"
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
    """Write the store surface the syllable body reads: streams, envelope, spans, words, posteriorgram, view.

    Every argument defaults to writing nothing for that derivative, which is how a test sets up an
    absence. A seeded posteriorgram brings a background view whose level follows its phonemes unless
    ``background`` is False.
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
        background: bool = True,
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
            if background:
                level = level_of_raster(posteriorgram)
                frames = int(round(duration_s / FRAME_S))
                level = np.concatenate([level, np.zeros(max(0, frames - len(level)))])[:frames]
                ids["background_model"] = seed_background_view(store, tmp_path, level)
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


def _gated(store: ProvStore, config: TriageConfig, family: str) -> Any:  # noqa: ANN401 — a Conformance
    """What VERDICT's gates make of the readings the syllable body left in the store."""
    return gated_from_store(store, SPEECH_EXPECTATIONS[family].pattern, settings=config)


def _detail(result: Result) -> dict[str, Any]:
    """The fields SPEECH's report carries for a syllable task."""
    return syllable_detail(result)


def _unmeasured(store: ProvStore) -> list[str]:
    """The operating points a body asked for and the configuration did not carry."""
    found = _measurements(store, UNMEASURED_POINTS)
    return [str(key) for entity in found for key in entity.attributes.get("value") or ()]


def _reading(store: ProvStore) -> dict[str, Any]:
    """The task layer's reading SPEECH wrote."""
    [reading] = _measurements(store, DDK_READING)
    return dict(reading.attributes)


PA = AudioHints(metadata={"task_token": "diadochokinesis-pa"})
PATAKA_HINT = AudioHints(metadata={"task_token": "diadochokinesis-pataka"})
BUTTERCUP_HINT = AudioHints(metadata={"task_token": "diadochokinesis-buttercup"})


class TestTheTaskLayerDecidesTheSyllableTask:
    """SPEECH writes the task layer's reading; VERDICT decides on it and on nothing the gates read."""

    def test_a_clean_pa_train_reads_present_with_one_event_per_syllable(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Eight /pa/ syllables: eight events, the target's identity, present."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        _run(store, ddk_config, tmp_path, PA)
        reading = _reading(store)
        assert reading["decision"] == "present"
        assert reading["unit"] == "syllable"
        assert reading["events_n"] == 8
        assert reading["identity"] > 0.9

    def test_a_pataka_train_reads_cycles(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Eighteen syllables of a three-syllable template are six cycle events."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, PATAKA_HINT)
        reading = _reading(store)
        assert reading["unit"] == "cycle"
        assert reading["events_n"] == 6
        assert reading["decision"] == "present"

    def test_buttercup_words_with_pauses_are_one_cycle_each(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Six words, each followed by a pause, are six cycles from the first to the last."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-buttercup", posteriorgram=_buttercup_raster(6))
        _run(store, ddk_config, tmp_path, BUTTERCUP_HINT)
        reading = _reading(store)
        assert reading["events_n"] == 6
        [span] = _spans_of(store)
        assert span.extent is not None
        assert span.extent[0] == pytest.approx(0.2, abs=0.03)

    def test_silence_is_no_syllable_train(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Nothing over the floor: absent, and no extent."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_raster([("<silent>", 6.0)]))
        _run(store, ddk_config, tmp_path, PA)
        assert _reading(store)["decision"] == "absent"
        assert _spans_of(store) == []

    def test_a_train_of_the_wrong_syllable_is_not_the_target(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """/ta/ said for /pa/ stands over the floor and reads as another activity on its consonant."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_raster(
                [("<silent>", 0.2), *[(p, d) for _ in range(8) for p, d in (("s", 0.05), ("iy", 0.2))]]
            ),
        )
        _run(store, ddk_config, tmp_path, PA)
        reading = _reading(store)
        assert reading["decision"] == "absent"
        assert reading["why"] == "not the target"

    def test_the_syllable_patterns_have_no_conformance_gate(self) -> None:
        """``repetitions_min`` and ``instructed_count_min_fraction`` no longer decide a syllable task."""
        assert CONFORMANCE_GATES[Pattern.SYLLABLE_TRAIN] == ()
        assert CONFORMANCE_GATES[Pattern.SYLLABLE_SEQUENCE] == ()
        assert "repetitions_min" not in GATE_SPECS

    def test_the_gates_answer_undetermined_and_leave_the_decision_to_the_reading(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A group with no conformance gate answers ``UNDETERMINED``, whatever was read."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        _run(store, ddk_config, tmp_path, PA)
        assert _gated(store, ddk_config, "diadochokinesis-pa") == UNDETERMINED


class TestTheInstructedCountIsAnAnnotation:
    """The v1 instructions say ``10 times``: what was produced is written beside it and decides nothing."""

    def test_the_count_is_written_against_the_instructed_one(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Found against required, in the unit the declaration names; no fraction is gated on."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        _run(store, ddk_config, tmp_path, PA)
        [counts] = _measurements(store, "counts")
        assert counts.attributes["entries"][REQUIRED_COUNT] == {"found": 8, "required": 10, "unit": "repetitions"}
        assert _measurements(store, INSTRUCTED_COUNT_FRACTION) == []
        assert _reading(store)["annotations"]["count_fraction"] == pytest.approx(0.8)

    def test_a_count_well_short_of_the_instruction_still_reads_present(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Two buttercups of ten is a short train, not an absent one, and no deviation."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-buttercup", posteriorgram=_buttercup_raster(2))
        result = _run(store, ddk_config, tmp_path, BUTTERCUP_HINT)
        reading = _reading(store)
        assert reading["events_n"] == 2
        assert reading["decision"] == "present"
        assert deviation_names(result.deviations) == ()
        [found] = _measurements(store, EVENTS_FOUND)
        assert found.attributes["value"] == 2
        assert found.attributes["required"] == 10

    def test_the_row_says_which_kind_of_count_it_carries(self) -> None:
        """A v1 syllable-repetition row carries the instruction's count; the timed v2 rows carry none."""
        row = SPEECH_EXPECTATIONS["diadochokinesis-pataka"]
        assert row.required_count == RequiredCount(10, CountUnit.REPETITIONS)
        assert row.typical_count is None
        assert SPEECH_EXPECTATIONS["diadochokinesis-v2-puhtuhkuh"].required_count is None
        assert not hasattr(row, "expected_event_count")

    def test_a_typical_count_cannot_be_declared_without_citing_its_measurement(self) -> None:
        """A measured number whose measurement is unrecorded is the defect this shape forbids."""
        with pytest.raises(ValueError, match="cites the measurement"):
            TypicalCount(10, CountUnit.REPETITIONS, "  ")


class TestTheTaskExtentIsTheTaskEventsSpan:
    """Exactly one span: the first task event to the last, with nothing added."""

    def test_the_extent_runs_from_the_first_event_to_the_last(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The only span this family proposes, and the task events mint it."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, PATAKA_HINT)
        [span] = _spans_of(store)
        assert span.attributes["role"] == "task_extent"
        assert span.attributes["production"] == TASK_FROM_EVENTS
        assert span.attributes["events_n"] == 6
        assert span.extent is not None
        assert span.extent[0] == pytest.approx(0.2, abs=0.03)
        assert span.extent[1] == pytest.approx(0.2 + 17 * 0.2 + 0.25, abs=0.05)

    def test_it_names_the_background_and_the_posteriorgram_it_was_read_off(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Under propose-only the derivation is the whole record of where the extent came from."""
        ids = seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, PATAKA_HINT)
        [span] = _spans_of(store)
        assert store.derived_from(span.id) == [ids["background_model"], ids["ppg_posteriorgram"]]

    def test_a_carrier_no_task_event_stands_on_proposes_no_task_extent(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """An amplitude carrier is not a task event, so it cannot say where the task was."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        assert _spans_of(store) == []

    def test_an_absent_background_reads_as_an_absent_input_not_as_found_nothing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The reading names the missing view and takes no decision."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7),
            background=False,
        )
        _run(store, ddk_config, tmp_path, PA)
        reading = _reading(store)
        assert reading["decision"] is None
        assert reading["absent"] == ["background_model"]
        assert _spans_of(store) == []

    def test_the_span_is_minted_into_speechs_own_family_like_every_other_role(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """DDK dissolved into SPEECH, so a ``ddk`` family would be a family no reader knows."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, PATAKA_HINT)
        assert _spans_of(store, "ddk") == []
        assert {str(span.attributes["role"]) for span in _spans_of(store)} == {"task_extent"}

    def test_a_task_extent_touching_the_recordings_edge_is_a_truncation(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The deviation keys on the span that ships."""
        raster = _cv_raster(["labial"] * 8, [0.25] * 7, lead_s=0.0)[:-50]
        seed_ddk_store(
            store,
            duration_s=raster.shape[0] * FRAME_S - 0.001,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=raster,
        )
        result = _run(store, ddk_config, tmp_path, PA)
        assert "truncation" in deviation_names(result.deviations)


class TestTheEnvelopeChannelIsTheModulationSpectrumAndNothingElse:
    """The modulation-rate channel is read beside the task events and decides nothing."""

    def test_the_rate_is_the_modulation_peak_and_states_its_unit(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A 5 Hz train reads 5 Hz, in syllables per second because the family repeats one syllable."""
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

    def test_the_task_events_two_rates_travel_beside_the_peak(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The reader can see which of the two the peak matched."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, PATAKA_HINT)
        [rate] = _measurements(store, RATE)
        assert rate.attributes["event_syllable_rate_hz"] > rate.attributes["event_cycle_rate_hz"] > 0.0

    def test_a_carrier_with_no_readable_modulation_is_no_carrier(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Envelope-first, then qualified by modulation: a flat carrier holds no rate."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_flat_envelope(6.0, [(1.0, 5.0)]),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path)
        assert _measurements(store, RATE) == []

    def test_the_train_fraction_is_taken_over_the_extent_that_ships(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Keyed on the surviving span, so it cannot describe an extent no reader ever sees."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(["labial"] * 17, [0.25] * 16, lead_s=1.0),
        )
        _run(store, ddk_config, tmp_path, PA)
        [span] = _spans_of(store)
        assert span.extent is not None
        [fraction] = _measurements(store, "train_fraction_of_recording")
        assert fraction.attributes["train_s"] == pytest.approx(span.extent[1] - span.extent[0], abs=0.001)
        assert fraction.attributes["value"] == pytest.approx(fraction.attributes["train_s"] / 6.0, abs=0.001)


class TestThePosteriorgramScoresAndNoLongerLocates:
    """The cyclic decode is gone; the identity alignment reads each task event's realised mass."""

    def test_the_decode_and_its_operating_point_are_gone(self, ddk_config: TriageConfig) -> None:
        """No cyclic topology, no chain length, and ``burst_window_ms`` is no config key."""
        from senselab.audio.workflows.triage.nodes import ddk  # noqa: PLC0415

        for name in ("decode_template", "repetitions_of", "arcs", "viterbi", "Decode", "min_phone_frames"):
            assert not hasattr(ddk, name)
        assert "burst_window_ms" not in PARAM_KEYS
        with pytest.raises(UnknownConfigKey):
            ddk_config.require("branch.burst_window_ms")

    def test_a_collapsed_sequence_reports_mass_per_position_rather_than_a_deviation(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """/papapa/ said for /pataka/: the alveolar and velar positions read nothing."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial"] * 18, [0.2] * 17),
        )
        result = _run(store, ddk_config, tmp_path, PATAKA_HINT)
        assert deviation_names(result.deviations) == ()
        assert "syllable_sequence_mismatch" not in DEVIATION_TYPES
        [mass] = _measurements(store, POSITION_MASS)
        assert mass.attributes["value"][0] > 0.9
        assert mass.attributes["value"][2] == pytest.approx(0.0, abs=0.01)
        assert mass.attributes["positions"] == list(PATAKA)

    @pytest.mark.parametrize("flap", ["t", "d", "r"])
    def test_the_alveolar_class_admits_all_three_spellings_of_butters_flap(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any], flap: str
    ) -> None:
        """ARPAbet-40 has no flap symbol, so /ɾ/ surfaces as ``t``, ``d`` or ``r`` — one segment."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-buttercup", posteriorgram=_buttercup_raster(6, flap=flap)
        )
        _run(store, ddk_config, tmp_path, BUTTERCUP_HINT)
        [mass] = _measurements(store, POSITION_MASS)
        assert mass.attributes["value"][2] > 0.9

    def test_a_substituted_nucleus_reads_low_at_its_own_position(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A "buttercap" reads the rhotic position low and the open ones high."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            posteriorgram=_buttercup_raster(6, nuclei=("ah", "ah", "ah")),
        )
        _run(store, ddk_config, tmp_path, BUTTERCUP_HINT)
        [mass] = _measurements(store, POSITION_MASS)
        assert mass.attributes["value"][3] == pytest.approx(0.0, abs=0.01)
        assert mass.attributes["value"][1] > 0.8

    def test_the_rate_and_regularity_are_the_task_events(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The syllable rate, the cycle rate and the period's dispersion, each an annotation."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17),
        )
        _run(store, ddk_config, tmp_path, PATAKA_HINT)
        [syllables] = _measurements(store, SYLLABLE_RATE)
        [cycles] = _measurements(store, CYCLE_RATE)
        [spread] = _measurements(store, PERIOD_CV)
        assert syllables.attributes["value"] == pytest.approx(5.0, abs=0.5)
        assert cycles.attributes["value"] == pytest.approx(5.0 / 3, abs=0.2)
        assert spread.attributes["value"] == pytest.approx(0.0, abs=0.1)
        [identity] = _measurements(store, IDENTITY)
        assert identity.attributes["value"] > 0.9


class TestAnAbsentInstrumentIsNotANegativeReading:
    """Three states stay distinct and each keeps its own record."""

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

    def test_an_absent_posteriorgram_leaves_the_reading_untaken(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The identity has no reading, and the reading names what it lacked."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            envelope=_train_envelope(6.0, (1.0, 5.0), 5.0, 0.1),
            spans=[(1.0, 5.0)],
        )
        _run(store, ddk_config, tmp_path, PA)
        [identity] = _measurements(store, IDENTITY)
        assert identity.attributes["value"] is None
        assert identity.attributes["unavailable"] == "ppg_posteriorgram"
        reading = _reading(store)
        assert reading["decision"] is None
        assert "ppg_posteriorgram" in reading["absent"]
        assert read_ddk(store, tmp_path, "plain").ppg is None, NO_PPG

    def test_a_silent_recording_is_a_reading_and_not_an_absence(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """This is a reading, and it is the only one of the three that can support a discard."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_raster([("<silent>", 6.0)]))
        _run(store, ddk_config, tmp_path, PA)
        reading = _reading(store)
        assert reading["decision"] == "absent"
        assert reading["absent"] == []
        [found] = _measurements(store, EVENTS_FOUND)
        assert found.attributes["value"] == 0

    def test_an_unmeasured_class_vocabulary_leaves_the_reading_untaken_and_names_the_key(
        self, store: ProvStore, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A branch does not refuse over an unmeasured point; the reading names it absent."""
        override = tmp_path / "cleared.yaml"
        override.write_text("branch:\n  phoneme_place_classes: null\n")
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        _run(store, load_triage_config(override), tmp_path, PA)
        reading = _reading(store)
        assert reading["decision"] is None
        assert reading["absent"] == ["branch.phoneme_place_classes"]
        assert "phoneme_place_classes" in _unmeasured(store)


class TestEveryReportKeyAReaderSelectsHasAWriter:
    """A key the table names with no writer is a claim about the graph that is not true."""

    def test_the_syllable_body_writes_every_ddk_key_the_table_names(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The fullest reading: both instruments, a counted family, a contested word."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-buttercup",
            envelope=_train_envelope(6.0, (0.2, 4.5), 5.0, 0.1),
            spans=[(0.2, 4.5)],
            posteriorgram=_buttercup_raster(6),
            words=[("buttercup", (0.5, 1.0))],
        )
        result = _run(store, ddk_config, tmp_path, BUTTERCUP_HINT)
        detail = _detail(result)
        named = [key for key in BRANCH_MEASURES["SPEECH"] if key.startswith("ddk_")]
        assert named
        assert [key for key in named if detail.get(key) is None] == []

    def test_the_detail_carries_exactly_the_keys_the_table_names(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A key the writer emits that no reader selects is invisible in the report."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        result = _run(store, ddk_config, tmp_path, PA)
        assert set(_detail(result)) <= set(BRANCH_MEASURES["SPEECH"])
        assert not [key for key in BRANCH_MEASURES["SPEECH"] if key.startswith("ppg_")]

    def test_the_report_names_one_realised_mass_number_per_template_position(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Which part of the sequence was realised."""
        seed_ddk_store(store, stem="sub-a_ses-1_task-diadochokinesis-buttercup", posteriorgram=_buttercup_raster(6))
        result = _run(store, ddk_config, tmp_path, BUTTERCUP_HINT)
        detail = _detail(result)
        assert detail["ddk_positions"] == list(BUTTERCUP)
        assert len(detail["ddk_realised_mass"]) == len(BUTTERCUP)


class TestTheDeclaredFamilySelectsTheBody:
    """The declared family picks which of SPEECH's bodies runs, and never supplies the answer."""

    def test_a_syllable_family_takes_the_syllable_body(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The in-family mode, over the row the declaration names."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-diadochokinesis-pa", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        assert mode_of("SPEECH", store) == ("align", "diadochokinesis-pa")
        _run(store, ddk_config, tmp_path)
        assert _reading(store)["decision"] == "present"
        assert [branch for branch in BRANCHES if mode_of(branch, store)[0] == "align"] == ["SPEECH"]

    def test_another_branchs_family_takes_the_out_of_family_mode(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A declared cough is AIRWAY's row, so SPEECH detects rather than expecting."""
        seed_ddk_store(
            store, stem="sub-a_ses-1_task-respiration-and-cough", posteriorgram=_cv_raster(["labial"] * 8, [0.25] * 7)
        )
        assert mode_of("SPEECH", store) == ("detect", "respiration-and-cough")
        _run(store, ddk_config, tmp_path)
        assert _measurements(store, DDK_READING) == []

    def test_the_hint_task_token_selects_the_syllable_body_over_the_path(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """A recording whose stem says nothing, declared by its sidecar metadata instead."""
        seed_ddk_store(
            store, stem="recording", posteriorgram=_cv_raster(["labial", "alveolar", "velar"] * 6, [0.2] * 17)
        )
        assert mode_of("SPEECH", store, PATAKA_HINT) == ("align", "diadochokinesis-pataka")
        _run(store, ddk_config, tmp_path, PATAKA_HINT)
        assert _reading(store)["unit"] == "cycle"


PA_TRAIN = (["labial"] * 8, [0.25] * 7)
"""Eight /pa/ onsets 0.25 s apart, covering roughly 0.2 s to 2.2 s."""

INSIDE = [("papapapa", (0.5, 1.0)), ("pataca", (1.2, 1.8)), ("[cough]", (0.8, 0.9))]
"""Two recogniser words over the syllable train, and a bracketed token that claims no word."""

OUTSIDE = ("hello", (3.0, 3.5))
"""A word the task extent does not reach."""


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
    """On a declared syllable task the task events say what was produced and the ASR does not.

    Nothing here deletes, replaces or hides recogniser text; the interlock that measures that is in
    ``pii_interlock_test.py``.
    """

    def test_the_reading_is_recorded_and_says_it_is_the_authority(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Without this record the two readings sit side by side and neither claims the task."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, PA)
        [reading] = _measurements(store, INSTRUMENT_READING)
        assert reading.attributes["authority"] == CV_AUTHORITY
        assert reading.attributes["supersedes"] == TRANSCRIPT_CLAIM
        assert reading.attributes["events_n"] == 8
        assert len(reading.attributes["onsets_s"]) == 8

    def test_its_value_is_the_realised_class_series_per_event(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """One inner list per task event, one number per template position, and no word text."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, PA)
        [reading] = _measurements(store, INSTRUMENT_READING)
        value = reading.attributes["value"]
        assert len(value) == 8
        assert all(len(row) == 2 for row in value)
        assert reading.attributes["classes"] == ["labial", "open"]

    def test_the_reading_is_taken_over_the_task_extent(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Nothing is claimed where no task event stands."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, PA)
        [reading] = _measurements(store, INSTRUMENT_READING)
        assert reading.extent is not None
        assert reading.extent[0] == pytest.approx(0.2, abs=0.05)
        assert reading.extent[1] < OUTSIDE[1][0]

    def test_every_lexical_word_inside_that_extent_is_contested(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Which recogniser words it contradicts; a bracketed token and an outside word are left alone."""
        ids = seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, PA)
        contested = _contest_assertions(store)
        assert [entity.attributes["reason"] for entity in contested] == [CONTRADICTED, CONTRADICTED]
        assert [entity.attributes["index"] for entity in contested] == [0, 1]
        derived = {store.derived_from(entity.id)[0] for entity in contested}
        assert derived == {ids["word-0"], ids["word-1"]}
        [reading] = _measurements(store, INSTRUMENT_READING)
        assert reading.attributes["contradicted_words_n"] == 2

    def test_the_contested_words_survive_verbatim_and_no_contest_carries_their_text(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """Mark, do not delete, and make no second copy of possibly-identifying text."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        _run(store, ddk_config, tmp_path, PA)
        live = live_entities(store, "word")
        assert [str(word.attributes["text"]) for word in live] == [text for text, _ in [*INSIDE, OUTSIDE]]
        blob = " ".join(str(entity.attributes) for entity in _contest_assertions(store))
        blob += " ".join(str(entity.attributes) for entity in _measurements(store, INSTRUMENT_READING))
        for text, _ in [*INSIDE, OUTSIDE]:
            assert text not in blob

    def test_the_contradiction_is_not_a_deviation(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """The instrument disagreeing with a recogniser is not a departure by the participant."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        result = _run(store, ddk_config, tmp_path, PA)
        subjects = {word.id for word in lexical_words(store)}
        kinds = {
            finding.kind for finding in result.deviations if subjects & set(finding.derived_from) and finding.start
        }
        assert "contest" in kinds
        assert "deviation" not in kinds

    def test_no_task_event_claims_nothing(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """No evidence the task was performed is no authority over anything the recogniser said."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pataka",
            posteriorgram=_raster([("<silent>", 2.0)]),
            words=[("pa", (0.3, 0.5))],
        )
        _run(store, ddk_config, tmp_path, PATAKA_HINT)
        assert _measurements(store, INSTRUMENT_READING) == []
        assert _contest_assertions(store) == []

    def test_the_branch_report_carries_how_many_words_the_instrument_contradicted(
        self, store: ProvStore, ddk_config: TriageConfig, tmp_path: Path, seed_ddk_store: Callable[..., Any]
    ) -> None:
        """``BRANCH_MEASURES`` names the key."""
        seed_ddk_store(
            store,
            stem="sub-a_ses-1_task-diadochokinesis-pa",
            posteriorgram=_cv_raster(*PA_TRAIN),
            words=[*INSIDE, OUTSIDE],
        )
        result = _run(store, ddk_config, tmp_path, PA)
        assert _detail(result)["ddk_contradicted_words_n"] == 2
        assert "ddk_contradicted_words_n" in BRANCH_MEASURES["SPEECH"]


class TestTheEmissionsAreOnePartitionOfThePhonemeInventory:
    """Every template position reads the mass of one part of one partition of the inventory."""

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
        """Silence is in the remainder; the identity alignment's fillers read it apart."""
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
        ppg = Posteriorgram(frames=_buttercup_raster(4), phonemes=PHONEME_LABELS, seconds_per_frame=FRAME_S)
        masses = position_masses(ppg, branch_params(ddk_config), BUTTERCUP)
        assert masses is not None
        assert masses.classes[1] != masses.classes[3]

    def test_the_open_class_holds_exactly_what_the_low_class_held_plus_two(self, ddk_config: TriageConfig) -> None:
        """The extension makes the inventory total; it does not move the DDK reading."""
        classes = branch_params(ddk_config).point("phoneme_vowel_classes")
        assert classes is not None
        assert set(classes["open"]) == {"aa", "ae", "ah", "ao", "aw", "ay"} | {"eh", "oy"}


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
        assert all(sequence is not None for sequence in firsts)
        sequences = [sequence for sequence in firsts if sequence is not None]
        assert {sequence[1] for sequence in sequences} == {"aa"}
        assert {sequence[0] for sequence in sequences} == {"p", "t", "k"}

    def test_a_row_survives_a_round_trip_through_its_mapping(self) -> None:
        """A run records which expectation it applied, and a phoneme tuple has to come back."""
        row = SPEECH_EXPECTATIONS["diadochokinesis-buttercup"]
        assert type(row).from_mapping(row.as_mapping()).sequence == BUTTERCUP


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
