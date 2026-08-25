"""VOICE — the phonation spans PREPROCESS detected, measured. There is no residual.

Praat is faked here: this module's subject is what VOICE does with a span it was handed, and the
phonation task's own tests own where Praat's refusals lie. The spans are seeded directly, so no test
here depends on ``propose_phonation_spans`` classifying anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import senselab.audio.workflows.triage.nodes.voice as voice_module
from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.phonation import PeriodMark
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import find_measurements
from senselab.audio.workflows.triage.nodes.voice import voice
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore
from tests.audio.workflows.triage.nodes.conftest import seed_preprocess_store

FAKE_F0 = 100.0
"""The fake point process's rate. Chosen against ``voice_config``'s range so neither doubling alias
lands inside it: 200 Hz is above its maximum and 50 Hz is below its minimum, which leaves the
period-doubling row inert unless a test asks for a range that makes it fire."""

_SEED_DURATION_S = 8.0


def _fake_hnr_track(
    audio: Audio, *, f0_min_hz: float, hop_s: float, silence_threshold: float, periods_per_window: float
) -> tuple[np.ndarray, np.ndarray]:
    """A constant 20 dB HNR track on the hop grid, spanning the audio."""
    n = int(round(audio.waveform.shape[-1] / audio.sampling_rate / hop_s))
    times = (np.arange(n) + 0.5) * hop_s
    return times, np.full(n, 20.0)


def _fake_f0_track(
    audio: Audio, *, f0_min_hz: float, f0_max_hz: float, hop_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A constant ``FAKE_F0`` track with constant strength, on the same hop grid."""
    n = int(round(audio.waveform.shape[-1] / audio.sampling_rate / hop_s))
    times = (np.arange(n) + 0.5) * hop_s
    return times, np.full(n, FAKE_F0), np.full(n, 0.9)


def _fake_period_marks(
    audio: Audio, start_s: float, end_s: float, *, f0_min_hz: float, f0_max_hz: float
) -> list[PeriodMark]:
    """Marks every ``1 / FAKE_F0`` seconds inside the queried extent."""
    period = 1.0 / FAKE_F0
    times = np.arange(start_s, end_s - period, period)
    return [PeriodMark(time_s=float(t), period_s=period, amplitude=0.1) for t in times]


@pytest.fixture(autouse=True)
def praat_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Praat is deterministic but slow; the phonation task's tests own the real calls."""
    monkeypatch.setattr(voice_module, "hnr_track", _fake_hnr_track)
    monkeypatch.setattr(voice_module, "f0_track", _fake_f0_track)
    monkeypatch.setattr(voice_module, "period_marks", _fake_period_marks)


@pytest.fixture
def voice_config(tmp_path: Path) -> TriageConfig:
    """The packaged configuration with one declared F0 range. A fixture, not a fit.

    The packaged ``voice.f0_range_hz`` is null because no single range serves both a low adult male
    fundamental and an infant voice; this states one so the branch can run.
    """
    return _override(tmp_path, "voice:\n  f0_range_hz: [75.0, 190.0]\n")


def _override(tmp_path: Path, text: str) -> TriageConfig:
    """The packaged configuration with one partial YAML deep-merged over it.

    Args:
        tmp_path: Where the override file is written.
        text: The partial YAML.

    Returns:
        The merged configuration.
    """
    path = tmp_path / "voice-override.yaml"
    path.write_text(text)
    return load_triage_config(path)


def _seed_voice_store(
    store: ProvStore,
    tmp_path: Path,
    *,
    phonation: list[tuple[Any, ...]],
    speech_spans: list[tuple[float, float]] | None = None,
    airway_labelled: list[tuple[float, float]] | None = None,
    hop_s: float = 0.01,
) -> None:
    """Seed the store VOICE reads: PREPROCESS's streams and phonation spans, plus other branches' work.

    The phonation spans go through the shared ``seed_preprocess_store``, so this module reads the one
    span schema PREPROCESS actually writes rather than a private copy of it.

    Args:
        store: The store to seed.
        tmp_path: Where the seeded stream and sidecars are written.
        phonation: ``[(start, end, production), ...]`` or ``[(start, end, production, member), ...]``.
        speech_spans: SPEECH's spans, written by a ``SPEECH`` activity. Present only so a test can
            show that they remove nothing.
        airway_labelled: Spans AIRWAY labelled, each with its ``label`` assertion. Same purpose.
        hop_s: The analysis grid PREPROCESS stated these spans on. VOICE reads the same grid from
            ``phonation_spans.hop_s``, so a value the packaged configuration does not declare would
            describe a store the configuration contradicts.

    Raises:
        ValueError: If ``hop_s`` is not the grid ``phonation_spans.hop_s`` declares.
    """
    declared_hop_s = load_triage_config().require("phonation_spans.hop_s")
    if hop_s != declared_hop_s:
        raise ValueError(
            f"hop_s {hop_s} is not phonation_spans.hop_s {declared_hop_s}; the store would be inconsistent"
        )
    seed_preprocess_store.__wrapped__(tmp_path)(store, duration_s=_SEED_DURATION_S, phonation=phonation)

    agent = store.agent(agent_type="software", version="senselab test-seed")
    if speech_spans:
        speech = store.activity(node="SPEECH", step="seed-voice", parameters={})
        store.was_associated_with(speech, agent)
        for start, end in speech_spans:
            span_id = store.entity(prov_type="span", extent=(start, end), attributes={"source": "words"})
            store.was_generated_by(span_id, speech)
    if airway_labelled:
        airway = store.activity(node="AIRWAY", step="seed-voice", parameters={})
        store.was_associated_with(airway, agent)
        for start, end in airway_labelled:
            span_id = store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"peak_over_floor_db": 30.0, "k_db": 18.0, "signal": "preemphasised", "merged_proposals": 1},
            )
            store.was_generated_by(span_id, airway)
            label_id = store.entity(
                prov_type="assertion", extent=(start, end), attributes={"verb": "label", "label": "Cough"}
            )
            store.was_generated_by(label_id, airway)
            store.was_derived_from(label_id, span_id)


def _stub_period_marks(monkeypatch: pytest.MonkeyPatch, *, marks: int) -> list[tuple[float, float]]:
    """Replace the point process with a recorder returning a fixed number of marks.

    Args:
        monkeypatch: The test's patcher.
        marks: How many marks each call returns.

    Returns:
        The list the recorder appends each call's ``(start_s, end_s)`` to.
    """
    calls: list[tuple[float, float]] = []

    def _marks(audio: Audio, start_s: float, end_s: float, *, f0_min_hz: float, f0_max_hz: float) -> list[PeriodMark]:
        calls.append((start_s, end_s))
        period = 1.0 / FAKE_F0
        return [PeriodMark(time_s=start_s + k * period, period_s=period, amplitude=0.1) for k in range(marks)]

    monkeypatch.setattr(voice_module, "period_marks", _marks)
    return calls


def _verdict_entity(store: ProvStore, node: str) -> Entity:
    """The verdict entity one node wrote."""
    return next(e for e in store.entities("verdict") if e.attributes.get("node") == node)


def _voice_spans(store: ProvStore) -> list[Entity]:
    """The spans VOICE itself wrote, in time order — told from PREPROCESS's by generating activity."""
    out = []
    for entity in store.entities("span"):
        activity_id = store.generated_by(entity.id)
        if activity_id is not None and store.get_activity(activity_id).node == "VOICE":
            out.append(entity)
    return sorted(out, key=lambda entity: entity.extent or (0.0, 0.0))


class TestTheSubjectIsPreprocessesSpans:
    """VOICE measures what PREPROCESS detected. Nothing is subtracted from anything."""

    def test_the_spans_are_preprocesses_phonation_spans(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """spans_n is the count of phonation spans in the store, not of a residual."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced"), (2.0, 2.8, "voiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["spans_n"] == 2

    def test_a_speech_span_removes_nothing(self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path) -> None:
        """branch-voice.md: 'Nothing another branch claimed is removed from this branch's subject'."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")], speech_spans=[(0.0, 1.5)])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["spans_n"] == 1

    def test_an_airway_label_removes_nothing(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Nothing this branch measures is conditioned on what another branch concluded."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")], airway_labelled=[(0.0, 1.5)])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["spans_n"] == 1

    def test_the_module_computes_no_residual(self) -> None:
        """The three residual helpers are deleted, not left unreachable."""
        for name in ("_subtract_intervals", "_airway_labelled", "_speech_spans"):
            assert not hasattr(voice_module, name)

    def test_no_phonation_span_fails(self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path) -> None:
        """This path is reached only when a hint forced the branch, which routing gates on the same fact."""
        _seed_voice_store(store, tmp_path, phonation=[])
        assert voice(store, "plain", voice_config, run_dir=tmp_path).verdict.outcome is Outcome.FAIL

    def test_the_kind_is_voice(self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path) -> None:
        """voice_no_words is gone; VERDICT joins branch to kind on this string."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        assert voice(store, "plain", voice_config, run_dir=tmp_path).verdict.kind == "voice"

    def test_the_packaged_config_refuses_before_the_store_is_touched(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """No range is declared by default, and the branch does not invent a population to serve."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        before = len(store.entities())
        with pytest.raises(ValueError, match="voice.f0_range_hz"):
            voice(store, "plain", config, run_dir=tmp_path)
        assert len(store.entities()) == before


class TestProductionModes:
    """Voiced, unvoiced and mixed are all measured; an unvoiced span is not a failure."""

    def test_an_unvoiced_span_is_measured(self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path) -> None:
        """A disordered voice sustaining without periodicity is exactly what must be measured."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "unvoiced")])
        result = voice(store, "plain", voice_config, run_dir=tmp_path)
        assert result.verdict.outcome is not Outcome.FAIL
        assert _verdict_entity(store, "VOICE").attributes["production"]["unvoiced"] == 1

    def test_an_unvoiced_span_carries_no_period_marks(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Absent, not zero and not interpolated: its duration, formants and level are its measurement."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "unvoiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        marks = find_measurements(store, "period_marks")
        assert marks and "n" not in marks[0].attributes
        assert marks[0].attributes["unmeasured"] == "unvoiced_span"

    def test_the_production_counts_are_reported(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The verdict's production block is a count per mode, as branch-voice.md's product names it."""
        _seed_voice_store(
            store,
            tmp_path,
            phonation=[(0.0, 1.0, "voiced"), (2.0, 3.0, "unvoiced"), (4.0, 5.0, "mixed")],
        )
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["production"] == {"voiced": 1, "unvoiced": 1, "mixed": 1}


class TestMptRecoverableProducts:
    """longest_span_s and its criterion, so a task measurement is not reassembled from fragments."""

    def test_longest_span_s_is_a_first_class_product(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The longest span's duration, reported directly."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.0, "voiced"), (2.0, 5.5, "voiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["longest_span_s"] == pytest.approx(3.5)

    def test_the_criterion_that_closed_it_travels_with_it(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A duration without its offset criterion is not a maximum phonation time."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 3.5, "voiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["longest_span_criterion"] == "f0_stability"

    def test_phonation_s_totals_every_span(self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path) -> None:
        """The total is over the spans, whatever their production mode."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.0, "voiced"), (2.0, 2.5, "unvoiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["phonation_s"] == pytest.approx(1.5)

    def test_a_declared_task_outside_its_range_flags_with_the_range_named(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """The task conditions how a duration is reported, never whether a span exists."""
        config = _override(
            tmp_path,
            "voice:\n  f0_range_hz: [75, 500]\n  task_duration_ranges: {maximum_phonation_time: [10.0, 40.0]}\n",
        )
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 3.5, "voiced")])
        hint = AudioHints(metadata={"task": "maximum_phonation_time"})
        result = voice(store, "plain", config, hint, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert "10.0" in result.verdict.why and "40.0" in result.verdict.why

    def test_a_null_task_range_leaves_the_row_inert(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Nobody derived a range, so no span is out of one."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 3.5, "voiced")])
        hint = AudioHints(metadata={"task": "maximum_phonation_time"})
        result = voice(store, "plain", voice_config, hint, run_dir=tmp_path)
        assert result.verdict.outcome is not Outcome.FLAG
        assert _verdict_entity(store, "VOICE").attributes["task_range"] == "not_evaluated"


class TestTheHalfFrameTolerance:
    """A frame stands for a hop-wide interval centred on its time (V20)."""

    def test_a_span_of_exactly_min_marks_s_is_measured(
        self,
        store: ProvStore,
        voice_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the tolerance this span reads one hop short and its marks are skipped."""
        hop_s = 0.01
        min_marks_s = 3.0 / 75.0
        start, end = 1.0, 1.0 + min_marks_s - hop_s
        _seed_voice_store(store, tmp_path, phonation=[(start, end, "voiced")], hop_s=hop_s)
        calls = _stub_period_marks(monkeypatch, marks=4)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert calls, "the frame-edge tolerance is one hop; this span reaches min_marks_s with it"
        assert _verdict_entity(store, "VOICE").attributes["marks_skipped_short_n"] == 0

    def test_a_span_one_hop_shorter_still_is_skipped(
        self,
        store: ProvStore,
        voice_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The tolerance is one hop, not an open-ended slack."""
        hop_s = 0.01
        min_marks_s = 3.0 / 75.0
        _seed_voice_store(store, tmp_path, phonation=[(1.0, 1.0 + min_marks_s - 2 * hop_s, "voiced")], hop_s=hop_s)
        calls = _stub_period_marks(monkeypatch, marks=4)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert calls == []
        marks = find_measurements(store, "period_marks")
        assert marks[0].attributes["unmeasured"] == "shorter_than_mark_window"

    def test_the_tolerance_is_recorded_as_the_hop_not_a_constant(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The activity's parameters must show where the tolerance came from."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")], hop_s=0.01)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        analyze = next(a for a in store.activities("VOICE") if a.step == "analyze")
        assert analyze.parameters["frame_edge_tolerance_s"] == pytest.approx(0.01)


class TestTheF0RangeServesAPopulation:
    """The range is declared, overridable per population, and a vacuous ratio is refused at load."""

    def test_a_population_override_replaces_the_range(self, store: ProvStore, tmp_path: Path) -> None:
        """Age and sex move the range; the hint names which population."""
        config = _override(
            tmp_path,
            "voice:\n  f0_range_hz: [75, 500]\n  f0_range_by_population: {adult_male: [60, 250]}\n",
        )
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        hint = AudioHints(metadata={"population": "adult_male"})
        voice(store, "plain", config, hint, run_dir=tmp_path)
        analyze = next(a for a in store.activities("VOICE") if a.step == "analyze")
        assert list(analyze.parameters["f0_range_hz"]) == [60.0, 250.0]

    def test_a_vacuous_ratio_is_refused_before_the_store_is_touched(self, store: ProvStore, tmp_path: Path) -> None:
        """A check that flags everything reports nothing, so it is refused rather than run and flagged."""
        config = _override(tmp_path, "voice:\n  f0_range_hz: [50, 800]\n  f0_range_ratio_max: 4.0\n")
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        before = len(store.entities())
        with pytest.raises(ValueError, match="f0_range_ratio_max"):
            voice(store, "plain", config, run_dir=tmp_path)
        assert len(store.entities()) == before

    def test_a_null_ratio_refuses_nothing(self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path) -> None:
        """Nobody fixed the bound, so no configuration exceeds it."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        assert voice(store, "plain", voice_config, run_dir=tmp_path).verdict.outcome is not Outcome.FAIL


class TestEdgesAreNamedApart:
    """The onset is a period where one exists; the offset is always a criterion."""

    def test_a_span_with_marks_has_a_period_onset(
        self,
        store: ProvStore,
        voice_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An observed event, named as one."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        _stub_period_marks(monkeypatch, marks=6)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        span = _voice_spans(store)[0]
        assert span.attributes["onset_kind"] == "period"
        assert span.attributes["offset_kind"] == "criterion"

    def test_a_marked_span_reports_both_f0_keys(
        self,
        store: ProvStore,
        voice_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two F0 values from two streams are two measurements, so the stream travels with the value."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        _stub_period_marks(monkeypatch, marks=6)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        detail = _verdict_entity(store, "VOICE").attributes
        assert detail["f0_median_hz"] > 0.0
        assert detail["f0_stream"] == "plain"

    def test_an_unmarked_span_reports_neither(
        self, store: ProvStore, voice_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Absent for a span with no period marks, rather than estimated from one."""
        _seed_voice_store(store, tmp_path, phonation=[(0.0, 1.5, "unvoiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        detail = _verdict_entity(store, "VOICE").attributes
        assert "f0_median_hz" not in detail
        assert "f0_stream" not in detail
