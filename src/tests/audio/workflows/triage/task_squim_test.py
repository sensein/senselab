"""Tests for SQUIM over fixed windows of the task extent."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import find_measurement, path_attributes, write_stream
from senselab.audio.workflows.triage.task_squim import (
    ABSENT,
    EXTENT_FROM_CUT,
    EXTENT_FROM_SPANS,
    MEASUREMENT,
    OK,
    PRESENT,
    WindowConfig,
    measure_task_squim,
    summarise,
    task_squim_parameters,
    task_windows,
)
from senselab.utils.prov_store import ProvStore

RATE = 16_000
CONFIG = WindowConfig(window_s=10.0, min_window_s=3.0, floor_s=1.0)


class TestTheWindows:
    """Non-overlapping fixed windows; a trailing window only at the minimum; one window or none below it."""

    def test_the_packaged_configuration_is_ten_three_one(self) -> None:
        """data/task_squim.yaml declares 10 s windows, a 3 s minimum and a 1 s floor."""
        assert WindowConfig.from_parameters() == CONFIG
        assert task_squim_parameters()["derivation"]

    def test_full_windows_tile_from_the_start(self) -> None:
        """An extent of exactly two windows is two windows, no tail."""
        assert task_windows(1.0, 21.0, CONFIG) == [(1.0, 11.0), (11.0, 21.0)]

    def test_a_tail_at_the_minimum_is_kept(self) -> None:
        """A 3 s remainder is a window of its own."""
        assert task_windows(0.0, 23.0, CONFIG) == [(0.0, 10.0), (10.0, 20.0), (20.0, 23.0)]

    def test_a_tail_under_the_minimum_is_dropped(self) -> None:
        """A 2.9 s remainder is not read."""
        assert task_windows(0.0, 22.9, CONFIG) == [(0.0, 10.0), (10.0, 20.0)]

    def test_an_extent_between_the_minimum_and_one_window_is_one_window(self) -> None:
        """A 7 s extent is one 7 s window."""
        assert task_windows(2.0, 9.0, CONFIG) == [(2.0, 9.0)]

    def test_an_extent_under_the_minimum_is_one_window_over_the_whole(self) -> None:
        """A 2 s extent, under the minimum but over the floor, is one window."""
        assert task_windows(4.0, 6.0, CONFIG) == [(4.0, 6.0)]

    def test_an_extent_at_the_floor_is_one_window(self) -> None:
        """Exactly the floor is read."""
        assert task_windows(4.0, 5.0, CONFIG) == [(4.0, 5.0)]

    def test_an_extent_under_the_floor_gets_none(self) -> None:
        """Under the floor, nothing is read."""
        assert task_windows(4.0, 4.99, CONFIG) == []
        assert task_windows(4.0, 4.0, CONFIG) == []

    def test_floating_point_edges_do_not_lose_a_window(self) -> None:
        """30 s of 0.1 s steps is three windows, not two and a dropped tail."""
        end = sum([0.1] * 300)
        assert len(task_windows(0.0, end, CONFIG)) == 3

    def test_an_inconsistent_configuration_is_refused(self) -> None:
        """The minimum may not exceed the window, nor the floor the minimum."""
        with pytest.raises(ValueError):
            WindowConfig.from_parameters({"windows": {"window_s": 2.0, "min_window_s": 3.0, "floor_s": 1.0}})

    def test_the_summary_is_the_median_of_the_windows_read(self) -> None:
        """Unmeasured windows count neither towards the medians, the count nor the seconds covered."""
        readings: list[dict[str, Any]] = [
            {"start_s": 0.0, "end_s": 10.0, "pesq": 2.0, "stoi": 0.9, "si_sdr": 10.0},
            {"start_s": 10.0, "end_s": 20.0, "pesq": 3.0, "stoi": 1.0, "si_sdr": 20.0},
            {"start_s": 20.0, "end_s": 24.0, "unmeasured": "RuntimeError"},
        ]
        assert summarise(readings) == {
            "pesq": 2.5,
            "stoi": 0.95,
            "si_sdr": 15.0,
            "windows_n": 2,
            "covered_s": 20.0,
        }


@pytest.fixture
def config() -> TriageConfig:
    """The packaged triage configuration."""
    return load_triage_config()


def _tone(seconds: float) -> Audio:
    t = torch.arange(int(round(RATE * seconds)), dtype=torch.float32) / RATE
    return Audio(waveform=(0.5 * torch.sin(2 * math.pi * 220.0 * t)).unsqueeze(0), sampling_rate=RATE)


def _stream(store: ProvStore, run_dir: Path, name: str, audio: Audio) -> str:
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    relative, _ = write_stream(audio, run_dir, name)
    return store.entity(
        prov_type="stream",
        extent=(0.0, audio.waveform.shape[-1] / RATE),
        attributes={"name": name, **path_attributes(relative, run_dir), "sampling_rate": RATE, "channels": 1},
    )


def _seed(tmp_path: Path, *, family: str = "caterpillar-passage", task: tuple[float, float] = (2.0, 25.0)) -> ProvStore:
    run_dir = tmp_path / "run"
    store = ProvStore(run_id="task-squim-test")
    _stream(store, run_dir, "recording", _tone(30.0))
    _stream(store, run_dir, "plain", _tone(30.0))
    store.entity(prov_type="span", extent=task, attributes={"role": "task_extent", "branch": "SPEECH"})
    store.entity(prov_type="verdict", extent=None, attributes={"node": "VERDICT", "declared_family": family})
    return store


class _Counting:
    """A fake scorer: records each window's length and returns fixed readings."""

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def __call__(self, audios: Sequence[Audio]) -> list[dict[str, float]]:
        self.lengths.extend(int(a.waveform.shape[-1]) for a in audios)
        assert all(a.sampling_rate == RATE and a.waveform.shape[0] == 1 for a in audios)
        return [{"pesq": 2.9, "stoi": 0.98, "si_sdr": 17.0} for _ in audios]


class TestTheStore:
    """One task_squim measurement on the plain stream, with its provenance, replaced and never edited."""

    def test_the_measurement_round_trips_through_the_store(self, tmp_path: Path, config: TriageConfig) -> None:
        """Windows, medians, count, seconds covered and the provenance survive a write and a read."""
        store = _seed(tmp_path)
        fake = _Counting()
        outcome = measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=lambda: fake)
        assert outcome.status == OK and outcome.changed
        # Extent 1.75-25.25 s (0.25 s pad): two full windows and a 3.5 s tail.
        assert fake.lengths == [160_000, 160_000, 56_000]

        path = tmp_path / "store.jsonl"
        store.write_jsonl(path)
        back = ProvStore.read_jsonl(path, run_id="task-squim-test")
        held = find_measurement(back, MEASUREMENT)
        assert held is not None
        a: dict[str, Any] = held.attributes
        assert (a["pesq"], a["stoi"], a["si_sdr"]) == (2.9, 0.98, 17.0)
        assert a["windows_n"] == 3 and a["covered_s"] == pytest.approx(23.5)
        assert a["signal"] == "plain" and a["extent_source"] == EXTENT_FROM_SPANS
        assert held.extent == (1.75, 25.25)
        assert a["window_config"] == CONFIG.as_dict()
        assert a["weights_sha256"] == task_squim_parameters()["model"]["weights_sha256"]
        assert [w["end_s"] - w["start_s"] for w in a["windows"]] == pytest.approx([10.0, 10.0, 3.5])
        model = [g for g in back.agents("model") if g.model_id == "torchaudio SQUIM_OBJECTIVE"]
        assert model and a["weights_sha256"] in str(model[0].unresolved_reason)

    def test_a_second_pass_over_the_same_inputs_writes_nothing(self, tmp_path: Path, config: TriageConfig) -> None:
        """The same plain stream, extent, windows and weights come back present."""
        store = _seed(tmp_path)
        measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        again = measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        assert again.status == PRESENT and not again.changed

    def test_a_moved_extent_retires_the_old_measurement(self, tmp_path: Path, config: TriageConfig) -> None:
        """A new extent writes a new measurement and invalidates the old one, which stays in the store."""
        store = _seed(tmp_path)
        measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        first = find_measurement(store, MEASUREMENT)
        assert first is not None
        store.entity(
            prov_type="measurement",
            extent=(5.0, 9.0),
            attributes={"name": "task_audio", "signal": "recording", "start_s": 5.0, "end_s": 9.0},
        )
        outcome = measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        second = find_measurement(store, MEASUREMENT)
        assert outcome.status == OK and second is not None and second.id != first.id
        assert store.is_invalidated(first.id)
        assert second.attributes["extent_source"] == EXTENT_FROM_CUT
        assert second.attributes["windows_n"] == 1

    def test_an_airway_family_gets_no_value(self, tmp_path: Path, config: TriageConfig) -> None:
        """Breath and cough families are not read."""
        store = _seed(tmp_path, family="voluntary-cough")
        outcome = measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        assert outcome.status == ABSENT and outcome.reason == "airway family"
        assert find_measurement(store, MEASUREMENT) is None

    def test_a_voice_family_gets_no_value(self, tmp_path: Path, config: TriageConfig) -> None:
        """Sustained vowels and glides are not read: SQUIM is speech-trained."""
        store = _seed(tmp_path, family="prolonged-vowel")
        outcome = measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        assert outcome.status == ABSENT and outcome.reason == "voice family"
        assert find_measurement(store, MEASUREMENT) is None

    def test_a_voice_family_measured_under_v1_is_retired(self, tmp_path: Path, config: TriageConfig) -> None:
        """A task_squim a VOICE recording carries from before is invalidated, not left standing."""
        store = _seed(tmp_path, family="prolonged-vowel")
        v1 = {**task_squim_parameters(), "no_value_branches": ["AIRWAY"]}
        measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting, parameters=v1)
        held = find_measurement(store, MEASUREMENT)
        assert held is not None
        outcome = measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        assert outcome.status == ABSENT and outcome.changed and store.is_invalidated(held.id)

    def test_no_task_extent_gets_no_value(self, tmp_path: Path, config: TriageConfig) -> None:
        """A recording with no live task-extent span is not read."""
        store = _seed(tmp_path)
        for span in store.entities("span"):
            activity = store.activity(node="TEST", step="retire", parameters={})
            store.was_invalidated_by(span.id, activity)
        outcome = measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        assert outcome.status == ABSENT and outcome.reason == "no task extent"

    def test_an_extent_under_the_floor_gets_no_value(self, tmp_path: Path, config: TriageConfig) -> None:
        """A 0.4 s task padded to 0.9 s is under the 1 s floor."""
        store = _seed(tmp_path, task=(5.0, 5.4))
        outcome = measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=_Counting)
        assert outcome.status == ABSENT and "floor" in str(outcome.reason)


def test_a_timestamped_manifest_stem_gives_its_family(tmp_path: Path) -> None:
    """Where the fold names no family, the manifest stem's launch timestamp is dropped before the task is read."""
    from senselab.audio.workflows.triage.task_squim import declared_family

    store = ProvStore(run_id="stem-test")
    assert declared_family(store, "sub-a_ses-b_task-voluntary-cough-2_20000101-000000") == "voluntary-cough"


def _windows_before_the_shared_planner(start_s: float, end_s: float, config: WindowConfig) -> list[tuple[float, float]]:
    """``task_windows`` as it stood before window planning moved into ``torchaudio_squim``."""
    eps = 1e-9
    duration = end_s - start_s
    if duration < config.floor_s - eps:
        return []
    if duration < config.min_window_s - eps:
        return [(start_s, end_s)]
    full = int(math.floor(duration / config.window_s + eps))
    windows = [(start_s + i * config.window_s, start_s + (i + 1) * config.window_s) for i in range(full)]
    tail_start = start_s + full * config.window_s
    if end_s - tail_start >= config.min_window_s - eps:
        windows.append((tail_start, end_s))
    return windows


def _chirp(seconds: float) -> Audio:
    t = torch.arange(int(round(RATE * seconds)), dtype=torch.float32) / RATE
    waveform = 0.2 * torch.sin(2 * math.pi * (100 + 20 * t) * t) * (1 + t / 40)
    return Audio(waveform=waveform.unsqueeze(0), sampling_rate=RATE)


def _cut_scorer(audios: Sequence[Audio]) -> list[dict[str, float]]:
    """A deterministic reading of each cut: its mean magnitude, its length and its end samples."""
    return [
        {
            "pesq": round(float(a.waveform.abs().mean()), 6),
            "stoi": a.waveform.shape[-1] / 1e6,
            "si_sdr": round(float(a.waveform[0, 0] + a.waveform[0, -1]), 6),
        }
        for a in audios
    ]


class TestTheSharedPlannerChangesNothing:
    """Triage's windows and medians are those it published before the planner moved."""

    def test_the_windows_match_the_previous_planner_on_a_grid(self) -> None:
        """Every extent from 0 to 45 s in 0.05 s steps, from three starts."""
        for start in (0.0, 1.75, 3.3):
            for k in range(901):
                end = start + k * 0.05
                assert task_windows(start, end, CONFIG) == _windows_before_the_shared_planner(start, end, CONFIG)

    @pytest.mark.parametrize(
        ("task", "windows", "medians"),
        [
            ((2.0, 25.0), [(1.75, 11.75), (11.75, 21.75), (21.75, 25.25)], (0.180644, 0.16, 0.545578, 3, 23.5)),
            ((0.5, 29.7), [(0.25, 10.25), (10.25, 20.25), (20.25, 29.95)], (0.175869, 0.16, 0.447459, 3, 29.7)),
            ((4.0, 6.2), [(3.75, 6.45)], (0.143554, 0.0432, 0.258684, 1, 2.7)),
            ((1.3, 14.1), [(1.05, 11.05), (11.05, 14.35)], (0.157171, 0.1064, 0.158042, 2, 13.3)),
            ((3.0, 15.2), [(2.75, 12.75)], (0.151995, 0.16, 0.469931, 1, 10.0)),
        ],
    )
    def test_the_measurement_matches_values_recorded_before_the_refactor(
        self,
        tmp_path: Path,
        config: TriageConfig,
        monkeypatch: pytest.MonkeyPatch,
        task: tuple[float, float],
        windows: list[tuple[float, float]],
        medians: tuple[float, float, float, int, float],
    ) -> None:
        """A chirp, a deterministic scorer of each cut, and the values the pre-refactor code wrote."""
        monkeypatch.setitem(globals(), "_tone", _chirp)
        store = _seed(tmp_path, task=task)
        measure_task_squim(store, config, run_dir=tmp_path / "run", scorer=lambda: _cut_scorer)
        held = find_measurement(store, MEASUREMENT)
        assert held is not None
        a = held.attributes
        assert [(w["start_s"], w["end_s"]) for w in a["windows"]] == pytest.approx(windows)
        pesq, stoi, si_sdr, windows_n, covered_s = medians
        assert a["pesq"] == pytest.approx(pesq, abs=2e-6)
        assert a["stoi"] == pytest.approx(stoi)
        assert a["si_sdr"] == pytest.approx(si_sdr, abs=2e-6)
        assert a["windows_n"] == windows_n and a["covered_s"] == pytest.approx(covered_s)


def test_the_shipped_scorer_reads_each_window_in_one_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """``squim_scorer`` asks the general function for a single pass at the default floor, so a window is read as cut."""
    from senselab.audio.tasks.features_extraction import torchaudio_squim
    from senselab.audio.workflows.triage import task_squim

    calls: list[dict[str, Any]] = []

    def fake(audios: list[Audio], **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        calls.append(kwargs)
        return [
            {"stoi": 0.9, "pesq": 2.0, "si_sdr": 10.0, "windows": [], "windows_n": 1, "span_s": 1.0} for _ in audios
        ]

    monkeypatch.setattr(torchaudio_squim, "extract_objective_quality_features_from_audios", fake)
    monkeypatch.setattr(task_squim, "verify_weights", lambda asset, sha256: sha256)
    task_squim.squim_scorer.cache_clear()
    try:
        assert task_squim.squim_scorer()([_tone(12.0)]) == [{"pesq": 2.0, "stoi": 0.9, "si_sdr": 10.0}]
    finally:
        task_squim.squim_scorer.cache_clear()
    assert calls and calls[0]["window_s"] is None and "floor_s" not in calls[0]
