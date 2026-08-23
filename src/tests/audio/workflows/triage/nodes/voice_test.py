"""VOICE — the residual fold, the two-condition gate, marks not contours. All else real.

Praat is faked by default, for speed and platform independence; the boundary tests that pin where its
own refusals lie put the real hnr_track, f0_track or period_marks back, since a fake cannot say where
Praat draws that line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

import senselab.audio.workflows.triage.nodes.voice as voice_module
from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.phonation import PeriodMark
from senselab.audio.tasks.phonation import f0_track as real_f0_track
from senselab.audio.tasks.phonation import hnr_track as real_hnr_track
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

FAKE_F0 = 220.0


def _fake_hnr_track(
    audio: Audio, *, f0_min_hz: float, hop_s: float, silence_threshold: float, periods_per_window: float
) -> tuple[np.ndarray, np.ndarray]:
    """A constant 20 dB HNR track on the hop grid, spanning the sliced audio."""
    n = int(round(audio.waveform.shape[-1] / audio.sampling_rate / hop_s))
    times = (np.arange(n) + 0.5) * hop_s
    return times, np.full(n, 20.0)


def _fake_f0_track(
    audio: Audio, *, f0_min_hz: float, f0_max_hz: float, hop_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A constant 220 Hz F0 track with constant strength, on the same hop grid."""
    n = int(round(audio.waveform.shape[-1] / audio.sampling_rate / hop_s))
    times = (np.arange(n) + 0.5) * hop_s
    return times, np.full(n, FAKE_F0), np.full(n, 0.9)


def _fake_period_marks(
    audio: Audio, start_s: float, end_s: float, *, f0_min_hz: float, f0_max_hz: float
) -> list[PeriodMark]:
    """Marks every 1/220 s inside the queried extent."""
    period = 1.0 / FAKE_F0
    times = np.arange(start_s, end_s - period, period)
    return [PeriodMark(time_s=float(t), period_s=period, amplitude=0.1) for t in times]


@pytest.fixture(autouse=True)
def praat_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Praat is deterministic but slow and platform-sensitive; the phonation tests own the real calls.

    The boundary tests below substitute the real functions back, because a fake cannot be the oracle
    for where Praat's own refusal lies.
    """
    monkeypatch.setattr(voice_module, "hnr_track", _fake_hnr_track)
    monkeypatch.setattr(voice_module, "f0_track", _fake_f0_track)
    monkeypatch.setattr(voice_module, "period_marks", _fake_period_marks)


def _cfg(tmp_path: Path, **phonation: object) -> TriageConfig:
    """An override supplying the four null phonation floors — fixtures for these tests, not recommendations."""
    values: dict[str, object] = {"f0_min_hz": 150.0, "f0_max_hz": 400.0, "hnr_floor_db": 5.0, "rms_floor": 0.01}
    values.update(phonation)
    path = tmp_path / "voice-override.yaml"
    path.write_text(yaml.safe_dump({"phonation": values}))
    return load_triage_config(path)


def _voice_spans(store: ProvStore) -> list[Entity]:
    """The voiced-run spans VOICE wrote, in time order."""
    spans = [e for e in store.entities("span") if voice_module._generating_node(store, e.id) == "VOICE"]
    return sorted(spans, key=lambda e: e.extent or (0.0, 0.0))


def _marks_measurements(store: ProvStore) -> list[Entity]:
    """VOICE's per-run period_marks measurements, in time order."""
    found = [e for e in store.entities("measurement") if e.attributes.get("name") == "period_marks"]
    return sorted(found, key=lambda e: e.extent or (0.0, 0.0))


def test_packaged_config_refuses_and_the_store_is_untouched(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """The gate cannot run ungated: the four phonation.* floors are null by design (N2)."""
    seed_voice_store(store, energetic=((1.0, 2.0),))
    before = store.fingerprint()
    with pytest.raises(ValueError, match="phonation\\."):
        voice_module.voice(store, "plain", load_triage_config(), run_dir=tmp_path)
    assert store.fingerprint() == before


def test_residual_subtracts_labelled_and_speech_but_not_unlabelled_spans(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """Energy minus airway-labelled minus speech; an unlabelled span is NOT excluded."""
    seed_voice_store(
        store,
        energetic=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
        airway_labelled=((1.0, 2.0),),
        speech_spans=((3.0, 4.0),),
        unlabelled_spans=((5.0, 6.0),),
    )
    voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    runs = _voice_spans(store)
    assert runs, "the unlabelled region must yield a voiced run"
    assert all(5.0 <= s <= 6.0 for r in runs for s in r.extent or ()), (
        "only the unlabelled region survives the fold; unclaimed activity is exactly what VOICE is for"
    )


def test_empty_residual_is_a_normal_fail(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """Every energetic interval belongs to another branch -> fail, with the verdict written."""
    seed_voice_store(store, energetic=((1.0, 2.0),), airway_labelled=((1.0, 2.0),))
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    assert result.verdict.outcome is Outcome.FAIL
    assert store.entities("verdict")


def test_the_gate_is_an_and_from_both_sides(
    seed_voice_store: Callable[..., dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """High HNR under the RMS floor is periodic room tone; high RMS under the HNR floor is noise.

    Neither passes alone. The fixtures hold one condition and starve the other.
    """
    # Periodic room tone: the HNR fake reads 20 dB but the wav is quiet, so RMS starves.
    quiet = ProvStore(run_id="gate-quiet")
    seed_voice_store(quiet, energetic=((1.0, 2.0),), loud=())
    result = voice_module.voice(quiet, "plain", _cfg(tmp_path), run_dir=tmp_path)
    assert result.verdict.outcome is Outcome.FAIL
    assert quiet.get_entity(result.verdict_entity_id).attributes["runs_n"] == 0

    # Broadband noise: the wav is loud but the HNR fake reads below the floor.
    noisy = ProvStore(run_id="gate-noisy")
    seed_voice_store(noisy, energetic=((1.0, 2.0),))

    def _noise_hnr(
        audio: Audio, *, f0_min_hz: float, hop_s: float, silence_threshold: float, periods_per_window: float
    ) -> tuple[np.ndarray, np.ndarray]:
        times, hnr = _fake_hnr_track(
            audio,
            f0_min_hz=f0_min_hz,
            hop_s=hop_s,
            silence_threshold=silence_threshold,
            periods_per_window=periods_per_window,
        )
        return times, np.full_like(hnr, -10.0)

    monkeypatch.setattr(voice_module, "hnr_track", _noise_hnr)
    result = voice_module.voice(noisy, "plain", _cfg(tmp_path), run_dir=tmp_path)
    assert result.verdict.outcome is Outcome.FAIL
    assert noisy.get_entity(result.verdict_entity_id).attributes["runs_n"] == 0

    # Both held: the gate opens.
    both = ProvStore(run_id="gate-both")
    seed_voice_store(both, energetic=((1.0, 2.0),))
    monkeypatch.setattr(voice_module, "hnr_track", _fake_hnr_track)
    result = voice_module.voice(both, "plain", _cfg(tmp_path), run_dir=tmp_path)
    assert result.verdict.outcome is Outcome.PASS
    assert both.get_entity(result.verdict_entity_id).attributes["runs_n"] == 1


def test_runs_are_elementary_never_merged(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-frame unvoiced gap yields two runs; nothing merges them."""
    seed_voice_store(store, energetic=((1.0, 2.0),))

    def _dipping_hnr(
        audio: Audio, *, f0_min_hz: float, hop_s: float, silence_threshold: float, periods_per_window: float
    ) -> tuple[np.ndarray, np.ndarray]:
        times, hnr = _fake_hnr_track(
            audio,
            f0_min_hz=f0_min_hz,
            hop_s=hop_s,
            silence_threshold=silence_threshold,
            periods_per_window=periods_per_window,
        )
        hnr[len(hnr) // 2] = -10.0  # below the floor for exactly one frame mid-interval
        return times, hnr

    monkeypatch.setattr(voice_module, "hnr_track", _dipping_hnr)
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    assert store.get_entity(result.verdict_entity_id).attributes["runs_n"] == 2
    first, second = _voice_spans(store)
    assert first.attributes["offset_criterion"] == "hnr", "the first run stopped because HNR fell"
    assert second.attributes["offset_criterion"] == "residual_end"


def test_marks_are_absent_outside_runs_and_absent_is_not_zero(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """period_marks is queried per run only; absent is not zero (N23).

    A markless gate-passing run records marks_n=0 with onset_kind='criterion', distinct from a run
    nobody measured.
    """
    seed_voice_store(store, energetic=((1.0, 2.0),))
    calls: list[tuple[float, float]] = []

    def _markless(
        audio: Audio, start_s: float, end_s: float, *, f0_min_hz: float, f0_max_hz: float
    ) -> list[PeriodMark]:
        calls.append((start_s, end_s))
        return []

    monkeypatch.setattr(voice_module, "period_marks", _markless)
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    (run,) = _voice_spans(store)
    assert run.attributes["marks_n"] == 0
    assert run.attributes["onset_kind"] == "criterion"
    assert len(calls) == 1, "queried once per voiced run, never outside one"
    start_s, end_s = calls[0]
    assert 1.0 <= start_s < end_s <= 2.0, "queried only inside the residual interval"
    (marks,) = _marks_measurements(store)
    assert marks.attributes["marks"] == [], "measured and empty — absent, not zero, not unmeasured"
    assert "f0_median_hz" not in store.get_entity(result.verdict_entity_id).attributes


def test_onset_is_a_period_and_offset_is_a_criterion(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """A marked run's span starts at its first mark; both edge kinds are named in the attributes."""
    seed_voice_store(store, energetic=((1.0, 2.0),))
    voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    (run,) = _voice_spans(store)
    assert run.attributes["onset_kind"] == "period"
    assert run.attributes["offset_kind"] == "criterion"
    (marks,) = _marks_measurements(store)
    assert marks.attributes["signal"] == "plain", "a measurement names the stream it was taken on"
    first_mark_time = marks.attributes["marks"][0]["time_s"]
    assert run.extent is not None and run.extent[0] == pytest.approx(first_mark_time)
    assert run.attributes["offset_criterion"] in {"hnr", "rms", "both", "residual_end"}
    assert run.attributes["offset_criterion"] == "residual_end", "this run ran into the interval's edge"


def test_period_doubling_alias_inside_the_range_flags(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """Median F0 * factor (or / factor) inside [f0_min, f0_max] -> ambiguous run, flagged (N21)."""
    seed_voice_store(store, energetic=((1.0, 2.0),))
    config = _cfg(tmp_path, f0_min_hz=100.0, f0_max_hz=500.0)  # marks at 220 Hz -> 440 also in range
    result = voice_module.voice(store, "plain", config, run_dir=tmp_path)
    assert result.verdict.outcome is Outcome.FLAG
    assert store.get_entity(result.verdict_entity_id).attributes["ambiguous_runs_n"] == 1


def test_gate_interval_flag_is_inert_while_unmeasured(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """phonation.*_interval keys are null: no near-edge flag fires; gate_interval: 'unmeasured' (N22)."""
    seed_voice_store(store, energetic=((1.0, 2.0),))
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["gate_interval"] == "unmeasured"
    assert verdict.attributes["flags"] == []
    assert result.verdict.outcome is Outcome.PASS


def test_hint_asserting_phonation_not_found_flags(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """hint.may_contain includes a voice.hint_tags tag and no run passes -> flag, not fail (N25)."""
    seed_voice_store(store, energetic=((1.0, 2.0),), loud=())  # RMS starves: no run passes the gate
    hint = AudioHints(may_contain=["phonation"])
    result = voice_module.voice(store, "plain", _cfg(tmp_path), hint, run_dir=tmp_path)
    assert result.verdict.outcome is Outcome.FLAG

    empty = ProvStore(run_id="hint-empty-residual")
    seed_voice_store(empty, energetic=((1.0, 2.0),), airway_labelled=((1.0, 2.0),))
    result = voice_module.voice(empty, "plain", _cfg(tmp_path), hint, run_dir=tmp_path)
    assert result.verdict.outcome is Outcome.FLAG, "an empty residual under the hint is also a flag"


def test_tracks_are_sidecars_on_the_hop_and_measurements_carry_used(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """voice_tracks npz exists with hop_s recorded; the activity records what it read."""
    ids = seed_voice_store(
        store,
        energetic=((1.0, 2.0),),
        airway_labelled=((3.0, 3.5),),
        speech_spans=((4.0, 4.5),),
        silence_windows=[{"start": 0.0, "end": 7.0, "is_silence": False}],
    )
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    (tracks,) = [e for e in store.entities("measurement") if e.attributes.get("name") == "voice_tracks"]
    assert tracks.attributes["hop_s"] == 0.01
    sidecar = np.load(tmp_path / tracks.attributes["path"])
    assert {"times_s", "rms", "hnr_db", "f0_times_s", "f0_hz", "f0_strength"} <= set(sidecar.files)
    assert len(sidecar["times_s"]) == len(sidecar["rms"]) == len(sidecar["hnr_db"])
    assert np.allclose(np.diff(sidecar["times_s"]), 0.01), "the three tracks share the analysis hop"

    activity_id = store.generated_by(result.verdict_entity_id)
    assert activity_id is not None
    used = set(store.uses_of(activity_id))
    read = {ids["stream"], ids["envelope"], ids["silence"], ids["labels"][0], ids["labelled_spans"][0]}
    read |= {ids["speech_spans"][0]}
    assert read <= used, "every entity read — envelope, spans, labels, silence — carries a used edge"


@pytest.mark.parametrize(
    ("f0_min_hz", "f0_max_hz", "expected"),
    [
        pytest.param(150.0, 500.0, True, id="only-the-times-factor-alias-in-range"),
        pytest.param(100.0, 400.0, True, id="only-the-divided-by-factor-alias-in-range"),
        pytest.param(100.0, 500.0, True, id="both-aliases-in-range"),
        pytest.param(150.0, 400.0, False, id="neither-alias-in-range"),
    ],
)
def test_alias_in_range_pins_each_clause(f0_min_hz: float, f0_max_hz: float, expected: bool) -> None:
    """220 Hz with factor 2.0 aliases to 440 and 110; each clause is pinned alone and together (N21)."""
    assert voice_module._alias_in_range(220.0, factor=2.0, f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz) is expected


def test_near_edge_flags_fire_when_both_intervals_are_supplied(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """Override-supplied [lo, hi] intervals arm the near-edge check (N22).

    A run whose gate values at onset fall inside is flagged per family, and the verdict records
    gate_interval 'measured'.
    """
    seed_voice_store(store, energetic=((1.0, 2.0),))
    config = _cfg(tmp_path, hnr_floor_interval_db=[0.0, 25.0], rms_floor_interval=[0.0, 1.0])
    result = voice_module.voice(store, "plain", config, run_dir=tmp_path)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["gate_interval"] == "measured"
    assert any(flag.startswith("near_gate_edge hnr") for flag in verdict.attributes["flags"])
    assert any(flag.startswith("near_gate_edge rms") for flag in verdict.attributes["flags"])
    assert result.verdict.outcome is Outcome.FLAG


def test_gate_interval_is_partial_when_exactly_one_interval_is_supplied(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path
) -> None:
    """Exactly one interval supplied -> gate_interval 'partial'; only that family's flag can fire (N22)."""
    seed_voice_store(store, energetic=((1.0, 2.0),))
    config = _cfg(tmp_path, hnr_floor_interval_db=[0.0, 25.0])
    result = voice_module.voice(store, "plain", config, run_dir=tmp_path)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["gate_interval"] == "partial"
    assert any(flag.startswith("near_gate_edge hnr") for flag in verdict.attributes["flags"])
    assert not any(flag.startswith("near_gate_edge rms") for flag in verdict.attributes["flags"])


def test_sub_window_residual_fragments_are_pruned_not_handed_to_praat(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rolling floor fragments the residual; Praat refuses a segment shorter than its window.

    The real tracks run here: a fragment shorter than ``periods_per_window / f0_min_hz`` reaches
    Praat only if the node fails to prune it, and Praat's refusal is the defect this pins.
    """
    monkeypatch.setattr(voice_module, "hnr_track", real_hnr_track)
    monkeypatch.setattr(voice_module, "f0_track", real_f0_track)
    seed_voice_store(store, energetic=((1.0, 1.02), (2.0, 2.5)))
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["short_intervals_n"] == 1, "the 20 ms fragment is pruned and counted"
    assert all((r.extent or (0.0, 0.0))[0] >= 2.0 for r in _voice_spans(store)), "no run comes from the fragment"


def test_a_residual_of_nothing_but_fragments_fails_and_says_so(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every residual interval shorter than the window leaves nothing analysable, which is a fail."""
    monkeypatch.setattr(voice_module, "hnr_track", real_hnr_track)
    monkeypatch.setattr(voice_module, "f0_track", real_f0_track)
    seed_voice_store(store, energetic=((1.0, 1.02), (2.0, 2.015)))
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    verdict = store.get_entity(result.verdict_entity_id)
    assert result.verdict.outcome is Outcome.FAIL
    assert verdict.attributes["short_intervals_n"] == 2
    assert "shorter than the minimum analysable duration" in result.verdict.why


@pytest.mark.parametrize("duration_ms", [30, 33, 36])
def test_a_fragment_in_the_unguarded_band_is_pruned(
    store: ProvStore,
    seed_voice_store: Callable[..., dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duration_ms: int,
) -> None:
    """Praat's harmonicity needs (periods_per_window + 1) / f0_min, not periods_per_window / f0_min.

    At f0_min 150 Hz and 4.5 periods per window the two differ by a factor of 1.2222, and every
    duration in [30 ms, 36.67 ms) sat inside that gap: long enough to survive a floor set at the
    window, short enough for Praat to refuse. Real hnr_track and f0_track run, so the refusal is the
    oracle rather than a fake standing in for it.
    """
    monkeypatch.setattr(voice_module, "hnr_track", real_hnr_track)
    monkeypatch.setattr(voice_module, "f0_track", real_f0_track)
    seed_voice_store(store, energetic=((1.0, 1.0 + duration_ms / 1000.0),))
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["short_intervals_n"] == 1
    assert _voice_spans(store) == []
    assert result.verdict.outcome is Outcome.FAIL


def test_a_fragment_just_over_the_praat_floor_is_analysed(
    store: ProvStore, seed_voice_store: Callable[..., dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of the boundary: 37 ms clears (periods_per_window + 1) / f0_min and is measured.

    Without this the pruning could be tightened arbitrarily and still look correct.
    """
    monkeypatch.setattr(voice_module, "hnr_track", real_hnr_track)
    monkeypatch.setattr(voice_module, "f0_track", real_f0_track)
    seed_voice_store(store, energetic=((1.0, 1.037),))
    result = voice_module.voice(store, "plain", _cfg(tmp_path), run_dir=tmp_path)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["short_intervals_n"] == 0, "37 ms is above the floor and must be analysed"
    assert verdict.attributes["runs_n"] >= 1, "a 220 Hz tone over the floor passes the gate"
