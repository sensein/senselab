"""The DDK task layer: syllables and cycles over the background, their identity and the decision."""

from typing import Sequence

import numpy as np
import pytest

from senselab.audio.workflows.triage.background_model import (
    BandFrames,
    Floor,
    Impulse,
    background_model_parameters,
    regions_of,
)
from senselab.audio.workflows.triage.ddk_task import (
    CYCLE,
    SYLLABLE,
    Identity,
    PositionMasses,
    Voicing,
    cycle_period_s,
    cycles_of,
    ddk_parameters,
    ddk_reading_of,
    decide_ddk,
    dispersion,
    identity_of,
    syllable_runs,
    trend,
)
from senselab.audio.workflows.triage.task_events import (
    ABSENT,
    PRESENT,
    REVIEW,
    GenericView,
    Rhythm,
    TaskEvent,
    task_events_parameters,
)

HOP = 0.01
FLOOR = -70.0

Span = tuple[float, float]


def _view(level: np.ndarray, impulses: Sequence[Span] = ()) -> GenericView:
    """A background whose broadband level over the floor is ``level`` per frame."""
    p = background_model_parameters()
    bands = len(p["band_edges_hz"]) - 1
    times = np.arange(len(level)) * HOP + HOP / 2
    band_db = FLOOR + np.asarray(level, dtype=float)[:, None] + np.zeros((len(level), bands))
    level_db = 10.0 * np.log10(np.sum(10.0 ** (band_db / 10.0), axis=1))
    floor = np.full(bands, FLOOR)
    held = tuple(Impulse((a + b) / 2, a, b, 30.0, 1.0) for a, b in impulses)
    frames = BandFrames(times, band_db, level_db)
    regions = tuple(regions_of(frames, floor, held, p))
    broadband = float(10.0 * np.log10(np.sum(10.0 ** (floor / 10.0))))
    return GenericView(frames, Floor(floor, "quiet_frames", 1.0, floor, None), held, regions, broadband, HOP)


def _train(duration_s: float, words: Sequence[Sequence[Span]], *, trough_db: float = 8.0) -> np.ndarray:
    """A level series: silence, and each syllable a 40 dB plateau whose first 50 ms is a closure trough."""
    level = np.zeros(int(round(duration_s / HOP)))
    for word in words:
        for start, end in word:
            a, b = int(round(start / HOP)), int(round(end / HOP))
            level[a:b] = 40.0
            level[a : a + 5] = trough_db
    return level


def _syllables(start: float, n: int, length: float) -> list[Span]:
    """``n`` contiguous syllables of ``length`` from ``start``."""
    return [(start + i * length, start + (i + 1) * length) for i in range(n)]


def _positions(n_frames: int, template: Sequence[str], spans: Sequence[tuple[Span, Sequence[float]]]) -> PositionMasses:
    """Position masses: silence everywhere, and inside each span each position's mass in turn."""
    masses = np.zeros((n_frames, len(template)))
    silence = np.ones(n_frames)
    for (start, end), per_position in spans:
        a, b = int(round(start / HOP)), int(round(end / HOP))
        width = max(1, (b - a) // len(per_position))
        silence[a:b] = 0.0
        for j, mass in enumerate(per_position):
            masses[a + j * width : a + (j + 1) * width, j] = mass
    return PositionMasses(tuple(template), tuple(f"c{j}" for j in range(len(template))), masses, silence, HOP)


Q = ddk_parameters()
P = task_events_parameters()


class TestSyllablesAreSplitAtTroughs:
    """Nuclei over BACKGROUND's activity regions, cut where the level or voicing dips."""

    def test_three_contiguous_syllables_are_three(self) -> None:
        """One region, two closure troughs inside it, three syllables tiling it."""
        view = _view(_train(2.0, [_syllables(0.5, 3, 0.2)]))
        [run] = syllable_runs(view, None, Q)
        assert len(run) == 3
        assert run[0][0] == pytest.approx(0.5, abs=0.02)
        assert run[-1][1] == pytest.approx(1.1, abs=0.02)

    def test_two_regions_are_two_runs(self) -> None:
        """A pause the background calls quiet closes a run."""
        view = _view(_train(3.0, [_syllables(0.5, 2, 0.2), _syllables(1.6, 2, 0.2)]))
        runs = syllable_runs(view, None, Q)
        assert [len(run) for run in runs] == [2, 2]

    def test_an_unvoiced_stretch_splits_a_flat_nucleus(self) -> None:
        """Voicing is a trough as well as level: a flat plateau unvoiced in its middle is two syllables."""
        level = np.zeros(200)
        level[50:110] = 40.0
        times = np.arange(200) * HOP + HOP / 2
        voiced = (times > 0.5) & ~((times > 0.78) & (times < 0.83))
        runs = syllable_runs(_view(level), Voicing(times, voiced), Q)
        assert [len(run) for run in runs] == [2]

    def test_a_stretch_longer_than_a_syllable_with_no_trough_is_none(self) -> None:
        """A held vowel is another activity, not a syllable."""
        level = np.zeros(300)
        level[50:250] = 40.0
        assert syllable_runs(_view(level), None, Q) == []


class TestCyclesAreGroupedFromSyllables:
    """The template's syllable count and the rhythm prior group syllables; a pause always closes a cycle."""

    def test_separate_words_are_separate_cycles_whatever_their_syllable_count(self) -> None:
        """Six bursts of two syllables each (a flapped "butter") are six cycles, not four."""
        runs = [_syllables(0.1 + i * 0.65, 2, 0.2) for i in range(6)]
        period = cycle_period_s(runs, 3, Rhythm(1.0 / 0.65, 10.0), Q)
        cycles = cycles_of(runs, period, 3)
        assert len(cycles) == 6

    def test_a_continuous_run_is_split_by_the_period(self) -> None:
        """Nine syllables at 0.2 s with a three-syllable template are three cycles."""
        [run] = [_syllables(0.0, 9, 0.2)]
        cycles = cycles_of([run], cycle_period_s([run], 3, None, Q), 3)
        assert [len(c) for c in cycles] == [3, 3, 3]

    def test_a_leading_partial_cycle_stands_as_its_own_event(self) -> None:
        """A word cut at its start is still an event; the identity reads what is missing."""
        runs = [_syllables(0.0, 2, 0.2), *(_syllables(0.8 + i * 0.8, 3, 0.2) for i in range(3))]
        cycles = cycles_of(runs, cycle_period_s(runs, 3, None, Q), 3)
        assert [len(c) for c in cycles] == [2, 3, 3, 3]

    def test_a_rhythm_far_from_the_syllables_is_not_used(self) -> None:
        """A modulation peak at three times the syllables' own cycle period is some other periodicity."""
        runs = [_syllables(0.0, 9, 0.2)]
        assert cycle_period_s(runs, 3, Rhythm(1.0 / 1.8, 10.0), Q) == pytest.approx(0.6)
        assert cycle_period_s(runs, 3, Rhythm(1.0 / 0.66, 10.0), Q) == pytest.approx(0.66)


class TestIdentityIsALocalAlignment:
    """Each event is scored against the template; skips, deletions and substitutions are allowed."""

    def test_every_position_realised_scores_its_mass(self) -> None:
        """A clean cycle reads each position at its class mass."""
        positions = _positions(100, ("p", "aa", "t", "aa"), [((0.2, 0.6), [0.9, 0.9, 0.9, 0.9])])
        identity = identity_of(positions, (0.2, 0.6), Q)
        assert identity.score == pytest.approx(0.9)
        assert all(m == pytest.approx(0.9) for m in identity.realised)

    def test_a_substitution_keeps_its_low_mass_at_its_own_position(self) -> None:
        """A wrong consonant is charged to its position at the mass its class held, not dropped."""
        positions = _positions(100, ("p", "aa", "t", "aa"), [((0.2, 0.6), [0.9, 0.9, 0.05, 0.9])])
        identity = identity_of(positions, (0.2, 0.6), Q)
        assert identity.realised[2] is not None and identity.realised[2] < 0.1
        assert identity.score < 0.75

    def test_a_partial_cycle_reads_its_missing_positions_as_deleted(self) -> None:
        """The leading half of a cycle missing: its positions are None and count 0."""
        positions = _positions(100, ("p", "aa", "t", "aa"), [((0.2, 0.4), [0.9, 0.9])])
        identity = identity_of(positions, (0.2, 0.4), Q)
        assert identity.realised[2] is None and identity.realised[3] is None
        assert identity.score == pytest.approx(0.45, abs=0.05)

    def test_silence_scores_nothing(self) -> None:
        """An event the posteriorgram calls silent holds no position."""
        positions = _positions(100, ("p", "aa"), [])
        assert identity_of(positions, (0.2, 0.6), Q).score == 0.0


def _event(snr: float = 40.0, local: float = 40.0, *, entangled: bool = False) -> TaskEvent:
    """One event read against the background."""
    return TaskEvent(0.0, 0.2, snr, local, entangled)


class TestTheDecisionIsTheSharedRule:
    """Present on clear events and identity; review on weak, entangled or ambiguous; absent on nothing."""

    def test_nothing_over_the_floor_is_absent(self) -> None:
        """No syllable train stands over the floor."""
        decision, why, _, _ = decide_ddk([_event(5.0)], [Identity(0.9, (0.9,))], P["decision"], Q, SYLLABLE)
        assert (decision, why) == (ABSENT, "no syllable train over the floor")

    def test_a_train_of_another_activity_is_absent(self) -> None:
        """Events over the floor whose identity is under ``identity_absent_max``."""
        low = Q["identity_absent_max"] / 2
        decision, why, _, _ = decide_ddk([_event()] * 5, [Identity(low, (low,))] * 5, P["decision"], Q, SYLLABLE)
        assert (decision, why) == (ABSENT, "not the target")

    def test_an_ambiguous_identity_is_review(self) -> None:
        """Between the two identity bounds."""
        mid = (Q["identity_absent_max"] + Q["identity_min"]) / 2
        decision, why, _, _ = decide_ddk([_event()] * 5, [Identity(mid, (mid,))] * 5, P["decision"], Q, SYLLABLE)
        assert (decision, why) == (REVIEW, "identity")

    def test_enough_clear_events_and_the_target_are_present(self) -> None:
        """``events_min`` clear events, the identity over ``identity_min``."""
        n = Q["events_min"][CYCLE]
        decision, _, inputs, identity = decide_ddk([_event()] * n, [Identity(0.8, (0.8,))] * n, P["decision"], Q, CYCLE)
        assert decision == PRESENT
        assert identity == pytest.approx(0.8)
        assert inputs["clear_free_n"] == n

    def test_too_few_clear_events_are_weak(self) -> None:
        """Events over the floor but under the local bound."""
        events = [_event(20.0, 10.0)] * 5
        decision, why, _, _ = decide_ddk(events, [Identity(0.8, (0.8,))] * 5, P["decision"], Q, SYLLABLE)
        assert (decision, why) == (REVIEW, "weak")

    def test_clear_events_all_touched_by_an_impulse_are_entangled(self) -> None:
        """Clear, but an impulse from outside the train touches each."""
        events = [_event(entangled=True)] * 5
        decision, why, _, _ = decide_ddk(events, [Identity(0.8, (0.8,))] * 5, P["decision"], Q, SYLLABLE)
        assert (decision, why) == (REVIEW, "entangled")


def _buttercup_case() -> tuple[GenericView, PositionMasses]:
    """Six bursts about 0.65 s apart from 0.015 s, each two syllables, as sub-06d289b0's v2 buttercup."""
    words = [_syllables(0.015 + i * 0.65, 2, 0.2) for i in range(6)]
    view = _view(_train(5.1, words))
    spans = [((w[0][0], w[-1][1]), [0.6, 0.7, 0.6, 0.3, 0.7, 0.7, 0.4]) for w in words]
    return view, _positions(510, ("b", "ah", "t", "er", "k", "ah", "p"), spans)


class TestTheReading:
    """Events, extent, decision and annotations, end to end over a background and a posteriorgram."""

    def test_the_buttercup_case_reads_six_cycles_from_its_first_burst_to_its_last(self) -> None:
        """Every burst is an event, the leading one included; the extent adds nothing."""
        view, positions = _buttercup_case()
        reading = ddk_reading_of(view, positions, sequence=True, syllables_per_cycle=3)
        assert reading.evidence is not None
        assert len(reading.evidence.events) == 6
        assert reading.extent is not None
        assert reading.extent[0] == pytest.approx(0.015, abs=0.02)
        assert reading.extent[1] == pytest.approx(0.015 + 5 * 0.65 + 0.4, abs=0.02)
        assert reading.decision == PRESENT

    def test_an_event_that_is_not_the_target_is_outside_the_extent(self) -> None:
        """A lead-in sound with no template mass is found, but is not a task event."""
        view, positions = _buttercup_case()
        level = view.frames.level_db - view.broadband_floor_db
        noisy = np.concatenate([np.zeros(10), level[10:]])
        noisy[455:475] = 40.0
        reading = ddk_reading_of(_view(noisy), positions, sequence=True, syllables_per_cycle=3)
        assert reading.evidence is not None and reading.extent is not None
        assert reading.evidence.events_found_n == 7
        assert reading.extent[1] < 4.0

    def test_the_counts_and_rates_are_annotations(self) -> None:
        """The count against the instruction, the rates, regularity and per-position mass."""
        view, positions = _buttercup_case()
        reading = ddk_reading_of(
            view, positions, sequence=True, syllables_per_cycle=3, required_count=10, recording_s=5.1
        )
        notes = reading.annotations
        assert notes["events_n"] == 6
        assert notes["count_fraction"] == pytest.approx(0.6)
        assert notes["cycle_rate_hz"] == pytest.approx(1 / 0.65, abs=0.05)
        assert notes["syllable_rate_hz"] == pytest.approx(5.0, abs=0.5)
        assert notes["period_cv"] == pytest.approx(0.0, abs=0.05)
        assert len(notes["realised_mass"]) == 7
        assert notes["train_fraction"] == pytest.approx((3.25 + 0.4) / 5.1, abs=0.02)
        assert reading.decision == PRESENT

    def test_an_impulse_inside_the_train_is_the_train_s_own(self) -> None:
        """A stop release is impulsive; only an impulse outside every run entangles."""
        view, positions = _buttercup_case()
        level = view.frames.level_db - view.broadband_floor_db
        inside = ddk_reading_of(_view(level, [(0.70, 0.705)]), positions, sequence=True, syllables_per_cycle=3)
        assert inside.evidence is not None
        assert not any(e.entangled for e in inside.evidence.found)
        outside = ddk_reading_of(_view(level, [(0.44, 0.445)]), positions, sequence=True, syllables_per_cycle=3)
        assert outside.evidence is not None
        assert any(e.entangled for e in outside.evidence.found)

    def test_a_missing_input_is_named_and_not_read(self) -> None:
        """No background or no posteriorgram: no decision, and the input named."""
        view, positions = _buttercup_case()
        assert ddk_reading_of(None, positions, sequence=True, syllables_per_cycle=3).absent == ("background_model",)
        missing = ddk_reading_of(view, None, sequence=True, syllables_per_cycle=3)
        assert missing.absent == ("ppg_posteriorgram",)
        assert missing.decision is None
        assert missing.record()["decision"] is None

    def test_a_single_syllable_family_takes_its_syllables_as_events(self) -> None:
        """Puh puh puh: each syllable is an event."""
        syllables = [(0.5 + i * 0.25, 0.5 + i * 0.25 + 0.2) for i in range(8)]
        view = _view(_train(3.0, [[s] for s in syllables]))
        positions = _positions(300, ("p", "ah"), [(s, [0.8, 0.8]) for s in syllables])
        reading = ddk_reading_of(view, positions, sequence=False, syllables_per_cycle=1)
        assert reading.unit == SYLLABLE
        assert reading.evidence is not None and len(reading.evidence.events) == 8
        assert reading.decision == PRESENT


class TestTheRegularityStatisticsAreParameterFree:
    """Applied to the task events' onset intervals."""

    def test_dispersion_is_zero_for_a_perfectly_even_train(self) -> None:
        """The coefficient of variation of a constant series is zero."""
        assert dispersion([0.2] * 6) == pytest.approx(0.0)

    def test_dispersion_is_none_below_its_own_domain(self) -> None:
        """A sample deviation needs two values."""
        assert dispersion([0.2]) is None
        assert dispersion([]) is None

    def test_the_trend_signs_festination_and_slowing(self) -> None:
        """Negative is festination and positive is slowing; no threshold names either."""
        festinating = trend([0.30, 0.28, 0.26, 0.24])
        slowing = trend([0.24, 0.26, 0.28, 0.30])
        assert festinating is not None and festinating < 0.0
        assert slowing is not None and slowing > 0.0


class TestTheParametersAreData:
    """Every bound is in ``data/task_events.yaml``; the deciding ones are marked unfitted there."""

    @pytest.mark.parametrize("key", ["events_min", "identity_min", "identity_absent_max"])
    def test_the_deciding_bounds_are_in_the_ddk_section(self, key: str) -> None:
        """N_min and the two identity bounds."""
        assert key in Q

    def test_the_identity_bounds_are_ordered(self) -> None:
        """Another activity sits under an ambiguous identity."""
        assert Q["identity_absent_max"] < Q["identity_min"]
