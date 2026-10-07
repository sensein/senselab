"""AIRWAY's task measures: what they write, and what VERDICT reads back."""

from __future__ import annotations

from senselab.audio.workflows.triage.breath_pattern import BreathPattern
from senselab.audio.workflows.triage.cough_pattern import CoughPattern
from senselab.audio.workflows.triage.nodes import airway_task
from senselab.audio.workflows.triage.nodes import verdict as verdict_module
from senselab.audio.workflows.triage.nodes.common import software_agent, write_measurement
from senselab.audio.workflows.triage.task_events import TaskEvent, TaskEvidence
from senselab.audio.workflows.triage.vocabulary import SUPERSEDES, standing_task_extents
from senselab.utils.prov_store import Entity, ProvStore


def test_the_measure_extent_stands_in_place_of_airways_and_settles_on_rerun() -> None:
    """The measure's extent supersedes AIRWAY's span, is kept unchanged on a rerun, and goes when absent."""
    store = ProvStore(run_id="breath-extent-test")
    airway = store.entity(prov_type="span", extent=(10.0, 12.0), attributes={"family": "airway", "role": "task_extent"})
    software = software_agent(store)
    extent = {"start_s": 1.0, "end_s": 30.0, "source": "breath_train", "phases": 10, "breaths": 5}

    def live() -> list[Entity]:
        return [s for s in store.entities("span") if not store.is_invalidated(s.id)]

    airway_task.settle_task_extent(store, store.activity(node="AIRWAY", step=None, parameters={}), software, extent)
    standing = standing_task_extents(live())
    assert [s.extent for s in standing] == [(1.0, 30.0)]
    assert standing[0].attributes[SUPERSEDES] == [airway]
    assert not store.is_invalidated(airway)

    again = store.activity(node="AIRWAY", step=None, parameters={"again": True})
    airway_task.settle_task_extent(store, again, software, extent)
    assert [s.id for s in standing_task_extents(live())] == [standing[0].id]

    gone = store.activity(node="AIRWAY", step=None, parameters={"none": True})
    airway_task.settle_task_extent(store, gone, software, None)
    assert [s.id for s in standing_task_extents(live())] == [airway]


def test_the_families_name_the_event_they_are_decided_on() -> None:
    """Every breath and cough family of the requirements profile names its kind; others name none."""
    assert airway_task.required_event("respiration-and-cough-fivebreaths") == "breath"
    assert airway_task.required_event("voluntary-cough") == "cough"
    assert airway_task.required_event("harvard-sentences-list") is None
    assert airway_task.required_event(None) is None


def test_an_absent_input_is_written_as_absent() -> None:
    """A measure that lacked its derivatives writes the names, and no reading."""
    assert airway_task.breath_attributes(("spectrogram_narrowband",)) == {"absent": ["spectrogram_narrowband"]}
    assert airway_task.cough_attributes(("spectrogram_narrowband",), 3) == {"absent": ["spectrogram_narrowband"]}


def _seed(name: str, attributes: dict[str, object]) -> ProvStore:
    store = ProvStore(run_id="airway-reading-test")
    software = software_agent(store)
    activity = store.activity(node="AIRWAY", step="align", parameters={})
    write_measurement(store, activity, software, name=name, signal="plain", attributes=attributes)
    return store


def _task_evidence(decision: str, phases: int) -> TaskEvidence:
    events = tuple(TaskEvent(float(i), float(i) + 0.5, 15.0) for i in range(phases))
    return TaskEvidence(events, events, None, (0.0, float(phases)), decision, "clear")


def test_verdict_reads_the_breath_reading_airway_wrote() -> None:
    """VERDICT's task evidence is AIRWAY's breath reading, unchanged: the decision and the breaths it counts."""
    read = BreathPattern(pattern="alternating_breaths", events_n=6, evidence=_task_evidence("present", 6))
    attributes = airway_task.breath_attributes(read)
    evidence = verdict_module._task_evidence(
        _seed(airway_task.BREATH_READING, attributes), "respiration-and-cough-fivebreaths"
    )
    assert evidence.breath_mode is not None
    assert evidence.breath_decision == "present"
    assert evidence.breath_train_breaths == 3
    assert evidence.breath_pattern == "alternating_breaths"
    assert evidence.owner_absent_inputs == ()


def test_a_breath_reading_without_the_plain_stream_is_absent() -> None:
    """With no task evidence the breath reading names the plain stream as its absent input."""
    attributes = airway_task.breath_attributes(BreathPattern(pattern="alternating_breaths", events_n=6))
    assert attributes["absent"] == ["plain"] and attributes["decision"] is None


def test_verdict_reads_the_cough_reading_airway_wrote() -> None:
    """VERDICT's task evidence is AIRWAY's cough reading, unchanged."""
    attributes = airway_task.cough_attributes(CoughPattern(onsets_s=(1.0, 2.0, 3.0)), 3)
    evidence = verdict_module._task_evidence(_seed(airway_task.COUGH_READING, attributes), "voluntary-cough")
    assert evidence.cough_onsets_n == 3
    assert evidence.owner_absent_inputs == ()


def test_a_breath_reading_from_before_the_task_layer_is_not_measured() -> None:
    """A stored breath reading with no ``decision`` names that field as the owner's absent input."""
    attributes = {"absent": [], "pattern": "alternating_breaths", "events_n": 6, "review": False, "reading": {}}
    evidence = verdict_module._task_evidence(
        _seed(airway_task.BREATH_READING, attributes), "respiration-and-cough-fivebreaths"
    )
    assert evidence.breath_decision is None
    assert evidence.owner_absent_inputs == (f"AIRWAY:{airway_task.BREATH_READING}.decision",)


def test_no_reading_without_an_airway_report_is_no_absent_input() -> None:
    """A store AIRWAY never reported on lacks nothing of AIRWAY's: the branch did not run."""
    evidence = verdict_module._task_evidence(ProvStore(run_id="unmeasured"), "respiration-and-cough-fivebreaths")
    assert evidence.breath_pattern is None
    assert evidence.owner_absent_inputs == ()


def test_the_contest_threshold_is_the_instructed_count_else_the_uncounted_minimum() -> None:
    """``data/discard_contested.yaml``: the whole instructed count; else one cough, or three breath events."""
    assert verdict_module.contest_events_min(5, "breath") == 5
    assert verdict_module.contest_events_min(3, "cough") == 3
    assert verdict_module.contest_events_min(None, "cough") == 1
    assert verdict_module.contest_events_min(None, "breath") == 3


def test_a_breath_contest_needs_a_train_that_rose() -> None:
    """Only a breath family names a train rise; a cough family's contest reads the detector alone."""
    assert verdict_module.contest_rise_db_min("breath") == 10.0
    assert verdict_module.contest_rise_db_min("cough") is None


def test_a_weak_train_does_not_contest_a_breath_discard() -> None:
    """The detector's events contest a no-breath discard only where the train itself rose clear of the floor."""
    from senselab.audio.workflows.triage.vocabulary import TaskEvidence, discard_contested

    def evidence(rise_db: float | None) -> TaskEvidence:
        return TaskEvidence(
            events_found_n=4, contest_events_min=3, contest_rise_db_min=10.0, breath_train_rise_db=rise_db
        )

    assert discard_contested(evidence(14.3))
    assert not discard_contested(evidence(7.9))
    assert not discard_contested(evidence(None))
    assert discard_contested(TaskEvidence(events_found_n=1, contest_events_min=1))
