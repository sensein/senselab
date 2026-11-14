"""Tests for the behaviour-event data structures."""

from pathlib import Path

import pytest

from senselab.utils.data_structures.events import (
    AnnotationSchema,
    BehaviorEvent,
    EventTrack,
    whole_clip_event,
)


def _event(onset: float, offset: float, label: str = "reach", **kwargs: object) -> BehaviorEvent:
    fields: dict = {"onset_s": onset, "offset_s": offset, "label": label, "source": "annotator_a"}
    fields.update(kwargs)
    return BehaviorEvent(**fields)


def _schema(**kwargs: object) -> AnnotationSchema:
    fields: dict = {"schema_id": "motion", "task_type": "segmentation", "labels": ["reach", "point", "wave"]}
    fields.update(kwargs)
    return AnnotationSchema(**fields)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(text)
    return path


def test_event_constructor_keeps_fields() -> None:
    """Every field given to the constructor is kept as given."""
    event = _event(0.5, 1.25, confidence=0.75, confidence_kind="prob", subject_ref="s01", track_id="t1")
    assert (event.onset_s, event.offset_s, event.label) == (0.5, 1.25, "reach")
    assert (event.confidence, event.confidence_kind) == (0.75, "prob")
    assert (event.source, event.subject_ref, event.track_id) == ("annotator_a", "s01", "t1")


def test_event_defaults_to_no_confidence() -> None:
    """Without a confidence the kind is ``none``, and the optional keys are absent."""
    event = _event(0.0, 1.0)
    assert (event.confidence, event.confidence_kind) == (None, "none")
    assert event.subject_ref is None and event.track_id is None


def test_track_sorts_by_onset_offset_label() -> None:
    """A track orders its events by onset, then offset, then label, whatever the input order."""
    events = [_event(2.0, 3.0), _event(0.0, 2.0, "wave"), _event(0.0, 2.0, "point"), _event(0.0, 1.0)]
    track = EventTrack(schema_id="motion", events=events)
    assert [(e.onset_s, e.offset_s, e.label) for e in track.events] == [
        (0.0, 1.0, "reach"),
        (0.0, 2.0, "point"),
        (0.0, 2.0, "wave"),
        (2.0, 3.0, "reach"),
    ]


@pytest.mark.parametrize(("onset", "offset"), [(1.0, 1.0), (2.0, 1.0)])
def test_onset_not_before_offset_raises(onset: float, offset: float) -> None:
    """Intervals are half-open, so an empty or reversed interval is rejected."""
    with pytest.raises(ValueError, match="onset_s"):
        _event(onset, offset)


def test_negative_onset_raises() -> None:
    """Event times are measured from the start of the clip, so they cannot be negative."""
    with pytest.raises(ValueError, match="onset_s"):
        _event(-0.5, 1.0)


def test_confidence_without_kind_raises() -> None:
    """A confidence value must say what kind of number it is."""
    with pytest.raises(ValueError, match="confidence_kind"):
        _event(0.0, 1.0, confidence=0.5)


def test_kind_without_confidence_raises() -> None:
    """A confidence kind other than ``none`` needs a value."""
    with pytest.raises(ValueError, match="confidence_kind"):
        _event(0.0, 1.0, confidence_kind="vote")


def test_track_event_past_duration_raises() -> None:
    """An event cannot end after the clip it belongs to."""
    with pytest.raises(ValueError, match="duration_s"):
        EventTrack(schema_id="motion", events=[_event(0.0, 2.0)], duration_s=1.5)


def test_check_accepts_schema_labels() -> None:
    """A track whose labels all come from the schema passes the check."""
    track = EventTrack(schema_id="motion", events=[_event(0.0, 1.0), _event(1.0, 2.0, "wave")])
    track.check(_schema())


def test_check_rejects_foreign_label() -> None:
    """A label outside the schema is rejected."""
    track = EventTrack(schema_id="motion", events=[_event(0.0, 1.0, "jump")])
    with pytest.raises(ValueError, match="not in schema"):
        track.check(_schema())


def test_check_rejects_other_schema() -> None:
    """A track checked against a schema with a different id is rejected."""
    track = EventTrack(schema_id="other", events=[_event(0.0, 1.0)])
    with pytest.raises(ValueError, match="schema_id"):
        track.check(_schema())


def test_yaml_boolean_labels_raise(tmp_path: Path) -> None:
    """Unquoted ``yes``/``no`` load as booleans, and the loader names the first one."""
    path = _write(tmp_path, "schema_id: answer\ntask_type: classification\nlabels: [yes, no]\n")
    with pytest.raises(ValueError, match=r"labels\[0\] loaded as a YAML boolean; quote it"):
        AnnotationSchema.from_yaml(path)


def test_yaml_quoted_labels_load_as_strings(tmp_path: Path) -> None:
    """Quoted labels stay strings."""
    path = _write(tmp_path, "schema_id: answer\ntask_type: classification\nlabels: ['yes', 'no']\n")
    schema = AnnotationSchema.from_yaml(path)
    assert schema.labels == ["yes", "no"]
    assert schema.schema_id == "answer" and schema.task_type == "classification"


def test_yaml_integer_labels_raise(tmp_path: Path) -> None:
    """A label that loads as a number is rejected, naming its index."""
    path = _write(tmp_path, "schema_id: count\ntask_type: counting\nlabels: [1, 2]\n")
    with pytest.raises(ValueError, match=r"labels\[0\] is not a string"):
        AnnotationSchema.from_yaml(path)


def test_reference_prevalence_not_summing_to_one_raises() -> None:
    """Reference prevalences are a distribution over the labels."""
    with pytest.raises(ValueError, match="sum to 1"):
        _schema(reference_prevalence={"reach": 0.5, "point": 0.25, "wave": 0.125})


def test_reference_prevalence_foreign_label_raises() -> None:
    """Every key of the reference prevalence must be a schema label."""
    with pytest.raises(ValueError, match="not in labels"):
        _schema(reference_prevalence={"reach": 0.5, "jump": 0.5})


def test_reference_prevalence_is_kept() -> None:
    """A valid reference prevalence, over a subset of the labels, is kept as given."""
    schema = _schema(reference_prevalence={"reach": 0.75, "wave": 0.25})
    assert schema.reference_prevalence == {"reach": 0.75, "wave": 0.25}


def test_whole_clip_event_spans_clip() -> None:
    """A whole-clip label is an event from 0 to the clip's duration, with no confidence."""
    event = whole_clip_event("wave", 12.5, "model_x")
    assert (event.onset_s, event.offset_s, event.label, event.source) == (0.0, 12.5, "wave", "model_x")
    assert (event.confidence, event.confidence_kind) == (None, "none")


@pytest.mark.parametrize("field", ["onset_s", "offset_s", "confidence"])
@pytest.mark.parametrize("bad", [True, "1.0", float("nan"), float("inf")])
def test_numbers_are_strict_and_finite(field: str, bad: object) -> None:
    """Times and confidences refuse bools, strings, NaN and infinities."""
    fields: dict = {"onset_s": 0.0, "offset_s": 2.0, "label": "reach", "source": "annotator_a"}
    fields.update({"confidence": 0.5, "confidence_kind": "prob", field: bad})
    with pytest.raises(ValueError):
        BehaviorEvent(**fields)


def test_integer_times_are_accepted_as_floats() -> None:
    """An int is a valid time and is stored as a float."""
    event = _event(0, 2)
    assert (event.onset_s, event.offset_s) == (0.0, 2.0) and isinstance(event.onset_s, float)


@pytest.mark.parametrize("field", ["subject_ref", "track_id"])
def test_empty_optional_strings_raise(field: str) -> None:
    """An optional key is absent (`None`) or non-empty, never the empty string."""
    with pytest.raises(ValueError):
        _event(0.0, 1.0, **{field: ""})


def test_track_events_are_an_immutable_tuple() -> None:
    """The stored events cannot be appended to behind the sort."""
    track = EventTrack(schema_id="motion", events=[_event(1.0, 2.0), _event(0.0, 1.0)])
    assert isinstance(track.events, tuple)
    assert not hasattr(track.events, "append")


def test_schema_refuses_unknown_keys() -> None:
    """A schema with an unknown key is refused; callers map their metadata first."""
    with pytest.raises(ValueError):
        _schema(owner="someone")


def test_yaml_boolean_background_label_raises(tmp_path: Path) -> None:
    """An unquoted ``no`` as the background label gets the same hint as a label."""
    text = "schema_id: answer\ntask_type: classification\nlabels: ['a', 'b']\nbackground_label: no\n"
    with pytest.raises(ValueError, match="background_label loaded as a YAML boolean; quote it"):
        AnnotationSchema.from_yaml(_write(tmp_path, text))


def test_yaml_boolean_prevalence_key_raises(tmp_path: Path) -> None:
    """An unquoted ``yes`` as a reference-prevalence key gets the same hint."""
    text = (
        "schema_id: answer\ntask_type: classification\nlabels: ['yes', 'no']\n"
        "reference_prevalence: {yes: 0.5, 'no': 0.5}\n"
    )
    with pytest.raises(ValueError, match="reference_prevalence key True loaded as a YAML boolean; quote it"):
        AnnotationSchema.from_yaml(_write(tmp_path, text))
