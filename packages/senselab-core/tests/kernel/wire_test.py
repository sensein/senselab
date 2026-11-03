"""Wire forms of kernel values: tagged plain data an isolated worker reads, decoded back by format tag."""

import json
import operator
from pathlib import Path

import numpy
import pytest
import soundfile
from senselab_core.kernel._base import Frozen
from senselab_core.kernel.declare import kernel_types
from senselab_core.kernel.formats import Formatted
from senselab_core.kernel.results import Metric, TextSpans, Transcript, Trials
from senselab_core.kernel.signals import Audio, PlainText, Recording, RegularGroup
from senselab_core.kernel.wire import (
    WIRE_FORMATS,
    WIRE_MIGRATION_STATEMENT,
    WireFormatRetired,
    from_wire_form,
    wire_form,
)
from senselab_core.runtime import audio_io
from senselab_core.testing.values import samples_of
from senselab_worker.codec import from_wire, to_wire

NEW_TYPES = (PlainText, Transcript, TextSpans, Trials, Metric)
PROTOCOL_IMPLEMENTATIONS = (RegularGroup, Audio, PlainText)
WIRE_TYPES: list[type[Frozen]] = sorted(WIRE_FORMATS.values(), key=operator.attrgetter("__name__"))


def _through_the_wire(value: Frozen, array_dir: Path) -> Frozen:
    """Encode `value` as a worker input, frame it as JSON, and decode it as the worker and the host would."""
    form = wire_form(value, reader=audio_io.READER)
    wire = json.loads(json.dumps(to_wire(form, array_dir=array_dir, at="inputs")))
    return from_wire_form(from_wire(wire, array_dir=array_dir))


@pytest.mark.contract
@pytest.mark.parametrize("cls", WIRE_TYPES, ids=operator.attrgetter("__name__"))
def test_every_new_type_round_trips_by_format_tag(cls: type[Frozen], tmp_path: Path) -> None:
    """Every sample of every wire type comes back through the codec by its format tag.

    A value held in memory comes back equal; a file-backed one with the same identity and samples.
    """
    for value in samples_of(cls):
        back = _through_the_wire(value, tmp_path)
        if not _file_backed(value):
            assert back == value
            continue
        assert getattr(back, "identity", None) == getattr(value, "identity", None)
        assert isinstance(back, RegularGroup | Recording) and isinstance(value, RegularGroup | Recording)
        assert _samples(back).tolist() == _samples(value).tolist()


@pytest.mark.contract
def test_every_port_type_has_a_wire_form() -> None:
    """Every concrete kernel type a port may name, every port protocol's implementation and every new type has one."""
    ported = {t for t in kernel_types().values() if t.__module__.startswith("senselab_core.kernel.")}
    concrete = {t for t in ported if issubclass(t, (Recording, Formatted)) and t is not Formatted}
    wanted = concrete | set(NEW_TYPES + PROTOCOL_IMPLEMENTATIONS)
    missing = sorted(t.__name__ for t in wanted if t not in WIRE_FORMATS.values())
    assert not missing


@pytest.mark.logic
def test_a_file_backed_audio_travels_as_its_samples(tmp_path: Path) -> None:
    """A file-backed audio travels as its samples and its file's stamp, never the path, and keeps its identity."""
    path = tmp_path / "tone.wav"
    soundfile.write(path, numpy.array([0.25, -0.5, 0.75], dtype=numpy.float32), 16_000, subtype="FLOAT")
    audio = audio_io.open_audio(path)
    form = wire_form(audio, reader=audio_io.READER)
    assert str(path) not in json.dumps(to_wire(form, array_dir=tmp_path, at="inputs"))
    assert form["identity"] == audio.identity and form["anchor"]["clock_id"] == audio.anchor.clock_id
    back = from_wire_form(form)
    assert isinstance(back, Audio) and back.identity == audio.identity and back.anchor == audio.anchor
    assert back.read().tolist() == audio.read(reader=audio_io.READER).tolist()
    with pytest.raises(ValueError, match="SourceReader"):
        wire_form(audio)


@pytest.mark.logic
def test_unknown_or_retired_tags_are_refused() -> None:
    """A wire form of an unregistered name or another major version is refused; a later minor is read."""
    form = wire_form(Metric(name="wer", value=0.1, unit="1", items=1))
    later = from_wire_form({**form, "format": "metric@2.7", "added": 1})
    assert later.model_dump(exclude={"format"}) == from_wire_form(form).model_dump(exclude={"format"})
    for tag in ("metric@3.0", "nothing@1.0", None):
        with pytest.raises(ValueError, match="format"):
            from_wire_form({**form, "format": tag})


@pytest.mark.logic
def test_metric1_refused() -> None:
    """A `metric@1` wire form is refused as retired, naming the migration statement; `metric@2` is read."""
    form = wire_form(Metric(name="wer", value=0.5, unit="1", items=2, numerator=1, denominator=2))
    assert form["format"] == "metric@2.0" and from_wire_form(form).model_dump() == form
    with pytest.raises(WireFormatRetired, match=WIRE_MIGRATION_STATEMENT):
        from_wire_form({**form, "format": "metric@1.0"})


@pytest.mark.logic
@pytest.mark.parametrize(
    "tag", ["recording@1.0", "audio@1.0", "regular-group@1.3", "recording@2.1", "audio@2.1", "regular-group@2.1"]
)
def test_retired_signal_forms_name_their_format_and_migration(tag: str) -> None:
    """A signal wire form of a retired major is refused with a typed error naming it and the migration statement."""
    form = wire_form(Audio.from_array(numpy.zeros(4, dtype=numpy.float32), 8_000))
    with pytest.raises(WireFormatRetired, match=WIRE_MIGRATION_STATEMENT) as info:
        from_wire_form({**form, "format": tag})
    assert info.value.found == tag and tag in str(info.value)


def _file_backed(value: Frozen) -> bool:
    groups = value.groups.values() if isinstance(value, Recording) else [value]
    return any(isinstance(g, RegularGroup) and g.source is not None for g in groups)


def _samples(value: RegularGroup | Recording) -> numpy.ndarray:
    groups = value.groups.values() if isinstance(value, Recording) else [value]
    return numpy.concatenate([g.read(reader=audio_io.READER) for g in groups if isinstance(g, RegularGroup)])
