"""Recordings, channel groups and audio."""

import copy
import hashlib
from pathlib import Path

import numpy
import pytest
from pydantic import ValidationError
from senselab_core.kernel._base import value_digest
from senselab_core.kernel.signals import (
    Audio,
    ChannelInfo,
    IrregularGroup,
    PlainText,
    Recording,
    RegularGroup,
    RegularSeries,
    SourceStamp,
    Text,
    audio_channels,
)
from senselab_core.kernel.time import TimeAnchor
from senselab_worker.canonical import canonical_npy_bytes
from senselab_worker.digest import identity_digest

RATE = 8_000


def _group(channels: int, rate: float, samples: int, kind: str = "ecg") -> RegularGroup:
    return RegularGroup(
        channels=tuple(ChannelInfo(name=f"c{i}", type=kind, units="mV") for i in range(channels)),
        sampling_rate=rate,
        data=numpy.zeros((channels, samples), dtype=numpy.float32),
    )


@pytest.mark.logic
def test_multirate_recording() -> None:
    """One recording holds groups at different rates."""
    r = Recording(
        groups={
            "ecg": RegularGroup(
                channels=(ChannelInfo(name="I", type="ecg", units="mV"),),
                sampling_rate=250.0,
                data=numpy.zeros((1, 250)),
            ),
            "acc": RegularGroup(
                channels=tuple(ChannelInfo(name=a, type="acc", units="g") for a in "xyz"),
                sampling_rate=50.0,
                data=numpy.zeros((3, 50)),
            ),
        },
        anchor=TimeAnchor(clock_id="dev", absolute_start_ns=None),
    )
    ecg, acc = r.groups["ecg"], r.groups["acc"]
    assert isinstance(ecg, RegularGroup) and ecg.sampling_rate == 250.0
    assert isinstance(acc, RegularGroup) and acc.data is not None and acc.data.shape == (3, 50)


@pytest.mark.logic
def test_recording_refuses_no_groups_or_an_empty_identity() -> None:
    """A recording has at least one group, and a supplied identity is a content identity."""
    anchor = TimeAnchor(clock_id="dev", absolute_start_ns=None)
    with pytest.raises(ValidationError):
        Recording(groups={}, anchor=anchor)
    with pytest.raises(ValidationError):
        Recording(groups={"ecg": _group(1, 250.0, 10)}, anchor=anchor, identity="")


@pytest.mark.logic
def test_regular_group_refuses_inconsistent_layout() -> None:
    """Channel count must match the data rows, data must be 2-D, and the rate positive and finite."""
    with pytest.raises(ValidationError, match="channels"):
        RegularGroup(
            channels=(ChannelInfo(name="a", type="ecg", units="mV"),), sampling_rate=1.0, data=numpy.zeros((2, 5))
        )
    with pytest.raises(ValidationError):
        RegularGroup(channels=(ChannelInfo(name="a", type="ecg", units="mV"),), sampling_rate=1.0, data=numpy.zeros(5))
    for rate in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            _group(1, rate, 5)
    with pytest.raises(ValidationError):
        RegularGroup(channels=(), sampling_rate=1.0, data=numpy.zeros((0, 5)))


@pytest.mark.logic
def test_regular_group_refuses_duplicate_channel_names_and_needs_one_backing() -> None:
    """Channel names are unique, and a group has exactly one of `data` and `source`."""
    a = ChannelInfo(name="a", type="ecg", units="mV")
    with pytest.raises(ValidationError, match="unique"):
        RegularGroup(channels=(a, a), sampling_rate=1.0, data=numpy.zeros((2, 5)))
    with pytest.raises(ValidationError):
        RegularGroup(channels=(a,), sampling_rate=1.0)


@pytest.mark.logic
def test_regular_group_data_is_read_only() -> None:
    """The model's array cannot be written through."""
    g = _group(1, 10.0, 5)
    assert g.data is not None
    with pytest.raises(ValueError):
        g.data[0, 0] = 1.0


@pytest.mark.logic
def test_regular_group_read_selects_samples_in_half_open_range() -> None:
    """`read` returns the samples whose times lie in [start_ns, end_ns)."""
    g = RegularGroup(
        channels=(ChannelInfo(name="a", type="ecg", units="mV"),),
        sampling_rate=4.0,
        data=numpy.arange(8.0).reshape(1, 8),
    )
    assert g.read().tolist() == [list(range(8))]
    assert g.read(250_000_000, 1_000_000_000).tolist() == [[1.0, 2.0, 3.0]]
    assert g.read(250_000_001, 1_000_000_001).tolist() == [[2.0, 3.0, 4.0]]
    assert g.read(start_ns=1_500_000_000).tolist() == [[6.0, 7.0]]
    assert g.read(5_000_000_000, 6_000_000_000).shape == (1, 0)


@pytest.mark.logic
def test_regular_group_read_refuses_bad_ranges() -> None:
    """Negative starts, inverted ranges and float times are refused."""
    g = _group(1, 4.0, 8)
    for start, end in ((-1, 10), (10, 5)):
        with pytest.raises(ValueError):
            g.read(start, end)
    with pytest.raises(TypeError):
        g.read(0.5, 10)  # type: ignore[arg-type]


@pytest.mark.logic
def test_irregular_group_validates_timestamps() -> None:
    """Timestamps are strictly increasing integers, one per data column."""
    ch = (ChannelInfo(name="hr", type="hr", units="bpm"),)
    g = IrregularGroup(channels=ch, timestamps_ns=numpy.array([0, 7, 30]), data=numpy.ones((1, 3)))
    assert g.timestamps_ns.dtype == numpy.int64 and not g.timestamps_ns.flags.writeable
    for ts, data in (
        (numpy.array([0, 7]), numpy.ones((1, 3))),
        (numpy.array([0, 7, 7]), numpy.ones((1, 3))),
        (numpy.array([0.0, 1.0, 2.0]), numpy.ones((1, 3))),
        (numpy.array([0, 1, 2]), numpy.ones((2, 3))),
        (numpy.array([[0, 1, 2]]), numpy.ones((1, 3))),
    ):
        with pytest.raises(ValidationError):
            IrregularGroup(channels=ch, timestamps_ns=ts, data=data)


@pytest.mark.logic
def test_audio_from_array_identity_construction() -> None:
    """In-memory audio's identity hashes the canonical array with the sampling rate and the channels."""
    x = numpy.linspace(-1, 1, 50, dtype=numpy.float32)
    a = Audio.from_array(x, 16_000)
    npy = hashlib.sha256(canonical_npy_bytes(x.reshape(1, -1))).hexdigest()
    channels = [{"name": "ch0", "type": "audio", "units": "FS", "placement": None, "reference": None}]
    expected = "content:" + identity_digest("audio@4", {"npy": npy, "sampling_rate": 16_000, "channels": channels})
    assert a.identity == expected
    assert Audio.from_array(x.reshape(1, -1), 16_000.0).identity == expected
    assert Audio.from_array(x, 8_000).identity != expected
    named = (ChannelInfo(name="left", type="audio", units="Pa"),)
    relabelled = RegularGroup(channels=named, sampling_rate=16_000, data=x.reshape(1, -1))
    assert Audio.from_group(relabelled).identity != expected
    group = a.groups["audio"]
    assert isinstance(group, RegularGroup) and group.read().shape == (1, 50)


@pytest.mark.logic
def test_audio_refuses_non_audio_layouts() -> None:
    """Audio is one regular group named `audio` whose channels are all audio with one unit."""
    anchor = TimeAnchor(clock_id="c", absolute_start_ns=None)
    with pytest.raises(ValidationError, match="exactly one regular group"):
        Audio(groups={"ecg": _group(1, 8000.0, 10, kind="audio")}, anchor=anchor)
    with pytest.raises(ValidationError, match="type 'audio'"):
        Audio(groups={"audio": _group(1, 8000.0, 10, kind="ecg")}, anchor=anchor)
    with pytest.raises(ValidationError, match="exactly one regular group"):
        Audio(groups={"audio": _group(1, 8000.0, 10, kind="audio"), "b": _group(1, 8.0, 1)}, anchor=anchor)
    held = Audio(groups={"audio": _group(1, 8000.0, 10, kind="audio")}, anchor=anchor).groups["audio"]
    assert isinstance(held, RegularGroup) and held.frames == 10
    mixed = RegularGroup(
        channels=(ChannelInfo(name="a", type="audio", units="FS"), ChannelInfo(name="b", type="audio", units="Pa")),
        sampling_rate=8000.0,
        data=numpy.zeros((2, 4), dtype=numpy.float32),
    )
    with pytest.raises(ValidationError, match="units"):
        Audio(groups={"audio": mixed}, anchor=anchor)
    with pytest.raises(ValueError):
        Audio.from_array(numpy.zeros((1, 2, 3)), 8000)


@pytest.mark.logic
def test_recording_groups_are_read_only() -> None:
    """The group mapping of a recording cannot be changed, and still serialises as a dict."""
    r = Recording(groups={"ecg": _group(1, 250.0, 10)}, anchor=TimeAnchor(clock_id="dev", absolute_start_ns=None))
    with pytest.raises(TypeError):
        r.groups["acc"] = _group(1, 50.0, 10)  # type: ignore[index]
    with pytest.raises(TypeError):
        del r.groups["ecg"]  # type: ignore[attr-defined]
    assert list(r.model_dump()["groups"]) == ["ecg"]


@pytest.mark.logic
def test_recording_round_trips_through_model_validate() -> None:
    """A recording's dump validates back to a recording with the same groups."""
    r = Recording(groups={"ecg": _group(1, 250.0, 10)}, anchor=TimeAnchor(clock_id="dev", absolute_start_ns=None))
    back = Recording.model_validate(r.model_dump())
    assert list(back.groups) == ["ecg"] and back.identity == r.identity and back == r
    with pytest.raises(TypeError):
        del back.groups["ecg"]  # type: ignore[attr-defined]


@pytest.mark.logic
def test_int16_reads_full_scale() -> None:
    """Int16 samples are held as float32 full scale, as the decoder reads them."""
    g = Audio.from_array(numpy.array([16384, -32768], dtype=numpy.int16), 16_000).groups["audio"]
    assert isinstance(g, RegularGroup)
    assert g.read().dtype == numpy.float32 and g.read().tolist() == [[0.5, -1.0]]


@pytest.mark.logic
@pytest.mark.parametrize("rate", [True, "16000", float("inf"), float("nan"), 0, -5])
def test_rate_refuses_bool_str_nonfinite(rate: object) -> None:
    """A sampling rate is a positive finite number; bools and strings are not rates."""
    with pytest.raises(ValidationError):
        Audio.from_array(numpy.zeros(4, dtype=numpy.float32), rate)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _group(1, rate, 4)  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize(
    "samples",
    [
        numpy.zeros(4, dtype=numpy.complex64),
        numpy.zeros(4, dtype=bool),
        numpy.array([0.0, None, 1.0], dtype=object),
        numpy.zeros(0, dtype=numpy.float32),
        numpy.zeros((2, 0), dtype=numpy.float32),
        numpy.zeros(4, dtype=numpy.int64),
        numpy.zeros(4, dtype="datetime64[ns]"),
    ],
    ids=["complex", "bool", "object", "empty", "empty-2d", "int64", "datetime"],
)
def test_from_array_refuses_complex_bool_empty(samples: numpy.ndarray) -> None:
    """Complex, boolean, object, datetime, empty and unconvertible integer samples are refused."""
    with pytest.raises(ValueError):
        Audio.from_array(samples, RATE)


@pytest.mark.logic
def test_audio_holds_float32_only() -> None:
    """An audio group's held samples are float32 full scale; other dtypes are converted by `from_array` only."""
    anchor = TimeAnchor(clock_id="c", absolute_start_ns=None)
    g = RegularGroup(
        channels=(ChannelInfo(name="a", type="audio", units="FS"),), sampling_rate=8.0, data=numpy.zeros((1, 4))
    )
    with pytest.raises(ValidationError, match="float32"):
        Audio(groups={"audio": g}, anchor=anchor)


@pytest.mark.logic
def test_recording_identity_is_independent_of_the_class_path() -> None:
    """A recording's identity names a declared type token, not the Python class that holds it."""
    groups = {"ecg": _group(1, 250.0, 10)}
    anchor = TimeAnchor(clock_id="dev", absolute_start_ns=None)
    moved = type("Moved", (Recording,), {"__module__": "elsewhere.module"})
    base = Recording(groups=groups, anchor=anchor)
    assert moved(groups=groups, anchor=anchor).identity == base.identity
    expected = value_digest("recording", {"groups": {"ecg": base.groups["ecg"].model_dump()}})
    assert base.identity == "content:" + expected


@pytest.mark.logic
def test_recording_identity_is_required_or_derived() -> None:
    """The identity field has no placeholder default: validation derives it or checks the supplied one."""
    assert Recording.model_fields["identity"].is_required()
    r = Recording(groups={"ecg": _group(1, 250.0, 10)}, anchor=TimeAnchor(clock_id="dev", absolute_start_ns=None))
    assert r.identity.startswith("content:") and Recording.model_validate(r.model_dump()) == r


@pytest.mark.logic
@pytest.mark.parametrize("dtype", ["int16", "int32"])
def test_float32_pcm_conversion_is_bit_identical_to_float64(dtype: str) -> None:
    """Integer PCM converted in float32 equals, bit for bit, the float64 division cast to float32."""
    info = numpy.iinfo(dtype)
    x = numpy.random.default_rng(1).integers(info.min, info.max, size=4096, endpoint=True, dtype=dtype)
    x[:3] = [info.min, info.max, 0]
    held = Audio.from_array(x, RATE).groups["audio"]
    assert isinstance(held, RegularGroup)
    expected = (x.astype(numpy.float64) / float(-info.min)).astype(numpy.float32)
    assert held.read()[0].tobytes() == expected.tobytes()


def _stamp(frames: int = 8, channels: int = 2) -> SourceStamp:
    return SourceStamp(
        size=1,
        mtime_ns=0,
        inode=0,
        device=0,
        frames=frames,
        channels=audio_channels(channels),
        sampling_rate=RATE,
        header="h",
        chunk_size=4,
        chunks=("d" * 64,),
        content="c" * 64,
    )


class _Reader:
    """A `SourceReader` that returns `samples` and records each call."""

    def __init__(self, samples: numpy.ndarray) -> None:
        self.samples = samples
        self.calls: list[tuple[Path, int, int]] = []

    def decode(self, source: Path, stamp: SourceStamp, first: int, count: int) -> numpy.ndarray:
        self.calls.append((source, first, count))
        return self.samples


@pytest.mark.logic
def test_a_source_group_validates_without_its_file(tmp_path: Path) -> None:
    """A source group needs the stamp a reader took; validating it reads no file, so the file may be absent."""
    missing = tmp_path / "absent.wav"
    group = RegularGroup(channels=audio_channels(2), sampling_rate=RATE, source=missing, source_stamp=_stamp())
    audio = Audio.from_group(group)
    assert audio.identity == "content:" + "c" * 64 and group.frames == 8
    with pytest.raises(ValidationError, match="stamp"):
        RegularGroup(channels=audio_channels(2), sampling_rate=RATE, source=missing)
    with pytest.raises(ValidationError, match="channels"):
        RegularGroup(channels=audio_channels(1), sampling_rate=RATE, source=missing, source_stamp=_stamp())


@pytest.mark.logic
def test_a_source_group_is_read_only_through_a_reader(tmp_path: Path) -> None:
    """`read` hands the frame range to the reader, refuses no reader, and checks what the reader returns."""
    group = RegularGroup(channels=audio_channels(2), sampling_rate=RATE, source=tmp_path / "x", source_stamp=_stamp())
    with pytest.raises(ValueError, match="SourceReader"):
        group.read()
    reader = _Reader(numpy.ones((2, 3), dtype=numpy.float32))
    assert group.read(10**9 // RATE, 4 * 10**9 // RATE, reader=reader).shape == (2, 3)
    assert reader.calls == [(tmp_path / "x", 1, 3)]
    for wrong in (numpy.ones((1, 3), dtype=numpy.float32), numpy.ones((2, 3)), numpy.full((2, 3), numpy.nan, "f4")):
        with pytest.raises(ValueError):
            group.read(10**9 // RATE, 4 * 10**9 // RATE, reader=_Reader(wrong))


@pytest.mark.logic
def test_regular_group_satisfies_regular_series() -> None:
    """A regular group and audio both read as a `RegularSeries`: channels, rate, frames and `read`."""
    audio = Audio.from_array(numpy.array([[0.25, -0.5, 0.75], [0.0, 0.5, -1.0]], dtype=numpy.float32), RATE)
    group = audio.groups["audio"]
    series: list[RegularSeries] = [group, audio] if isinstance(group, RegularGroup) else []
    assert len(series) == 2 and all(isinstance(s, RegularSeries) for s in series)
    for s in series:
        assert (s.channels, s.sampling_rate, s.frames) == (audio_channels(2), RATE, 3)
        assert s.read(0, 2 * 10**9 // RATE).tolist() == [[0.25, -0.5], [0.0, 0.5]]
    assert not isinstance(IrregularGroup(channels=audio_channels(1), timestamps_ns=[0], data=[[0.0]]), RegularSeries)


@pytest.mark.logic
def test_plain_text_is_canonical_and_content_identified() -> None:
    """Plain text is held in canonical form, its identity derived from it; a forged identity is refused."""
    composed, decomposed = PlainText(content="caf\u00e9"), PlainText(content="cafe\u0301")
    assert isinstance(composed, Text) and composed.content == decomposed.content == "caf\u00e9"
    assert composed.identity == decomposed.identity == "content:" + value_digest("text", {"content": "caf\u00e9"})
    assert composed.identity != PlainText(content="cafe").identity and composed.format == "text@1.0"
    with pytest.raises(ValidationError, match="identity"):
        PlainText(content="cafe", identity=composed.identity)


@pytest.mark.logic
def test_samples_carry_their_files_stamp_only_as_decoded_from_it(tmp_path: Path) -> None:
    """Decoded samples keep the file's identity; other held samples under its stamp, or changed ones, are refused."""
    samples = numpy.arange(16, dtype=numpy.float32).reshape(2, 8) / 16
    source = RegularGroup(channels=audio_channels(2), sampling_rate=RATE, source=tmp_path / "a", source_stamp=_stamp())
    decoded = RegularGroup.decoded_from(source, _Reader(samples))
    assert Audio.from_group(decoded).identity == "content:" + "c" * 64
    assert copy.deepcopy(decoded) == decoded and decoded.read().tolist() == samples.tolist()
    zeros = numpy.zeros((2, 8), dtype=numpy.float32)
    with pytest.raises(ValidationError, match="decoded_from"):
        RegularGroup(channels=audio_channels(2), sampling_rate=RATE, data=zeros, source_stamp=_stamp())
    with pytest.raises(ValidationError, match="decoded_from"):
        decoded.model_copy(update={"data": zeros})
    with pytest.raises(ValidationError, match="channels and frames"):
        decoded.model_copy(update={"source_stamp": _stamp(frames=7)})
    with pytest.raises(ValueError, match="source group"):
        RegularGroup.decoded_from(decoded, _Reader(samples))
