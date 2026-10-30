"""Recordings, channel groups and audio."""

from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
import soundfile
from pydantic import ValidationError
from senselab_core.kernel.signals import Audio, ChannelInfo, IrregularGroup, Recording, RegularGroup
from senselab_core.kernel.time import TimeAnchor
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.hashing import content_identity, sha256_bytes

RATE = 8_000


def write_wav(tmp_path: Path, frames: int = RATE, channels: int = 2, rate: int = RATE) -> Path:
    """Write a deterministic float WAV whose sample k of channel c is (k % 100 + c) / 1000."""
    k = numpy.arange(frames)
    data = numpy.stack([(k % 100 + c) / 1000 for c in range(channels)], axis=1).astype(numpy.float32)
    path = tmp_path / "x.wav"
    soundfile.write(path, data, rate, subtype="FLOAT")
    return path


def _group(channels: int, rate: float, samples: int, kind: str = "ecg") -> RegularGroup:
    return RegularGroup(
        channels=tuple(ChannelInfo(name=f"c{i}", type=kind, units="mV") for i in range(channels)),
        sampling_rate=rate,
        data=numpy.zeros((channels, samples)),
    )


@pytest.fixture
def read_spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Record the frame count of every `SoundFile.read` call."""
    calls: list[int] = []
    original = soundfile.SoundFile.read

    def spy(self: soundfile.SoundFile, frames: int = -1, *args: object, **kwargs: object) -> object:
        calls.append(frames)
        return original(self, frames, *args, **kwargs)

    monkeypatch.setattr(soundfile.SoundFile, "read", spy)
    yield calls


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
        identity="x",
    )
    ecg, acc = r.groups["ecg"], r.groups["acc"]
    assert isinstance(ecg, RegularGroup) and ecg.sampling_rate == 250.0
    assert isinstance(acc, RegularGroup) and acc.data is not None and acc.data.shape == (3, 50)


@pytest.mark.logic
def test_recording_refuses_no_groups_or_no_identity() -> None:
    """A recording has at least one group and an identity."""
    anchor = TimeAnchor(clock_id="dev", absolute_start_ns=None)
    with pytest.raises(ValidationError):
        Recording(groups={}, anchor=anchor, identity="x")
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
def test_audio_identity_is_raw_bytes(tmp_path: Path) -> None:
    """A file-backed audio's identity is the content identity root of its bytes."""
    p = write_wav(tmp_path)
    a = Audio.from_path(p)
    assert a.identity == content_identity(p)["root"]
    group = a.groups["audio"]
    assert isinstance(group, RegularGroup) and group.sampling_rate == RATE and len(group.channels) == 2
    assert group.frames == RATE and a.anchor.clock_id == a.identity


@pytest.mark.logic
def test_audio_time_range_read_is_lazy(tmp_path: Path, read_spy: list[int]) -> None:
    """Opening reads no samples; a range read decodes only the requested frames."""
    p = write_wav(tmp_path)
    a = Audio.from_path(p)
    assert read_spy == []
    group = a.groups["audio"]
    assert isinstance(group, RegularGroup)
    x = group.read(100_000_000, 200_000_000)
    assert read_spy == [800]
    k = numpy.arange(800, 1600)
    expected = numpy.stack([(k % 100 + c) / 1000 for c in range(2)]).astype(numpy.float32)
    assert x.shape == (2, 800) and x.dtype == numpy.float32 and numpy.array_equal(x, expected)


@pytest.mark.logic
def test_audio_from_path_reads_to_end_of_file(tmp_path: Path) -> None:
    """A range running past the end returns the frames up to the end."""
    group = Audio.from_path(write_wav(tmp_path, frames=100)).groups["audio"]
    assert isinstance(group, RegularGroup)
    assert group.read(start_ns=10_000_000).shape == (2, 20)


@pytest.mark.logic
def test_file_group_refuses_a_layout_the_file_does_not_have(tmp_path: Path) -> None:
    """A source-backed group's channel count and rate must match the file header."""
    p = write_wav(tmp_path)
    ch = (ChannelInfo(name="a", type="audio", units="FS"),)
    with pytest.raises(ValidationError, match="channels"):
        RegularGroup(channels=ch, sampling_rate=RATE, source=p)
    two = (ch[0], ChannelInfo(name="b", type="audio", units="FS"))
    with pytest.raises(ValidationError, match="rate"):
        RegularGroup(channels=two, sampling_rate=RATE * 2, source=p)


@pytest.mark.logic
def test_audio_from_array_identity_construction() -> None:
    """In-memory audio's identity hashes the canonical array with the sampling rate."""
    x = numpy.linspace(-1, 1, 50, dtype=numpy.float32)
    a = Audio.from_array(x, 16_000)
    expected = sha256_bytes(
        jcs_dumps({"npy": sha256_bytes(canonical_npy_bytes(x.reshape(1, -1))), "sampling_rate": 16_000})
    )
    assert a.identity == expected
    assert Audio.from_array(x.reshape(1, -1), 16_000.0).identity == expected
    assert Audio.from_array(x, 8_000).identity != expected
    group = a.groups["audio"]
    assert isinstance(group, RegularGroup) and group.read().shape == (1, 50)


@pytest.mark.logic
def test_audio_refuses_non_audio_layouts() -> None:
    """Audio is one regular group named `audio` whose channels are all audio with one unit."""
    anchor = TimeAnchor(clock_id="c", absolute_start_ns=None)
    with pytest.raises(ValidationError):
        Audio(groups={"ecg": _group(1, 8000.0, 10, kind="audio")}, anchor=anchor, identity="x")
    with pytest.raises(ValidationError):
        Audio(groups={"audio": _group(1, 8000.0, 10, kind="ecg")}, anchor=anchor, identity="x")
    with pytest.raises(ValidationError):
        Audio(
            groups={"audio": _group(1, 8000.0, 10, kind="audio"), "b": _group(1, 8.0, 1)}, anchor=anchor, identity="x"
        )
    mixed = RegularGroup(
        channels=(ChannelInfo(name="a", type="audio", units="FS"), ChannelInfo(name="b", type="audio", units="Pa")),
        sampling_rate=8000.0,
        data=numpy.zeros((2, 4)),
    )
    with pytest.raises(ValidationError, match="units"):
        Audio(groups={"audio": mixed}, anchor=anchor, identity="x")
    with pytest.raises(ValueError):
        Audio.from_array(numpy.zeros((1, 2, 3)), 8000)


@pytest.mark.logic
def test_audio_from_path_refuses_a_file_that_is_not_audio(tmp_path: Path) -> None:
    """A file soundfile cannot parse raises its typed error."""
    p = tmp_path / "x.wav"
    p.write_bytes(b"not audio at all")
    with pytest.raises(soundfile.LibsndfileError):
        Audio.from_path(p)


@pytest.mark.logic
def test_recording_groups_are_read_only() -> None:
    """The group mapping of a recording cannot be changed, and still serialises as a dict."""
    r = Recording(
        groups={"ecg": _group(1, 250.0, 10)}, anchor=TimeAnchor(clock_id="dev", absolute_start_ns=None), identity="x"
    )
    with pytest.raises(TypeError):
        r.groups["acc"] = _group(1, 50.0, 10)  # type: ignore[index]
    with pytest.raises(TypeError):
        del r.groups["ecg"]  # type: ignore[attr-defined]
    assert list(r.model_dump()["groups"]) == ["ecg"]


@pytest.mark.logic
def test_audio_from_path_reads_the_header_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening a file reads its header once."""
    p = write_wav(tmp_path)
    calls: list[str] = []
    original = soundfile.info

    def spy(file: str, verbose: bool = False) -> object:
        calls.append(file)
        return original(file, verbose)

    monkeypatch.setattr(soundfile, "info", spy)
    Audio.from_path(p)
    assert calls == [str(p)]
