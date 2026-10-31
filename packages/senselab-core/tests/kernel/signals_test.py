"""Recordings, channel groups and audio."""

import builtins
import copy
import os
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
import soundfile
from pydantic import ValidationError
from senselab_core.kernel import signals
from senselab_core.kernel._base import value_digest
from senselab_core.kernel.signals import (
    Audio,
    ChannelInfo,
    IrregularGroup,
    Recording,
    RegularGroup,
    StaleSource,
)
from senselab_core.kernel.time import TimeAnchor
from senselab_worker.canonical import canonical_npy_bytes, tagged_digest
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
        data=numpy.zeros((channels, samples), dtype=numpy.float32),
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
def test_audio_identity_is_raw_bytes(tmp_path: Path) -> None:
    """A file-backed audio's identity is the content identity root of its bytes."""
    p = write_wav(tmp_path)
    a = Audio.from_path(p)
    assert a.identity == "content:" + content_identity(p)["root"]
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
    expected = "content:" + tagged_digest(
        "audio@2", {"npy": sha256_bytes(canonical_npy_bytes(x.reshape(1, -1))), "sampling_rate": 16_000}
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
def test_audio_from_path_refuses_a_file_that_is_not_audio(tmp_path: Path) -> None:
    """A file soundfile cannot parse raises its typed error."""
    p = tmp_path / "x.wav"
    p.write_bytes(b"not audio at all")
    with pytest.raises(soundfile.LibsndfileError):
        Audio.from_path(p)


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
def test_audio_from_path_reads_the_header_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening a file opens it as sound once, to read its header and stamp it."""
    p = write_wav(tmp_path)
    calls: list[object] = []
    original = soundfile.SoundFile.__init__

    def spy(self: soundfile.SoundFile, file: object, *args: object, **kwargs: object) -> None:
        calls.append(file)
        original(self, file, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(soundfile.SoundFile, "__init__", spy)
    Audio.from_path(p)
    assert len(calls) == 1


@pytest.mark.logic
def test_audio_pickles_and_deep_copies(tmp_path: Path) -> None:
    """Pickling and deep copying keep identity, groups and samples; copies stay read-only mappings."""
    for audio in (
        Audio.from_array(numpy.arange(6.0, dtype=numpy.float32), 8_000),
        Audio.from_path(write_wav(tmp_path)),
    ):
        for copied in (pickle.loads(pickle.dumps(audio)), audio.model_copy(deep=True)):
            assert copied.identity == audio.identity and list(copied.groups) == list(audio.groups)
            g, h = copied.groups["audio"], audio.groups["audio"]
            assert isinstance(g, RegularGroup) and isinstance(h, RegularGroup)
            assert g.frames == h.frames and numpy.array_equal(g.read(), h.read())
            with pytest.raises(TypeError):
                copied.groups["b"] = g  # type: ignore[index]


@pytest.mark.logic
def test_recording_round_trips_through_model_validate() -> None:
    """A recording's dump validates back to a recording with the same groups."""
    r = Recording(groups={"ecg": _group(1, 250.0, 10)}, anchor=TimeAnchor(clock_id="dev", absolute_start_ns=None))
    back = Recording.model_validate(r.model_dump())
    assert list(back.groups) == ["ecg"] and back.identity == r.identity and back == r
    with pytest.raises(TypeError):
        del back.groups["ecg"]  # type: ignore[attr-defined]


def _rewrite(path: Path, frames: int, value: float) -> None:
    """Rewrite `path` in place and move its mtime one second on, so the change shows on any mtime granularity."""
    before = path.stat().st_mtime_ns
    soundfile.write(path, numpy.full((frames, 2), value, dtype=numpy.float32), RATE, subtype="FLOAT")
    os.utime(path, ns=(before + 10**9, before + 10**9))


@pytest.mark.logic
@pytest.mark.parametrize("frames", [RATE, RATE // 4])
def test_overwritten_source_raises_stale_source(tmp_path: Path, frames: int) -> None:
    """A source rewritten after open, at the same or another length, is refused at read."""
    p = write_wav(tmp_path)
    group = Audio.from_path(p).groups["audio"]
    assert isinstance(group, RegularGroup)
    _rewrite(p, frames, 0.5)
    with pytest.raises(StaleSource):
        group.read()
    with pytest.raises(StaleSource):
        group.read(0, 1_000_000)


@pytest.mark.logic
def test_a14_rewritten_source_never_reads_under_the_old_identity(tmp_path: Path) -> None:
    """A-14: a WAV overwritten in place, or shrunk, cannot be read or copied under the old identity."""
    p = write_wav(tmp_path, frames=4 * RATE)
    a = Audio.from_path(p)
    _rewrite(p, RATE, 0.5)
    group = a.groups["audio"]
    assert isinstance(group, RegularGroup) and group.frames == 4 * RATE
    with pytest.raises(StaleSource):
        group.read()
    with pytest.raises((StaleSource, ValidationError)):
        a.model_copy()
    assert Audio.from_path(p).identity != a.identity


@pytest.mark.logic
def test_source_stamp_records_the_opened_file(tmp_path: Path) -> None:
    """A source group carries the stamp of the file it opened, and a data group carries none."""
    p = write_wav(tmp_path)
    group = Audio.from_path(p).groups["audio"]
    assert isinstance(group, RegularGroup) and group.source_stamp is not None
    st = p.stat()
    stamp = group.source_stamp
    assert (stamp.size, stamp.mtime_ns, stamp.inode, stamp.device) == (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev)
    assert stamp.frames == RATE
    data_group = Audio.from_array(numpy.zeros(4, dtype=numpy.float32), RATE).groups["audio"]
    assert isinstance(data_group, RegularGroup) and data_group.source_stamp is None


_LSB = {"int16": 2.0**-15, "int32": 2.0**-31, "float32": 0.0, "float64": 0.0}
_SUBTYPE = {"int16": "PCM_16", "int32": "PCM_32", "float32": "FLOAT", "float64": "DOUBLE"}


@pytest.mark.logic
@pytest.mark.parametrize("dtype", sorted(_LSB))
def test_from_array_matches_from_path(tmp_path: Path, dtype: str) -> None:
    """In-memory samples read as the same float32 full scale the decoder reads from a file of them."""
    rng = numpy.random.default_rng(0)
    if dtype.startswith("int"):
        info = numpy.iinfo(dtype)
        x = rng.integers(info.min, info.max, size=(2, 64), endpoint=True, dtype=dtype)
        x[:, :2] = [[info.min, info.max], [0, -1]]
    else:
        x = rng.uniform(-1, 1, size=(2, 64)).astype(dtype)
    path = tmp_path / "x.wav"
    soundfile.write(path, x.T, RATE, subtype=_SUBTYPE[dtype])
    held, decoded = Audio.from_array(x, RATE).groups["audio"], Audio.from_path(path).groups["audio"]
    assert isinstance(held, RegularGroup) and isinstance(decoded, RegularGroup)
    assert held.read().dtype == decoded.read().dtype == numpy.float32
    numpy.testing.assert_allclose(held.read(), decoded.read(), rtol=0, atol=_LSB[dtype])


@pytest.mark.logic
def test_a16_int16_reads_full_scale() -> None:
    """A-16: int16 samples are held as float32 full scale, as the decoder reads them."""
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


@pytest.fixture
def open_spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record the path of every `open` and `SoundFile` construction."""
    calls: list[str] = []
    original_open = builtins.open
    original_sound = soundfile.SoundFile.__init__

    def spy_open(file: object, *args: object, **kwargs: object) -> object:
        calls.append(f"open:{file}")
        return original_open(file, *args, **kwargs)  # type: ignore[call-overload]

    def spy_sound(self: soundfile.SoundFile, file: object, *args: object, **kwargs: object) -> None:
        calls.append("sound")
        original_sound(self, file, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(soundfile.SoundFile, "__init__", spy_sound)
    yield calls


@pytest.mark.logic
def test_copies_of_a_source_recording_do_not_read_the_file(tmp_path: Path, open_spy: list[str]) -> None:
    """Opening hashes and stamps the file once; copies check it by stat and never read it."""
    p = write_wav(tmp_path)
    open_spy.clear()
    a = Audio.from_path(p)
    assert open_spy.count(f"open:{p}") == 2 and len(open_spy) == 3, open_spy
    open_spy.clear()
    r = Recording.model_validate(a.model_dump())
    copies = [copy.copy(a), copy.deepcopy(a), pickle.loads(pickle.dumps(a)), a.model_copy(), copy.deepcopy(r)]
    assert open_spy == []
    assert all(c.identity == a.identity for c in copies)


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
def test_memo_stays_at_its_bound_and_evicts_the_least_recently_used() -> None:
    """Inserting past the bound keeps the size at the bound; the least recently used entry goes first."""
    memo = signals.Memo(3)
    for i in range(10):
        memo.put(i, i)
        assert len(memo) == min(i + 1, 3)
    assert memo.get(7) == 7
    memo.put(10, 10)
    assert [memo.get(k) for k in (7, 8, 9, 10)] == [7, None, 9, 10]
    with pytest.raises(ValueError):
        signals.Memo(0)


@pytest.mark.logic
def test_source_memos_are_bounded_and_rederive_evicted_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With room for one file, opening a second evicts the first; a copy of the first re-derives its identity."""
    monkeypatch.setattr(signals, "_VERIFIED", signals.Memo(1))
    monkeypatch.setattr(signals, "_ROOTS", signals.Memo(1))
    hashed: list[object] = []
    original = signals.content_identity

    def spy(path: object, **kwargs: object) -> object:
        hashed.append(path)
        return original(path)  # type: ignore[arg-type]

    monkeypatch.setattr(signals, "content_identity", spy)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = Audio.from_path(write_wav(tmp_path / "a"))
    second = Audio.from_path(write_wav(tmp_path / "b", frames=RATE // 2))
    assert len(signals._VERIFIED) == len(signals._ROOTS) == 1 and len(hashed) == 2
    again = copy.copy(first)
    assert again == first and again.identity == first.identity != second.identity
    assert len(hashed) == 3 and len(signals._VERIFIED) == len(signals._ROOTS) == 1


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
