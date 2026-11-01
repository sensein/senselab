"""Sound files read through the runtime: stamping, lazy decoding, staleness and the stamp memo."""

import builtins
import copy
import multiprocessing as mp
import os
import pickle
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy
import pytest
import soundfile
from pydantic import ValidationError
from senselab_core.kernel.signals import Audio, ChannelInfo, Recording, RegularGroup, StaleSource
from senselab_core.runtime import audio_io
from senselab_core.runtime.audio_io import READER, open_audio, source_group
from senselab_worker.hashing import content_identity

RATE = 8_000


def write_wav(tmp_path: Path, frames: int = RATE, channels: int = 2, rate: int = RATE) -> Path:
    """Write a deterministic float WAV whose sample k of channel c is (k % 100 + c) / 1000."""
    k = numpy.arange(frames)
    data = numpy.stack([(k % 100 + c) / 1000 for c in range(channels)], axis=1).astype(numpy.float32)
    path = tmp_path / "x.wav"
    soundfile.write(path, data, rate, subtype="FLOAT")
    return path


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
def test_audio_identity_is_raw_bytes(tmp_path: Path) -> None:
    """A file-backed audio's identity is the content identity root of its bytes."""
    p = write_wav(tmp_path)
    a = open_audio(p)
    assert a.identity == "content:" + content_identity(p)["root"]
    group = a.groups["audio"]
    assert isinstance(group, RegularGroup) and group.sampling_rate == RATE and len(group.channels) == 2
    assert group.frames == RATE and a.anchor.clock_id == a.identity


@pytest.mark.logic
def test_audio_time_range_read_is_lazy(tmp_path: Path, read_spy: list[int]) -> None:
    """Opening reads no samples; a range read decodes only the requested frames."""
    p = write_wav(tmp_path)
    a = open_audio(p)
    assert read_spy == []
    group = a.groups["audio"]
    assert isinstance(group, RegularGroup)
    x = group.read(100_000_000, 200_000_000, reader=READER)
    assert read_spy == [800]
    k = numpy.arange(800, 1600)
    expected = numpy.stack([(k % 100 + c) / 1000 for c in range(2)]).astype(numpy.float32)
    assert x.shape == (2, 800) and x.dtype == numpy.float32 and numpy.array_equal(x, expected)


@pytest.mark.logic
def test_open_audio_reads_to_end_of_file(tmp_path: Path) -> None:
    """A range running past the end returns the frames up to the end."""
    group = open_audio(write_wav(tmp_path, frames=100)).groups["audio"]
    assert isinstance(group, RegularGroup)
    assert group.read(start_ns=10_000_000, reader=READER).shape == (2, 20)


@pytest.mark.logic
def test_file_group_refuses_a_layout_the_file_does_not_have(tmp_path: Path) -> None:
    """A source-backed group's channel count and rate must match the file header."""
    p = write_wav(tmp_path)
    ch = (ChannelInfo(name="a", type="audio", units="FS"),)
    with pytest.raises(ValidationError, match="channels"):
        source_group(ch, RATE, p)
    two = (ch[0], ChannelInfo(name="b", type="audio", units="FS"))
    with pytest.raises(ValidationError, match="rate"):
        source_group(two, RATE * 2, p)


@pytest.mark.logic
def test_open_audio_refuses_a_file_that_is_not_audio(tmp_path: Path) -> None:
    """A file soundfile cannot parse raises its typed error."""
    p = tmp_path / "x.wav"
    p.write_bytes(b"not audio at all")
    with pytest.raises(soundfile.LibsndfileError):
        open_audio(p)


@pytest.mark.logic
def test_open_audio_reads_the_header_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening a file opens it as sound once, to read its header and stamp it."""
    p = write_wav(tmp_path)
    calls: list[object] = []
    original = soundfile.SoundFile.__init__

    def spy(self: soundfile.SoundFile, file: object, *args: object, **kwargs: object) -> None:
        calls.append(file)
        original(self, file, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(soundfile.SoundFile, "__init__", spy)
    open_audio(p)
    assert len(calls) == 1


@pytest.mark.logic
def test_audio_pickles_and_deep_copies(tmp_path: Path) -> None:
    """Pickling and deep copying keep identity, groups and samples; copies stay read-only mappings."""
    for audio in (
        Audio.from_array(numpy.arange(6.0, dtype=numpy.float32), 8_000),
        open_audio(write_wav(tmp_path)),
    ):
        for copied in (pickle.loads(pickle.dumps(audio)), audio.model_copy(deep=True)):
            assert copied.identity == audio.identity and list(copied.groups) == list(audio.groups)
            g, h = copied.groups["audio"], audio.groups["audio"]
            assert isinstance(g, RegularGroup) and isinstance(h, RegularGroup)
            assert g.frames == h.frames and numpy.array_equal(g.read(reader=READER), h.read(reader=READER))
            with pytest.raises(TypeError):
                copied.groups["b"] = g  # type: ignore[index]


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
    group = open_audio(p).groups["audio"]
    assert isinstance(group, RegularGroup)
    _rewrite(p, frames, 0.5)
    with pytest.raises(StaleSource):
        group.read(reader=READER)
    with pytest.raises(StaleSource):
        group.read(0, 1_000_000, reader=READER)


@pytest.mark.logic
def test_rewritten_source_never_reads_under_the_old_identity(tmp_path: Path) -> None:
    """A WAV overwritten in place, or shrunk, cannot be read or re-admitted under the old identity."""
    p = write_wav(tmp_path, frames=4 * RATE)
    a = open_audio(p)
    _rewrite(p, RATE, 0.5)
    group = a.groups["audio"]
    assert isinstance(group, RegularGroup) and group.frames == 4 * RATE
    with pytest.raises(StaleSource):
        group.read(reader=READER)
    assert group.source_stamp is not None
    with pytest.raises(StaleSource):
        source_group(group.channels, RATE, p, group.source_stamp)
    assert open_audio(p).identity != a.identity


@pytest.mark.logic
def test_source_stamp_records_the_opened_file(tmp_path: Path) -> None:
    """A source group carries the stamp of the file it opened, and a data group carries none."""
    p = write_wav(tmp_path)
    group = open_audio(p).groups["audio"]
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
def test_from_array_matches_open_audio(tmp_path: Path, dtype: str) -> None:
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
    held, decoded = Audio.from_array(x, RATE).groups["audio"], open_audio(path).groups["audio"]
    assert isinstance(held, RegularGroup) and isinstance(decoded, RegularGroup)
    assert held.read().dtype == decoded.read(reader=READER).dtype == numpy.float32
    numpy.testing.assert_allclose(held.read(), decoded.read(reader=READER), rtol=0, atol=_LSB[dtype])


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
    """Opening hashes and stamps the file through one open; copies check it by stat and never read it."""
    p = write_wav(tmp_path)
    open_spy.clear()
    a = open_audio(p)
    assert open_spy.count(f"open:{p}") == 1 and len(open_spy) == 2, open_spy
    open_spy.clear()
    r = Recording.model_validate(a.model_dump())
    copies = [copy.copy(a), copy.deepcopy(a), pickle.loads(pickle.dumps(a)), a.model_copy(), copy.deepcopy(r)]
    assert open_spy == []
    assert all(c.identity == a.identity for c in copies)


@pytest.mark.logic
def test_memo_stays_at_its_bound_and_evicts_the_least_recently_used() -> None:
    """Inserting past the bound keeps the size at the bound; the least recently used entry goes first."""
    memo = audio_io.Memo(3)
    for i in range(10):
        memo.put(i, i)
        assert len(memo) == min(i + 1, 3)
    assert memo.get(7) == 7
    memo.put(10, 10)
    assert [memo.get(k) for k in (7, 8, 9, 10)] == [7, None, 9, 10]
    with pytest.raises(ValueError):
        audio_io.Memo(0)


def _restamped_in_a_fresh_process(pickled: bytes) -> tuple[int, int, Audio]:
    """Load and check `pickled` in this (fresh) process; return the memo size after loading and after the check."""
    loaded = pickle.loads(pickled)
    loaded_only = len(audio_io.VERIFIED)
    group = loaded.groups["audio"]
    audio_io.check(group.source_stamp, group.source)
    return loaded_only, len(audio_io.VERIFIED), copy.copy(loaded)


@pytest.mark.logic
def test_source_memos_are_bounded_and_rederive_missing_entries(tmp_path: Path) -> None:
    """The per-process memo is bounded; a process holding no entry for a value's file re-stamps the file."""
    assert audio_io.VERIFIED.maxsize == audio_io.SOURCE_MEMO_ENTRIES
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = open_audio(write_wav(tmp_path / "a"))
    second = open_audio(write_wav(tmp_path / "b", frames=RATE // 2))
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        job = pool.submit(_restamped_in_a_fresh_process, pickle.dumps(first))
        loaded_only, checked, again = job.result(timeout=120)
    assert (loaded_only, checked) == (0, 1)
    assert again == first and again.identity == first.identity != second.identity
