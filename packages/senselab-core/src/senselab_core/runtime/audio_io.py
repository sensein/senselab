"""Sound files on disk: stamping, decoding, and the per-process memo of stamps already checked.

`open_audio` is the entry point for audio from a file, `source_group` for any regular group read from a
sound file, and `READER` the `SourceReader` that decodes their samples.
"""

import os
import threading
from collections import OrderedDict
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy
import soundfile
from senselab_worker.hashing import content_identity

from senselab_core.kernel._base import value_digest
from senselab_core.kernel.signals import (
    AUDIO_DTYPE,
    Audio,
    ChannelInfo,
    RegularGroup,
    SourceStamp,
    StaleSource,
    audio_channels,
)

SOURCE_MEMO_ENTRIES = 65_536
"""How many (path, stamp) entries the per-process stamp memo keeps; the least recently used is evicted first."""


class Memo:
    """A thread-safe mapping that keeps at most `maxsize` entries, evicting the least recently used.

    A miss is always safe: callers recompute what was evicted.
    """

    def __init__(self, maxsize: int) -> None:
        """Start empty, keeping at most `maxsize` (at least 1) entries."""
        if maxsize < 1:
            raise ValueError(f"a memo keeps at least one entry, got {maxsize}")
        self.maxsize = maxsize
        self._items: OrderedDict[Hashable, object] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> object | None:
        """Return the value held for `key`, marking it recently used, or `None`."""
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: Hashable, value: object) -> None:
        """Hold `value` for `key`, evicting the least recently used entry beyond `maxsize`."""
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry; callers recompute on their next miss."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        """Return the number of entries held."""
        with self._lock:
            return len(self._items)


VERIFIED = Memo(SOURCE_MEMO_ENTRIES)
"""The ctime, by (path, stamp digest), of each file this process stamped; a stamp with one is checked by `os.stat`."""


def _remember(path: str | os.PathLike[str], stamp: SourceStamp, ctime_ns: int) -> None:
    VERIFIED.put((os.fspath(path), stamp.canonical_digest()), ctime_ns)


@contextmanager
def _opened(path: str | os.PathLike[str]) -> Iterator[tuple[soundfile.SoundFile, SourceStamp, int]]:
    """Open `path` once and yield the sound file, the stamp of that open file (content read in full) and its ctime.

    Raises:
        StaleSource: if the file's stat fields changed while its content was read.
    """
    with open(path, "rb") as raw:
        st = os.fstat(raw.fileno())
        content = content_identity(raw)["root"]
        raw.seek(0)
        after = os.fstat(raw.fileno())
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (st.st_size, st.st_mtime_ns, st.st_ctime_ns):
            raise StaleSource(f"{path} changed while it was read")
        with soundfile.SoundFile(raw) as f:
            header = {
                "frames": f.frames,
                "channels": f.channels,
                "samplerate": f.samplerate,
                "format": f.format,
                "subtype": f.subtype,
            }
            stamp = SourceStamp(
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
                inode=st.st_ino,
                device=st.st_dev,
                frames=f.frames,
                channels=f.channels,
                sampling_rate=f.samplerate,
                header=value_digest("soundfile.info", header),
                content=content,
            )
            yield f, stamp, st.st_ctime_ns


def _require_stamp(stamp: SourceStamp, current: SourceStamp, path: object) -> None:
    if current != stamp:
        raise StaleSource(f"{path} changed since it was opened: recorded {stamp!r}, now {current!r}")


def stamp(path: str | os.PathLike[str]) -> SourceStamp:
    """Open the sound file at `path`, read every byte and its header, and return its current stamp.

    The stamp is remembered, so `check` of it at this path is an `os.stat`.

    Raises:
        StaleSource: if the file changed while it was read.
        soundfile.LibsndfileError: if the file is not a sound file.
    """
    with _opened(path) as (_, taken, ctime_ns):
        pass
    _remember(path, taken, ctime_ns)
    return taken


def check(recorded: SourceStamp, path: str | os.PathLike[str]) -> None:
    """Check that the file at `path` still has the stamp `recorded`.

    A stamp this process took from the file at `path` is checked by `os.stat` alone, its ctime included,
    so a rewrite that restores size and mtime is caught; any other stamp is checked by stamping the file
    again, which reads every byte.

    Raises:
        StaleSource: if the file differs from the stamp.
    """
    ctime_ns = VERIFIED.get((os.fspath(path), recorded.canonical_digest()))
    if ctime_ns is not None:
        st = os.stat(path)
        if (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev, st.st_ctime_ns) != (
            recorded.size,
            recorded.mtime_ns,
            recorded.inode,
            recorded.device,
            ctime_ns,
        ):
            raise StaleSource(f"{path} changed since it was opened: recorded {recorded!r}")
        return
    with _opened(path) as (_, current, ctime_ns):
        _require_stamp(recorded, current, path)
    _remember(path, recorded, ctime_ns)


class SoundFileReader:
    """The `SourceReader` over soundfile: every decode stamps the file again, content included."""

    def decode(self, source: Path, stamp: SourceStamp, first: int, count: int) -> numpy.ndarray:
        """Return `(channels, count)` float32 samples from frame `first`, after checking the file still has `stamp`.

        Raises:
            StaleSource: if the file at `source` differs from `stamp`.
        """
        with _opened(source) as (f, current, _):
            _require_stamp(stamp, current, source)
            if count == 0:
                return numpy.zeros((stamp.channels, 0), dtype=AUDIO_DTYPE)
            f.seek(first)
            samples = f.read(count, dtype="float32", always_2d=True)
        decoded: numpy.ndarray = samples.T
        return decoded


READER = SoundFileReader()
"""The process's sound-file reader; pass it to `RegularGroup.read` for a source group."""


def source_group(
    channels: tuple[ChannelInfo, ...],
    sampling_rate: float,
    path: str | os.PathLike[str],
    recorded: SourceStamp | None = None,
) -> RegularGroup:
    """Return a regular group read from the sound file at `path`, stamping it, or checking `recorded` against it.

    Raises:
        StaleSource: if `recorded` is given and the file differs from it.
        ValueError: if the file's header does not match `channels` and `sampling_rate`.
    """
    if recorded is None:
        recorded = stamp(path)
    else:
        check(recorded, path)
    return RegularGroup(channels=channels, sampling_rate=sampling_rate, source=Path(path), source_stamp=recorded)


def open_audio(path: str | os.PathLike[str]) -> Audio:
    """Open a sound file lazily: its bytes are hashed once and its header read; no sample is decoded.

    Its identity is its content identity, and the recording's clock is named by its identity. Its
    samples are read by `group.read(..., reader=READER)`.

    Raises:
        StaleSource: if the file changed while it was read.
        soundfile.LibsndfileError: if the file is not a sound file.
    """
    taken = stamp(path)
    return Audio.from_group(source_group(audio_channels(taken.channels), taken.sampling_rate, path, taken))
