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
from senselab_worker.hashing import DEFAULT_CHUNK_SIZE, chunk_digest, content_identity, content_root

from senselab_core.kernel._base import value_digest
from senselab_core.kernel.signals import (
    Audio,
    ChannelInfo,
    RegularGroup,
    SourceStamp,
    StaleSource,
    audio_channels,
)

SOURCE_MEMO_ENTRIES = 65_536
"""How many (path, stamp) entries the per-process stamp memo keeps; the least recently used is evicted first."""

DECODER = f"soundfile {soundfile.__version__}, libsndfile {soundfile.__libsndfile_version__}"
"""The decoder `open_audio` reads with, as an execution context records it."""


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


def _stat_fields(st: os.stat_result) -> tuple[int, int, int, int]:
    return st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev


def _stamped_fields(stamp: SourceStamp) -> tuple[int, int, int, int]:
    return stamp.size, stamp.mtime_ns, stamp.inode, stamp.device


def _header(f: soundfile.SoundFile) -> str:
    fields = {"frames": f.frames, "channels": f.channels, "samplerate": f.samplerate, "format": f.format}
    return value_digest("soundfile.info", {**fields, "subtype": f.subtype})


@contextmanager
def _opened(
    path: str | os.PathLike[str], chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Iterator[tuple[soundfile.SoundFile, SourceStamp, int]]:
    """Open `path` once and yield the sound file, the stamp of that open file (content read in full) and its ctime.

    Raises:
        StaleSource: if the file's stat fields changed while its content was read.
    """
    with open(path, "rb") as raw:
        st = os.fstat(raw.fileno())
        content = content_identity(raw, chunk_size=chunk_size)
        raw.seek(0)
        after = os.fstat(raw.fileno())
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (st.st_size, st.st_mtime_ns, st.st_ctime_ns):
            raise StaleSource(f"{path} changed while it was read")
        with soundfile.SoundFile(raw) as f:
            stamp = SourceStamp(
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
                inode=st.st_ino,
                device=st.st_dev,
                frames=f.frames,
                channels=f.channels,
                sampling_rate=f.samplerate,
                header=_header(f),
                chunk_size=chunk_size,
                chunks=tuple(content["chunks"]),
                content=content["root"],
            )
            yield f, stamp, st.st_ctime_ns


def _require_stamp(stamp: SourceStamp, current: SourceStamp, path: object) -> None:
    if current != stamp:
        raise StaleSource(f"{path} changed since it was opened: recorded {stamp!r}, now {current!r}")


def stamp(path: str | os.PathLike[str], *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> SourceStamp:
    """Open the sound file at `path`, read every byte and its header, and return its current stamp.

    `chunk_size` is the grid its content is hashed on. The stamp is remembered, so `check` of it at this
    path is an `os.stat`.

    Raises:
        StaleSource: if the file changed while it was read.
        soundfile.LibsndfileError: if the file is not a sound file.
    """
    with _opened(path, chunk_size) as (_, taken, ctime_ns):
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
        if (*_stat_fields(st), st.st_ctime_ns) != (*_stamped_fields(recorded), ctime_ns):
            raise StaleSource(f"{path} changed since it was opened: recorded {recorded!r}")
        return
    with _opened(path, recorded.chunk_size) as (_, current, ctime_ns):
        _require_stamp(recorded, current, path)
    _remember(path, recorded, ctime_ns)


class _ChunkChecked:
    """A read-only binary file over `fd` that serves bytes only from chunks whose digest the stamp records.

    Each chunk a read touches is read whole by `os.pread`, hashed and compared before any of its bytes is
    served; the last chunk read is kept for the reads that follow. A chunk that differs ends the read short and is
    held in `stale`, since the decoder's callback cannot raise through it.
    """

    def __init__(self, fd: int, stamp: SourceStamp, path: object) -> None:
        self._fd, self._stamp, self._path = fd, stamp, path
        self._pos = 0
        self._index = -1
        self._chunk = b""
        self.stale: StaleSource | None = None

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._pos, os.SEEK_END: self._stamp.size}[whence]
        self._pos = max(base + offset, 0)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buffer: memoryview) -> int:
        view = memoryview(buffer).cast("B")
        done = 0
        while done < len(view) and self._pos < self._stamp.size and self.stale is None:
            index, offset = divmod(self._pos, self._stamp.chunk_size)
            try:
                chunk = self._verified(index)
            except StaleSource as exc:
                self.stale = exc
                break
            take = min(len(view) - done, len(chunk) - offset)
            view[done : done + take] = chunk[offset : offset + take]
            done += take
            self._pos += take
        return done

    def _verified(self, index: int) -> bytes:
        if index != self._index:
            size = self._stamp.chunk_size
            parts: list[bytes] = []
            while (got := sum(map(len, parts))) < size and (part := os.pread(self._fd, size - got, index * size + got)):
                parts.append(part)
            chunk = b"".join(parts)
            if chunk_digest(chunk) != self._stamp.chunks[index]:
                raise StaleSource(f"{self._path} changed since it was opened: chunk {index} differs")
            self._index, self._chunk = index, chunk
        return self._chunk


class SoundFileReader:
    """The `SourceReader` over soundfile: a decode verifies only the content chunks it reads."""

    def decode(self, source: Path, stamp: SourceStamp, first: int, count: int) -> numpy.ndarray:
        """Return `(channels, count)` float32 samples from frame `first`, after checking the file still has `stamp`.

        The file's stat fields and sound header are checked against `stamp`, the stamp's content against the root of
        its chunk digests, and every chunk the decoder reads is hashed and compared with its digest before it is
        decoded.

        Raises:
            StaleSource: if the file at `source` differs from `stamp` in its stat fields, its header or a chunk read,
                or the stamp's content is not the root of its chunks.
        """
        fd = os.open(source, os.O_RDONLY)
        try:
            rooted = content_root(stamp.chunk_size, stamp.chunks) == stamp.content
            if _stat_fields(os.fstat(fd)) != _stamped_fields(stamp) or not rooted:
                raise StaleSource(f"{source} changed since it was opened: recorded {stamp!r}")
            checked = _ChunkChecked(fd, stamp, source)
            try:
                with soundfile.SoundFile(checked) as f:
                    if _header(f) != stamp.header:
                        raise StaleSource(f"{source} changed since it was opened: its sound header differs")
                    f.seek(first)
                    samples = f.read(count, dtype="float32", always_2d=True)
            except soundfile.LibsndfileError:
                if checked.stale is None:
                    raise
            if checked.stale is not None:
                raise checked.stale
        finally:
            os.close(fd)
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
