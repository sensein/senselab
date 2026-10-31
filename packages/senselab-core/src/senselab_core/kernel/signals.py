"""Recordings made of named channel groups, and audio."""

import math
import os
import threading
from collections import OrderedDict
from collections.abc import Hashable, Iterator, Mapping
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self

import numpy
import soundfile
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PlainSerializer,
    StringConstraints,
    TypeAdapter,
    model_validator,
)
from senselab_worker.canonical import canonical_npy_bytes
from senselab_worker.digest import identity_digest, npy_digest
from senselab_worker.hashing import content_identity

from senselab_core.kernel._base import (
    Frozen,
    FrozenMapping,
    NonEmptyStr,
    NsArray,
    Rate,
    as_numeric_array,
    require_ns,
    value_digest,
)
from senselab_core.kernel.identity import ContentIdentity
from senselab_core.kernel.time import TimeAnchor

AUDIO_GROUP = "audio"
AUDIO_UNITS = "FS"
"""Digital full scale: the units of decoded audio samples."""

AUDIO_DTYPE = numpy.dtype(numpy.float32)
"""The dtype audio samples are held and read in: float32 full scale."""

SIGNAL_DTYPES: Mapping[str, float | None] = FrozenMapping(
    {"int16": 2.0**15, "int32": 2.0**31, "float32": None, "float64": None}
)
"""The sample dtypes `Audio.from_array` accepts, by name, with the full scale an integer dtype is divided by.

Integer PCM maps to float32 full scale as the decoder maps it (`x / 2**(bits - 1)`); floats are cast.
"""

_NS_PER_S = 10**9
SourceStampKey = str
"""A source stamp's `canonical_digest`, the key of the per-process stamp and root memos."""

SOURCE_MEMO_ENTRIES = 65_536
"""How many (path, stamp) entries each per-process source memo keeps; the least recently used is evicted first."""


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

    def __len__(self) -> int:
        """Return the number of entries held."""
        with self._lock:
            return len(self._items)


_VERIFIED = Memo(SOURCE_MEMO_ENTRIES)
"""(path, stamp) pairs this process stamped from the file and kept; a copy with one is checked by `os.stat` alone."""


def _remember(path: str | os.PathLike[str], stamp: "SourceStamp") -> None:
    _VERIFIED.put((os.fspath(path), stamp.canonical_digest()), True)


def _finite_signal(v: object) -> numpy.ndarray:
    a = as_numeric_array(v, 2)
    if a.dtype.kind == "f" and not numpy.isfinite(a).all():
        raise ValueError("signal samples must be finite; NaN and infinite samples are refused")
    return a


SignalArray = Annotated[numpy.ndarray, BeforeValidator(_finite_signal)]
"""A frozen integer or float array of shape `(channels, samples)` whose samples are all finite."""


class StaleSource(RuntimeError):
    """A source file no longer matches the stamp recorded when it was opened."""


class SourceStamp(Frozen):
    """What a source file was when it was opened: its stat fields, a digest of its sound header, and its content.

    `header` is `value_digest("soundfile.info", {frames, channels, samplerate, format, subtype})`;
    `frames`, `channels` and `sampling_rate` repeat the header fields a group's layout is checked against.
    `content` is the root of the file's content identity (`content_identity`), taken over every byte
    when the stamp is taken; it is the identity root of a recording of that file.
    """

    size: int = Field(ge=0)
    mtime_ns: int
    inode: int = Field(ge=0)
    device: int = Field(ge=0)
    frames: int = Field(ge=0)
    channels: int = Field(ge=1)
    sampling_rate: Rate
    header: NonEmptyStr
    content: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @classmethod
    def of(cls, path: str | os.PathLike[str]) -> "SourceStamp":
        """Open the file at `path`, read every byte, and return its current stamp.

        Raises:
            StaleSource: if the file changed while it was read.
        """
        with _opened(path) as (_, stamp):
            return stamp

    def checked(self, path: str | os.PathLike[str]) -> None:
        """Check that the file at `path` still has this stamp.

        A stamp this process took from the file at `path` is checked by `os.stat` alone; any other
        stamp is checked by stamping the file again, which reads every byte. Samples are never
        decoded from the file without checking its content (`RegularGroup.read`).

        Raises:
            StaleSource: if the file differs from the stamp.
        """
        if _VERIFIED.get((os.fspath(path), self.canonical_digest())) is not None:
            if not self.same_file(path):
                raise StaleSource(f"{path} changed since it was opened: recorded {self!r}")
            return
        _require_stamp(self, SourceStamp.of(path), path)
        _remember(path, self)

    def same_file(self, path: str | os.PathLike[str]) -> bool:
        """Return whether the stat fields of the file at `path` still match, without reading its header."""
        st = os.stat(path)
        return (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev) == (self.size, self.mtime_ns, self.inode, self.device)


@contextmanager
def _opened(path: str | os.PathLike[str]) -> Iterator[tuple[soundfile.SoundFile, SourceStamp]]:
    """Open `path` once and yield the sound file with the stamp of that open file, its content read in full.

    Raises:
        StaleSource: if the file's stat fields changed while its content was read.
    """
    with open(path, "rb") as raw:
        st = os.fstat(raw.fileno())
        content = content_identity(path)["root"]
        after = os.fstat(raw.fileno())
        if (after.st_size, after.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
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
            yield f, stamp


def _require_stamp(stamp: SourceStamp, current: SourceStamp, path: object) -> None:
    if current != stamp:
        raise StaleSource(f"{path} changed since it was opened: recorded {stamp!r}, now {current!r}")


class ChannelInfo(Frozen):
    """One channel: its name, signal type (`audio`, `ecg`, ...), units, and optional placement and reference."""

    name: NonEmptyStr
    type: NonEmptyStr
    units: NonEmptyStr
    placement: str | None = None
    reference: str | None = None


def _check_channels(channels: tuple[ChannelInfo, ...], rows: int) -> None:
    names = [c.name for c in channels]
    if len(set(names)) != len(names):
        raise ValueError(f"channel names must be unique, got {names}")
    if rows != len(channels):
        raise ValueError(f"{len(channels)} channels declared, {rows} channels in the data")


def _frame_at(t_ns: int, sampling_rate: float) -> int:
    """Index of the first sample at or after `t_ns`, sample k lying at k / sampling_rate seconds."""
    return math.ceil(Fraction(t_ns) * Fraction(sampling_rate) / _NS_PER_S)


class RegularGroup(Frozen):
    """Channels sampled together at one rate, held in `data` `(channels, samples)` or read from the sound file `source`.

    Exactly one of `data` and `source` is given. A source group carries `source_stamp`, the stamp of
    the file when it was opened: recorded at validation when not supplied, and checked against the
    file when supplied (`SourceStamp.checked`, raising `StaleSource` if it differs), so a copy made
    in this process does not read the file. The header must match the declared channel count and
    rate; samples are decoded only by `read`, which stamps the file again, content included, and
    refuses a file that no longer matches.
    """

    channels: tuple[ChannelInfo, ...] = Field(min_length=1)
    sampling_rate: Rate
    data: SignalArray | None = None
    source: Path | None = None
    source_stamp: SourceStamp | None = None

    @model_validator(mode="before")
    @classmethod
    def _stamp_source(cls, data: object) -> object:
        if not isinstance(data, Mapping) or data.get("source") is None:
            return data
        supplied = data.get("source_stamp")
        if supplied is None:
            taken = SourceStamp.of(data["source"])
            _remember(data["source"], taken)
            return {**data, "source_stamp": taken}
        stamp = supplied if isinstance(supplied, SourceStamp) else SourceStamp.model_validate(supplied)
        stamp.checked(data["source"])
        return {**data, "source_stamp": stamp}

    @model_validator(mode="after")
    def _check_layout(self) -> "RegularGroup":
        if (self.data is None) == (self.source is None):
            raise ValueError("a regular group needs exactly one of data and source")
        if self.data is not None:
            if self.source_stamp is not None:
                raise ValueError("a source stamp belongs to a source group, not to held data")
            rows = self.data.shape[0]
        else:
            assert self.source_stamp is not None
            if self.source_stamp.sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"declared rate {self.sampling_rate} Hz, file rate {self.source_stamp.sampling_rate} Hz"
                )
            rows = self.source_stamp.channels
        _check_channels(self.channels, rows)
        return self

    @property
    def frames(self) -> int:
        """Number of samples per channel, from the held data or the source stamp."""
        if self.data is not None:
            return int(self.data.shape[1])
        assert self.source_stamp is not None
        return self.source_stamp.frames

    def read(self, start_ns: int | None = None, end_ns: int | None = None) -> numpy.ndarray:
        """Return the `(channels, samples)` samples whose times lie in `[start_ns, end_ns)` from the group start.

        Omitted bounds mean the start and the end; a range running past the end stops there. Held
        data is returned as a read-only view; a file is decoded to float32 for the requested frames
        only, after checking it still matches `source_stamp`.

        Raises:
            TypeError: if a bound is not an integer.
            ValueError: if `start_ns` is negative or `end_ns < start_ns`.
            StaleSource: if the source file changed since it was opened.
        """
        start = 0 if start_ns is None else require_ns(start_ns, "start_ns")
        if start < 0:
            raise ValueError(f"start_ns must be non-negative, got {start}")
        end = None if end_ns is None else require_ns(end_ns, "end_ns")
        if end is not None and end < start:
            raise ValueError(f"end_ns {end} is before start_ns {start}")
        frames = self.frames
        first = min(_frame_at(start, self.sampling_rate), frames)
        stop = frames if end is None else min(_frame_at(end, self.sampling_rate), frames)
        if self.data is not None:
            return self.data[:, first:stop]
        return self._decode(first, stop - first)

    def _decode(self, first: int, count: int) -> numpy.ndarray:
        assert self.source is not None and self.source_stamp is not None
        with _opened(self.source) as (f, current):
            _require_stamp(self.source_stamp, current, self.source)
            if count == 0:
                return numpy.zeros((len(self.channels), 0), dtype=AUDIO_DTYPE)
            f.seek(first)
            samples = f.read(count, dtype="float32", always_2d=True)
        return numpy.ascontiguousarray(samples.T)


class IrregularGroup(Frozen):
    """Channels sampled together at the strictly increasing times `timestamps_ns`, one per `data` column."""

    channels: tuple[ChannelInfo, ...] = Field(min_length=1)
    timestamps_ns: NsArray
    data: SignalArray

    @model_validator(mode="after")
    def _check_layout(self) -> "IrregularGroup":
        _check_channels(self.channels, self.data.shape[0])
        if self.data.shape[1] != self.timestamps_ns.shape[0]:
            raise ValueError(f"{self.timestamps_ns.shape[0]} timestamps for {self.data.shape[1]} samples")
        if numpy.any(numpy.diff(self.timestamps_ns) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        return self


Groups = Annotated[
    Mapping[NonEmptyStr, RegularGroup | IrregularGroup],
    Field(min_length=1),
    AfterValidator(FrozenMapping),
    PlainSerializer(dict),
]
"""A non-empty read-only mapping of group name to group; serialised as a dict."""


class Recording(Frozen):
    """Named channel groups sharing one clock, anchored by `anchor`; `groups` is read-only.

    `identity` is derived from the content by validation: omitted, it is filled in; supplied, it must
    equal the derived one. A recording of one source group is identified by `content:` and the
    content identity root of the file. Otherwise it is `content:` and `value_digest(IDENTITY_TYPE,
    {"groups": ...})`, each source group standing as its channels, rate and content identity root.
    `IDENTITY_TYPE` is a declared token, so moving, renaming or subclassing the class keeps identities.
    """

    IDENTITY_TYPE: ClassVar[str] = "recording"

    groups: Groups
    anchor: TimeAnchor
    identity: ContentIdentity

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            groups: Mapping[str, "RegularGroup | IrregularGroup"],
            anchor: TimeAnchor,
            identity: str | None = None,
        ) -> None:
            """Validate `groups` and `anchor`, deriving `identity` or checking the one supplied."""

    @model_validator(mode="wrap")
    @classmethod
    def _derive_identity(cls, data: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        if isinstance(data, Mapping) and "groups" in data:
            groups = _GROUPS.validate_python(data["groups"])
            derived = cls.identity_of(groups)
            supplied = data.get("identity")
            if supplied is not None and supplied != derived:
                raise ValueError(f"identity {supplied!r} is not the content identity {derived!r}")
            data = {**data, "groups": groups, "identity": derived}
        return handler(data)

    @classmethod
    def identity_of(cls, groups: Mapping[str, "RegularGroup | IrregularGroup"]) -> str:
        """Return the identity a recording of this type over `groups` has.

        Raises:
            ValueError: if `groups` is not a layout this type holds.
            StaleSource: if a source file changed since it was opened.
        """
        values = list(groups.values())
        if len(values) == 1 and isinstance(values[0], RegularGroup) and values[0].source is not None:
            return "content:" + _source_root(values[0])
        fields = {name: _group_content(g) for name, g in groups.items()}
        return "content:" + value_digest(cls.IDENTITY_TYPE, {"groups": fields})


def _source_root(group: RegularGroup) -> str:
    """Return the content identity root of a source group's file, as its checked stamp holds it."""
    assert group.source is not None and group.source_stamp is not None
    return group.source_stamp.content


def _group_content(group: RegularGroup | IrregularGroup) -> object:
    if isinstance(group, RegularGroup) and group.source is not None:
        return {**group.model_dump(exclude={"source", "source_stamp"}), "content": _source_root(group)}
    return group.model_dump()


def _audio_group(groups: Mapping[str, RegularGroup | IrregularGroup]) -> RegularGroup:
    group = groups.get(AUDIO_GROUP)
    if list(groups) != [AUDIO_GROUP] or not isinstance(group, RegularGroup):
        raise ValueError(f"audio has exactly one regular group named {AUDIO_GROUP!r}, got {list(groups)}")
    channels = group.channels
    if any(c.type != "audio" for c in channels):
        raise ValueError(f"audio channels must have type 'audio', got {[c.type for c in channels]}")
    if len({c.units for c in channels}) != 1:
        raise ValueError(f"audio channels must share units, got {[c.units for c in channels]}")
    if group.data is not None and group.data.dtype != AUDIO_DTYPE:
        raise ValueError(f"audio samples are held as float32 full scale, got {group.data.dtype}")
    return group


_GROUPS: TypeAdapter[Mapping[str, RegularGroup | IrregularGroup]] = TypeAdapter(
    Groups, config=ConfigDict(arbitrary_types_allowed=True)
)


class Audio(Recording):
    """A recording of one regular group named `audio` whose channels are all of type `audio`, in one unit.

    Held samples are float32 (`AUDIO_DTYPE`). An in-memory audio's identity is `"content:" +
    identity_digest("audio@3", {"npy": npy_digest("npy@1", <the (channels, samples) data>),
    "sampling_rate": rate})`; a file-backed one's is `content:` and the root of the file's content identity.
    """

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            groups: Mapping[str, "RegularGroup | IrregularGroup"],
            anchor: TimeAnchor,
            identity: str | None = None,
        ) -> None:
            """Validate `groups` and `anchor`, deriving `identity` or checking the one supplied."""

    @classmethod
    def identity_of(cls, groups: Mapping[str, "RegularGroup | IrregularGroup"]) -> str:
        """Return the identity of audio over `groups`, after checking they are an audio layout.

        Raises:
            ValueError: if `groups` is not one regular `audio` group of float32 audio channels in one unit.
            StaleSource: if the source file changed since it was opened.
        """
        group = _audio_group(groups)
        if group.data is None:
            return "content:" + _source_root(group)
        npy = npy_digest("npy@1", group.data)
        return "content:" + identity_digest("audio@3", {"npy": npy, "sampling_rate": group.sampling_rate})

    @classmethod
    def _of(cls, group: RegularGroup) -> "Audio":
        identity = cls.identity_of({AUDIO_GROUP: group})
        anchor = TimeAnchor(clock_id=identity, absolute_start_ns=None)
        return cls(groups={AUDIO_GROUP: group}, anchor=anchor, identity=identity)

    @staticmethod
    def _channels(count: int) -> tuple[ChannelInfo, ...]:
        return tuple(ChannelInfo(name=f"ch{i}", type="audio", units=AUDIO_UNITS) for i in range(count))

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "Audio":
        """Open a sound file lazily: its bytes are hashed once and its header read; no sample is decoded.

        Its identity is its content identity, and the recording's clock is named by its identity.
        """
        stamp = SourceStamp.of(path)
        _remember(path, stamp)
        group = RegularGroup.model_validate(
            {
                "channels": cls._channels(stamp.channels),
                "sampling_rate": stamp.sampling_rate,
                "source": Path(path),
                "source_stamp": stamp,
            }
        )
        return cls._of(group)

    @classmethod
    def from_array(cls, a: numpy.ndarray, sampling_rate: float) -> "Audio":
        """Hold a float32 full-scale copy of samples shaped `(samples,)` or `(channels, samples)`.

        Integer PCM is divided by its full scale (`SIGNAL_DTYPES`) as the decoder divides it; floats
        are cast. The integer conversion is done in float32 (round to float32, then scale by a power of
        two) and is bit-identical to dividing in float64 and casting to float32. The recording's clock
        is named by its identity.

        Raises:
            ValueError: if `a` is neither 1-D nor 2-D, is empty, holds a NaN or infinite sample or one
                that overflows float32, or its dtype is not in `SIGNAL_DTYPES` (complex, boolean, object,
                datetime and other integer widths are refused).
            pydantic.ValidationError: if `sampling_rate` is not a `Rate`.
        """
        if isinstance(a, numpy.ndarray) and type(a) is not numpy.ndarray:
            raise ValueError(f"not a plain numpy.ndarray: {type(a).__qualname__}")
        x = numpy.asarray(a)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"audio samples must be 1-D or (channels, samples), got shape {x.shape}")
        if x.size == 0:
            raise ValueError(f"audio samples must not be empty, got shape {x.shape}")
        if x.dtype.name not in SIGNAL_DTYPES:
            raise ValueError(f"audio samples must be one of {sorted(SIGNAL_DTYPES)}, got {x.dtype}")
        scale = SIGNAL_DTYPES[x.dtype.name]
        if scale is None and not numpy.isfinite(x).all():
            raise ValueError("audio samples must be finite; NaN and infinite samples are refused")
        with numpy.errstate(over="ignore"):
            held = x.astype(AUDIO_DTYPE) if scale is None else x.astype(AUDIO_DTYPE) / AUDIO_DTYPE.type(scale)
        if not numpy.isfinite(held).all():
            raise ValueError("audio samples must lie within float32 full scale; a sample overflows float32")
        group = RegularGroup(channels=cls._channels(x.shape[0]), sampling_rate=sampling_rate, data=held)
        return cls._of(group)
