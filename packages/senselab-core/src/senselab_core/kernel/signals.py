"""Recordings made of named channel groups, and audio."""

import math
import os
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Annotated

import numpy
import soundfile
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    PlainSerializer,
    PrivateAttr,
    ValidationInfo,
    model_validator,
)
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.hashing import content_identity, sha256_bytes

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, NsArray, as_numeric_array, require_ns
from senselab_core.kernel.time import TimeAnchor

AUDIO_GROUP = "audio"
AUDIO_UNITS = "FS"
"""Digital full scale: the units of decoded audio samples."""

_NS_PER_S = 10**9
_HEADER = "senselab_core.sound_header"

SignalArray = Annotated[numpy.ndarray, BeforeValidator(lambda v: as_numeric_array(v, 2))]
"""A read-only numeric array of shape `(channels, samples)`."""


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

    Exactly one of `data` and `source` is given. A `source` header must match the declared channel
    count and rate; samples are decoded only by `read`.
    """

    channels: tuple[ChannelInfo, ...] = Field(min_length=1)
    sampling_rate: float = Field(gt=0, allow_inf_nan=False)
    data: SignalArray | None = None
    source: Path | None = None
    _frames: int = PrivateAttr()

    @model_validator(mode="after")
    def _check_layout(self, validation: ValidationInfo) -> "RegularGroup":
        if getattr(self, "_frames", None) is not None:
            return self
        if (self.data is None) == (self.source is None):
            raise ValueError("a regular group needs exactly one of data and source")
        if self.data is not None:
            rows, self._frames = self.data.shape
        else:
            assert self.source is not None
            info = (validation.context or {}).get(_HEADER) or soundfile.info(os.fspath(self.source))
            if info.samplerate != self.sampling_rate:
                raise ValueError(f"declared rate {self.sampling_rate} Hz, file rate {info.samplerate} Hz")
            rows, self._frames = info.channels, info.frames
        _check_channels(self.channels, rows)
        return self

    @property
    def frames(self) -> int:
        """Number of samples per channel."""
        return self._frames

    def read(self, start_ns: int | None = None, end_ns: int | None = None) -> numpy.ndarray:
        """Return the `(channels, samples)` samples whose times lie in `[start_ns, end_ns)` from the group start.

        Omitted bounds mean the start and the end; a range running past the end stops there. In-memory
        data is returned as a read-only view; a file is decoded to float32 for the requested frames only.

        Raises:
            TypeError: if a bound is not an integer.
            ValueError: if `start_ns` is negative or `end_ns < start_ns`.
        """
        start = 0 if start_ns is None else require_ns(start_ns, "start_ns")
        if start < 0:
            raise ValueError(f"start_ns must be non-negative, got {start}")
        end = None if end_ns is None else require_ns(end_ns, "end_ns")
        if end is not None and end < start:
            raise ValueError(f"end_ns {end} is before start_ns {start}")
        first = min(_frame_at(start, self.sampling_rate), self._frames)
        stop = self._frames if end is None else min(_frame_at(end, self.sampling_rate), self._frames)
        if self.data is not None:
            return self.data[:, first:stop]
        return self._decode(first, stop - first)

    def _decode(self, first: int, count: int) -> numpy.ndarray:
        if count == 0:
            return numpy.zeros((len(self.channels), 0), dtype=numpy.float32)
        assert self.source is not None
        with soundfile.SoundFile(os.fspath(self.source)) as f:
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
    """Named channel groups sharing one clock, anchored by `anchor`, with content `identity`; `groups` is read-only."""

    groups: Groups
    anchor: TimeAnchor
    identity: NonEmptyStr


class Audio(Recording):
    """A recording of one regular group named `audio` whose channels are all of type `audio`, in one unit."""

    @model_validator(mode="after")
    def _check_audio(self) -> "Audio":
        if list(self.groups) != [AUDIO_GROUP] or not isinstance(self.groups[AUDIO_GROUP], RegularGroup):
            raise ValueError(f"audio has exactly one regular group named {AUDIO_GROUP!r}, got {list(self.groups)}")
        channels = self.groups[AUDIO_GROUP].channels
        if any(c.type != "audio" for c in channels):
            raise ValueError(f"audio channels must have type 'audio', got {[c.type for c in channels]}")
        if len({c.units for c in channels}) != 1:
            raise ValueError(f"audio channels must share units, got {[c.units for c in channels]}")
        return self

    @classmethod
    def _of(cls, group: RegularGroup, identity: str) -> "Audio":
        anchor = TimeAnchor(clock_id=identity, absolute_start_ns=None)
        return cls(groups={AUDIO_GROUP: group}, anchor=anchor, identity=identity)

    @staticmethod
    def _channels(count: int) -> tuple[ChannelInfo, ...]:
        return tuple(ChannelInfo(name=f"ch{i}", type="audio", units=AUDIO_UNITS) for i in range(count))

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "Audio":
        """Open a sound file lazily: only its header is read, and its identity is the root of its content identity.

        The recording's clock is named by its identity.
        """
        info = soundfile.info(os.fspath(path))
        group = RegularGroup.model_validate(
            {"channels": cls._channels(info.channels), "sampling_rate": info.samplerate, "source": Path(path)},
            context={_HEADER: info},
        )
        return cls._of(group, content_identity(path)["root"])

    @classmethod
    def from_array(cls, a: numpy.ndarray, sampling_rate: float) -> "Audio":
        """Wrap a copy of samples shaped `(samples,)` or `(channels, samples)`.

        The identity is `sha256_bytes(jcs_dumps({"npy": sha256_bytes(canonical_npy_bytes(x)),
        "sampling_rate": float(sampling_rate)}))`, where `x` is the `(channels, samples)` array. The
        recording's clock is named by its identity.

        Raises:
            ValueError: if `a` is neither 1-D nor 2-D.
        """
        x = numpy.array(a, copy=True)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"audio samples must be 1-D or (channels, samples), got shape {x.shape}")
        group = RegularGroup(channels=cls._channels(x.shape[0]), sampling_rate=sampling_rate, data=x)
        npy_hash = sha256_bytes(canonical_npy_bytes(x))
        return cls._of(group, sha256_bytes(jcs_dumps({"npy": npy_hash, "sampling_rate": group.sampling_rate})))
