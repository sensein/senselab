"""Recordings made of named channel groups, and audio."""

import math
import re
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Protocol, Self, get_args, runtime_checkable

import numpy
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
from senselab_worker.canonical import canonical_text
from senselab_worker.digest import identity_digest

from senselab_core.kernel._base import (
    ArrayForm,
    Frozen,
    FrozenMapping,
    NonEmptyStr,
    NsArray,
    Rate,
    as_numeric_array,
    held_npy_digest,
    require_finite,
    require_ns,
    value_digest,
)
from senselab_core.kernel.formats import FormatTag, Formatted
from senselab_core.kernel.identity import TEXT_IDENTITY_FORMS, ContentIdentity
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
SignalArray = Annotated[
    numpy.ndarray,
    BeforeValidator(lambda v: require_finite(as_numeric_array(v, 2), "signal samples", nan_is_missing=True)),
    ArrayForm(("number", "null"), 2, None, "Shape (channels, samples); a null marks a missing sample."),
]
"""A frozen integer or float array of shape `(channels, samples)`: no infinite sample; a NaN marks a missing one.

A NaN is held as the one quiet NaN without payload. A regular group refuses NaN; an irregular group holds it.
"""

SourcePath = Annotated[NonEmptyStr, BeforeValidator(lambda v: v.as_posix() if isinstance(v, PurePath) else v)]
"""A sound file's path as posix text; a `PurePath` is accepted and held as its `as_posix()`."""


class StaleSource(RuntimeError):
    """A source file no longer matches the stamp recorded when it was opened."""


Hex64 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ChannelInfo(Frozen):
    """One channel: its name, signal type (`audio`, `ecg`, ...), units, and optional placement and reference."""

    name: NonEmptyStr
    type: NonEmptyStr
    units: NonEmptyStr
    placement: str | None = None
    reference: str | None = None


class SourceStamp(Frozen):
    """What a source file was when a reader opened it: its stat fields, a digest of its sound header, and its content.

    `header` is `value_digest("soundfile.info", {frames, channels, samplerate, format, subtype})`;
    `frames` and `sampling_rate` repeat header fields a group's layout is checked against; `channels` is the channel
    layout the stamp was taken under, so a stamp of the same file under another layout is another stamp.
    `chunks` are the sha256 of each consecutive `chunk_size`-byte chunk of the file and `content` their root, as
    `content_identity` gives them; `content` is the identity root of a recording of that file. A windowed read
    verifies only the chunks it touches. Stamps are taken and checked by a `SourceReader`
    (`senselab_core.runtime.audio_io`).
    """

    size: int = Field(ge=0)
    mtime_ns: int
    inode: int = Field(ge=0)
    device: int = Field(ge=0)
    frames: int = Field(ge=0)
    channels: tuple[ChannelInfo, ...] = Field(min_length=1)
    sampling_rate: Rate
    header: NonEmptyStr
    chunk_size: int = Field(ge=1)
    chunks: tuple[Hex64, ...]
    content: Hex64

    @model_validator(mode="after")
    def _one_digest_per_chunk(self) -> "SourceStamp":
        if len(self.chunks) != -(-self.size // self.chunk_size):
            raise ValueError(f"{len(self.chunks)} chunk digests for {self.size} bytes in {self.chunk_size}-byte chunks")
        return self


class SourceReader(Protocol):
    """Decodes a source group's samples from its file; `senselab_core.runtime.audio_io.READER` implements it."""

    def decode(self, source: Path, stamp: SourceStamp, first: int, count: int) -> numpy.ndarray:
        """Return `(channels, count)` float32 samples from frame `first`, after checking the file still has `stamp`.

        Raises:
            StaleSource: if the file at `source` differs from `stamp`.
        """
        ...


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

    Exactly one of `data` and `source` is given. A source group carries `source_stamp`, the stamp a
    `SourceReader` took of the file; validation reads no file. The stamp's layout and rate must match the
    declared channel count and rate. Samples are decoded only by `read` through a reader, which
    checks the file still has the stamp, content included. A held group carries a file's stamp only as
    `decoded_from` builds it: with `decoded`, the `npy@1` digest of the samples the reader decoded, which
    the held samples must still have; it is then identified as that file is.
    """

    channels: tuple[ChannelInfo, ...] = Field(min_length=1)
    sampling_rate: Rate
    data: SignalArray | None = None
    source: SourcePath | None = None
    source_stamp: SourceStamp | None = None
    decoded: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None

    @model_validator(mode="after")
    def _check_layout(self) -> "RegularGroup":
        if (self.data is None) == (self.source is None):
            raise ValueError("a regular group needs exactly one of data and source")
        stamp = self.source_stamp
        if self.source is not None and stamp is None:
            raise ValueError("a source group carries the stamp a reader took of its file")
        if self.decoded is not None and (self.data is None or stamp is None):
            raise ValueError("`decoded` belongs only to held samples carrying their file's stamp")
        if stamp is not None and stamp.channels != self.channels:
            raise ValueError("declared channels differ from the channel layout the stamp was taken under")
        if stamp is not None and stamp.sampling_rate != self.sampling_rate:
            raise ValueError(f"declared rate {self.sampling_rate} Hz, file rate {stamp.sampling_rate} Hz")
        if self.data is not None:
            require_finite(self.data, "a regular group's samples")
            if stamp is not None and self.data.shape != (len(stamp.channels), stamp.frames):
                raise ValueError(f"held samples {self.data.shape} are not the stamped file's channels and frames")
            if stamp is not None and self.decoded != held_npy_digest(self.data):
                raise ValueError("held samples carry a file's stamp only as decoded from it (`decoded_from`)")
            rows = self.data.shape[0]
        else:
            assert stamp is not None
            rows = len(stamp.channels)
        _check_channels(self.channels, rows)
        return self

    @classmethod
    def decoded_from(cls, group: "RegularGroup", reader: SourceReader) -> "RegularGroup":
        """Return a source group's samples decoded by `reader`, held with the file's stamp and their digest.

        Raises:
            ValueError: if `group` holds its samples, or as `read`.
            StaleSource: if the file changed since it was stamped.
        """
        if group.source is None:
            raise ValueError("only a source group is decoded; a held group is its own samples")
        data = group.read(reader=reader)
        fields = {"channels": group.channels, "sampling_rate": group.sampling_rate, "data": data}
        return cls(**fields, source_stamp=group.source_stamp, decoded=held_npy_digest(data))

    @property
    def frames(self) -> int:
        """Number of samples per channel, from the held data or the source stamp."""
        if self.data is not None:
            return int(self.data.shape[1])
        assert self.source_stamp is not None
        return self.source_stamp.frames

    def read(
        self, start_ns: int | None = None, end_ns: int | None = None, reader: SourceReader | None = None
    ) -> numpy.ndarray:
        """Return the `(channels, samples)` samples whose times lie in `[start_ns, end_ns)` from the group start.

        Omitted bounds mean the start and the end; a range running past the end stops there. Held
        data is returned as a read-only view; a source group is decoded by `reader` to float32 for
        the requested frames only, after the reader checks the file still matches `source_stamp`.

        Raises:
            TypeError: if a bound is not an integer.
            ValueError: if `start_ns` is negative or `end_ns < start_ns`, a source group is read
                without a reader, or a decoded sample is not finite or not of the group's layout.
            StaleSource: if the source file changed since it was stamped.
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
        assert self.source is not None and self.source_stamp is not None
        if reader is None:
            raise ValueError(f"{self.source} is a source group; read it through a SourceReader")
        samples = reader.decode(Path(self.source), self.source_stamp, first, stop - first)
        expected = (len(self.channels), stop - first)
        if samples.shape != expected or samples.dtype != AUDIO_DTYPE:
            raise ValueError(f"decoded {samples.dtype} {samples.shape} from {self.source}, expected float32 {expected}")
        return require_finite(samples, f"samples decoded from {self.source}")


@runtime_checkable
class RegularSeries(Protocol):
    """Channels sampled together at one rate: what a task over regularly sampled channels reads.

    `RegularGroup` and `Audio` implement it.
    """

    @property
    def channels(self) -> tuple[ChannelInfo, ...]:
        """The channels, one per row of what `read` returns."""
        ...

    @property
    def sampling_rate(self) -> float:
        """The sampling rate in Hz."""
        ...

    @property
    def frames(self) -> int:
        """The number of samples per channel."""
        ...

    def read(
        self, start_ns: int | None = None, end_ns: int | None = None, reader: SourceReader | None = None
    ) -> numpy.ndarray:
        """Return the `(channels, samples)` samples in `[start_ns, end_ns)`, as `RegularGroup.read`."""
        ...


class IrregularGroup(Frozen):
    """Channels sampled together at the strictly increasing times `timestamps_ns`, one per `data` column.

    A NaN in `data` marks a sample missing from its channel at that time.
    """

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
    equal the derived one. It is `content:` and `value_digest(IDENTITY_TYPE, {"groups": ...})`, each stamped
    group standing as its channels, rate and content identity root.
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
        """
        fields = {name: _group_content(g) for name, g in groups.items()}
        return "content:" + value_digest(cls.IDENTITY_TYPE, {"groups": fields})


def _source_root(group: RegularGroup) -> str:
    """Return the content identity root of a stamped group's file, as its stamp holds it."""
    assert group.source_stamp is not None
    return group.source_stamp.content


def _group_content(group: RegularGroup | IrregularGroup) -> object:
    if isinstance(group, RegularGroup) and group.source_stamp is not None:
        held = group.model_dump(exclude={"source", "source_stamp", "decoded"})
        return {**held, "data": None, "content": _source_root(group)}
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


def audio_channels(count: int) -> tuple[ChannelInfo, ...]:
    """Return `count` audio channels in full-scale units, named `ch0`, `ch1`, ..."""
    return tuple(ChannelInfo(name=f"ch{i}", type="audio", units=AUDIO_UNITS) for i in range(count))


_GROUPS: TypeAdapter[Mapping[str, RegularGroup | IrregularGroup]] = TypeAdapter(
    Groups, config=ConfigDict(arbitrary_types_allowed=True)
)


class Audio(Recording):
    """A recording of one regular group named `audio` whose channels are all of type `audio`, in one unit.

    Held samples are float32 (`AUDIO_DTYPE`). An in-memory audio's identity is `"content:" +
    identity_digest("audio@4", {"npy": npy_digest("npy@1", <the (channels, samples) data>),
    "sampling_rate": rate, "channels": <each ChannelInfo's model_dump()>})`; a stamped one's (file-backed, or held
    with its file's stamp) is `"content:" + identity_digest("audio-file@1", {"content": <the root of the file's content
    identity>, "channels": <each ChannelInfo's model_dump()>})`, so relabelled channels are another identity.
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
        """
        group = _audio_group(groups)
        if group.source_stamp is not None or group.data is None:
            file_fields = {"content": _source_root(group), "channels": [c.model_dump() for c in group.channels]}
            return "content:" + identity_digest("audio-file@1", file_fields)
        npy = held_npy_digest(group.data)
        channels = [c.model_dump() for c in group.channels]
        fields = {"npy": npy, "sampling_rate": group.sampling_rate, "channels": channels}
        return "content:" + identity_digest("audio@4", fields)

    @property
    def group(self) -> RegularGroup:
        """The one `audio` group."""
        group = self.groups[AUDIO_GROUP]
        assert isinstance(group, RegularGroup)
        return group

    @property
    def channels(self) -> tuple[ChannelInfo, ...]:
        """The audio channels."""
        return self.group.channels

    @property
    def sampling_rate(self) -> float:
        """The sampling rate in Hz."""
        return self.group.sampling_rate

    @property
    def frames(self) -> int:
        """The number of samples per channel."""
        return self.group.frames

    def read(
        self, start_ns: int | None = None, end_ns: int | None = None, reader: SourceReader | None = None
    ) -> numpy.ndarray:
        """Return the audio group's samples in `[start_ns, end_ns)`, as `RegularGroup.read`."""
        return self.group.read(start_ns, end_ns, reader)

    @classmethod
    def from_group(cls, group: RegularGroup) -> "Audio":
        """Return the audio of one regular group of audio channels, its clock named by its identity.

        Raises:
            ValueError: if `group` is not float32 audio channels in one unit.
        """
        identity = cls.identity_of({AUDIO_GROUP: group})
        anchor = TimeAnchor(clock_id=identity, absolute_start_ns=None)
        return cls(groups={AUDIO_GROUP: group}, anchor=anchor, identity=identity)

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
        if x.ndim != 2:  # noqa: PLR2004 - the (channels, samples) layout has two axes
            raise ValueError(f"audio samples must be 1-D or (channels, samples), got shape {x.shape}")
        if x.size == 0:
            raise ValueError(f"audio samples must not be empty, got shape {x.shape}")
        if x.dtype.name not in SIGNAL_DTYPES:
            raise ValueError(f"audio samples must be one of {sorted(SIGNAL_DTYPES)}, got {x.dtype}")
        scale = SIGNAL_DTYPES[x.dtype.name]
        with numpy.errstate(over="ignore", invalid="ignore"):
            held = x.astype(AUDIO_DTYPE) if scale is None else x.astype(AUDIO_DTYPE) / AUDIO_DTYPE.type(scale)
        held = require_finite(held, "audio samples, as float32")
        group = RegularGroup(channels=audio_channels(x.shape[0]), sampling_rate=sampling_rate, data=held)
        return cls.from_group(group)


_TEXT_IDENTITY = re.compile(get_args(ContentIdentity)[1].pattern)


@runtime_checkable
class Text(Protocol):
    """Text a task reads: its content in canonical form (`canonical_text`) and that content's identity.

    `PlainText` implements it.
    """

    @property
    def content(self) -> str:
        """The text, in canonical form; character offsets into it count code points."""
        ...

    @property
    def identity(self) -> str:
        """The content identity of `content`."""
        ...


class PlainText(Formatted):
    """Text held in canonical form (`canonical_text`), identified by its content.

    `identity` is `"content:" + value_digest(IDENTITY_TYPE, {"content": content})`, derived by validation:
    omitted, it is filled in; supplied, it must equal the derived one.
    """

    IDENTITY_TYPE: ClassVar[str] = "text"

    format: FormatTag = "text@1.0"
    content: str
    identity: ContentIdentity

    if TYPE_CHECKING:

        def __init__(self, *, content: str, identity: str | None = None, format: str = "text@1.0") -> None:
            """Hold `content` in canonical form, deriving `identity` or checking the one supplied."""

    @model_validator(mode="before")
    @classmethod
    def _derive_identity(cls, data: Any) -> Any:  # noqa: ANN401 - pydantic's hook signature
        if isinstance(data, Mapping) and isinstance(data.get("content"), str):
            content = canonical_text(data["content"])
            derived = "content:" + value_digest(cls.IDENTITY_TYPE, {"content": content})
            supplied = data.get("identity")
            if supplied is not None and supplied != derived:
                raise ValueError(f"identity {supplied!r} is not the content identity {derived!r}")
            data = {**data, "content": content, "identity": derived}
        return data


def text_identity(field: str, given: object) -> str:
    """Return the identity a `TextIdentity` field named `field` takes from `given`.

    Raises:
        ValueError: `given` is neither `content:` and 64 lowercase hex digits nor `{"content": <text>}`; names the
            field and both forms.
    """
    if isinstance(given, str) and _TEXT_IDENTITY.fullmatch(given):
        return given
    if isinstance(given, Mapping) and set(given) == {"content"} and isinstance(given["content"], str):
        return PlainText(content=given["content"]).identity
    forms = " or ".join(TEXT_IDENTITY_FORMS)
    raise ValueError(f"field {field!r} takes {forms}; got {given!r}")
