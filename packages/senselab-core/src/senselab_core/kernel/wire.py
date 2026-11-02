"""Wire forms of kernel values: plain data tagged with a format, which an isolated worker reads without the kernel.

A value's wire form is its `model_dump()` with `format` set to its registered tag (`WIRE_FORMATS`): nested
values are objects, arrays stay arrays (the worker codec sends them as `.npy` files), tuples are lists. A
regular group read from a file travels as `RegularGroup.decoded_from` builds it, its decoded samples with the file's
stamp, so it keeps the file's identity and no path reaches a worker.
"""

from collections.abc import Mapping
from typing import Any

from senselab_core.kernel._base import Frozen, FrozenMapping
from senselab_core.kernel.formats import Formatted, readable
from senselab_core.kernel.results import Annotation, Embedding, Events, Intervals, Metric, TextSpans, Transcript, Trials
from senselab_core.kernel.signals import Audio, PlainText, Recording, RegularGroup, SourceReader
from senselab_core.kernel.time import TimeMap


def _tag(cls: type[Formatted]) -> str:
    return str(cls.model_fields["format"].default)


_FORMATTED: tuple[type[Formatted], ...] = (
    TimeMap,
    Annotation,
    Embedding,
    Intervals,
    Events,
    PlainText,
    Transcript,
    TextSpans,
    Trials,
    Metric,
)
_SIGNALS: dict[str, type[Frozen]] = {"recording@2.1": Recording, "audio@2.1": Audio, "regular-group@2.1": RegularGroup}
WIRE_FORMATS: Mapping[str, type[Frozen]] = FrozenMapping(_SIGNALS | {_tag(t): t for t in _FORMATTED})
"""The wire format of every kernel value a port may carry, by tag; a `Formatted` type's tag is its own format."""

RETIRED_WIRE_FORMATS = frozenset({"recording@1", "audio@1", "regular-group@1"})
"""Wire formats 2.x no longer reads, as `<name>@<major>`."""
WIRE_MIGRATION_STATEMENT = "specs/20261030-140022-senselab-2x-api/migration-ledger.md#retired-wire-formats"
"""Where the statement on retired wire formats is: they are not read; re-reading the source migrates."""


class WireFormatRetired(ValueError):
    """A wire form in a retired format (`RETIRED_WIRE_FORMATS`).

    Attributes:
        found: The format tag the form declares.
    """

    def __init__(self, found: str) -> None:
        """Name the declared format and the migration statement."""
        super().__init__(
            f"a wire form declares the retired format {found}; it is not read, and migrating a value is re-reading"
            f" its source or recomputing it under this version. See {WIRE_MIGRATION_STATEMENT}"
        )
        self.found = found


_TAGS: Mapping[type[Frozen], str] = FrozenMapping({t: tag for tag, t in WIRE_FORMATS.items()})


def wire_form(value: Frozen, reader: SourceReader | None = None) -> dict[str, Any]:
    """Return `value`'s wire form; a file-backed regular group is decoded by `reader` into held samples.

    Raises:
        ValueError: if `value`'s type has no wire format, or it holds a file-backed group and `reader` is None.
        StaleSource: if a file changed since it was stamped.
    """
    tag = _TAGS.get(type(value))
    if tag is None:
        raise ValueError(f"{type(value).__name__} has no wire format; registered: {sorted(WIRE_FORMATS)}")
    form = value.model_dump()
    if isinstance(value, RegularGroup):
        form = _held(value, reader)
    elif isinstance(value, Recording):
        groups = value.groups.items()
        form["groups"] = {n: _held(g, reader) if isinstance(g, RegularGroup) else g.model_dump() for n, g in groups}
    return {**form, "format": value.format if isinstance(value, Formatted) else tag}


def _held(group: RegularGroup, reader: SourceReader | None) -> dict[str, Any]:
    if group.source is None:
        return group.model_dump()
    if reader is None:
        raise ValueError(f"{group.source} is a source group; read it through a SourceReader")
    return RegularGroup.decoded_from(group, reader).model_dump()


def from_wire_form(form: Mapping[str, Any]) -> Frozen:
    """Return the kernel value a wire form holds, read by the registered type of its tag's name and major version.

    Raises:
        WireFormatRetired: if `form`'s format is retired.
        ValueError: if `form` has no `format` naming a registered wire format of a readable version.
        pydantic.ValidationError: if the fields do not validate as that type.
    """
    found = form.get("format")
    if isinstance(found, str) and found.split(".")[0] in RETIRED_WIRE_FORMATS:
        raise WireFormatRetired(found)
    cls = next((t for tag, t in WIRE_FORMATS.items() if readable(tag, found)), None)
    if cls is None:
        raise ValueError(f"no wire format reads {found!r}; registered: {sorted(WIRE_FORMATS)}")
    fields = dict(form)
    if not issubclass(cls, Formatted):
        del fields["format"]
    return cls.model_validate(fields)
