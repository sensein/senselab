"""A job's outputs between the backend, the store and the caller: wire forms, declared types and clocks."""

from collections.abc import Iterable, Mapping

from senselab_core.kernel._base import Frozen
from senselab_core.kernel.declare import TaskSpec, kernel_types
from senselab_core.kernel.signals import Recording
from senselab_core.kernel.time import TimeAnchor, TimeMap, rebind
from senselab_core.kernel.wire import WireFormatRetired, from_wire_form, wire_form
from senselab_core.runtime.audio_io import READER

Anchors = tuple[tuple[TimeAnchor, TimeAnchor], ...]
"""Pairs of a recording's identity clock anchor and the anchor the caller's recording carries."""


class AmbiguousClock(ValueError):
    """Two recording inputs of one content carry different anchors: a stored output has no one clock to re-bind to."""


class OutputTypeMismatch(ValueError):
    """A backend's output for a kernel-typed port is not a wire form of the port's declared type."""


def anchors_of(values: Iterable[object]) -> Anchors:
    """Return, for each recording input not on the clock its identity names, that clock's anchor and its own.

    Raises:
        AmbiguousClock: two recording inputs of one content carry different anchors.
    """
    readers: dict[str, TimeAnchor] = {}
    for v in values:
        if isinstance(v, Recording) and readers.setdefault(v.identity, v.anchor) != v.anchor:
            raise AmbiguousClock(f"recording inputs of {v.identity} carry anchors on two clocks; pass one anchor")
    return tuple(
        (TimeAnchor(clock_id=identity, absolute_start_ns=None), anchor)
        for identity, anchor in readers.items()
        if anchor.clock_id != identity
    )


def rebound(outputs: dict[str, object], anchors: Anchors) -> dict[str, object]:
    """Return `outputs` re-bound from each pair's first anchor to its second, the two naming one content's start."""
    for stored, reader in anchors:
        timemap = TimeMap(source=stored.clock_id, target=reader.clock_id, offset_ns=0)
        outputs = {port: rebind(value, stored, reader, timemap) for port, value in outputs.items()}
    return outputs


def decoded(spec: TaskSpec, outputs: Mapping[str, object]) -> dict[str, object]:
    """Return `outputs` with each port declared as a kernel type read from its wire form (`from_wire_form`).

    Raises:
        OutputTypeMismatch: an output of a kernel-typed port is not a wire form of its declared type.
        WireFormatRetired: a stored output is in a retired wire format.
    """
    types = kernel_types()
    out = dict(outputs)
    for port, declared in spec.outputs.items():
        if declared not in types or port not in outputs:
            continue
        form = outputs[port]
        try:
            value = from_wire_form(form) if isinstance(form, Mapping) else form
        except WireFormatRetired:
            raise
        except ValueError as exc:
            raise OutputTypeMismatch(f"output {port!r} of {spec.name!r} is not a wire form: {exc}") from exc
        if not isinstance(value, types[declared]):
            found = form.get("format") if isinstance(form, Mapping) else type(form).__name__
            raise OutputTypeMismatch(f"output {port!r} of {spec.name!r} is declared {declared}, got {found}")
        out[port] = value
    return out


def as_wire(values: Mapping[str, object]) -> dict[str, object]:
    """Return `values` with each kernel value as its wire form."""
    return {port: wire_form(v, READER) if isinstance(v, Frozen) else v for port, v in values.items()}


def as_stored(
    spec: TaskSpec, computed: Mapping[str, object], anchors: Anchors
) -> tuple[dict[str, object], dict[str, object]]:
    """Return computed outputs decoded, and as the store holds them: on their identity clocks, as wire forms."""
    read = decoded(spec, computed)
    on_identity_clocks = rebound(read, tuple((reader, stored) for stored, reader in anchors))
    outputs = {
        p: wire_form(v, READER) if isinstance(v, Frozen) and v is not read[p] else computed[p]
        for p, v in on_identity_clocks.items()
    }
    return read, outputs
